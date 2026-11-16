//! Drop-in HTTP servers for Cohere/Jina (`POST /v1/rerank`), Qdrant (`:6333`
//! `POST /collections/{name}/points/search`), and Prometheus (`GET /metrics`).

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use osprey_core::RawVectorRecord;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::net::TcpListener;

use crate::delta_buffer::HybridSearchEngine;
use crate::error::{ProEngineError, ProResult};
use crate::metrics::ProMetricsRegistry;
use crate::rerank_service::LutRankReranker;

/// Default Qdrant-compatible HTTP bind address (`:6333`).
pub const DEFAULT_HTTP_ADDR: &str = "0.0.0.0:6333";

/// Shared application state powering all drop-in HTTP routes and `/metrics`.
#[derive(Clone, Debug)]
pub struct HttpServerState {
    collections: Arc<RwLock<HashMap<String, Arc<HybridSearchEngine>>>>,
    reranker: Arc<LutRankReranker>,
    metrics: Arc<ProMetricsRegistry>,
    default_nprobe: usize,
}

impl HttpServerState {
    /// Creates a new [`HttpServerState`] with a shared [`LutRankReranker`] and [`ProMetricsRegistry`].
    #[must_use]
    pub fn new(reranker: Arc<LutRankReranker>, metrics: Arc<ProMetricsRegistry>) -> Self {
        Self {
            collections: Arc::new(RwLock::new(HashMap::new())),
            reranker,
            metrics,
            default_nprobe: 16,
        }
    }

    /// Registers or replaces a named collection engine for Qdrant-compatible routes.
    pub fn register_collection(
        &self,
        name: impl Into<String>,
        engine: Arc<HybridSearchEngine>,
    ) -> ProResult<()> {
        let mut guard = self.collections.write().map_err(|_| {
            ProEngineError::Io(std::io::Error::other("collections RwLock poisoned"))
        })?;
        guard.insert(name.into(), engine);
        Ok(())
    }

    /// Resolves a collection engine by name.
    pub fn get_collection(&self, name: &str) -> ProResult<Option<Arc<HybridSearchEngine>>> {
        let guard = self.collections.read().map_err(|_| {
            ProEngineError::Io(std::io::Error::other("collections RwLock poisoned"))
        })?;
        Ok(guard.get(name).cloned())
    }
}

/// Builds the Axum [`Router`] exposing Cohere/Jina `/v1/rerank`, Qdrant `/collections/...`,
/// `/healthz`, and Prometheus `/metrics`.
pub fn build_http_router(state: HttpServerState) -> Router {
    Router::new()
        .route("/healthz", get(handle_healthz))
        .route("/metrics", get(handle_metrics))
        .route("/v1/rerank", post(handle_cohere_rerank))
        .route(
            "/collections/{name}/points/search",
            post(handle_qdrant_search),
        )
        .route("/collections/{name}/points", put(handle_qdrant_upsert))
        .route("/collections/{name}", get(handle_qdrant_collection_info))
        .with_state(state)
}

/// Binds and runs the drop-in HTTP server on `addr` until `shutdown` resolves.
pub async fn serve_http<F>(addr: SocketAddr, state: HttpServerState, shutdown: F) -> ProResult<()>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    let listener = TcpListener::bind(addr).await?;
    let router = build_http_router(state);
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

async fn handle_healthz() -> impl IntoResponse {
    Json(json!({ "status": "ok", "engine": "osprey-pro" }))
}

async fn handle_metrics(State(state): State<HttpServerState>) -> Response {
    let body = state.metrics.render_prometheus();
    (
        StatusCode::OK,
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
        .into_response()
}

/// Cohere / Jina document input: either plain string or `{"text": "..."}`.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum RerankDocumentInput {
    /// Raw document text.
    Text(String),
    /// Structured document object with `"text"` field.
    Object {
        /// Document text body.
        text: String,
    },
}

impl RerankDocumentInput {
    fn as_str(&self) -> &str {
        match self {
            Self::Text(s) => s.as_str(),
            Self::Object { text } => text.as_str(),
        }
    }
}

/// Cohere / Jina compatible `POST /v1/rerank` request payload.
#[derive(Clone, Debug, Deserialize)]
pub struct CohereRerankRequest {
    /// Optional model identifier (logged/ignored; executed via local `LUT-Rank`).
    pub model: Option<String>,
    /// Query text to score documents against.
    pub query: String,
    /// Candidate documents to rerank.
    pub documents: Vec<RerankDocumentInput>,
    /// Maximum number of top results to return.
    pub top_n: Option<usize>,
    /// Whether to include document text in each result object.
    pub return_documents: Option<bool>,
}

#[derive(Clone, Debug, Serialize)]
struct RerankDocumentText {
    text: String,
}

#[derive(Clone, Debug, Serialize)]
struct CohereRerankResultItem {
    index: usize,
    relevance_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    document: Option<RerankDocumentText>,
}

#[derive(Clone, Debug, Serialize)]
struct CohereRerankResponse {
    id: String,
    results: Vec<CohereRerankResultItem>,
    meta: Value,
}

async fn handle_cohere_rerank(
    State(state): State<HttpServerState>,
    Json(req): Json<CohereRerankRequest>,
) -> Response {
    let doc_strs: Vec<&str> = req
        .documents
        .iter()
        .map(RerankDocumentInput::as_str)
        .collect();
    let hits =
        match state
            .reranker
            .rerank_texts(&req.query, &doc_strs, req.top_n, Some(&state.metrics))
        {
            Ok(hits) => hits,
            Err(err) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({ "message": err.to_string() })),
                )
                    .into_response();
            }
        };

    let return_docs = req.return_documents.unwrap_or(false);
    let results: Vec<CohereRerankResultItem> = hits
        .into_iter()
        .map(|h| CohereRerankResultItem {
            index: h.index,
            relevance_score: h.relevance_score,
            document: if return_docs {
                Some(RerankDocumentText {
                    text: doc_strs[h.index].to_owned(),
                })
            } else {
                None
            },
        })
        .collect();

    let resp = CohereRerankResponse {
        id: "osprey-lutrank-v1".to_owned(),
        results,
        meta: json!({
            "api_version": { "version": "1" },
            "billed_units": { "search_units": 1 },
            "model": req.model.unwrap_or_else(|| "osprey-lutrank-v1".to_owned())
        }),
    };
    (StatusCode::OK, Json(resp)).into_response()
}

/// Optional Qdrant search tuning parameters.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct QdrantSearchParams {
    /// Oversampling factor / probe hint mapped to `top_m` candidate budget.
    pub hnsw_ef: Option<usize>,
    /// Explicit cluster probe count.
    pub nprobe: Option<usize>,
}

/// Qdrant compatible `POST /collections/{name}/points/search` request body.
#[derive(Clone, Debug, Deserialize)]
pub struct QdrantSearchRequest {
    /// Dense query vector.
    pub vector: Vec<f32>,
    /// Number of top results to return.
    #[serde(default = "default_qdrant_limit")]
    pub limit: usize,
    /// Optional result offset for pagination.
    pub offset: Option<usize>,
    /// Whether to return JSON payloads (default: `true`).
    pub with_payload: Option<bool>,
    /// Optional minimum exact score cutoff.
    pub score_threshold: Option<f32>,
    /// Optional search parameters.
    pub params: Option<QdrantSearchParams>,
}

const fn default_qdrant_limit() -> usize {
    10
}

#[derive(Clone, Debug, Serialize)]
struct QdrantScoredPoint {
    id: u64,
    version: u64,
    score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    payload: Option<Value>,
}

async fn handle_qdrant_search(
    State(state): State<HttpServerState>,
    Path(collection_name): Path<String>,
    Json(req): Json<QdrantSearchRequest>,
) -> Response {
    let start = Instant::now();
    let engine = match state.get_collection(&collection_name) {
        Ok(Some(eng)) => eng,
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({
                    "status": { "error": format!("Not found: Collection `{collection_name}` doesn't exist!") },
                    "time": start.elapsed().as_secs_f64()
                })),
            )
                .into_response();
        }
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "status": { "error": err.to_string() } })),
            )
                .into_response();
        }
    };

    let offset = req.offset.unwrap_or(0);
    let fetch_k = req.limit.saturating_add(offset);
    let params = req.params.unwrap_or_default();
    let nprobe = params.nprobe.unwrap_or(state.default_nprobe);
    let top_m = params
        .hnsw_ef
        .unwrap_or_else(|| fetch_k.saturating_mul(8).max(64));

    let results = match engine
        .search_hybrid_async(&req.vector, nprobe, top_m, fetch_k)
        .await
    {
        Ok(r) => r,
        Err(err) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "status": { "error": err.to_string() },
                    "time": start.elapsed().as_secs_f64()
                })),
            )
                .into_response();
        }
    };

    let with_payload = req.with_payload.unwrap_or(true);
    let points: Vec<QdrantScoredPoint> = results
        .into_iter()
        .filter(|r| {
            req.score_threshold
                .is_none_or(|thresh| r.exact_score >= thresh)
        })
        .skip(offset)
        .take(req.limit)
        .map(|r| {
            let payload = if with_payload {
                if r.payload_json.is_empty() {
                    Some(json!({}))
                } else {
                    Some(
                        serde_json::from_slice::<Value>(&r.payload_json).unwrap_or_else(|_| {
                            Value::String(String::from_utf8_lossy(&r.payload_json).into_owned())
                        }),
                    )
                }
            } else {
                None
            };
            QdrantScoredPoint {
                id: r.doc_id,
                version: 1,
                score: r.exact_score,
                payload,
            }
        })
        .collect();

    (
        StatusCode::OK,
        Json(json!({
            "result": points,
            "status": "ok",
            "time": start.elapsed().as_secs_f64()
        })),
    )
        .into_response()
}

/// Qdrant-compatible upsert point entry (`PUT /collections/{name}/points`).
#[derive(Clone, Debug, Deserialize)]
pub struct QdrantPointStruct {
    /// Numeric point ID.
    pub id: u64,
    /// Dense vector.
    pub vector: Vec<f32>,
    /// Optional JSON payload.
    pub payload: Option<Value>,
}

/// Qdrant-compatible batch upsert payload (`PUT /collections/{name}/points`).
#[derive(Clone, Debug, Deserialize)]
pub struct QdrantUpsertRequest {
    /// Points to upsert into WAL + `LsmDeltaBuffer`.
    pub points: Vec<QdrantPointStruct>,
}

async fn handle_qdrant_upsert(
    State(state): State<HttpServerState>,
    Path(collection_name): Path<String>,
    Json(req): Json<QdrantUpsertRequest>,
) -> Response {
    let start = Instant::now();
    let engine = match state.get_collection(&collection_name) {
        Ok(Some(eng)) => eng,
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({ "status": { "error": format!("Collection `{collection_name}` not found") } })),
            )
                .into_response();
        }
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "status": { "error": err.to_string() } })),
            )
                .into_response();
        }
    };

    let payload_bytes: Vec<Vec<u8>> = req
        .points
        .iter()
        .map(|p| match &p.payload {
            Some(val) => serde_json::to_vec(val).unwrap_or_else(|_| b"{}".to_vec()),
            None => b"{}".to_vec(),
        })
        .collect();

    let records: Vec<RawVectorRecord<'_>> = req
        .points
        .iter()
        .zip(payload_bytes.iter())
        .map(|(p, payload)| RawVectorRecord {
            doc_id: p.id,
            vector: &p.vector,
            payload_json: payload,
        })
        .collect();

    match engine.delta_buffer().ingest_batch(&records) {
        Ok(seq) => {
            state.metrics.observe_wal_ingest(records.len());
            (
                StatusCode::OK,
                Json(json!({
                    "result": { "operation_id": seq, "status": "completed" },
                    "status": "ok",
                    "time": start.elapsed().as_secs_f64()
                })),
            )
                .into_response()
        }
        Err(err) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "status": { "error": err.to_string() } })),
        )
            .into_response(),
    }
}

async fn handle_qdrant_collection_info(
    State(state): State<HttpServerState>,
    Path(collection_name): Path<String>,
) -> Response {
    let start = Instant::now();
    let engine = match state.get_collection(&collection_name) {
        Ok(Some(eng)) => eng,
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({ "status": { "error": format!("Collection `{collection_name}` not found") } })),
            )
                .into_response();
        }
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "status": { "error": err.to_string() } })),
            )
                .into_response();
        }
    };

    let delta_live = engine.delta_buffer().live_count().unwrap_or(0);
    let segments = engine.loaded_segment_count().unwrap_or(0);
    (
        StatusCode::OK,
        Json(json!({
            "result": {
                "status": "green",
                "segments_count": segments,
                "delta_vectors_count": delta_live,
                "config": {
                    "params": {
                        "vectors": {
                            "size": engine.delta_buffer().orig_dim(),
                            "distance": "Dot"
                        }
                    }
                }
            },
            "status": "ok",
            "time": start.elapsed().as_secs_f64()
        })),
    )
        .into_response()
}
