//! Drop-in HTTP servers for Cohere/Jina (`POST /v1/rerank`), Qdrant (`:6333`
//! `POST /collections/{name}/points/search`), Prometheus (`GET /metrics`), and
//! built-in Auto-Embeddings `/v1/embeddings` (OpenAI-compatible contract).

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, RwLock};

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
use crate::embedder::Embedder;
use crate::error::{ProEngineError, ProResult};
use crate::license::{OspreyLicense, VectorQuotaState, COMMUNITY_MAX_VECTORS};
use crate::metrics::ProMetricsRegistry;
use crate::rerank_service::LutRankReranker;

pub const DEFAULT_HTTP_ADDR: &str = "0.0.0.0:6333";

#[derive(Clone, Debug)]
pub struct HttpServerState {
    collections: Arc<RwLock<HashMap<String, Arc<HybridSearchEngine>>>>,
    reranker: Arc<LutRankReranker>,
    embedder: Arc<Embedder>,
    metrics: Arc<ProMetricsRegistry>,
    default_nprobe: usize,
    quota: VectorQuotaState,
}

impl HttpServerState {
    #[must_use]
    pub fn new(
        reranker: Arc<LutRankReranker>,
        embedder: Arc<Embedder>,
        metrics: Arc<ProMetricsRegistry>,
    ) -> Self {
        Self::with_quota(reranker, embedder, metrics, VectorQuotaState::new())
    }

    #[must_use]
    pub fn with_quota(
        reranker: Arc<LutRankReranker>,
        embedder: Arc<Embedder>,
        metrics: Arc<ProMetricsRegistry>,
        quota: VectorQuotaState,
    ) -> Self {
        Self {
            collections: Arc::new(RwLock::new(HashMap::new())),
            reranker,
            embedder,
            metrics,
            default_nprobe: 16,
            quota,
        }
    }

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

    pub fn get_collection(&self, name: &str) -> ProResult<Option<Arc<HybridSearchEngine>>> {
        let guard = self.collections.read().map_err(|_| {
            ProEngineError::Io(std::io::Error::other("collections RwLock poisoned"))
        })?;
        Ok(guard.get(name).cloned())
    }

    /// Activates a Pro license, unlocking unlimited vectors and multi-node features.
    pub fn activate_license(&self, license: OspreyLicense) {
        self.quota.activate_license(license);
    }

    /// Returns the current vector count across all collections.
    #[must_use]
    pub fn total_vector_count(&self) -> u64 {
        self.quota.total_vector_count()
    }

    /// Checks whether the total vector count is allowed under the active license.
    /// Returns `Ok(())` if allowed, or `ProEngineError::CommunityTierLimitReached` if blocked.
    pub fn check_vector_limit(&self, additional: u64) -> Result<(), ProEngineError> {
        self.quota.check_vector_limit(additional)
    }

    pub fn increment_vector_count(&self, delta: u64) {
        self.quota.increment_vector_count(delta);
    }
}

pub fn build_http_router(state: HttpServerState) -> Router {
    Router::new()
        .route("/healthz", get(handle_healthz))
        .route("/metrics", get(handle_metrics))
        .route("/v1/rerank", post(handle_cohere_rerank))
        .route("/v1/embeddings", post(handle_embeddings))
        .route(
            "/collections/{name}/points/search",
            post(handle_qdrant_search),
        )
        .route("/collections/{name}/points", put(handle_qdrant_upsert))
        .route("/collections/{name}", get(handle_qdrant_collection_info))
        .with_state(state)
}

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

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum RerankDocumentInput {
    Text(String),
    Object { text: String },
}

impl RerankDocumentInput {
    fn as_str(&self) -> &str {
        match self {
            Self::Text(s) => s.as_str(),
            Self::Object { text } => text.as_str(),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct CohereRerankRequest {
    pub model: Option<String>,
    pub query: String,
    pub documents: Vec<RerankDocumentInput>,
    pub top_n: Option<usize>,
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

#[derive(Clone, Debug, Default, Deserialize)]
pub struct QdrantSearchParams {
    pub hnsw_ef: Option<usize>,
    pub nprobe: Option<usize>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct QdrantSearchRequest {
    pub vector: Option<Vec<f32>>,
    pub query_text: Option<String>,
    #[serde(default = "default_qdrant_limit")]
    pub limit: usize,
    pub offset: Option<usize>,
    pub with_payload: Option<bool>,
    pub score_threshold: Option<f32>,
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
    let start = std::time::Instant::now();
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

    let query_vec = match (req.vector, req.query_text) {
        (Some(v), _) if !v.is_empty() => v,
        (_, Some(ref text)) => match state.embedder.embed_one(text).await {
            Ok(emb) => emb,
            Err(err) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({
                        "status": { "error": format!("embedding query_text failed: {err}") },
                        "time": start.elapsed().as_secs_f64()
                    })),
                )
                    .into_response();
            }
        },
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "status": { "error": "either 'vector' or 'query_text' is required" },
                    "time": start.elapsed().as_secs_f64()
                })),
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
        .search_hybrid_async(&query_vec, nprobe, top_m, fetch_k)
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

#[derive(Clone, Debug, Deserialize)]
pub struct QdrantPointStruct {
    pub id: u64,
    #[serde(default)]
    pub vector: Vec<f32>,
    pub text: Option<String>,
    pub payload: Option<Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct QdrantUpsertRequest {
    pub points: Vec<QdrantPointStruct>,
}

async fn handle_qdrant_upsert(
    State(state): State<HttpServerState>,
    Path(collection_name): Path<String>,
    Json(req): Json<QdrantUpsertRequest>,
) -> Response {
    let start = std::time::Instant::now();
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

    // Collect texts that need embedding
    let mut texts_to_embed: Vec<String> = Vec::new();
    let mut text_indices: Vec<usize> = Vec::new();
    for (idx, point) in req.points.iter().enumerate() {
        if let Some(ref text) = point.text {
            if point.vector.is_empty() {
                texts_to_embed.push(text.clone());
                text_indices.push(idx);
            }
        }
    }

    let embedding_count = u64::try_from(texts_to_embed.len()).unwrap_or(0);

    // Enforce community tier vector limit
    if let Err(limit_err) = state.check_vector_limit(
        u64::try_from(req.points.len()).unwrap_or(0)
    ) {
        let status_code = if matches!(limit_err, ProEngineError::CommunityTierLimitReached { .. }) {
            StatusCode::from_u16(402).unwrap_or(StatusCode::PAYMENT_REQUIRED)
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        };
        return (
            status_code,
            Json(json!({
                "status": {
                    "error": format!("{limit_err}"),
                    "current_count": state.total_vector_count(),
                }
            })),
        )
            .into_response();
    }

    let embedded: Vec<Vec<f32>> = if !texts_to_embed.is_empty() {
        match state.embedder.embed_batch(&texts_to_embed).await {
            Ok(emb) => emb,
            Err(err) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({
                        "status": {
                            "error": format!("embedding text failed: {err}")
                        }
                    })),
                )
                    .into_response();
            }
        }
    } else {
        Vec::new()
    };

    // Build records: use explicit vector or embedded text vector
    let needs_vector_context = if !embedded.is_empty() {
        let mut map = HashMap::with_capacity(embedded.len());
        for (emb_idx, point_idx) in text_indices.iter().enumerate() {
            if emb_idx < embedded.len() {
                map.insert(*point_idx, embedded[emb_idx].clone());
            }
        }
        Some(map)
    } else {
        None
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
        .enumerate()
        .zip(payload_bytes.iter())
        .filter_map(|((idx, p), payload)| {
let vector_ref = if !p.vector.is_empty() {
                Some(p.vector.as_slice())
            } else if let Some(ctx) = &needs_vector_context {
                ctx.get(&idx).map(|v| v.as_slice())
            } else {
                None
            };
            vector_ref.map(|v| RawVectorRecord {
                doc_id: p.id,
                vector: v,
                payload_json: payload,
            })
        })
        .collect();

    if records.is_empty() && !req.points.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "status": { "error": "points require either 'vector' or 'text' for embedding" } })),
        )
            .into_response();
    }

    let ingested_count = records.len();
    match engine.delta_buffer().ingest_batch(&records) {
        Ok(seq) => {
            state.metrics.observe_wal_ingest(ingested_count);
            state.increment_vector_count(u64::try_from(ingested_count).unwrap_or(0));
            (
                StatusCode::OK,
                Json(json!({
                    "result": {
                        "operation_id": seq,
                        "status": "completed",
                        "embedded_texts": embedding_count,
                    },
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
    let start = std::time::Instant::now();
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
    let total_vecs = state.total_vector_count();
    let limit_reached = state.quota.is_limit_reached();
    let license_info = state
        .quota
        .active_license_snapshot()
        .map(|lic| {
            json!({
                "tier": lic.tier,
                "max_vectors": lic.max_vectors,
                "is_unlimited": lic.is_unlimited_vectors(),
                "limit_reached": limit_reached,
                "read_only": limit_reached,
            })
        })
        .unwrap_or_else(|| {
            json!({
                "tier": "community",
                "max_vectors": COMMUNITY_MAX_VECTORS,
                "is_unlimited": false,
                "limit_reached": limit_reached,
                "read_only": limit_reached,
            })
        });

    (
        StatusCode::OK,
        Json(json!({
            "result": {
                "status": "green",
                "segments_count": segments,
                "delta_vectors_count": delta_live,
                "total_vectors_count": total_vecs,
                "license": license_info,
                "embedder": {
                    "configured": state.embedder.is_configured(),
                    "model": state.embedder.model(),
                },
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

// ---------------------------------------------------------------------------
// Built-in OpenAI-compatible /v1/embeddings endpoint (zero ONNX/Torch)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Deserialize)]
struct EmbeddingsRequest {
    model: Option<String>,
    input: EmbeddingsInput,
    #[allow(dead_code)]
    #[serde(default)]
    encoding_format: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum EmbeddingsInput {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Clone, Debug, Serialize)]
struct EmbeddingsUsage {
    prompt_tokens: u64,
    total_tokens: u64,
}

#[derive(Clone, Debug, Serialize)]
struct EmbeddingsDataItem {
    object: String,
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Clone, Debug, Serialize)]
struct EmbeddingsResponse {
    object: String,
    data: Vec<EmbeddingsDataItem>,
    model: String,
    usage: EmbeddingsUsage,
}

async fn handle_embeddings(
    State(state): State<HttpServerState>,
    Json(req): Json<EmbeddingsRequest>,
) -> Response {
    let model = req
        .model
        .unwrap_or_else(|| state.embedder.model().to_owned());

    let texts: Vec<String> = match req.input {
        EmbeddingsInput::Single(s) => vec![s],
        EmbeddingsInput::Batch(arr) => arr,
    };

    match state.embedder.embed_batch(&texts).await {
        Ok(embeddings) => {
            let data: Vec<EmbeddingsDataItem> = embeddings
                .into_iter()
                .enumerate()
                .map(|(idx, embedding)| EmbeddingsDataItem {
                    object: "embedding".to_owned(),
                    embedding,
                    index: idx,
                })
                .collect();
            let token_count = texts
                .iter()
                .map(|t| t.split_whitespace().count() as u64)
                .sum::<u64>()
                .max(1);
            let resp = EmbeddingsResponse {
                object: "list".to_owned(),
                data,
                model,
                usage: EmbeddingsUsage {
                    prompt_tokens: token_count,
                    total_tokens: token_count,
                },
            };
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(err) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": { "message": err.to_string(), "type": "server_error" } })),
        )
            .into_response(),
    }
}