//! Drop-in HTTP servers for Cohere/Jina (`POST /v1/rerank`), Qdrant (`:6333`
//! `POST /collections/{name}/points/search`), Prometheus (`GET /metrics`), and
//! built-in Auto-Embeddings `/v1/embeddings` (OpenAI-compatible contract).

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use axum::extract::{DefaultBodyLimit, Path, Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::{self, Next};
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
use crate::metrics::ProMetricsRegistry;
use crate::rerank_service::LutRankReranker;

pub const DEFAULT_HTTP_ADDR: &str = "0.0.0.0:6333";

/// Maximum HTTP JSON request body size (64 MiB) for large batch upserts.
pub const MAX_HTTP_BODY_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct HttpServerState {
    collections: Arc<RwLock<HashMap<String, Arc<HybridSearchEngine>>>>,
    reranker: Arc<LutRankReranker>,
    embedder: Arc<Embedder>,
    metrics: Arc<ProMetricsRegistry>,
    default_nprobe: usize,
    total_vectors: Arc<AtomicU64>,
    api_key: Option<String>,
    read_only: bool,
}

impl HttpServerState {
    #[must_use]
    pub fn new(
        reranker: Arc<LutRankReranker>,
        embedder: Arc<Embedder>,
        metrics: Arc<ProMetricsRegistry>,
    ) -> Self {
        Self::with_vector_counter(reranker, embedder, metrics, Arc::new(AtomicU64::new(0)))
    }

    #[must_use]
    pub fn with_vector_counter(
        reranker: Arc<LutRankReranker>,
        embedder: Arc<Embedder>,
        metrics: Arc<ProMetricsRegistry>,
        total_vectors: Arc<AtomicU64>,
    ) -> Self {
        Self {
            collections: Arc::new(RwLock::new(HashMap::new())),
            reranker,
            embedder,
            metrics,
            default_nprobe: 16,
            total_vectors,
            api_key: None,
            read_only: false,
        }
    }

    /// Configures an optional static HTTP API key (`api-key` or `Authorization: Bearer`).
    #[must_use]
    pub fn with_api_key(mut self, api_key: Option<String>) -> Self {
        self.api_key = api_key
            .map(|k| k.trim().to_owned())
            .filter(|k| !k.is_empty());
        self
    }

    /// Marks this HTTP server read-only (for `--role reader` nodes), rejecting vector writes.
    #[must_use]
    pub const fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
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

    /// Returns the tracked total vector count across all collections.
    #[must_use]
    pub fn total_vector_count(&self) -> u64 {
        self.total_vectors.load(Ordering::Acquire)
    }

    /// Increments the tracked total vector count by `delta` after successful ingestion.
    pub fn increment_vector_count(&self, delta: u64) {
        self.total_vectors.fetch_add(delta, Ordering::Release);
    }
}

pub fn build_http_router(state: HttpServerState) -> Router {
    let auth_state = state.clone();
    Router::new()
        .route("/healthz", get(handle_healthz))
        .route("/metrics", get(handle_metrics))
        .route("/v1/rerank", post(handle_cohere_rerank))
        .route("/v1/embeddings", post(handle_embeddings))
        .route("/collections", get(handle_qdrant_list_collections))
        .route(
            "/collections/{name}",
            get(handle_qdrant_collection_info).put(handle_qdrant_create_collection),
        )
        .route("/collections/{name}/points", put(handle_qdrant_upsert))
        .route(
            "/collections/{name}/points/search",
            post(handle_qdrant_search),
        )
        .route(
            "/collections/{name}/points/delete",
            post(handle_qdrant_delete),
        )
        .layer(DefaultBodyLimit::max(MAX_HTTP_BODY_BYTES))
        .layer(middleware::from_fn_with_state(
            auth_state,
            api_key_auth_middleware,
        ))
        .with_state(state)
}

async fn api_key_auth_middleware(
    State(state): State<HttpServerState>,
    req: Request,
    next: Next,
) -> Response {
    let Some(expected_key) = state.api_key.as_deref() else {
        return next.run(req).await;
    };

    if req.uri().path() == "/healthz" {
        return next.run(req).await;
    }

    let provided_api_key = req
        .headers()
        .get("api-key")
        .and_then(|v| v.to_str().ok())
        .map(str::trim);

    let provided_bearer = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|raw| {
            let trimmed = raw.trim();
            if trimmed.len() >= 7 && trimmed[..7].eq_ignore_ascii_case("bearer ") {
                Some(trimmed[7..].trim())
            } else {
                None
            }
        });

    let authorized = provided_api_key
        .is_some_and(|k| constant_time_eq(k.as_bytes(), expected_key.as_bytes()))
        || provided_bearer.is_some_and(|k| constant_time_eq(k.as_bytes(), expected_key.as_bytes()));

    if !authorized {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({
                "status": { "error": "unauthorized: valid 'api-key' or 'Authorization: Bearer' header required" }
            })),
        )
            .into_response();
    }

    next.run(req).await
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let max_len = a.len().max(b.len());
    let mut diff = u8::from(a.len() != b.len());
    for i in 0..max_len {
        let ba = a.get(i).copied().unwrap_or(0);
        let bb = b.get(i).copied().unwrap_or(0);
        diff |= ba ^ bb;
    }
    diff == 0
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
    Json(json!({ "status": "ok", "engine": "osprey" }))
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
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub documents: Vec<RerankDocumentInput>,
    #[serde(default)]
    pub query_tokens: Option<Vec<Vec<f32>>>,
    #[serde(default)]
    pub document_tokens: Option<Vec<Vec<Vec<f32>>>>,
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

    let hits = match (&req.query_tokens, &req.document_tokens) {
        (Some(q_toks), Some(d_toks)) => {
            if q_toks.is_empty() {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({ "message": "query_tokens must not be empty" })),
                )
                    .into_response();
            }
            let q_refs: Vec<&[f32]> = q_toks.iter().map(Vec::as_slice).collect();
            let d_refs: Vec<Vec<&[f32]>> = d_toks
                .iter()
                .map(|doc| doc.iter().map(Vec::as_slice).collect())
                .collect();
            match state.reranker.rerank_token_embeddings(
                &q_refs,
                &d_refs,
                req.top_n,
                Some(&state.metrics),
            ) {
                Ok(hits) => hits,
                Err(err) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(json!({ "message": err.to_string() })),
                    )
                        .into_response();
                }
            }
        }
        (Some(_), None) | (None, Some(_)) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "message": "both 'query_tokens' and 'document_tokens' must be provided together"
                })),
            )
                .into_response();
        }
        (None, None) => {
            let Some(query) = req.query.as_deref() else {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({
                        "message": "either 'query' + 'documents' or 'query_tokens' + 'document_tokens' is required"
                    })),
                )
                    .into_response();
            };
            match state
                .reranker
                .rerank_texts(query, &doc_strs, req.top_n, Some(&state.metrics))
            {
                Ok(hits) => hits,
                Err(err) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(json!({ "message": err.to_string() })),
                    )
                        .into_response();
                }
            }
        }
    };

    let return_docs = req.return_documents.unwrap_or(false);
    let results: Vec<CohereRerankResultItem> = hits
        .into_iter()
        .map(|h| CohereRerankResultItem {
            index: h.index,
            relevance_score: h.relevance_score,
            document: if return_docs {
                doc_strs.get(h.index).map(|text| RerankDocumentText {
                    text: (*text).to_owned(),
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
    if state.read_only {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({
                "status": { "error": "node is running in read-only (--role reader) mode; writes are not permitted" },
                "time": start.elapsed().as_secs_f64()
            })),
        )
            .into_response();
    }
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

    let expected_dim = engine.delta_buffer().orig_dim();
    for rec in &records {
        if rec.vector.len() != expected_dim {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "status": {
                        "error": format!(
                            "vector dimension mismatch for point {}: expected {expected_dim}, got {}",
                            rec.doc_id,
                            rec.vector.len()
                        )
                    },
                    "time": start.elapsed().as_secs_f64()
                })),
            )
                .into_response();
        }
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

#[derive(Clone, Debug, Deserialize)]
struct QdrantDeleteRequest {
    points: Option<Vec<u64>>,
}

async fn handle_qdrant_delete(
    State(state): State<HttpServerState>,
    Path(collection_name): Path<String>,
    Json(req): Json<QdrantDeleteRequest>,
) -> Response {
    let start = std::time::Instant::now();
    if state.read_only {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({
                "status": { "error": "node is running in read-only (--role reader) mode; deletes are not permitted" },
                "time": start.elapsed().as_secs_f64()
            })),
        )
            .into_response();
    }

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

    let Some(doc_ids) = req.points else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "status": { "error": "'points' array of numeric IDs is required" } })),
        )
            .into_response();
    };

    match engine.delta_buffer().delete_docs(&doc_ids) {
        Ok(seq) => (
            StatusCode::OK,
            Json(json!({
                "result": {
                    "operation_id": seq,
                    "status": "completed",
                    "deleted": doc_ids.len(),
                },
                "status": "ok",
                "time": start.elapsed().as_secs_f64()
            })),
        )
            .into_response(),
        Err(err) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "status": { "error": err.to_string() } })),
        )
            .into_response(),
    }
}

async fn handle_qdrant_list_collections(State(state): State<HttpServerState>) -> Response {
    let start = std::time::Instant::now();
    let names: Vec<Value> = match state.collections.read() {
        Ok(guard) => {
            let mut keys: Vec<&String> = guard.keys().collect();
            keys.sort();
            keys.into_iter()
                .map(|name| json!({ "name": name }))
                .collect()
        }
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "status": { "error": err.to_string() } })),
            )
                .into_response();
        }
    };

    (
        StatusCode::OK,
        Json(json!({
            "result": {
                "collections": names
            },
            "status": "ok",
            "time": start.elapsed().as_secs_f64()
        })),
    )
        .into_response()
}

async fn handle_qdrant_create_collection(
    State(state): State<HttpServerState>,
    Path(collection_name): Path<String>,
    body: Option<Json<Value>>,
) -> Response {
    let start = std::time::Instant::now();
    let requested_dim = body
        .as_ref()
        .and_then(|Json(v)| v.get("vectors"))
        .and_then(|v| v.get("size"))
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok());

    match state.get_collection(&collection_name) {
        Ok(Some(engine)) => {
            let expected_dim = engine.delta_buffer().orig_dim();
            if let Some(dim) = requested_dim {
                if dim != expected_dim {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(json!({
                            "status": {
                                "error": format!(
                                    "collection `{collection_name}` expects vector dimension {expected_dim}, got {dim}"
                                )
                            },
                            "time": start.elapsed().as_secs_f64()
                        })),
                    )
                        .into_response();
                }
            }
            (
                StatusCode::OK,
                Json(json!({
                    "result": true,
                    "status": "ok",
                    "time": start.elapsed().as_secs_f64()
                })),
            )
                .into_response()
        }
        Ok(None) => (
            StatusCode::OK,
            Json(json!({
                "result": true,
                "status": "ok",
                "time": start.elapsed().as_secs_f64()
            })),
        )
            .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
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

    (
        StatusCode::OK,
        Json(json!({
            "result": {
                "status": "green",
                "segments_count": segments,
                "delta_vectors_count": delta_live,
                "total_vectors_count": total_vecs,
                "read_only": state.read_only,
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
