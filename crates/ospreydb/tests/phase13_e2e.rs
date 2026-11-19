#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use object_store::memory::InMemory;
use object_store::ObjectStore;
use osprey_storage::{
    MinipageLruCache, RangePlannerConfig, RemoteStoreConfig, DEFAULT_MINIPAGE_CACHE_BYTES,
};
use ospreydb::{
    build_http_router, ClusterManifest, Embedder, EmbedderConfig, HttpServerState,
    HybridSearchEngine, LsmDeltaBuffer, LutRankReranker, ManifestStore, ProMetricsRegistry,
    DEFAULT_LUTRANK_SEED, DEFAULT_MANIFEST_KEY,
};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

fn basis_vec(dim: usize, hot: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim];
    v[hot % dim] = 1.0;
    v[(hot + 1) % dim] = 0.2;
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut v {
        *x /= n;
    }
    v
}

// ---------------------------------------------------------------------------
// Test: Unlimited Vectors (>5M) via HTTP Upsert Without Any Paywall
// ---------------------------------------------------------------------------
#[tokio::test]
async fn http_upsert_allows_unlimited_vectors_above_5m_without_paywall() {
    let dim = 64_usize;
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let manifest_store = ManifestStore::new(Arc::clone(&store), DEFAULT_MANIFEST_KEY);
    manifest_store
        .create_if_absent(&ClusterManifest::new_empty())
        .await
        .unwrap();

    let delta = Arc::new(LsmDeltaBuffer::new(dim, dim, 42, Vec::new(), None).unwrap());
    let metrics = Arc::new(ProMetricsRegistry::new());
    let engine = Arc::new(HybridSearchEngine::with_metrics(
        Arc::clone(&delta),
        Arc::clone(&store),
        manifest_store,
        Arc::new(MinipageLruCache::new(DEFAULT_MINIPAGE_CACHE_BYTES)),
        RemoteStoreConfig::default(),
        RangePlannerConfig::default(),
        Arc::clone(&metrics),
    ));

    let embedder = Arc::new(Embedder::new(EmbedderConfig::default()));
    let reranker = Arc::new(LutRankReranker::new(DEFAULT_LUTRANK_SEED).unwrap());
    let counter = Arc::new(AtomicU64::new(5_000_000));
    let state =
        HttpServerState::with_vector_counter(reranker, embedder, Arc::clone(&metrics), counter);
    state.register_collection("test", engine).unwrap();

    let seed_vec = basis_vec(dim, 0);
    delta
        .ingest_batch(&[ospreydb::RawVectorRecord {
            doc_id: 42,
            vector: &seed_vec,
            payload_json: br#"{"title":"existing-doc"}"#,
        }])
        .unwrap();

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel::<()>();

    let server_handle = tokio::spawn(async move {
        axum::serve(listener, build_http_router(state))
            .with_graceful_shutdown(async {
                let _ = rx.await;
            })
            .await
            .unwrap();
    });

    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    // Upsert above 5M vectors succeeds with 200 OK
    let resp = client
        .put(format!("{base_url}/collections/test/points"))
        .json(&json!({
            "points": [
                { "id": 5_000_001, "vector": basis_vec(dim, 0) }
            ]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status().as_u16(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");

    let search_resp = client
        .post(format!("{base_url}/collections/test/points/search"))
        .json(&json!({
            "vector": seed_vec,
            "limit": 3,
            "with_payload": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(search_resp.status().as_u16(), 200);
    let search_json: Value = search_resp.json().await.unwrap();
    assert_eq!(search_json["status"], "ok");

    let info_resp = client
        .get(format!("{base_url}/collections/test"))
        .send()
        .await
        .unwrap();
    assert_eq!(info_resp.status().as_u16(), 200);
    let info_json: Value = info_resp.json().await.unwrap();
    assert_eq!(info_json["result"]["total_vectors_count"], 5_000_001);
    assert_eq!(info_json["result"]["read_only"], false);
    assert!(info_json["result"]["license"].is_null());

    // License activation route has been removed completely
    let license_route = client
        .post(format!("{base_url}/v1/license/activate"))
        .json(&json!({ "license": "anything" }))
        .send()
        .await
        .unwrap();
    assert_eq!(license_route.status().as_u16(), 404);

    let _ = tx.send(());
    let _ = server_handle.await;
}

// ---------------------------------------------------------------------------
// Test: Text-based upsert and query_text search (via mock/non-configured fallback)
// ---------------------------------------------------------------------------
#[tokio::test]
async fn text_based_upsert_with_explicit_vector_delegates_to_direct_ingest() {
    let dim = 64_usize;
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let manifest_store = ManifestStore::new(Arc::clone(&store), DEFAULT_MANIFEST_KEY);
    manifest_store
        .create_if_absent(&ClusterManifest::new_empty())
        .await
        .unwrap();

    let delta = Arc::new(LsmDeltaBuffer::new(dim, dim, 77, Vec::new(), None).unwrap());
    let metrics = Arc::new(ProMetricsRegistry::new());
    let engine = Arc::new(HybridSearchEngine::with_metrics(
        Arc::clone(&delta),
        Arc::clone(&store),
        manifest_store,
        Arc::new(MinipageLruCache::new(DEFAULT_MINIPAGE_CACHE_BYTES)),
        RemoteStoreConfig::default(),
        RangePlannerConfig::default(),
        Arc::clone(&metrics),
    ));

    let embedder = Arc::new(Embedder::new(EmbedderConfig::default()));
    let reranker = Arc::new(LutRankReranker::new(DEFAULT_LUTRANK_SEED).unwrap());
    let state = HttpServerState::new(reranker, embedder, metrics);
    state.register_collection("text_vectors", engine).unwrap();

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel::<()>();

    let server_handle = tokio::spawn(async move {
        axum::serve(listener, build_http_router(state))
            .with_graceful_shutdown(async {
                let _ = rx.await;
            })
            .await
            .unwrap();
    });

    let client = reqwest::Client::new();
    let base_url = format!("http://{addr}");

    // 1. Upsert with explicit vector (direct path)
    let v1 = basis_vec(dim, 10);
    let upsert_resp: Value = client
        .put(format!("{base_url}/collections/text_vectors/points"))
        .json(&json!({
            "points": [
                { "id": 1, "vector": v1, "payload": { "title": "document_one" } }
            ]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(upsert_resp["status"], "ok");

    // 2. Upsert with only `text` but no `vector` and no embedder configured -> error
    let text_only_resp = client
        .put(format!("{base_url}/collections/text_vectors/points"))
        .json(&json!({
            "points": [
                { "id": 2, "text": "a document about vector search", "payload": { "title": "doc_two" } }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(text_only_resp.status().as_u16(), 400);

    // 3. Search with explicit vector
    let search_resp: Value = client
        .post(format!("{base_url}/collections/text_vectors/points/search"))
        .json(&json!({
            "vector": v1,
            "limit": 3,
            "with_payload": true
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(search_resp["status"], "ok");
    assert_eq!(search_resp["result"][0]["id"], 1);

    // 4. Search with query_text but no embedder configured -> error
    let query_text_resp: Value = client
        .post(format!("{base_url}/collections/text_vectors/points/search"))
        .json(&json!({
            "query_text": "document one vector",
            "limit": 3
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(query_text_resp["status"]["error"]
        .as_str()
        .unwrap()
        .contains("embedding"));

    let _ = tx.send(());
    let _ = server_handle.await;
}

// ---------------------------------------------------------------------------
// Test: CAS ETag conflict and retry loop preserves all updates
// ---------------------------------------------------------------------------
#[tokio::test]
async fn manifest_cas_etag_conflict_retry_preserves_all_writer_segments() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let manifest_store = ManifestStore::new(Arc::clone(&store), DEFAULT_MANIFEST_KEY);

    manifest_store
        .create_if_absent(&ClusterManifest::new_empty())
        .await
        .unwrap();

    let mut handles = Vec::new();
    for worker_id in 0..16_u64 {
        let ms = manifest_store.clone();
        handles.push(tokio::spawn(async move {
            ms.update_with_cas_retry(64, |m| {
                let id = m.allocate_segment_id();
                m.segments.push(
                    ospreydb::SegmentDescriptor::new(
                        id,
                        format!("w{worker_id}/seg-{id}.osi"),
                        format!("w{worker_id}/seg-{id}.osv"),
                        64,
                        64,
                        worker_id.wrapping_add(1),
                        100,
                        &roaring::RoaringTreemap::new(),
                    )
                    .unwrap(),
                );
                Ok(())
            })
            .await
        }));
    }

    for h in handles {
        h.await.unwrap().unwrap();
    }

    let final_snap = manifest_store.load().await.unwrap().unwrap();
    assert_eq!(final_snap.manifest.segments.len(), 16);
    assert!(final_snap.manifest.epoch >= 17);
}

// ---------------------------------------------------------------------------
// Test: Full collection info JSON exposes vector count and embedder metadata
// ---------------------------------------------------------------------------
#[tokio::test]
async fn collection_info_exposes_vector_count_and_embedder_metadata() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let manifest_store = ManifestStore::new(Arc::clone(&store), DEFAULT_MANIFEST_KEY);
    manifest_store
        .create_if_absent(&ClusterManifest::new_empty())
        .await
        .unwrap();

    let dim = 64_usize;
    let delta = Arc::new(LsmDeltaBuffer::new(dim, dim, 2001, Vec::new(), None).unwrap());
    let metrics = Arc::new(ProMetricsRegistry::new());
    let engine = Arc::new(HybridSearchEngine::with_metrics(
        Arc::clone(&delta),
        store,
        manifest_store,
        Arc::new(MinipageLruCache::new(DEFAULT_MINIPAGE_CACHE_BYTES)),
        RemoteStoreConfig::default(),
        RangePlannerConfig::default(),
        Arc::clone(&metrics),
    ));

    let embedder = Arc::new(Embedder::new(EmbedderConfig::default()));
    let reranker = Arc::new(LutRankReranker::new(DEFAULT_LUTRANK_SEED).unwrap());
    let state = HttpServerState::new(reranker, embedder, metrics);
    state.register_collection("info_test", engine).unwrap();

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel::<()>();

    let server_handle = tokio::spawn(async move {
        axum::serve(listener, build_http_router(state))
            .with_graceful_shutdown(async {
                let _ = rx.await;
            })
            .await
            .unwrap();
    });

    let client = reqwest::Client::new();
    let info: Value = client
        .get(format!("http://{addr}/collections/info_test"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    assert_eq!(info["status"], "ok");
    assert_eq!(info["result"]["status"], "green");
    assert_eq!(info["result"]["total_vectors_count"], 0);
    assert_eq!(info["result"]["read_only"], false);
    assert!(info["result"]["license"].is_null());
    assert!(info["result"]["embedder"]["configured"]
        .as_bool()
        .is_some_and(|c| !c));
    assert!(info["result"]["embedder"]["model"]
        .as_str()
        .is_some_and(|s| !s.is_empty()));

    let _ = tx.send(());
    let _ = server_handle.await;
}
