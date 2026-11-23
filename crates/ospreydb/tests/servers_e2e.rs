#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use arrow::array::{
    BinaryArray, FixedSizeListArray, Float32Array, Float64Array, ListArray, StringArray,
    UInt32Array, UInt64Array,
};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use arrow::record_batch::RecordBatch;
use arrow_flight::decode::FlightRecordBatchStream;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::error::FlightError;
use arrow_flight::flight_service_client::FlightServiceClient;
use arrow_flight::flight_service_server::FlightServiceServer;
use arrow_flight::{Criteria, FlightDescriptor, Ticket};
use futures::stream::{self, TryStreamExt};
use object_store::memory::InMemory;
use object_store::ObjectStore;
use osprey_core::RawVectorRecord;
use osprey_storage::{
    MinipageLruCache, OsvWriterConfig, RangePlannerConfig, RemoteStoreConfig,
    DEFAULT_MINIPAGE_CACHE_BYTES,
};
use ospreydb::{
    build_http_router, ClusterManifest, CompactorConfig, CowSlabCompactor, Embedder,
    EmbedderConfig, HttpServerState, HybridSearchEngine, LsmDeltaBuffer, LutRankReranker,
    ManifestStore, OspreyFlightService, ProMetricsRegistry, DEFAULT_LUTRANK_SEED,
    DEFAULT_MANIFEST_KEY,
};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio_stream::wrappers::TcpListenerStream;

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

#[tokio::test]
async fn arrow_flight_do_put_do_exchange_and_do_get_e2e() {
    let dim = 64_usize;
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let manifest_store = ManifestStore::new(Arc::clone(&store), DEFAULT_MANIFEST_KEY);
    manifest_store
        .create_if_absent(&ClusterManifest::new_empty())
        .await
        .unwrap();

    let delta = Arc::new(LsmDeltaBuffer::new(dim, dim, 55, Vec::new(), None).unwrap());
    let metrics = Arc::new(ProMetricsRegistry::new());
    let engine = Arc::new(HybridSearchEngine::with_metrics(
        Arc::clone(&delta),
        Arc::clone(&store),
        manifest_store.clone(),
        Arc::new(MinipageLruCache::new(DEFAULT_MINIPAGE_CACHE_BYTES)),
        RemoteStoreConfig::default(),
        RangePlannerConfig::default(),
        Arc::clone(&metrics),
    ));

    // Pre-compact 1 base segment so DoGet/ListFlights returns segment metadata too
    let base_vecs: Vec<Vec<f32>> = (0..32).map(|i| basis_vec(dim, i)).collect();
    let base_payloads: Vec<Vec<u8>> = (0..32)
        .map(|i| format!(r#"{{"id":{i},"src":"base"}}"#).into_bytes())
        .collect();
    let base_batch: Vec<RawVectorRecord<'_>> = (0..32)
        .map(|i| RawVectorRecord {
            doc_id: (i + 1) as u64,
            vector: &base_vecs[i],
            payload_json: &base_payloads[i],
        })
        .collect();
    delta.ingest_batch(&base_batch).unwrap();
    let compactor = CowSlabCompactor::new(
        CompactorConfig {
            delta_flush_threshold: 10,
            target_clusters: 4,
            fwht_seed: 55,
            osv_writer_config: OsvWriterConfig {
                enable_zstd_dict: false,
                ..OsvWriterConfig::default()
            },
            ..CompactorConfig::default()
        },
        Arc::clone(&store),
        manifest_store,
    );
    compactor.compact_now(&delta, None).await.unwrap();
    assert!(engine.refresh_from_manifest().await.unwrap());

    let reranker = Arc::new(LutRankReranker::new(DEFAULT_LUTRANK_SEED).unwrap());
    let svc = OspreyFlightService::new(Arc::clone(&engine), reranker);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel::<()>();

    let server_handle = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(FlightServiceServer::new(svc))
            .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                let _ = rx.await;
            })
            .await
            .unwrap();
    });

    let endpoint = format!("http://{addr}");
    let mut client = FlightServiceClient::connect(endpoint).await.unwrap();

    // 1. DoPut: stream RecordBatch of new vectors zero-copy
    let put_schema = Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::UInt64, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                i32::try_from(dim).unwrap(),
            ),
            false,
        ),
        Field::new("payload_json", DataType::Binary, false),
    ]));
    let v_1000 = basis_vec(dim, 40);
    let v_1001 = basis_vec(dim, 41);
    let fsl = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        vec![
            Some(v_1000.iter().copied().map(Some).collect::<Vec<_>>()),
            Some(v_1001.iter().copied().map(Some).collect::<Vec<_>>()),
        ],
        i32::try_from(dim).unwrap(),
    );
    let put_batch = RecordBatch::try_new(
        Arc::clone(&put_schema),
        vec![
            Arc::new(UInt64Array::from(vec![1000_u64, 1001_u64])),
            Arc::new(fsl),
            Arc::new(BinaryArray::from_vec(vec![
                br#"{"src":"flight_put","doc":1000}"#.as_slice(),
                br#"{"src":"flight_put","doc":1001}"#.as_slice(),
            ])),
        ],
    )
    .unwrap();

    let put_data: Vec<_> = FlightDataEncoderBuilder::new()
        .with_schema(put_schema)
        .build(stream::iter(vec![Ok(put_batch)]))
        .try_collect()
        .await
        .unwrap();
    let put_res: Vec<_> = client
        .do_put(stream::iter(put_data))
        .await
        .unwrap()
        .into_inner()
        .try_collect()
        .await
        .unwrap();
    assert_eq!(put_res.len(), 1);
    let meta: Value = serde_json::from_slice(&put_res[0].app_metadata).unwrap();
    assert_eq!(meta["ingested"], 2);

    // 2. DoExchange (Vector Search): send query RecordBatch and decode returned hits
    let search_schema = Arc::new(Schema::new(vec![
        Field::new("query_id", DataType::UInt64, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                i32::try_from(dim).unwrap(),
            ),
            false,
        ),
        Field::new("top_k", DataType::UInt32, false),
    ]));
    let q_fsl = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        vec![Some(v_1000.iter().copied().map(Some).collect::<Vec<_>>())],
        i32::try_from(dim).unwrap(),
    );
    let search_batch = RecordBatch::try_new(
        Arc::clone(&search_schema),
        vec![
            Arc::new(UInt64Array::from(vec![7_u64])),
            Arc::new(q_fsl),
            Arc::new(UInt32Array::from(vec![3_u32])),
        ],
    )
    .unwrap();
    let ex_data: Vec<_> = FlightDataEncoderBuilder::new()
        .with_schema(search_schema)
        .build(stream::iter(vec![Ok(search_batch)]))
        .try_collect()
        .await
        .unwrap();
    let resp_stream = client
        .do_exchange(stream::iter(ex_data))
        .await
        .unwrap()
        .into_inner();
    let resp_batches: Vec<RecordBatch> =
        FlightRecordBatchStream::new_from_flight_data(resp_stream.map_err(FlightError::Tonic))
            .try_collect()
            .await
            .unwrap();
    assert_eq!(resp_batches.len(), 1);
    let top_docs = resp_batches[0]
        .column_by_name("doc_id")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(top_docs.value(0), 1000);

    // 3. DoExchange (LUT-Rank Rerank)
    let rerank_schema = Arc::new(Schema::new(vec![
        Field::new("query", DataType::Utf8, false),
        Field::new("document", DataType::Utf8, false),
        Field::new("top_n", DataType::UInt32, false),
    ]));
    let rerank_batch = RecordBatch::try_new(
        Arc::clone(&rerank_schema),
        vec![
            Arc::new(StringArray::from(vec![
                "arrow flight zero copy",
                "arrow flight zero copy",
            ])),
            Arc::new(StringArray::from(vec![
                "unrelated recipe blog post",
                "apache arrow flight zero copy grpc vector search",
            ])),
            Arc::new(UInt32Array::from(vec![2_u32, 2_u32])),
        ],
    )
    .unwrap();
    let rr_data: Vec<_> = FlightDataEncoderBuilder::new()
        .with_schema(rerank_schema)
        .build(stream::iter(vec![Ok(rerank_batch)]))
        .try_collect()
        .await
        .unwrap();
    let rr_resp = client
        .do_exchange(stream::iter(rr_data))
        .await
        .unwrap()
        .into_inner();
    let rr_batches: Vec<RecordBatch> =
        FlightRecordBatchStream::new_from_flight_data(rr_resp.map_err(FlightError::Tonic))
            .try_collect()
            .await
            .unwrap();
    let doc_indices = rr_batches[0]
        .column_by_name("doc_index")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(doc_indices.value(0), 1);

    // 4. ListFlights & DoGet
    let flights: Vec<_> = client
        .list_flights(Criteria::default())
        .await
        .unwrap()
        .into_inner()
        .try_collect()
        .await
        .unwrap();
    assert_eq!(flights.len(), 1);
    let info = client
        .get_flight_info(FlightDescriptor::new_path(vec!["segments".to_owned()]))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(info.endpoint.len(), 1);

    let get_stream = client
        .do_get(Ticket::new("segments"))
        .await
        .unwrap()
        .into_inner();
    let seg_batches: Vec<RecordBatch> =
        FlightRecordBatchStream::new_from_flight_data(get_stream.map_err(FlightError::Tonic))
            .try_collect()
            .await
            .unwrap();
    assert_eq!(seg_batches[0].num_rows(), 1);

    let _ = tx.send(());
    let _ = server_handle.await;
}

#[tokio::test]
async fn dropin_cohere_rerank_qdrant_search_and_prometheus_metrics_e2e() {
    let dim = 64_usize;
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let manifest_store = ManifestStore::new(Arc::clone(&store), DEFAULT_MANIFEST_KEY);
    manifest_store
        .create_if_absent(&ClusterManifest::new_empty())
        .await
        .unwrap();

    let delta = Arc::new(LsmDeltaBuffer::new(dim, dim, 99, Vec::new(), None).unwrap());
    let metrics = Arc::new(ProMetricsRegistry::new());
    let engine = Arc::new(HybridSearchEngine::with_metrics(
        Arc::clone(&delta),
        Arc::clone(&store),
        manifest_store.clone(),
        Arc::new(MinipageLruCache::new(DEFAULT_MINIPAGE_CACHE_BYTES)),
        RemoteStoreConfig::default(),
        RangePlannerConfig::default(),
        Arc::clone(&metrics),
    ));

    // Populate base segment so search triggers range planner + S3 stats in Prometheus metrics
    let vecs: Vec<Vec<f32>> = (0..40).map(|i| basis_vec(dim, i)).collect();
    let payloads: Vec<Vec<u8>> = (0..40)
        .map(|i| format!(r#"{{"title":"doc_{i}"}}"#).into_bytes())
        .collect();
    let batch: Vec<RawVectorRecord<'_>> = (0..40)
        .map(|i| RawVectorRecord {
            doc_id: (i + 1) as u64,
            vector: &vecs[i],
            payload_json: &payloads[i],
        })
        .collect();
    delta.ingest_batch(&batch).unwrap();

    let compactor = CowSlabCompactor::new(
        CompactorConfig {
            delta_flush_threshold: 10,
            target_clusters: 4,
            fwht_seed: 99,
            osv_writer_config: OsvWriterConfig {
                enable_zstd_dict: false,
                minipage_target_bytes: 512,
                ..OsvWriterConfig::default()
            },
            ..CompactorConfig::default()
        },
        Arc::clone(&store),
        manifest_store,
    );
    compactor.compact_now(&delta, None).await.unwrap();
    assert!(engine.refresh_from_manifest().await.unwrap());

    let reranker = Arc::new(LutRankReranker::new(DEFAULT_LUTRANK_SEED).unwrap());
    let state = HttpServerState::new(
        reranker,
        Arc::new(Embedder::new(EmbedderConfig::default())),
        Arc::clone(&metrics),
    );
    state
        .register_collection("papers", Arc::clone(&engine))
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

    // 1. Cohere/Jina Drop-In POST /v1/rerank
    let rerank_resp: Value = client
        .post(format!("{base_url}/v1/rerank"))
        .json(&json!({
            "model": "rerank-english-v3.0",
            "query": "S3 RaBitQ vector search",
            "documents": [
                "italian pasta cooking guide",
                { "text": "S3 RaBitQ vector search with DP range coalescing" },
                "bicycle repair manual"
            ],
            "top_n": 2,
            "return_documents": true
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(rerank_resp["results"].as_array().unwrap().len(), 2);
    assert_eq!(rerank_resp["results"][0]["index"], 1);
    assert!(rerank_resp["results"][0]["document"]["text"]
        .as_str()
        .unwrap()
        .contains("RaBitQ"));

    // 2. Qdrant Drop-In PUT /collections/{name}/points + POST /collections/{name}/points/search
    let live_v = basis_vec(dim, 50);
    let upsert_resp: Value = client
        .put(format!("{base_url}/collections/papers/points"))
        .json(&json!({
            "points": [
                { "id": 5000, "vector": live_v, "payload": { "title": "realtime_delta_paper" } }
            ]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(upsert_resp["status"], "ok");

    let qdrant_resp: Value = client
        .post(format!("{base_url}/collections/papers/points/search"))
        .json(&json!({
            "vector": vecs[10],
            "limit": 3,
            "with_payload": true
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(qdrant_resp["status"], "ok");
    assert_eq!(qdrant_resp["result"][0]["id"], 11);
    assert_eq!(qdrant_resp["result"][0]["payload"]["title"], "doc_10");

    // 3. Prometheus GET /metrics
    let prom_text = client
        .get(format!("{base_url}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(prom_text.contains("osprey_pro_rabitq_scan_duration_microseconds_bucket"));
    assert!(prom_text.contains("osprey_pro_rabitq_scan_duration_microseconds_p50_microseconds"));
    assert!(prom_text.contains("osprey_pro_rabitq_scan_duration_microseconds_p95_microseconds"));
    assert!(prom_text.contains("osprey_pro_rabitq_scan_duration_microseconds_p99_microseconds"));
    assert!(prom_text.contains("osprey_pro_range_planner_duration_microseconds_count"));
    assert!(prom_text.contains("osprey_pro_s3_calls_saved_total"));
    assert!(prom_text.contains("osprey_pro_lutrank_tokens_per_sec"));

    let _ = tx.send(());
    let _ = server_handle.await;
}

#[tokio::test]
async fn arrow_flight_allows_unlimited_do_put_above_5m_vectors_without_paywall() {
    let dim = 64_usize;
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let manifest_store = ManifestStore::new(Arc::clone(&store), DEFAULT_MANIFEST_KEY);
    manifest_store
        .create_if_absent(&ClusterManifest::new_empty())
        .await
        .unwrap();

    let delta = Arc::new(LsmDeltaBuffer::new(dim, dim, 77, Vec::new(), None).unwrap());
    let existing_vec = basis_vec(dim, 3);
    delta
        .ingest_batch(&[RawVectorRecord {
            doc_id: 777,
            vector: &existing_vec,
            payload_json: br#"{"doc":"existing"}"#,
        }])
        .unwrap();

    let engine = Arc::new(HybridSearchEngine::new(
        Arc::clone(&delta),
        Arc::clone(&store),
        manifest_store,
        Arc::new(MinipageLruCache::new(DEFAULT_MINIPAGE_CACHE_BYTES)),
        RemoteStoreConfig::default(),
        RangePlannerConfig::default(),
    ));
    let reranker = Arc::new(LutRankReranker::new(DEFAULT_LUTRANK_SEED).unwrap());
    let counter = Arc::new(AtomicU64::new(5_000_000));
    let svc = OspreyFlightService::with_vector_counter(
        Arc::clone(&engine),
        reranker,
        Arc::clone(&counter),
    );

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel::<()>();

    let server_handle = tokio::spawn(async move {
        let incoming = TcpListenerStream::new(listener);
        tonic::transport::Server::builder()
            .add_service(FlightServiceServer::new(svc))
            .serve_with_incoming_shutdown(incoming, async {
                let _ = rx.await;
            })
            .await
            .unwrap();
    });

    let endpoint = format!("http://{addr}");
    let mut client = FlightServiceClient::connect(endpoint).await.unwrap();

    // 1. DoPut must succeed freely above 5M vectors (100% Apache-2.0, zero paywall)
    let put_schema = Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::UInt64, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                i32::try_from(dim).unwrap(),
            ),
            false,
        ),
    ]));
    let new_vec = basis_vec(dim, 9);
    let fsl = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        vec![Some(new_vec.into_iter().map(Some).collect::<Vec<_>>())],
        i32::try_from(dim).unwrap(),
    );
    let put_batch = RecordBatch::try_new(
        Arc::clone(&put_schema),
        vec![Arc::new(UInt64Array::from(vec![888_u64])), Arc::new(fsl)],
    )
    .unwrap();
    let flight_data: Vec<_> = FlightDataEncoderBuilder::new()
        .with_schema(put_schema)
        .build(stream::iter(vec![Ok(put_batch)]))
        .try_collect()
        .await
        .unwrap();

    let put_res: Vec<_> = client
        .do_put(stream::iter(flight_data))
        .await
        .expect("DoPut must succeed without vector cap")
        .into_inner()
        .try_collect()
        .await
        .unwrap();
    let meta: Value = serde_json::from_slice(&put_res[0].app_metadata).unwrap();
    assert_eq!(meta["ingested"], 1);
    assert_eq!(
        counter.load(std::sync::atomic::Ordering::Acquire),
        5_000_001
    );

    // 2. DoExchange vector search works normally
    let search_schema = Arc::new(Schema::new(vec![
        Field::new("query_id", DataType::UInt64, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                i32::try_from(dim).unwrap(),
            ),
            false,
        ),
        Field::new("top_k", DataType::UInt32, false),
    ]));
    let search_fsl = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        vec![Some(existing_vec.into_iter().map(Some).collect::<Vec<_>>())],
        i32::try_from(dim).unwrap(),
    );
    let search_batch = RecordBatch::try_new(
        Arc::clone(&search_schema),
        vec![
            Arc::new(UInt64Array::from(vec![1_u64])),
            Arc::new(search_fsl),
            Arc::new(UInt32Array::from(vec![3_u32])),
        ],
    )
    .unwrap();
    let ex_data: Vec<_> = FlightDataEncoderBuilder::new()
        .with_schema(search_schema)
        .build(stream::iter(vec![Ok(search_batch)]))
        .try_collect()
        .await
        .unwrap();
    let resp_stream = client
        .do_exchange(stream::iter(ex_data))
        .await
        .unwrap()
        .into_inner();
    let resp_batches: Vec<RecordBatch> =
        FlightRecordBatchStream::new_from_flight_data(resp_stream.map_err(FlightError::Tonic))
            .try_collect()
            .await
            .unwrap();
    assert_eq!(resp_batches.len(), 1);
    let top_docs = resp_batches[0]
        .column_by_name("doc_id")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(top_docs.value(0), 777);

    // 3. ListFlights & DoGet metadata work normally
    let flights: Vec<_> = client
        .list_flights(Criteria::default())
        .await
        .unwrap()
        .into_inner()
        .try_collect()
        .await
        .unwrap();
    assert_eq!(flights.len(), 1);

    let get_stream = client
        .do_get(Ticket::new("segments"))
        .await
        .unwrap()
        .into_inner();
    let seg_batches: Vec<RecordBatch> =
        FlightRecordBatchStream::new_from_flight_data(get_stream.map_err(FlightError::Tonic))
            .try_collect()
            .await
            .unwrap();
    assert_eq!(
        seg_batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
        0
    );

    let _ = tx.send(());
    let _ = server_handle.await;
}

#[tokio::test]
async fn arrow_flight_accepts_list_float64_and_list_float32_in_do_put_and_do_exchange() {
    let dim = 64_usize;
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let manifest_store = ManifestStore::new(Arc::clone(&store), DEFAULT_MANIFEST_KEY);
    manifest_store
        .create_if_absent(&ClusterManifest::new_empty())
        .await
        .unwrap();

    let delta = Arc::new(LsmDeltaBuffer::new(dim, dim, 123, Vec::new(), None).unwrap());
    let engine = Arc::new(HybridSearchEngine::new(
        Arc::clone(&delta),
        Arc::clone(&store),
        manifest_store,
        Arc::new(MinipageLruCache::new(DEFAULT_MINIPAGE_CACHE_BYTES)),
        RemoteStoreConfig::default(),
        RangePlannerConfig::default(),
    ));
    let reranker = Arc::new(LutRankReranker::new(DEFAULT_LUTRANK_SEED).unwrap());
    let svc = OspreyFlightService::new(Arc::clone(&engine), reranker);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel::<()>();

    let server_handle = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(FlightServiceServer::new(svc))
            .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                let _ = rx.await;
            })
            .await
            .unwrap();
    });

    let mut client = FlightServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap();

    // 1. DoPut with List<Float64> (PyArrow default for pa.array([[0.12, -0.04, ...]]))
    let v_f32 = basis_vec(dim, 15);
    let v_f64: Vec<f64> = v_f32.iter().map(|&x| f64::from(x)).collect();
    let list_f64 = ListArray::new(
        Arc::new(Field::new("item", DataType::Float64, true)),
        OffsetBuffer::from_lengths([dim]),
        Arc::new(Float64Array::from(v_f64.clone())),
        None,
    );
    let put_schema = Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::UInt64, false),
        Field::new(
            "vector",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            false,
        ),
    ]));
    let put_batch = RecordBatch::try_new(
        Arc::clone(&put_schema),
        vec![
            Arc::new(UInt64Array::from(vec![9001_u64])),
            Arc::new(list_f64),
        ],
    )
    .unwrap();
    let put_data: Vec<_> = FlightDataEncoderBuilder::new()
        .with_schema(put_schema)
        .build(stream::iter(vec![Ok(put_batch)]))
        .try_collect()
        .await
        .unwrap();
    let put_res: Vec<_> = client
        .do_put(stream::iter(put_data))
        .await
        .unwrap()
        .into_inner()
        .try_collect()
        .await
        .unwrap();
    let meta: Value = serde_json::from_slice(&put_res[0].app_metadata).unwrap();
    assert_eq!(meta["ingested"], 1);

    // 2. DoExchange search with List<Float32>
    let list_f32 = ListArray::new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        OffsetBuffer::from_lengths([dim]),
        Arc::new(Float32Array::from(v_f32)),
        None,
    );
    let search_schema = Arc::new(Schema::new(vec![
        Field::new("query_id", DataType::UInt64, false),
        Field::new(
            "vector",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
            false,
        ),
        Field::new("top_k", DataType::UInt32, false),
    ]));
    let search_batch = RecordBatch::try_new(
        Arc::clone(&search_schema),
        vec![
            Arc::new(UInt64Array::from(vec![1_u64])),
            Arc::new(list_f32),
            Arc::new(UInt32Array::from(vec![1_u32])),
        ],
    )
    .unwrap();
    let ex_data: Vec<_> = FlightDataEncoderBuilder::new()
        .with_schema(search_schema)
        .build(stream::iter(vec![Ok(search_batch)]))
        .try_collect()
        .await
        .unwrap();
    let resp_stream = client
        .do_exchange(stream::iter(ex_data))
        .await
        .unwrap()
        .into_inner();
    let resp_batches: Vec<RecordBatch> =
        FlightRecordBatchStream::new_from_flight_data(resp_stream.map_err(FlightError::Tonic))
            .try_collect()
            .await
            .unwrap();
    let doc_col = resp_batches[0]
        .column_by_name("doc_id")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(doc_col.value(0), 9001);

    let _ = tx.send(());
    let _ = server_handle.await;
}

#[tokio::test]
async fn http_qdrant_compat_delete_token_rerank_and_api_key_auth_e2e() {
    let dim = 64_usize;
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let manifest_store = ManifestStore::new(Arc::clone(&store), DEFAULT_MANIFEST_KEY);
    manifest_store
        .create_if_absent(&ClusterManifest::new_empty())
        .await
        .unwrap();

    let delta = Arc::new(LsmDeltaBuffer::new(dim, dim, 321, Vec::new(), None).unwrap());
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

    let reranker = Arc::new(LutRankReranker::new(DEFAULT_LUTRANK_SEED).unwrap());
    let state = HttpServerState::new(
        reranker,
        Arc::new(Embedder::new(EmbedderConfig::default())),
        metrics,
    )
    .with_api_key(Some("osprey-secret-123".to_owned()));
    state.register_collection("default", engine).unwrap();

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

    // 1. /healthz stays open even when --api-key is configured
    let health = client
        .get(format!("{base_url}/healthz"))
        .send()
        .await
        .unwrap();
    assert_eq!(health.status().as_u16(), 200);

    // 2. Protected routes reject missing/wrong API key with 401
    let unauth = client
        .get(format!("{base_url}/collections"))
        .send()
        .await
        .unwrap();
    assert_eq!(unauth.status().as_u16(), 401);

    // 3. GET /collections & PUT /collections/{name} succeed with Bearer or api-key header
    let list_resp: Value = client
        .get(format!("{base_url}/collections"))
        .bearer_auth("osprey-secret-123")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(list_resp["status"], "ok");
    assert_eq!(list_resp["result"]["collections"][0]["name"], "default");

    let create_resp: Value = client
        .put(format!("{base_url}/collections/default"))
        .header("api-key", "osprey-secret-123")
        .json(&json!({ "vectors": { "size": dim, "distance": "Dot" } }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(create_resp["status"], "ok");
    assert_eq!(create_resp["result"], true);

    // 4. Dimension mismatch on upsert returns HTTP 400
    let mismatch_resp = client
        .put(format!("{base_url}/collections/default/points"))
        .header("api-key", "osprey-secret-123")
        .json(&json!({
            "points": [{ "id": 1, "vector": [1.0, 2.0, 3.0] }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(mismatch_resp.status().as_u16(), 400);

    // 5. Upsert + Delete removes vector from subsequent search
    let v1 = basis_vec(dim, 4);
    let v2 = basis_vec(dim, 8);
    let upsert_ok = client
        .put(format!("{base_url}/collections/default/points"))
        .header("api-key", "osprey-secret-123")
        .json(&json!({
            "points": [
                { "id": 10, "vector": v1 },
                { "id": 20, "vector": v2 }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(upsert_ok.status().as_u16(), 200);

    let del_resp: Value = client
        .post(format!("{base_url}/collections/default/points/delete"))
        .bearer_auth("osprey-secret-123")
        .json(&json!({ "points": [10] }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(del_resp["status"], "ok");

    let search_after_del: Value = client
        .post(format!("{base_url}/collections/default/points/search"))
        .header("api-key", "osprey-secret-123")
        .json(&json!({ "vector": v1, "limit": 5 }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ids: Vec<u64> = search_after_del["result"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|p| p["id"].as_u64())
        .collect();
    assert!(!ids.contains(&10));
    assert!(ids.contains(&20));

    // 6. Explicit ColBERT token embeddings in POST /v1/rerank
    let mut q_tok = vec![0.0_f32; 128];
    q_tok[0] = 1.0;
    let mut d0_tok = vec![0.0_f32; 128];
    d0_tok[5] = 1.0;
    let mut d1_tok = vec![0.0_f32; 128];
    d1_tok[0] = 1.0;
    let tok_rerank: Value = client
        .post(format!("{base_url}/v1/rerank"))
        .bearer_auth("osprey-secret-123")
        .json(&json!({
            "query_tokens": [q_tok],
            "document_tokens": [[d0_tok], [d1_tok]],
            "documents": ["orthogonal", "exact match"],
            "return_documents": true
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(tok_rerank["results"][0]["index"], 1);
    assert_eq!(tok_rerank["results"][0]["document"]["text"], "exact match");

    let _ = tx.send(());
    let _ = server_handle.await;
}
