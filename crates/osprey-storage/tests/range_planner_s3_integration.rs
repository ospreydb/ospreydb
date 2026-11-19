//! End-to-end Phase 9 S3 integration test validating:
//! - `SimulatedLatencyObjectStore` wrapping `object_store::memory::InMemory` with TTFB + tail spikes;
//! - TSP + DP range coalescing `Range-GET` reduction ($\ge 10\times$–$18\times$), `< 200 us` DP plan time, and
//!   $100\%$ exact `f16` refinement equivalence via `RefineStage::refine_async`;
//! - `MinipageLruCache` hot-page offloading on repeat queries;
//! - Speculative P99 **Hedged Requests** cutting simulated `50 ms` S3 tail latency spikes.

use std::fmt;
use std::future::Future;
use std::ops::Range;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures::stream::BoxStream;
use half::f16;
use object_store::memory::InMemory;
use object_store::path::Path as ObjectPath;
use object_store::{
    GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore, PutMultipartOpts,
    PutOptions, PutPayload, PutResult, Result as ObjectStoreResult,
};
use osprey_core::{OspreyError, OspreyResult};
use osprey_storage::{
    solve_tsp_locality_largest_first, CandidatePageRef, MinipageLruCache, OsvSlabReader,
    OsvVectorRecord, OsvWriter, OsvWriterConfig, RangePlannerConfig, RefineStage,
    RemoteStoreConfig,
};

type BoxFut<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// `object_store::ObjectStore` wrapper over `InMemory` simulating S3 TTFB, jitter,
/// and configurable tail-latency spikes while counting HTTP `Range-GET`s and bytes transferred.
struct SimulatedLatencyObjectStore {
    inner: InMemory,
    base_ttfb_ms: u64,
    jitter_ms: u64,
    tail_spike_call_idx: Option<usize>,
    tail_spike_ms: u64,
    total_http_range_gets: AtomicUsize,
    total_bytes_transferred: AtomicUsize,
    rng_state: AtomicU64,
}

impl fmt::Debug for SimulatedLatencyObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SimulatedLatencyObjectStore")
            .field("base_ttfb_ms", &self.base_ttfb_ms)
            .field("jitter_ms", &self.jitter_ms)
            .field("tail_spike_ms", &self.tail_spike_ms)
            .field(
                "total_http_range_gets",
                &self.total_http_range_gets.load(Ordering::Relaxed),
            )
            .field(
                "total_bytes_transferred",
                &self.total_bytes_transferred.load(Ordering::Relaxed),
            )
            .finish()
    }
}

impl fmt::Display for SimulatedLatencyObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SimulatedLatencyObjectStore(InMemory)")
    }
}

impl SimulatedLatencyObjectStore {
    fn new(base_ttfb_ms: u64, jitter_ms: u64) -> Self {
        Self {
            inner: InMemory::new(),
            base_ttfb_ms,
            jitter_ms,
            tail_spike_call_idx: None,
            tail_spike_ms: 0,
            total_http_range_gets: AtomicUsize::new(0),
            total_bytes_transferred: AtomicUsize::new(0),
            rng_state: AtomicU64::new(0x9e37_79b9_7f4a_7c15),
        }
    }

    fn with_tail_spike(
        base_ttfb_ms: u64,
        jitter_ms: u64,
        spike_on_call_idx: usize,
        tail_spike_ms: u64,
    ) -> Self {
        Self {
            inner: InMemory::new(),
            base_ttfb_ms,
            jitter_ms,
            tail_spike_call_idx: Some(spike_on_call_idx),
            tail_spike_ms,
            total_http_range_gets: AtomicUsize::new(0),
            total_bytes_transferred: AtomicUsize::new(0),
            rng_state: AtomicU64::new(0x9e37_79b9_7f4a_7c15),
        }
    }

    fn next_jitter_ms(&self) -> u64 {
        if self.jitter_ms == 0 {
            return 0;
        }
        let mut x = self
            .rng_state
            .fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        (x ^ (x >> 31)) % (self.jitter_ms + 1)
    }

    async fn simulate_network_ttfb(&self, call_idx: usize) {
        let mut delay_ms = self.base_ttfb_ms.saturating_add(self.next_jitter_ms());
        if self.tail_spike_call_idx == Some(call_idx) {
            delay_ms = delay_ms.saturating_add(self.tail_spike_ms);
        }
        if delay_ms > 0 {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        }
    }

    fn http_range_gets(&self) -> usize {
        self.total_http_range_gets.load(Ordering::Relaxed)
    }

    fn bytes_transferred(&self) -> usize {
        self.total_bytes_transferred.load(Ordering::Relaxed)
    }
}

impl ObjectStore for SimulatedLatencyObjectStore {
    fn put_opts<'life0, 'life1, 'async_trait>(
        &'life0 self,
        location: &'life1 ObjectPath,
        payload: PutPayload,
        opts: PutOptions,
    ) -> BoxFut<'async_trait, ObjectStoreResult<PutResult>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move { self.inner.put_opts(location, payload, opts).await })
    }

    fn put_multipart_opts<'life0, 'life1, 'async_trait>(
        &'life0 self,
        location: &'life1 ObjectPath,
        opts: PutMultipartOpts,
    ) -> BoxFut<'async_trait, ObjectStoreResult<Box<dyn MultipartUpload>>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move { self.inner.put_multipart_opts(location, opts).await })
    }

    fn get_opts<'life0, 'life1, 'async_trait>(
        &'life0 self,
        location: &'life1 ObjectPath,
        options: GetOptions,
    ) -> BoxFut<'async_trait, ObjectStoreResult<GetResult>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move { self.inner.get_opts(location, options).await })
    }

    fn get_range<'life0, 'life1, 'async_trait>(
        &'life0 self,
        location: &'life1 ObjectPath,
        range: Range<usize>,
    ) -> BoxFut<'async_trait, ObjectStoreResult<Bytes>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move {
            let call_idx = self.total_http_range_gets.fetch_add(1, Ordering::SeqCst);
            self.simulate_network_ttfb(call_idx).await;
            let bytes = self.inner.get_range(location, range).await?;
            self.total_bytes_transferred
                .fetch_add(bytes.len(), Ordering::Relaxed);
            Ok(bytes)
        })
    }

    fn get_ranges<'life0, 'life1, 'life2, 'async_trait>(
        &'life0 self,
        location: &'life1 ObjectPath,
        ranges: &'life2 [Range<usize>],
    ) -> BoxFut<'async_trait, ObjectStoreResult<Vec<Bytes>>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        'life2: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move {
            let call_idx = self
                .total_http_range_gets
                .fetch_add(ranges.len(), Ordering::SeqCst);
            self.simulate_network_ttfb(call_idx).await;
            let bufs = self.inner.get_ranges(location, ranges).await?;
            let total: usize = bufs.iter().map(Bytes::len).sum();
            self.total_bytes_transferred
                .fetch_add(total, Ordering::Relaxed);
            Ok(bufs)
        })
    }

    fn delete<'life0, 'life1, 'async_trait>(
        &'life0 self,
        location: &'life1 ObjectPath,
    ) -> BoxFut<'async_trait, ObjectStoreResult<()>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move { self.inner.delete(location).await })
    }

    fn list<'life0>(
        &'life0 self,
        prefix: Option<&ObjectPath>,
    ) -> BoxStream<'life0, ObjectStoreResult<ObjectMeta>> {
        self.inner.list(prefix)
    }

    fn list_with_delimiter<'life0, 'life1, 'async_trait>(
        &'life0 self,
        prefix: Option<&'life1 ObjectPath>,
    ) -> BoxFut<'async_trait, ObjectStoreResult<ListResult>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move { self.inner.list_with_delimiter(prefix).await })
    }

    fn copy<'life0, 'life1, 'life2, 'async_trait>(
        &'life0 self,
        from: &'life1 ObjectPath,
        to: &'life2 ObjectPath,
    ) -> BoxFut<'async_trait, ObjectStoreResult<()>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        'life2: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move { self.inner.copy(from, to).await })
    }

    fn copy_if_not_exists<'life0, 'life1, 'life2, 'async_trait>(
        &'life0 self,
        from: &'life1 ObjectPath,
        to: &'life2 ObjectPath,
    ) -> BoxFut<'async_trait, ObjectStoreResult<()>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        'life2: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move { self.inner.copy_if_not_exists(from, to).await })
    }
}

struct BenchDataset {
    raw_slab: Vec<u8>,
    zstd_dict: Option<Vec<u8>>,
    queries: Vec<(Vec<f32>, Vec<CandidatePageRef>)>,
}

fn build_tsp_dataset(
    n: usize,
    d: usize,
    k: usize,
    num_queries: usize,
    top_m: usize,
) -> OspreyResult<BenchDataset> {
    let mut centroids_f16 = Vec::with_capacity(k * d);
    let mut cluster_counts = vec![0_u64; k];
    let logical_perm: Vec<usize> = (0..k).map(|i| (i * 37 + 13) % k).collect();
    let mut inverse_pos = vec![0usize; k];
    for (pos, &logical_id) in logical_perm.iter().enumerate() {
        inverse_pos[logical_id] = pos;
    }

    for &inv_pos in inverse_pos.iter().take(k) {
        let t = inv_pos as f32 / k as f32;
        for dim in 0..d {
            let phase = (dim as f32) * 0.15;
            let val = (t * std::f32::consts::PI * 2.0 + phase).cos() * 0.8
                + (t * 4.0 + phase * 0.5).sin() * 0.2;
            centroids_f16.push(f16::from_f32(val));
        }
    }

    let mut vectors_f16 = Vec::with_capacity(n);
    let mut payloads = Vec::with_capacity(n);
    let mut doc_cluster = Vec::with_capacity(n);
    let mut state = 0x5eed_cafe_1234_5678_u64;

    for i in 0..n {
        let logical_c = i % k;
        cluster_counts[logical_c] += 1;
        doc_cluster.push(u32::try_from(logical_c).unwrap_or(0));

        let mut vec_i = Vec::with_capacity(d);
        for dim_idx in 0..d {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let jitter = (((state >> 40) as f32) / ((1_u64 << 24) as f32) - 0.5) * 0.08;
            let base = centroids_f16[logical_c * d + dim_idx].to_f32();
            vec_i.push(f16::from_f32(base + jitter));
        }
        vectors_f16.push(vec_i);
        payloads.push(
            format!(
                r#"{{"doc_id":{i},"cluster":{logical_c},"tenant":"osprey-bench","tier":"hot","region":"us-east-1","seq":{}}}"#,
                i * 7
            )
            .into_bytes(),
        );
    }

    let tsp_perm = solve_tsp_locality_largest_first(&centroids_f16, k, d, &cluster_counts, 50)?;
    let records: Vec<OsvVectorRecord<'_>> = (0..n)
        .map(|i| OsvVectorRecord {
            doc_id: i as u64,
            cluster_id: doc_cluster[i],
            vector_f16: &vectors_f16[i],
            payload: &payloads[i],
        })
        .collect();

    let writer = OsvWriter::new(OsvWriterConfig {
        minipage_target_bytes: 16 * 1024,
        zstd_level: 3,
        enable_zstd_dict: true,
        max_dict_bytes: 32 * 1024,
    });
    let mut raw_slab = Vec::new();
    let write_report = writer.write_slab(&mut raw_slab, &records, d, Some(&tsp_perm))?;
    let reader = OsvSlabReader::from_slice(&raw_slab)?;

    let zstd_dict = if reader.header().zstd_dict_len > 0 {
        let s = usize::try_from(reader.header().zstd_dict_offset)
            .map_err(|_| OspreyError::CorruptedFooter)?;
        let l = usize::try_from(reader.header().zstd_dict_len)
            .map_err(|_| OspreyError::CorruptedFooter)?;
        Some(raw_slab[s..s + l].to_vec())
    } else {
        None
    };

    let mut docs_by_physical_cluster: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (doc_idx, &logical_c) in doc_cluster.iter().enumerate() {
        let phys_c = usize::try_from(tsp_perm.to_physical(logical_c)?).unwrap_or(0);
        docs_by_physical_cluster[phys_c].push(doc_idx);
    }

    let mut queries = Vec::with_capacity(num_queries);
    for q_idx in 0..num_queries {
        let anchor_phys = (q_idx * 5) % (k - 18);
        let probe_clusters = anchor_phys..(anchor_phys + 18);

        let mut candidates = Vec::with_capacity(top_m);
        let mut step = 0usize;
        'outer: while candidates.len() < top_m {
            for phys_c in probe_clusters.clone() {
                let cluster_docs = &docs_by_physical_cluster[phys_c];
                if step < cluster_docs.len() {
                    let doc_idx = cluster_docs[step];
                    let doc_id = doc_idx as u64;
                    let (minipage_id, slot_idx) = write_report
                        .location_for_doc(doc_id)
                        .ok_or(OspreyError::CorruptedFooter)?;
                    candidates.push(CandidatePageRef {
                        doc_id,
                        minipage_id,
                        slot_idx,
                        approx_score_bits: 0,
                    });
                    if candidates.len() == top_m {
                        break 'outer;
                    }
                }
            }
            step += 1;
        }

        let center_logical =
            usize::try_from(tsp_perm.to_logical(u32::try_from(anchor_phys + 9).unwrap_or(0))?)
                .unwrap_or(0);
        let query_f32: Vec<f32> = (0..d)
            .map(|dim_idx| centroids_f16[center_logical * d + dim_idx].to_f32())
            .collect();
        queries.push((query_f32, candidates));
    }

    Ok(BenchDataset {
        raw_slab,
        zstd_dict,
        queries,
    })
}

#[tokio::test(flavor = "multi_thread")]
async fn range_planner_s3_tsp_integration_reduces_range_gets_10x_plus_with_exact_f16_and_lru_hits(
) -> OspreyResult<()> {
    const N: usize = 8_192;
    const D: usize = 128;
    const K: usize = 128;
    const NUM_QUERIES: usize = 20;
    const TOP_M: usize = 150;
    const TOP_K: usize = 10;

    let dataset = build_tsp_dataset(N, D, K, NUM_QUERIES, TOP_M)?;
    let reader = OsvSlabReader::from_slice(&dataset.raw_slab)?;
    let directory = reader.directory();

    let obj_path = ObjectPath::from("bench/segment_0.osv");
    let payload = Bytes::from(dataset.raw_slab);

    let baseline_sim = Arc::new(SimulatedLatencyObjectStore::new(0, 0));
    baseline_sim
        .put(&obj_path, PutPayload::from_bytes(payload.clone()))
        .await
        .map_err(|e| OspreyError::Io(std::io::Error::other(e)))?;
    let baseline_store: Arc<dyn ObjectStore> = baseline_sim.clone();

    let coalesce_sim = Arc::new(SimulatedLatencyObjectStore::new(12, 3));
    coalesce_sim
        .put(&obj_path, PutPayload::from_bytes(payload))
        .await
        .map_err(|e| OspreyError::Io(std::io::Error::other(e)))?;
    let coalesce_store: Arc<dyn ObjectStore> = coalesce_sim.clone();

    let no_coalesce_cfg = RangePlannerConfig {
        tau_req: 0.0,
        tau_byte: 1.0,
        max_coalesce_bytes: 1,
    };
    let remote_cfg = RemoteStoreConfig {
        hedged_request_delay_ms: 0,
        ..RemoteStoreConfig::default()
    };

    let baseline_stage = RefineStage::new(1, no_coalesce_cfg, dataset.zstd_dict.clone());
    let coalesce_stage = RefineStage::new(0, RangePlannerConfig::default(), dataset.zstd_dict);
    let hot_cache = MinipageLruCache::new(64 * 1024 * 1024);

    let mut total_raw_page_requests = 0usize;
    let mut total_coalesced_range_gets = 0usize;
    let mut max_coalesce_plan_us = 0_u64;

    for (query_f32, candidates) in &dataset.queries {
        // Without DP coalescing: 1 Range-GET per distinct minipage.
        let uncached_base = MinipageLruCache::new(0);
        let (base_res, base_stats) = baseline_stage
            .refine_async(
                query_f32,
                candidates,
                TOP_K,
                directory,
                &uncached_base,
                &baseline_store,
                &obj_path,
                &remote_cfg,
            )
            .await?;
        assert_eq!(
            base_stats.coalesced_range_gets,
            base_stats.raw_page_requests
        );

        // With DP coalescing + TSP (cold pass per query for pure coalescing telemetry).
        let uncached_coalesce = MinipageLruCache::new(0);
        let (coalesce_res, coalesce_stats) = coalesce_stage
            .refine_async(
                query_f32,
                candidates,
                TOP_K,
                directory,
                &uncached_coalesce,
                &coalesce_store,
                &obj_path,
                &remote_cfg,
            )
            .await?;

        assert_eq!(coalesce_res, base_res);
        assert!(
            coalesce_stats.range_plan_time_us < 200,
            "DP coalescing plan time {} us exceeded 200 us budget",
            coalesce_stats.range_plan_time_us
        );

        total_raw_page_requests += coalesce_stats.raw_page_requests;
        total_coalesced_range_gets += coalesce_stats.coalesced_range_gets;
        max_coalesce_plan_us = max_coalesce_plan_us.max(coalesce_stats.range_plan_time_us);

        // Warm shared LRU cache on first pass.
        let _ = coalesce_stage
            .refine_async(
                query_f32,
                candidates,
                TOP_K,
                directory,
                &hot_cache,
                &coalesce_store,
                &obj_path,
                &remote_cfg,
            )
            .await?;
    }

    let reduction_ratio = total_raw_page_requests as f64 / total_coalesced_range_gets as f64;
    println!(
        "\n[Phase 9 DP coalescing+S3 Integration] queries={NUM_QUERIES} | no_coalesce_http_gets={} | coalesce_http_gets={total_coalesced_range_gets} | reduction={reduction_ratio:.2}x | max_coalesce_plan_us={max_coalesce_plan_us} | s3_bytes_transferred={}",
        baseline_sim.http_range_gets(),
        coalesce_sim.bytes_transferred()
    );

    assert_eq!(baseline_sim.http_range_gets(), total_raw_page_requests);
    assert!(
        reduction_ratio >= 10.0,
        "Expected >= 10x Range-GET reduction with DP coalescing + TSP, got {reduction_ratio:.2}x"
    );

    // Second pass over the same 20 queries must hit MinipageLruCache 100% with 0 S3 calls.
    let gets_before_warm = coalesce_sim.http_range_gets();
    for (query_f32, candidates) in &dataset.queries {
        let (_res, warm_stats) = coalesce_stage
            .refine_async(
                query_f32,
                candidates,
                TOP_K,
                directory,
                &hot_cache,
                &coalesce_store,
                &obj_path,
                &remote_cfg,
            )
            .await?;
        assert_eq!(warm_stats.cache_hits, warm_stats.raw_page_requests);
        assert_eq!(warm_stats.coalesced_range_gets, 0);
    }
    assert_eq!(coalesce_sim.http_range_gets(), gets_before_warm);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn hedged_requests_cut_tail_latency_on_simulated_50ms_s3_stall() -> OspreyResult<()> {
    let orig_dim = 4usize;
    let v0 = [f16::from_f32(0.4), f16::from_f32(0.1), f16::ZERO, f16::ZERO];
    let v1 = [f16::from_f32(0.9), f16::from_f32(0.2), f16::ZERO, f16::ZERO];
    let records = [
        OsvVectorRecord {
            doc_id: 10,
            cluster_id: 0,
            vector_f16: &v0,
            payload: br#"{"doc":10}"#,
        },
        OsvVectorRecord {
            doc_id: 20,
            cluster_id: 1,
            vector_f16: &v1,
            payload: br#"{"doc":20}"#,
        },
    ];

    let mut raw_slab = Vec::new();
    let report = OsvWriter::new(OsvWriterConfig {
        enable_zstd_dict: false,
        ..OsvWriterConfig::default()
    })
    .write_slab(&mut raw_slab, &records, orig_dim, None)?;
    let reader = OsvSlabReader::from_slice(&raw_slab)?;

    // Inject a 60 ms tail spike on primary call #0; speculative hedge fires at 15 ms with 5 ms TTFB.
    // Primary call #0 stalls for 155ms (5ms base + 150ms spike); hedged call #1 completes in 5ms.
    // Using a 150ms spike keeps the assertion robust against Windows' 15.625ms timer tick granularity.
    let sim = Arc::new(SimulatedLatencyObjectStore::with_tail_spike(5, 0, 0, 150));
    let obj_path = ObjectPath::from("hedge/seg0.osv");
    sim.put(&obj_path, PutPayload::from_bytes(Bytes::from(raw_slab)))
        .await
        .map_err(|e| OspreyError::Io(std::io::Error::other(e)))?;
    let store: Arc<dyn ObjectStore> = sim.clone();

    let candidates: Vec<CandidatePageRef> = [10_u64, 20_u64]
        .into_iter()
        .filter_map(|doc_id| {
            report
                .location_for_doc(doc_id)
                .map(|(minipage_id, slot_idx)| CandidatePageRef {
                    doc_id,
                    minipage_id,
                    slot_idx,
                    approx_score_bits: 0,
                })
        })
        .collect();

    let cache = MinipageLruCache::new(64 * 1024);
    let stage = RefineStage::new(0, RangePlannerConfig::default(), None);
    let remote_cfg = RemoteStoreConfig {
        hedged_request_delay_ms: 15,
        request_timeout_ms: 500,
        ..RemoteStoreConfig::default()
    };

    let t0 = Instant::now();
    let (results, stats) = stage
        .refine_async(
            &[1.0_f32, 0.0, 0.0, 0.0],
            &candidates,
            1,
            reader.directory(),
            &cache,
            &store,
            &obj_path,
            &remote_cfg,
        )
        .await?;
    let elapsed = t0.elapsed();

    println!(
        "[Phase 9 Hedged Request] elapsed_ms={:.2} (primary_stall=155ms, hedge_delay=15ms) | total_http_range_gets={}",
        elapsed.as_secs_f64() * 1000.0,
        sim.http_range_gets()
    );

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].doc_id, 20);
    assert_eq!(stats.coalesced_range_gets, 1);
    assert_eq!(sim.http_range_gets(), 2);
    assert!(
        elapsed < Duration::from_millis(100),
        "Hedged request took {:?}, failing to beat 155ms primary tail spike",
        elapsed
    );
    Ok(())
}
