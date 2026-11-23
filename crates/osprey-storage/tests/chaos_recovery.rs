//! Chaos recovery integration tests for `ChaosObjectStore` + DP range coalescing + `RefineStage`.
//!
//! Validates:
//! 1. Deterministic HTTP 500 / 503 (`object_store::Error::Generic`) injection on the first `N`
//!    attempts and full recovery via full-jitter exponential backoff retries;
//! 2. Tail-latency spikes (`120 ms`) triggering speculative P99 hedged requests that win cleanly;
//! 3. Partial (truncated) responses and in-transit byte corruptions caught safely by
//!    `page_xxh64` / bounds checks (`OspreyError::ChecksumMismatch` / `BufferTooSmall`) with
//!    zero panics across `get_opts`, `get_range`, and `get_ranges`;
//! 4. End-to-end 100% query recovery with zero data loss under a 30% HTTP 503 failure rate
//!    combined with `120 ms` tail-latency spikes across a multi-query TSP + DP range coalescing workload.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use half::f16;
use object_store::memory::InMemory;
use object_store::path::Path as ObjectPath;
use object_store::{GetOptions, ObjectStore, PutPayload};
use osprey_core::math::SplitMix64;
use osprey_core::{OspreyError, OspreyResult};
use osprey_storage::{
    solve_tsp_locality_largest_first, CandidatePageRef, ChaosConfig, ChaosObjectStore,
    MinipageLruCache, OsvSlabReader, OsvVectorRecord, OsvWriter, OsvWriterConfig,
    RangePlannerConfig, RefineStage, RemoteStoreConfig,
};

struct ChaosDataset {
    raw_slab: Vec<u8>,
    zstd_dict: Option<Vec<u8>>,
    queries: Vec<(Vec<f32>, Vec<CandidatePageRef>)>,
}

fn build_chaos_dataset(
    n: usize,
    d: usize,
    k: usize,
    num_queries: usize,
    top_m: usize,
) -> OspreyResult<ChaosDataset> {
    let mut centroids_f16 = Vec::with_capacity(k * d);
    let mut cluster_counts = vec![0_u64; k];
    let mut rng = SplitMix64::new(0x1234_5678_9ABC_DEF0);

    for c in 0..k {
        let phase_base = (c as f32) / (k as f32);
        for dim in 0..d {
            let phase = (dim as f32) * 0.17;
            let val = (phase_base * std::f32::consts::PI * 2.0 + phase).cos() * 0.75
                + (phase_base * 3.0 + phase * 0.5).sin() * 0.25;
            centroids_f16.push(f16::from_f32(val));
        }
    }

    let mut vectors_f16 = Vec::with_capacity(n);
    let mut payloads = Vec::with_capacity(n);
    let mut doc_cluster = Vec::with_capacity(n);

    for i in 0..n {
        let logical_c = i % k;
        cluster_counts[logical_c] += 1;
        doc_cluster.push(u32::try_from(logical_c).unwrap());

        let mut vec_i = Vec::with_capacity(d);
        for dim_idx in 0..d {
            let unit = ((rng.next_u64() >> 40) as f32) / ((1_u64 << 24) as f32) - 0.5;
            let base = centroids_f16[logical_c * d + dim_idx].to_f32();
            vec_i.push(f16::from_f32(base + unit * 0.1));
        }
        vectors_f16.push(vec_i);
        payloads.push(
            format!(
                r#"{{"doc_id":{i},"cluster":{logical_c},"tenant":"chaos-test","seq":{}}}"#,
                i * 11
            )
            .into_bytes(),
        );
    }

    let tsp_perm = solve_tsp_locality_largest_first(&centroids_f16, k, d, &cluster_counts, 30)?;
    let records: Vec<OsvVectorRecord<'_>> = (0..n)
        .map(|i| OsvVectorRecord {
            doc_id: i as u64,
            cluster_id: doc_cluster[i],
            vector_f16: &vectors_f16[i],
            payload: &payloads[i],
        })
        .collect();

    let writer = OsvWriter::new(OsvWriterConfig {
        minipage_target_bytes: 2 * 1024,
        zstd_level: 3,
        enable_zstd_dict: true,
        max_dict_bytes: 16 * 1024,
    });
    let mut raw_slab = Vec::new();
    let write_report = writer.write_slab(&mut raw_slab, &records, d, Some(&tsp_perm))?;
    let reader = OsvSlabReader::from_slice(&raw_slab)?;

    let zstd_dict = if reader.header().zstd_dict_len > 0 {
        let s = usize::try_from(reader.header().zstd_dict_offset).unwrap();
        let l = usize::try_from(reader.header().zstd_dict_len).unwrap();
        Some(raw_slab[s..s + l].to_vec())
    } else {
        None
    };

    let mut docs_by_physical_cluster: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (doc_idx, &logical_c) in doc_cluster.iter().enumerate() {
        let phys_c = usize::try_from(tsp_perm.to_physical(logical_c)?).unwrap();
        docs_by_physical_cluster[phys_c].push(doc_idx);
    }

    let window = k.min(6);
    let mut queries = Vec::with_capacity(num_queries);
    for q_idx in 0..num_queries {
        let anchor_phys = (q_idx * 3) % (k - window + 1).max(1);
        let probe_clusters = anchor_phys..(anchor_phys + window);

        let mut candidates = Vec::with_capacity(top_m);
        let mut step = 0usize;
        'outer: while candidates.len() < top_m {
            for phys_c in probe_clusters.clone() {
                let cluster_docs = &docs_by_physical_cluster[phys_c];
                if step < cluster_docs.len() {
                    let doc_id = cluster_docs[step] as u64;
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
            usize::try_from(tsp_perm.to_logical(u32::try_from(anchor_phys + window / 2).unwrap())?)
                .unwrap();
        let query_f32: Vec<f32> = (0..d)
            .map(|dim_idx| centroids_f16[center_logical * d + dim_idx].to_f32())
            .collect();
        queries.push((query_f32, candidates));
    }

    Ok(ChaosDataset {
        raw_slab,
        zstd_dict,
        queries,
    })
}

fn make_small_slab() -> OspreyResult<(Vec<u8>, Vec<CandidatePageRef>)> {
    let orig_dim = 4usize;
    let v0 = [
        f16::from_f32(0.25),
        f16::from_f32(0.1),
        f16::ZERO,
        f16::ZERO,
    ];
    let v1 = [
        f16::from_f32(0.95),
        f16::from_f32(0.2),
        f16::ZERO,
        f16::ZERO,
    ];
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

    Ok((raw_slab, candidates))
}

#[tokio::test(flavor = "multi_thread")]
async fn chaos_recovers_100_percent_from_first_n_http_500_503_errors() -> OspreyResult<()> {
    let (raw_slab, candidates) = make_small_slab()?;
    let reader = OsvSlabReader::from_slice(&raw_slab)?;
    let obj_path = ObjectPath::from("chaos/first_n.osv");

    let inner = InMemory::new();
    inner
        .put(&obj_path, PutPayload::from_bytes(Bytes::from(raw_slab)))
        .await
        .map_err(|e| OspreyError::Io(std::io::Error::other(e)))?;

    let chaos = Arc::new(ChaosObjectStore::new(
        inner,
        ChaosConfig {
            fail_first_n: 3,
            ..ChaosConfig::default()
        },
    ));
    let store: Arc<dyn ObjectStore> = chaos.clone();

    let stage = RefineStage::new(0, RangePlannerConfig::default(), None);
    let cache = MinipageLruCache::new(64 * 1024);
    let remote_cfg = RemoteStoreConfig {
        hedged_request_delay_ms: 0,
        max_retries: 4,
        ..RemoteStoreConfig::default()
    };

    let (results, stats) = stage
        .refine_async(
            &[1.0_f32, 0.0, 0.0, 0.0],
            &candidates,
            2,
            reader.directory(),
            &cache,
            &store,
            &obj_path,
            &remote_cfg,
        )
        .await?;

    let c_stats = chaos.stats();
    assert_eq!(c_stats.http_errors_injected, 3);
    assert_eq!(c_stats.total_reads, 4);
    assert_eq!(stats.coalesced_range_gets, 1);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].doc_id, 20);
    assert_eq!(results[1].doc_id, 10);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn chaos_hedged_request_beats_120ms_tail_latency_spike() -> OspreyResult<()> {
    let (raw_slab, candidates) = make_small_slab()?;
    let reader = OsvSlabReader::from_slice(&raw_slab)?;
    let obj_path = ObjectPath::from("chaos/tail_120ms.osv");

    let inner = InMemory::new();
    inner
        .put(&obj_path, PutPayload::from_bytes(Bytes::from(raw_slab)))
        .await
        .map_err(|e| OspreyError::Io(std::io::Error::other(e)))?;

    let chaos = Arc::new(ChaosObjectStore::new(
        inner,
        ChaosConfig {
            tail_latency_calls: vec![0],
            tail_latency_ms: 120,
            base_latency_ms: 2,
            ..ChaosConfig::default()
        },
    ));
    let store: Arc<dyn ObjectStore> = chaos.clone();

    let stage = RefineStage::new(0, RangePlannerConfig::default(), None);
    let cache = MinipageLruCache::new(64 * 1024);
    let remote_cfg = RemoteStoreConfig {
        hedged_request_delay_ms: 15,
        request_timeout_ms: 500,
        max_retries: 2,
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

    let c_stats = chaos.stats();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].doc_id, 20);
    assert_eq!(stats.coalesced_range_gets, 1);
    assert_eq!(c_stats.tail_spikes_injected, 1);
    assert!(c_stats.total_reads >= 2);
    assert!(
        elapsed < Duration::from_millis(95),
        "Hedged secondary request took {elapsed:?}, failing to beat 120 ms tail spike"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn chaos_detects_corrupted_bytes_and_partial_responses_without_panic() -> OspreyResult<()> {
    let (raw_slab, candidates) = make_small_slab()?;
    let reader = OsvSlabReader::from_slice(&raw_slab)?;
    let obj_path = ObjectPath::from("chaos/corrupt.osv");

    // 1. In-transit byte corruption -> caught by page_xxh64 before Zstd decompression.
    let inner_corrupt = InMemory::new();
    inner_corrupt
        .put(
            &obj_path,
            PutPayload::from_bytes(Bytes::from(raw_slab.clone())),
        )
        .await
        .map_err(|e| OspreyError::Io(std::io::Error::other(e)))?;

    let chaos_corrupt = Arc::new(ChaosObjectStore::new(
        inner_corrupt,
        ChaosConfig {
            corrupt_bytes_calls: vec![0],
            ..ChaosConfig::default()
        },
    ));
    let store_corrupt: Arc<dyn ObjectStore> = chaos_corrupt.clone();

    let stage = RefineStage::new(0, RangePlannerConfig::default(), None);
    let remote_cfg = RemoteStoreConfig {
        hedged_request_delay_ms: 0,
        ..RemoteStoreConfig::default()
    };

    let err = stage
        .refine_async(
            &[1.0_f32, 0.0, 0.0, 0.0],
            &candidates,
            2,
            reader.directory(),
            &MinipageLruCache::new(0),
            &store_corrupt,
            &obj_path,
            &remote_cfg,
        )
        .await
        .expect_err("byte corruption must fail page_xxh64 validation");
    assert!(
        matches!(err, OspreyError::ChecksumMismatch { .. }),
        "expected ChecksumMismatch, got {err:?}"
    );
    assert_eq!(chaos_corrupt.stats().corruptions_injected, 1);

    // 2. Partial (truncated) response -> caught cleanly with BufferTooSmall (no panic).
    let inner_partial = InMemory::new();
    inner_partial
        .put(
            &obj_path,
            PutPayload::from_bytes(Bytes::from(raw_slab.clone())),
        )
        .await
        .map_err(|e| OspreyError::Io(std::io::Error::other(e)))?;

    let chaos_partial = Arc::new(ChaosObjectStore::new(
        inner_partial,
        ChaosConfig {
            partial_response_calls: vec![0],
            ..ChaosConfig::default()
        },
    ));
    let store_partial: Arc<dyn ObjectStore> = chaos_partial.clone();

    let err_partial = stage
        .refine_async(
            &[1.0_f32, 0.0, 0.0, 0.0],
            &candidates,
            2,
            reader.directory(),
            &MinipageLruCache::new(0),
            &store_partial,
            &obj_path,
            &remote_cfg,
        )
        .await
        .expect_err("truncated response must fail cleanly with BufferTooSmall");
    assert!(
        matches!(err_partial, OspreyError::BufferTooSmall { .. }),
        "expected BufferTooSmall, got {err_partial:?}"
    );
    assert_eq!(chaos_partial.stats().partial_responses_injected, 1);

    // 3. Direct get_opts + get_range interception validation.
    let inner_direct = InMemory::new();
    let sample_payload = Bytes::from_static(b"0123456789abcdef_osprey_chaos_payload");
    inner_direct
        .put(&obj_path, PutPayload::from_bytes(sample_payload.clone()))
        .await
        .map_err(|e| OspreyError::Io(std::io::Error::other(e)))?;

    let chaos_direct = ChaosObjectStore::new(
        inner_direct,
        ChaosConfig {
            fail_first_n: 1,
            partial_response_calls: vec![1],
            corrupt_bytes_calls: vec![2],
            ..ChaosConfig::default()
        },
    );

    // Call 0 via get_opts -> HTTP 500/503 Generic error.
    let r0 = chaos_direct
        .get_opts(&obj_path, GetOptions::default())
        .await;
    assert!(matches!(r0, Err(object_store::Error::Generic { .. })));

    // Call 1 via get_range -> partial response strictly shorter than requested range.
    let r1 = chaos_direct
        .get_range(&obj_path, 0..sample_payload.len())
        .await
        .map_err(|e| OspreyError::Io(std::io::Error::other(e)))?;
    assert!(r1.len() < sample_payload.len());

    // Call 2 via get_opts -> corrupted bytes differing from ground truth.
    let r2 = chaos_direct
        .get_opts(&obj_path, GetOptions::default())
        .await
        .map_err(|e| OspreyError::Io(std::io::Error::other(e)))?
        .bytes()
        .await
        .map_err(|e| OspreyError::Io(std::io::Error::other(e)))?;
    assert_eq!(r2.len(), sample_payload.len());
    assert_ne!(r2, sample_payload);

    let d_stats = chaos_direct.stats();
    assert_eq!(d_stats.http_errors_injected, 1);
    assert_eq!(d_stats.partial_responses_injected, 1);
    assert_eq!(d_stats.corruptions_injected, 1);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn chaos_coalesce_30pct_http_503_and_tail_spikes_100pct_query_recovery() -> OspreyResult<()> {
    const N: usize = 2_048;
    const D: usize = 64;
    const K: usize = 32;
    const NUM_QUERIES: usize = 25;
    const TOP_M: usize = 80;
    const TOP_K: usize = 10;

    let dataset = build_chaos_dataset(N, D, K, NUM_QUERIES, TOP_M)?;
    let reader = OsvSlabReader::from_slice(&dataset.raw_slab)?;
    let directory = reader.directory();
    let obj_path = ObjectPath::from("chaos/coalesce_30pct.osv");
    let slab_bytes = Bytes::from(dataset.raw_slab);

    // Ground truth against clean InMemory store.
    let clean_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    clean_store
        .put(&obj_path, PutPayload::from_bytes(slab_bytes.clone()))
        .await
        .map_err(|e| OspreyError::Io(std::io::Error::other(e)))?;

    let stage = RefineStage::new(0, RangePlannerConfig::default(), dataset.zstd_dict);
    let clean_cfg = RemoteStoreConfig {
        hedged_request_delay_ms: 0,
        ..RemoteStoreConfig::default()
    };

    let mut expected_results = Vec::with_capacity(NUM_QUERIES);
    for (query_f32, candidates) in &dataset.queries {
        let (res, _) = stage
            .refine_async(
                query_f32,
                candidates,
                TOP_K,
                directory,
                &MinipageLruCache::new(0),
                &clean_store,
                &obj_path,
                &clean_cfg,
            )
            .await?;
        expected_results.push(res);
    }

    // Chaos store: 30% HTTP 500/503 transient failure rate + 20% 120 ms tail-latency spikes.
    let inner_chaos = InMemory::new();
    inner_chaos
        .put(&obj_path, PutPayload::from_bytes(slab_bytes))
        .await
        .map_err(|e| OspreyError::Io(std::io::Error::other(e)))?;

    let chaos = Arc::new(ChaosObjectStore::new(
        inner_chaos,
        ChaosConfig {
            seed: 0x4348_414F_535F_3330,
            http_error_rate: 0.30,
            tail_latency_rate: 0.20,
            tail_latency_ms: 120,
            base_latency_ms: 1,
            ..ChaosConfig::default()
        },
    ));
    let chaos_store: Arc<dyn ObjectStore> = chaos.clone();

    let chaos_remote_cfg = RemoteStoreConfig {
        hedged_request_delay_ms: 20,
        request_timeout_ms: 500,
        max_retries: 8,
        ..RemoteStoreConfig::default()
    };

    for (q_idx, (query_f32, candidates)) in dataset.queries.iter().enumerate() {
        let (actual_res, stats) = stage
            .refine_async(
                query_f32,
                candidates,
                TOP_K,
                directory,
                &MinipageLruCache::new(0),
                &chaos_store,
                &obj_path,
                &chaos_remote_cfg,
            )
            .await?;

        assert!(stats.coalesced_range_gets >= 1);
        assert_eq!(
            actual_res, expected_results[q_idx],
            "Query {q_idx} diverged from ground truth under 30% HTTP 503 chaos"
        );
    }

    let c_stats = chaos.stats();
    println!(
        "\n[Phase 11 Chaos Recovery] queries={NUM_QUERIES}/{} | total_reads={} | http_500_503={} | tail_spikes_120ms={}",
        NUM_QUERIES, c_stats.total_reads, c_stats.http_errors_injected, c_stats.tail_spikes_injected
    );
    assert!(
        c_stats.http_errors_injected >= 5,
        "expected significant HTTP 500/503 injections under 30% rate, got {}",
        c_stats.http_errors_injected
    );
    assert!(
        c_stats.tail_spikes_injected >= 1,
        "expected >= 1 120 ms tail spike under 20% rate"
    );

    Ok(())
}
