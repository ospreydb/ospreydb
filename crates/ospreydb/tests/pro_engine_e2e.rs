#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use object_store::memory::InMemory;
use object_store::ObjectStore;
use osprey_core::{RawVectorRecord, StorageEngineExt};
use osprey_storage::{
    MinipageLruCache, OsvWriterConfig, RangePlannerConfig, RemoteStoreConfig,
    DEFAULT_MINIPAGE_CACHE_BYTES,
};
use ospreydb::{
    ClusterManifest, CompactorConfig, CowSlabCompactor, HybridSearchEngine, LsmDeltaBuffer,
    ManifestStore, OspreyWal, TombstoneGarbageCollector, TombstoneGcConfig, WalConfig,
    DEFAULT_DELTA_FLUSH_THRESHOLD, DEFAULT_MANIFEST_KEY, DEFAULT_TOMBSTONE_GC_THRESHOLD,
};

static E2E_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_wal_path(tag: &str) -> PathBuf {
    let n = E2E_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "osprey_pro_e2e_{}_{}_{}.wal",
        tag,
        std::process::id(),
        n
    ))
}

fn make_unit_basis_vector(dim: usize, hot_idx: usize, secondary_idx: usize, w2: f32) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim];
    v[hot_idx % dim] = 1.0;
    v[secondary_idx % dim] += w2;
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut v {
        *x /= norm;
    }
    v
}

#[tokio::test]
async fn wal_crash_recovery_rebuilds_delta_buffer() {
    let wal_path = temp_wal_path("crash_recover");
    let _ = fs::remove_file(&wal_path);

    let dim = 64_usize;
    let v1 = make_unit_basis_vector(dim, 0, 1, 0.1);
    let v2 = make_unit_basis_vector(dim, 5, 6, 0.2);

    {
        let (wal, _) = OspreyWal::open(&wal_path, WalConfig::default()).unwrap();
        let wal = Arc::new(wal);
        let delta = LsmDeltaBuffer::new(dim, dim, 42, Vec::new(), Some(wal)).unwrap();
        delta
            .ingest_batch(&[
                RawVectorRecord {
                    doc_id: 11,
                    vector: &v1,
                    payload_json: br#"{"doc":11}"#,
                },
                RawVectorRecord {
                    doc_id: 22,
                    vector: &v2,
                    payload_json: br#"{"doc":22}"#,
                },
            ])
            .unwrap();
        delta.delete_docs(&[11]).unwrap();
    }

    // Simulate restart: reopen WAL, replay into a fresh LsmDeltaBuffer
    let (wal_reopened, replayed) = OspreyWal::open(&wal_path, WalConfig::default()).unwrap();
    assert_eq!(replayed.len(), 3);
    let delta_rebuilt =
        LsmDeltaBuffer::new(dim, dim, 42, Vec::new(), Some(Arc::new(wal_reopened))).unwrap();
    let applied = delta_rebuilt.recover_from_wal(&replayed).unwrap();
    assert_eq!(applied, 3);
    assert_eq!(delta_rebuilt.live_count().unwrap(), 1);
    assert!(delta_rebuilt.is_masked_in_base(11));

    let hits = delta_rebuilt.search_exact(&v2, 5).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, 22);
    assert_eq!(hits[0].payload_json, br#"{"doc":22}"#);

    let _ = fs::remove_file(&wal_path);
}

#[tokio::test]
async fn cow_compactor_stateless_discovery_and_hybrid_search_e2e() {
    let wal_path = temp_wal_path("hybrid_cow");
    let _ = fs::remove_file(&wal_path);

    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let manifest_store = ManifestStore::new(Arc::clone(&store), DEFAULT_MANIFEST_KEY);
    manifest_store
        .create_if_absent(&ClusterManifest::new_empty())
        .await
        .unwrap();

    let dim = 64_usize;
    let (wal, _) = OspreyWal::open(&wal_path, WalConfig::default()).unwrap();
    let wal = Arc::new(wal);

    // Writer node: ingest 200 base vectors and compact into S3 .osi + .osv
    let writer_delta =
        Arc::new(LsmDeltaBuffer::new(dim, dim, 1234, Vec::new(), Some(Arc::clone(&wal))).unwrap());
    let mut base_vecs = Vec::with_capacity(200);
    let mut payloads = Vec::with_capacity(200);
    for i in 0..200_usize {
        base_vecs.push(make_unit_basis_vector(dim, i % dim, (i * 3) % dim, 0.25));
        payloads.push(format!(r#"{{"id":{i},"gen":"base"}}"#).into_bytes());
    }
    let raw_batch: Vec<RawVectorRecord<'_>> = (0..200_usize)
        .map(|i| RawVectorRecord {
            doc_id: (i + 1) as u64,
            vector: &base_vecs[i],
            payload_json: &payloads[i],
        })
        .collect();
    // Exercise StorageEngineExt trait directly as well
    let ext: &dyn StorageEngineExt<Error = ospreydb::ProEngineError> = &*writer_delta;
    let _seq = ext.append_batch(&raw_batch).unwrap();
    assert_eq!(writer_delta.live_count().unwrap(), 200);

    let compactor = CowSlabCompactor::new(
        CompactorConfig {
            delta_flush_threshold: 150,
            target_clusters: 8,
            fwht_seed: 1234,
            osv_writer_config: OsvWriterConfig {
                enable_zstd_dict: false,
                ..OsvWriterConfig::default()
            },
            ..CompactorConfig::default()
        },
        Arc::clone(&store),
        manifest_store.clone(),
    );

    let outcome = compactor
        .compact_if_needed(&writer_delta, Some(&wal))
        .await
        .unwrap()
        .expect("compaction should trigger above threshold");
    assert_eq!(outcome.vectors_compacted, 200);
    assert_eq!(outcome.manifest_epoch, 2);
    assert_eq!(writer_delta.live_count().unwrap(), 0);
    // WAL was cleanly reset after CoW commit
    let (_, after_compact_wal) = OspreyWal::open(&wal_path, WalConfig::default()).unwrap();
    assert!(after_compact_wal.is_empty());

    // Two stateless reader nodes discover the new segment via manifest.json ETag polling
    let reader_a = HybridSearchEngine::new(
        Arc::clone(&writer_delta),
        Arc::clone(&store),
        manifest_store.clone(),
        Arc::new(MinipageLruCache::new(DEFAULT_MINIPAGE_CACHE_BYTES)),
        RemoteStoreConfig::default(),
        RangePlannerConfig::default(),
    );
    let reader_b = HybridSearchEngine::new(
        Arc::new(LsmDeltaBuffer::new(dim, dim, 1234, Vec::new(), None).unwrap()),
        Arc::clone(&store),
        manifest_store.clone(),
        Arc::new(MinipageLruCache::new(DEFAULT_MINIPAGE_CACHE_BYTES)),
        RemoteStoreConfig::default(),
        RangePlannerConfig::default(),
    );

    assert!(reader_a.refresh_from_manifest().await.unwrap());
    assert!(!reader_a.refresh_from_manifest().await.unwrap()); // unchanged ETag
    assert!(reader_b.refresh_from_manifest().await.unwrap());
    assert_eq!(reader_a.loaded_segment_count().unwrap(), 1);
    assert_eq!(reader_b.loaded_segment_count().unwrap(), 1);

    // Insert a real-time delta vector (doc 9999), update doc 1, and delete doc 2
    let live_vec = make_unit_basis_vector(dim, 17, 19, 0.35);
    let updated_doc1_vec = make_unit_basis_vector(dim, 29, 31, 0.15);
    let t_ingest = Instant::now();
    writer_delta
        .ingest_batch(&[
            RawVectorRecord {
                doc_id: 9999,
                vector: &live_vec,
                payload_json: br#"{"id":9999,"gen":"delta"}"#,
            },
            RawVectorRecord {
                doc_id: 1,
                vector: &updated_doc1_vec,
                payload_json: br#"{"id":1,"gen":"delta_v2"}"#,
            },
        ])
        .unwrap();
    writer_delta.delete_docs(&[2]).unwrap();
    let delta_hits = writer_delta.search_exact(&live_vec, 1).unwrap();
    let ingest_search_us = t_ingest.elapsed().as_secs_f64() * 1_000_000.0;
    assert_eq!(delta_hits[0].doc_id, 9999);
    eprintln!("real-time delta ingest+delete+search: {ingest_search_us:.1} us");

    // Hybrid search transparently merges base .osi/.osv with live delta buffer
    let hybrid_live = reader_a
        .search_hybrid_async(&live_vec, 8, 64, 5)
        .await
        .unwrap();
    assert!(!hybrid_live.is_empty());
    assert_eq!(hybrid_live[0].doc_id, 9999);
    assert_eq!(hybrid_live[0].payload_json, br#"{"id":9999,"gen":"delta"}"#);

    // Querying updated doc 1 returns the new delta_v2 payload, not the old base payload
    let hybrid_doc1 = reader_a
        .search_hybrid_async(&updated_doc1_vec, 8, 64, 3)
        .await
        .unwrap();
    assert_eq!(hybrid_doc1[0].doc_id, 1);
    assert_eq!(hybrid_doc1[0].payload_json, br#"{"id":1,"gen":"delta_v2"}"#);

    // Deleted doc 2 must never appear in hybrid search results
    let all_top = reader_a
        .search_hybrid_async(&base_vecs[1], 8, 200, 200)
        .await
        .unwrap();
    assert!(all_top.iter().all(|r| r.doc_id != 2));

    let _ = fs::remove_file(&wal_path);
}

#[tokio::test]
async fn background_compactor_and_tombstone_gc_rewrite_slabs_above_20_percent() {
    assert_eq!(DEFAULT_DELTA_FLUSH_THRESHOLD, 50_000);
    assert!((DEFAULT_TOMBSTONE_GC_THRESHOLD - 0.20).abs() < f64::EPSILON);

    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let manifest_store = ManifestStore::new(Arc::clone(&store), DEFAULT_MANIFEST_KEY);

    let dim = 64_usize;
    let delta = Arc::new(LsmDeltaBuffer::new(dim, dim, 777, Vec::new(), None).unwrap());

    let compactor = CowSlabCompactor::new(
        CompactorConfig {
            delta_flush_threshold: 100,
            target_clusters: 4,
            fwht_seed: 777,
            osv_writer_config: OsvWriterConfig {
                enable_zstd_dict: false,
                ..OsvWriterConfig::default()
            },
            poll_interval: Duration::from_millis(20),
            ..CompactorConfig::default()
        },
        Arc::clone(&store),
        manifest_store.clone(),
    );

    let handle = compactor.spawn_background(Arc::clone(&delta), None);

    let vecs: Vec<Vec<f32>> = (0..100)
        .map(|i| make_unit_basis_vector(dim, i % dim, (i + 7) % dim, 0.2))
        .collect();
    let payloads: Vec<Vec<u8>> = (0..100)
        .map(|i| format!(r#"{{"doc":{i}}}"#).into_bytes())
        .collect();
    let batch: Vec<RawVectorRecord<'_>> = (0..100)
        .map(|i| RawVectorRecord {
            doc_id: (i + 1) as u64,
            vector: &vecs[i],
            payload_json: &payloads[i],
        })
        .collect();

    delta.ingest_batch(&batch).unwrap();
    handle.notify_write();

    // Wait for background compactor to flush segment 1
    let mut flushed = false;
    for _ in 0..100 {
        if let Some(snap) = manifest_store.load().await.unwrap() {
            if !snap.manifest.segments.is_empty() {
                flushed = true;
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    handle.shutdown().await;
    assert!(flushed, "background compactor did not materialize segment");

    let snap = manifest_store.load().await.unwrap().unwrap();
    assert_eq!(snap.manifest.segments.len(), 1);
    let seg1 = &snap.manifest.segments[0];
    assert_eq!(seg1.num_vectors, 100);

    let gc = TombstoneGarbageCollector::new(
        TombstoneGcConfig {
            tombstone_ratio_threshold: 0.20,
            delete_obsolete_objects: true,
            target_clusters: 4,
            osv_writer_config: OsvWriterConfig {
                enable_zstd_dict: false,
                ..OsvWriterConfig::default()
            },
            ..TombstoneGcConfig::default()
        },
        Arc::clone(&store),
        manifest_store.clone(),
    )
    .unwrap();

    // Mark 15% deleted (<= 20% threshold) -> GC must NOT rewrite
    let first_15: Vec<u64> = (1..=15).collect();
    gc.mark_tombstones_for_segment(seg1.segment_id, &first_15)
        .await
        .unwrap();
    let gc_noop = gc.collect_garbage().await.unwrap();
    assert_eq!(gc_noop.segments_rewritten, 0);

    // Mark 15 more deleted (total 30% > 20% threshold) -> GC MUST rewrite CoW slab
    let next_15: Vec<u64> = (16..=30).collect();
    gc.mark_tombstones_for_segment(seg1.segment_id, &next_15)
        .await
        .unwrap();
    let gc_report = gc.collect_garbage().await.unwrap();
    assert_eq!(gc_report.segments_scanned, 1);
    assert_eq!(gc_report.segments_rewritten, 1);
    assert_eq!(gc_report.tombstones_reclaimed, 30);

    let after_gc = manifest_store.load().await.unwrap().unwrap();
    assert_eq!(after_gc.manifest.segments.len(), 1);
    let new_seg = &after_gc.manifest.segments[0];
    assert_ne!(new_seg.segment_id, seg1.segment_id);
    assert_eq!(new_seg.num_vectors, 70);
    assert_eq!(new_seg.tombstone_count, 0);

    // Reader node discovers the GC-rewritten segment and queries surviving doc 50
    let reader = HybridSearchEngine::new(
        Arc::new(LsmDeltaBuffer::new(dim, dim, 777, Vec::new(), None).unwrap()),
        Arc::clone(&store),
        manifest_store,
        Arc::new(MinipageLruCache::new(DEFAULT_MINIPAGE_CACHE_BYTES)),
        RemoteStoreConfig::default(),
        RangePlannerConfig::default(),
    );
    assert!(reader.refresh_from_manifest().await.unwrap());
    let hits = reader
        .search_hybrid_async(&vecs[49], 4, 70, 5)
        .await
        .unwrap();
    assert!(hits.iter().any(|h| h.doc_id == 50));
    assert!(hits.iter().all(|h| h.doc_id > 30));
}
