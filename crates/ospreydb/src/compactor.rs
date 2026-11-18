//! Background Copy-on-Write `.osv`/`.osi` slab compactor (`Tarefa 164`) and
//! `RoaringBitmap` Tombstone Garbage Collector (`Tarefa 166`).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use half::f16;
use object_store::path::Path as ObjectPath;
use object_store::{ObjectStore, PutPayload};
use osprey_core::math::padded_dimension;
use osprey_core::{
    fit_binary_kmeans, AlignedBuffer, BinaryKMeansConfig, CompactionReport, OsiIndex,
    VectorLocation,
};
use osprey_simd::SimdEngine;
use osprey_storage::{
    parse_minipage, solve_tsp_locality, OsvSlabView, OsvVectorRecord, OsvWriter, OsvWriterConfig,
};
use roaring::RoaringBitmap;
use tokio::sync::Notify;
use tokio::task::JoinHandle;

use crate::delta_buffer::{FrozenDeltaSnapshot, LsmDeltaBuffer};
use crate::error::{ProEngineError, ProResult};
use crate::manifest::{ClusterManifest, ManifestStore, SegmentDescriptor};
use crate::wal::OspreyWal;

/// Default Delta Buffer vector count threshold before triggering CoW slab compaction.
pub const DEFAULT_DELTA_FLUSH_THRESHOLD: usize = 50_000;

/// Default tombstone ratio threshold (`20%`) above which GC rewrites `.osv` slabs.
pub const DEFAULT_TOMBSTONE_GC_THRESHOLD: f64 = 0.20;

static ARTIFACT_NONCE: AtomicU64 = AtomicU64::new(1);

/// Configuration for [`CowSlabCompactor`].
#[derive(Clone, Debug)]
pub struct CompactorConfig {
    /// Minimum live delta buffer vectors to trigger automatic background compaction (`default: 50_000`).
    pub delta_flush_threshold: usize,
    /// Target cluster count `K` per compacted segment (clamped to `1..=num_vectors`).
    pub target_clusters: usize,
    /// Deterministic FWHT seed for newly built segments.
    pub fwht_seed: u64,
    /// Object store prefix where `.osi` and `.osv` segment blobs are stored.
    pub segment_prefix: String,
    /// `.osv` slab writer configuration (16 KB minipages, Zstd + optional dict).
    pub osv_writer_config: OsvWriterConfig,
    /// Maximum CAS retry attempts when committing `manifest.json`.
    pub cas_max_retries: usize,
    /// Periodic background check interval when running as a spawned task.
    pub poll_interval: Duration,
}

impl Default for CompactorConfig {
    fn default() -> Self {
        Self {
            delta_flush_threshold: DEFAULT_DELTA_FLUSH_THRESHOLD,
            target_clusters: 64,
            fwht_seed: 0x4F53_5052_0001,
            segment_prefix: "segments".to_owned(),
            osv_writer_config: OsvWriterConfig::default(),
            cas_max_retries: 16,
            poll_interval: Duration::from_secs(5),
        }
    }
}

/// Result emitted by a Copy-on-Write compaction cycle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CowCompactionOutcome {
    /// Assigned segment ID in `manifest.json` (`0` if only tombstones were flushed).
    pub report: CompactionReport,
    /// New manifest epoch committed via S3 CAS.
    pub manifest_epoch: u64,
    /// Object key of the newly created `.osi` index (if vectors were compacted).
    pub osi_key: Option<String>,
    /// Object key of the newly created `.osv` slab (if vectors were compacted).
    pub osv_key: Option<String>,
    /// Number of vectors materialized into the new immutable segment.
    pub vectors_compacted: usize,
}

/// Handle controlling a running background [`CowSlabCompactor`] worker.
#[derive(Debug)]
pub struct CompactorHandle {
    notify: Arc<Notify>,
    shutdown: Arc<AtomicBool>,
    join_handle: JoinHandle<()>,
}

impl CompactorHandle {
    /// Signals the background compactor to check `delta_flush_threshold` immediately.
    pub fn notify_write(&self) {
        self.notify.notify_one();
    }

    /// Gracefully stops the background compactor task and awaits its completion.
    pub async fn shutdown(self) {
        self.shutdown.store(true, Ordering::Release);
        self.notify.notify_one();
        let _ = self.join_handle.await;
    }
}

/// Copy-on-Write slab compactor that freezes delta snapshots without blocking writers,
/// builds locality-ordered `.osv` + `.osi` segments in S3, and commits via `manifest.json` CAS.
#[derive(Clone, Debug)]
pub struct CowSlabCompactor {
    config: CompactorConfig,
    store: Arc<dyn ObjectStore>,
    manifest_store: ManifestStore,
    simd: SimdEngine,
}

impl CowSlabCompactor {
    /// Creates a new [`CowSlabCompactor`].
    #[must_use]
    pub fn new(
        config: CompactorConfig,
        store: Arc<dyn ObjectStore>,
        manifest_store: ManifestStore,
    ) -> Self {
        Self {
            config,
            store,
            manifest_store,
            simd: SimdEngine::new_auto(),
        }
    }

    /// Returns the compactor configuration.
    #[must_use]
    pub const fn config(&self) -> &CompactorConfig {
        &self.config
    }

    /// Checks `delta.live_count()` against `delta_flush_threshold` and runs CoW compaction
    /// only when the threshold is reached.
    pub async fn compact_if_needed(
        &self,
        delta: &LsmDeltaBuffer,
        wal: Option<&OspreyWal>,
    ) -> ProResult<Option<CowCompactionOutcome>> {
        if delta.live_count()? < self.config.delta_flush_threshold {
            return Ok(None);
        }
        self.compact_now(delta, wal).await
    }

    /// Unconditionally freezes the current delta buffer via Copy-on-Write, uploads new
    /// `.osv`/`.osi` segments to `ObjectStore`, and commits them atomically to `manifest.json`.
    pub async fn compact_now(
        &self,
        delta: &LsmDeltaBuffer,
        wal: Option<&OspreyWal>,
    ) -> ProResult<Option<CowCompactionOutcome>> {
        let Some(snapshot) = delta.take_snapshot_for_compaction()? else {
            return Ok(None);
        };

        let num_vectors = snapshot.doc_ids.len();
        let (built_opt, osi_key, osv_key) = if num_vectors > 0 {
            let built = build_segment_artifacts(
                &snapshot,
                self.config.target_clusters,
                self.config.fwht_seed,
                &self.config.osv_writer_config,
                &self.simd,
            )?;

            let tag = next_artifact_tag();
            let prefix = self.config.segment_prefix.trim_matches('/');
            let osi_key = if prefix.is_empty() {
                format!("seg-{tag}.osi")
            } else {
                format!("{prefix}/seg-{tag}.osi")
            };
            let osv_key = if prefix.is_empty() {
                format!("seg-{tag}.osv")
            } else {
                format!("{prefix}/seg-{tag}.osv")
            };

            // Upload immutable slabs via Copy-on-Write BEFORE publishing in manifest.json.
            self.store
                .put(
                    &ObjectPath::from(osv_key.as_str()),
                    PutPayload::from_bytes(Bytes::from(built.osv_bytes.clone())),
                )
                .await?;
            self.store
                .put(
                    &ObjectPath::from(osi_key.as_str()),
                    PutPayload::from_bytes(Bytes::from(built.osi_bytes.clone())),
                )
                .await?;

            (Some(built), Some(osi_key), Some(osv_key))
        } else {
            (None, None, None)
        };

        let deleted_to_apply = snapshot.tombstones;
        let mut upserted_set = RoaringBitmap::new();
        for &doc_id in &snapshot.doc_ids {
            if let Ok(id_u32) = u32::try_from(doc_id) {
                upserted_set.insert(id_u32);
            }
        }
        let mut mask_existing = deleted_to_apply.clone();
        mask_existing |= &upserted_set;

        let reclaimed_tombstones = deleted_to_apply.len();
        let mut assigned_segment_id = 0_u64;
        let mut existing_segments_count = 0_usize;

        let cas_snapshot = self
            .manifest_store
            .update_with_cas_retry(self.config.cas_max_retries, |manifest| {
                existing_segments_count = manifest.segments.len();
                if !mask_existing.is_empty() {
                    for existing_seg in &mut manifest.segments {
                        let mut bm = existing_seg.decode_tombstones()?;
                        bm |= &mask_existing;
                        existing_seg.set_tombstones(&bm)?;
                    }
                }
                if !upserted_set.is_empty() && !manifest.global_tombstones_b64.is_empty() {
                    let mut global = manifest.decode_global_tombstones()?;
                    global -= &upserted_set;
                    manifest.set_global_tombstones(&global)?;
                }

                if let (Some(built), Some(osi_k), Some(osv_k)) = (&built_opt, &osi_key, &osv_key) {
                    let seg_id = manifest.allocate_segment_id();
                    assigned_segment_id = seg_id;
                    let desc = SegmentDescriptor::new(
                        seg_id,
                        osi_k.clone(),
                        osv_k.clone(),
                        built.orig_dim,
                        built.padded_dim,
                        self.config.fwht_seed,
                        u64::try_from(num_vectors).unwrap_or(u64::MAX),
                        &RoaringBitmap::new(),
                    )?;
                    manifest.segments.push(desc);
                }
                Ok(())
            })
            .await?;

        if let Some(wal) = wal {
            wal.reset()?;
        }

        Ok(Some(CowCompactionOutcome {
            report: CompactionReport {
                merged_segments: existing_segments_count,
                reclaimed_tombstones,
                new_segment_id: assigned_segment_id,
            },
            manifest_epoch: cas_snapshot.manifest.epoch,
            osi_key,
            osv_key,
            vectors_compacted: num_vectors,
        }))
    }

    /// Spawns an asynchronous background compaction loop monitored by [`CompactorHandle`].
    #[must_use]
    pub fn spawn_background(
        self,
        delta: Arc<LsmDeltaBuffer>,
        wal: Option<Arc<OspreyWal>>,
    ) -> CompactorHandle {
        let notify = Arc::new(Notify::new());
        let shutdown = Arc::new(AtomicBool::new(false));

        let worker_notify = Arc::clone(&notify);
        let worker_shutdown = Arc::clone(&shutdown);
        let poll_interval = self.config.poll_interval;

        let join_handle = tokio::spawn(async move {
            while !worker_shutdown.load(Ordering::Acquire) {
                let _ = tokio::time::timeout(poll_interval, worker_notify.notified()).await;
                if worker_shutdown.load(Ordering::Acquire) {
                    break;
                }
                if let Err(err) = self.compact_if_needed(&delta, wal.as_deref()).await {
                    tracing::warn!(error = %err, "background CoW compaction cycle failed");
                }
            }
        });

        CompactorHandle {
            notify,
            shutdown,
            join_handle,
        }
    }
}

/// Configuration for [`TombstoneGarbageCollector`].
#[derive(Clone, Debug)]
pub struct TombstoneGcConfig {
    /// Minimum deleted ratio (`0.0..=1.0`) in a segment to trigger CoW rewrite (`default: 0.20`).
    pub tombstone_ratio_threshold: f64,
    /// Whether to delete superseded `.osi`/`.osv` objects after `manifest.json` CAS commit.
    pub delete_obsolete_objects: bool,
    /// Target cluster count for rewritten segments.
    pub target_clusters: usize,
    /// Object store prefix for newly rewritten segments.
    pub segment_prefix: String,
    /// `.osv` slab writer configuration.
    pub osv_writer_config: OsvWriterConfig,
    /// Maximum CAS retry attempts when updating `manifest.json`.
    pub cas_max_retries: usize,
}

impl Default for TombstoneGcConfig {
    fn default() -> Self {
        Self {
            tombstone_ratio_threshold: DEFAULT_TOMBSTONE_GC_THRESHOLD,
            delete_obsolete_objects: true,
            target_clusters: 64,
            segment_prefix: "segments".to_owned(),
            osv_writer_config: OsvWriterConfig::default(),
            cas_max_retries: 16,
        }
    }
}

/// Summary of a [`TombstoneGarbageCollector::collect_garbage`] execution.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TombstoneGcReport {
    /// Number of segments inspected in `manifest.json`.
    pub segments_scanned: usize,
    /// Number of fragmented segments rewritten via Copy-on-Write.
    pub segments_rewritten: usize,
    /// Number of segments dropped completely because 100% of their vectors were deleted.
    pub segments_dropped: usize,
    /// Total deleted vectors physically reclaimed.
    pub tombstones_reclaimed: u64,
    /// Total compressed `.osv` bytes saved across rewritten/dropped segments.
    pub osv_bytes_reclaimed: u64,
}

/// Background Garbage Collector that rewrites `.osv`/`.osi` segments whose `RoaringBitmap`
/// deletion ratio exceeds `tombstone_ratio_threshold` (default `> 20%`).
#[derive(Clone, Debug)]
pub struct TombstoneGarbageCollector {
    config: TombstoneGcConfig,
    store: Arc<dyn ObjectStore>,
    manifest_store: ManifestStore,
    simd: SimdEngine,
}

impl TombstoneGarbageCollector {
    /// Creates a new [`TombstoneGarbageCollector`].
    pub fn new(
        config: TombstoneGcConfig,
        store: Arc<dyn ObjectStore>,
        manifest_store: ManifestStore,
    ) -> ProResult<Self> {
        if !(0.0..1.0).contains(&config.tombstone_ratio_threshold) {
            return Err(ProEngineError::InvalidConfig(
                "tombstone_ratio_threshold must be in [0.0, 1.0)",
            ));
        }
        Ok(Self {
            config,
            store,
            manifest_store,
            simd: SimdEngine::new_auto(),
        })
    }

    /// Applies `deleted_doc_ids` to a segment's `RoaringBitmap` tombstones in `manifest.json`
    /// atomically via CAS.
    pub async fn mark_tombstones_for_segment(
        &self,
        segment_id: u64,
        deleted_doc_ids: &[u64],
    ) -> ProResult<ClusterManifest> {
        let snap = self
            .manifest_store
            .update_with_cas_retry(self.config.cas_max_retries, |manifest| {
                if let Some(seg) = manifest
                    .segments
                    .iter_mut()
                    .find(|s| s.segment_id == segment_id)
                {
                    let mut bm = seg.decode_tombstones()?;
                    for &doc_id in deleted_doc_ids {
                        if let Ok(id_u32) = u32::try_from(doc_id) {
                            bm.insert(id_u32);
                        }
                    }
                    seg.set_tombstones(&bm)?;
                }
                Ok(())
            })
            .await?;
        Ok(snap.manifest)
    }

    /// Scans `manifest.json` and rewrites every `.osv`/`.osi` segment whose deleted vector ratio
    /// strictly exceeds `tombstone_ratio_threshold`.
    pub async fn collect_garbage(&self) -> ProResult<TombstoneGcReport> {
        let Some(snapshot) = self.manifest_store.load().await? else {
            return Ok(TombstoneGcReport::default());
        };
        let manifest = snapshot.manifest;
        let global_tombstones = manifest.decode_global_tombstones()?;

        let mut report = TombstoneGcReport {
            segments_scanned: manifest.segments.len(),
            ..TombstoneGcReport::default()
        };

        for seg in &manifest.segments {
            let local_tombs = seg.decode_tombstones()?;
            let mut combined = local_tombs;
            combined |= &global_tombstones;

            // Fast skip when neither local tombstones nor global tombstones can exceed threshold.
            if combined.is_empty() {
                continue;
            }
            if global_tombstones.is_empty()
                && seg.tombstone_ratio() <= self.config.tombstone_ratio_threshold
            {
                continue;
            }

            let osv_path = ObjectPath::from(seg.osv_key.as_str());
            let old_osv_bytes = self.store.get(&osv_path).await?.bytes().await?;
            let aligned_osv = AlignedBuffer::<u8>::from_slice(&old_osv_bytes)?;
            let old_osv_len = u64::try_from(aligned_osv.len()).unwrap_or(0);
            let osv_view = OsvSlabView::from_bytes(aligned_osv.as_slice())?;
            let orig_dim = usize::from(osv_view.header.orig_dim);

            let mut total_docs = 0_u64;
            let mut deleted_docs = 0_u64;
            let mut live_doc_ids = Vec::new();
            let mut live_vectors_f32 = Vec::new();
            let mut live_payloads = Vec::new();
            let mut vec_scratch = vec![f16::ZERO; orig_dim];

            let num_pages = osv_view.directory.len();
            for page_id in 0..num_pages {
                let raw_page = osv_view.read_minipage(page_id)?;
                let page_view = parse_minipage(&raw_page)?;
                let count = usize::from(page_view.vector_count);
                for slot in 0..count {
                    total_docs = total_docs.saturating_add(1);
                    let doc_id = page_view.doc_id(slot)?;
                    let is_deleted =
                        u32::try_from(doc_id).is_ok_and(|id_u32| combined.contains(id_u32));
                    if is_deleted {
                        deleted_docs = deleted_docs.saturating_add(1);
                        continue;
                    }
                    page_view.read_vector_f16(slot, &mut vec_scratch)?;
                    let payload = page_view.payload(slot)?;

                    live_doc_ids.push(doc_id);
                    for &v in &vec_scratch {
                        live_vectors_f32.push(v.to_f32());
                    }
                    live_payloads.push(payload.to_vec());
                }
            }

            if total_docs == 0 {
                continue;
            }
            #[allow(clippy::cast_precision_loss)]
            let actual_ratio = (deleted_docs as f64) / (total_docs as f64);
            if actual_ratio <= self.config.tombstone_ratio_threshold {
                continue;
            }

            if live_doc_ids.is_empty() {
                // Drop empty segment atomically via CAS.
                let old_seg_id = seg.segment_id;
                let _ = self
                    .manifest_store
                    .update_with_cas_retry(self.config.cas_max_retries, |m| {
                        m.segments.retain(|s| s.segment_id != old_seg_id);
                        Ok(())
                    })
                    .await?;

                if self.config.delete_obsolete_objects {
                    let _ = self
                        .store
                        .delete(&ObjectPath::from(seg.osv_key.as_str()))
                        .await;
                    let _ = self
                        .store
                        .delete(&ObjectPath::from(seg.osi_key.as_str()))
                        .await;
                }

                report.segments_dropped += 1;
                report.tombstones_reclaimed =
                    report.tombstones_reclaimed.saturating_add(deleted_docs);
                report.osv_bytes_reclaimed = report.osv_bytes_reclaimed.saturating_add(old_osv_len);
                continue;
            }

            let live_snapshot = FrozenDeltaSnapshot {
                orig_dim,
                vectors_f32: live_vectors_f32,
                doc_ids: live_doc_ids,
                payloads: live_payloads,
                tombstones: RoaringBitmap::new(),
            };
            let built = build_segment_artifacts(
                &live_snapshot,
                self.config.target_clusters,
                seg.fwht_seed,
                &self.config.osv_writer_config,
                &self.simd,
            )?;

            let new_osv_len = u64::try_from(built.osv_bytes.len()).unwrap_or(0);
            let tag = next_artifact_tag();
            let prefix = self.config.segment_prefix.trim_matches('/');
            let new_osi_key = if prefix.is_empty() {
                format!("seg-{tag}.osi")
            } else {
                format!("{prefix}/seg-{tag}.osi")
            };
            let new_osv_key = if prefix.is_empty() {
                format!("seg-{tag}.osv")
            } else {
                format!("{prefix}/seg-{tag}.osv")
            };

            self.store
                .put(
                    &ObjectPath::from(new_osv_key.as_str()),
                    PutPayload::from_bytes(Bytes::from(built.osv_bytes)),
                )
                .await?;
            self.store
                .put(
                    &ObjectPath::from(new_osi_key.as_str()),
                    PutPayload::from_bytes(Bytes::from(built.osi_bytes)),
                )
                .await?;

            let old_seg_id = seg.segment_id;
            let live_count_u64 = u64::try_from(live_snapshot.doc_ids.len()).unwrap_or(u64::MAX);
            let new_osi_k = new_osi_key.clone();
            let new_osv_k = new_osv_key.clone();
            let fwht_seed = seg.fwht_seed;

            self.manifest_store
                .update_with_cas_retry(self.config.cas_max_retries, |m| {
                    let new_id = m.allocate_segment_id();
                    let new_desc = SegmentDescriptor::new(
                        new_id,
                        new_osi_k.clone(),
                        new_osv_k.clone(),
                        built.orig_dim,
                        built.padded_dim,
                        fwht_seed,
                        live_count_u64,
                        &RoaringBitmap::new(),
                    )?;
                    m.segments.retain(|s| s.segment_id != old_seg_id);
                    m.segments.push(new_desc);
                    Ok(())
                })
                .await?;

            if self.config.delete_obsolete_objects {
                let _ = self
                    .store
                    .delete(&ObjectPath::from(seg.osv_key.as_str()))
                    .await;
                let _ = self
                    .store
                    .delete(&ObjectPath::from(seg.osi_key.as_str()))
                    .await;
            }

            report.segments_rewritten += 1;
            report.tombstones_reclaimed = report.tombstones_reclaimed.saturating_add(deleted_docs);
            report.osv_bytes_reclaimed = report
                .osv_bytes_reclaimed
                .saturating_add(old_osv_len.saturating_sub(new_osv_len));
        }

        Ok(report)
    }
}

struct BuiltSegmentArtifacts {
    orig_dim: u16,
    padded_dim: u16,
    osi_bytes: Vec<u8>,
    osv_bytes: Vec<u8>,
}

fn build_segment_artifacts(
    snapshot: &FrozenDeltaSnapshot,
    target_clusters: usize,
    fwht_seed: u64,
    osv_writer_config: &OsvWriterConfig,
    simd: &SimdEngine,
) -> ProResult<BuiltSegmentArtifacts> {
    let n = snapshot.doc_ids.len();
    if n == 0 {
        return Err(ProEngineError::InvalidConfig(
            "cannot build segment artifacts from empty snapshot",
        ));
    }

    let orig_dim = snapshot.orig_dim;
    let padded_dim = padded_dimension(orig_dim)?;
    let num_clusters = target_clusters.max(1).min(n);

    let kmeans_cfg = BinaryKMeansConfig::new(num_clusters);
    let mut kmeans = fit_binary_kmeans(
        &snapshot.vectors_f32,
        n,
        orig_dim,
        fwht_seed,
        &kmeans_cfg,
        simd,
    )?;

    let perm = solve_tsp_locality(&kmeans.centroids_f16, num_clusters, padded_dim, 0, 16)?;
    let reordered_centroids =
        perm.reorder_centroids_f16_aligned(&kmeans.centroids_f16, padded_dim)?;
    let physical_assignments = perm.remap_cluster_ids(&kmeans.assignments)?;

    // Convert f32 vectors to f16 for `.osv` slab packing.
    let vectors_f16: Vec<f16> = snapshot
        .vectors_f32
        .iter()
        .map(|&v| f16::from_f32(v))
        .collect();

    let mut osv_records = Vec::with_capacity(n);
    for i in 0..n {
        osv_records.push(OsvVectorRecord {
            doc_id: snapshot.doc_ids[i],
            cluster_id: kmeans.assignments[i],
            vector_f16: &vectors_f16[i * orig_dim..(i + 1) * orig_dim],
            payload: &snapshot.payloads[i],
        });
    }

    let osv_writer = OsvWriter::new(osv_writer_config.clone());
    let (osv_bytes, osv_report) =
        osv_writer.write_slab_to_bytes(&osv_records, orig_dim, Some(&perm))?;

    let mut locations = Vec::with_capacity(n);
    for &doc_id in &snapshot.doc_ids {
        let (minipage_id, slot_idx) = osv_report
            .location_for_doc(doc_id)
            .ok_or(osprey_core::OspreyError::CorruptedFooter)?;
        locations.push(VectorLocation {
            slab_id: 0,
            minipage_id,
            slot_idx,
        });
    }

    // Align Binary K-Means centroids and assignments with TSP physical order for `.osi`.
    kmeans.centroids_f16 = reordered_centroids;
    kmeans.assignments = physical_assignments;

    let osi_index = OsiIndex::build(
        &snapshot.vectors_f32,
        &snapshot.doc_ids,
        &locations,
        &kmeans,
        orig_dim,
        fwht_seed,
        RoaringBitmap::new(),
        simd,
    )?;
    let osi_bytes = osi_index.to_aligned_bytes()?.as_slice().to_vec();

    let orig_dim_u16 = u16::try_from(orig_dim).map_err(|_| ProEngineError::DimensionMismatch {
        expected: usize::from(u16::MAX),
        actual: orig_dim,
    })?;
    let padded_dim_u16 =
        u16::try_from(padded_dim).map_err(|_| ProEngineError::DimensionMismatch {
            expected: usize::from(u16::MAX),
            actual: padded_dim,
        })?;

    Ok(BuiltSegmentArtifacts {
        orig_dim: orig_dim_u16,
        padded_dim: padded_dim_u16,
        osi_bytes,
        osv_bytes,
    })
}

fn next_artifact_tag() -> String {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let seq = ARTIFACT_NONCE.fetch_add(1, Ordering::Relaxed);
    format!("{ts}-{seq}")
}
