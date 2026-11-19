//! Real-time in-memory `LsmDeltaBuffer` (quantizing against active `.osi` `f16` centroids)
//! and concurrent `HybridSearchEngine` merging `.osi`/`.osv` base segments with live deltas.

use std::cmp::Ordering as CmpOrdering;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use half::f16;
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;
use osprey_core::format::osv::OSV_FLAG_ZSTD_ENABLED;
use osprey_core::math::{padded_dimension, prepare_and_pad_vector, RademacherSigns};
use osprey_core::{
    cast_minipage_directory, encode_prerotated_rabitq_residual_in_place, AlignedBuffer,
    CompactionReport, DeltaBufferExt, MinipageDirectoryEntry, OsiHeader, OsiIndex, OsiMappedIndex,
    OspreyError, OsvHeader, RabitqEntryHeader, RawVectorRecord, ScoredCandidate, Stage1Candidate,
    TransposedQuery4Bit,
};
use osprey_simd::{SimdBackend, SimdEngine};
use osprey_storage::{
    CandidatePageRef, MinipageLruCache, OsvSlabReader, RangePlannerConfig, RefineStage,
    RefinedSearchResult, RemoteStoreConfig,
};
use rayon::prelude::*;
use roaring::{RoaringBitmap, RoaringTreemap};

use crate::error::{ProEngineError, ProResult};
use crate::manifest::{ClusterManifest, ManifestStore};
use crate::metrics::ProMetricsRegistry;
use crate::wal::{OspreyWal, WalOpKind, WalRecord};

/// Minimum batch size before fanning out FWHT + RaBitQ encoding across `rayon` threads.
const PARALLEL_INGEST_THRESHOLD: usize = 32;

/// Single live or superseded entry stored inside [`LsmDeltaBuffer`].
#[derive(Clone, Debug)]
struct DeltaEntry {
    doc_id: u64,
    cluster_id: u32,
    res_norm: f16,
    l1_norm_factor: f16,
    seq_no: u64,
    vector_f16: Vec<f16>,
    payload_json: Vec<u8>,
}

#[derive(Debug, Default)]
struct DeltaBufferState {
    entries: Vec<DeltaEntry>,
    codes: Vec<u64>,
    superseded: Vec<bool>,
    doc_index: HashMap<u64, usize>,
    masked_in_base: Arc<RoaringTreemap>,
    deleted_docs: RoaringTreemap,
    deleted_seq: HashMap<u64, u64>,
    live_count: usize,
    next_seq_no: u64,
}

/// Owned immutable delta snapshot extracted via Copy-on-Write for compaction.
#[derive(Clone, Debug)]
pub struct FrozenDeltaSnapshot {
    /// Original unpadded embedding dimensionality.
    pub orig_dim: usize,
    /// Original `f32` vectors of live delta entries (`live_count * orig_dim`).
    pub vectors_f32: Vec<f32>,
    /// Live document IDs aligned with `vectors_f32`.
    pub doc_ids: Vec<u64>,
    /// Live JSON metadata payloads aligned with `doc_ids`.
    pub payloads: Vec<Vec<u8>>,
    /// Accumulated `RoaringTreemap` of deletions/updates to apply to older segments.
    pub tombstones: RoaringTreemap,
    /// Inclusive WAL/delta sequence watermark captured by this snapshot.
    pub watermark_seq: u64,
}

/// Thread-safe in-memory LSM Delta Buffer backed optionally by [`OspreyWal`].
///
/// New vectors are rotated via FWHT, assigned to the nearest existing `.osi` `f16` centroid,
/// and quantized into 1-bit `RaBitQ` codes + exact `f16` vectors so they become searchable
/// in `< 1 ms` without retraining `1-Bit Binary K-Means`.
pub struct LsmDeltaBuffer {
    orig_dim: usize,
    padded_dim: usize,
    words_per_vec: usize,
    fwht_seed: u64,
    signs: RademacherSigns,
    centroids_f16: Vec<f16>,
    num_clusters: usize,
    simd: SimdEngine,
    wal: Option<Arc<OspreyWal>>,
    live_count_atomic: AtomicUsize,
    has_masked_atomic: AtomicBool,
    state: RwLock<DeltaBufferState>,
}

impl std::fmt::Debug for LsmDeltaBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LsmDeltaBuffer")
            .field("orig_dim", &self.orig_dim)
            .field("padded_dim", &self.padded_dim)
            .field("fwht_seed", &self.fwht_seed)
            .field("num_clusters", &self.num_clusters)
            .finish_non_exhaustive()
    }
}

impl LsmDeltaBuffer {
    /// Creates an [`LsmDeltaBuffer`] using existing `.osi` `f16` centroids (`[K, padded_dim]`).
    ///
    /// If `centroids_f16` is empty, initializes a single neutral zero centroid so writes can
    /// start before the first immutable base segment is compacted.
    pub fn new(
        orig_dim: usize,
        padded_dim: usize,
        fwht_seed: u64,
        centroids_f16: Vec<f16>,
        wal: Option<Arc<OspreyWal>>,
    ) -> ProResult<Self> {
        let expected_padded = padded_dimension(orig_dim)?;
        if padded_dim != expected_padded {
            return Err(ProEngineError::DimensionMismatch {
                expected: expected_padded,
                actual: padded_dim,
            });
        }

        let (centroids_f16, num_clusters) = if centroids_f16.is_empty() {
            (vec![f16::ZERO; padded_dim], 1_usize)
        } else {
            if !centroids_f16.len().is_multiple_of(padded_dim) {
                return Err(ProEngineError::DimensionMismatch {
                    expected: padded_dim,
                    actual: centroids_f16.len() % padded_dim,
                });
            }
            let k = centroids_f16.len() / padded_dim;
            (centroids_f16, k)
        };

        let signs = RademacherSigns::from_seed(padded_dim, fwht_seed)?;
        Ok(Self {
            orig_dim,
            padded_dim,
            words_per_vec: padded_dim / 64,
            fwht_seed,
            signs,
            centroids_f16,
            num_clusters,
            simd: SimdEngine::new_auto(),
            wal,
            live_count_atomic: AtomicUsize::new(0),
            has_masked_atomic: AtomicBool::new(false),
            state: RwLock::new(DeltaBufferState {
                next_seq_no: 1,
                ..DeltaBufferState::default()
            }),
        })
    }

    /// Creates an [`LsmDeltaBuffer`] inheriting `orig_dim`, `padded_dim`, `fwht_seed`, and
    /// `f16` centroids directly from an active [`OsiIndex`].
    pub fn from_osi_index(index: &OsiIndex, wal: Option<Arc<OspreyWal>>) -> ProResult<Self> {
        let hdr = index.header();
        Self::new(
            usize::from(hdr.orig_dim),
            usize::from(hdr.padded_dim),
            hdr.fwht_seed,
            index.centroids_f16().to_vec(),
            wal,
        )
    }

    /// Returns the original unpadded vector dimensionality.
    #[must_use]
    pub const fn orig_dim(&self) -> usize {
        self.orig_dim
    }

    /// Returns the FWHT-padded dimensionality.
    #[must_use]
    pub const fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    /// Returns the deterministic FWHT Rademacher seed.
    #[must_use]
    pub const fn fwht_seed(&self) -> u64 {
        self.fwht_seed
    }

    /// Returns the count of live (non-superseded, non-deleted) entries in the delta buffer.
    pub fn live_count(&self) -> ProResult<usize> {
        let guard = self
            .state
            .read()
            .map_err(|_| ProEngineError::Io(std::io::Error::other("delta RwLock poisoned")))?;
        Ok(guard.live_count)
    }

    /// Lock-free fast check returning `true` when the delta buffer has zero live entries.
    #[inline]
    #[must_use]
    pub fn is_empty_fast(&self) -> bool {
        self.live_count_atomic.load(Ordering::Acquire) == 0
    }

    /// Returns `true` if `doc_id` has been updated or deleted in this delta buffer, meaning
    /// any older copy in base `.osi`/`.osv` segments must be suppressed.
    pub fn is_masked_in_base(&self, doc_id: u64) -> bool {
        if !self.has_masked_atomic.load(Ordering::Acquire) {
            return false;
        }
        let Ok(guard) = self.state.read() else {
            return false;
        };
        guard.masked_in_base.contains(doc_id)
    }

    /// Returns a cloned snapshot of all `doc_id`s tombstoned or superseded by this delta buffer.
    pub fn masked_tombstones(&self) -> ProResult<RoaringTreemap> {
        if !self.has_masked_atomic.load(Ordering::Acquire) {
            return Ok(RoaringTreemap::new());
        }
        let guard = self
            .state
            .read()
            .map_err(|_| ProEngineError::Io(std::io::Error::other("delta RwLock poisoned")))?;
        Ok((*guard.masked_in_base).clone())
    }

    /// Zero-copy `Arc` snapshot of masked base `doc_id`s, returning `Ok(None)` lock-free when empty.
    pub fn masked_tombstones_if_nonempty(&self) -> ProResult<Option<Arc<RoaringTreemap>>> {
        if !self.has_masked_atomic.load(Ordering::Acquire) {
            return Ok(None);
        }
        let guard = self
            .state
            .read()
            .map_err(|_| ProEngineError::Io(std::io::Error::other("delta RwLock poisoned")))?;
        if guard.masked_in_base.is_empty() {
            Ok(None)
        } else {
            Ok(Some(Arc::clone(&guard.masked_in_base)))
        }
    }

    /// Replays persisted [`WalRecord`]s into memory on startup without re-writing the WAL.
    pub fn recover_from_wal(&self, records: &[WalRecord]) -> ProResult<usize> {
        if records.is_empty() {
            return Ok(0);
        }

        let mut applied = 0_usize;
        for rec in records {
            match rec.op {
                WalOpKind::Upsert => {
                    let raw = [RawVectorRecord {
                        doc_id: rec.doc_id,
                        vector: &rec.vector,
                        payload_json: &rec.payload_json,
                    }];
                    self.ingest_in_memory(&raw, Some(rec.seq_no))?;
                    applied += 1;
                }
                WalOpKind::Delete => {
                    self.delete_in_memory(&[rec.doc_id], Some(rec.seq_no))?;
                    applied += 1;
                }
            }
        }
        Ok(applied)
    }

    /// Persists `records` to [`OspreyWal`] (if configured) and quantizes them into memory
    /// against existing `.osi` `f16` centroids so they are searchable in `< 1 ms`.
    pub fn ingest_batch(&self, records: &[RawVectorRecord<'_>]) -> ProResult<u64> {
        if records.is_empty() {
            let guard = self
                .state
                .read()
                .map_err(|_| ProEngineError::Io(std::io::Error::other("delta RwLock poisoned")))?;
            return Ok(guard.next_seq_no.saturating_sub(1));
        }

        for rec in records {
            if rec.vector.len() != self.orig_dim {
                return Err(ProEngineError::DimensionMismatch {
                    expected: self.orig_dim,
                    actual: rec.vector.len(),
                });
            }
        }

        let wal_seq = if let Some(wal) = &self.wal {
            Some(wal.append_upserts(records)?)
        } else {
            None
        };

        self.ingest_in_memory(records, wal_seq)
    }

    /// Persists logical deletions to [`OspreyWal`] (if configured) and masks `doc_ids`
    /// in both the delta buffer and older base segments.
    pub fn delete_docs(&self, doc_ids: &[u64]) -> ProResult<u64> {
        if doc_ids.is_empty() {
            let guard = self
                .state
                .read()
                .map_err(|_| ProEngineError::Io(std::io::Error::other("delta RwLock poisoned")))?;
            return Ok(guard.next_seq_no.saturating_sub(1));
        }

        let wal_seq = if let Some(wal) = &self.wal {
            Some(wal.append_deletes(doc_ids)?)
        } else {
            None
        };

        let last_seq = self.delete_in_memory(doc_ids, wal_seq)?;
        Ok(wal_seq.unwrap_or(last_seq))
    }

    fn encode_single_record(&self, rec: &RawVectorRecord<'_>) -> ProResult<(DeltaEntry, Vec<u64>)> {
        if rec.vector.len() != self.orig_dim {
            return Err(ProEngineError::DimensionMismatch {
                expected: self.orig_dim,
                actual: rec.vector.len(),
            });
        }

        let mut scratch = vec![0.0_f32; self.padded_dim];
        prepare_and_pad_vector(rec.vector, &mut scratch)?;
        self.simd
            .fwht_rotate_in_place(&mut scratch, self.signs.xor_masks())?;

        // Find nearest existing `.osi` f16 centroid in rotated space.
        let mut best_cluster = 0_u32;
        let mut best_dot = f32::NEG_INFINITY;
        for k in 0..self.num_clusters {
            let c_slice = &self.centroids_f16[k * self.padded_dim..(k + 1) * self.padded_dim];
            let mut dot = 0.0_f32;
            for (&x, &c) in scratch.iter().zip(c_slice.iter()) {
                dot += x * c.to_f32();
            }
            if dot > best_dot {
                best_dot = dot;
                best_cluster = u32::try_from(k).unwrap_or(0);
            }
        }

        let c_idx = usize::try_from(best_cluster).unwrap_or(0);
        let centroid = &self.centroids_f16[c_idx * self.padded_dim..(c_idx + 1) * self.padded_dim];
        let mut code_words = vec![0_u64; self.words_per_vec];
        let encoded =
            encode_prerotated_rabitq_residual_in_place(&mut scratch, centroid, &mut code_words)?;

        let vector_f16: Vec<f16> = rec.vector.iter().map(|&v| f16::from_f32(v)).collect();
        Ok((
            DeltaEntry {
                doc_id: rec.doc_id,
                cluster_id: best_cluster,
                res_norm: encoded.res_norm,
                l1_norm_factor: encoded.l1_norm_factor,
                seq_no: 0,
                vector_f16,
                payload_json: rec.payload_json.to_vec(),
            },
            code_words,
        ))
    }

    fn ingest_in_memory(
        &self,
        records: &[RawVectorRecord<'_>],
        wal_seq_end: Option<u64>,
    ) -> ProResult<u64> {
        let encoded_pairs: Vec<(DeltaEntry, Vec<u64>)> =
            if records.len() >= PARALLEL_INGEST_THRESHOLD {
                records
                    .par_iter()
                    .map(|rec| self.encode_single_record(rec))
                    .collect::<ProResult<Vec<_>>>()?
            } else {
                let mut out = Vec::with_capacity(records.len());
                for rec in records {
                    out.push(self.encode_single_record(rec)?);
                }
                out
            };

        let mut guard = self
            .state
            .write()
            .map_err(|_| ProEngineError::Io(std::io::Error::other("delta RwLock poisoned")))?;

        let mut seq = guard.next_seq_no;
        if let Some(end_seq) = wal_seq_end {
            let count_u64 = u64::try_from(encoded_pairs.len()).unwrap_or(0);
            seq = end_seq.saturating_sub(count_u64.saturating_sub(1)).max(1);
        }

        guard.entries.reserve(encoded_pairs.len());
        guard.superseded.reserve(encoded_pairs.len());
        guard
            .codes
            .reserve(encoded_pairs.len() * self.words_per_vec);

        for (mut entry, code_words) in encoded_pairs {
            entry.seq_no = seq;
            seq = seq.saturating_add(1);

            let slot = guard.entries.len();
            let doc_id = entry.doc_id;
            if let Some(prev_slot) = guard.doc_index.insert(doc_id, slot) {
                if !guard.superseded[prev_slot] {
                    guard.superseded[prev_slot] = true;
                    guard.live_count = guard.live_count.saturating_sub(1);
                }
            }
            Arc::make_mut(&mut guard.masked_in_base).insert(doc_id);
            guard.deleted_docs.remove(doc_id);
            guard.deleted_seq.remove(&doc_id);
            guard.entries.push(entry);
            guard.superseded.push(false);
            guard.codes.extend_from_slice(&code_words);
            guard.live_count = guard.live_count.saturating_add(1);
        }

        guard.next_seq_no = seq;
        self.live_count_atomic
            .store(guard.live_count, Ordering::Release);
        self.has_masked_atomic
            .store(!guard.masked_in_base.is_empty(), Ordering::Release);
        Ok(seq.saturating_sub(1))
    }

    fn delete_in_memory(&self, doc_ids: &[u64], wal_seq_end: Option<u64>) -> ProResult<u64> {
        let mut guard = self
            .state
            .write()
            .map_err(|_| ProEngineError::Io(std::io::Error::other("delta RwLock poisoned")))?;

        let state = &mut *guard;
        let mut seq = state.next_seq_no;
        if let Some(end_seq) = wal_seq_end {
            let count_u64 = u64::try_from(doc_ids.len()).unwrap_or(0);
            seq = end_seq.saturating_sub(count_u64.saturating_sub(1)).max(1);
        }

        let masked = Arc::make_mut(&mut state.masked_in_base);
        for &doc_id in doc_ids {
            let cur_seq = seq;
            seq = seq.saturating_add(1);
            masked.insert(doc_id);
            state.deleted_docs.insert(doc_id);
            state.deleted_seq.insert(doc_id, cur_seq);
            if let Some(slot) = state.doc_index.remove(&doc_id) {
                if !state.superseded[slot] {
                    state.superseded[slot] = true;
                    state.live_count = state.live_count.saturating_sub(1);
                }
            }
        }
        state.next_seq_no = seq;
        self.live_count_atomic
            .store(state.live_count, Ordering::Release);
        self.has_masked_atomic
            .store(!state.masked_in_base.is_empty(), Ordering::Release);
        Ok(seq.saturating_sub(1))
    }

    /// Stage-1 asymmetric `RaBitQ` search over live delta entries for an already-rotated query.
    pub fn search_stage1_rotated(
        &self,
        q_rot: &[f32],
        top_k: usize,
    ) -> ProResult<Vec<Stage1Candidate>> {
        if q_rot.len() != self.padded_dim {
            return Err(ProEngineError::DimensionMismatch {
                expected: self.padded_dim,
                actual: q_rot.len(),
            });
        }
        if top_k == 0 || self.is_empty_fast() {
            return Ok(Vec::new());
        }

        let guard = self
            .state
            .read()
            .map_err(|_| ProEngineError::Io(std::io::Error::other("delta RwLock poisoned")))?;

        if guard.live_count == 0 {
            return Ok(Vec::new());
        }

        let transposed = TransposedQuery4Bit::try_from_rotated_query(q_rot)?;
        let mut centroid_dots = vec![0.0_f32; self.num_clusters];
        for (k, dot_out) in centroid_dots.iter_mut().enumerate() {
            let c_slice = &self.centroids_f16[k * self.padded_dim..(k + 1) * self.padded_dim];
            *dot_out = osprey_simd::dot_f32_f16(q_rot, c_slice);
        }

        let padded_dim_u16 = u16::try_from(self.padded_dim).unwrap_or(u16::MAX);

        let mut candidates = Vec::with_capacity(guard.live_count);
        for (idx, entry) in guard.entries.iter().enumerate() {
            if guard.superseded[idx] {
                continue;
            }
            let code_start = idx * self.words_per_vec;
            let code = &guard.codes[code_start..code_start + self.words_per_vec];
            let dot_int4_sign = self.simd.rabitq_4bit_dot_u64(
                &transposed.planes,
                &transposed.sign_mask,
                transposed.query_abs_sum,
                code,
            );
            let hdr = RabitqEntryHeader {
                doc_id: entry.doc_id,
                minipage_id: u32::MAX,
                slab_id: u16::MAX,
                slot_idx: u16::try_from(idx).unwrap_or(u16::MAX),
                res_norm: entry.res_norm,
                l1_norm_factor: entry.l1_norm_factor,
                _reserved: 0,
            };
            let c_dot = centroid_dots
                .get(usize::try_from(entry.cluster_id).unwrap_or(0))
                .copied()
                .unwrap_or(0.0);
            let approx_score = OsiIndex::reconstruct_asymmetric_score(
                c_dot,
                &hdr,
                transposed.q_scale,
                dot_int4_sign,
                padded_dim_u16,
            );
            candidates.push(Stage1Candidate {
                doc_id: entry.doc_id,
                minipage_id: u32::MAX,
                slot_idx: u16::try_from(idx).unwrap_or(u16::MAX),
                approx_score,
            });
        }

        candidates.sort_by(|a, b| {
            b.approx_score
                .partial_cmp(&a.approx_score)
                .unwrap_or(CmpOrdering::Equal)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });
        candidates.truncate(top_k);
        Ok(candidates)
    }

    /// Exact `f16` search over all live delta records (Stage-1 RaBitQ pruning for large deltas,
    /// followed by exact `f16` dot-product refinement and payload extraction).
    pub fn search_exact(
        &self,
        query_f32: &[f32],
        top_k: usize,
    ) -> ProResult<Vec<RefinedSearchResult>> {
        if query_f32.len() != self.orig_dim {
            return Err(ProEngineError::DimensionMismatch {
                expected: self.orig_dim,
                actual: query_f32.len(),
            });
        }
        for (index, &val) in query_f32.iter().enumerate() {
            if !val.is_finite() {
                return Err(OspreyError::NonFiniteEmbedding { index }.into());
            }
        }
        if top_k == 0 || self.is_empty_fast() {
            return Ok(Vec::new());
        }

        let guard = self
            .state
            .read()
            .map_err(|_| ProEngineError::Io(std::io::Error::other("delta RwLock poisoned")))?;

        if guard.live_count == 0 {
            return Ok(Vec::new());
        }

        let mut q_rot = vec![0.0_f32; self.padded_dim];
        prepare_and_pad_vector(query_f32, &mut q_rot)?;
        self.simd
            .fwht_rotate_in_place(&mut q_rot, self.signs.xor_masks())?;
        let transposed = TransposedQuery4Bit::try_from_rotated_query(&q_rot)?;

        let mut centroid_dots = vec![0.0_f32; self.num_clusters];
        for (k, dot_out) in centroid_dots.iter_mut().enumerate() {
            let c_slice = &self.centroids_f16[k * self.padded_dim..(k + 1) * self.padded_dim];
            *dot_out = osprey_simd::dot_f32_f16(&q_rot, c_slice);
        }
        let padded_dim_u16 = u16::try_from(self.padded_dim).unwrap_or(u16::MAX);

        // For large delta buffers (> 256 live items), use 1-bit RaBitQ overfetch shortlist first;
        // for small deltas, refine every live entry directly for 100% recall.
        let shortlist_slots: Vec<usize> = if guard.live_count <= 256 {
            (0..guard.entries.len())
                .filter(|&i| !guard.superseded[i])
                .collect()
        } else {
            let overfetch = top_k.saturating_mul(16).max(128).min(guard.live_count);
            let mut stage1: Vec<(usize, f32)> = Vec::with_capacity(guard.live_count);
            for (idx, entry) in guard.entries.iter().enumerate() {
                if guard.superseded[idx] {
                    continue;
                }
                let code_start = idx * self.words_per_vec;
                let code = &guard.codes[code_start..code_start + self.words_per_vec];
                let dot_int4_sign = self.simd.rabitq_4bit_dot_u64(
                    &transposed.planes,
                    &transposed.sign_mask,
                    transposed.query_abs_sum,
                    code,
                );
                let hdr = RabitqEntryHeader {
                    doc_id: entry.doc_id,
                    minipage_id: u32::MAX,
                    slab_id: u16::MAX,
                    slot_idx: 0,
                    res_norm: entry.res_norm,
                    l1_norm_factor: entry.l1_norm_factor,
                    _reserved: 0,
                };
                let c_dot = centroid_dots
                    .get(usize::try_from(entry.cluster_id).unwrap_or(0))
                    .copied()
                    .unwrap_or(0.0);
                let approx = OsiIndex::reconstruct_asymmetric_score(
                    c_dot,
                    &hdr,
                    transposed.q_scale,
                    dot_int4_sign,
                    padded_dim_u16,
                );
                stage1.push((idx, approx));
            }
            stage1.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(CmpOrdering::Equal));
            stage1.truncate(overfetch);
            stage1.into_iter().map(|(idx, _)| idx).collect()
        };

        let mut refined = Vec::with_capacity(shortlist_slots.len());
        for idx in shortlist_slots {
            let entry = &guard.entries[idx];
            let exact_score = exact_f16_dot(query_f32, &entry.vector_f16);
            refined.push(RefinedSearchResult {
                doc_id: entry.doc_id,
                exact_score,
                minipage_id: u32::MAX,
                slot_idx: u16::try_from(idx).unwrap_or(u16::MAX),
                payload_json: entry.payload_json.clone(),
            });
        }

        refined.sort_by(|a, b| {
            b.exact_score
                .partial_cmp(&a.exact_score)
                .unwrap_or(CmpOrdering::Equal)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });
        refined.truncate(top_k);
        Ok(refined)
    }

    /// Non-destructive Copy-on-Write snapshot extraction for background slab compaction.
    ///
    /// Keeps all live entries and tombstones searchable in memory until
    /// [`Self::commit_compaction_snapshot`] is called after S3 upload, manifest CAS commit,
    /// and `HybridSearchEngine` segment refresh succeed (eliminating any invisibility window).
    pub fn snapshot_for_compaction(&self) -> ProResult<Option<FrozenDeltaSnapshot>> {
        let guard = self
            .state
            .read()
            .map_err(|_| ProEngineError::Io(std::io::Error::other("delta RwLock poisoned")))?;

        if guard.live_count == 0 && guard.masked_in_base.is_empty() {
            return Ok(None);
        }

        let watermark_seq = guard.next_seq_no.saturating_sub(1);
        let mut vectors_f32 = Vec::with_capacity(guard.live_count * self.orig_dim);
        let mut doc_ids = Vec::with_capacity(guard.live_count);
        let mut payloads = Vec::with_capacity(guard.live_count);

        for (idx, entry) in guard.entries.iter().enumerate() {
            if guard.superseded[idx] {
                continue;
            }
            doc_ids.push(entry.doc_id);
            for &val in &entry.vector_f16 {
                vectors_f32.push(val.to_f32());
            }
            payloads.push(entry.payload_json.clone());
        }

        Ok(Some(FrozenDeltaSnapshot {
            orig_dim: self.orig_dim,
            vectors_f32,
            doc_ids,
            payloads,
            tombstones: guard.deleted_docs.clone(),
            watermark_seq,
        }))
    }

    /// Commits a completed compaction up to `watermark_seq`, removing only entries and
    /// tombstones already materialized into the S3 segment while preserving concurrent writes
    /// that arrived with `seq_no > watermark_seq`.
    pub fn commit_compaction_snapshot(&self, watermark_seq: u64) -> ProResult<()> {
        let mut guard = self
            .state
            .write()
            .map_err(|_| ProEngineError::Io(std::io::Error::other("delta RwLock poisoned")))?;

        let old = std::mem::take(&mut *guard);
        guard.next_seq_no = old.next_seq_no;

        let mut remaining_entries = Vec::new();
        let mut remaining_codes = Vec::new();
        let mut remaining_superseded = Vec::new();
        let mut remaining_doc_index = HashMap::new();
        let mut remaining_masked = RoaringTreemap::new();
        let mut remaining_deleted = RoaringTreemap::new();
        let mut remaining_deleted_seq = HashMap::new();
        let mut remaining_live = 0_usize;

        for (idx, entry) in old.entries.into_iter().enumerate() {
            if old.superseded[idx] || entry.seq_no <= watermark_seq {
                continue;
            }
            let new_slot = remaining_entries.len();
            let doc_id = entry.doc_id;
            remaining_doc_index.insert(doc_id, new_slot);
            remaining_masked.insert(doc_id);
            remaining_entries.push(entry);
            remaining_superseded.push(false);
            let code_start = idx * self.words_per_vec;
            remaining_codes
                .extend_from_slice(&old.codes[code_start..code_start + self.words_per_vec]);
            remaining_live = remaining_live.saturating_add(1);
        }

        for (doc_id, del_seq) in old.deleted_seq {
            if del_seq > watermark_seq {
                remaining_masked.insert(doc_id);
                remaining_deleted.insert(doc_id);
                remaining_deleted_seq.insert(doc_id, del_seq);
            }
        }

        guard.entries = remaining_entries;
        guard.codes = remaining_codes;
        guard.superseded = remaining_superseded;
        guard.doc_index = remaining_doc_index;
        guard.masked_in_base = Arc::new(remaining_masked);
        guard.deleted_docs = remaining_deleted;
        guard.deleted_seq = remaining_deleted_seq;
        guard.live_count = remaining_live;
        self.live_count_atomic
            .store(guard.live_count, Ordering::Release);
        self.has_masked_atomic
            .store(!guard.masked_in_base.is_empty(), Ordering::Release);
        Ok(())
    }

    /// Copy-on-Write atomic snapshot extraction for background slab compaction.
    ///
    /// Swaps out the current buffer for a fresh empty state so concurrent writes proceed
    /// without blocking. Returns `None` if there are no live entries and no tombstones.
    pub fn take_snapshot_for_compaction(&self) -> ProResult<Option<FrozenDeltaSnapshot>> {
        let Some(snapshot) = self.snapshot_for_compaction()? else {
            return Ok(None);
        };
        self.commit_compaction_snapshot(snapshot.watermark_seq)?;
        Ok(Some(snapshot))
    }
}

impl DeltaBufferExt for LsmDeltaBuffer {
    type Error = ProEngineError;

    fn append_batch(&self, records: &[RawVectorRecord<'_>]) -> Result<u64, Self::Error> {
        self.ingest_batch(records)
    }

    fn search_deltas(
        &self,
        q_rot: &[f32],
        top_k: usize,
    ) -> Result<Vec<Stage1Candidate>, Self::Error> {
        self.search_stage1_rotated(q_rot, top_k)
    }

    fn compact(&self, segment_ids: &[u64]) -> Result<CompactionReport, Self::Error> {
        let snapshot = self.take_snapshot_for_compaction()?;
        let reclaimed = snapshot.as_ref().map_or(0, |s| s.tombstones.len());
        Ok(CompactionReport {
            merged_segments: segment_ids.len(),
            reclaimed_tombstones: reclaimed,
            new_segment_id: 0,
        })
    }
}

/// Resident `.osi` index representation supporting zero-copy `memmap2` or owned buffers.
#[derive(Debug)]
pub enum SegmentOsi {
    /// Zero-copy `Mmap` or single-copy `AlignedBuffer`-backed `.osi` index.
    Mapped(OsiMappedIndex),
    /// Owned heap-resident `.osi` index.
    Owned(OsiIndex),
}

impl SegmentOsi {
    /// Returns the validated `.osi` header.
    #[inline]
    #[must_use]
    pub fn header(&self) -> &OsiHeader {
        match self {
            Self::Mapped(m) => m.header(),
            Self::Owned(o) => o.header(),
        }
    }

    /// Returns the `.osi` tombstone bitmap.
    #[inline]
    #[must_use]
    pub fn tombstones(&self) -> &RoaringBitmap {
        match self {
            Self::Mapped(m) => m.tombstones(),
            Self::Owned(o) => o.tombstones(),
        }
    }

    /// Executes unfiltered Stage-1 asymmetric `RaBitQ` search.
    #[inline]
    pub fn search_candidates(
        &self,
        query: &[f32],
        nprobe: usize,
        top_m: usize,
        simd: &SimdEngine,
    ) -> osprey_core::OspreyResult<Vec<ScoredCandidate>> {
        match self {
            Self::Mapped(m) => m.search_candidates(query, nprobe, top_m, simd),
            Self::Owned(o) => o.search_candidates(query, nprobe, top_m, simd),
        }
    }

    /// Executes Stage-1 asymmetric `RaBitQ` search with a caller-supplied predicate.
    #[inline]
    pub fn search_candidates_filtered_with<F>(
        &self,
        query: &[f32],
        nprobe: usize,
        top_m: usize,
        simd: &SimdEngine,
        allow_entry: F,
    ) -> osprey_core::OspreyResult<Vec<ScoredCandidate>>
    where
        F: Fn(u32, u64) -> bool,
    {
        match self {
            Self::Mapped(m) => {
                m.search_candidates_filtered_with(query, nprobe, top_m, simd, allow_entry)
            }
            Self::Owned(o) => {
                o.search_candidates_filtered_with(query, nprobe, top_m, simd, allow_entry)
            }
        }
    }
}

/// Loaded immutable segment cached by a stateless reader node.
#[derive(Debug)]
pub struct LoadedSegment {
    /// Segment identifier from `manifest.json`.
    pub segment_id: u64,
    /// Object store path of the `.osv` slab.
    pub osv_path: ObjectPath,
    /// Parsed resident `.osi` 1-bit index (zero-copy `mmap` on `file://`).
    pub osi: Arc<SegmentOsi>,
    /// `.osv` minipage directory entries.
    pub osv_directory: Vec<MinipageDirectoryEntry>,
    /// `RefineStage` configured with `slab_id`, `RangePlannerConfig`, and optional Zstd dict.
    pub refine_stage: RefineStage,
    /// Segment-local tombstones from `manifest.json`.
    pub tombstones: RoaringTreemap,
}

/// Executes blocking CPU/mmap work safely on both `multi_thread` and `current_thread` Tokio runtimes.
#[inline]
fn block_in_place_safe<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread {
            return tokio::task::block_in_place(f);
        }
    }
    f()
}

/// Hybrid search coordinator combining immutable S3 `.osi`/`.osv` segments with the live
/// in-memory [`LsmDeltaBuffer`].
#[derive(Debug)]
pub struct HybridSearchEngine {
    delta: Arc<LsmDeltaBuffer>,
    store: Arc<dyn ObjectStore>,
    manifest_store: ManifestStore,
    cache: Arc<MinipageLruCache>,
    remote_config: RemoteStoreConfig,
    range_planner_config: RangePlannerConfig,
    local_store_root: Option<PathBuf>,
    simd: SimdEngine,
    segments: RwLock<Vec<Arc<LoadedSegment>>>,
    global_tombstones: RwLock<Arc<RoaringTreemap>>,
    known_etag: RwLock<Option<String>>,
    known_epoch: RwLock<Option<u64>>,
    metrics: Arc<ProMetricsRegistry>,
}

impl HybridSearchEngine {
    /// Creates a new [`HybridSearchEngine`] bound to `store`, `manifest_store`, and `delta`.
    #[must_use]
    pub fn new(
        delta: Arc<LsmDeltaBuffer>,
        store: Arc<dyn ObjectStore>,
        manifest_store: ManifestStore,
        cache: Arc<MinipageLruCache>,
        remote_config: RemoteStoreConfig,
        range_planner_config: RangePlannerConfig,
    ) -> Self {
        Self::with_metrics(
            delta,
            store,
            manifest_store,
            cache,
            remote_config,
            range_planner_config,
            Arc::new(ProMetricsRegistry::new()),
        )
    }

    /// Creates a [`HybridSearchEngine`] sharing an external [`ProMetricsRegistry`].
    #[must_use]
    pub fn with_metrics(
        delta: Arc<LsmDeltaBuffer>,
        store: Arc<dyn ObjectStore>,
        manifest_store: ManifestStore,
        cache: Arc<MinipageLruCache>,
        remote_config: RemoteStoreConfig,
        range_planner_config: RangePlannerConfig,
        metrics: Arc<ProMetricsRegistry>,
    ) -> Self {
        Self {
            delta,
            store,
            manifest_store,
            cache,
            remote_config,
            range_planner_config,
            local_store_root: None,
            simd: SimdEngine::new_auto(),
            segments: RwLock::new(Vec::new()),
            global_tombstones: RwLock::new(Arc::new(RoaringTreemap::new())),
            known_etag: RwLock::new(None),
            known_epoch: RwLock::new(None),
            metrics,
        }
    }

    /// Configures a local filesystem root directory (`file://` object store) to enable
    /// zero-copy `mmap` for `.osi` and direct `mmap`/`pread` fast-path for `.osv`.
    #[must_use]
    pub fn with_local_store_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.local_store_root = Some(root.into());
        self
    }

    /// Returns the Prometheus metrics registry bound to this search engine.
    #[must_use]
    pub fn metrics(&self) -> &Arc<ProMetricsRegistry> {
        &self.metrics
    }

    /// Returns the active [`ManifestStore`].
    #[must_use]
    pub fn manifest_store(&self) -> &ManifestStore {
        &self.manifest_store
    }

    /// Returns a reference to the active [`LsmDeltaBuffer`].
    #[must_use]
    pub fn delta_buffer(&self) -> &Arc<LsmDeltaBuffer> {
        &self.delta
    }

    /// Returns the number of immutable segments currently loaded by this reader node.
    pub fn loaded_segment_count(&self) -> ProResult<usize> {
        let guard = self
            .segments
            .read()
            .map_err(|_| ProEngineError::Io(std::io::Error::other("segments RwLock poisoned")))?;
        Ok(guard.len())
    }

    /// Returns the total live vector count across loaded segments and the in-memory delta buffer.
    pub fn total_live_vector_count(&self) -> ProResult<u64> {
        let seg_guard = self
            .segments
            .read()
            .map_err(|_| ProEngineError::Io(std::io::Error::other("segments RwLock poisoned")))?;
        let seg_live: u64 = seg_guard
            .iter()
            .map(|seg| {
                let total = seg.osi.header().num_vectors;
                let dead = seg.tombstones.len().max(seg.osi.tombstones().len());
                total.saturating_sub(dead)
            })
            .sum();
        let delta_live = u64::try_from(self.delta.live_count()?).unwrap_or(0);
        Ok(seg_live.saturating_add(delta_live))
    }

    /// Polls `manifest.json` via `HeadObject` ETag check and incrementally loads any new
    /// immutable `.osi` + `.osv` segment metadata without ZooKeeper or etcd.
    ///
    /// Returns `Ok(true)` if a new manifest epoch was observed and applied.
    pub async fn refresh_from_manifest(&self) -> ProResult<bool> {
        let known = self
            .known_etag
            .read()
            .map_err(|_| ProEngineError::Io(std::io::Error::other("etag RwLock poisoned")))?
            .clone();

        let snapshot = match self
            .manifest_store
            .poll_if_changed(known.as_deref())
            .await?
        {
            Some(snap) => snap,
            None => return Ok(false),
        };

        // Fallback deduplication for ObjectStore backends that do not emit ETags:
        // skip reloading segments if both epoch and segment count/ETag are unchanged.
        if known.is_none() && snapshot.version.e_tag.is_none() {
            let known_epoch = *self
                .known_epoch
                .read()
                .map_err(|_| ProEngineError::Io(std::io::Error::other("epoch RwLock poisoned")))?;
            if known_epoch == Some(snapshot.manifest.epoch) {
                return Ok(false);
            }
        }

        self.apply_manifest_snapshot(&snapshot.manifest, snapshot.version.e_tag)
            .await?;
        Ok(true)
    }

    /// Synchronizes loaded segments directly from a known [`ClusterManifest`].
    pub async fn apply_manifest_snapshot(
        &self,
        manifest: &ClusterManifest,
        new_etag: Option<String>,
    ) -> ProResult<()> {
        let existing_map: HashMap<u64, Arc<LoadedSegment>> = {
            let guard = self.segments.read().map_err(|_| {
                ProEngineError::Io(std::io::Error::other("segments RwLock poisoned"))
            })?;
            guard
                .iter()
                .map(|s| (s.segment_id, Arc::clone(s)))
                .collect()
        };

        let mut next_segments = Vec::with_capacity(manifest.segments.len());
        for desc in &manifest.segments {
            let seg_tombstones = desc.decode_tombstones()?;
            if let Some(existing) = existing_map.get(&desc.segment_id) {
                next_segments.push(Arc::new(LoadedSegment {
                    segment_id: existing.segment_id,
                    osv_path: existing.osv_path.clone(),
                    osi: Arc::clone(&existing.osi),
                    osv_directory: existing.osv_directory.clone(),
                    refine_stage: existing.refine_stage.clone(),
                    tombstones: seg_tombstones,
                }));
            } else {
                let loaded = self.load_segment_from_store(desc, seg_tombstones).await?;
                next_segments.push(Arc::new(loaded));
            }
        }

        let global_tombs = manifest.decode_global_tombstones()?;
        {
            let mut seg_guard = self.segments.write().map_err(|_| {
                ProEngineError::Io(std::io::Error::other("segments RwLock poisoned"))
            })?;
            *seg_guard = next_segments;
        }
        {
            let mut tomb_guard = self.global_tombstones.write().map_err(|_| {
                ProEngineError::Io(std::io::Error::other("tombstone RwLock poisoned"))
            })?;
            *tomb_guard = Arc::new(global_tombs);
        }
        {
            let mut etag_guard = self
                .known_etag
                .write()
                .map_err(|_| ProEngineError::Io(std::io::Error::other("etag RwLock poisoned")))?;
            *etag_guard = new_etag;
        }
        {
            let mut epoch_guard = self
                .known_epoch
                .write()
                .map_err(|_| ProEngineError::Io(std::io::Error::other("epoch RwLock poisoned")))?;
            *epoch_guard = Some(manifest.epoch);
        }

        Ok(())
    }

    async fn load_segment_from_store(
        &self,
        desc: &crate::manifest::SegmentDescriptor,
        tombstones: RoaringTreemap,
    ) -> ProResult<LoadedSegment> {
        let osi_path = ObjectPath::from(desc.osi_key.as_str());
        let osv_path = ObjectPath::from(desc.osv_key.as_str());
        #[allow(clippy::cast_possible_truncation)]
        let slab_id = (desc.segment_id & 0xFFFF) as u16;

        if let Some(root) = &self.local_store_root {
            let local_osi_path = root.join(desc.osi_key.trim_start_matches('/'));
            let local_osv_path = root.join(desc.osv_key.trim_start_matches('/'));
            if local_osi_path.exists() && local_osv_path.exists() {
                let osi = Arc::new(SegmentOsi::Mapped(OsiMappedIndex::open_mmap(
                    &local_osi_path,
                )?));
                let reader = OsvSlabReader::open_mmap(&local_osv_path)?;
                let osv_header = *reader.header();
                let osv_directory = reader.directory().to_vec();
                let zstd_dict = if osv_header.zstd_dict_len > 0 {
                    use std::io::{Read, Seek, SeekFrom};
                    let mut f = std::fs::File::open(&local_osv_path)?;
                    let len = usize::try_from(osv_header.zstd_dict_len)
                        .map_err(|_| OspreyError::CorruptedFooter)?;
                    f.seek(SeekFrom::Start(osv_header.zstd_dict_offset))?;
                    let mut dict = vec![0_u8; len];
                    f.read_exact(&mut dict)?;
                    Some(dict)
                } else {
                    None
                };
                drop(reader);
                let refine_stage = RefineStage::open_local_path(
                    slab_id,
                    self.range_planner_config,
                    &osv_header,
                    zstd_dict,
                    &local_osv_path,
                )?;
                return Ok(LoadedSegment {
                    segment_id: desc.segment_id,
                    osv_path,
                    osi,
                    osv_directory,
                    refine_stage,
                    tombstones,
                });
            }
        }

        let osi_bytes = self.store.get(&osi_path).await?.bytes().await?;
        let aligned_osi = AlignedBuffer::<u8>::from_slice(&osi_bytes)?;
        drop(osi_bytes);
        let osi = Arc::new(SegmentOsi::Mapped(OsiMappedIndex::from_aligned_buffer(
            aligned_osi,
        )?));

        let osv_meta = self.store.head(&osv_path).await?;
        let hdr_bytes = self.store.get_range(&osv_path, 0..64).await?;
        let aligned_hdr = AlignedBuffer::<u8>::from_slice(&hdr_bytes)?;
        let osv_header = *OsvHeader::from_bytes(aligned_hdr.as_slice())?;

        let dir_offset = usize::try_from(osv_header.directory_offset)
            .map_err(|_| OspreyError::CorruptedFooter)?;
        let file_size = osv_meta.size;
        if dir_offset > file_size {
            return Err(OspreyError::CorruptedFooter.into());
        }

        let mut ranges = Vec::with_capacity(2);
        ranges.push(dir_offset..file_size);
        let has_dict = osv_header.zstd_dict_len > 0;
        if has_dict {
            let dict_start = usize::try_from(osv_header.zstd_dict_offset)
                .map_err(|_| OspreyError::CorruptedFooter)?;
            let dict_len = usize::try_from(osv_header.zstd_dict_len)
                .map_err(|_| OspreyError::CorruptedFooter)?;
            let dict_end = dict_start
                .checked_add(dict_len)
                .ok_or(OspreyError::CorruptedFooter)?;
            ranges.push(dict_start..dict_end);
        }

        let fetched = self.store.get_ranges(&osv_path, &ranges).await?;
        let osv_directory =
            cast_minipage_directory(&fetched[0], osv_header.num_minipages)?.to_vec();
        let zstd_dict = if has_dict {
            Some(fetched[1].to_vec())
        } else {
            None
        };

        let zstd_enabled = (osv_header.flags & OSV_FLAG_ZSTD_ENABLED) != 0;
        let refine_stage = RefineStage::new(slab_id, self.range_planner_config, zstd_dict)
            .with_zstd_enabled(zstd_enabled);

        Ok(LoadedSegment {
            segment_id: desc.segment_id,
            osv_path,
            osi,
            osv_directory,
            refine_stage,
            tombstones,
        })
    }

    /// Executes a concurrent hybrid search merging all active `.osi`/`.osv` immutable segments
    /// with the live [`LsmDeltaBuffer`].
    pub async fn search_hybrid_async(
        &self,
        query_f32: &[f32],
        nprobe: usize,
        top_m_per_segment: usize,
        top_k: usize,
    ) -> ProResult<Vec<RefinedSearchResult>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        if query_f32.len() != self.delta.orig_dim() {
            return Err(ProEngineError::DimensionMismatch {
                expected: self.delta.orig_dim(),
                actual: query_f32.len(),
            });
        }

        let segments_snapshot = {
            let guard = self.segments.read().map_err(|_| {
                ProEngineError::Io(std::io::Error::other("segments RwLock poisoned"))
            })?;
            guard.clone()
        };
        let global_tombs = {
            let guard = self.global_tombstones.read().map_err(|_| {
                ProEngineError::Io(std::io::Error::other("tombstone RwLock poisoned"))
            })?;
            Arc::clone(&*guard)
        };
        let delta_masked = self.delta.masked_tombstones_if_nonempty()?;
        let delta_is_empty = self.delta.is_empty_fast();
        let has_global_tombs = !global_tombs.is_empty();

        if segments_snapshot.is_empty() {
            if delta_is_empty {
                return Ok(Vec::new());
            }
            let stage1_start = Instant::now();
            let hits = block_in_place_safe(|| self.delta.search_exact(query_f32, top_k))?;
            let stage1_us = u64::try_from(stage1_start.elapsed().as_micros()).unwrap_or(u64::MAX);
            self.metrics.observe_rabitq_scan(stage1_us);
            return Ok(hits);
        }

        // Fast-path: single segment + empty delta skips rayon::join / par_iter contention completely.
        let stage1_start = Instant::now();
        let (delta_hits, stage1_per_seg) = if segments_snapshot.len() == 1 && delta_is_empty {
            let seg = &segments_snapshot[0];
            let no_external_tombs =
                !has_global_tombs && seg.tombstones.is_empty() && delta_masked.is_none();
            let cands = block_in_place_safe(|| {
                if no_external_tombs {
                    seg.osi
                        .search_candidates(query_f32, nprobe, top_m_per_segment, &self.simd)
                } else {
                    let dm_ref = delta_masked.as_deref();
                    seg.osi.search_candidates_filtered_with(
                        query_f32,
                        nprobe,
                        top_m_per_segment,
                        &self.simd,
                        |_entry_idx, doc_id| {
                            (!has_global_tombs || !global_tombs.contains(doc_id))
                                && (seg.tombstones.is_empty() || !seg.tombstones.contains(doc_id))
                                && dm_ref.is_none_or(|m| !m.contains(doc_id))
                        },
                    )
                }
            })?;
            (Vec::new(), vec![(Arc::clone(seg), cands)])
        } else {
            let delta_ref = &self.delta;
            let simd_ref = &self.simd;
            let dm_ref = delta_masked.as_deref();
            let (delta_res, segment_stage1_res) = block_in_place_safe(|| {
                if delta_is_empty {
                    let seg_res = segments_snapshot
                        .par_iter()
                        .map(|seg| {
                            let cands = if !has_global_tombs
                                && seg.tombstones.is_empty()
                                && dm_ref.is_none()
                            {
                                seg.osi.search_candidates(
                                    query_f32,
                                    nprobe,
                                    top_m_per_segment,
                                    simd_ref,
                                )?
                            } else {
                                seg.osi.search_candidates_filtered_with(
                                    query_f32,
                                    nprobe,
                                    top_m_per_segment,
                                    simd_ref,
                                    |_entry_idx, doc_id| {
                                        (!has_global_tombs || !global_tombs.contains(doc_id))
                                            && (seg.tombstones.is_empty()
                                                || !seg.tombstones.contains(doc_id))
                                            && dm_ref.is_none_or(|m| !m.contains(doc_id))
                                    },
                                )?
                            };
                            Ok::<_, ProEngineError>((Arc::clone(seg), cands))
                        })
                        .collect::<ProResult<Vec<_>>>();
                    (Ok(Vec::new()), seg_res)
                } else {
                    rayon::join(
                        || delta_ref.search_exact(query_f32, top_k),
                        || {
                            segments_snapshot
                                .par_iter()
                                .map(|seg| {
                                    let cands = seg.osi.search_candidates_filtered_with(
                                        query_f32,
                                        nprobe,
                                        top_m_per_segment,
                                        simd_ref,
                                        |_entry_idx, doc_id| {
                                            (!has_global_tombs || !global_tombs.contains(doc_id))
                                                && (seg.tombstones.is_empty()
                                                    || !seg.tombstones.contains(doc_id))
                                                && dm_ref.is_none_or(|m| !m.contains(doc_id))
                                        },
                                    )?;
                                    Ok::<_, ProEngineError>((Arc::clone(seg), cands))
                                })
                                .collect::<ProResult<Vec<_>>>()
                        },
                    )
                }
            });
            (delta_res?, segment_stage1_res?)
        };
        let stage1_us = u64::try_from(stage1_start.elapsed().as_micros()).unwrap_or(u64::MAX);
        self.metrics.observe_rabitq_scan(stage1_us);

        // Single-segment fast-path: bypass try_join_all, HashMap, and re-sort when delta is empty.
        if stage1_per_seg.len() == 1 && delta_hits.is_empty() {
            let (seg, cands) = stage1_per_seg.into_iter().next().ok_or_else(|| {
                ProEngineError::Io(std::io::Error::other(
                    "missing segment in single-segment path",
                ))
            })?;
            if cands.is_empty() {
                return Ok(Vec::new());
            }
            let page_refs: Vec<CandidatePageRef> =
                cands.iter().map(CandidatePageRef::from).collect();
            let (refined, stats) = seg
                .refine_stage
                .refine_async(
                    query_f32,
                    &page_refs,
                    top_k,
                    &seg.osv_directory,
                    &self.cache,
                    &self.store,
                    &seg.osv_path,
                    &self.remote_config,
                )
                .await?;
            self.metrics.observe_range_plan_and_s3(
                stats.range_plan_time_us,
                stats.raw_page_requests,
                stats.coalesced_range_gets,
                stats.cache_hits,
            );
            self.metrics.observe_refine_breakdown(
                stats.cache_lookup_us,
                stats.disk_read_us,
                stats.zstd_decode_us,
                stats.f16_score_us,
            );
            if !has_global_tombs && delta_masked.is_none() {
                return Ok(refined);
            }
            let dm_ref = delta_masked.as_deref();
            return Ok(refined
                .into_iter()
                .filter(|hit| {
                    (!has_global_tombs || !global_tombs.contains(hit.doc_id))
                        && dm_ref.is_none_or(|m| !m.contains(hit.doc_id))
                })
                .collect());
        }

        // Dispatch async Stage-3 `.osv` refinements concurrently across all segments.
        let refine_futs = stage1_per_seg.into_iter().map(|(seg, cands)| {
            let cache = Arc::clone(&self.cache);
            let store = Arc::clone(&self.store);
            let remote_cfg = self.remote_config.clone();
            let metrics = Arc::clone(&self.metrics);
            async move {
                if cands.is_empty() {
                    return Ok::<Vec<RefinedSearchResult>, ProEngineError>(Vec::new());
                }
                let page_refs: Vec<CandidatePageRef> =
                    cands.iter().map(CandidatePageRef::from).collect();
                let (refined, stats) = seg
                    .refine_stage
                    .refine_async(
                        query_f32,
                        &page_refs,
                        top_k,
                        &seg.osv_directory,
                        &cache,
                        &store,
                        &seg.osv_path,
                        &remote_cfg,
                    )
                    .await?;
                metrics.observe_range_plan_and_s3(
                    stats.range_plan_time_us,
                    stats.raw_page_requests,
                    stats.coalesced_range_gets,
                    stats.cache_hits,
                );
                metrics.observe_refine_breakdown(
                    stats.cache_lookup_us,
                    stats.disk_read_us,
                    stats.zstd_decode_us,
                    stats.f16_score_us,
                );
                Ok(refined)
            }
        });

        let base_hits_nested = futures::future::try_join_all(refine_futs).await?;

        // Merge and deduplicate: delta entries take precedence over older base versions.
        let mut by_doc: HashMap<u64, RefinedSearchResult> = HashMap::new();
        for hit in delta_hits {
            by_doc.insert(hit.doc_id, hit);
        }
        let dm_ref = delta_masked.as_deref();
        for seg_hits in base_hits_nested {
            for hit in seg_hits {
                if (has_global_tombs && global_tombs.contains(hit.doc_id))
                    || dm_ref.is_some_and(|m| m.contains(hit.doc_id))
                {
                    continue;
                }
                by_doc.entry(hit.doc_id).or_insert(hit);
            }
        }

        let mut merged: Vec<RefinedSearchResult> = by_doc.into_values().collect();
        merged.sort_by(|a, b| {
            b.exact_score
                .total_cmp(&a.exact_score)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });
        merged.truncate(top_k);
        Ok(merged)
    }
}

#[inline]
fn exact_f16_dot(query_f32: &[f32], vec_f16: &[f16]) -> f32 {
    osprey_simd::dot_f32_f16(query_f32, vec_f16)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::time::Instant;

    use super::*;

    #[test]
    fn delta_ingest_makes_vectors_searchable_under_1ms() {
        let delta = LsmDeltaBuffer::new(64, 64, 42, Vec::new(), None).unwrap();

        let mut base_vec = vec![0.0_f32; 64];
        base_vec[0] = 1.0;
        let mut target_vec = vec![0.0_f32; 64];
        target_vec[1] = 1.0;

        delta
            .ingest_batch(&[RawVectorRecord {
                doc_id: 1,
                vector: &base_vec,
                payload_json: br#"{"label":"base"}"#,
            }])
            .unwrap();

        let t0 = Instant::now();
        delta
            .ingest_batch(&[RawVectorRecord {
                doc_id: 2,
                vector: &target_vec,
                payload_json: br#"{"label":"target"}"#,
            }])
            .unwrap();
        let hits = delta.search_exact(&target_vec, 1).unwrap();
        let elapsed_us = t0.elapsed().as_secs_f64() * 1_000_000.0;

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, 2);
        assert_eq!(hits[0].payload_json, br#"{"label":"target"}"#);
        assert!((hits[0].exact_score - 1.0).abs() < 1e-3);
        eprintln!("delta ingest + search latency: {elapsed_us:.1} us");
        if !cfg!(debug_assertions) {
            assert!(
                elapsed_us < 1000.0,
                "expected < 1 ms ingest+search, got {elapsed_us:.1} us"
            );
        }
    }

    #[test]
    fn upsert_and_delete_mask_old_entries() {
        let delta = LsmDeltaBuffer::new(64, 64, 7, Vec::new(), None).unwrap();

        let mut v1 = vec![0.0_f32; 64];
        v1[0] = 1.0;
        let mut v1_updated = vec![0.0_f32; 64];
        v1_updated[2] = 1.0;

        delta
            .ingest_batch(&[RawVectorRecord {
                doc_id: 100,
                vector: &v1,
                payload_json: br#"{"v":1}"#,
            }])
            .unwrap();
        delta
            .ingest_batch(&[RawVectorRecord {
                doc_id: 100,
                vector: &v1_updated,
                payload_json: br#"{"v":2}"#,
            }])
            .unwrap();

        assert_eq!(delta.live_count().unwrap(), 1);
        assert!(delta.is_masked_in_base(100));

        let hits = delta.search_exact(&v1_updated, 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].payload_json, br#"{"v":2}"#);

        delta.delete_docs(&[100]).unwrap();
        assert_eq!(delta.live_count().unwrap(), 0);
        assert!(delta.search_exact(&v1_updated, 5).unwrap().is_empty());
    }

    #[test]
    fn supports_full_64bit_doc_ids_in_tombstones_and_masking() {
        let delta = LsmDeltaBuffer::new(64, 64, 11, Vec::new(), None).unwrap();
        let high_id = u64::MAX - 42;

        let mut vec = vec![0.0_f32; 64];
        vec[3] = 1.0;

        delta
            .ingest_batch(&[RawVectorRecord {
                doc_id: high_id,
                vector: &vec,
                payload_json: br#"{"high":true}"#,
            }])
            .unwrap();
        assert!(delta.is_masked_in_base(high_id));
        assert!(delta.masked_tombstones().unwrap().contains(high_id));

        delta.delete_docs(&[high_id]).unwrap();
        assert_eq!(delta.live_count().unwrap(), 0);
        assert!(delta.is_masked_in_base(high_id));

        let snap = delta.snapshot_for_compaction().unwrap().unwrap();
        assert!(snap.tombstones.contains(high_id));
    }

    #[test]
    fn commit_compaction_snapshot_preserves_concurrent_writes_after_watermark() {
        let delta = LsmDeltaBuffer::new(64, 64, 19, Vec::new(), None).unwrap();
        let mut v1 = vec![0.0_f32; 64];
        v1[0] = 1.0;
        let mut v2 = vec![0.0_f32; 64];
        v2[1] = 1.0;

        delta
            .ingest_batch(&[RawVectorRecord {
                doc_id: 1,
                vector: &v1,
                payload_json: br#"{"gen":1}"#,
            }])
            .unwrap();

        let snap = delta.snapshot_for_compaction().unwrap().unwrap();
        assert_eq!(snap.doc_ids, vec![1]);

        // Concurrent write arrives BEFORE commit_compaction_snapshot
        delta
            .ingest_batch(&[RawVectorRecord {
                doc_id: 2,
                vector: &v2,
                payload_json: br#"{"gen":2}"#,
            }])
            .unwrap();
        assert_eq!(delta.live_count().unwrap(), 2);

        delta
            .commit_compaction_snapshot(snap.watermark_seq)
            .unwrap();
        assert_eq!(delta.live_count().unwrap(), 1);
        let hits = delta.search_exact(&v2, 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, 2);
    }

    #[tokio::test]
    async fn hybrid_search_local_uncompressed_single_segment_current_thread_and_metrics() {
        use object_store::local::LocalFileSystem;
        use osprey_storage::OsvWriterConfig;

        use crate::compactor::{CompactorConfig, CowSlabCompactor};

        let dir = tempfile::tempdir().unwrap();
        let store: Arc<dyn ObjectStore> =
            Arc::new(LocalFileSystem::new_with_prefix(dir.path()).unwrap());
        let manifest_store = ManifestStore::new(Arc::clone(&store), "manifest.json");
        let seed = 0x4F53_5052_0001_u64;
        let dim = 64usize;
        let delta = Arc::new(LsmDeltaBuffer::new(dim, dim, seed, Vec::new(), None).unwrap());

        let mut target = vec![0.0_f32; dim];
        target[25] = 1.0;
        let mut recs = Vec::new();
        let mut storage = Vec::with_capacity(16);
        for i in 0..16usize {
            let mut v = vec![0.0_f32; dim];
            if i == 7 {
                v[25] = 1.0;
            } else {
                v[i % dim] = 1.0;
                v[(i + 1) % dim] = 0.25;
            }
            storage.push(v);
        }
        for (i, v) in storage.iter().enumerate() {
            recs.push(RawVectorRecord {
                doc_id: (i as u64) + 100,
                vector: v,
                payload_json: br#"{"ok":true}"#,
            });
        }
        delta.ingest_batch(&recs).unwrap();

        let compactor = CowSlabCompactor::new(
            CompactorConfig {
                delta_flush_threshold: 1,
                target_clusters: 4,
                fwht_seed: seed,
                osv_writer_config: OsvWriterConfig::uncompressed(),
                ..CompactorConfig::default()
            },
            Arc::clone(&store),
            manifest_store.clone(),
        );
        let cache = Arc::new(MinipageLruCache::from_mb(4));
        let engine = HybridSearchEngine::new(
            Arc::clone(&delta),
            Arc::clone(&store),
            manifest_store,
            cache,
            RemoteStoreConfig::default(),
            RangePlannerConfig::default(),
        )
        .with_local_store_root(dir.path());

        let outcome = compactor
            .compact_now_with_engine(&delta, None, Some(&engine))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(outcome.vectors_compacted, 16);
        assert_eq!(delta.live_count().unwrap(), 0);
        assert!(delta.is_empty_fast());
        assert_eq!(engine.loaded_segment_count().unwrap(), 1);

        // Executes inside default tokio `current_thread` runtime without panicking in
        // `block_in_place_safe` and uses zero-copy `.osi` + uncompressed `.osv` mmap fast-path.
        let results = engine.search_hybrid_async(&target, 4, 8, 3).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].doc_id, 107);

        let prom = engine.metrics().render_prometheus();
        assert!(prom.contains("osprey_pro_stage1_duration_microseconds_count 1"));
        assert!(prom.contains("osprey_pro_f16_score_duration_microseconds_count 1"));
    }
}
