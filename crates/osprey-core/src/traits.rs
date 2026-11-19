//! Extension traits and immutable reader contracts for OspreyDB.

use crate::error::OspreyResult;
use crate::format::osi::OsiHeader;
use crate::format::osr::{DocTokenRange, OsrHeader};

/// Candidate returned by the asymmetric Stage-1 `.osi` scan.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stage1Candidate {
    /// Logical document identifier.
    pub doc_id: u64,
    /// Target `.osv` minipage identifier.
    pub minipage_id: u32,
    /// Vector slot index inside the minipage.
    pub slot_idx: u16,
    /// Approximate Stage-1 asymmetric inner-product score.
    pub approx_score: f32,
}

/// Zero-copy reader contract for `.osi` resident 1-bit index files.
pub trait VectorIndexReader: Send + Sync {
    /// Returns the validated `.osi` header.
    fn header(&self) -> &OsiHeader;

    /// Selects the top `nprobe` clusters closest to the FWHT-rotated query `q_rot`.
    fn probe_clusters(&self, q_rot: &[f32], nprobe: usize) -> OspreyResult<Vec<u32>>;

    /// Scans the probed clusters using asymmetric `RaBitQ` and tombstone filtering.
    fn scan_clusters(
        &self,
        q_rot: &[f32],
        cluster_ids: &[u32],
        top_c: usize,
    ) -> OspreyResult<Vec<Stage1Candidate>>;

    /// Returns `true` if `doc_id` is marked as logically deleted in the tombstone bitmap.
    fn is_deleted(&self, doc_id: u64) -> bool;
}

/// Zero-copy reader contract for `.osr` nibble-packed token rerank stores.
pub trait RerankStoreReader: Send + Sync {
    /// Returns the validated `.osr` header.
    fn header(&self) -> &OsrHeader;

    /// Resolves a `doc_id` to its contiguous `DocTokenRange` via binary search.
    fn lookup_doc(&self, doc_id: u64) -> OspreyResult<Option<DocTokenRange>>;

    /// Returns a 64-byte-aligned byte slice (`token_count * 64` bytes) for `range`.
    fn token_bytes(&self, range: &DocTokenRange) -> OspreyResult<&[u8]>;
}

/// Raw vector record submitted for continuous streaming ingestion into the `osprey` daemon.
#[derive(Clone, Copy, Debug)]
pub struct RawVectorRecord<'a> {
    /// Logical document identifier.
    pub doc_id: u64,
    /// Dense input vector slice.
    pub vector: &'a [f32],
    /// Serialized JSON metadata payload.
    pub payload_json: &'a [u8],
}

/// Asynchronous compaction summary emitted by the `osprey` Copy-on-Write slab compactor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactionReport {
    /// Number of delta/base segments merged.
    pub merged_segments: usize,
    /// Number of deleted tombstones physically reclaimed.
    pub reclaimed_tombstones: u64,
    /// Resulting canonical segment identifier.
    pub new_segment_id: u64,
}

/// Extension hook for continuous WAL writes, LSM delta buffers, and CoW compaction (`osprey` daemon).
pub trait DeltaBufferExt: Send + Sync {
    /// Backend-specific error type returned by the extended storage engine.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Appends a mutable batch to the `fsync` WAL and quantizes against active `.osi` centroids.
    fn append_batch(&self, records: &[RawVectorRecord<'_>]) -> Result<u64, Self::Error>;

    /// Searches in-memory delta buffers not yet compacted into immutable `.osv` slabs.
    fn search_deltas(
        &self,
        q_rot: &[f32],
        top_k: usize,
    ) -> Result<Vec<Stage1Candidate>, Self::Error>;

    /// Triggers CoW slab compaction and tombstone garbage collection on `segment_ids`.
    fn compact(&self, segment_ids: &[u64]) -> Result<CompactionReport, Self::Error>;
}

/// Alias matching `ARCHITECTURE.md` §1.3 naming (`StorageEngineExt`).
pub use DeltaBufferExt as StorageEngineExt;
