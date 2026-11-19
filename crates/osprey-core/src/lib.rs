//! Core canonical binary formats, aligned memory primitives, mathematical kernels,
//! and extension traits for OspreyDB.

#[cfg(target_endian = "big")]
compile_error!(
    "OspreyDB requires a little-endian target architecture for zero-copy mmap formats (.osi, .osv, .osr)."
);

pub mod aligned;
pub mod cluster;
pub mod error;
pub mod format;
pub mod lutrank;
pub mod math;
pub mod rabitq;
pub mod search;
pub mod topk;
pub mod traits;

pub use aligned::{ensure_aligned_64, AlignedBuffer, CACHE_LINE_ALIGN};
pub use cluster::{
    binarize_rotated_vectors, fit_binary_kmeans, fit_binary_kmeans_from_rotated,
    fit_binary_kmeans_from_rotated_with_init, init_kmeans_plusplus_1bit,
    preprocess_and_binarize_batch, BinaryKMeansConfig, BinaryKMeansKernel, BinaryKMeansResult,
    ScalarClusterKernel,
};
pub use error::{OspreyError, OspreyResult};
pub use format::osi::{
    cast_centroids_f16, cast_centroids_f16_mut, centroids_padded_bytes, centroids_unpadded_bytes,
    cluster_block_bytes, deserialize_tombstone_footer, serialize_tombstone_footer, OsiClusterSpan,
    OsiClusterView, OsiHeader, RabitqEntryHeader,
};
pub use format::osr::{cast_doc_token_ranges, DocTokenRange, OsrHeader};
pub use format::osv::{cast_minipage_directory, MinipageDirectoryEntry, OsvHeader};
pub use format::{
    align_up_64, checked_align_up_64, FORMAT_KIND_OSI, FORMAT_KIND_OSR, FORMAT_KIND_OSV,
    FORMAT_VERSION_V1, OSPREY_MAGIC,
};
pub use lutrank::{
    build_nested_lattice_codebooks, encode_prerotated_token, encode_prerotated_token_codes,
    encode_token, pack_nibbles, score_doc_maxsim_scalar, score_token_with_luts_scalar,
    synthesize_query_token_luts, unpack_nibbles, LatticeCodebook2D, LutRankKernel,
    NestedLatticeCodebooks, OsrDocumentTokens, OsrStore, OsrTokenCode, QuantizedQueryLut,
    QueryTokenLuts, DEFAULT_LUTRANK_GAMMA, LATTICE_1D_LEVELS, LUTRANK_SUBSPACES, LUTRANK_TOKEN_DIM,
    OSR_TOKEN_BYTES,
};
pub use rabitq::{
    encode_prerotated_rabitq_residual, encode_prerotated_rabitq_residual_in_place,
    encode_rabitq_residual, encode_rabitq_residual_scalar, OsiIndex, OsiMappedView, ProbedCluster,
    RabitqEncodedResidual, RabitqFwhtKernel, RabitqScanKernel, ScalarRabitqKernel,
    TransposedQuery4Bit, VectorLocation, OSI_FLAG_UNROTATED_CENTROIDS, QUERY_QUANT_EPS,
    RABITQ_F16_MAX_CLAMP, RABITQ_L1_MIN_NORMAL_F32, RABITQ_NORM_EPS,
};
pub use search::{
    ResolvedSearchPlan, SearchOptions, DEFAULT_LOW_DIM_OVERSAMPLE_MULTIPLIER,
    LOW_DIM_OVERSAMPLE_THRESHOLD, LOW_DIM_REFINE_POOL,
};
pub use topk::{BoundedTopKHeap, ScoredCandidate};
pub use traits::{
    CompactionReport, DeltaBufferExt, RawVectorRecord, RerankStoreReader, Stage1Candidate,
    StorageEngineExt, VectorIndexReader,
};
