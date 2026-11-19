//! Zero-copy `memmap2` readers, TSP slab packing, DP range coalescing,
//! Zstd minipage codecs, `RoaringBitmap` tombstones, and async `object_store` I/O.

pub mod chaos_store;
pub mod formats_ingest;
pub mod lake_ingest;
pub mod osv_slab;
pub mod page_cache;
pub mod range_planner;
pub mod refine;
pub mod remote_store;
pub mod tsp_locality;

pub use chaos_store::{ChaosConfig, ChaosObjectStore, ChaosStatsSnapshot};

pub use formats_ingest::{
    JsonlParsedRecord, JsonlRecordReader, NpyDtype, NpyF16Batch, NpyHeader, NpyMatrixReader,
    NpyVectorBatch,
};
pub use lake_ingest::{
    extract_batch_colbert_tokens_f32, extract_batch_doc_ids, extract_batch_vectors_f32,
    hash_category, inspect_parquet_schema, map_string_id_to_u64, DetectedLakeSchema,
    FilterPredicate, IngestColumnHints, MinipageZoneMap, OutOfCoreIngestConfig,
    OutOfCoreIngestPipeline, OutOfCoreIngestReport, ParquetBatchReader, ParquetBatchStream,
    ProjectedColumnMap, RowFilterAttributes, ScalarFilterColumnMapping, ScalarFilterIndex,
    TrainedLakeCodebook, DEFAULT_INGEST_BATCH_SIZE, DEFAULT_MAX_TRAINING_SAMPLE_VECTORS,
    DEFAULT_OUT_OF_CORE_BATCH_ROWS,
};
pub use osprey_core::{OspreyError, OspreyResult};
pub use osprey_simd as simd;
pub use osv_slab::{
    build_minipages, parse_minipage, train_payload_zstd_dictionary, DecodedMinipage, MinipageView,
    OsvSlabReader, OsvSlabView, OsvVectorLocation, OsvVectorRecord, OsvWriteReport, OsvWriter,
    OsvWriterConfig, StreamingOsvWriter, DEFAULT_MINIPAGE_TARGET_BYTES,
    DEFAULT_ZSTD_DICT_MAX_BYTES, MINIPAGE_INTERNAL_HEADER_BYTES,
};
pub use page_cache::{
    MinipageCacheKey, MinipageCacheStats, MinipageLruCache, DEFAULT_MINIPAGE_CACHE_BYTES,
};
pub use range_planner::{
    extract_unique_page_intervals, plan_coalesced_ranges_stack, CandidatePageRef, CoalescedRange,
    PageByteInterval, RangePlannerConfig, DEFAULT_COALESCE_MAX_K, DEFAULT_COALESCE_TAU_BYTE,
    DEFAULT_COALESCE_TAU_REQ, DEFAULT_MAX_COALESCE_BYTES,
};
pub use refine::{LocalOsvBacking, RefineStage, RefineStats, RefinedSearchResult};
pub use remote_store::{
    build_client_options, build_object_store, fetch_coalesced_ranges_resilient,
    fetch_coalesced_ranges_with_transport, slice_and_decompress_coalesced, AsyncRangeTransport,
    RemoteStoreConfig, DEFAULT_HEDGED_REQUEST_DELAY_MS, DEFAULT_HTTP2_KEEP_ALIVE_SECS,
    DEFAULT_MAX_CONCURRENT_STREAMS, DEFAULT_MAX_RETRIES, DEFAULT_POOL_MAX_IDLE_PER_HOST,
    DEFAULT_REQUEST_TIMEOUT_MS,
};
pub use tsp_locality::{
    compute_centroid_distance_matrix, open_tour_length, solve_tsp_locality,
    solve_tsp_locality_largest_first, tsp_nearest_neighbor, tsp_two_opt_refine,
    ClusterLocalityPermutation, TSP_PARALLEL_CLUSTER_THRESHOLD,
};
