//! Commercial extensions for OspreyDB (`ospreydb`).
//!
//! This crate consumes `osprey-core`, `osprey-simd`, and `osprey-storage`
//! strictly as external dependencies, implementing offline `Ed25519` license
//! verification ([`OfflineLicenseVerifier`]), crash-safe WAL ([`OspreyWal`]),
//! real-time [`LsmDeltaBuffer`] + [`HybridSearchEngine`], Copy-on-Write
//! [`CowSlabCompactor`], [`TombstoneGarbageCollector`], atomic S3
//! `manifest.json` CAS coordination ([`ManifestStore`]), Apache Arrow Flight
//! gRPC ([`OspreyFlightService`]), Drop-In Cohere/Qdrant HTTP APIs
//! ([`HttpServerState`]), and Prometheus observability ([`ProMetricsRegistry`]).

pub mod compactor;
pub mod delta_buffer;
pub mod embedder;
pub mod error;
pub mod flight_server;
pub mod http_server;
pub mod license;
pub mod manifest;
pub mod metrics;
pub mod rerank_service;
pub mod wal;

pub use compactor::{
    CompactorConfig, CompactorHandle, CowCompactionOutcome, CowSlabCompactor,
    TombstoneGarbageCollector, TombstoneGcConfig, TombstoneGcReport, DEFAULT_DELTA_FLUSH_THRESHOLD,
    DEFAULT_TOMBSTONE_GC_THRESHOLD,
};
pub use delta_buffer::{FrozenDeltaSnapshot, HybridSearchEngine, LoadedSegment, LsmDeltaBuffer};
pub use embedder::{Embedder, EmbedderConfig};
pub use error::{ProEngineError, ProLicenseError, ProResult};
pub use flight_server::{serve_arrow_flight, OspreyFlightService, DEFAULT_ARROW_FLIGHT_ADDR};
pub use http_server::{
    build_http_router, serve_http, CohereRerankRequest, HttpServerState, QdrantPointStruct,
    QdrantSearchParams, QdrantSearchRequest, QdrantUpsertRequest, RerankDocumentInput,
    DEFAULT_HTTP_ADDR,
};
pub use license::{
    capability_from_str, capability_name, current_unix_secs, Ed25519LicenseVerifier,
    OfflineLicenseVerifier, OspreyLicense, VectorQuotaState, COMMUNITY_MAX_VECTORS,
    EMBEDDED_PUBLIC_KEY, LICENSE_GRACE_PERIOD_SECS, LICENSE_TOKEN_PREFIX, MAX_CLOCK_SKEW_SECS,
    MAX_LICENSE_TOKEN_BYTES,
};
pub use manifest::{
    ClusterManifest, ManifestSnapshot, ManifestStore, SegmentDescriptor, DEFAULT_MANIFEST_KEY,
    MANIFEST_SCHEMA_VERSION_V1,
};
pub use metrics::{MetricsSnapshot, ProMetricsRegistry};
pub use osprey_core::{
    CompactionReport, DeltaBufferExt, LicenseClaims, LicenseGuard, LicenseVerifier, ProCapability,
    RawVectorRecord, Stage1Candidate, StorageEngineExt,
};
pub use rerank_service::{LutRankReranker, RerankHit, DEFAULT_LUTRANK_SEED};
pub use wal::{
    OspreyWal, WalConfig, WalOpKind, WalRecord, WalRecoveryMode, WalSyncPolicy, WAL_MAGIC,
    WAL_VERSION_V1,
};
