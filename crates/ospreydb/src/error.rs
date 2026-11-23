//! Error types for `ospreydb` WAL, LSM delta buffer, Copy-on-Write compaction,
//! tombstone GC, and S3 CAS manifest coordination.

use thiserror::Error;

/// Unified result type for `ospreydb` storage and coordination operations.
pub type ProResult<T> = Result<T, ProEngineError>;

/// Errors produced by `ospreydb` continuous ingestion, WAL, compaction, and S3 CAS manifest.
#[derive(Debug, Error)]
pub enum ProEngineError {
    /// Core OspreyDB binary format, math, or SIMD error.
    #[error("core engine error: {0}")]
    Core(#[from] osprey_core::OspreyError),

    /// WAL file magic bytes did not match `"OPWL"`.
    #[error("invalid WAL magic: expected [0x4f, 0x50, 0x57, 0x4c], found {found:?}")]
    InvalidWalMagic {
        /// Observed 4-byte magic prefix.
        found: [u8; 4],
    },

    /// WAL format version is unsupported.
    #[error("unsupported WAL version: expected {expected}, found {found}")]
    UnsupportedWalVersion {
        /// Expected version number.
        expected: u16,
        /// Observed version number.
        found: u16,
    },

    /// WAL record `xxHash64` checksum verification failed.
    #[error(
        "WAL record checksum mismatch at byte offset {offset}: expected 0x{expected:016x}, computed 0x{computed:016x}"
    )]
    WalChecksumMismatch {
        /// Byte offset of the corrupted frame.
        offset: u64,
        /// Stored `xxHash64` trailer.
        expected: u64,
        /// Recomputed `xxHash64`.
        computed: u64,
    },

    /// Structural corruption in a WAL record frame.
    #[error("WAL record corrupted at byte offset {offset}: {reason}")]
    WalCorrupted {
        /// Byte offset where corruption was detected.
        offset: u64,
        /// Human-readable corruption reason.
        reason: &'static str,
    },

    /// Conditional S3 `PutObject` (`If-Match` / `ETag`) CAS update conflicted.
    #[error("S3 manifest CAS conflict after {attempts} attempt(s)")]
    CasConflict {
        /// Number of CAS attempts performed.
        attempts: usize,
    },

    /// Conditional manifest creation failed because it already exists.
    #[error("manifest already exists at '{path}'")]
    ManifestAlreadyExists {
        /// Object store path of the manifest.
        path: String,
    },

    /// Referenced manifest or segment does not exist.
    #[error("manifest or segment not found at '{path}'")]
    NotFound {
        /// Object store path.
        path: String,
    },

    /// Underlying `object_store` I/O failure.
    #[error("object store error: {0}")]
    ObjectStore(#[from] object_store::Error),

    /// Manifest JSON serialization or deserialization failure.
    #[error("manifest JSON error: {0}")]
    ManifestJson(#[from] serde_json::Error),

    /// Base64 decoding error for serialized `RoaringBitmap` tombstones.
    #[error("tombstone bitmap base64 decode error: {0}")]
    TombstoneBase64(#[from] base64::DecodeError),

    /// Input vector dimension mismatched index or delta buffer dimension.
    #[error("vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected vector dimensionality.
        expected: usize,
        /// Observed vector dimensionality.
        actual: usize,
    },

    /// Invalid configuration or runtime argument.
    #[error("invalid configuration: {0}")]
    InvalidConfig(&'static str),

    /// Filesystem or OS I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
