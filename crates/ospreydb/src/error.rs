//! Error types for `ospreydb` commercial licensing, WAL, LSM delta buffer,
//! Copy-on-Write compaction, tombstone GC, and S3 CAS manifest coordination.

use thiserror::Error;

/// Unified result type for `ospreydb` storage and coordination operations.
pub type ProResult<T> = Result<T, ProEngineError>;

/// Errors produced during offline `Ed25519` license parsing and verification.
#[derive(Debug, Error)]
pub enum ProLicenseError {
    /// Token does not match `OSPREY-PRO-v1.<base64url_payload>.<base64url_sig>`.
    #[error("invalid license token format: expected OSPREY-PRO-v1.<payload>.<signature>")]
    InvalidFormat,

    /// Token has an unrecognized `OSPREY-PRO-*` version prefix.
    #[error("unsupported license token version: {found}")]
    UnsupportedVersion {
        /// Observed version prefix.
        found: String,
    },

    /// Token exceeds the maximum allowed byte size.
    #[error("license token exceeds maximum allowed size ({len} > {max} bytes)")]
    TokenTooLarge {
        /// Observed token length in bytes.
        len: usize,
        /// Maximum allowed token length in bytes.
        max: usize,
    },

    /// Base64url decoding failed on either the payload or signature segment.
    #[error("invalid base64url encoding in license token: {0}")]
    InvalidBase64(#[from] base64::DecodeSliceError),

    /// Base64url decoding into a dynamically sized buffer failed.
    #[error("invalid base64url encoding in license payload: {0}")]
    InvalidPayloadBase64(#[from] base64::DecodeError),

    /// Decoded Ed25519 signature did not have the expected 64-byte length.
    #[error("invalid Ed25519 signature length: expected {expected} bytes, got {actual}")]
    InvalidSignatureLength {
        /// Expected byte length (64).
        expected: usize,
        /// Actual decoded byte length.
        actual: usize,
    },

    /// Embedded or supplied 32-byte Ed25519 public key failed curve validation.
    #[error("invalid Ed25519 public verifying key")]
    InvalidPublicKey,

    /// Ed25519 cryptographic signature verification failed.
    #[error("invalid Ed25519 license signature")]
    InvalidSignature,

    /// Signed JSON payload could not be deserialized into [`crate::OspreyLicense`].
    #[error("invalid license JSON payload: {0}")]
    InvalidPayloadJson(#[from] serde_json::Error),

    /// Semantic validation of license claims failed.
    #[error("invalid license claim: {reason}")]
    InvalidClaim {
        /// Reason the claim failed validation.
        reason: &'static str,
    },

    /// License `issued_at` timestamp is further in the future than allowed clock skew.
    #[error("license issued_at ({issued_at}) is in the future relative to {now_unix_secs}")]
    IssuedInFuture {
        /// Token issuance timestamp (Unix seconds UTC).
        issued_at: u64,
        /// Current clock timestamp (Unix seconds UTC).
        now_unix_secs: u64,
    },

    /// License has expired past the 30-day renewal grace period at `now_unix_secs`.
    #[error(
        "license expired at {expires_at} and exceeded the 30-day renewal grace period (current time {now_unix_secs})"
    )]
    LicenseExpired {
        /// Token expiration timestamp (Unix seconds UTC).
        expires_at: u64,
        /// Current clock timestamp (Unix seconds UTC).
        now_unix_secs: u64,
    },

    /// Required commercial feature flag is not granted by the verified license.
    #[error("license does not grant required feature '{feature}'")]
    MissingFeature {
        /// Required commercial feature identifier.
        feature: String,
    },

    /// Cluster node count exceeds `max_nodes` permitted by the license.
    #[error("cluster node count {active_nodes} exceeds licensed limit of {max_nodes}")]
    NodeLimitExceeded {
        /// Licensed maximum node count (`0` means unlimited).
        max_nodes: u32,
        /// Observed active cluster nodes.
        active_nodes: u32,
    },

    /// System clock could not be converted to Unix epoch seconds.
    #[error("system clock is before Unix epoch")]
    SystemClockError,
}

/// Errors produced by `ospreydb` continuous ingestion, WAL, compaction, and S3 CAS manifest.
#[derive(Debug, Error)]
pub enum ProEngineError {
    /// Core OspreyDB binary format, math, or SIMD error.
    #[error("core engine error: {0}")]
    Core(#[from] osprey_core::OspreyError),

    /// Commercial license verification or capability gating error.
    #[error("license error: {0}")]
    License(#[from] ProLicenseError),

    /// Community tier hard vector limit reached (5M) — requires a Pro license activation.
    #[error(
        "Community Tier limit ({limit} vectors) reached. Activate a Pro license via \
         'osprey activate --license <KEY>' — see https://ospreydb.dev/pricing"
    )]
    CommunityTierLimitReached {
        /// Hard limit (defaults to [`crate::license::COMMUNITY_MAX_VECTORS`]).
        limit: u64,
    },

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
