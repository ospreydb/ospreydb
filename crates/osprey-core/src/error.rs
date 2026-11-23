//! Canonical error hierarchy for OspreyDB core, SIMD, and storage subsystems.

use thiserror::Error;

/// Standard fallible result alias across the OspreyDB workspace.
pub type OspreyResult<T> = Result<T, OspreyError>;

/// Zero-allocation error hierarchy for binary parsing, alignment, math invariants, and I/O.
#[derive(Debug, Error)]
pub enum OspreyError {
    /// File magic bytes did not match `"OSPR"` (`[0x4F, 0x53, 0x50, 0x52]`).
    #[error("invalid magic bytes: expected [0x4f, 0x53, 0x50, 0x52], found {found:?}")]
    InvalidMagic {
        /// Observed 4-byte magic prefix.
        found: [u8; 4],
    },

    /// Artifact version does not match the version supported by this binary.
    #[error("unsupported format version: expected {expected}, found {found}")]
    UnsupportedVersion {
        /// Observed format version.
        found: u16,
        /// Expected format version.
        expected: u16,
    },

    /// Format kind byte (`0x01` = `.osi`, `0x02` = `.osv`, `0x03` = `.osr`) mismatched parser.
    #[error("invalid format kind: expected 0x{expected:02x}, found 0x{found:02x}")]
    InvalidFormatKind {
        /// Observed format discriminant byte.
        found: u8,
        /// Expected format discriminant byte.
        expected: u8,
    },

    /// `xxHash64` checksum validation failed on a header, minipage, or global footer.
    #[error("xxHash64 checksum mismatch: expected 0x{expected:016x}, computed 0x{computed:016x}")]
    ChecksumMismatch {
        /// Stored checksum value.
        expected: u64,
        /// Recomputed checksum value.
        computed: u64,
    },

    /// Pointer or section offset violates 64-byte cache-line alignment.
    #[error("unaligned memory at 0x{address:x}: required {required_align}-byte alignment")]
    UnalignedMemory {
        /// Observed virtual address or offset.
        address: usize,
        /// Required alignment in bytes (typically 64).
        required_align: usize,
    },

    /// Embedding dimensionality violates format or power-of-two invariants.
    #[error("invalid vector dimension: {dim}")]
    InvalidDimension {
        /// Invalid dimension count.
        dim: usize,
    },

    /// Input embedding contained `NaN` or `±Infinity`.
    #[error("non-finite embedding value at coordinate index {index}")]
    NonFiniteEmbedding {
        /// Coordinate index of the non-finite value.
        index: usize,
    },

    /// Slice or mapped region was smaller than required by the binary layout.
    #[error("buffer too small: needed at least {needed} bytes, got {actual}")]
    BufferTooSmall {
        /// Minimum required byte length.
        needed: usize,
        /// Actual byte length available.
        actual: usize,
    },

    /// Global trailer or tombstone footer failed structural validation.
    #[error("corrupted file footer or tombstone trailer")]
    CorruptedFooter,

    /// Aligned memory allocation overflowed `usize` or returned null.
    #[error("aligned allocation failed for {size} bytes (align={align})")]
    AllocationFailed {
        /// Requested byte size.
        size: usize,
        /// Requested alignment in bytes.
        align: usize,
    },

    /// Underlying OS / filesystem / `memmap2` I/O failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
