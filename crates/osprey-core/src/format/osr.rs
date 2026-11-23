//! Zero-copy binary layout for the `.osr` (Osprey SIMD-Interleaved Token Rerank Store) format.

use std::mem::{align_of, offset_of, size_of};

use bytemuck::{Pod, PodCastError, Zeroable};
use xxhash_rust::xxh64::xxh64;

use crate::aligned::{ensure_aligned_64, CACHE_LINE_ALIGN};
use crate::error::{OspreyError, OspreyResult};
use crate::format::{FORMAT_KIND_OSR, FORMAT_VERSION_V1, OSPREY_MAGIC};

/// Canonical `.osr` token dimensionality (ColBERT standard).
pub const OSR_CANONICAL_TOKEN_DIM: u16 = 128;
/// Canonical `.osr` 2D subspace dimensionality.
pub const OSR_CANONICAL_SUBSPACE_DIM: u8 = 2;
/// Canonical `.osr` nested lattice layer count.
pub const OSR_CANONICAL_NUM_LAYERS: u8 = 2;
/// Canonical `.osr` residual contraction factor (`gamma = 0.28`).
pub const OSR_CANONICAL_GAMMA: f32 = 0.28;

/// Fixed 64-byte header of an `.osr` late-interaction rerank store (`0x00..0x3F`).
#[repr(C, align(64))]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct OsrHeader {
    /// Global magic bytes `"OSPR"`.
    pub magic: [u8; 4],
    /// Format discriminant (`0x03` for `.osr`).
    pub format_kind: u8,
    /// Subspace dimension (`2` by default).
    pub subspace_dim: u8,
    /// Number of nested lattice layers (`2` by default).
    pub num_layers: u8,
    /// Explicit alignment padding / format flags (`0`).
    pub _pad0: u8,
    /// Physical layout version (`1`).
    pub version: u16,
    /// Token embedding dimensionality (`128` by default).
    pub token_dim: u16,
    /// Residual lattice scale contraction factor (`0.28` by default).
    pub gamma: f32,
    /// Deterministic SplitMix64 seed for token FWHT rotation.
    pub fwht_seed: u64,
    /// Number of documents indexed in the document table.
    pub num_docs: u64,
    /// Total number of 64-byte tokens stored in `tensors_offset`.
    pub total_tokens: u64,
    /// 64-byte-aligned byte offset to the `DocTokenRange` table (sorted by `doc_id`).
    pub doc_table_offset: u64,
    /// 64-byte-aligned byte offset to the contiguous 64-byte-per-token nibble tensors.
    pub tensors_offset: u64,
    /// `xxHash64` checksum (seed `0`) over the preceding 56 bytes (`0x00..0x38`).
    pub header_xxh64: u64,
}

const _: () = {
    assert!(size_of::<OsrHeader>() == 64);
    assert!(align_of::<OsrHeader>() == 64);
    assert!(offset_of!(OsrHeader, magic) == 0x00);
    assert!(offset_of!(OsrHeader, format_kind) == 0x04);
    assert!(offset_of!(OsrHeader, subspace_dim) == 0x05);
    assert!(offset_of!(OsrHeader, num_layers) == 0x06);
    assert!(offset_of!(OsrHeader, _pad0) == 0x07);
    assert!(offset_of!(OsrHeader, version) == 0x08);
    assert!(offset_of!(OsrHeader, token_dim) == 0x0A);
    assert!(offset_of!(OsrHeader, gamma) == 0x0C);
    assert!(offset_of!(OsrHeader, fwht_seed) == 0x10);
    assert!(offset_of!(OsrHeader, num_docs) == 0x18);
    assert!(offset_of!(OsrHeader, total_tokens) == 0x20);
    assert!(offset_of!(OsrHeader, doc_table_offset) == 0x28);
    assert!(offset_of!(OsrHeader, tensors_offset) == 0x30);
    assert!(offset_of!(OsrHeader, header_xxh64) == 0x38);
};

impl OsrHeader {
    /// Creates a new `.osr` header with `header_xxh64` computed and sealed.
    #[allow(clippy::too_many_arguments)]
    pub fn new_sealed(
        subspace_dim: u8,
        num_layers: u8,
        token_dim: u16,
        gamma: f32,
        fwht_seed: u64,
        num_docs: u64,
        total_tokens: u64,
        doc_table_offset: u64,
        tensors_offset: u64,
    ) -> OspreyResult<Self> {
        let mut header = Self {
            magic: OSPREY_MAGIC,
            format_kind: FORMAT_KIND_OSR,
            subspace_dim,
            num_layers,
            _pad0: 0,
            version: FORMAT_VERSION_V1,
            token_dim,
            gamma,
            fwht_seed,
            num_docs,
            total_tokens,
            doc_table_offset,
            tensors_offset,
            header_xxh64: 0,
        };
        header.seal_checksum();
        header.validate()?;
        Ok(header)
    }

    /// Computes `xxHash64` (seed `0`) over the first 56 bytes (`0x00..0x38`).
    #[inline]
    pub fn compute_header_xxh64(&self) -> u64 {
        let bytes = bytemuck::bytes_of(self);
        xxh64(&bytes[..56], 0)
    }

    /// Recomputes and writes `header_xxh64`.
    #[inline]
    pub fn seal_checksum(&mut self) {
        self.header_xxh64 = self.compute_header_xxh64();
    }

    /// Validates magic bytes, format kind, version, `header_xxh64`, lattice params, and 64B offsets.
    pub fn validate(&self) -> OspreyResult<()> {
        if self.magic != OSPREY_MAGIC {
            return Err(OspreyError::InvalidMagic { found: self.magic });
        }
        if self.format_kind != FORMAT_KIND_OSR {
            return Err(OspreyError::InvalidFormatKind {
                found: self.format_kind,
                expected: FORMAT_KIND_OSR,
            });
        }
        if self.version != FORMAT_VERSION_V1 {
            return Err(OspreyError::UnsupportedVersion {
                found: self.version,
                expected: FORMAT_VERSION_V1,
            });
        }

        let computed = self.compute_header_xxh64();
        if self.header_xxh64 != computed {
            return Err(OspreyError::ChecksumMismatch {
                expected: self.header_xxh64,
                computed,
            });
        }

        if self.token_dim == 0
            || self.subspace_dim == 0
            || self.num_layers == 0
            || !usize::from(self.token_dim).is_multiple_of(usize::from(self.subspace_dim))
        {
            return Err(OspreyError::InvalidDimension {
                dim: usize::from(self.token_dim),
            });
        }

        if !self.gamma.is_finite() || self.gamma <= 0.0 {
            return Err(OspreyError::NonFiniteEmbedding { index: 0 });
        }

        for offset in [self.doc_table_offset, self.tensors_offset] {
            if !offset.is_multiple_of(64) {
                return Err(OspreyError::UnalignedMemory {
                    address: usize::try_from(offset).unwrap_or(usize::MAX),
                    required_align: CACHE_LINE_ALIGN,
                });
            }
        }

        Ok(())
    }

    /// Casts and validates an `OsrHeader` zero-copy from a 64-byte-aligned slice.
    pub fn from_bytes(bytes: &[u8]) -> OspreyResult<&Self> {
        if bytes.len() < size_of::<Self>() {
            return Err(OspreyError::BufferTooSmall {
                needed: size_of::<Self>(),
                actual: bytes.len(),
            });
        }
        ensure_aligned_64(bytes)?;
        let header: &Self = bytemuck::from_bytes(&bytes[..size_of::<Self>()]);
        header.validate()?;
        Ok(header)
    }
}

/// Maps a document `doc_id` to its contiguous range of 64-byte nibble-packed tokens (24 bytes).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct DocTokenRange {
    /// Logical document identifier (sorted ascending in the `.osr` document table).
    pub doc_id: u64,
    /// Starting token index inside `tensors_offset` (byte offset = `token_start_idx * 64`).
    pub token_start_idx: u64,
    /// Number of contiguous tokens belonging to this document.
    pub token_count: u16,
    /// Explicit zero padding closing 24 bytes without implicit compiler padding.
    pub _reserved: [u8; 6],
}

const _: () = {
    assert!(size_of::<DocTokenRange>() == 24);
    assert!(align_of::<DocTokenRange>() == 8);
    assert!(offset_of!(DocTokenRange, doc_id) == 0x00);
    assert!(offset_of!(DocTokenRange, token_start_idx) == 0x08);
    assert!(offset_of!(DocTokenRange, token_count) == 0x10);
    assert!(offset_of!(DocTokenRange, _reserved) == 0x12);
};

/// Casts the `.osr` document table slice zero-copy.
pub fn cast_doc_token_ranges(bytes: &[u8], num_docs: u64) -> OspreyResult<&[DocTokenRange]> {
    let count = usize::try_from(num_docs).map_err(|_| OspreyError::AllocationFailed {
        size: usize::MAX,
        align: align_of::<DocTokenRange>(),
    })?;
    let needed =
        count
            .checked_mul(size_of::<DocTokenRange>())
            .ok_or(OspreyError::AllocationFailed {
                size: usize::MAX,
                align: align_of::<DocTokenRange>(),
            })?;
    if bytes.len() < needed {
        return Err(OspreyError::BufferTooSmall {
            needed,
            actual: bytes.len(),
        });
    }
    bytemuck::try_cast_slice(&bytes[..needed]).map_err(|err| match err {
        PodCastError::TargetAlignmentGreaterAndInputNotAligned => OspreyError::UnalignedMemory {
            address: bytes.as_ptr() as usize,
            required_align: align_of::<DocTokenRange>(),
        },
        _ => OspreyError::BufferTooSmall {
            needed,
            actual: bytes.len(),
        },
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::aligned::AlignedBuffer;

    #[test]
    fn osr_header_and_doc_token_range_roundtrip() {
        let header = OsrHeader::new_sealed(
            OSR_CANONICAL_SUBSPACE_DIM,
            OSR_CANONICAL_NUM_LAYERS,
            OSR_CANONICAL_TOKEN_DIM,
            OSR_CANONICAL_GAMMA,
            0x9876_5432,
            2,
            128,
            64,
            128,
        )
        .unwrap();

        let mut hdr_buf = AlignedBuffer::<u8>::new(64).unwrap();
        hdr_buf.copy_from_slice(bytemuck::bytes_of(&header));
        assert_eq!(OsrHeader::from_bytes(&hdr_buf).unwrap(), &header);

        hdr_buf[20] ^= 0x01;
        assert!(matches!(
            OsrHeader::from_bytes(&hdr_buf).unwrap_err(),
            OspreyError::ChecksumMismatch { .. }
        ));

        let ranges = [
            DocTokenRange {
                doc_id: 10,
                token_start_idx: 0,
                token_count: 64,
                _reserved: [0; 6],
            },
            DocTokenRange {
                doc_id: 42,
                token_start_idx: 64,
                token_count: 64,
                _reserved: [0; 6],
            },
        ];

        let table_buf = AlignedBuffer::from_slice(&ranges).unwrap();
        let parsed = cast_doc_token_ranges(table_buf.as_ref(), 2).unwrap();
        assert_eq!(parsed, &ranges);
    }
}
