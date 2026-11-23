//! Zero-copy binary layout for the `.osv` (Osprey Locality-Reordered Vector Slabs) format.

use std::mem::{align_of, offset_of, size_of};

use bytemuck::{Pod, PodCastError, Zeroable};
use xxhash_rust::xxh64::xxh64;

use crate::aligned::{ensure_aligned_64, CACHE_LINE_ALIGN};
use crate::error::{OspreyError, OspreyResult};
use crate::format::{FORMAT_KIND_OSV, FORMAT_VERSION_V1, OSPREY_MAGIC};

/// Compression flag bit: Zstd frame compression enabled for minipages.
pub const OSV_FLAG_ZSTD_ENABLED: u16 = 0x0001;
/// Compression flag bit: segment embeds a trained Zstd dictionary at `zstd_dict_offset`.
pub const OSV_FLAG_HAS_ZSTD_DICT: u16 = 0x0002;
/// Compression flag bit: Zstd content checksum enabled on frames.
pub const OSV_FLAG_ZSTD_CHECKSUM: u16 = 0x0004;

/// Directory entry flag bit: minipage contains scalar zone map metadata.
pub const MINIPAGE_FLAG_HAS_ZONE_MAP: u16 = 0x0001;
/// Directory entry flag bit: uncompressed minipage exceeded target 16 KiB due to large payload.
pub const MINIPAGE_FLAG_OVERSIZED: u16 = 0x0002;

/// Fixed 64-byte header of an `.osv` vector slab file (`0x00..0x3F`).
#[repr(C, align(64))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct OsvHeader {
    /// Global magic bytes `"OSPR"`.
    pub magic: [u8; 4],
    /// Format discriminant (`0x02` for `.osv`).
    pub format_kind: u8,
    /// Zstd compression level (`0` if uncompressed or default).
    pub zstd_level: i8,
    /// Physical layout version (`1`).
    pub version: u16,
    /// Original unpadded vector dimensionality (`d`).
    pub orig_dim: u16,
    /// Bitflags (`OSV_FLAG_*`).
    pub flags: u16,
    /// Total number of minipages in the directory.
    pub num_minipages: u32,
    /// Total number of vectors across all minipages.
    pub total_vectors: u64,
    /// 64-byte-aligned byte offset to the optional Zstd dictionary (`0` if absent).
    pub zstd_dict_offset: u64,
    /// Byte length of the optional Zstd dictionary (`0` if absent).
    pub zstd_dict_len: u64,
    /// 64-byte-aligned byte offset to the `MinipageDirectoryEntry` array.
    pub directory_offset: u64,
    /// Explicit reserved field (`0`).
    pub _reserved: u64,
    /// `xxHash64` checksum (seed `0`) over the preceding 56 bytes (`0x00..0x38`).
    pub header_xxh64: u64,
}

const _: () = {
    assert!(size_of::<OsvHeader>() == 64);
    assert!(align_of::<OsvHeader>() == 64);
    assert!(offset_of!(OsvHeader, magic) == 0x00);
    assert!(offset_of!(OsvHeader, format_kind) == 0x04);
    assert!(offset_of!(OsvHeader, zstd_level) == 0x05);
    assert!(offset_of!(OsvHeader, version) == 0x06);
    assert!(offset_of!(OsvHeader, orig_dim) == 0x08);
    assert!(offset_of!(OsvHeader, flags) == 0x0A);
    assert!(offset_of!(OsvHeader, num_minipages) == 0x0C);
    assert!(offset_of!(OsvHeader, total_vectors) == 0x10);
    assert!(offset_of!(OsvHeader, zstd_dict_offset) == 0x18);
    assert!(offset_of!(OsvHeader, zstd_dict_len) == 0x20);
    assert!(offset_of!(OsvHeader, directory_offset) == 0x28);
    assert!(offset_of!(OsvHeader, _reserved) == 0x30);
    assert!(offset_of!(OsvHeader, header_xxh64) == 0x38);
};

impl OsvHeader {
    /// Creates a new `.osv` header with `header_xxh64` computed and sealed.
    #[allow(clippy::too_many_arguments)]
    pub fn new_sealed(
        zstd_level: i8,
        orig_dim: u16,
        flags: u16,
        num_minipages: u32,
        total_vectors: u64,
        zstd_dict_offset: u64,
        zstd_dict_len: u64,
        directory_offset: u64,
    ) -> OspreyResult<Self> {
        let mut header = Self {
            magic: OSPREY_MAGIC,
            format_kind: FORMAT_KIND_OSV,
            zstd_level,
            version: FORMAT_VERSION_V1,
            orig_dim,
            flags,
            num_minipages,
            total_vectors,
            zstd_dict_offset,
            zstd_dict_len,
            directory_offset,
            _reserved: 0,
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

    /// Validates magic bytes, format kind, version, `header_xxh64`, dimensions, and 64B offsets.
    pub fn validate(&self) -> OspreyResult<()> {
        if self.magic != OSPREY_MAGIC {
            return Err(OspreyError::InvalidMagic { found: self.magic });
        }
        if self.format_kind != FORMAT_KIND_OSV {
            return Err(OspreyError::InvalidFormatKind {
                found: self.format_kind,
                expected: FORMAT_KIND_OSV,
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

        if self.orig_dim == 0 {
            return Err(OspreyError::InvalidDimension { dim: 0 });
        }

        for offset in [self.zstd_dict_offset, self.directory_offset] {
            if !offset.is_multiple_of(64) {
                return Err(OspreyError::UnalignedMemory {
                    address: usize::try_from(offset).unwrap_or(usize::MAX),
                    required_align: CACHE_LINE_ALIGN,
                });
            }
        }

        Ok(())
    }

    /// Casts and validates an `OsvHeader` zero-copy from a 64-byte-aligned slice.
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

/// Directory entry describing a single Zstd-compressed minipage in `.osv` (48 bytes exact).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct MinipageDirectoryEntry {
    /// Absolute byte offset of the compressed minipage frame within the `.osv` object.
    pub byte_offset: u64,
    /// Minimum `doc_id` stored in this minipage (zone map fast-path).
    pub min_doc_id: u64,
    /// Maximum `doc_id` stored in this minipage (zone map fast-path).
    pub max_doc_id: u64,
    /// `xxHash64` checksum (seed `0`) over the compressed frame bytes `[byte_offset..+compressed_size]`.
    pub page_xxh64: u64,
    /// Compressed byte length to fetch via S3 `Range-GET`.
    pub compressed_size: u32,
    /// Exact uncompressed byte length for pre-sized decompression buffers.
    pub uncompressed_size: u32,
    /// TSP-reordered cluster identifier owning this minipage.
    pub cluster_id: u32,
    /// Number of vectors packed inside this minipage.
    pub vector_count: u16,
    /// Minipage flags (`MINIPAGE_FLAG_*`).
    pub flags: u16,
}

const _: () = {
    assert!(size_of::<MinipageDirectoryEntry>() == 48);
    assert!(align_of::<MinipageDirectoryEntry>() == 8);
    assert!(offset_of!(MinipageDirectoryEntry, byte_offset) == 0x00);
    assert!(offset_of!(MinipageDirectoryEntry, min_doc_id) == 0x08);
    assert!(offset_of!(MinipageDirectoryEntry, max_doc_id) == 0x10);
    assert!(offset_of!(MinipageDirectoryEntry, page_xxh64) == 0x18);
    assert!(offset_of!(MinipageDirectoryEntry, compressed_size) == 0x20);
    assert!(offset_of!(MinipageDirectoryEntry, uncompressed_size) == 0x24);
    assert!(offset_of!(MinipageDirectoryEntry, cluster_id) == 0x28);
    assert!(offset_of!(MinipageDirectoryEntry, vector_count) == 0x2C);
    assert!(offset_of!(MinipageDirectoryEntry, flags) == 0x2E);
};

/// Casts the `.osv` minipage directory slice zero-copy.
pub fn cast_minipage_directory(
    bytes: &[u8],
    num_minipages: u32,
) -> OspreyResult<&[MinipageDirectoryEntry]> {
    let count = usize::try_from(num_minipages).map_err(|_| OspreyError::AllocationFailed {
        size: usize::MAX,
        align: align_of::<MinipageDirectoryEntry>(),
    })?;
    let needed = count
        .checked_mul(size_of::<MinipageDirectoryEntry>())
        .ok_or(OspreyError::AllocationFailed {
            size: usize::MAX,
            align: align_of::<MinipageDirectoryEntry>(),
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
            required_align: align_of::<MinipageDirectoryEntry>(),
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
    fn osv_header_and_directory_roundtrip() {
        let header = OsvHeader::new_sealed(
            3,
            1536,
            OSV_FLAG_ZSTD_ENABLED | OSV_FLAG_HAS_ZSTD_DICT,
            2,
            100,
            64,
            65536,
            131_072,
        )
        .unwrap();

        let mut hdr_buf = AlignedBuffer::<u8>::new(64).unwrap();
        hdr_buf.copy_from_slice(bytemuck::bytes_of(&header));
        assert_eq!(OsvHeader::from_bytes(&hdr_buf).unwrap(), &header);

        hdr_buf[10] ^= 0x80;
        assert!(matches!(
            OsvHeader::from_bytes(&hdr_buf).unwrap_err(),
            OspreyError::ChecksumMismatch { .. }
        ));

        let entries = [
            MinipageDirectoryEntry {
                byte_offset: 65600,
                min_doc_id: 1,
                max_doc_id: 50,
                page_xxh64: 0xDEAD_BEEF_CAFE_BABE,
                compressed_size: 4096,
                uncompressed_size: 16384,
                cluster_id: 0,
                vector_count: 50,
                flags: MINIPAGE_FLAG_HAS_ZONE_MAP,
            },
            MinipageDirectoryEntry {
                byte_offset: 69696,
                min_doc_id: 51,
                max_doc_id: 100,
                page_xxh64: 0x1122_3344_5566_7788,
                compressed_size: 4200,
                uncompressed_size: 16384,
                cluster_id: 1,
                vector_count: 50,
                flags: MINIPAGE_FLAG_HAS_ZONE_MAP,
            },
        ];

        let dir_buf = AlignedBuffer::from_slice(&entries).unwrap();
        let parsed_dir = cast_minipage_directory(dir_buf.as_ref(), 2).unwrap();
        assert_eq!(parsed_dir, &entries);
    }
}
