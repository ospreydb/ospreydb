//! Zero-copy binary layout for the `.osi` (Osprey Resident 1-Bit Index) format.
//!
//! All multi-byte integers and IEEE-754 floats are Little-Endian. Major tabular
//! sections start on 64-byte cache-line boundaries (`offset % 64 == 0`).

use std::io::Cursor;
use std::mem::{align_of, offset_of, size_of};

use bytemuck::{Pod, PodCastError, Zeroable};
use half::f16;
use roaring::RoaringBitmap;
use xxhash_rust::xxh64::xxh64;

use crate::aligned::{ensure_aligned_64, CACHE_LINE_ALIGN};
use crate::error::{OspreyError, OspreyResult};
use crate::format::{checked_align_up_64, FORMAT_KIND_OSI, FORMAT_VERSION_V1, OSPREY_MAGIC};

#[cfg(not(target_endian = "little"))]
compile_error!("OspreyDB binary formats require a little-endian target architecture");

/// `OsiHeader::flags` bit indicating that stored `centroids_f16` reside in unrotated padded
/// space (so cluster probing uses `q_padded` and only intra-cluster residuals are FWHT-rotated).
pub const OSI_FLAG_UNROTATED_CENTROIDS: u8 = 0x02;

/// `OsiHeader::flags` bit indicating 4-bit `RaBitQ` quantization stored in transposed
/// 32-vector FastScan (`vpshufb`) blocks instead of 1-bit `u64` words.
pub const OSI_FLAG_RABITQ_4BIT: u8 = 0x04;

/// `OsiHeader::flags` bit (paired with [`OSI_FLAG_RABITQ_4BIT`]) indicating progressive bitplane
/// 4-bit layout for high-dimensional indexes (`padded_dim >= 512`): MSB plane `b3` (`1-bit`)
/// packed contiguously first per cluster, followed by `[b2, b1, b0]` (`3-bit` extension).
pub const OSI_FLAG_RABITQ_PROGRESSIVE: u8 = 0x08;

/// Number of vectors packed per transposed 4-bit FastScan (`vpshufb`) block.
pub const FASTSCAN_BLOCK_SIZE: usize = 32;

/// Fixed 64-byte header of a `.osi` index file (`0x00..0x3F`).
#[repr(C, align(64))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct OsiHeader {
    /// Global magic bytes `"OSPR"`.
    pub magic: [u8; 4],
    /// Format discriminant (`0x01` for `.osi`).
    pub format_kind: u8,
    /// Bitflags for index capabilities (`0x00` in v1).
    pub flags: u8,
    /// Physical layout version (`1`).
    pub version: u16,
    /// Original unpadded embedding dimensionality.
    pub orig_dim: u16,
    /// FWHT power-of-two padded dimensionality (`>= 64`, multiple of 64).
    pub padded_dim: u16,
    /// Number of `1-Bit Binary K-Means` clusters `K`.
    pub num_clusters: u32,
    /// Total indexed vectors `N`.
    pub num_vectors: u64,
    /// Deterministic SplitMix64 seed for FWHT Rademacher signs.
    pub fwht_seed: u64,
    /// 64-byte aligned byte offset to the `f16` TSP-ordered centroids table.
    pub centroids_offset: u64,
    /// 64-byte aligned byte offset to the cluster-grouped `RaBitQ` directory.
    pub entries_offset: u64,
    /// 64-byte aligned byte offset to the serialized `RoaringBitmap` tombstone footer.
    pub tombstones_offset: u64,
    /// `xxHash64` checksum (seed `0`) over the preceding 56 bytes (`0x00..0x38`).
    pub header_xxh64: u64,
}

const _: () = {
    assert!(size_of::<OsiHeader>() == 64);
    assert!(align_of::<OsiHeader>() == 64);
    assert!(offset_of!(OsiHeader, magic) == 0x00);
    assert!(offset_of!(OsiHeader, format_kind) == 0x04);
    assert!(offset_of!(OsiHeader, flags) == 0x05);
    assert!(offset_of!(OsiHeader, version) == 0x06);
    assert!(offset_of!(OsiHeader, orig_dim) == 0x08);
    assert!(offset_of!(OsiHeader, padded_dim) == 0x0A);
    assert!(offset_of!(OsiHeader, num_clusters) == 0x0C);
    assert!(offset_of!(OsiHeader, num_vectors) == 0x10);
    assert!(offset_of!(OsiHeader, fwht_seed) == 0x18);
    assert!(offset_of!(OsiHeader, centroids_offset) == 0x20);
    assert!(offset_of!(OsiHeader, entries_offset) == 0x28);
    assert!(offset_of!(OsiHeader, tombstones_offset) == 0x30);
    assert!(offset_of!(OsiHeader, header_xxh64) == 0x38);
};

impl OsiHeader {
    /// Creates a new `.osi` header with `header_xxh64` computed and sealed.
    #[allow(clippy::too_many_arguments)]
    pub fn new_sealed(
        flags: u8,
        orig_dim: u16,
        padded_dim: u16,
        num_clusters: u32,
        num_vectors: u64,
        fwht_seed: u64,
        centroids_offset: u64,
        entries_offset: u64,
        tombstones_offset: u64,
    ) -> OspreyResult<Self> {
        let mut header = Self {
            magic: OSPREY_MAGIC,
            format_kind: FORMAT_KIND_OSI,
            flags,
            version: FORMAT_VERSION_V1,
            orig_dim,
            padded_dim,
            num_clusters,
            num_vectors,
            fwht_seed,
            centroids_offset,
            entries_offset,
            tombstones_offset,
            header_xxh64: 0,
        };
        header.seal_checksum();
        header.validate()?;
        Ok(header)
    }

    /// Computes `xxHash64` (seed `0`) over the first 56 bytes (`0x00..0x38`) of the header.
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

    /// Returns the `RaBitQ` quantization bit-width (`1` or `4`) encoded in `flags`.
    #[inline]
    pub const fn rabitq_bits(&self) -> u8 {
        if (self.flags & OSI_FLAG_RABITQ_4BIT) != 0 {
            4
        } else {
            1
        }
    }

    /// Returns `true` when this `.osi` index stores 4-bit `RaBitQ` quantization.
    #[inline]
    pub const fn is_4bit(&self) -> bool {
        (self.flags & OSI_FLAG_RABITQ_4BIT) != 0
    }

    /// Returns `true` when this `.osi` index stores bitplane-sliced progressive 4-bit `RaBitQ`.
    #[inline]
    pub const fn is_progressive_4bit(&self) -> bool {
        (self.flags & (OSI_FLAG_RABITQ_4BIT | OSI_FLAG_RABITQ_PROGRESSIVE))
            == (OSI_FLAG_RABITQ_4BIT | OSI_FLAG_RABITQ_PROGRESSIVE)
    }

    /// Validates magic bytes, format kind, version, `header_xxh64`, dimensions, and 64B offsets.
    pub fn validate(&self) -> OspreyResult<()> {
        if self.magic != OSPREY_MAGIC {
            return Err(OspreyError::InvalidMagic { found: self.magic });
        }
        if self.format_kind != FORMAT_KIND_OSI {
            return Err(OspreyError::InvalidFormatKind {
                found: self.format_kind,
                expected: FORMAT_KIND_OSI,
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

        if self.orig_dim == 0
            || self.padded_dim < 64
            || self.padded_dim < self.orig_dim
            || !self.padded_dim.is_power_of_two()
        {
            return Err(OspreyError::InvalidDimension {
                dim: usize::from(self.padded_dim),
            });
        }

        for offset in [
            self.centroids_offset,
            self.entries_offset,
            self.tombstones_offset,
        ] {
            if !offset.is_multiple_of(64) {
                return Err(OspreyError::UnalignedMemory {
                    address: usize::try_from(offset).unwrap_or(usize::MAX),
                    required_align: CACHE_LINE_ALIGN,
                });
            }
        }

        Ok(())
    }

    /// Casts and validates an `OsiHeader` zero-copy from a 64-byte-aligned slice.
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

/// Per-vector metadata record in the `.osi` directory (24 bytes, zero implicit padding).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct RabitqEntryHeader {
    /// Global document identifier.
    pub doc_id: u64,
    /// Target minipage index within the `.osv` slab file.
    pub minipage_id: u32,
    /// Target slab identifier (0 for single-slab segments).
    pub slab_id: u16,
    /// Vector slot index within the minipage.
    pub slot_idx: u16,
    /// Euclidean norm of the residual `||x - c_k||_2` in IEEE-754 `f16`.
    pub res_norm: f16,
    /// Asymmetric `RaBitQ` L1 projection factor `||P u||_1 / D` in IEEE-754 `f16`.
    pub l1_norm_factor: f16,
    /// Explicit zero padding closing 24 bytes (multiple of `align_of::<u64>()`).
    pub _reserved: u32,
}

const _: () = {
    assert!(size_of::<RabitqEntryHeader>() == 24);
    assert!(align_of::<RabitqEntryHeader>() == 8);
    assert!(offset_of!(RabitqEntryHeader, doc_id) == 0x00);
    assert!(offset_of!(RabitqEntryHeader, minipage_id) == 0x08);
    assert!(offset_of!(RabitqEntryHeader, slab_id) == 0x0C);
    assert!(offset_of!(RabitqEntryHeader, slot_idx) == 0x0E);
    assert!(offset_of!(RabitqEntryHeader, res_norm) == 0x10);
    assert!(offset_of!(RabitqEntryHeader, l1_norm_factor) == 0x12);
    assert!(offset_of!(RabitqEntryHeader, _reserved) == 0x14);
};

/// Cluster span descriptor in the `.osi` directory (16 bytes, naturally aligned).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct OsiClusterSpan {
    /// Byte offset of this cluster's contiguous block relative to `entries_offset` (`% 64 == 0`).
    pub cluster_offset: u64,
    /// Number of vectors assigned to this cluster.
    pub vector_count: u32,
    /// Explicit reserved flags / padding (`0`).
    pub _reserved: u32,
}

const _: () = {
    assert!(size_of::<OsiClusterSpan>() == 16);
    assert!(align_of::<OsiClusterSpan>() == 8);
    assert!(offset_of!(OsiClusterSpan, cluster_offset) == 0x00);
    assert!(offset_of!(OsiClusterSpan, vector_count) == 0x08);
    assert!(offset_of!(OsiClusterSpan, _reserved) == 0x0C);
};

/// Returns the raw unpadded byte size of the `f16` centroid table (`K * padded_dim * 2`).
#[inline]
pub fn centroids_unpadded_bytes(num_clusters: u32, padded_dim: u16) -> OspreyResult<usize> {
    let k = usize::try_from(num_clusters).map_err(|_| OspreyError::InvalidDimension {
        dim: usize::from(padded_dim),
    })?;
    let d = usize::from(padded_dim);
    k.checked_mul(d)
        .and_then(|elems| elems.checked_mul(size_of::<f16>()))
        .ok_or(OspreyError::AllocationFailed {
            size: usize::MAX,
            align: CACHE_LINE_ALIGN,
        })
}

/// Returns the 64-byte-padded byte size of the `f16` centroid table.
#[inline]
pub fn centroids_padded_bytes(num_clusters: u32, padded_dim: u16) -> OspreyResult<usize> {
    let raw = centroids_unpadded_bytes(num_clusters, padded_dim)?;
    checked_align_up_64(raw).ok_or(OspreyError::AllocationFailed {
        size: usize::MAX,
        align: CACHE_LINE_ALIGN,
    })
}

/// Exposes the TSP-ordered `f16` centroid table as a zero-copy slice.
pub fn cast_centroids_f16(
    bytes: &[u8],
    num_clusters: u32,
    padded_dim: u16,
) -> OspreyResult<&[f16]> {
    let needed = centroids_unpadded_bytes(num_clusters, padded_dim)?;
    if bytes.len() < needed {
        return Err(OspreyError::BufferTooSmall {
            needed,
            actual: bytes.len(),
        });
    }
    bytemuck::try_cast_slice(&bytes[..needed]).map_err(|err| map_pod_cast_err(err, bytes))
}

/// Exposes a mutable zero-copy `f16` centroid slice for `.osi` index construction.
pub fn cast_centroids_f16_mut(
    bytes: &mut [u8],
    num_clusters: u32,
    padded_dim: u16,
) -> OspreyResult<&mut [f16]> {
    let needed = centroids_unpadded_bytes(num_clusters, padded_dim)?;
    if bytes.len() < needed {
        return Err(OspreyError::BufferTooSmall {
            needed,
            actual: bytes.len(),
        });
    }
    let addr = bytes.as_ptr() as usize;
    bytemuck::try_cast_slice_mut(&mut bytes[..needed]).map_err(|err| match err {
        PodCastError::TargetAlignmentGreaterAndInputNotAligned => OspreyError::UnalignedMemory {
            address: addr,
            required_align: align_of::<f16>(),
        },
        _ => OspreyError::BufferTooSmall {
            needed,
            actual: needed,
        },
    })
}

/// Zero-copy view over one cluster's contiguous 1-bit `RaBitQ` codes and entry headers.
#[derive(Copy, Clone, Debug)]
pub struct OsiClusterView<'a> {
    /// Contiguous `[u64; vector_count * words_per_vec]` 1-bit sign codes (64B-aligned start).
    pub codes: &'a [u64],
    /// Contiguous `RabitqEntryHeader` metadata records (`vector_count` elements).
    pub entries: &'a [RabitqEntryHeader],
    /// Number of `u64` words per vector (`padded_dim / 64`).
    pub words_per_vec: usize,
}

impl<'a> OsiClusterView<'a> {
    /// Returns the `[u64; padded_dim / 64]` bitcode slice for local vector index `idx`.
    #[inline]
    pub fn vector_code(&self, idx: usize) -> Option<&'a [u64]> {
        let start = idx.checked_mul(self.words_per_vec)?;
        let end = start.checked_add(self.words_per_vec)?;
        self.codes.get(start..end)
    }

    /// Parses a contiguous cluster block (`[codes_u64 (pad 64B) | entries (pad 64B)]`) zero-copy.
    pub fn from_cluster_bytes(
        bytes: &'a [u8],
        vector_count: usize,
        padded_dim: u16,
    ) -> OspreyResult<Self> {
        let codes_bytes = cluster_codes_bytes_for_bits(vector_count, padded_dim, 1)?;
        let (entries_offset, entries_bytes, needed, _) =
            cluster_layout_parts(codes_bytes, vector_count)?;
        ensure_aligned_64(bytes)?;
        if bytes.len() < needed {
            return Err(OspreyError::BufferTooSmall {
                needed,
                actual: bytes.len(),
            });
        }

        let words_per_vec = usize::from(padded_dim) / 64;
        let codes: &'a [u64] = bytemuck::try_cast_slice(&bytes[..codes_bytes])
            .map_err(|err| map_pod_cast_err(err, &bytes[..codes_bytes]))?;
        let entries_slice = &bytes[entries_offset..entries_offset + entries_bytes];
        let entries: &'a [RabitqEntryHeader] = bytemuck::try_cast_slice(entries_slice)
            .map_err(|err| map_pod_cast_err(err, entries_slice))?;

        Ok(Self {
            codes,
            entries,
            words_per_vec,
        })
    }
}

/// Zero-copy view over one cluster's transposed 32-vector 4-bit FastScan blocks and entry headers.
#[derive(Copy, Clone, Debug)]
pub struct OsiClusterView4Bit<'a> {
    /// Contiguous transposed 4-bit FastScan blocks (`num_blocks * 16 * padded_dim` bytes).
    pub blocks: &'a [u8],
    /// Contiguous `RabitqEntryHeader` metadata records (`vector_count` elements).
    pub entries: &'a [RabitqEntryHeader],
    /// Number of 32-vector FastScan blocks in this cluster.
    pub num_blocks: usize,
}

impl<'a> OsiClusterView4Bit<'a> {
    /// Parses a contiguous 4-bit FastScan cluster block (`[blocks (pad 64B) | entries (pad 64B)]`)
    /// zero-copy.
    pub fn from_cluster_bytes(
        bytes: &'a [u8],
        vector_count: usize,
        padded_dim: u16,
    ) -> OspreyResult<Self> {
        let codes_bytes = cluster_codes_bytes_for_bits(vector_count, padded_dim, 4)?;
        let (entries_offset, entries_bytes, needed, _) =
            cluster_layout_parts(codes_bytes, vector_count)?;
        ensure_aligned_64(bytes)?;
        if bytes.len() < needed {
            return Err(OspreyError::BufferTooSmall {
                needed,
                actual: bytes.len(),
            });
        }

        let num_blocks = vector_count.div_ceil(FASTSCAN_BLOCK_SIZE);
        let blocks = &bytes[..codes_bytes];
        let entries_slice = &bytes[entries_offset..entries_offset + entries_bytes];
        let entries: &'a [RabitqEntryHeader] = bytemuck::try_cast_slice(entries_slice)
            .map_err(|err| map_pod_cast_err(err, entries_slice))?;

        Ok(Self {
            blocks,
            entries,
            num_blocks,
        })
    }
}

/// Returns the byte length of one 32-vector 4-bit FastScan block (`16 * padded_dim`, always `% 64 == 0`).
#[inline]
pub fn fastscan_block_bytes(padded_dim: u16) -> OspreyResult<usize> {
    if padded_dim < 64 || !padded_dim.is_power_of_two() {
        return Err(OspreyError::InvalidDimension {
            dim: usize::from(padded_dim),
        });
    }
    usize::from(padded_dim)
        .checked_mul(FASTSCAN_BLOCK_SIZE / 2)
        .ok_or(OspreyError::AllocationFailed {
            size: usize::MAX,
            align: CACHE_LINE_ALIGN,
        })
}

/// Returns the unpadded code-section byte size for `vector_count` vectors at `bits` (`1` or `4`).
pub fn cluster_codes_bytes_for_bits(
    vector_count: usize,
    padded_dim: u16,
    bits: u8,
) -> OspreyResult<usize> {
    if padded_dim < 64 || !padded_dim.is_power_of_two() {
        return Err(OspreyError::InvalidDimension {
            dim: usize::from(padded_dim),
        });
    }
    match bits {
        1 => {
            let words_per_vec = usize::from(padded_dim) / 64;
            vector_count
                .checked_mul(words_per_vec)
                .and_then(|w| w.checked_mul(size_of::<u64>()))
                .ok_or(OspreyError::AllocationFailed {
                    size: usize::MAX,
                    align: CACHE_LINE_ALIGN,
                })
        }
        4 => {
            let num_blocks = vector_count.div_ceil(FASTSCAN_BLOCK_SIZE);
            let block_bytes = fastscan_block_bytes(padded_dim)?;
            num_blocks
                .checked_mul(block_bytes)
                .ok_or(OspreyError::AllocationFailed {
                    size: usize::MAX,
                    align: CACHE_LINE_ALIGN,
                })
        }
        other => Err(OspreyError::InvalidDimension {
            dim: usize::from(other),
        }),
    }
}

fn cluster_layout_parts(
    codes_bytes: usize,
    vector_count: usize,
) -> OspreyResult<(usize, usize, usize, usize)> {
    let entries_offset = checked_align_up_64(codes_bytes).ok_or(OspreyError::AllocationFailed {
        size: usize::MAX,
        align: CACHE_LINE_ALIGN,
    })?;
    let entries_bytes = vector_count
        .checked_mul(size_of::<RabitqEntryHeader>())
        .ok_or(OspreyError::AllocationFailed {
            size: usize::MAX,
            align: CACHE_LINE_ALIGN,
        })?;
    let needed =
        entries_offset
            .checked_add(entries_bytes)
            .ok_or(OspreyError::AllocationFailed {
                size: usize::MAX,
                align: CACHE_LINE_ALIGN,
            })?;
    let total = checked_align_up_64(needed).ok_or(OspreyError::AllocationFailed {
        size: usize::MAX,
        align: CACHE_LINE_ALIGN,
    })?;
    Ok((entries_offset, entries_bytes, needed, total))
}

/// Returns the unpadded code-section byte size for `vector_count` vectors matching `flags`.
pub fn cluster_codes_bytes_for_flags(
    vector_count: usize,
    padded_dim: u16,
    flags: u8,
) -> OspreyResult<usize> {
    if padded_dim < 64 || !padded_dim.is_power_of_two() {
        return Err(OspreyError::InvalidDimension {
            dim: usize::from(padded_dim),
        });
    }
    if (flags & OSI_FLAG_RABITQ_4BIT) == 0 {
        return cluster_codes_bytes_for_bits(vector_count, padded_dim, 1);
    }
    if (flags & OSI_FLAG_RABITQ_PROGRESSIVE) != 0 {
        let words_per_vec = usize::from(padded_dim) / 64;
        return vector_count
            .checked_mul(words_per_vec)
            .and_then(|w| w.checked_mul(4 * size_of::<u64>()))
            .ok_or(OspreyError::AllocationFailed {
                size: usize::MAX,
                align: CACHE_LINE_ALIGN,
            });
    }
    cluster_codes_bytes_for_bits(vector_count, padded_dim, 4)
}

/// Returns the 64-byte-aligned byte span required for one cluster block matching `flags`.
pub fn cluster_block_bytes_for_flags(
    vector_count: usize,
    padded_dim: u16,
    flags: u8,
) -> OspreyResult<usize> {
    let codes_bytes = cluster_codes_bytes_for_flags(vector_count, padded_dim, flags)?;
    let (_, _, _, total) = cluster_layout_parts(codes_bytes, vector_count)?;
    Ok(total)
}

/// Zero-copy view over one cluster's progressive bitplane 4-bit `RaBitQ` data and entry headers.
#[derive(Copy, Clone, Debug)]
pub struct OsiClusterViewProgressive4Bit<'a> {
    /// Contiguous MSB (`b3`) 1-bit `u64` words (`vector_count * words_per_vec`).
    pub plane3: &'a [u64],
    /// Contiguous `[b2, b1, b0]` 3-bit extension `u64` words (`vector_count * 3 * words_per_vec`).
    pub low_planes: &'a [u64],
    /// Contiguous `RabitqEntryHeader` metadata records (`vector_count` elements).
    pub entries: &'a [RabitqEntryHeader],
    /// `u64` words per vector per bitplane (`padded_dim / 64`).
    pub words_per_vec: usize,
}

impl<'a> OsiClusterViewProgressive4Bit<'a> {
    /// Parses a progressive bitplane 4-bit cluster block (`[codes (pad 64B) | entries (pad 64B)]`).
    pub fn from_cluster_bytes(
        bytes: &'a [u8],
        vector_count: usize,
        padded_dim: u16,
    ) -> OspreyResult<Self> {
        let flags = OSI_FLAG_RABITQ_4BIT | OSI_FLAG_RABITQ_PROGRESSIVE;
        let codes_bytes = cluster_codes_bytes_for_flags(vector_count, padded_dim, flags)?;
        let (entries_offset, entries_bytes, needed, _) =
            cluster_layout_parts(codes_bytes, vector_count)?;
        ensure_aligned_64(bytes)?;
        if bytes.len() < needed {
            return Err(OspreyError::BufferTooSmall {
                needed,
                actual: bytes.len(),
            });
        }
        let words_per_vec = usize::from(padded_dim) / 64;
        let all_words: &'a [u64] = bytemuck::try_cast_slice(&bytes[..codes_bytes])
            .map_err(|err| map_pod_cast_err(err, &bytes[..codes_bytes]))?;
        let plane3_words = vector_count.saturating_mul(words_per_vec);
        let (plane3, low_planes) = all_words.split_at(plane3_words);
        let entries_slice = &bytes[entries_offset..entries_offset + entries_bytes];
        let entries: &'a [RabitqEntryHeader] = bytemuck::try_cast_slice(entries_slice)
            .map_err(|err| map_pod_cast_err(err, entries_slice))?;
        Ok(Self {
            plane3,
            low_planes,
            entries,
            words_per_vec,
        })
    }
}

/// Returns the 64-byte-aligned byte span required for one 1-bit cluster block (`codes + entries`).
pub fn cluster_block_bytes(vector_count: usize, padded_dim: u16) -> OspreyResult<usize> {
    cluster_block_bytes_for_bits(vector_count, padded_dim, 1)
}

/// Returns the 64-byte-aligned byte span required for one cluster block at `bits` (`1` or `4`).
pub fn cluster_block_bytes_for_bits(
    vector_count: usize,
    padded_dim: u16,
    bits: u8,
) -> OspreyResult<usize> {
    let codes_bytes = cluster_codes_bytes_for_bits(vector_count, padded_dim, bits)?;
    let (_, _, _, total) = cluster_layout_parts(codes_bytes, vector_count)?;
    Ok(total)
}

/// Serializes the `.osi` tombstone footer:
/// `[bitmap_byte_len: u64 LE][portable RoaringBitmap bytes][tombstone_xxh64: u64 LE]`.
///
/// The `tombstone_xxh64` checksum covers `[bitmap_byte_len || portable RoaringBitmap bytes]`
/// with seed `0`.
pub fn serialize_tombstone_footer(bitmap: &RoaringBitmap, out: &mut Vec<u8>) -> OspreyResult<()> {
    let payload_len = bitmap.serialized_size();
    let payload_len_u64 = u64::try_from(payload_len).map_err(|_| OspreyError::CorruptedFooter)?;

    let start = out.len();
    out.reserve(size_of::<u64>() + payload_len + size_of::<u64>());
    out.extend_from_slice(&payload_len_u64.to_le_bytes());

    bitmap.serialize_into(&mut *out)?;
    let checksum = xxh64(&out[start..], 0);
    out.extend_from_slice(&checksum.to_le_bytes());
    Ok(())
}

/// Validates `tombstone_xxh64` before deserializing the `RoaringBitmap` tombstone trailer.
///
/// Returns `(bitmap, total_footer_bytes_consumed)`.
pub fn deserialize_tombstone_footer(bytes: &[u8]) -> OspreyResult<(RoaringBitmap, usize)> {
    const MIN_FOOTER_BYTES: usize = size_of::<u64>() + size_of::<u64>();
    if bytes.len() < MIN_FOOTER_BYTES {
        return Err(OspreyError::BufferTooSmall {
            needed: MIN_FOOTER_BYTES,
            actual: bytes.len(),
        });
    }

    let len_bytes: [u8; 8] = bytes[..8]
        .try_into()
        .map_err(|_| OspreyError::CorruptedFooter)?;
    let payload_len_u64 = u64::from_le_bytes(len_bytes);
    let payload_len = usize::try_from(payload_len_u64).map_err(|_| OspreyError::CorruptedFooter)?;

    let checksum_offset = 8usize
        .checked_add(payload_len)
        .ok_or(OspreyError::CorruptedFooter)?;
    let total_len = checksum_offset
        .checked_add(size_of::<u64>())
        .ok_or(OspreyError::CorruptedFooter)?;

    if bytes.len() < total_len {
        return Err(OspreyError::BufferTooSmall {
            needed: total_len,
            actual: bytes.len(),
        });
    }

    let stored_bytes: [u8; 8] = bytes[checksum_offset..total_len]
        .try_into()
        .map_err(|_| OspreyError::CorruptedFooter)?;
    let expected = u64::from_le_bytes(stored_bytes);
    let computed = xxh64(&bytes[..checksum_offset], 0);
    if expected != computed {
        return Err(OspreyError::ChecksumMismatch { expected, computed });
    }

    let bitmap = RoaringBitmap::deserialize_from(Cursor::new(&bytes[8..checksum_offset]))
        .map_err(|_| OspreyError::CorruptedFooter)?;

    Ok((bitmap, total_len))
}

#[inline]
fn map_pod_cast_err(err: PodCastError, slice: &[u8]) -> OspreyError {
    match err {
        PodCastError::TargetAlignmentGreaterAndInputNotAligned => OspreyError::UnalignedMemory {
            address: slice.as_ptr() as usize,
            required_align: CACHE_LINE_ALIGN,
        },
        _ => OspreyError::BufferTooSmall {
            needed: slice.len(),
            actual: slice.len(),
        },
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::aligned::AlignedBuffer;

    #[test]
    fn osi_header_roundtrip_and_corruption_rejection() {
        let header =
            OsiHeader::new_sealed(0, 384, 512, 64, 8192, 0x1234_5678, 64, 65600, 131_072).unwrap();

        let mut buf = AlignedBuffer::<u8>::new(64).unwrap();
        buf.copy_from_slice(bytemuck::bytes_of(&header));

        let parsed = OsiHeader::from_bytes(&buf).unwrap();
        assert_eq!(parsed, &header);

        // Corrupt a single payload bit -> must fail with ChecksumMismatch
        buf[12] ^= 0x01;
        let err = OsiHeader::from_bytes(&buf).unwrap_err();
        assert!(matches!(err, OspreyError::ChecksumMismatch { .. }));

        // Corrupt magic -> InvalidMagic
        buf[0] = b'X';
        let err = OsiHeader::from_bytes(&buf).unwrap_err();
        assert!(matches!(err, OspreyError::InvalidMagic { .. }));
    }

    #[test]
    fn centroids_f16_zero_copy_slice_and_padding() {
        assert_eq!(centroids_unpadded_bytes(3, 64).unwrap(), 3 * 64 * 2);
        assert_eq!(centroids_padded_bytes(3, 64).unwrap(), 384);
        assert_eq!(centroids_unpadded_bytes(1, 32 + 32).unwrap(), 128);

        let mut buf = AlignedBuffer::<u8>::new(centroids_padded_bytes(2, 64).unwrap()).unwrap();
        let slice_mut = cast_centroids_f16_mut(&mut buf, 2, 64).unwrap();
        slice_mut[0] = f16::from_f32(1.5);
        slice_mut[127] = f16::from_f32(-0.25);

        let slice = cast_centroids_f16(&buf, 2, 64).unwrap();
        assert_eq!(slice.len(), 128);
        assert!((slice[0].to_f32() - 1.5).abs() < 1e-3);
        assert!((slice[127].to_f32() + 0.25).abs() < 1e-3);
    }

    #[test]
    fn cluster_view_and_rabitq_entry_header_zero_copy() {
        let vec_count = 3usize;
        let padded_dim = 128u16; // 2 u64 words per vector
        let block_bytes = cluster_block_bytes(vec_count, padded_dim).unwrap();
        assert_eq!(block_bytes % 64, 0);

        let mut buf = AlignedBuffer::<u8>::new(block_bytes).unwrap();
        // Write 3 vector bitcodes (3 * 2 * 8 = 48 bytes, padded to 64 bytes)
        let codes: &mut [u64] = bytemuck::try_cast_slice_mut(&mut buf[..48]).unwrap();
        codes[0] = 0xAA55_0000_FFFF_1111;
        codes[1] = 0x1234_5678_9ABC_DEF0;
        codes[4] = 0xFFFF_FFFF_FFFF_FFFF;

        // Write 3 RabitqEntryHeader structs starting at offset 64
        let entries: &mut [RabitqEntryHeader] =
            bytemuck::try_cast_slice_mut(&mut buf[64..64 + 3 * 24]).unwrap();
        entries[0] = RabitqEntryHeader {
            doc_id: 100,
            minipage_id: 7,
            slab_id: 1,
            slot_idx: 2,
            res_norm: f16::from_f32(0.75),
            l1_norm_factor: f16::from_f32(0.07),
            _reserved: 0,
        };
        entries[2].doc_id = 999;

        let view = OsiClusterView::from_cluster_bytes(&buf, vec_count, padded_dim).unwrap();
        assert_eq!(
            view.vector_code(0),
            Some(&[0xAA55_0000_FFFF_1111, 0x1234_5678_9ABC_DEF0][..])
        );
        assert_eq!(view.vector_code(2), Some(&[0xFFFF_FFFF_FFFF_FFFF, 0][..]));
        assert_eq!(view.entries[0].doc_id, 100);
        assert_eq!(view.entries[0].minipage_id, 7);
        assert_eq!(view.entries[2].doc_id, 999);
    }

    #[test]
    fn tombstone_footer_roundtrip_and_checksum() {
        let mut bm = RoaringBitmap::new();
        bm.insert(0);
        bm.insert(42);
        bm.insert(1_000_000);

        let mut encoded = Vec::new();
        serialize_tombstone_footer(&bm, &mut encoded).unwrap();

        let (decoded, consumed) = deserialize_tombstone_footer(&encoded).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, bm);

        // Corrupt a byte inside the serialized RoaringBitmap payload
        encoded[10] ^= 0xFF;
        let err = deserialize_tombstone_footer(&encoded).unwrap_err();
        assert!(matches!(err, OspreyError::ChecksumMismatch { .. }));
    }
}
