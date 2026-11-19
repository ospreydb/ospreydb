//! Locality-ordered 16 KB minipage builder, optional Zstd dictionary trainer,
//! and parallel `.osv` slab writer.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::mem::size_of;
use std::path::Path;

use bytemuck::{cast_slice, try_cast_slice, Pod, PodCastError, Zeroable};
use half::f16;
use memmap2::Mmap;
use osprey_core::{
    cast_minipage_directory,
    format::osv::{MINIPAGE_FLAG_OVERSIZED, OSV_FLAG_HAS_ZSTD_DICT, OSV_FLAG_ZSTD_ENABLED},
    AlignedBuffer, MinipageDirectoryEntry, OspreyError, OspreyResult, OsvHeader,
};
use rayon::prelude::*;
use xxhash_rust::xxh64::xxh64;
use zstd::dict::DecoderDictionary;

use crate::tsp_locality::ClusterLocalityPermutation;

/// Default target uncompressed byte size per `.osv` minipage (16 KB).
pub const DEFAULT_MINIPAGE_TARGET_BYTES: usize = 16 * 1024;

/// Size in bytes of the internal uncompressed minipage header (`vector_count`, `orig_dim`, `cluster_id`).
pub const MINIPAGE_INTERNAL_HEADER_BYTES: usize = 8;

/// Minimum number of sample minipages required before attempting Zstd dictionary training.
pub const MIN_DICT_TRAINING_PAGES: usize = 8;

/// Minimum total sample byte volume required before attempting Zstd dictionary training (32 KB).
pub const MIN_DICT_TRAINING_BYTES: usize = 32 * 1024;

/// Default maximum size of a trained Zstd dictionary (32 KB).
pub const DEFAULT_ZSTD_DICT_MAX_BYTES: usize = 32 * 1024;

/// Input vector + metadata record to pack into a `.osv` slab.
#[derive(Clone, Copy, Debug)]
pub struct OsvVectorRecord<'a> {
    /// Global logical document identifier.
    pub doc_id: u64,
    /// Cluster identifier (logical if passed with a [`ClusterLocalityPermutation`], or physical otherwise).
    pub cluster_id: u32,
    /// Exact unpadded embedding vector in `half::f16` (length must equal `orig_dim`).
    pub vector_f16: &'a [f16],
    /// Raw UTF-8 JSON/metadata payload bytes for this document.
    pub payload: &'a [u8],
}

/// Physical location of a stored vector inside a `.osv` slab.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OsvVectorLocation {
    /// Zero-based index of the minipage in the `.osv` directory.
    pub minipage_id: u32,
    /// Zero-based slot index inside the uncompressed minipage.
    pub slot_idx: u16,
    /// TSP-reordered physical cluster ID owning the minipage.
    pub physical_cluster_id: u32,
}

/// Built uncompressed minipage ready for Zstd compression and directory indexing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UncompressedMinipage {
    /// TSP-reordered physical cluster ID owning all vectors in this minipage.
    pub cluster_id: u32,
    /// Number of vector slots packed into this minipage.
    pub vector_count: u16,
    /// Original unpadded embedding dimensionality.
    pub orig_dim: u16,
    /// Minimum `doc_id` in this minipage.
    pub min_doc_id: u64,
    /// Maximum `doc_id` in this minipage.
    pub max_doc_id: u64,
    /// Directory flags (`0` or `MINIPAGE_FLAG_OVERSIZED`).
    pub flags: u16,
    /// Document IDs in slot order (`0..vector_count`).
    pub doc_ids: Vec<u64>,
    /// Deterministic Little-Endian uncompressed minipage byte payload.
    pub bytes: Vec<u8>,
}

/// Zero-copy/borrowed view over an uncompressed `.osv` minipage buffer.
#[derive(Clone, Copy, Debug)]
pub struct MinipageView<'a> {
    /// Number of vectors in the minipage.
    pub vector_count: u16,
    /// Original vector dimensionality (`orig_dim`).
    pub orig_dim: u16,
    /// TSP-reordered physical cluster ID.
    pub cluster_id: u32,
    /// Raw Little-Endian `doc_id` bytes (`vector_count * 8` bytes).
    doc_ids_le: &'a [u8],
    /// Raw Little-Endian `half::f16` vector bytes (`vector_count * orig_dim * 2` bytes).
    vectors_le: &'a [u8],
    /// Raw Little-Endian payload offsets (`(vector_count + 1) * 4` bytes).
    payload_offsets_le: &'a [u8],
    /// Concatenated payload bytes referenced by `payload_offsets_le`.
    payloads: &'a [u8],
}

impl<'a> MinipageView<'a> {
    /// Returns the `doc_id` stored at `slot_idx`.
    pub fn doc_id(&self, slot_idx: usize) -> OspreyResult<u64> {
        if slot_idx >= usize::from(self.vector_count) {
            return Err(OspreyError::InvalidDimension { dim: slot_idx });
        }
        let start = slot_idx * 8;
        let bytes: [u8; 8] = self.doc_ids_le[start..start + 8]
            .try_into()
            .map_err(|_| OspreyError::CorruptedFooter)?;
        Ok(u64::from_le_bytes(bytes))
    }

    /// Reads the exact `half::f16` vector at `slot_idx` into `out` (`out.len() == orig_dim`).
    pub fn read_vector_f16(&self, slot_idx: usize, out: &mut [f16]) -> OspreyResult<()> {
        if slot_idx >= usize::from(self.vector_count) {
            return Err(OspreyError::InvalidDimension { dim: slot_idx });
        }
        let dim = usize::from(self.orig_dim);
        if out.len() != dim {
            return Err(OspreyError::InvalidDimension { dim: out.len() });
        }
        let start = slot_idx * dim * 2;
        let slice = &self.vectors_le[start..start + dim * 2];
        let (chunks, _) = slice.as_chunks::<2>();
        for (dst, &chunk) in out.iter_mut().zip(chunks) {
            *dst = f16::from_le_bytes(chunk);
        }
        Ok(())
    }

    /// Returns a zero-copy `&[half::f16]` slice for `slot_idx` when the buffer is 2-byte aligned.
    #[inline]
    pub fn vector_f16_slice(&self, slot_idx: usize) -> Option<&'a [f16]> {
        if slot_idx >= usize::from(self.vector_count) {
            return None;
        }
        let dim = usize::from(self.orig_dim);
        let start = slot_idx.checked_mul(dim)?.checked_mul(2)?;
        let end = start.checked_add(dim.checked_mul(2)?)?;
        let slice = self.vectors_le.get(start..end)?;
        try_cast_slice(slice).ok()
    }

    /// Computes the exact `f32 x f16` inner product for `slot_idx` using SIMD F16C/AVX-512 when aligned.
    #[inline]
    pub fn compute_exact_f16_dot(&self, slot_idx: usize, query_f32: &[f32]) -> OspreyResult<f32> {
        if slot_idx >= usize::from(self.vector_count) {
            return Err(OspreyError::InvalidDimension { dim: slot_idx });
        }
        let dim = usize::from(self.orig_dim);
        if query_f32.len() != dim {
            return Err(OspreyError::InvalidDimension {
                dim: query_f32.len(),
            });
        }
        if let Some(vec_f16) = self.vector_f16_slice(slot_idx) {
            return Ok(osprey_simd::dot_f32_f16(query_f32, vec_f16));
        }
        let start = slot_idx * dim * 2;
        let slice = &self.vectors_le[start..start + dim * 2];
        let (chunks, _) = slice.as_chunks::<2>();
        let mut acc0 = 0.0_f32;
        let mut acc1 = 0.0_f32;
        let mut acc2 = 0.0_f32;
        let mut acc3 = 0.0_f32;
        let mut i = 0usize;
        while i + 4 <= dim {
            acc0 += f16::from_le_bytes(chunks[i]).to_f32() * query_f32[i];
            acc1 += f16::from_le_bytes(chunks[i + 1]).to_f32() * query_f32[i + 1];
            acc2 += f16::from_le_bytes(chunks[i + 2]).to_f32() * query_f32[i + 2];
            acc3 += f16::from_le_bytes(chunks[i + 3]).to_f32() * query_f32[i + 3];
            i += 4;
        }
        let mut sum = (acc0 + acc1) + (acc2 + acc3);
        while i < dim {
            sum += f16::from_le_bytes(chunks[i]).to_f32() * query_f32[i];
            i += 1;
        }
        Ok(sum)
    }

    /// Returns the payload byte slice for `slot_idx` without heap allocation.
    pub fn payload(&self, slot_idx: usize) -> OspreyResult<&'a [u8]> {
        if slot_idx >= usize::from(self.vector_count) {
            return Err(OspreyError::InvalidDimension { dim: slot_idx });
        }
        let o0 = read_u32_le(self.payload_offsets_le, slot_idx * 4)?;
        let o1 = read_u32_le(self.payload_offsets_le, (slot_idx + 1) * 4)?;
        let start = usize::try_from(o0).unwrap_or(usize::MAX);
        let end = usize::try_from(o1).unwrap_or(usize::MAX);
        if start > end || end > self.payloads.len() {
            return Err(OspreyError::CorruptedFooter);
        }
        Ok(&self.payloads[start..end])
    }
}

/// Parses and bounds-checks an uncompressed `.osv` minipage byte buffer.
pub fn parse_minipage(bytes: &[u8]) -> OspreyResult<MinipageView<'_>> {
    if bytes.len() < 12 {
        return Err(OspreyError::BufferTooSmall {
            needed: 12,
            actual: bytes.len(),
        });
    }

    let vector_count = read_u16_le(bytes, 0)?;
    let orig_dim = read_u16_le(bytes, 2)?;
    let cluster_id = read_u32_le(bytes, 4)?;
    if orig_dim == 0 {
        return Err(OspreyError::InvalidDimension { dim: 0 });
    }

    let count = usize::from(vector_count);
    let dim = usize::from(orig_dim);

    let doc_ids_len = count
        .checked_mul(8)
        .ok_or(OspreyError::InvalidDimension { dim: count })?;
    let vectors_len = count
        .checked_mul(dim)
        .and_then(|v| v.checked_mul(2))
        .ok_or(OspreyError::InvalidDimension { dim: count })?;
    let offsets_len = (count + 1)
        .checked_mul(4)
        .ok_or(OspreyError::InvalidDimension { dim: count })?;

    let doc_ids_start = MINIPAGE_INTERNAL_HEADER_BYTES;
    let vectors_start = doc_ids_start
        .checked_add(doc_ids_len)
        .ok_or(OspreyError::InvalidDimension { dim: count })?;
    let offsets_start = vectors_start
        .checked_add(vectors_len)
        .ok_or(OspreyError::InvalidDimension { dim: count })?;
    let payloads_start = offsets_start
        .checked_add(offsets_len)
        .ok_or(OspreyError::InvalidDimension { dim: count })?;

    if bytes.len() < payloads_start {
        return Err(OspreyError::BufferTooSmall {
            needed: payloads_start,
            actual: bytes.len(),
        });
    }

    let doc_ids_le = &bytes[doc_ids_start..vectors_start];
    let vectors_le = &bytes[vectors_start..offsets_start];
    let payload_offsets_le = &bytes[offsets_start..payloads_start];
    let payloads = &bytes[payloads_start..];

    let first_off = read_u32_le(payload_offsets_le, 0)?;
    let last_off = read_u32_le(payload_offsets_le, count * 4)?;
    if first_off != 0 || usize::try_from(last_off).unwrap_or(usize::MAX) != payloads.len() {
        return Err(OspreyError::CorruptedFooter);
    }

    Ok(MinipageView {
        vector_count,
        orig_dim,
        cluster_id,
        doc_ids_le,
        vectors_le,
        payload_offsets_le,
        payloads,
    })
}

/// Configuration parameters for [`OsvWriter`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OsvWriterConfig {
    /// Target uncompressed size per minipage in bytes (default: `16 * 1024`).
    pub minipage_target_bytes: usize,
    /// Zstd compression level (default: `3`).
    pub zstd_level: i32,
    /// Whether to train and embed a Zstd dictionary when enough sample pages exist.
    pub enable_zstd_dict: bool,
    /// Maximum Zstd dictionary byte size when training (default: `32 * 1024`).
    pub max_dict_bytes: usize,
}

impl OsvWriterConfig {
    /// Uncompressed `.osv` configuration (`zstd_level = 0`, no dictionary) for zero-copy local NVMe `mmap`.
    #[inline]
    pub const fn uncompressed() -> Self {
        Self {
            minipage_target_bytes: DEFAULT_MINIPAGE_TARGET_BYTES,
            zstd_level: 0,
            enable_zstd_dict: false,
            max_dict_bytes: 0,
        }
    }
}

impl Default for OsvWriterConfig {
    #[inline]
    fn default() -> Self {
        Self {
            minipage_target_bytes: DEFAULT_MINIPAGE_TARGET_BYTES,
            zstd_level: 3,
            enable_zstd_dict: true,
            max_dict_bytes: DEFAULT_ZSTD_DICT_MAX_BYTES,
        }
    }
}

/// Output summary and exact `(doc_id -> (minipage_id, slot_idx))` map from [`OsvWriter::write_slab`].
#[derive(Clone, Debug)]
pub struct OsvWriteReport {
    /// Sealed 64-byte `.osv` file header.
    pub header: OsvHeader,
    /// Directory entries for all minipages (`0..num_minipages`).
    pub directory: Vec<MinipageDirectoryEntry>,
    /// Mapping from `doc_id` to exact physical `(minipage_id, slot_idx, physical_cluster_id)`.
    pub locations_by_doc_id: HashMap<u64, OsvVectorLocation>,
    /// Total uncompressed payload size across all minipages.
    pub total_uncompressed_bytes: u64,
    /// Total compressed frame size across all minipages.
    pub total_compressed_bytes: u64,
    /// Size in bytes of the embedded Zstd dictionary (`0` if none).
    pub zstd_dict_bytes: usize,
}

impl OsvWriteReport {
    /// Returns the exact `(minipage_id, slot_idx)` location for `doc_id` to embed in `.osi`.
    #[inline]
    pub fn location_for_doc(&self, doc_id: u64) -> Option<(u32, u16)> {
        self.locations_by_doc_id
            .get(&doc_id)
            .map(|loc| (loc.minipage_id, loc.slot_idx))
    }
}

/// Parallel `.osv` slab writer implementing locality-ordered 16 KB minipage packing,
/// optional Zstd dictionary training, and `xxHash64` checksum sealing.
#[derive(Clone, Debug, Default)]
pub struct OsvWriter {
    /// Writer configuration.
    pub config: OsvWriterConfig,
}

impl OsvWriter {
    /// Creates an [`OsvWriter`] with the specified configuration.
    #[inline]
    pub const fn new(config: OsvWriterConfig) -> Self {
        Self { config }
    }

    /// Packs, compresses in parallel via `rayon`, and writes a complete `.osv` slab to `writer`.
    ///
    /// If `permutation` is provided, each record's logical `cluster_id` is remapped to its
    /// TSP physical cluster ID before grouping into minipages. If `permutation` is `None`,
    /// `record.cluster_id` is used directly as the physical cluster ID.
    pub fn write_slab<W: Write>(
        &self,
        writer: &mut W,
        records: &[OsvVectorRecord<'_>],
        orig_dim: usize,
        permutation: Option<&ClusterLocalityPermutation>,
    ) -> OspreyResult<OsvWriteReport> {
        let zstd_level_i8 = i8::try_from(self.config.zstd_level)
            .map_err(|_| OspreyError::InvalidDimension { dim: 0 })?;
        let orig_dim_u16 =
            u16::try_from(orig_dim).map_err(|_| OspreyError::InvalidDimension { dim: orig_dim })?;
        if orig_dim_u16 == 0 {
            return Err(OspreyError::InvalidDimension { dim: 0 });
        }

        let pages = build_minipages(
            records,
            orig_dim,
            self.config.minipage_target_bytes,
            permutation,
        )?;

        let zstd_enabled = self.config.zstd_level != 0;
        let zstd_dict =
            if zstd_enabled && self.config.enable_zstd_dict && self.config.max_dict_bytes > 0 {
                let sample_pages: Vec<Vec<u8>> = pages.iter().map(|p| p.bytes.clone()).collect();
                train_payload_zstd_dictionary(&sample_pages, self.config.max_dict_bytes)?
            } else {
                Vec::new()
            };

        let compressed_pages = if zstd_enabled {
            compress_minipages_parallel(&pages, self.config.zstd_level, &zstd_dict)?
        } else {
            Vec::new()
        };

        let mut flags = if zstd_enabled {
            OSV_FLAG_ZSTD_ENABLED
        } else {
            0
        };
        let (zstd_dict_offset, zstd_dict_len, slabs_offset) = if zstd_dict.is_empty() {
            (0_u64, 0_u64, 64_u64)
        } else {
            flags |= OSV_FLAG_HAS_ZSTD_DICT;
            let dict_offset = 64_u64;
            let dict_len =
                u64::try_from(zstd_dict.len()).map_err(|_| OspreyError::InvalidDimension {
                    dim: zstd_dict.len(),
                })?;
            let dict_end =
                dict_offset
                    .checked_add(dict_len)
                    .ok_or(OspreyError::InvalidDimension {
                        dim: zstd_dict.len(),
                    })?;
            let after_dict = align_up_64_u64(dict_end)?;
            (dict_offset, dict_len, after_dict)
        };

        let mut cursor = slabs_offset;
        let page_count = if zstd_enabled {
            compressed_pages.len()
        } else {
            pages.len()
        };
        let mut directory = Vec::with_capacity(page_count);
        let mut total_uncompressed_bytes = 0_u64;
        let mut total_compressed_bytes = 0_u64;

        if zstd_enabled {
            for (page_idx, page) in compressed_pages.iter().enumerate() {
                let compressed_size = u32::try_from(page.compressed_bytes.len()).map_err(|_| {
                    OspreyError::InvalidDimension {
                        dim: page.compressed_bytes.len(),
                    }
                })?;
                let uncompressed_size = u32::try_from(page.uncompressed_size).map_err(|_| {
                    OspreyError::InvalidDimension {
                        dim: page.uncompressed_size,
                    }
                })?;

                directory.push(MinipageDirectoryEntry {
                    byte_offset: cursor,
                    min_doc_id: page.min_doc_id,
                    max_doc_id: page.max_doc_id,
                    page_xxh64: page.page_xxh64,
                    compressed_size,
                    uncompressed_size,
                    cluster_id: page.cluster_id,
                    vector_count: page.vector_count,
                    flags: page.flags,
                });

                let comp_u64 = u64::from(compressed_size);
                let uncomp_u64 = u64::from(uncompressed_size);
                total_compressed_bytes = total_compressed_bytes
                    .checked_add(comp_u64)
                    .ok_or(OspreyError::InvalidDimension { dim: page_idx })?;
                total_uncompressed_bytes = total_uncompressed_bytes
                    .checked_add(uncomp_u64)
                    .ok_or(OspreyError::InvalidDimension { dim: page_idx })?;
                cursor = cursor
                    .checked_add(comp_u64)
                    .ok_or(OspreyError::InvalidDimension { dim: page_idx })?;
            }
        } else {
            for (page_idx, page) in pages.iter().enumerate() {
                // Keep every uncompressed minipage 8-byte aligned so `&[u64]`/`&[f16]` can cast zero-copy on mmap.
                cursor = align_up_8_u64(cursor)?;
                let len_u32 =
                    u32::try_from(page.bytes.len()).map_err(|_| OspreyError::InvalidDimension {
                        dim: page.bytes.len(),
                    })?;
                let page_xxh64 = xxh64(&page.bytes, 0);
                directory.push(MinipageDirectoryEntry {
                    byte_offset: cursor,
                    min_doc_id: page.min_doc_id,
                    max_doc_id: page.max_doc_id,
                    page_xxh64,
                    compressed_size: len_u32,
                    uncompressed_size: len_u32,
                    cluster_id: page.cluster_id,
                    vector_count: page.vector_count,
                    flags: page.flags,
                });

                let len_u64 = u64::from(len_u32);
                total_compressed_bytes = total_compressed_bytes
                    .checked_add(len_u64)
                    .ok_or(OspreyError::InvalidDimension { dim: page_idx })?;
                total_uncompressed_bytes = total_uncompressed_bytes
                    .checked_add(len_u64)
                    .ok_or(OspreyError::InvalidDimension { dim: page_idx })?;
                cursor = cursor
                    .checked_add(len_u64)
                    .ok_or(OspreyError::InvalidDimension { dim: page_idx })?;
            }
        }

        let directory_offset = align_up_64_u64(cursor)?;
        let num_minipages =
            u32::try_from(directory.len()).map_err(|_| OspreyError::InvalidDimension {
                dim: directory.len(),
            })?;
        let total_vectors = u64::try_from(records.len())
            .map_err(|_| OspreyError::InvalidDimension { dim: records.len() })?;

        let header = OsvHeader::new_sealed(
            zstd_level_i8,
            orig_dim_u16,
            flags,
            num_minipages,
            total_vectors,
            zstd_dict_offset,
            zstd_dict_len,
            directory_offset,
        )?;

        // 1. Write 64-byte sealed OsvHeader.
        writer.write_all(bytemuck::bytes_of(&header))?;

        // 2. Write optional Zstd dictionary + 64B zero padding.
        if !zstd_dict.is_empty() {
            writer.write_all(&zstd_dict)?;
            let dict_end = 64_u64 + zstd_dict_len;
            write_zero_padding(writer, slabs_offset - dict_end)?;
        }

        // 3. Write minipage frames + 64B zero padding before directory.
        if zstd_enabled {
            for page in &compressed_pages {
                writer.write_all(&page.compressed_bytes)?;
            }
        } else {
            let mut written_cursor = slabs_offset;
            for page in &pages {
                let aligned = align_up_8_u64(written_cursor)?;
                if aligned > written_cursor {
                    write_zero_padding(writer, aligned - written_cursor)?;
                }
                writer.write_all(&page.bytes)?;
                written_cursor = aligned + u64::try_from(page.bytes.len()).unwrap_or(0);
            }
        }
        write_zero_padding(writer, directory_offset - cursor)?;

        // 4. Write contiguous MinipageDirectoryEntry array (num_minipages * 48 bytes).
        writer.write_all(cast_slice(&directory))?;

        let mut locations_by_doc_id = HashMap::with_capacity(records.len());
        for (page_idx, page) in pages.iter().enumerate() {
            let minipage_id = u32::try_from(page_idx)
                .map_err(|_| OspreyError::InvalidDimension { dim: page_idx })?;
            for (slot_i, &doc_id) in page.doc_ids.iter().enumerate() {
                let slot_idx = u16::try_from(slot_i)
                    .map_err(|_| OspreyError::InvalidDimension { dim: slot_i })?;
                locations_by_doc_id.insert(
                    doc_id,
                    OsvVectorLocation {
                        minipage_id,
                        slot_idx,
                        physical_cluster_id: page.cluster_id,
                    },
                );
            }
        }

        Ok(OsvWriteReport {
            header,
            directory,
            locations_by_doc_id,
            total_uncompressed_bytes,
            total_compressed_bytes,
            zstd_dict_bytes: zstd_dict.len(),
        })
    }

    /// Convenience wrapper writing the `.osv` slab into an owned `Vec<u8>`.
    pub fn write_slab_to_bytes(
        &self,
        records: &[OsvVectorRecord<'_>],
        orig_dim: usize,
        permutation: Option<&ClusterLocalityPermutation>,
    ) -> OspreyResult<(Vec<u8>, OsvWriteReport)> {
        let mut buf = Vec::new();
        let report = self.write_slab(&mut buf, records, orig_dim, permutation)?;
        Ok((buf, report))
    }
}

/// Incremental single-file `.osv` slab writer that streams minipage chunks without holding the
/// full `f16` corpus in memory (`Task: streaming import under bounded RSS`).
///
/// The 64-byte `OsvHeader` is written as a placeholder first and rewritten in-place by
/// [`StreamingOsvWriter::finish`] once the final directory offset, minipage count, and vector
/// count are known. Only the (small) `MinipageDirectoryEntry` table is retained across chunks.
pub struct StreamingOsvWriter<W: Write + Seek> {
    writer: W,
    config: OsvWriterConfig,
    orig_dim: u16,
    zstd_dict: Vec<u8>,
    directory: Vec<MinipageDirectoryEntry>,
    cursor: u64,
    total_vectors: u64,
    total_uncompressed_bytes: u64,
    total_compressed_bytes: u64,
}

impl<W: Write + Seek> StreamingOsvWriter<W> {
    /// Creates a streaming writer and writes the placeholder header (+ optional Zstd dictionary).
    pub fn new(
        mut writer: W,
        config: OsvWriterConfig,
        orig_dim: usize,
        zstd_dict: Vec<u8>,
    ) -> OspreyResult<Self> {
        let orig_dim_u16 =
            u16::try_from(orig_dim).map_err(|_| OspreyError::InvalidDimension { dim: orig_dim })?;
        if orig_dim_u16 == 0 {
            return Err(OspreyError::InvalidDimension { dim: 0 });
        }

        // Placeholder 64-byte header, rewritten by `finish`.
        writer.write_all(&[0_u8; 64])?;

        let cursor = if zstd_dict.is_empty() {
            64_u64
        } else {
            let dict_len =
                u64::try_from(zstd_dict.len()).map_err(|_| OspreyError::InvalidDimension {
                    dim: zstd_dict.len(),
                })?;
            writer.write_all(&zstd_dict)?;
            let after_dict = 64_u64
                .checked_add(dict_len)
                .ok_or(OspreyError::AllocationFailed {
                    size: usize::MAX,
                    align: 64,
                })?;
            let slabs_offset = align_up_64_u64(after_dict)?;
            write_zero_padding(&mut writer, slabs_offset - after_dict)?;
            slabs_offset
        };

        Ok(Self {
            writer,
            config,
            orig_dim: orig_dim_u16,
            zstd_dict,
            directory: Vec::new(),
            cursor,
            total_vectors: 0,
            total_uncompressed_bytes: 0,
            total_compressed_bytes: 0,
        })
    }

    /// Builds, compresses, and appends the minipages for `records`, returning each record's
    /// global `(minipage_id, slot_idx, physical_cluster_id)` location.
    pub fn write_chunk(
        &mut self,
        records: &[OsvVectorRecord<'_>],
        permutation: Option<&ClusterLocalityPermutation>,
    ) -> OspreyResult<Vec<(u64, OsvVectorLocation)>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }

        let prev_len = self.directory.len();
        let pages = build_minipages(
            records,
            usize::from(self.orig_dim),
            self.config.minipage_target_bytes,
            permutation,
        )?;
        let zstd_enabled = self.config.zstd_level != 0;

        if zstd_enabled {
            let compressed_pages =
                compress_minipages_parallel(&pages, self.config.zstd_level, &self.zstd_dict)?;

            for page in compressed_pages {
                let compressed_size = u32::try_from(page.compressed_bytes.len()).map_err(|_| {
                    OspreyError::InvalidDimension {
                        dim: page.compressed_bytes.len(),
                    }
                })?;
                let uncompressed_size = u32::try_from(page.uncompressed_size).map_err(|_| {
                    OspreyError::InvalidDimension {
                        dim: page.uncompressed_size,
                    }
                })?;

                let minipage_id = u32::try_from(self.directory.len()).map_err(|_| {
                    OspreyError::InvalidDimension {
                        dim: self.directory.len(),
                    }
                })?;

                self.directory.push(MinipageDirectoryEntry {
                    byte_offset: self.cursor,
                    min_doc_id: page.min_doc_id,
                    max_doc_id: page.max_doc_id,
                    page_xxh64: page.page_xxh64,
                    compressed_size,
                    uncompressed_size,
                    cluster_id: page.cluster_id,
                    vector_count: page.vector_count,
                    flags: page.flags,
                });

                self.writer.write_all(&page.compressed_bytes)?;

                let dim_err = usize::try_from(minipage_id).unwrap_or(usize::MAX);
                self.total_compressed_bytes = self
                    .total_compressed_bytes
                    .checked_add(u64::from(compressed_size))
                    .ok_or(OspreyError::InvalidDimension { dim: dim_err })?;
                self.total_uncompressed_bytes = self
                    .total_uncompressed_bytes
                    .checked_add(u64::from(uncompressed_size))
                    .ok_or(OspreyError::InvalidDimension { dim: dim_err })?;
                self.cursor = self
                    .cursor
                    .checked_add(u64::from(compressed_size))
                    .ok_or(OspreyError::InvalidDimension { dim: dim_err })?;
            }
        } else {
            for page in &pages {
                let aligned = align_up_8_u64(self.cursor)?;
                if aligned > self.cursor {
                    write_zero_padding(&mut self.writer, aligned - self.cursor)?;
                    self.cursor = aligned;
                }
                let len_u32 =
                    u32::try_from(page.bytes.len()).map_err(|_| OspreyError::InvalidDimension {
                        dim: page.bytes.len(),
                    })?;
                let minipage_id = u32::try_from(self.directory.len()).map_err(|_| {
                    OspreyError::InvalidDimension {
                        dim: self.directory.len(),
                    }
                })?;
                let page_xxh64 = xxh64(&page.bytes, 0);

                self.directory.push(MinipageDirectoryEntry {
                    byte_offset: self.cursor,
                    min_doc_id: page.min_doc_id,
                    max_doc_id: page.max_doc_id,
                    page_xxh64,
                    compressed_size: len_u32,
                    uncompressed_size: len_u32,
                    cluster_id: page.cluster_id,
                    vector_count: page.vector_count,
                    flags: page.flags,
                });

                self.writer.write_all(&page.bytes)?;

                let dim_err = usize::try_from(minipage_id).unwrap_or(usize::MAX);
                let len_u64 = u64::from(len_u32);
                self.total_compressed_bytes = self
                    .total_compressed_bytes
                    .checked_add(len_u64)
                    .ok_or(OspreyError::InvalidDimension { dim: dim_err })?;
                self.total_uncompressed_bytes = self
                    .total_uncompressed_bytes
                    .checked_add(len_u64)
                    .ok_or(OspreyError::InvalidDimension { dim: dim_err })?;
                self.cursor = self
                    .cursor
                    .checked_add(len_u64)
                    .ok_or(OspreyError::InvalidDimension { dim: dim_err })?;
            }
        }

        let mut locations = Vec::with_capacity(records.len());

        // Recover exact per-doc locations from the uncompressed pages (slot order == doc order).
        // Pages are appended in the same order `build_minipages` returns them, so the global
        // minipage id is `prev_len + page_idx`.
        for (page_idx, page) in pages.iter().enumerate() {
            let global_id = u32::try_from(
                prev_len
                    .checked_add(page_idx)
                    .ok_or(OspreyError::InvalidDimension { dim: usize::MAX })?,
            )
            .map_err(|_| OspreyError::InvalidDimension { dim: page_idx })?;
            for (slot_i, &doc_id) in page.doc_ids.iter().enumerate() {
                let slot_idx = u16::try_from(slot_i)
                    .map_err(|_| OspreyError::InvalidDimension { dim: slot_i })?;
                locations.push((
                    doc_id,
                    OsvVectorLocation {
                        minipage_id: global_id,
                        slot_idx,
                        physical_cluster_id: page.cluster_id,
                    },
                ));
            }
        }

        self.total_vectors = self
            .total_vectors
            .checked_add(
                u64::try_from(records.len())
                    .map_err(|_| OspreyError::InvalidDimension { dim: records.len() })?,
            )
            .ok_or(OspreyError::InvalidDimension { dim: records.len() })?;

        Ok(locations)
    }

    /// Seals the slab: writes the directory, rewrites the header in-place, and returns the report.
    pub fn finish(mut self) -> OspreyResult<OsvWriteReport> {
        let directory_offset = align_up_64_u64(self.cursor)?;
        write_zero_padding(&mut self.writer, directory_offset - self.cursor)?;
        self.writer.write_all(cast_slice(&self.directory))?;

        let zstd_level_i8 = i8::try_from(self.config.zstd_level)
            .map_err(|_| OspreyError::InvalidDimension { dim: 0 })?;
        let mut flags = if self.config.zstd_level != 0 {
            OSV_FLAG_ZSTD_ENABLED
        } else {
            0
        };
        let (zstd_dict_offset, zstd_dict_len) = if self.zstd_dict.is_empty() {
            (0_u64, 0_u64)
        } else {
            flags |= OSV_FLAG_HAS_ZSTD_DICT;
            (
                64_u64,
                u64::try_from(self.zstd_dict.len()).map_err(|_| OspreyError::InvalidDimension {
                    dim: self.zstd_dict.len(),
                })?,
            )
        };
        let num_minipages =
            u32::try_from(self.directory.len()).map_err(|_| OspreyError::InvalidDimension {
                dim: self.directory.len(),
            })?;

        let header = OsvHeader::new_sealed(
            zstd_level_i8,
            self.orig_dim,
            flags,
            num_minipages,
            self.total_vectors,
            zstd_dict_offset,
            zstd_dict_len,
            directory_offset,
        )?;

        self.writer.seek(SeekFrom::Start(0))?;
        self.writer.write_all(bytemuck::bytes_of(&header))?;
        self.writer.flush()?;

        Ok(OsvWriteReport {
            header,
            directory: self.directory,
            locations_by_doc_id: HashMap::new(),
            total_uncompressed_bytes: self.total_uncompressed_bytes,
            total_compressed_bytes: self.total_compressed_bytes,
            zstd_dict_bytes: self.zstd_dict.len(),
        })
    }
}

/// Groups records into TSP-ordered 16 KB uncompressed minipages.
///
/// Records are deterministically sorted by `(physical_cluster_id, doc_id)` and packed into
/// minipages belonging to the same `physical_cluster_id` up to `target_bytes` (always placing
/// at least 1 vector per minipage).
pub fn build_minipages(
    records: &[OsvVectorRecord<'_>],
    orig_dim: usize,
    target_bytes: usize,
    permutation: Option<&ClusterLocalityPermutation>,
) -> OspreyResult<Vec<UncompressedMinipage>> {
    let orig_dim_u16 =
        u16::try_from(orig_dim).map_err(|_| OspreyError::InvalidDimension { dim: orig_dim })?;
    if orig_dim_u16 == 0 {
        return Err(OspreyError::InvalidDimension { dim: 0 });
    }
    if target_bytes == 0 {
        return Err(OspreyError::InvalidDimension { dim: 0 });
    }
    if records.is_empty() {
        return Ok(Vec::new());
    }

    let mut ordered: Vec<(u32, u64, usize)> = Vec::with_capacity(records.len());
    for (rec_idx, record) in records.iter().enumerate() {
        if record.vector_f16.len() != orig_dim {
            return Err(OspreyError::InvalidDimension {
                dim: record.vector_f16.len(),
            });
        }
        for (coord_idx, &val) in record.vector_f16.iter().enumerate() {
            if !val.is_finite() {
                return Err(OspreyError::NonFiniteEmbedding {
                    index: rec_idx * orig_dim + coord_idx,
                });
            }
        }
        let physical_cluster_id = match permutation {
            Some(perm) => perm.to_physical(record.cluster_id)?,
            None => record.cluster_id,
        };
        ordered.push((physical_cluster_id, record.doc_id, rec_idx));
    }

    // Deterministic order: physical TSP cluster order first, then doc_id ascending.
    ordered.sort_by_key(|&(cluster_id, doc_id, _)| (cluster_id, doc_id));

    let fixed_per_vec = 8_usize + orig_dim * 2 + 4;
    let empty_page_overhead = MINIPAGE_INTERNAL_HEADER_BYTES + 4; // header(8) + offsets[0](4)

    let mut pages = Vec::new();
    let mut start = 0usize;

    while start < ordered.len() {
        let cluster_id = ordered[start].0;
        let mut end = start;
        let mut page_bytes = empty_page_overhead;
        let mut payload_bytes = 0usize;

        while end < ordered.len() {
            let (rec_cluster, _, rec_idx) = ordered[end];
            if rec_cluster != cluster_id {
                break;
            }
            let payload_len = records[rec_idx].payload.len();
            let entry_cost = fixed_per_vec
                .checked_add(payload_len)
                .ok_or(OspreyError::InvalidDimension { dim: payload_len })?;
            let next_page_bytes = page_bytes
                .checked_add(entry_cost)
                .ok_or(OspreyError::InvalidDimension { dim: entry_cost })?;
            let next_payload_bytes = payload_bytes
                .checked_add(payload_len)
                .ok_or(OspreyError::InvalidDimension { dim: payload_len })?;

            let count = end - start;
            let fits_slot_limit = count < usize::from(u16::MAX);
            let fits_u32_payload = u32::try_from(next_payload_bytes).is_ok();
            let fits_u32_page = u32::try_from(next_page_bytes).is_ok();

            if count > 0
                && (next_page_bytes > target_bytes
                    || !fits_slot_limit
                    || !fits_u32_payload
                    || !fits_u32_page)
            {
                break;
            }
            if count == 0 && (!fits_u32_payload || !fits_u32_page) {
                return Err(OspreyError::InvalidDimension { dim: payload_len });
            }

            page_bytes = next_page_bytes;
            payload_bytes = next_payload_bytes;
            end += 1;
        }

        let group = &ordered[start..end];
        let page = serialize_minipage(cluster_id, orig_dim_u16, target_bytes, group, records)?;
        pages.push(page);
        start = end;
    }

    Ok(pages)
}

/// Trains a compact Zstd dictionary from sample minipages when volume is sufficient.
///
/// Requires at least `8` sample pages and `32 KB` total sample bytes. When there is
/// insufficient data (or `max_dict_bytes == 0`), returns `Ok(Vec::new())` so callers
/// fall back seamlessly to standard Zstd compression without a dictionary.
pub fn train_payload_zstd_dictionary(
    sample_pages: &[Vec<u8>],
    max_dict_bytes: usize,
) -> OspreyResult<Vec<u8>> {
    if max_dict_bytes == 0 || sample_pages.len() < MIN_DICT_TRAINING_PAGES {
        return Ok(Vec::new());
    }

    let total_bytes: usize = sample_pages.iter().map(Vec::len).sum();
    if total_bytes < MIN_DICT_TRAINING_BYTES {
        return Ok(Vec::new());
    }

    // ZDICT requires total sample size to be larger than the requested dictionary size.
    let target_dict_size = max_dict_bytes.min(total_bytes / 4);
    if target_dict_size < 256 {
        return Ok(Vec::new());
    }

    match zstd::dict::from_samples(sample_pages, target_dict_size) {
        Ok(dict) => Ok(dict),
        Err(_) => Ok(Vec::new()),
    }
}

/// Read-only view of an `.osv` slab stored in memory or mapped via `memmap2`.
#[derive(Clone, Copy, Debug)]
pub struct OsvSlabView<'a> {
    /// Validated 64-byte `.osv` header.
    pub header: &'a OsvHeader,
    /// Optional embedded Zstd dictionary bytes.
    pub zstd_dict: &'a [u8],
    /// Contiguous minipage directory entries.
    pub directory: &'a [MinipageDirectoryEntry],
    /// Entire `.osv` byte slice.
    bytes: &'a [u8],
}

impl<'a> OsvSlabView<'a> {
    /// Parses and validates an `.osv` slab byte slice (`header`, optional dictionary, and directory).
    pub fn from_bytes(bytes: &'a [u8]) -> OspreyResult<Self> {
        let header = OsvHeader::from_bytes(bytes)?;
        let zstd_dict = if (header.flags & OSV_FLAG_HAS_ZSTD_DICT) != 0 {
            if header.zstd_dict_len == 0 {
                return Err(OspreyError::CorruptedFooter);
            }
            let start = usize::try_from(header.zstd_dict_offset)
                .map_err(|_| OspreyError::CorruptedFooter)?;
            if start < size_of::<OsvHeader>() {
                return Err(OspreyError::CorruptedFooter);
            }
            let len =
                usize::try_from(header.zstd_dict_len).map_err(|_| OspreyError::CorruptedFooter)?;
            let end = start.checked_add(len).ok_or(OspreyError::CorruptedFooter)?;
            if end > bytes.len() {
                return Err(OspreyError::BufferTooSmall {
                    needed: end,
                    actual: bytes.len(),
                });
            }
            &bytes[start..end]
        } else {
            &[]
        };

        let dir_offset =
            usize::try_from(header.directory_offset).map_err(|_| OspreyError::CorruptedFooter)?;
        if dir_offset < size_of::<OsvHeader>() {
            return Err(OspreyError::CorruptedFooter);
        }
        let dir_bytes = bytes.get(dir_offset..).ok_or(OspreyError::BufferTooSmall {
            needed: dir_offset,
            actual: bytes.len(),
        })?;
        let directory = cast_minipage_directory(dir_bytes, header.num_minipages)?;

        Ok(Self {
            header,
            zstd_dict,
            directory,
            bytes,
        })
    }

    /// Validates `page_xxh64` before decompressing and parsing minipage `minipage_id`.
    pub fn read_minipage(&self, minipage_id: usize) -> OspreyResult<Vec<u8>> {
        let entry = self
            .directory
            .get(minipage_id)
            .ok_or(OspreyError::InvalidDimension { dim: minipage_id })?;
        let start = usize::try_from(entry.byte_offset).map_err(|_| OspreyError::CorruptedFooter)?;
        if start < size_of::<OsvHeader>() {
            return Err(OspreyError::CorruptedFooter);
        }
        let len =
            usize::try_from(entry.compressed_size).map_err(|_| OspreyError::CorruptedFooter)?;
        let end = start.checked_add(len).ok_or(OspreyError::CorruptedFooter)?;
        let compressed = self
            .bytes
            .get(start..end)
            .ok_or(OspreyError::BufferTooSmall {
                needed: end,
                actual: self.bytes.len(),
            })?;

        let computed = xxh64(compressed, 0);
        if computed != entry.page_xxh64 {
            return Err(OspreyError::ChecksumMismatch {
                expected: entry.page_xxh64,
                computed,
            });
        }

        let uncomp_len =
            usize::try_from(entry.uncompressed_size).map_err(|_| OspreyError::CorruptedFooter)?;
        const MAX_MINIPAGE_UNCOMPRESSED_BYTES: usize = 128 * 1024 * 1024;
        if uncomp_len > MAX_MINIPAGE_UNCOMPRESSED_BYTES {
            return Err(OspreyError::CorruptedFooter);
        }
        if (self.header.flags & OSV_FLAG_ZSTD_ENABLED) == 0 {
            if compressed.len() != uncomp_len {
                return Err(OspreyError::CorruptedFooter);
            }
            return Ok(compressed.to_vec());
        }
        if let Ok(Some(frame_content_size)) = zstd::zstd_safe::get_frame_content_size(compressed) {
            if frame_content_size != u64::from(entry.uncompressed_size) {
                return Err(OspreyError::CorruptedFooter);
            }
        }
        let mut decompressor = if self.zstd_dict.is_empty() {
            zstd::bulk::Decompressor::new()?
        } else {
            zstd::bulk::Decompressor::with_dictionary(self.zstd_dict)?
        };
        let mut out = vec![0_u8; uncomp_len];
        let written = decompressor.decompress_to_buffer(compressed, &mut out)?;
        if written != uncomp_len {
            return Err(OspreyError::CorruptedFooter);
        }
        Ok(out)
    }
}

/// Decoded uncompressed `.osv` minipage backed by a 64-byte-aligned buffer.
///
/// Guarantees O(1) zero-copy access to `f16` vectors (`&[half::f16]`), `doc_id`s,
/// and UTF-8 JSON payloads without extra heap allocations.
pub struct DecodedMinipage {
    buffer: AlignedBuffer<u8>,
    vector_count: u16,
    orig_dim: u16,
    cluster_id: u32,
    doc_ids_start: usize,
    vectors_start: usize,
    offsets_start: usize,
    payloads_start: usize,
}

impl std::fmt::Debug for DecodedMinipage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecodedMinipage")
            .field("vector_count", &self.vector_count)
            .field("orig_dim", &self.orig_dim)
            .field("cluster_id", &self.cluster_id)
            .field("byte_len", &self.buffer.len())
            .finish()
    }
}

impl DecodedMinipage {
    /// Validates and wraps a 64-byte-aligned uncompressed minipage buffer.
    pub fn from_aligned_buffer(buffer: AlignedBuffer<u8>) -> OspreyResult<Self> {
        let bytes = buffer.as_slice();
        if bytes.len() < 12 {
            return Err(OspreyError::BufferTooSmall {
                needed: 12,
                actual: bytes.len(),
            });
        }

        let vector_count = read_u16_le(bytes, 0)?;
        let orig_dim = read_u16_le(bytes, 2)?;
        let cluster_id = read_u32_le(bytes, 4)?;
        if orig_dim == 0 {
            return Err(OspreyError::InvalidDimension { dim: 0 });
        }

        let count = usize::from(vector_count);
        let dim = usize::from(orig_dim);

        let doc_ids_len = count
            .checked_mul(8)
            .ok_or(OspreyError::InvalidDimension { dim: count })?;
        let vectors_len = count
            .checked_mul(dim)
            .and_then(|v| v.checked_mul(2))
            .ok_or(OspreyError::InvalidDimension { dim: count })?;
        let offsets_len = (count + 1)
            .checked_mul(4)
            .ok_or(OspreyError::InvalidDimension { dim: count })?;

        let doc_ids_start = MINIPAGE_INTERNAL_HEADER_BYTES;
        let vectors_start = doc_ids_start
            .checked_add(doc_ids_len)
            .ok_or(OspreyError::InvalidDimension { dim: count })?;
        let offsets_start = vectors_start
            .checked_add(vectors_len)
            .ok_or(OspreyError::InvalidDimension { dim: count })?;
        let payloads_start = offsets_start
            .checked_add(offsets_len)
            .ok_or(OspreyError::InvalidDimension { dim: count })?;

        if bytes.len() < payloads_start {
            return Err(OspreyError::BufferTooSmall {
                needed: payloads_start,
                actual: bytes.len(),
            });
        }

        // Verify zero-copy cast invariants for `doc_ids` ([u64]) and `vectors` ([f16]).
        let _: &[u64] = try_cast_slice(&bytes[doc_ids_start..vectors_start])
            .map_err(|_| OspreyError::CorruptedFooter)?;
        let _: &[f16] = try_cast_slice(&bytes[vectors_start..offsets_start])
            .map_err(|_| OspreyError::CorruptedFooter)?;

        let payload_offsets_le = &bytes[offsets_start..payloads_start];
        let first_off = read_u32_le(payload_offsets_le, 0)?;
        let last_off = read_u32_le(payload_offsets_le, count * 4)?;
        let payloads_len = bytes.len() - payloads_start;
        if first_off != 0 || usize::try_from(last_off).unwrap_or(usize::MAX) != payloads_len {
            return Err(OspreyError::CorruptedFooter);
        }

        Ok(Self {
            buffer,
            vector_count,
            orig_dim,
            cluster_id,
            doc_ids_start,
            vectors_start,
            offsets_start,
            payloads_start,
        })
    }

    /// Returns the TSP-reordered physical `cluster_id` owning this minipage.
    #[inline]
    pub const fn cluster_id(&self) -> u32 {
        self.cluster_id
    }

    /// Returns the number of vector slots stored in this minipage.
    #[inline]
    pub const fn vector_count(&self) -> u16 {
        self.vector_count
    }

    /// Returns the exact unpadded embedding dimensionality (`orig_dim`).
    #[inline]
    pub const fn orig_dim(&self) -> u16 {
        self.orig_dim
    }

    /// Returns the uncompressed byte length of the backing 64-byte-aligned buffer.
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns the zero-copy slice of all `doc_id`s in slot order (`0..vector_count`).
    #[inline]
    pub fn doc_ids(&self) -> &[u64] {
        let bytes = &self.buffer.as_slice()[self.doc_ids_start..self.vectors_start];
        try_cast_slice(bytes).unwrap_or(&[])
    }

    /// Returns the `doc_id` at `slot_idx` in O(1).
    #[inline]
    pub fn doc_id(&self, slot_idx: u16) -> OspreyResult<u64> {
        let idx = usize::from(slot_idx);
        self.doc_ids()
            .get(idx)
            .copied()
            .ok_or(OspreyError::InvalidDimension { dim: idx })
    }

    /// Returns a zero-copy O(1) `&[half::f16]` slice for the vector at `slot_idx`.
    #[inline]
    pub fn vector_f16(&self, slot_idx: u16) -> OspreyResult<&[f16]> {
        if slot_idx >= self.vector_count {
            return Err(OspreyError::InvalidDimension {
                dim: usize::from(slot_idx),
            });
        }
        let dim = usize::from(self.orig_dim);
        let all_vecs: &[f16] =
            try_cast_slice(&self.buffer.as_slice()[self.vectors_start..self.offsets_start])
                .map_err(|_| OspreyError::CorruptedFooter)?;
        let start = usize::from(slot_idx)
            .checked_mul(dim)
            .ok_or(OspreyError::CorruptedFooter)?;
        let end = start.checked_add(dim).ok_or(OspreyError::CorruptedFooter)?;
        all_vecs.get(start..end).ok_or(OspreyError::CorruptedFooter)
    }

    /// Returns the raw payload byte slice for `slot_idx` in O(1) without allocation.
    #[inline]
    pub fn payload_bytes(&self, slot_idx: u16) -> OspreyResult<&[u8]> {
        if slot_idx >= self.vector_count {
            return Err(OspreyError::InvalidDimension {
                dim: usize::from(slot_idx),
            });
        }
        let idx = usize::from(slot_idx);
        let offsets_slice = &self.buffer.as_slice()[self.offsets_start..self.payloads_start];
        let o0 = read_u32_le(offsets_slice, idx * 4)?;
        let o1 = read_u32_le(offsets_slice, (idx + 1) * 4)?;
        let start = usize::try_from(o0).unwrap_or(usize::MAX);
        let end = usize::try_from(o1).unwrap_or(usize::MAX);
        let payloads = &self.buffer.as_slice()[self.payloads_start..];
        if start > end || end > payloads.len() {
            return Err(OspreyError::CorruptedFooter);
        }
        Ok(&payloads[start..end])
    }

    /// Returns the UTF-8 JSON/metadata payload `&str` for `slot_idx` in O(1).
    #[inline]
    pub fn payload_str(&self, slot_idx: u16) -> OspreyResult<&str> {
        let bytes = self.payload_bytes(slot_idx)?;
        std::str::from_utf8(bytes).map_err(|err| {
            OspreyError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, err))
        })
    }

    /// Computes the exact `f16` inner product against `query_f32` in `f32` accumulation,
    /// validating slot bounds and query dimensionality.
    #[inline]
    pub fn try_compute_exact_f16_dot(&self, slot_idx: u16, query_f32: &[f32]) -> OspreyResult<f32> {
        let vec_f16 = self.vector_f16(slot_idx)?;
        if query_f32.len() != vec_f16.len() {
            return Err(OspreyError::InvalidDimension {
                dim: query_f32.len(),
            });
        }
        Ok(exact_f16_f32_dot(vec_f16, query_f32))
    }

    /// Computes the exact `f16` inner product against `query_f32` for `slot_idx`.
    ///
    /// Returns `0.0` if `slot_idx` is out of bounds or `query_f32.len() != orig_dim`.
    #[inline]
    pub fn compute_exact_f16_dot(&self, slot_idx: u16, query_f32: &[f32]) -> f32 {
        self.try_compute_exact_f16_dot(slot_idx, query_f32)
            .unwrap_or(0.0_f32)
    }
}

enum SlabBacking {
    Mmap(Mmap),
    Aligned(AlignedBuffer<u8>),
}

impl SlabBacking {
    #[inline]
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Mmap(mmap) => mmap.as_ref(),
            Self::Aligned(buf) => buf.as_slice(),
        }
    }
}

/// Zero-copy local `.osv` slab reader backed by `memmap2::Mmap` or an aligned in-memory buffer.
pub struct OsvSlabReader {
    backing: SlabBacking,
    header: OsvHeader,
    dir_start: usize,
    dir_end: usize,
    decoder_dict: Option<DecoderDictionary<'static>>,
}

impl std::fmt::Debug for OsvSlabReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OsvSlabReader")
            .field("header", &self.header)
            .field("num_minipages", &self.directory().len())
            .field("has_zstd_dict", &self.decoder_dict.is_some())
            .finish()
    }
}

impl OsvSlabReader {
    /// Opens and memory-maps an `.osv` slab file from disk via `memmap2::Mmap`.
    ///
    /// Validates the 64-byte `OsvHeader`, casts the `MinipageDirectoryEntry` slice zero-copy,
    /// and pre-parses the optional Zstd dictionary if present.
    pub fn open_mmap<P: AsRef<Path>>(path: P) -> OspreyResult<Self> {
        let file = File::open(path)?;
        // SAFETY: The `.osv` slab file is immutable for the lifetime of the read-only mapping.
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_backing(SlabBacking::Mmap(mmap))
    }

    /// Opens an `.osv` slab from a 64-byte-aligned in-memory buffer.
    pub fn from_bytes(bytes: AlignedBuffer<u8>) -> OspreyResult<Self> {
        Self::from_backing(SlabBacking::Aligned(bytes))
    }

    /// Copies a byte slice into a 64-byte-aligned buffer and opens it as an [`OsvSlabReader`].
    pub fn from_slice(bytes: &[u8]) -> OspreyResult<Self> {
        Self::from_bytes(AlignedBuffer::from_slice(bytes)?)
    }

    fn from_backing(backing: SlabBacking) -> OspreyResult<Self> {
        let slice = backing.as_slice();
        let header = *OsvHeader::from_bytes(slice)?;

        let decoder_dict =
            if header.zstd_dict_len > 0 || (header.flags & OSV_FLAG_HAS_ZSTD_DICT) != 0 {
                if header.zstd_dict_len == 0 {
                    return Err(OspreyError::CorruptedFooter);
                }
                let dict_start = usize::try_from(header.zstd_dict_offset)
                    .map_err(|_| OspreyError::CorruptedFooter)?;
                if dict_start < size_of::<OsvHeader>() {
                    return Err(OspreyError::CorruptedFooter);
                }
                let dict_len = usize::try_from(header.zstd_dict_len)
                    .map_err(|_| OspreyError::CorruptedFooter)?;
                let dict_end = dict_start
                    .checked_add(dict_len)
                    .ok_or(OspreyError::CorruptedFooter)?;
                let dict_slice =
                    slice
                        .get(dict_start..dict_end)
                        .ok_or(OspreyError::BufferTooSmall {
                            needed: dict_end,
                            actual: slice.len(),
                        })?;
                Some(DecoderDictionary::copy(dict_slice))
            } else {
                None
            };

        let dir_start =
            usize::try_from(header.directory_offset).map_err(|_| OspreyError::CorruptedFooter)?;
        if dir_start < size_of::<OsvHeader>() {
            return Err(OspreyError::CorruptedFooter);
        }
        let num_pages =
            usize::try_from(header.num_minipages).map_err(|_| OspreyError::CorruptedFooter)?;
        let dir_bytes_len = num_pages
            .checked_mul(size_of::<MinipageDirectoryEntry>())
            .ok_or(OspreyError::CorruptedFooter)?;
        let dir_end = dir_start
            .checked_add(dir_bytes_len)
            .ok_or(OspreyError::CorruptedFooter)?;
        if dir_end > slice.len() {
            return Err(OspreyError::BufferTooSmall {
                needed: dir_end,
                actual: slice.len(),
            });
        }

        // Validate zero-copy alignment and pod casting of `MinipageDirectoryEntry` upfront.
        let _: &[MinipageDirectoryEntry] =
            try_cast_slice(&slice[dir_start..dir_end]).map_err(|err| match err {
                PodCastError::TargetAlignmentGreaterAndInputNotAligned => {
                    OspreyError::UnalignedMemory {
                        address: slice[dir_start..dir_end].as_ptr() as usize,
                        required_align: std::mem::align_of::<MinipageDirectoryEntry>(),
                    }
                }
                _ => OspreyError::BufferTooSmall {
                    needed: dir_bytes_len,
                    actual: slice[dir_start..dir_end].len(),
                },
            })?;

        Ok(Self {
            backing,
            header,
            dir_start,
            dir_end,
            decoder_dict,
        })
    }

    /// Returns the validated 64-byte `.osv` header.
    #[inline]
    pub const fn header(&self) -> &OsvHeader {
        &self.header
    }

    /// Returns the zero-copy `MinipageDirectoryEntry` slice mapped over the `.osv` slab.
    #[inline]
    pub fn directory(&self) -> &[MinipageDirectoryEntry] {
        let dir_bytes = &self.backing.as_slice()[self.dir_start..self.dir_end];
        try_cast_slice(dir_bytes).unwrap_or(&[])
    }

    /// Returns `true` if the slab contains an embedded Zstd dictionary.
    #[inline]
    pub const fn has_zstd_dict(&self) -> bool {
        self.decoder_dict.is_some()
    }

    /// Verifies `page_xxh64` over the compressed frame and decompresses minipage `minipage_id`.
    ///
    /// Returns [`OspreyError::ChecksumMismatch`] before invoking Zstd if even a single bit
    /// in the compressed frame is corrupted.
    pub fn decompress_minipage(&self, minipage_id: u32) -> OspreyResult<DecodedMinipage> {
        let idx = usize::try_from(minipage_id)
            .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
        let entry = self
            .directory()
            .get(idx)
            .ok_or(OspreyError::InvalidDimension { dim: idx })?;

        let start = usize::try_from(entry.byte_offset).map_err(|_| OspreyError::CorruptedFooter)?;
        if start < size_of::<OsvHeader>() {
            return Err(OspreyError::CorruptedFooter);
        }
        let len =
            usize::try_from(entry.compressed_size).map_err(|_| OspreyError::CorruptedFooter)?;
        let end = start.checked_add(len).ok_or(OspreyError::CorruptedFooter)?;
        let slab_bytes = self.backing.as_slice();
        let compressed = slab_bytes
            .get(start..end)
            .ok_or(OspreyError::BufferTooSmall {
                needed: end,
                actual: slab_bytes.len(),
            })?;

        if (self.header.flags & OSV_FLAG_ZSTD_ENABLED) == 0 {
            let computed = xxh64(compressed, 0);
            if computed != entry.page_xxh64 {
                return Err(OspreyError::ChecksumMismatch {
                    expected: entry.page_xxh64,
                    computed,
                });
            }
            let uncomp_len = usize::try_from(entry.uncompressed_size)
                .map_err(|_| OspreyError::CorruptedFooter)?;
            if compressed.len() != uncomp_len {
                return Err(OspreyError::CorruptedFooter);
            }
            let aligned = AlignedBuffer::<u8>::from_slice(compressed)?;
            let decoded = DecodedMinipage::from_aligned_buffer(aligned)?;
            if decoded.cluster_id() != entry.cluster_id
                || decoded.vector_count() != entry.vector_count
                || decoded.orig_dim() != self.header.orig_dim
            {
                return Err(OspreyError::CorruptedFooter);
            }
            return Ok(decoded);
        }

        let mut decompressor = match &self.decoder_dict {
            Some(dict) => zstd::bulk::Decompressor::with_prepared_dictionary(dict)?,
            None => zstd::bulk::Decompressor::new()?,
        };
        verify_and_decompress_minipage_with(
            compressed,
            entry,
            Some(self.header.orig_dim),
            &mut decompressor,
        )
    }
}

/// Validates `page_xxh64` before Zstd decompression, allocates an exact 64B-aligned buffer,
/// decompresses `compressed` with `decompressor`, and checks directory metadata invariants.
pub(crate) fn verify_and_decompress_minipage_with(
    compressed: &[u8],
    entry: &MinipageDirectoryEntry,
    expected_orig_dim: Option<u16>,
    decompressor: &mut zstd::bulk::Decompressor<'_>,
) -> OspreyResult<DecodedMinipage> {
    let computed = xxh64(compressed, 0);
    if computed != entry.page_xxh64 {
        return Err(OspreyError::ChecksumMismatch {
            expected: entry.page_xxh64,
            computed,
        });
    }

    let uncomp_len =
        usize::try_from(entry.uncompressed_size).map_err(|_| OspreyError::CorruptedFooter)?;
    const MAX_MINIPAGE_UNCOMPRESSED_BYTES: usize = 128 * 1024 * 1024;
    if uncomp_len > MAX_MINIPAGE_UNCOMPRESSED_BYTES {
        return Err(OspreyError::CorruptedFooter);
    }
    if let Ok(Some(frame_content_size)) = zstd::zstd_safe::get_frame_content_size(compressed) {
        if frame_content_size != u64::from(entry.uncompressed_size) {
            return Err(OspreyError::CorruptedFooter);
        }
    }
    let mut out = AlignedBuffer::<u8>::new(uncomp_len)?;
    let written = decompressor.decompress_to_buffer(compressed, out.as_mut_slice())?;
    if written != uncomp_len {
        return Err(OspreyError::CorruptedFooter);
    }

    let decoded = DecodedMinipage::from_aligned_buffer(out)?;
    if decoded.cluster_id() != entry.cluster_id || decoded.vector_count() != entry.vector_count {
        return Err(OspreyError::CorruptedFooter);
    }
    if let Some(dim) = expected_orig_dim {
        if decoded.orig_dim() != dim {
            return Err(OspreyError::CorruptedFooter);
        }
    }
    Ok(decoded)
}

#[inline]
fn exact_f16_f32_dot(vec_f16: &[f16], query_f32: &[f32]) -> f32 {
    osprey_simd::dot_f32_f16(query_f32, vec_f16)
}

struct CompressedMinipage {
    compressed_bytes: Vec<u8>,
    page_xxh64: u64,
    uncompressed_size: usize,
    cluster_id: u32,
    vector_count: u16,
    min_doc_id: u64,
    max_doc_id: u64,
    flags: u16,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct InternalMinipageHeader {
    vector_count: u16,
    orig_dim: u16,
    cluster_id: u32,
}

fn serialize_minipage(
    cluster_id: u32,
    orig_dim: u16,
    target_bytes: usize,
    group: &[(u32, u64, usize)],
    records: &[OsvVectorRecord<'_>],
) -> OspreyResult<UncompressedMinipage> {
    let vector_count = u16::try_from(group.len())
        .map_err(|_| OspreyError::InvalidDimension { dim: group.len() })?;
    let dim = usize::from(orig_dim);
    let total_payload: usize = group
        .iter()
        .map(|&(_, _, idx)| records[idx].payload.len())
        .sum();
    let total_size = MINIPAGE_INTERNAL_HEADER_BYTES
        + group.len() * (8 + dim * 2)
        + (group.len() + 1) * 4
        + total_payload;

    let mut bytes = Vec::with_capacity(total_size);
    let hdr = InternalMinipageHeader {
        vector_count: vector_count.to_le(),
        orig_dim: orig_dim.to_le(),
        cluster_id: cluster_id.to_le(),
    };
    bytes.extend_from_slice(bytemuck::bytes_of(&hdr));

    let mut doc_ids = Vec::with_capacity(group.len());
    let mut min_doc_id = u64::MAX;
    let mut max_doc_id = 0_u64;

    for &(_, doc_id, _) in group {
        doc_ids.push(doc_id);
        min_doc_id = min_doc_id.min(doc_id);
        max_doc_id = max_doc_id.max(doc_id);
        bytes.extend_from_slice(&doc_id.to_le_bytes());
    }

    for &(_, _, rec_idx) in group {
        for &val in records[rec_idx].vector_f16 {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
    }

    let mut offset = 0_u32;
    bytes.extend_from_slice(&offset.to_le_bytes());
    for &(_, _, rec_idx) in group {
        let len_u32 = u32::try_from(records[rec_idx].payload.len()).map_err(|_| {
            OspreyError::InvalidDimension {
                dim: records[rec_idx].payload.len(),
            }
        })?;
        offset = offset
            .checked_add(len_u32)
            .ok_or(OspreyError::InvalidDimension {
                dim: usize::try_from(offset).unwrap_or(usize::MAX),
            })?;
        bytes.extend_from_slice(&offset.to_le_bytes());
    }

    for &(_, _, rec_idx) in group {
        bytes.extend_from_slice(records[rec_idx].payload);
    }

    let flags = if bytes.len() > target_bytes {
        MINIPAGE_FLAG_OVERSIZED
    } else {
        0
    };

    Ok(UncompressedMinipage {
        cluster_id,
        vector_count,
        orig_dim,
        min_doc_id,
        max_doc_id,
        flags,
        doc_ids,
        bytes,
    })
}

fn compress_minipages_parallel(
    pages: &[UncompressedMinipage],
    zstd_level: i32,
    zstd_dict: &[u8],
) -> OspreyResult<Vec<CompressedMinipage>> {
    pages
        .par_iter()
        .map_init(
            || -> std::io::Result<zstd::bulk::Compressor<'_>> {
                if zstd_dict.is_empty() {
                    zstd::bulk::Compressor::new(zstd_level)
                } else {
                    zstd::bulk::Compressor::with_dictionary(zstd_level, zstd_dict)
                }
            },
            |compressor_res, page| -> OspreyResult<CompressedMinipage> {
                let compressor = match compressor_res {
                    Ok(c) => c,
                    Err(err) => {
                        return Err(OspreyError::Io(std::io::Error::new(
                            err.kind(),
                            err.to_string(),
                        )));
                    }
                };
                let compressed_bytes = compressor.compress(&page.bytes)?;
                let page_xxh64 = xxh64(&compressed_bytes, 0);
                Ok(CompressedMinipage {
                    compressed_bytes,
                    page_xxh64,
                    uncompressed_size: page.bytes.len(),
                    cluster_id: page.cluster_id,
                    vector_count: page.vector_count,
                    min_doc_id: page.min_doc_id,
                    max_doc_id: page.max_doc_id,
                    flags: page.flags,
                })
            },
        )
        .collect()
}

#[inline]
fn align_up_8_u64(offset: u64) -> OspreyResult<u64> {
    let rem = offset % 8;
    if rem == 0 {
        Ok(offset)
    } else {
        offset
            .checked_add(8 - rem)
            .ok_or(OspreyError::AllocationFailed {
                size: usize::MAX,
                align: 8,
            })
    }
}

#[inline]
fn align_up_64_u64(offset: u64) -> OspreyResult<u64> {
    let rem = offset % 64;
    if rem == 0 {
        Ok(offset)
    } else {
        offset
            .checked_add(64 - rem)
            .ok_or(OspreyError::AllocationFailed {
                size: usize::MAX,
                align: 64,
            })
    }
}

fn write_zero_padding<W: Write>(writer: &mut W, len: u64) -> OspreyResult<()> {
    const ZEROS: [u8; 64] = [0_u8; 64];
    let mut remaining = usize::try_from(len).map_err(|_| OspreyError::AllocationFailed {
        size: usize::MAX,
        align: 64,
    })?;
    while remaining > 0 {
        let chunk = remaining.min(ZEROS.len());
        writer.write_all(&ZEROS[..chunk])?;
        remaining -= chunk;
    }
    Ok(())
}

#[inline]
fn read_u16_le(bytes: &[u8], offset: usize) -> OspreyResult<u16> {
    let slice: [u8; 2] = bytes
        .get(offset..offset + 2)
        .and_then(|s| s.try_into().ok())
        .ok_or(OspreyError::CorruptedFooter)?;
    Ok(u16::from_le_bytes(slice))
}

#[inline]
fn read_u32_le(bytes: &[u8], offset: usize) -> OspreyResult<u32> {
    let slice: [u8; 4] = bytes
        .get(offset..offset + 4)
        .and_then(|s| s.try_into().ok())
        .ok_or(OspreyError::CorruptedFooter)?;
    Ok(u32::from_le_bytes(slice))
}

#[cfg(test)]
mod tests {
    use osprey_core::AlignedBuffer;

    use super::*;

    fn make_f16_vec(seed: usize, dim: usize) -> Vec<f16> {
        (0..dim)
            .map(|d| f16::from_f32(((seed * 13 + d * 7) % 100) as f32 * 0.05 - 2.0))
            .collect()
    }

    #[test]
    fn minipage_builder_respects_16kb_target_cluster_boundaries_and_oversized() -> OspreyResult<()>
    {
        let dim = 128usize;
        let num_vecs = 200usize;
        let vecs: Vec<Vec<f16>> = (0..num_vecs).map(|i| make_f16_vec(i, dim)).collect();
        let payloads: Vec<Vec<u8>> = (0..num_vecs)
            .map(|i| {
                format!(
                    r#"{{"id":{i},"category":"cat_{}","score":{:.3}}}"#,
                    i % 5,
                    i as f32 * 0.1
                )
                .into_bytes()
            })
            .collect();

        // Assign first 120 vectors to logical cluster 0, next 80 to logical cluster 1.
        let mut records = Vec::with_capacity(num_vecs);
        for i in 0..num_vecs {
            records.push(OsvVectorRecord {
                doc_id: 1000 + i as u64,
                cluster_id: if i < 120 { 0 } else { 1 },
                vector_f16: &vecs[i],
                payload: &payloads[i],
            });
        }

        // Reverse logical clusters via permutation: logical 0 -> physical 1, logical 1 -> physical 0.
        let perm = ClusterLocalityPermutation::from_tour(vec![1, 0])?;
        let pages = build_minipages(&records, dim, DEFAULT_MINIPAGE_TARGET_BYTES, Some(&perm))?;

        // Must be ordered strictly by physical cluster 0 first, then physical cluster 1.
        assert!(pages.len() >= 3);
        let mut seen_cluster_1 = false;
        let mut total_vecs_packed = 0usize;

        for page in &pages {
            assert!(page.vector_count >= 1);
            assert!(page.bytes.len() <= DEFAULT_MINIPAGE_TARGET_BYTES);
            if page.cluster_id == 1 {
                seen_cluster_1 = true;
            } else {
                assert_eq!(page.cluster_id, 0);
                assert!(!seen_cluster_1, "physical cluster 0 must precede cluster 1");
            }
            let view = parse_minipage(&page.bytes)?;
            assert_eq!(view.vector_count, page.vector_count);
            assert_eq!(view.orig_dim, 128);
            assert_eq!(view.cluster_id, page.cluster_id);

            let mut buf = vec![f16::ZERO; dim];
            for slot in 0..usize::from(view.vector_count) {
                let doc_id = view.doc_id(slot)?;
                let orig_idx = usize::try_from(doc_id - 1000).unwrap_or(usize::MAX);
                view.read_vector_f16(slot, &mut buf)?;
                assert_eq!(buf, vecs[orig_idx]);
                assert_eq!(view.payload(slot)?, payloads[orig_idx].as_slice());
            }
            total_vecs_packed += usize::from(page.vector_count);
        }
        assert_eq!(total_vecs_packed, num_vecs);

        // Oversized single vector (> 16 KB payload) must still pack into 1 page with MINIPAGE_FLAG_OVERSIZED.
        let huge_payload = vec![b'x'; 20_000];
        let oversized_rec = [OsvVectorRecord {
            doc_id: 42,
            cluster_id: 0,
            vector_f16: &vecs[0],
            payload: &huge_payload,
        }];
        let oversized_pages =
            build_minipages(&oversized_rec, dim, DEFAULT_MINIPAGE_TARGET_BYTES, None)?;
        assert_eq!(oversized_pages.len(), 1);
        assert_eq!(oversized_pages[0].vector_count, 1);
        assert_eq!(oversized_pages[0].flags, MINIPAGE_FLAG_OVERSIZED);
        let parsed = parse_minipage(&oversized_pages[0].bytes)?;
        assert_eq!(parsed.payload(0)?, huge_payload.as_slice());
        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Invokes zstd-sys C FFI
    fn zstd_dictionary_trainer_respects_thresholds_and_trains_when_sufficient() -> OspreyResult<()>
    {
        // Below 8 pages -> empty dictionary.
        let small_pages = vec![vec![b'a'; 16 * 1024]; 7];
        assert!(
            train_payload_zstd_dictionary(&small_pages, DEFAULT_ZSTD_DICT_MAX_BYTES)?.is_empty()
        );

        // >= 8 pages but < 32 KB total -> empty dictionary.
        let tiny_pages = vec![vec![b'a'; 1024]; 10];
        assert!(
            train_payload_zstd_dictionary(&tiny_pages, DEFAULT_ZSTD_DICT_MAX_BYTES)?.is_empty()
        );

        // Realistic 16 pages of structured JSON-like minipages (>= 128 KB total) -> trains non-empty dict.
        let dim = 64usize;
        let vecs: Vec<Vec<f16>> = (0..640).map(|i| make_f16_vec(i, dim)).collect();
        let payloads: Vec<Vec<u8>> = (0..640)
            .map(|i| {
                format!(
                    r#"{{"document_identifier":"doc_{i:06}","tenant":"acme-corp","region":"us-east-1","status":"active","rank":{i},"tags":["vector","osprey","colbert","slab"]}}"#
                )
                .into_bytes()
            })
            .collect();
        let records: Vec<OsvVectorRecord<'_>> = (0..640)
            .map(|i| OsvVectorRecord {
                doc_id: i as u64,
                cluster_id: u32::try_from(i / 80).unwrap_or(0),
                vector_f16: &vecs[i],
                payload: &payloads[i],
            })
            .collect();

        let pages = build_minipages(&records, dim, DEFAULT_MINIPAGE_TARGET_BYTES, None)?;
        assert!(pages.len() >= 8);
        let sample_pages: Vec<Vec<u8>> = pages.iter().map(|p| p.bytes.clone()).collect();
        let dict = train_payload_zstd_dictionary(&sample_pages, 16 * 1024)?;
        assert!(!dict.is_empty());
        assert!(dict.len() <= 16 * 1024);
        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Invokes zstd-sys C FFI and rayon threadpool
    fn osv_writer_end_to_end_roundtrip_with_dict_and_without_dict() -> OspreyResult<()> {
        let dim = 64usize;
        let num_vecs = 600usize;
        let vecs: Vec<Vec<f16>> = (0..num_vecs).map(|i| make_f16_vec(i, dim)).collect();
        let payloads: Vec<Vec<u8>> = (0..num_vecs)
            .map(|i| {
                format!(
                    r#"{{"doc":{i},"title":"OspreyDB Vector Slab Entry #{i}","shard":"s3-us-east","active":true}}"#
                )
                .into_bytes()
            })
            .collect();

        let records: Vec<OsvVectorRecord<'_>> = (0..num_vecs)
            .map(|i| OsvVectorRecord {
                doc_id: 50_000 + i as u64,
                cluster_id: u32::try_from(i % 6).unwrap_or(0),
                vector_f16: &vecs[i],
                payload: &payloads[i],
            })
            .collect();

        // Remap logical clusters [0..6] via reverse permutation.
        let perm = ClusterLocalityPermutation::from_tour(vec![5, 4, 3, 2, 1, 0])?;

        // Case 1: With dictionary training enabled.
        let writer_with_dict = OsvWriter::default();
        let (slab_bytes, report) =
            writer_with_dict.write_slab_to_bytes(&records, dim, Some(&perm))?;

        assert!(report.zstd_dict_bytes > 0);
        assert_eq!(report.header.magic, *b"OSPR");
        assert_eq!(report.header.format_kind, 0x02);
        assert_eq!(report.header.total_vectors, num_vecs as u64);
        assert_eq!(report.header.directory_offset % 64, 0);
        assert_eq!(report.header.zstd_dict_offset % 64, 0);
        assert_eq!(
            usize::try_from(report.header.directory_offset).unwrap_or(usize::MAX)
                + report.directory.len() * 48,
            slab_bytes.len()
        );

        let aligned_slab = AlignedBuffer::from_slice(&slab_bytes)?;
        let view = OsvSlabView::from_bytes(&aligned_slab)?;
        assert_eq!(view.directory.len(), report.directory.len());

        // Verify every doc_id maps cleanly to (minipage_id, slot_idx) and decodes identically.
        let mut decoded_vec = vec![f16::ZERO; dim];
        let mut page_cache: HashMap<u32, Vec<u8>> = HashMap::new();
        for (i, rec) in records.iter().enumerate() {
            let (minipage_id, slot_idx) = report
                .location_for_doc(rec.doc_id)
                .ok_or(OspreyError::CorruptedFooter)?;
            let page_bytes = match page_cache.get(&minipage_id) {
                Some(buf) => buf,
                None => {
                    let raw =
                        view.read_minipage(usize::try_from(minipage_id).unwrap_or(usize::MAX))?;
                    page_cache.insert(minipage_id, raw);
                    &page_cache[&minipage_id]
                }
            };
            let mp_view = parse_minipage(page_bytes)?;
            let slot = usize::from(slot_idx);
            assert_eq!(mp_view.doc_id(slot)?, rec.doc_id);
            assert_eq!(mp_view.cluster_id, perm.to_physical(rec.cluster_id)?);
            mp_view.read_vector_f16(slot, &mut decoded_vec)?;
            assert_eq!(decoded_vec, vecs[i]);
            assert_eq!(mp_view.payload(slot)?, payloads[i].as_slice());
        }

        // Verify checksum tampering detection before Zstd decompression.
        let mut tampered = aligned_slab.as_slice().to_vec();
        let first_offset = usize::try_from(view.directory[0].byte_offset).unwrap_or(0);
        tampered[first_offset] ^= 0xFF;
        let tampered_aligned = AlignedBuffer::from_slice(&tampered)?;
        let tampered_view = OsvSlabView::from_bytes(&tampered_aligned)?;
        assert!(matches!(
            tampered_view.read_minipage(0),
            Err(OspreyError::ChecksumMismatch { .. })
        ));

        // Case 2: Dictionary disabled (small or explicit off).
        let writer_no_dict = OsvWriter::new(OsvWriterConfig {
            enable_zstd_dict: false,
            ..OsvWriterConfig::default()
        });
        let (no_dict_bytes, no_dict_report) =
            writer_no_dict.write_slab_to_bytes(&records[..10], dim, None)?;
        assert_eq!(no_dict_report.zstd_dict_bytes, 0);
        assert_eq!(no_dict_report.header.zstd_dict_offset, 0);
        assert_eq!(no_dict_report.header.zstd_dict_len, 0);
        let no_dict_aligned = AlignedBuffer::from_slice(&no_dict_bytes)?;
        let no_dict_view = OsvSlabView::from_bytes(&no_dict_aligned)?;
        assert_eq!(no_dict_view.header.total_vectors, 10);
        assert!(no_dict_view.zstd_dict.is_empty());
        let mp0 = no_dict_view.read_minipage(0)?;
        assert_eq!(parse_minipage(&mp0)?.vector_count, 2);
        let total_no_dict_vecs: u64 = no_dict_view
            .directory
            .iter()
            .map(|e| u64::from(e.vector_count))
            .sum();
        assert_eq!(total_no_dict_vecs, 10);

        // Case 3: Empty records produces valid 64B header-only .osv file.
        let (empty_bytes, empty_report) = writer_no_dict.write_slab_to_bytes(&[], dim, None)?;
        assert_eq!(empty_bytes.len(), 64);
        assert_eq!(empty_report.header.num_minipages, 0);
        assert_eq!(empty_report.header.directory_offset, 64);
        let empty_aligned = AlignedBuffer::from_slice(&empty_bytes)?;
        let empty_view = OsvSlabView::from_bytes(&empty_aligned)?;
        assert!(empty_view.directory.is_empty());
        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Invokes zstd-sys C FFI
    fn osv_slab_reader_decodes_zero_copy_odd_dims_and_exact_dot() -> OspreyResult<()> {
        // Odd dim (3) and odd vector count (3) verifies payload_offsets alignment handling.
        let dim = 3usize;
        let vecs = [
            make_f16_vec(1, dim),
            make_f16_vec(2, dim),
            make_f16_vec(3, dim),
        ];
        let payloads = [
            br#"{"k":"a"}"#.as_slice(),
            br#"{"k":"bb"}"#.as_slice(),
            br#"{"k":"ccc"}"#.as_slice(),
        ];
        let records = [
            OsvVectorRecord {
                doc_id: 10,
                cluster_id: 0,
                vector_f16: &vecs[0],
                payload: payloads[0],
            },
            OsvVectorRecord {
                doc_id: 20,
                cluster_id: 0,
                vector_f16: &vecs[1],
                payload: payloads[1],
            },
            OsvVectorRecord {
                doc_id: 30,
                cluster_id: 0,
                vector_f16: &vecs[2],
                payload: payloads[2],
            },
        ];

        let writer = OsvWriter::default();
        let (bytes, _) = writer.write_slab_to_bytes(&records, dim, None)?;
        let aligned = AlignedBuffer::from_slice(&bytes)?;
        let reader = OsvSlabReader::from_bytes(aligned)?;

        assert_eq!(reader.directory().len(), 1);
        assert!(!reader.has_zstd_dict());

        let decoded = reader.decompress_minipage(0)?;
        assert_eq!(decoded.vector_count(), 3);
        assert_eq!(decoded.orig_dim(), 3);
        assert_eq!(decoded.doc_ids(), &[10, 20, 30]);

        let query = [1.5_f32, -2.0, 0.25];
        for (slot, rec) in records.iter().enumerate() {
            let slot_u16 = u16::try_from(slot).unwrap_or(0);
            assert_eq!(decoded.doc_id(slot_u16)?, rec.doc_id);
            assert_eq!(decoded.vector_f16(slot_u16)?, rec.vector_f16);
            assert_eq!(decoded.payload_bytes(slot_u16)?, rec.payload);
            assert_eq!(
                decoded.payload_str(slot_u16)?,
                std::str::from_utf8(rec.payload).unwrap_or("")
            );

            let expected: f32 = rec
                .vector_f16
                .iter()
                .zip(query.iter())
                .map(|(a, &b)| a.to_f32() * b)
                .sum();
            let got = decoded.compute_exact_f16_dot(slot_u16, &query);
            assert!((got - expected).abs() < 1e-5);
        }

        // Out-of-bounds checks
        assert!(matches!(
            decoded.vector_f16(3),
            Err(OspreyError::InvalidDimension { dim: 3 })
        ));
        assert!(matches!(
            reader.decompress_minipage(1),
            Err(OspreyError::InvalidDimension { dim: 1 })
        ));
        assert_eq!(decoded.compute_exact_f16_dot(3, &query), 0.0);
        assert_eq!(decoded.compute_exact_f16_dot(0, &[1.0]), 0.0);
        Ok(())
    }

    #[test]
    fn osv_uncompressed_mode_roundtrip_and_8byte_alignment() -> OspreyResult<()> {
        let dim = 17usize; // odd dim to exercise 8-byte padding between minipages
        let vecs: Vec<Vec<f16>> = (0..30).map(|i| make_f16_vec(i, dim)).collect();
        let payloads: Vec<Vec<u8>> = (0..30)
            .map(|i| format!(r#"{{"doc":{i}}}"#).into_bytes())
            .collect();
        let records: Vec<OsvVectorRecord<'_>> = (0..30)
            .map(|i| OsvVectorRecord {
                doc_id: 1000 + i as u64,
                cluster_id: u32::try_from(i % 5).unwrap_or(0),
                vector_f16: &vecs[i],
                payload: &payloads[i],
            })
            .collect();

        let (bytes, report) = OsvWriter::new(OsvWriterConfig::uncompressed())
            .write_slab_to_bytes(&records, dim, None)?;
        assert_eq!(report.header.flags & OSV_FLAG_ZSTD_ENABLED, 0);
        assert_eq!(report.header.zstd_level, 0);
        assert_eq!(report.zstd_dict_bytes, 0);
        for entry in &report.directory {
            assert_eq!(entry.byte_offset % 8, 0);
            assert_eq!(entry.compressed_size, entry.uncompressed_size);
        }

        let aligned = AlignedBuffer::from_slice(&bytes)?;
        let reader = OsvSlabReader::from_bytes(aligned)?;
        for (page_id, _) in reader.directory().iter().enumerate() {
            let decoded = reader.decompress_minipage(u32::try_from(page_id).unwrap_or(0))?;
            assert_eq!(decoded.orig_dim(), 17);
        }

        // Also verify StreamingOsvWriter in uncompressed mode
        let mut cursor = std::io::Cursor::new(Vec::new());
        let mut streaming = StreamingOsvWriter::new(
            &mut cursor,
            OsvWriterConfig::uncompressed(),
            dim,
            Vec::new(),
        )?;
        let locs1 = streaming.write_chunk(&records[..15], None)?;
        let locs2 = streaming.write_chunk(&records[15..], None)?;
        assert_eq!(locs1.len() + locs2.len(), 30);
        let stream_report = streaming.finish()?;
        assert_eq!(stream_report.header.flags & OSV_FLAG_ZSTD_ENABLED, 0);
        for entry in &stream_report.directory {
            assert_eq!(entry.byte_offset % 8, 0);
        }
        let stream_aligned = AlignedBuffer::from_slice(&cursor.into_inner())?;
        let stream_view = OsvSlabView::from_bytes(&stream_aligned)?;
        let page0 = stream_view.read_minipage(0)?;
        let mp = parse_minipage(&page0)?;
        assert!(mp.vector_f16_slice(0).is_some());
        Ok(())
    }
}
