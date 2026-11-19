//! Asymmetric `RaBitQ` quantizer (`arXiv:2405.12497`, `arXiv:2409.09913`) and `.osi` cluster-grouped index builder.
//!
//! Implements intra-cluster residual extraction, exact `f64` L2 norm computation,
//! anti-subnormal `f16` clamping, FWHT orthogonal rotation, L1 projection factor
//! computation, 1-bit Little-Endian bit-packing, and parallel `OsiIndex` construction
//! with validated zero-copy `.osi` serialization/deserialization.

use std::collections::HashSet;
use std::mem::size_of;

use half::f16;
#[cfg(not(miri))]
use rayon::prelude::*;
use roaring::RoaringBitmap;
use xxhash_rust::xxh64::xxh64;

use crate::aligned::{ensure_aligned_64, AlignedBuffer, CACHE_LINE_ALIGN};
use crate::cluster::BinaryKMeansResult;
use crate::error::{OspreyError, OspreyResult};
use crate::format::osi::{
    cast_centroids_f16, cast_centroids_f16_mut, centroids_padded_bytes, cluster_block_bytes,
    deserialize_tombstone_footer, serialize_tombstone_footer, OsiClusterSpan, OsiClusterView,
    OsiHeader, RabitqEntryHeader,
};
use crate::format::{checked_align_up_64, FORMAT_KIND_OSI, FORMAT_VERSION_V1, OSPREY_MAGIC};
use crate::math::{
    fwht_scalar_in_place, padded_dimension, prepare_and_pad_vector, prepare_and_pad_vector_f16,
    RademacherSigns,
};
use crate::topk::{BoundedTopKHeap, ScoredCandidate};

/// Minimum additive epsilon in residual normalization `u_i = r_i / (||r||_2 + 1e-8)`.
pub const RABITQ_NORM_EPS: f64 = 1e-8;

/// Upper safety clamp for `f16` scalar factors (`< f16::MAX = 65504.0`).
pub const RABITQ_F16_MAX_CLAMP: f32 = 65_500.0;

/// Lower anti-subnormal clamp for `l1_norm_factor` (`>= f16::MIN_POSITIVE = 2^-14`).
pub const RABITQ_L1_MIN_NORMAL_F32: f32 = 6.1035e-5;

/// `OsiHeader::flags` bit indicating that stored `centroids_f16` reside in unrotated padded
/// space (so cluster probing uses `q_padded` and only intra-cluster residuals are FWHT-rotated).
pub const OSI_FLAG_UNROTATED_CENTROIDS: u8 = 0x02;

/// Pair `(cluster_id, centroid_ip)` returned by cluster probing.
pub type ProbedCluster = (u32, f32);

/// Hardware-accelerated or scalar FWHT rotation hook consumed by `RaBitQ` encoding.
///
/// Implemented by [`ScalarRabitqKernel`] in `osprey-core` and by `SimdEngine` /
/// `ScalarBackend` in `osprey-simd`, keeping the workspace crate DAG acyclic.
pub trait RabitqFwhtKernel: Send + Sync {
    /// Applies XOR sign masks and orthogonal FWHT `O(d log d)` in-place.
    fn fwht_rotate_in_place(&self, data: &mut [f32], signs_xor_mask: &[u32]) -> OspreyResult<()>;
}

/// Hardware-accelerated or scalar Stage-1 scan kernel hook consumed by [`OsiIndex::search_candidates`].
pub trait RabitqScanKernel: RabitqFwhtKernel {
    /// Hints to the CPU cache hierarchy to prefetch the 64-byte cache line starting at `bytes`
    /// into L1 data cache (`_MM_HINT_T0`) ahead of sequential `.osi` scanning.
    #[inline]
    fn prefetch_l1(&self, _bytes: &[u8]) {}

    /// Computes the exact `int4 x 1-bit` inner product `sum_i q_int[i] * (2 * b_x[i] - 1)`
    /// via bitwise XNOR + AND + POPCNT with zero floating-point operations in the inner loop.
    fn rabitq_4bit_dot_u64(
        &self,
        query_planes: &[[u64; 4]],
        query_signs: &[u64],
        query_abs_sum: i32,
        db_bits: &[u64],
    ) -> i32;
}

/// Portable scalar implementation of [`RabitqFwhtKernel`] and [`RabitqScanKernel`] for
/// standalone `osprey-core` usage.
#[derive(Copy, Clone, Debug, Default)]
pub struct ScalarRabitqKernel;

impl RabitqScanKernel for ScalarRabitqKernel {
    #[inline]
    fn rabitq_4bit_dot_u64(
        &self,
        query_planes: &[[u64; 4]],
        query_signs: &[u64],
        query_abs_sum: i32,
        db_bits: &[u64],
    ) -> i32 {
        let mut weighted_matches = 0_i64;
        for ((planes, &q_sign), &db_word) in query_planes
            .iter()
            .zip(query_signs.iter())
            .zip(db_bits.iter())
        {
            let match_word = !(q_sign ^ db_word);
            let p0 = i64::from((planes[0] & match_word).count_ones());
            let p1 = i64::from((planes[1] & match_word).count_ones());
            let p2 = i64::from((planes[2] & match_word).count_ones());
            let p3 = i64::from((planes[3] & match_word).count_ones());
            weighted_matches += p0 + (p1 << 1) + (p2 << 2) + (p3 << 3);
        }
        let res = weighted_matches.wrapping_mul(2) - i64::from(query_abs_sum);
        i32::try_from(res).unwrap_or(i32::MAX)
    }
}

impl RabitqFwhtKernel for ScalarRabitqKernel {
    #[inline]
    fn fwht_rotate_in_place(&self, data: &mut [f32], signs_xor_mask: &[u32]) -> OspreyResult<()> {
        let d = data.len();
        if d == 0 || !d.is_power_of_two() || d != signs_xor_mask.len() {
            return Err(OspreyError::InvalidDimension { dim: d });
        }

        for (val, &mask) in data.iter_mut().zip(signs_xor_mask.iter()) {
            *val = f32::from_bits(val.to_bits() ^ mask);
        }

        let mut h = 1usize;
        while h < d {
            let step = h * 2;
            for i in (0..d).step_by(step) {
                for j in i..(i + h) {
                    let a = data[j];
                    let b = data[j + h];
                    data[j] = a + b;
                    data[j + h] = a - b;
                }
            }
            h = step;
        }

        let d_u32 = u32::try_from(d).map_err(|_| OspreyError::InvalidDimension { dim: d })?;
        #[allow(clippy::cast_possible_truncation)]
        let inv_sqrt_d = (1.0_f64 / f64::from(d_u32).sqrt()) as f32;
        for val in data.iter_mut() {
            *val *= inv_sqrt_d;
        }

        Ok(())
    }
}

/// Physical storage location metadata for a vector inside `.osv` slabs/minipages.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct VectorLocation {
    /// Target `.osv` slab identifier (`0` for single-slab segments).
    pub slab_id: u16,
    /// Target `.osv` minipage identifier.
    pub minipage_id: u32,
    /// Slot index within the target minipage.
    pub slot_idx: u16,
}

/// Scalar `f16` parameters emitted by asymmetric `RaBitQ` residual quantization.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RabitqEncodedResidual {
    /// Clamped Euclidean norm `||x - c_k||_2` in IEEE-754 `f16`.
    pub res_norm: f16,
    /// Anti-subnormal clamped L1 projection factor `||v||_1 / D` in IEEE-754 `f16`.
    pub l1_norm_factor: f16,
}

/// Minimum `q_max` threshold below which `q_scale` defaults to `1.0`.
pub const QUERY_QUANT_EPS: f32 = 1e-12;

/// Symmetrically quantized `int4` (`[-7, +7]`) query transposed into 4 unsigned magnitude
/// bit-planes and 1 sign mask (`bit = 1` iff `q_int[i] >= 0`) for allocation-free
/// bitwise XNOR+AND+POPCNT inner products.
#[derive(Clone, Debug, PartialEq)]
pub struct TransposedQuery4Bit {
    /// Bit-planes `[p0, p1, p2, p3]` of `|q_int|` packed in Little-Endian `u64` words (`len = padded_dim / 64`).
    pub planes: Vec<[u64; 4]>,
    /// Sign mask (`bit b == 1` iff `q_int[64*w + b] >= 0`) (`len = padded_dim / 64`).
    pub sign_mask: Vec<u64>,
    /// Precomputed `sum_i |q_int[i]|`, subtracted once per database dot product.
    pub query_abs_sum: i32,
    /// Symmetric quantization step `q_max / 7.0` (or `1.0` for near-zero queries).
    pub q_scale: f32,
    /// Rotated `f32` query retained for validation, centroid probing, and diagnostics.
    pub q_rot: Vec<f32>,
}

impl TransposedQuery4Bit {
    /// Quantizes `q_rot` symmetrically to `[-7, +7]` (`q_scale = q_max / 7.0`) and transposes
    /// magnitude bit-planes and sign masks in registers without intermediate bit-matrix allocations.
    pub fn from_rotated_query(q_rot: &[f32]) -> Self {
        let mut q_max = 0.0_f32;
        for &val in q_rot {
            if val.is_finite() {
                let abs_val = val.abs();
                if abs_val > q_max {
                    q_max = abs_val;
                }
            }
        }
        let q_scale = if q_max > QUERY_QUANT_EPS {
            q_max / 7.0_f32
        } else {
            1.0_f32
        };
        let inv_scale = 1.0_f32 / q_scale;

        let n_words = q_rot.len().div_ceil(64);
        let mut planes = Vec::with_capacity(n_words);
        let mut sign_mask = Vec::with_capacity(n_words);
        let mut query_abs_sum = 0_i32;

        for chunk in q_rot.chunks(64) {
            let mut p0 = 0_u64;
            let mut p1 = 0_u64;
            let mut p2 = 0_u64;
            let mut p3 = 0_u64;
            let mut s_mask = 0_u64;

            for (b, &val) in chunk.iter().enumerate() {
                let finite_val = if val.is_finite() { val } else { 0.0_f32 };
                #[allow(clippy::cast_possible_truncation)]
                let q_int = (finite_val * inv_scale)
                    .round_ties_even()
                    .clamp(-7.0_f32, 7.0_f32) as i8;

                if q_int >= 0 {
                    s_mask |= 1_u64 << b;
                }
                let mag = q_int.unsigned_abs();
                query_abs_sum += i32::from(mag);

                let mag_u64 = u64::from(mag);
                p0 |= (mag_u64 & 1) << b;
                p1 |= ((mag_u64 >> 1) & 1) << b;
                p2 |= ((mag_u64 >> 2) & 1) << b;
                p3 |= ((mag_u64 >> 3) & 1) << b;
            }

            // Pad trailing bits (only if `q_rot.len()` was not a multiple of 64) as non-negative zero.
            if chunk.len() < 64 {
                for b in chunk.len()..64 {
                    s_mask |= 1_u64 << b;
                }
            }

            planes.push([p0, p1, p2, p3]);
            sign_mask.push(s_mask);
        }

        Self {
            planes,
            sign_mask,
            query_abs_sum,
            q_scale,
            q_rot: q_rot.to_vec(),
        }
    }

    /// Validated fallible constructor enforcing `padded_dim >= 64`, power-of-two dimension,
    /// and strict finiteness of every query coordinate.
    pub fn try_from_rotated_query(q_rot: &[f32]) -> OspreyResult<Self> {
        let d = q_rot.len();
        if d < 64 || !d.is_power_of_two() {
            return Err(OspreyError::InvalidDimension { dim: d });
        }
        for (index, &val) in q_rot.iter().enumerate() {
            if !val.is_finite() {
                return Err(OspreyError::NonFiniteEmbedding { index });
            }
        }
        Ok(Self::from_rotated_query(q_rot))
    }

    /// Constructs a [`TransposedQuery4Bit`] directly from pre-quantized `int4` coordinates
    /// (`[-15, +15]`), useful for bit-exact differential testing and custom quantizers.
    pub fn from_quantized_i8(q_int: &[i8], q_scale: f32, q_rot: Vec<f32>) -> OspreyResult<Self> {
        let d = q_int.len();
        if d < 64 || !d.is_power_of_two() {
            return Err(OspreyError::InvalidDimension { dim: d });
        }
        let n_words = d / 64;
        let mut planes = Vec::with_capacity(n_words);
        let mut sign_mask = Vec::with_capacity(n_words);
        let mut query_abs_sum = 0_i32;

        for chunk in q_int.as_chunks::<64>().0 {
            let mut p0 = 0_u64;
            let mut p1 = 0_u64;
            let mut p2 = 0_u64;
            let mut p3 = 0_u64;
            let mut s_mask = 0_u64;

            for (b, &q_val) in chunk.iter().enumerate() {
                let clamped = q_val.clamp(-15, 15);
                if clamped >= 0 {
                    s_mask |= 1_u64 << b;
                }
                let mag = clamped.unsigned_abs();
                query_abs_sum += i32::from(mag);

                let mag_u64 = u64::from(mag);
                p0 |= (mag_u64 & 1) << b;
                p1 |= ((mag_u64 >> 1) & 1) << b;
                p2 |= ((mag_u64 >> 2) & 1) << b;
                p3 |= ((mag_u64 >> 3) & 1) << b;
            }

            planes.push([p0, p1, p2, p3]);
            sign_mask.push(s_mask);
        }

        Ok(Self {
            planes,
            sign_mask,
            query_abs_sum,
            q_scale,
            q_rot,
        })
    }

    /// Portable scalar reference evaluation of the exact 4-bit x 1-bit inner product
    /// `sum_i q_int[i] * (2 * b_x[i] - 1)` against packed 1-bit `db_bits`.
    #[inline]
    pub fn dot_1bit_scalar(&self, db_bits: &[u64]) -> i32 {
        let mut weighted_matches = 0_u64;
        for ((planes, &q_sign), &db_word) in self
            .planes
            .iter()
            .zip(self.sign_mask.iter())
            .zip(db_bits.iter())
        {
            let match_word = !(q_sign ^ db_word);
            let p0 = u64::from((planes[0] & match_word).count_ones());
            let p1 = u64::from((planes[1] & match_word).count_ones());
            let p2 = u64::from((planes[2] & match_word).count_ones());
            let p3 = u64::from((planes[3] & match_word).count_ones());
            weighted_matches += p0 + (p1 << 1) + (p2 << 2) + (p3 << 3);
        }
        let twice = i64::try_from(weighted_matches)
            .unwrap_or(i64::MAX)
            .wrapping_mul(2);
        let res = twice - i64::from(self.query_abs_sum);
        i32::try_from(res).unwrap_or(i32::MAX)
    }
}

/// Validates dimensions and finiteness, extracts `r = x - c_k` in-place, accumulates exact
/// `f64` L2 norm, clamps `res_norm` to `[0.0, 65500.0]`, and normalizes `u = r / (||r||_2 + 1e-8)`.
fn extract_and_normalize_residual_in_place(
    x_and_u: &mut [f32],
    centroid_f16: &[f16],
) -> OspreyResult<f16> {
    let padded_dim = x_and_u.len();
    if padded_dim < 64 || !padded_dim.is_power_of_two() {
        return Err(OspreyError::InvalidDimension { dim: padded_dim });
    }
    if centroid_f16.len() != padded_dim {
        return Err(OspreyError::InvalidDimension {
            dim: centroid_f16.len(),
        });
    }

    let mut sq_sum = 0.0_f64;
    for (index, (xi, &ci_f16)) in x_and_u.iter_mut().zip(centroid_f16.iter()).enumerate() {
        if !xi.is_finite() {
            return Err(OspreyError::NonFiniteEmbedding { index });
        }
        let ci = ci_f16.to_f32();
        if !ci.is_finite() {
            return Err(OspreyError::NonFiniteEmbedding { index });
        }
        let r_i = *xi - ci;
        let r_f64 = f64::from(r_i);
        sq_sum += r_f64 * r_f64;
        *xi = r_i;
    }

    let norm_l2 = sq_sum.sqrt();
    #[allow(clippy::cast_possible_truncation)]
    let res_norm_f32 = (norm_l2 as f32).clamp(0.0, RABITQ_F16_MAX_CLAMP);
    let res_norm = f16::from_f32(res_norm_f32);

    let inv_denom = 1.0_f64 / (norm_l2 + RABITQ_NORM_EPS);
    for u_i in x_and_u.iter_mut() {
        #[allow(clippy::cast_possible_truncation)]
        {
            *u_i = (f64::from(*u_i) * inv_denom) as f32;
        }
    }

    Ok(res_norm)
}

/// Validates dimensions and finiteness, copies `x_padded` into `scratch_u`, and normalizes
/// the intra-cluster residual `u = (x - c_k) / (||r||_2 + 1e-8)`.
fn extract_and_normalize_residual(
    x_padded: &[f32],
    centroid_f16: &[f16],
    scratch_u: &mut [f32],
) -> OspreyResult<f16> {
    if scratch_u.len() != x_padded.len() {
        return Err(OspreyError::InvalidDimension {
            dim: scratch_u.len(),
        });
    }
    scratch_u.copy_from_slice(x_padded);
    extract_and_normalize_residual_in_place(scratch_u, centroid_f16)
}

/// Computes `l1_norm_factor = (1/D) * sum(|v_i|)` with anti-subnormal clamping and packs
/// 1-bit sign codes (`v_i >= 0.0` -> bit `1`) in Little-Endian `u64` words.
fn compute_l1_and_pack_1bit(v: &[f32], out_words: &mut [u64]) -> OspreyResult<f16> {
    let padded_dim = v.len();
    if padded_dim < 64 || !padded_dim.is_power_of_two() {
        return Err(OspreyError::InvalidDimension { dim: padded_dim });
    }
    let words_per_vec = padded_dim / 64;
    if out_words.len() != words_per_vec {
        return Err(OspreyError::BufferTooSmall {
            needed: words_per_vec * size_of::<u64>(),
            actual: std::mem::size_of_val(out_words),
        });
    }

    let mut sum_abs = 0.0_f64;
    for &val in v {
        sum_abs += f64::from(val.abs());
    }
    let d_u32 =
        u32::try_from(padded_dim).map_err(|_| OspreyError::InvalidDimension { dim: padded_dim })?;
    let l1_norm_factor = sum_abs / f64::from(d_u32);
    #[allow(clippy::cast_possible_truncation)]
    let l1_clamped = (l1_norm_factor as f32).clamp(RABITQ_L1_MIN_NORMAL_F32, RABITQ_F16_MAX_CLAMP);
    let l1_f16 = f16::from_f32(l1_clamped);

    for (w, out_word) in out_words.iter_mut().enumerate() {
        let base = w * 64;
        let mut word = 0_u64;
        for b in 0..64 {
            if v[base + b] >= 0.0_f32 {
                word |= 1_u64 << b;
            }
        }
        *out_word = word;
    }

    Ok(l1_f16)
}

/// Encodes one sanitized, zero-padded residual into asymmetric `RaBitQ`:
/// 1. Promotes `centroid_f16` to `f32`, computes `r = x - c_k`, accumulates `||r||_2` in `f64`,
///    and clamps `res_norm` to `[0.0, 65500.0]`.
/// 2. Normalizes `u = r / (||r||_2 + 1e-8)` into `scratch_u` (zero heap allocation).
/// 3. Applies FWHT rotation `v = P * u` in-place using `signs`.
/// 4. Computes `l1_norm_factor = (1/D) * sum(|v_i|)` with anti-subnormal clamp `[6.1035e-5, 65500.0]`.
/// 5. Packs sign bits (`v_i >= 0.0` -> `1`) into `out_words` in Little-Endian `u64` layout.
pub fn encode_rabitq_residual<K: RabitqFwhtKernel + ?Sized>(
    x_padded: &[f32],
    centroid_f16: &[f16],
    signs: &RademacherSigns,
    simd: &K,
    scratch_u: &mut [f32],
    out_words: &mut [u64],
) -> OspreyResult<RabitqEncodedResidual> {
    if signs.padded_dim() != x_padded.len() {
        return Err(OspreyError::InvalidDimension {
            dim: x_padded.len(),
        });
    }
    let res_norm = extract_and_normalize_residual(x_padded, centroid_f16, scratch_u)?;
    simd.fwht_rotate_in_place(scratch_u, signs.xor_masks())?;
    let l1_norm_factor = compute_l1_and_pack_1bit(scratch_u, out_words)?;
    Ok(RabitqEncodedResidual {
        res_norm,
        l1_norm_factor,
    })
}

/// Encodes an already FWHT-rotated vector and rotated `f16` centroid into asymmetric `RaBitQ`
/// in-place (overwriting `x_and_u` with normalized `u`), without a second FWHT rotation.
pub fn encode_prerotated_rabitq_residual_in_place(
    x_and_u: &mut [f32],
    centroid_rotated_f16: &[f16],
    out_words: &mut [u64],
) -> OspreyResult<RabitqEncodedResidual> {
    let res_norm = extract_and_normalize_residual_in_place(x_and_u, centroid_rotated_f16)?;
    let l1_norm_factor = compute_l1_and_pack_1bit(x_and_u, out_words)?;
    Ok(RabitqEncodedResidual {
        res_norm,
        l1_norm_factor,
    })
}

/// Encodes an already FWHT-rotated vector and rotated `f16` centroid into asymmetric `RaBitQ`
/// without applying a redundant second FWHT rotation.
pub fn encode_prerotated_rabitq_residual(
    x_rotated: &[f32],
    centroid_rotated_f16: &[f16],
    scratch_u: &mut [f32],
    out_words: &mut [u64],
) -> OspreyResult<RabitqEncodedResidual> {
    let res_norm = extract_and_normalize_residual(x_rotated, centroid_rotated_f16, scratch_u)?;
    let l1_norm_factor = compute_l1_and_pack_1bit(scratch_u, out_words)?;
    Ok(RabitqEncodedResidual {
        res_norm,
        l1_norm_factor,
    })
}

/// Convenience helper invoking the scalar FWHT kernel for [`encode_rabitq_residual`].
pub fn encode_rabitq_residual_scalar(
    x_padded: &[f32],
    centroid_f16: &[f16],
    signs: &RademacherSigns,
    scratch_u: &mut [f32],
    out_words: &mut [u64],
) -> OspreyResult<RabitqEncodedResidual> {
    let expected_words = x_padded.len() / 64;
    if out_words.len() != expected_words {
        return Err(OspreyError::BufferTooSmall {
            needed: expected_words * size_of::<u64>(),
            actual: std::mem::size_of_val(out_words),
        });
    }
    let res_norm = extract_and_normalize_residual(x_padded, centroid_f16, scratch_u)?;
    fwht_scalar_in_place(scratch_u, signs)?;
    let l1_norm_factor = compute_l1_and_pack_1bit(scratch_u, out_words)?;
    Ok(RabitqEncodedResidual {
        res_norm,
        l1_norm_factor,
    })
}

/// Cluster-grouped `.osi` (Osprey Resident 1-Bit Index) containing `f16` centroids,
/// `RabitqEntryHeader` metadata, 64-byte-aligned 1-bit codes, and `RoaringBitmap` tombstones.
pub struct OsiIndex {
    header: OsiHeader,
    centroids_f16: AlignedBuffer<f16>,
    cluster_offsets: Vec<u32>,
    entries: Vec<RabitqEntryHeader>,
    codes: AlignedBuffer<u64>,
    tombstones: RoaringBitmap,
}

impl std::fmt::Debug for OsiIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OsiIndex")
            .field("header", &self.header)
            .field("centroids_len", &self.centroids_f16.len())
            .field("cluster_offsets_len", &self.cluster_offsets.len())
            .field("entries_len", &self.entries.len())
            .field("codes_len", &self.codes.len())
            .field("tombstones_len", &self.tombstones.len())
            .finish()
    }
}

/// Zero-copy borrowed view over a validated 64-byte-aligned `.osi` memory buffer or mmap.
#[derive(Debug)]
pub struct OsiMappedView<'a> {
    header: &'a OsiHeader,
    centroids_f16: &'a [f16],
    cluster_spans: &'a [OsiClusterSpan],
    entries_section: &'a [u8],
    tombstones: RoaringBitmap,
}

impl<'a> OsiMappedView<'a> {
    /// Validates alignment, header/file/tombstone `xxHash64` checksums, and section bounds,
    /// returning a zero-copy view borrowing directly from `bytes`.
    pub fn from_bytes(bytes: &'a [u8]) -> OspreyResult<Self> {
        ensure_aligned_64(bytes)?;
        let header = OsiHeader::from_bytes(bytes)?;

        let k = usize::try_from(header.num_clusters)
            .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
        if k == 0 {
            return Err(OspreyError::InvalidDimension { dim: 0 });
        }

        let cent_off =
            usize::try_from(header.centroids_offset).map_err(|_| OspreyError::CorruptedFooter)?;
        let ent_off =
            usize::try_from(header.entries_offset).map_err(|_| OspreyError::CorruptedFooter)?;
        let tomb_off =
            usize::try_from(header.tombstones_offset).map_err(|_| OspreyError::CorruptedFooter)?;

        if cent_off < size_of::<OsiHeader>() || ent_off < cent_off || tomb_off < ent_off {
            return Err(OspreyError::CorruptedFooter);
        }

        // Minimum trailer: 16-byte tombstone header/checksum + 8-byte global file_xxh64.
        let min_file_len = tomb_off
            .checked_add(24)
            .ok_or(OspreyError::CorruptedFooter)?;
        if bytes.len() < min_file_len {
            return Err(OspreyError::BufferTooSmall {
                needed: min_file_len,
                actual: bytes.len(),
            });
        }

        let payload_end = bytes.len() - size_of::<u64>();
        let file_hash_bytes: [u8; 8] = bytes[payload_end..]
            .try_into()
            .map_err(|_| OspreyError::CorruptedFooter)?;
        let expected_file_hash = u64::from_le_bytes(file_hash_bytes);
        let computed_file_hash = xxh64(&bytes[..payload_end], 0);
        if expected_file_hash != computed_file_hash {
            return Err(OspreyError::ChecksumMismatch {
                expected: expected_file_hash,
                computed: computed_file_hash,
            });
        }

        let cent_padded = centroids_padded_bytes(header.num_clusters, header.padded_dim)?;
        let cent_end = cent_off
            .checked_add(cent_padded)
            .ok_or(OspreyError::CorruptedFooter)?;
        if cent_end > ent_off {
            return Err(OspreyError::BufferTooSmall {
                needed: cent_end,
                actual: ent_off,
            });
        }
        let centroids_f16 = cast_centroids_f16(
            &bytes[cent_off..ent_off],
            header.num_clusters,
            header.padded_dim,
        )?;

        let spans_bytes = k
            .checked_mul(size_of::<OsiClusterSpan>())
            .ok_or(OspreyError::CorruptedFooter)?;
        let spans_padded = checked_align_up_64(spans_bytes).ok_or(OspreyError::CorruptedFooter)?;
        let spans_end = ent_off
            .checked_add(spans_padded)
            .ok_or(OspreyError::CorruptedFooter)?;
        if spans_end > tomb_off {
            return Err(OspreyError::BufferTooSmall {
                needed: spans_end,
                actual: tomb_off,
            });
        }

        let spans_slice = &bytes[ent_off..ent_off + spans_bytes];
        let cluster_spans: &'a [OsiClusterSpan] =
            bytemuck::try_cast_slice(spans_slice).map_err(|_| OspreyError::UnalignedMemory {
                address: spans_slice.as_ptr() as usize,
                required_align: CACHE_LINE_ALIGN,
            })?;

        let entries_section = &bytes[ent_off..tomb_off];
        let mut total_vecs = 0_u64;
        let mut total_block_bytes = 0_usize;
        for span in cluster_spans {
            let rel_off =
                usize::try_from(span.cluster_offset).map_err(|_| OspreyError::CorruptedFooter)?;
            if !rel_off.is_multiple_of(CACHE_LINE_ALIGN) || rel_off < spans_padded {
                return Err(OspreyError::UnalignedMemory {
                    address: ent_off.saturating_add(rel_off),
                    required_align: CACHE_LINE_ALIGN,
                });
            }
            let vec_count =
                usize::try_from(span.vector_count).map_err(|_| OspreyError::CorruptedFooter)?;
            let block_len = cluster_block_bytes(vec_count, header.padded_dim)?;
            let block_end = rel_off
                .checked_add(block_len)
                .ok_or(OspreyError::CorruptedFooter)?;
            if block_end > entries_section.len() {
                return Err(OspreyError::BufferTooSmall {
                    needed: ent_off.saturating_add(block_end),
                    actual: tomb_off,
                });
            }
            total_block_bytes = total_block_bytes
                .checked_add(block_len)
                .ok_or(OspreyError::CorruptedFooter)?;
            if total_block_bytes > entries_section.len() - spans_padded {
                return Err(OspreyError::CorruptedFooter);
            }
            // Validate zero-copy alignment and castability of every cluster block up front.
            let _ = OsiClusterView::from_cluster_bytes(
                &entries_section[rel_off..block_end],
                vec_count,
                header.padded_dim,
            )?;
            total_vecs = total_vecs
                .checked_add(u64::from(span.vector_count))
                .ok_or(OspreyError::CorruptedFooter)?;
        }

        if total_vecs != header.num_vectors {
            return Err(OspreyError::CorruptedFooter);
        }

        let (tombstones, consumed) = deserialize_tombstone_footer(&bytes[tomb_off..payload_end])?;
        let trailing = payload_end - (tomb_off + consumed);
        if trailing >= size_of::<u64>() {
            return Err(OspreyError::CorruptedFooter);
        }

        Ok(Self {
            header,
            centroids_f16,
            cluster_spans,
            entries_section,
            tombstones,
        })
    }

    /// Returns the validated `.osi` header.
    #[inline]
    pub fn header(&self) -> &'a OsiHeader {
        self.header
    }

    /// Returns the flat `K * padded_dim` `f16` centroid table.
    #[inline]
    pub fn centroids_f16(&self) -> &'a [f16] {
        self.centroids_f16
    }

    /// Returns the `K` cluster span descriptors.
    #[inline]
    pub fn cluster_spans(&self) -> &'a [OsiClusterSpan] {
        self.cluster_spans
    }

    /// Returns a zero-copy view over one cluster's codes and entry headers.
    #[inline]
    pub fn cluster_view(&self, cluster_id: u32) -> OspreyResult<OsiClusterView<'a>> {
        let idx = usize::try_from(cluster_id)
            .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
        let span = self
            .cluster_spans
            .get(idx)
            .ok_or(OspreyError::InvalidDimension { dim: idx })?;
        let rel_off =
            usize::try_from(span.cluster_offset).map_err(|_| OspreyError::CorruptedFooter)?;
        let vec_count =
            usize::try_from(span.vector_count).map_err(|_| OspreyError::CorruptedFooter)?;
        let block_len = cluster_block_bytes(vec_count, self.header.padded_dim)?;
        OsiClusterView::from_cluster_bytes(
            &self.entries_section[rel_off..rel_off + block_len],
            vec_count,
            self.header.padded_dim,
        )
    }

    /// Returns the deserialized tombstone `RoaringBitmap`.
    #[inline]
    pub fn tombstones(&self) -> &RoaringBitmap {
        &self.tombstones
    }

    /// Returns `true` if `doc_id` is marked deleted in the tombstone bitmap.
    #[inline]
    pub fn is_deleted(&self, doc_id: u64) -> bool {
        u32::try_from(doc_id).is_ok_and(|id| self.tombstones.contains(id))
    }
}

impl OsiIndex {
    /// Builds a cluster-grouped `.osi` index in parallel (`rayon`) from raw unrotated `f32`
    /// vectors (`[N, orig_dim]`) and a [`BinaryKMeansResult`] whose `centroids_f16` reside
    /// in FWHT-rotated space (`fwht_seed`).
    ///
    /// Each vector is sanitized, L2-normalized, zero-padded to `padded_dim`, and rotated once
    /// via `simd.fwht_rotate_in_place` before residual extraction and 1-bit `RaBitQ` packing.
    #[allow(clippy::too_many_arguments)]
    pub fn build<K: RabitqFwhtKernel + ?Sized>(
        vectors: &[f32],
        doc_ids: &[u64],
        locations: &[VectorLocation],
        kmeans: &BinaryKMeansResult,
        orig_dim: usize,
        fwht_seed: u64,
        tombstones: RoaringBitmap,
        simd: &K,
    ) -> OspreyResult<Self> {
        let padded_dim = padded_dimension(orig_dim)?;
        let n = doc_ids.len();
        let expected_len = n
            .checked_mul(orig_dim)
            .ok_or(OspreyError::InvalidDimension { dim: n })?;
        if vectors.len() < expected_len {
            return Err(OspreyError::BufferTooSmall {
                needed: expected_len * size_of::<f32>(),
                actual: std::mem::size_of_val(vectors),
            });
        }
        if vectors.len() != expected_len {
            return Err(OspreyError::InvalidDimension { dim: vectors.len() });
        }

        let signs = RademacherSigns::from_seed(padded_dim, fwht_seed)?;
        let xor_masks = signs.xor_masks();

        Self::build_with_encoder(
            0,
            doc_ids,
            locations,
            kmeans,
            orig_dim,
            padded_dim,
            fwht_seed,
            tombstones,
            |scratch, src_idx, centroid_f16, out_words| {
                let raw = &vectors[src_idx * orig_dim..(src_idx + 1) * orig_dim];
                prepare_and_pad_vector(raw, scratch)?;
                simd.fwht_rotate_in_place(scratch, xor_masks)?;
                encode_prerotated_rabitq_residual_in_place(scratch, centroid_f16, out_words)
            },
        )
    }

    /// Builds a cluster-grouped `.osi` index in parallel from already `f16`-quantized vectors
    /// (`[N, orig_dim]`) without materializing the full `f32` corpus in memory.
    ///
    /// Each vector is promoted `f16 -> f32`, L2-normalized, zero-padded to `padded_dim`, and
    /// rotated once via `simd.fwht_rotate_in_place` before residual extraction and 1-bit
    /// `RaBitQ` packing. This mirrors the exact-`f16` vectors persisted in `.osv`, keeping
    /// Stage-1 and Stage-2 consistent while bounding peak RSS during streaming import.
    #[allow(clippy::too_many_arguments)]
    pub fn build_from_f16<K: RabitqFwhtKernel + ?Sized>(
        vectors_f16: &[f16],
        doc_ids: &[u64],
        locations: &[VectorLocation],
        kmeans: &BinaryKMeansResult,
        orig_dim: usize,
        fwht_seed: u64,
        tombstones: RoaringBitmap,
        simd: &K,
    ) -> OspreyResult<Self> {
        let padded_dim = padded_dimension(orig_dim)?;
        let n = doc_ids.len();
        let expected_len = n
            .checked_mul(orig_dim)
            .ok_or(OspreyError::InvalidDimension { dim: n })?;
        if vectors_f16.len() != expected_len {
            return Err(OspreyError::InvalidDimension {
                dim: vectors_f16.len(),
            });
        }

        let signs = RademacherSigns::from_seed(padded_dim, fwht_seed)?;
        let xor_masks = signs.xor_masks();

        Self::build_with_encoder(
            0,
            doc_ids,
            locations,
            kmeans,
            orig_dim,
            padded_dim,
            fwht_seed,
            tombstones,
            |scratch, src_idx, centroid_f16, out_words| {
                let raw_f16 = &vectors_f16[src_idx * orig_dim..(src_idx + 1) * orig_dim];
                prepare_and_pad_vector_f16(raw_f16, scratch)?;
                simd.fwht_rotate_in_place(scratch, xor_masks)?;
                encode_prerotated_rabitq_residual_in_place(scratch, centroid_f16, out_words)
            },
        )
    }

    /// Builds a cluster-grouped `.osi` index in parallel from already zero-padded **unrotated**
    /// vectors (`[N, padded_dim]`) and unrotated `centroids_f16`, applying FWHT to each
    /// normalized residual (`encode_rabitq_residual`).
    #[allow(clippy::too_many_arguments)]
    pub fn build_from_padded_unrotated<K: RabitqFwhtKernel + ?Sized>(
        padded_vectors: &[f32],
        doc_ids: &[u64],
        locations: &[VectorLocation],
        kmeans: &BinaryKMeansResult,
        orig_dim: usize,
        fwht_seed: u64,
        tombstones: RoaringBitmap,
        simd: &K,
    ) -> OspreyResult<Self> {
        let padded_dim = padded_dimension(orig_dim)?;
        let n = doc_ids.len();
        let expected_len = n
            .checked_mul(padded_dim)
            .ok_or(OspreyError::InvalidDimension { dim: n })?;
        if padded_vectors.len() < expected_len {
            return Err(OspreyError::BufferTooSmall {
                needed: expected_len * size_of::<f32>(),
                actual: std::mem::size_of_val(padded_vectors),
            });
        }
        if padded_vectors.len() != expected_len {
            return Err(OspreyError::InvalidDimension {
                dim: padded_vectors.len(),
            });
        }

        let signs = RademacherSigns::from_seed(padded_dim, fwht_seed)?;

        Self::build_with_encoder(
            OSI_FLAG_UNROTATED_CENTROIDS,
            doc_ids,
            locations,
            kmeans,
            orig_dim,
            padded_dim,
            fwht_seed,
            tombstones,
            |scratch, src_idx, centroid_f16, out_words| {
                let vec_padded = &padded_vectors[src_idx * padded_dim..(src_idx + 1) * padded_dim];
                encode_rabitq_residual(vec_padded, centroid_f16, &signs, simd, scratch, out_words)
            },
        )
    }

    /// Builds a cluster-grouped `.osi` index in parallel from already FWHT-rotated `f32`
    /// vectors (`[N, padded_dim]`) and rotated `centroids_f16` (e.g., `fit_binary_kmeans_from_rotated`).
    #[allow(clippy::too_many_arguments)]
    pub fn build_from_rotated(
        rotated_vectors: &[f32],
        doc_ids: &[u64],
        locations: &[VectorLocation],
        kmeans: &BinaryKMeansResult,
        orig_dim: usize,
        fwht_seed: u64,
        tombstones: RoaringBitmap,
    ) -> OspreyResult<Self> {
        let padded_dim = padded_dimension(orig_dim)?;
        let n = doc_ids.len();
        let expected_len = n
            .checked_mul(padded_dim)
            .ok_or(OspreyError::InvalidDimension { dim: n })?;
        if rotated_vectors.len() < expected_len {
            return Err(OspreyError::BufferTooSmall {
                needed: expected_len * size_of::<f32>(),
                actual: std::mem::size_of_val(rotated_vectors),
            });
        }
        if rotated_vectors.len() != expected_len {
            return Err(OspreyError::InvalidDimension {
                dim: rotated_vectors.len(),
            });
        }

        Self::build_with_encoder(
            0,
            doc_ids,
            locations,
            kmeans,
            orig_dim,
            padded_dim,
            fwht_seed,
            tombstones,
            |scratch, src_idx, centroid_f16, out_words| {
                let vec_rot = &rotated_vectors[src_idx * padded_dim..(src_idx + 1) * padded_dim];
                encode_prerotated_rabitq_residual(vec_rot, centroid_f16, scratch, out_words)
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_with_encoder<F>(
        flags: u8,
        doc_ids: &[u64],
        locations: &[VectorLocation],
        kmeans: &BinaryKMeansResult,
        orig_dim: usize,
        padded_dim: usize,
        fwht_seed: u64,
        tombstones: RoaringBitmap,
        encode_fn: F,
    ) -> OspreyResult<Self>
    where
        F: Fn(&mut [f32], usize, &[f16], &mut [u64]) -> OspreyResult<RabitqEncodedResidual>
            + Send
            + Sync,
    {
        let n = doc_ids.len();
        if locations.len() != n {
            return Err(OspreyError::InvalidDimension {
                dim: locations.len(),
            });
        }
        if kmeans.assignments.len() != n {
            return Err(OspreyError::InvalidDimension {
                dim: kmeans.assignments.len(),
            });
        }
        if u32::try_from(n).is_err() {
            return Err(OspreyError::InvalidDimension { dim: n });
        }

        let cent_len = kmeans.centroids_f16.len();
        if cent_len == 0 || !cent_len.is_multiple_of(padded_dim) {
            return Err(OspreyError::InvalidDimension { dim: cent_len });
        }
        let num_clusters = cent_len / padded_dim;
        let k_u32 = u32::try_from(num_clusters)
            .map_err(|_| OspreyError::InvalidDimension { dim: num_clusters })?;

        for (index, &c_f16) in kmeans.centroids_f16.iter().enumerate() {
            if !c_f16.to_f32().is_finite() {
                return Err(OspreyError::NonFiniteEmbedding { index });
            }
        }

        let mut counts = vec![0_u32; num_clusters];
        for &assign in &kmeans.assignments {
            let c_idx = usize::try_from(assign)
                .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
            let slot = counts
                .get_mut(c_idx)
                .ok_or(OspreyError::InvalidDimension { dim: c_idx })?;
            *slot += 1;
        }

        let mut cluster_offsets = Vec::with_capacity(num_clusters + 1);
        cluster_offsets.push(0_u32);
        let mut running = 0_u32;
        for &count in &counts {
            running += count;
            cluster_offsets.push(running);
        }

        // Sort physical storage order by (cluster_id, minipage_id, slot_idx, doc_id) so each
        // cluster is contiguous in RAM/mmap and preserves .osv minipage locality.
        let mut order: Vec<usize> = (0..n).collect();
        #[cfg(not(miri))]
        order.par_sort_unstable_by_key(|&idx| {
            let c = kmeans.assignments[idx];
            let loc = locations[idx];
            (c, loc.minipage_id, loc.slot_idx, doc_ids[idx])
        });
        #[cfg(miri)]
        order.sort_unstable_by_key(|&idx| {
            let c = kmeans.assignments[idx];
            let loc = locations[idx];
            (c, loc.minipage_id, loc.slot_idx, doc_ids[idx])
        });

        let words_per_vec = padded_dim / 64;
        let total_words = n
            .checked_mul(words_per_vec)
            .ok_or(OspreyError::AllocationFailed {
                size: usize::MAX,
                align: CACHE_LINE_ALIGN,
            })?;
        let mut codes = AlignedBuffer::<u64>::new(total_words)?;
        let mut entries = vec![
            RabitqEntryHeader {
                doc_id: 0,
                minipage_id: 0,
                slab_id: 0,
                slot_idx: 0,
                res_norm: f16::ZERO,
                l1_norm_factor: f16::ZERO,
                _reserved: 0,
            };
            n
        ];

        let centroids_slice = kmeans.centroids_f16.as_slice();
        if n > 0 {
            #[cfg(not(miri))]
            let chunk_vecs = 256usize;
            #[cfg(not(miri))]
            order
                .par_chunks(chunk_vecs)
                .zip(entries.par_chunks_mut(chunk_vecs))
                .zip(
                    codes
                        .as_mut_slice()
                        .par_chunks_mut(chunk_vecs * words_per_vec),
                )
                .try_for_each_init(
                    || vec![0.0_f32; padded_dim],
                    |scratch, ((ord_chunk, ent_chunk), code_chunk)| -> OspreyResult<()> {
                        for ((&src_idx, dst_ent), dst_code) in ord_chunk
                            .iter()
                            .zip(ent_chunk.iter_mut())
                            .zip(code_chunk.chunks_exact_mut(words_per_vec))
                        {
                            let cluster_id = usize::try_from(kmeans.assignments[src_idx])
                                .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
                            let centroid = &centroids_slice
                                [cluster_id * padded_dim..(cluster_id + 1) * padded_dim];
                            let encoded = encode_fn(scratch, src_idx, centroid, dst_code)?;

                            let loc = locations[src_idx];
                            *dst_ent = RabitqEntryHeader {
                                doc_id: doc_ids[src_idx],
                                minipage_id: loc.minipage_id,
                                slab_id: loc.slab_id,
                                slot_idx: loc.slot_idx,
                                res_norm: encoded.res_norm,
                                l1_norm_factor: encoded.l1_norm_factor,
                                _reserved: 0,
                            };
                        }
                        Ok(())
                    },
                )?;
            #[cfg(miri)]
            {
                let mut scratch = vec![0.0_f32; padded_dim];
                for ((&src_idx, dst_ent), dst_code) in order
                    .iter()
                    .zip(entries.iter_mut())
                    .zip(codes.as_mut_slice().chunks_exact_mut(words_per_vec))
                {
                    let cluster_id = usize::try_from(kmeans.assignments[src_idx])
                        .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
                    let centroid =
                        &centroids_slice[cluster_id * padded_dim..(cluster_id + 1) * padded_dim];
                    let encoded = encode_fn(&mut scratch, src_idx, centroid, dst_code)?;

                    let loc = locations[src_idx];
                    *dst_ent = RabitqEntryHeader {
                        doc_id: doc_ids[src_idx],
                        minipage_id: loc.minipage_id,
                        slab_id: loc.slab_id,
                        slot_idx: loc.slot_idx,
                        res_norm: encoded.res_norm,
                        l1_norm_factor: encoded.l1_norm_factor,
                        _reserved: 0,
                    };
                }
            }
        }

        let centroids_f16 = AlignedBuffer::from_slice(centroids_slice)?;
        let orig_dim_u16 =
            u16::try_from(orig_dim).map_err(|_| OspreyError::InvalidDimension { dim: orig_dim })?;
        let padded_dim_u16 = u16::try_from(padded_dim)
            .map_err(|_| OspreyError::InvalidDimension { dim: padded_dim })?;
        let n_u64 = u64::try_from(n).map_err(|_| OspreyError::InvalidDimension { dim: n })?;

        let header = compute_sealed_osi_header(
            flags,
            orig_dim_u16,
            padded_dim_u16,
            k_u32,
            n_u64,
            fwht_seed,
            &cluster_offsets,
        )?;

        Ok(Self {
            header,
            centroids_f16,
            cluster_offsets,
            entries,
            codes,
            tombstones,
        })
    }

    /// Returns the validated `.osi` header.
    #[inline]
    pub fn header(&self) -> &OsiHeader {
        &self.header
    }

    /// Returns the flat `K * padded_dim` `f16` centroid table.
    #[inline]
    pub fn centroids_f16(&self) -> &[f16] {
        self.centroids_f16.as_slice()
    }

    /// Returns the `f16` centroid slice for `cluster_id`.
    #[inline]
    pub fn centroid(&self, cluster_id: u32) -> Option<&[f16]> {
        let k = usize::try_from(cluster_id).ok()?;
        let d = usize::from(self.header.padded_dim);
        let start = k.checked_mul(d)?;
        let end = start.checked_add(d)?;
        self.centroids_f16.get(start..end)
    }

    /// Returns the `K + 1` cluster offset boundaries (`cluster_offsets[k]..cluster_offsets[k + 1]`).
    #[inline]
    pub fn cluster_offsets(&self) -> &[u32] {
        &self.cluster_offsets
    }

    /// Returns all `RabitqEntryHeader` records ordered physically by cluster.
    #[inline]
    pub fn entries(&self) -> &[RabitqEntryHeader] {
        &self.entries
    }

    /// Returns the contiguous `RabitqEntryHeader` slice belonging to `cluster_id`.
    #[inline]
    pub fn cluster_entries(&self, cluster_id: u32) -> Option<&[RabitqEntryHeader]> {
        let k = usize::try_from(cluster_id).ok()?;
        let start = usize::try_from(*self.cluster_offsets.get(k)?).ok()?;
        let end = usize::try_from(*self.cluster_offsets.get(k + 1)?).ok()?;
        self.entries.get(start..end)
    }

    /// Returns the 64-byte-aligned contiguous 1-bit `u64` codes buffer across all entries.
    #[inline]
    pub fn codes(&self) -> &[u64] {
        self.codes.as_slice()
    }

    /// Returns the contiguous 1-bit `u64` code slice for all entries in `cluster_id`.
    #[inline]
    pub fn cluster_codes(&self, cluster_id: u32) -> Option<&[u64]> {
        let k = usize::try_from(cluster_id).ok()?;
        let start_idx = usize::try_from(*self.cluster_offsets.get(k)?).ok()?;
        let end_idx = usize::try_from(*self.cluster_offsets.get(k + 1)?).ok()?;
        let w = self.words_per_vec();
        self.codes
            .get(start_idx.checked_mul(w)?..end_idx.checked_mul(w)?)
    }

    /// Returns the `[u64; padded_dim / 64]` code slice for global physical index `entry_idx`.
    #[inline]
    pub fn entry_code(&self, entry_idx: usize) -> Option<&[u64]> {
        let w = self.words_per_vec();
        let start = entry_idx.checked_mul(w)?;
        let end = start.checked_add(w)?;
        self.codes.get(start..end)
    }

    /// Returns the number of `u64` words per vector (`padded_dim / 64`).
    #[inline]
    pub fn words_per_vec(&self) -> usize {
        usize::from(self.header.padded_dim) / 64
    }

    /// Returns the `RoaringBitmap` of logically deleted document IDs.
    #[inline]
    pub fn tombstones(&self) -> &RoaringBitmap {
        &self.tombstones
    }

    /// Returns mutable access to the `RoaringBitmap` tombstones.
    #[inline]
    pub fn tombstones_mut(&mut self) -> &mut RoaringBitmap {
        &mut self.tombstones
    }

    /// Returns `true` if `doc_id` is present in the tombstone bitmap.
    #[inline]
    pub fn is_deleted(&self, doc_id: u64) -> bool {
        u32::try_from(doc_id).is_ok_and(|id| self.tombstones.contains(id))
    }

    /// Serializes this `OsiIndex` into a 64-byte-aligned `.osi` binary buffer
    /// adhering strictly to Phase 3 layout, 64B section alignment, and `xxHash64` checksums.
    pub fn to_aligned_bytes(&self) -> OspreyResult<AlignedBuffer<u8>> {
        let k = usize::try_from(self.header.num_clusters)
            .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
        if self.cluster_offsets.len() != k + 1 {
            return Err(OspreyError::InvalidDimension {
                dim: self.cluster_offsets.len(),
            });
        }

        let header = compute_sealed_osi_header(
            self.header.flags,
            self.header.orig_dim,
            self.header.padded_dim,
            self.header.num_clusters,
            self.header.num_vectors,
            self.header.fwht_seed,
            &self.cluster_offsets,
        )?;

        let mut tomb_trailer = Vec::new();
        serialize_tombstone_footer(&self.tombstones, &mut tomb_trailer)?;
        let tomb_padded_len = tomb_trailer.len().next_multiple_of(size_of::<u64>());

        let tomb_off = usize::try_from(header.tombstones_offset).map_err(|_| {
            OspreyError::AllocationFailed {
                size: usize::MAX,
                align: CACHE_LINE_ALIGN,
            }
        })?;
        let total_len = tomb_off
            .checked_add(tomb_padded_len)
            .and_then(|v| v.checked_add(size_of::<u64>()))
            .ok_or(OspreyError::AllocationFailed {
                size: usize::MAX,
                align: CACHE_LINE_ALIGN,
            })?;

        let mut out = AlignedBuffer::<u8>::new(total_len)?;
        let buf = out.as_mut_slice();

        // 1. OsiHeader (64 bytes at 0x00)
        buf[..size_of::<OsiHeader>()].copy_from_slice(bytemuck::bytes_of(&header));

        // 2. Centroids table (64B aligned at centroids_offset)
        let cent_off =
            usize::try_from(header.centroids_offset).map_err(|_| OspreyError::CorruptedFooter)?;
        let ent_off =
            usize::try_from(header.entries_offset).map_err(|_| OspreyError::CorruptedFooter)?;
        let dst_centroids = cast_centroids_f16_mut(
            &mut buf[cent_off..ent_off],
            header.num_clusters,
            header.padded_dim,
        )?;
        dst_centroids.copy_from_slice(self.centroids_f16.as_slice());

        // 3. OsiClusterSpan table + Cluster-SoA blocks (64B aligned at entries_offset)
        let spans_bytes = k * size_of::<OsiClusterSpan>();
        let spans_padded = checked_align_up_64(spans_bytes).ok_or(OspreyError::CorruptedFooter)?;
        let words_per_vec = self.words_per_vec();
        let mut rel_cursor = spans_padded;

        for c in 0..k {
            let start = usize::try_from(self.cluster_offsets[c])
                .map_err(|_| OspreyError::CorruptedFooter)?;
            let end = usize::try_from(self.cluster_offsets[c + 1])
                .map_err(|_| OspreyError::CorruptedFooter)?;
            let count = end.saturating_sub(start);
            let count_u32 =
                u32::try_from(count).map_err(|_| OspreyError::InvalidDimension { dim: count })?;

            let span = OsiClusterSpan {
                cluster_offset: u64::try_from(rel_cursor)
                    .map_err(|_| OspreyError::CorruptedFooter)?,
                vector_count: count_u32,
                _reserved: 0,
            };
            let span_off = ent_off + c * size_of::<OsiClusterSpan>();
            buf[span_off..span_off + size_of::<OsiClusterSpan>()]
                .copy_from_slice(bytemuck::bytes_of(&span));

            let block_len = cluster_block_bytes(count, header.padded_dim)?;
            let block_abs = ent_off + rel_cursor;
            let codes_bytes = count * words_per_vec * size_of::<u64>();
            let entries_rel =
                checked_align_up_64(codes_bytes).ok_or(OspreyError::CorruptedFooter)?;
            let entries_bytes = count * size_of::<RabitqEntryHeader>();

            let src_codes = &self.codes.as_slice()[start * words_per_vec..end * words_per_vec];
            buf[block_abs..block_abs + codes_bytes]
                .copy_from_slice(bytemuck::cast_slice(src_codes));
            let src_entries = &self.entries[start..end];
            buf[block_abs + entries_rel..block_abs + entries_rel + entries_bytes]
                .copy_from_slice(bytemuck::cast_slice(src_entries));

            rel_cursor += block_len;
        }

        // 4. Tombstone footer + global file_xxh64
        buf[tomb_off..tomb_off + tomb_trailer.len()].copy_from_slice(&tomb_trailer);
        let payload_end = total_len - size_of::<u64>();
        let file_hash = xxh64(&buf[..payload_end], 0);
        buf[payload_end..].copy_from_slice(&file_hash.to_le_bytes());

        Ok(out)
    }

    /// Validates and opens an `.osi` byte slice (64-byte aligned, e.g. from `memmap2` or
    /// [`Self::to_aligned_bytes`]), verifying `header_xxh64`, `file_xxh64`, `tombstone_xxh64`,
    /// and every cluster block boundary before loading.
    pub fn from_bytes(bytes: &[u8]) -> OspreyResult<Self> {
        let view = OsiMappedView::from_bytes(bytes)?;
        let header = *view.header();
        let k = usize::try_from(header.num_clusters)
            .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
        let n = usize::try_from(header.num_vectors)
            .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
        let words_per_vec = usize::from(header.padded_dim) / 64;

        let centroids_f16 = AlignedBuffer::from_slice(view.centroids_f16())?;
        let mut cluster_offsets = Vec::with_capacity(k + 1);
        cluster_offsets.push(0_u32);
        let mut entries = Vec::with_capacity(n);
        let mut codes = AlignedBuffer::<u64>::new(n.checked_mul(words_per_vec).ok_or(
            OspreyError::AllocationFailed {
                size: usize::MAX,
                align: CACHE_LINE_ALIGN,
            },
        )?)?;

        let mut running = 0_u32;
        let mut code_cursor = 0usize;
        for c in 0..k {
            let c_u32 = u32::try_from(c).map_err(|_| OspreyError::InvalidDimension { dim: c })?;
            let c_view = view.cluster_view(c_u32)?;
            entries.extend_from_slice(c_view.entries);
            let len_words = c_view.codes.len();
            codes.as_mut_slice()[code_cursor..code_cursor + len_words]
                .copy_from_slice(c_view.codes);
            code_cursor += len_words;

            let cnt_u32 =
                u32::try_from(c_view.entries.len()).map_err(|_| OspreyError::CorruptedFooter)?;
            running = running
                .checked_add(cnt_u32)
                .ok_or(OspreyError::CorruptedFooter)?;
            cluster_offsets.push(running);
        }

        Ok(Self {
            header,
            centroids_f16,
            cluster_offsets,
            entries,
            codes,
            tombstones: view.tombstones().clone(),
        })
    }

    /// Reconstructs the approximate Stage-1 inner product for an entry against rotated query
    /// `q_rot` (`scale` and `dot_int4_sign`) for testing and scalar verification.
    #[inline]
    pub fn reconstruct_asymmetric_score(
        centroid_dot: f32,
        entry: &RabitqEntryHeader,
        q_scale: f32,
        dot_int4_sign: i32,
        padded_dim: u16,
    ) -> f32 {
        let denom =
            f32::from(padded_dim) * entry.l1_norm_factor.to_f32().max(RABITQ_L1_MIN_NORMAL_F32);
        #[allow(clippy::cast_precision_loss)]
        let scale_factor = (q_scale * (dot_int4_sign as f32)) / denom;
        centroid_dot + entry.res_norm.to_f32() * scale_factor
    }

    /// Returns unique `(slab_id, minipage_id, slot_idx)` locators across all entries (diagnostics).
    pub fn unique_minipage_count(&self) -> usize {
        let mut seen = HashSet::with_capacity(self.entries.len() / 8 + 1);
        for e in &self.entries {
            seen.insert((e.slab_id, e.minipage_id));
        }
        seen.len()
    }

    /// Selects the top `nprobe` clusters (`1 <= effective_nprobe <= K`) maximizing exact
    /// `f16`-promoted inner product `<q_vec, c_k>` using a stack-allocated `BoundedTopKHeap`.
    pub fn probe_clusters(&self, q_vec: &[f32], nprobe: usize) -> OspreyResult<Vec<ProbedCluster>> {
        let padded_dim = usize::from(self.header.padded_dim);
        if q_vec.len() != padded_dim {
            return Err(OspreyError::InvalidDimension { dim: q_vec.len() });
        }
        for (index, &val) in q_vec.iter().enumerate() {
            if !val.is_finite() {
                return Err(OspreyError::NonFiniteEmbedding { index });
            }
        }
        let k_total = usize::try_from(self.header.num_clusters)
            .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
        let eff_nprobe = nprobe.min(k_total);
        if eff_nprobe == 0 {
            return Ok(Vec::new());
        }

        if eff_nprobe <= 512 {
            self.probe_clusters_with_cap::<512>(q_vec, k_total, eff_nprobe)
        } else if eff_nprobe <= 4096 {
            self.probe_clusters_with_cap::<4096>(q_vec, k_total, eff_nprobe)
        } else {
            Err(OspreyError::InvalidDimension { dim: eff_nprobe })
        }
    }

    fn probe_clusters_with_cap<const CAP: usize>(
        &self,
        q_vec: &[f32],
        k_total: usize,
        eff_nprobe: usize,
    ) -> OspreyResult<Vec<ProbedCluster>> {
        let padded_dim = usize::from(self.header.padded_dim);
        let mut heap = BoundedTopKHeap::<CAP>::try_new(eff_nprobe)?;
        let centroids = self.centroids_f16.as_slice();

        for k in 0..k_total {
            let centroid = &centroids[k * padded_dim..(k + 1) * padded_dim];
            let mut dot = 0.0_f32;
            for (&q_val, &c_f16) in q_vec.iter().zip(centroid.iter()) {
                dot += q_val * c_f16.to_f32();
            }
            let k_u32 = u32::try_from(k).map_err(|_| OspreyError::InvalidDimension { dim: k })?;
            heap.push(ScoredCandidate {
                score: dot,
                doc_id: u64::from(k_u32),
                entry_idx: k_u32,
                minipage_id: 0,
                slot_idx: 0,
            });
        }

        Ok(heap
            .into_sorted_slice()
            .iter()
            .map(|c| (c.entry_idx, c.score))
            .collect())
    }

    /// Normalizes and zero-pads a raw `query` (`orig_dim`) to `q_padded` and selects the top
    /// `nprobe` clusters maximizing `<q_padded, c_k>`.
    pub fn prepare_and_probe_clusters(
        &self,
        query: &[f32],
        nprobe: usize,
    ) -> OspreyResult<(Vec<f32>, Vec<ProbedCluster>)> {
        let orig_dim = usize::from(self.header.orig_dim);
        let padded_dim = usize::from(self.header.padded_dim);
        if query.len() != orig_dim {
            return Err(OspreyError::InvalidDimension { dim: query.len() });
        }

        let mut q_padded = vec![0.0_f32; padded_dim];
        prepare_and_pad_vector(query, &mut q_padded)?;
        let probed = self.probe_clusters(&q_padded, nprobe)?;
        Ok((q_padded, probed))
    }

    /// Executes Stage-1 asymmetric `RaBitQ` search.
    pub fn search_candidates<K: RabitqScanKernel + ?Sized>(
        &self,
        query: &[f32],
        nprobe: usize,
        top_m: usize,
        simd: &K,
    ) -> OspreyResult<Vec<ScoredCandidate>> {
        self.search_candidates_filtered(query, nprobe, top_m, simd, None)
    }

    /// Executes Stage-1 asymmetric `RaBitQ` search combining `RoaringBitmap` tombstones with
    /// an optional `allowed_docs: Option<&RoaringBitmap>` scalar filter bitmap in $O(1)$ per entry.
    pub fn search_candidates_filtered<K: RabitqScanKernel + ?Sized>(
        &self,
        query: &[f32],
        nprobe: usize,
        top_m: usize,
        simd: &K,
        allowed_docs: Option<&RoaringBitmap>,
    ) -> OspreyResult<Vec<ScoredCandidate>> {
        self.search_candidates_filtered_with(query, nprobe, top_m, simd, |_entry_idx, doc_id| {
            allowed_docs.is_none_or(|bm| u32::try_from(doc_id).is_ok_and(|id| bm.contains(id)))
        })
    }

    /// Executes Stage-1 asymmetric `RaBitQ` search using a caller-supplied $O(1)$ entry/doc
    /// predicate (`allow_entry(entry_idx, doc_id)`) combined with `.osi` tombstones.
    pub fn search_candidates_filtered_with<K: RabitqScanKernel + ?Sized, F>(
        &self,
        query: &[f32],
        nprobe: usize,
        top_m: usize,
        simd: &K,
        allow_entry: F,
    ) -> OspreyResult<Vec<ScoredCandidate>>
    where
        F: Fn(u32, u64) -> bool,
    {
        let orig_dim = usize::from(self.header.orig_dim);
        let padded_dim = usize::from(self.header.padded_dim);
        if query.len() != orig_dim {
            return Err(OspreyError::InvalidDimension { dim: query.len() });
        }

        let mut q_buf = vec![0.0_f32; padded_dim];
        prepare_and_pad_vector(query, &mut q_buf)?;

        if nprobe == 0 || top_m == 0 || self.header.num_vectors == 0 {
            return Ok(Vec::new());
        }

        let signs = RademacherSigns::from_seed(padded_dim, self.header.fwht_seed)?;
        let probed = if (self.header.flags & OSI_FLAG_UNROTATED_CENTROIDS) != 0 {
            // Centroids stored in unrotated padded space: probe with `q_padded`, then rotate for residuals.
            let p = self.probe_clusters(&q_buf, nprobe)?;
            simd.fwht_rotate_in_place(&mut q_buf, signs.xor_masks())?;
            p
        } else {
            // Canonical `.osi`: centroids & residuals share FWHT-rotated space (`ARCHITECTURE.md` §3.3).
            simd.fwht_rotate_in_place(&mut q_buf, signs.xor_masks())?;
            self.probe_clusters(&q_buf, nprobe)?
        };

        let q_transposed = TransposedQuery4Bit::try_from_rotated_query(&q_buf)?;

        if top_m <= 512 {
            self.scan_probed_clusters::<512, K, F>(
                &probed,
                &q_transposed,
                top_m,
                simd,
                &allow_entry,
            )
        } else if top_m <= 4096 {
            self.scan_probed_clusters::<4096, K, F>(
                &probed,
                &q_transposed,
                top_m,
                simd,
                &allow_entry,
            )
        } else {
            Err(OspreyError::InvalidDimension { dim: top_m })
        }
    }

    fn scan_probed_clusters<const CAP: usize, K: RabitqScanKernel + ?Sized, F>(
        &self,
        probed: &[(u32, f32)],
        q_transposed: &TransposedQuery4Bit,
        top_m: usize,
        simd: &K,
        allow_entry: &F,
    ) -> OspreyResult<Vec<ScoredCandidate>>
    where
        F: Fn(u32, u64) -> bool,
    {
        let mut heap = BoundedTopKHeap::<CAP>::try_new(top_m)?;
        let words_per_vec = self.words_per_vec();
        let inv_dim = 1.0_f32 / f32::from(self.header.padded_dim);
        let q_scale = q_transposed.q_scale;
        let query_abs_sum = q_transposed.query_abs_sum;
        let q_planes = q_transposed.planes.as_slice();
        let q_signs = q_transposed.sign_mask.as_slice();
        let all_codes = self.codes.as_slice();

        for &(cluster_id, centroid_ip) in probed {
            let k_idx = usize::try_from(cluster_id)
                .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
            let start = usize::try_from(
                *self
                    .cluster_offsets
                    .get(k_idx)
                    .ok_or(OspreyError::InvalidDimension { dim: k_idx })?,
            )
            .map_err(|_| OspreyError::CorruptedFooter)?;
            let end = usize::try_from(
                *self
                    .cluster_offsets
                    .get(k_idx + 1)
                    .ok_or(OspreyError::InvalidDimension { dim: k_idx + 1 })?,
            )
            .map_err(|_| OspreyError::CorruptedFooter)?;

            const PREFETCH_AHEAD: usize = 2;
            for entry_idx in start..end {
                if entry_idx + PREFETCH_AHEAD < end {
                    let pf_idx = entry_idx + PREFETCH_AHEAD;
                    let pf_code_start = pf_idx * words_per_vec;
                    let pf_code_end = pf_code_start + words_per_vec;
                    let pf_bytes: &[u8] =
                        bytemuck::cast_slice(&all_codes[pf_code_start..pf_code_end]);
                    simd.prefetch_l1(pf_bytes);
                    if pf_bytes.len() > 64 {
                        simd.prefetch_l1(&pf_bytes[64..]);
                    }
                    simd.prefetch_l1(bytemuck::bytes_of(&self.entries[pf_idx]));
                }

                let entry = &self.entries[entry_idx];
                let entry_idx_u32 = u32::try_from(entry_idx).unwrap_or(u32::MAX);

                // O(1) RoaringBitmap tombstone & scalar filter check prior to SIMD scan.
                if self.is_deleted(entry.doc_id) || !allow_entry(entry_idx_u32, entry.doc_id) {
                    continue;
                }

                let code_start = entry_idx * words_per_vec;
                let entry_bits = &all_codes[code_start..code_start + words_per_vec];
                let dot_int =
                    simd.rabitq_4bit_dot_u64(q_planes, q_signs, query_abs_sum, entry_bits);

                // Unbiased asymmetric RaBitQ estimator reconstruction.
                let res_norm = entry.res_norm.to_f32();
                let l1_factor = entry.l1_norm_factor.to_f32().max(RABITQ_L1_MIN_NORMAL_F32);
                #[allow(clippy::cast_precision_loss)]
                let est_score =
                    centroid_ip + res_norm * (((dot_int as f32) * q_scale) * inv_dim / l1_factor);

                heap.push(ScoredCandidate {
                    score: est_score,
                    doc_id: entry.doc_id,
                    entry_idx: entry_idx_u32,
                    minipage_id: entry.minipage_id,
                    slot_idx: entry.slot_idx,
                });
            }
        }

        Ok(heap.into_sorted_slice().to_vec())
    }
}

fn compute_sealed_osi_header(
    flags: u8,
    orig_dim: u16,
    padded_dim: u16,
    num_clusters: u32,
    num_vectors: u64,
    fwht_seed: u64,
    cluster_offsets: &[u32],
) -> OspreyResult<OsiHeader> {
    let centroids_offset = u64::try_from(size_of::<OsiHeader>()).unwrap_or(64);
    let cent_padded = centroids_padded_bytes(num_clusters, padded_dim)?;
    let entries_offset = centroids_offset
        .checked_add(
            u64::try_from(cent_padded).map_err(|_| OspreyError::AllocationFailed {
                size: cent_padded,
                align: CACHE_LINE_ALIGN,
            })?,
        )
        .ok_or(OspreyError::AllocationFailed {
            size: usize::MAX,
            align: CACHE_LINE_ALIGN,
        })?;

    let k = usize::try_from(num_clusters)
        .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
    let spans_bytes =
        k.checked_mul(size_of::<OsiClusterSpan>())
            .ok_or(OspreyError::AllocationFailed {
                size: usize::MAX,
                align: CACHE_LINE_ALIGN,
            })?;
    let mut entries_section_bytes =
        checked_align_up_64(spans_bytes).ok_or(OspreyError::AllocationFailed {
            size: usize::MAX,
            align: CACHE_LINE_ALIGN,
        })?;

    for c in 0..k {
        let count = usize::try_from(cluster_offsets[c + 1] - cluster_offsets[c])
            .map_err(|_| OspreyError::CorruptedFooter)?;
        let block_len = cluster_block_bytes(count, padded_dim)?;
        entries_section_bytes =
            entries_section_bytes
                .checked_add(block_len)
                .ok_or(OspreyError::AllocationFailed {
                    size: usize::MAX,
                    align: CACHE_LINE_ALIGN,
                })?;
    }

    let tombstones_offset = entries_offset
        .checked_add(u64::try_from(entries_section_bytes).map_err(|_| {
            OspreyError::AllocationFailed {
                size: entries_section_bytes,
                align: CACHE_LINE_ALIGN,
            }
        })?)
        .ok_or(OspreyError::AllocationFailed {
            size: usize::MAX,
            align: CACHE_LINE_ALIGN,
        })?;

    let header = OsiHeader::new_sealed(
        flags,
        orig_dim,
        padded_dim,
        num_clusters,
        num_vectors,
        fwht_seed,
        centroids_offset,
        entries_offset,
        tombstones_offset,
    )?;

    debug_assert_eq!(header.magic, OSPREY_MAGIC);
    debug_assert_eq!(header.format_kind, FORMAT_KIND_OSI);
    debug_assert_eq!(header.version, FORMAT_VERSION_V1);

    Ok(header)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::math::SplitMix64;

    fn make_synthetic_vectors(n: usize, orig_dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = SplitMix64::new(seed);
        let mut out = Vec::with_capacity(n * orig_dim);
        for _ in 0..n {
            let mut sq = 0.0_f64;
            let start = out.len();
            for d in 0..orig_dim {
                let raw = rng.next_u64();
                // Map u64 deterministically into [-1.0, 1.0] with anisotropic decay
                #[allow(clippy::cast_precision_loss)]
                let unit = ((raw >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64)) * 2.0 - 1.0;
                #[allow(clippy::cast_precision_loss)]
                let scale = 1.0 / (1.0 + (d as f64) * 0.15).sqrt();
                let val = unit * scale;
                sq += val * val;
                #[allow(clippy::cast_possible_truncation)]
                out.push(val as f32);
            }
            let inv = 1.0_f64 / sq.sqrt();
            for v in &mut out[start..] {
                #[allow(clippy::cast_possible_truncation)]
                {
                    *v = (f64::from(*v) * inv) as f32;
                }
            }
        }
        out
    }

    #[test]
    fn residual_extraction_normalization_fwht_and_l1_packing() {
        let padded_dim = 256usize;
        let fwht_seed = 0x4F53_5052_0006;
        let signs = RademacherSigns::from_seed(padded_dim, fwht_seed).unwrap();
        let simd = ScalarRabitqKernel;

        let x = make_synthetic_vectors(1, padded_dim, 0x1111);
        let c_raw = make_synthetic_vectors(1, padded_dim, 0x2222);
        let c_f16: Vec<f16> = c_raw.iter().map(|&v| f16::from_f32(v * 0.5)).collect();

        let mut scratch = vec![0.0_f32; padded_dim];
        let mut code_words = vec![0_u64; padded_dim / 64];
        let encoded =
            encode_rabitq_residual(&x, &c_f16, &signs, &simd, &mut scratch, &mut code_words)
                .unwrap();

        assert!(encoded.res_norm.to_f32() > 0.1);
        assert!(encoded.res_norm.to_f32() <= RABITQ_F16_MAX_CLAMP);
        assert!(encoded.l1_norm_factor.to_f32() >= RABITQ_L1_MIN_NORMAL_F32);
        // Under FWHT delocalization of a unit residual, E[|v_i|] ~= sqrt(2 / (pi * d))
        // wait: ||v||_2 = 1 => E[|v_i|] ~= sqrt(2/pi) / sqrt(d)
        let expected_l1_mean = (2.0_f32 / std::f32::consts::PI).sqrt() / (padded_dim as f32).sqrt();
        assert!(
            (encoded.l1_norm_factor.to_f32() - expected_l1_mean).abs() < 0.02,
            "l1={}, expected~={}",
            encoded.l1_norm_factor.to_f32(),
            expected_l1_mean
        );

        // Verify Little-Endian bit packing parity directly against `scratch[i] >= 0.0`
        for (i, &val) in scratch.iter().enumerate() {
            let packed_bit = (code_words[i / 64] >> (i % 64)) & 1;
            let expected_bit = u64::from(val >= 0.0);
            assert_eq!(packed_bit, expected_bit, "bit mismatch at {i}");
        }

        // Verify mathematical inner product unbiasedness:
        // <q, r> ~= res_norm * (sum_i q_rot[i] * sign(v_i)) / (D * l1_norm_factor)
        let q = make_synthetic_vectors(1, padded_dim, 0x3333);
        let mut q_rot = q.clone();
        simd.fwht_rotate_in_place(&mut q_rot, signs.xor_masks())
            .unwrap();

        let exact_qr: f64 = q
            .iter()
            .zip(x.iter().zip(c_f16.iter()))
            .map(|(&qi, (&xi, &ci))| f64::from(qi) * f64::from(xi - ci.to_f32()))
            .sum();

        let mut dot_q_sign = 0.0_f64;
        for (i, &qri) in q_rot.iter().enumerate() {
            let bit = (code_words[i / 64] >> (i % 64)) & 1;
            let s = if bit == 1 { 1.0_f64 } else { -1.0_f64 };
            dot_q_sign += f64::from(qri) * s;
        }
        let est_qr = f64::from(encoded.res_norm.to_f32())
            * (dot_q_sign / (padded_dim as f64 * f64::from(encoded.l1_norm_factor.to_f32())));
        assert!(
            (est_qr - exact_qr).abs() < 0.15,
            "exact={exact_qr}, est={est_qr}"
        );
    }

    #[test]
    fn zero_residual_and_non_finite_handling() {
        let padded_dim = 64usize;
        let signs = RademacherSigns::from_seed(padded_dim, 42).unwrap();
        let simd = ScalarRabitqKernel;

        let c_f16: Vec<f16> = (0..padded_dim)
            .map(|i| f16::from_f32((i as f32) * 0.01))
            .collect();
        let x_exact: Vec<f32> = c_f16.iter().map(|c| c.to_f32()).collect();

        let mut scratch = vec![0.0_f32; padded_dim];
        let mut words = vec![0_u64; 1];
        let res = encode_rabitq_residual(&x_exact, &c_f16, &signs, &simd, &mut scratch, &mut words)
            .unwrap();
        assert_eq!(res.res_norm, f16::ZERO);
        assert!(res.l1_norm_factor.to_f32() >= RABITQ_L1_MIN_NORMAL_F32);
        assert!(res.l1_norm_factor.is_finite());

        // Non-finite input must fail immediately with NonFiniteEmbedding
        let mut bad = x_exact;
        bad[17] = f32::NAN;
        let err = encode_rabitq_residual(&bad, &c_f16, &signs, &simd, &mut scratch, &mut words)
            .unwrap_err();
        assert!(matches!(err, OspreyError::NonFiniteEmbedding { index: 17 }));

        bad[17] = f32::INFINITY;
        let err_inf = encode_rabitq_residual(&bad, &c_f16, &signs, &simd, &mut scratch, &mut words)
            .unwrap_err();
        assert!(matches!(
            err_inf,
            OspreyError::NonFiniteEmbedding { index: 17 }
        ));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn osi_index_build_sort_serialize_roundtrip_and_corruption_detection() {
        let n = 128usize;
        let orig_dim = 384usize;
        let padded_dim = padded_dimension(orig_dim).unwrap();
        assert_eq!(padded_dim, 512);
        let k = 8usize;
        let fwht_seed = 0x4F53_5052_0006;
        let simd = ScalarRabitqKernel;

        let vectors = make_synthetic_vectors(n, orig_dim, 0xCAFE_0001);
        let signs = RademacherSigns::from_seed(padded_dim, fwht_seed).unwrap();

        // Build rotated centroids & assignments intentionally including an empty cluster (cluster 5)
        let mut centroids_f16 = AlignedBuffer::<f16>::new(k * padded_dim).unwrap();
        let mut scratch = vec![0.0_f32; padded_dim];
        for c in 0..k {
            prepare_and_pad_vector(&vectors[c * orig_dim..(c + 1) * orig_dim], &mut scratch)
                .unwrap();
            simd.fwht_rotate_in_place(&mut scratch, signs.xor_masks())
                .unwrap();
            for (d, &val) in scratch.iter().enumerate() {
                centroids_f16[c * padded_dim + d] = f16::from_f32(val);
            }
        }

        let mut assignments = Vec::with_capacity(n);
        let mut doc_ids = Vec::with_capacity(n);
        let mut locations = Vec::with_capacity(n);
        for i in 0..n {
            // Skip cluster 5 to verify empty cluster handling
            let mut c = u32::try_from(i % k).unwrap();
            if c == 5 {
                c = 2;
            }
            assignments.push(c);
            // Reverse doc_ids to verify sorting inside clusters
            doc_ids.push(u64::try_from(1000 + (n - i)).unwrap());
            locations.push(VectorLocation {
                slab_id: 0,
                minipage_id: u32::try_from(i % 4).unwrap(),
                slot_idx: u16::try_from(i / 4).unwrap(),
            });
        }

        let kmeans = BinaryKMeansResult {
            centroids_f16,
            binary_centroids: vec![0_u64; k * (padded_dim / 64)],
            assignments: assignments.clone(),
            final_hamming_inertia: 0,
            iterations_run: 1,
        };

        let mut tombstones = RoaringBitmap::new();
        tombstones.insert(1005);
        tombstones.insert(1042);

        let index = OsiIndex::build(
            &vectors,
            &doc_ids,
            &locations,
            &kmeans,
            orig_dim,
            fwht_seed,
            tombstones.clone(),
            &simd,
        )
        .unwrap();

        // Verify cluster offsets & empty cluster
        assert_eq!(index.cluster_offsets().len(), k + 1);
        assert_eq!(index.cluster_offsets()[0], 0);
        assert_eq!(index.cluster_offsets()[k], u32::try_from(n).unwrap());
        assert_eq!(index.cluster_entries(5).unwrap().len(), 0);
        assert_eq!(index.cluster_codes(5).unwrap().len(), 0);
        assert_eq!((index.codes().as_ptr() as usize) % 64, 0);
        assert!(index.is_deleted(1005));
        assert!(index.is_deleted(1042));
        assert!(!index.is_deleted(1006));

        // Verify intra-cluster sorting by (minipage_id, slot_idx, doc_id)
        for c in 0..u32::try_from(k).unwrap() {
            let ents = index.cluster_entries(c).unwrap();
            for win in ents.windows(2) {
                let a = (win[0].minipage_id, win[0].slot_idx, win[0].doc_id);
                let b = (win[1].minipage_id, win[1].slot_idx, win[1].doc_id);
                assert!(a <= b, "cluster {c} not sorted: {a:?} > {b:?}");
            }
        }

        // Serialize and re-open zero-copy & owned
        let bytes = index.to_aligned_bytes().unwrap();
        assert_eq!((bytes.as_ptr() as usize) % 64, 0);

        let mapped = OsiMappedView::from_bytes(&bytes).unwrap();
        assert_eq!(mapped.header(), index.header());
        assert_eq!(mapped.centroids_f16(), index.centroids_f16());
        assert_eq!(mapped.tombstones(), &tombstones);

        let reopened = OsiIndex::from_bytes(&bytes).unwrap();
        assert_eq!(reopened.header(), index.header());
        assert_eq!(reopened.centroids_f16(), index.centroids_f16());
        assert_eq!(reopened.cluster_offsets(), index.cluster_offsets());
        assert_eq!(reopened.entries(), index.entries());
        assert_eq!(reopened.codes(), index.codes());
        assert_eq!(reopened.tombstones(), index.tombstones());

        // Unaligned slice rejection
        let unaligned_err = OsiIndex::from_bytes(&bytes[1..]).unwrap_err();
        assert!(matches!(
            unaligned_err,
            OspreyError::UnalignedMemory {
                required_align: 64,
                ..
            }
        ));

        // Payload corruption -> global file_xxh64 ChecksumMismatch
        let mut corrupted = AlignedBuffer::from_slice(&bytes).unwrap();
        let cent_off = usize::try_from(index.header().centroids_offset).unwrap();
        corrupted[cent_off + 3] ^= 0x01;
        let err = OsiIndex::from_bytes(&corrupted).unwrap_err();
        assert!(matches!(err, OspreyError::ChecksumMismatch { .. }));

        // Header corruption -> ChecksumMismatch
        let mut hdr_corrupted = AlignedBuffer::from_slice(&bytes).unwrap();
        hdr_corrupted[20] ^= 0x80;
        let hdr_err = OsiIndex::from_bytes(&hdr_corrupted).unwrap_err();
        assert!(matches!(hdr_err, OspreyError::ChecksumMismatch { .. }));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn golden_rabitq_encoding_and_osi_build_parity() {
        let skm_raw: &[u8] = include_bytes!("../../../reference/golden/binary_kmeans_golden.bin");
        let rab_raw: &[u8] = include_bytes!("../../../reference/golden/rabitq_golden.bin");

        let n = 512usize;
        let dim = 128usize;
        let k = 16usize;
        let words_per_vec = dim / 64;

        // Read rotated DB vectors from binary_kmeans_golden.bin [0 .. 512*128*4]
        let mut db_rot = vec![0.0_f32; n * dim];
        for (i, dst) in db_rot.iter_mut().enumerate() {
            let off = i * 4;
            *dst = f32::from_le_bytes(skm_raw[off..off + 4].try_into().unwrap());
        }

        // Read cluster_id, res_norm, l1_norm_factor, rabitq_code, centroids_f16 from rabitq_golden.bin
        let mut cluster_ids = vec![0_u32; n];
        for (i, dst) in cluster_ids.iter_mut().enumerate() {
            let off = i * 4;
            *dst = u32::from_le_bytes(rab_raw[off..off + 4].try_into().unwrap());
        }

        let mut golden_res_norm = vec![f16::ZERO; n];
        for (i, dst) in golden_res_norm.iter_mut().enumerate() {
            let off = 2048 + i * 2;
            *dst = f16::from_le_bytes(rab_raw[off..off + 2].try_into().unwrap());
        }

        let mut golden_l1 = vec![f16::ZERO; n];
        for (i, dst) in golden_l1.iter_mut().enumerate() {
            let off = 3072 + i * 2;
            *dst = f16::from_le_bytes(rab_raw[off..off + 2].try_into().unwrap());
        }

        let mut golden_codes = vec![0_u64; n * words_per_vec];
        for (i, dst) in golden_codes.iter_mut().enumerate() {
            let off = 4096 + i * 8;
            *dst = u64::from_le_bytes(rab_raw[off..off + 8].try_into().unwrap());
        }

        let mut centroids_f16 = AlignedBuffer::<f16>::new(k * dim).unwrap();
        for (i, dst) in centroids_f16.iter_mut().enumerate() {
            let off = 143_360 + i * 2;
            *dst = f16::from_le_bytes(rab_raw[off..off + 2].try_into().unwrap());
        }

        let mut scratch = vec![0.0_f32; dim];
        let mut out_words = vec![0_u64; words_per_vec];
        for i in 0..n {
            let c = usize::try_from(cluster_ids[i]).unwrap();
            let enc = encode_prerotated_rabitq_residual(
                &db_rot[i * dim..(i + 1) * dim],
                &centroids_f16[c * dim..(c + 1) * dim],
                &mut scratch,
                &mut out_words,
            )
            .unwrap();

            assert!(
                (enc.res_norm.to_f32() - golden_res_norm[i].to_f32()).abs() <= 1e-3,
                "vec {i} res_norm mismatch: got {}, golden {}",
                enc.res_norm.to_f32(),
                golden_res_norm[i].to_f32()
            );
            assert!(
                (enc.l1_norm_factor.to_f32() - golden_l1[i].to_f32()).abs() <= 1e-3,
                "vec {i} l1 mismatch: got {}, golden {}",
                enc.l1_norm_factor.to_f32(),
                golden_l1[i].to_f32()
            );
            assert_eq!(
                &out_words[..],
                &golden_codes[i * words_per_vec..(i + 1) * words_per_vec],
                "vec {i} rabitq_code bit mismatch"
            );
        }

        // Also build full OsiIndex from rotated golden inputs and check cluster grouping + roundtrip
        let doc_ids: Vec<u64> = (0..n).map(|i| u64::try_from(i).unwrap()).collect();
        let locations = vec![VectorLocation::default(); n];
        let kmeans = BinaryKMeansResult {
            centroids_f16,
            binary_centroids: vec![0_u64; k * words_per_vec],
            assignments: cluster_ids.clone(),
            final_hamming_inertia: 0,
            iterations_run: 12,
        };
        let osi = OsiIndex::build_from_rotated(
            &db_rot,
            &doc_ids,
            &locations,
            &kmeans,
            dim,
            0x4F53_5052_0002,
            RoaringBitmap::new(),
        )
        .unwrap();
        let bytes = osi.to_aligned_bytes().unwrap();
        let reopened = OsiIndex::from_bytes(&bytes).unwrap();
        assert_eq!(reopened.entries().len(), n);
        for ent in reopened.entries() {
            let orig_idx = usize::try_from(ent.doc_id).unwrap();
            assert_eq!(
                reopened
                    .entry_code(
                        reopened
                            .entries()
                            .iter()
                            .position(|e| e.doc_id == ent.doc_id)
                            .unwrap()
                    )
                    .unwrap(),
                &golden_codes[orig_idx * words_per_vec..(orig_idx + 1) * words_per_vec]
            );
        }
    }

    #[test]
    fn build_from_padded_unrotated_and_empty_index_roundtrip() {
        let orig_dim = 64usize;
        let padded_dim = 64usize;
        let k = 4usize;
        let n = 16usize;
        let fwht_seed = 0x4F53_5052_9999;
        let simd = ScalarRabitqKernel;

        let padded_vecs = make_synthetic_vectors(n, padded_dim, 0x7777);
        let centroids_f16 = AlignedBuffer::from_slice(
            &padded_vecs[..k * padded_dim]
                .iter()
                .map(|&v| f16::from_f32(v))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let kmeans = BinaryKMeansResult {
            centroids_f16,
            binary_centroids: vec![0_u64; k],
            assignments: (0..n).map(|i| u32::try_from(i % k).unwrap()).collect(),
            final_hamming_inertia: 0,
            iterations_run: 1,
        };
        let doc_ids: Vec<u64> = (0..n).map(|i| u64::try_from(i + 10).unwrap()).collect();
        let locations = vec![VectorLocation::default(); n];

        let idx = OsiIndex::build_from_padded_unrotated(
            &padded_vecs,
            &doc_ids,
            &locations,
            &kmeans,
            orig_dim,
            fwht_seed,
            RoaringBitmap::new(),
            &simd,
        )
        .unwrap();
        let bytes = idx.to_aligned_bytes().unwrap();
        let rt = OsiIndex::from_bytes(&bytes).unwrap();
        assert_eq!(rt.entries().len(), n);
        let score = OsiIndex::reconstruct_asymmetric_score(
            0.5,
            &rt.entries()[0],
            0.1,
            12,
            rt.header().padded_dim,
        );
        assert!(score.is_finite());

        // Empty N=0, K=4 index roundtrip
        let empty_kmeans = BinaryKMeansResult {
            centroids_f16: AlignedBuffer::<f16>::new(k * padded_dim).unwrap(),
            binary_centroids: vec![0_u64; k],
            assignments: Vec::new(),
            final_hamming_inertia: 0,
            iterations_run: 0,
        };
        let empty_idx = OsiIndex::build(
            &[],
            &[],
            &[],
            &empty_kmeans,
            orig_dim,
            fwht_seed,
            RoaringBitmap::new(),
            &simd,
        )
        .unwrap();
        assert_eq!(empty_idx.cluster_offsets(), &[0, 0, 0, 0, 0]);
        let empty_bytes = empty_idx.to_aligned_bytes().unwrap();
        let empty_rt = OsiIndex::from_bytes(&empty_bytes).unwrap();
        assert_eq!(empty_rt.entries().len(), 0);
        assert_eq!(empty_rt.cluster_offsets(), &[0, 0, 0, 0, 0]);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn transposed_query_4bit_matches_naive_and_golden_rabitq() {
        let rab_raw: &[u8] = include_bytes!("../../../reference/golden/rabitq_golden.bin");
        let n = 512usize;
        let dim = 128usize;
        let k = 16usize;
        let n_queries = 16usize;
        let words_per_vec = dim / 64;

        let mut cluster_ids = vec![0_u32; n];
        for (i, dst) in cluster_ids.iter_mut().enumerate() {
            *dst = u32::from_le_bytes(rab_raw[i * 4..(i + 1) * 4].try_into().unwrap());
        }
        let mut res_norms = vec![f16::ZERO; n];
        for (i, dst) in res_norms.iter_mut().enumerate() {
            let off = 2048 + i * 2;
            *dst = f16::from_le_bytes(rab_raw[off..off + 2].try_into().unwrap());
        }
        let mut l1_factors = vec![f16::ZERO; n];
        for (i, dst) in l1_factors.iter_mut().enumerate() {
            let off = 3072 + i * 2;
            *dst = f16::from_le_bytes(rab_raw[off..off + 2].try_into().unwrap());
        }
        let mut codes = vec![0_u64; n * words_per_vec];
        for (i, dst) in codes.iter_mut().enumerate() {
            let off = 4096 + i * 8;
            *dst = u64::from_le_bytes(rab_raw[off..off + 8].try_into().unwrap());
        }
        let mut centroids_f16 = vec![f16::ZERO; k * dim];
        for (i, dst) in centroids_f16.iter_mut().enumerate() {
            let off = 143_360 + i * 2;
            *dst = f16::from_le_bytes(rab_raw[off..off + 2].try_into().unwrap());
        }

        for qi in 0..n_queries {
            let mut q_rot = vec![0.0_f32; dim];
            for (d, dst) in q_rot.iter_mut().enumerate() {
                let off = 151_552 + (qi * dim + d) * 4;
                *dst = f32::from_le_bytes(rab_raw[off..off + 4].try_into().unwrap());
            }
            let q_scale_off = 159_744 + qi * 4;
            let golden_q_scale =
                f32::from_le_bytes(rab_raw[q_scale_off..q_scale_off + 4].try_into().unwrap());

            let mut golden_q_int4 = vec![0_i8; dim];
            for (d, dst) in golden_q_int4.iter_mut().enumerate() {
                let off = 159_808 + qi * dim + d;
                *dst = i8::from_le_bytes([rab_raw[off]]);
            }

            let tq = TransposedQuery4Bit::try_from_rotated_query(&q_rot).unwrap();
            assert!((tq.q_scale - golden_q_scale).abs() <= 1e-6);

            let tq_golden_int4 = TransposedQuery4Bit::from_quantized_i8(
                &golden_q_int4,
                golden_q_scale,
                q_rot.clone(),
            )
            .unwrap();

            // Verify exact algebraic identity against naive dot product on all 512 DB codes
            let mut c_dots = vec![0.0_f32; k];
            for c in 0..k {
                let mut acc = 0.0_f64;
                for d in 0..dim {
                    acc += f64::from(centroids_f16[c * dim + d].to_f32()) * f64::from(q_rot[d]);
                }
                #[allow(clippy::cast_possible_truncation)]
                {
                    c_dots[c] = acc as f32;
                }
            }

            for vi in 0..n {
                let db_slice = &codes[vi * words_per_vec..(vi + 1) * words_per_vec];
                let dot_bit = tq_golden_int4.dot_1bit_scalar(db_slice);
                let mut dot_naive = 0_i32;
                for d in 0..dim {
                    let bit = (db_slice[d / 64] >> (d % 64)) & 1;
                    let sigma = if bit == 1 { 1_i32 } else { -1_i32 };
                    dot_naive += i32::from(golden_q_int4[d]) * sigma;
                }
                assert_eq!(dot_bit, dot_naive, "query {qi} vec {vi}");

                // Also verify full asymmetric score matches golden asym_scores
                let dot_from_rot = tq.dot_1bit_scalar(db_slice);
                let c_idx = usize::try_from(cluster_ids[vi]).unwrap();
                let entry = RabitqEntryHeader {
                    doc_id: 0,
                    minipage_id: 0,
                    slab_id: 0,
                    slot_idx: 0,
                    res_norm: res_norms[vi],
                    l1_norm_factor: l1_factors[vi],
                    _reserved: 0,
                };
                let est = OsiIndex::reconstruct_asymmetric_score(
                    c_dots[c_idx],
                    &entry,
                    tq.q_scale,
                    dot_from_rot,
                    u16::try_from(dim).unwrap(),
                );
                let asym_off = 162_880 + (qi * n + vi) * 4;
                let golden_score =
                    f32::from_le_bytes(rab_raw[asym_off..asym_off + 4].try_into().unwrap());
                assert!(
                    (est - golden_score).abs() <= 1e-4,
                    "q={qi} v={vi} est={est} golden={golden_score}"
                );
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn search_candidates_probes_filters_tombstones_and_retrieves_exact_nearest_neighbor() {
        let n = 256usize;
        let orig_dim = 384usize;
        let padded_dim = padded_dimension(orig_dim).unwrap();
        let k = 8usize;
        let fwht_seed = 0x4F53_5052_0089;
        let simd = ScalarRabitqKernel;

        // 1. Test canonical rotated-centroid `.osi` (`OsiIndex::build`)
        let vectors = make_synthetic_vectors(n, orig_dim, 0xBEEF_0001);
        let signs = RademacherSigns::from_seed(padded_dim, fwht_seed).unwrap();
        let mut centroids_rot_f16 = AlignedBuffer::<f16>::new(k * padded_dim).unwrap();
        let mut scratch = vec![0.0_f32; padded_dim];
        for c in 0..k {
            prepare_and_pad_vector(&vectors[c * orig_dim..(c + 1) * orig_dim], &mut scratch)
                .unwrap();
            simd.fwht_rotate_in_place(&mut scratch, signs.xor_masks())
                .unwrap();
            for (d, &val) in scratch.iter().enumerate() {
                centroids_rot_f16[c * padded_dim + d] = f16::from_f32(val);
            }
        }

        // Assign each vector to its closest centroid in rotated space
        let mut assignments = Vec::with_capacity(n);
        for i in 0..n {
            prepare_and_pad_vector(&vectors[i * orig_dim..(i + 1) * orig_dim], &mut scratch)
                .unwrap();
            simd.fwht_rotate_in_place(&mut scratch, signs.xor_masks())
                .unwrap();
            let mut best_c = 0_u32;
            let mut best_dot = f32::NEG_INFINITY;
            for c in 0..k {
                let mut dot = 0.0_f32;
                for (d, &sv) in scratch.iter().enumerate() {
                    dot += sv * centroids_rot_f16[c * padded_dim + d].to_f32();
                }
                if dot > best_dot {
                    best_dot = dot;
                    best_c = u32::try_from(c).unwrap();
                }
            }
            assignments.push(best_c);
        }

        let kmeans = BinaryKMeansResult {
            centroids_f16: centroids_rot_f16,
            binary_centroids: vec![0_u64; k * (padded_dim / 64)],
            assignments,
            final_hamming_inertia: 0,
            iterations_run: 1,
        };
        let doc_ids: Vec<u64> = (0..n).map(|i| u64::try_from(i + 100).unwrap()).collect();
        let locations: Vec<VectorLocation> = (0..n)
            .map(|i| VectorLocation {
                slab_id: 0,
                minipage_id: u32::try_from(i / 16).unwrap(),
                slot_idx: u16::try_from(i % 16).unwrap(),
            })
            .collect();

        // Mark doc 115 (exact match of query) as deleted via RoaringBitmap tombstone
        let mut tombstones = RoaringBitmap::new();
        tombstones.insert(115);

        let index = OsiIndex::build(
            &vectors, &doc_ids, &locations, &kmeans, orig_dim, fwht_seed, tombstones, &simd,
        )
        .unwrap();

        // Query == vector 15 (doc_id 115, deleted) -> doc 115 must NOT appear
        let query_deleted = &vectors[15 * orig_dim..(15 + 1) * orig_dim];
        let res_deleted = index
            .search_candidates(query_deleted, 4, 20, &simd)
            .unwrap();
        assert_eq!(res_deleted.len(), 20);
        assert!(res_deleted.iter().all(|c| c.doc_id != 115));
        assert!(res_deleted.windows(2).all(|w| w[0].score >= w[1].score));

        // Query == vector 42 (doc_id 142, alive) -> doc 142 must rank #1!
        let query_alive = &vectors[42 * orig_dim..(42 + 1) * orig_dim];
        let res_alive = index.search_candidates(query_alive, 4, 10, &simd).unwrap();
        assert_eq!(res_alive.len(), 10);
        assert_eq!(res_alive[0].doc_id, 142);

        // 2. Also test unrotated-centroid `.osi` (`build_from_padded_unrotated`) + `prepare_and_probe_clusters`
        let mut padded_vecs = vec![0.0_f32; n * padded_dim];
        for i in 0..n {
            prepare_and_pad_vector(
                &vectors[i * orig_dim..(i + 1) * orig_dim],
                &mut padded_vecs[i * padded_dim..(i + 1) * padded_dim],
            )
            .unwrap();
        }
        let mut unrot_centroids = AlignedBuffer::<f16>::new(k * padded_dim).unwrap();
        for c in 0..k {
            for d in 0..padded_dim {
                unrot_centroids[c * padded_dim + d] =
                    f16::from_f32(padded_vecs[c * padded_dim + d]);
            }
        }
        let unrot_kmeans = BinaryKMeansResult {
            centroids_f16: unrot_centroids,
            binary_centroids: vec![0_u64; k * (padded_dim / 64)],
            assignments: (0..n).map(|i| u32::try_from(i % k).unwrap()).collect(),
            final_hamming_inertia: 0,
            iterations_run: 1,
        };
        let unrot_index = OsiIndex::build_from_padded_unrotated(
            &padded_vecs,
            &doc_ids,
            &locations,
            &unrot_kmeans,
            orig_dim,
            fwht_seed,
            RoaringBitmap::new(),
            &simd,
        )
        .unwrap();

        let (q_pad, probed) = unrot_index
            .prepare_and_probe_clusters(query_alive, 3)
            .unwrap();
        assert_eq!(q_pad.len(), padded_dim);
        assert_eq!(probed.len(), 3);
        assert!(probed.windows(2).all(|w| w[0].1 >= w[1].1));

        let unrot_res = unrot_index
            .search_candidates(query_alive, k, 10, &simd)
            .unwrap();
        assert_eq!(unrot_res[0].doc_id, 142);
    }
}
