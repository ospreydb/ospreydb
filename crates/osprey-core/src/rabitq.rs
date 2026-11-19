//! Asymmetric `RaBitQ` quantizer (`arXiv:2405.12497`, `arXiv:2409.09913`) and `.osi` cluster-grouped index builder.
//!
//! Implements intra-cluster residual extraction, exact `f64` L2 norm computation,
//! anti-subnormal `f16` clamping, FWHT orthogonal rotation, L1 projection factor
//! computation, 1-bit Little-Endian bit-packing, and parallel `OsiIndex` construction
//! with validated zero-copy `.osi` serialization/deserialization.

use std::cell::RefCell;
use std::collections::HashSet;
use std::fs::File;
use std::mem::size_of;
use std::path::Path;

use half::f16;
use memmap2::Mmap;
#[cfg(not(miri))]
use rayon::prelude::*;
use roaring::RoaringBitmap;
use xxhash_rust::xxh64::xxh64;

use crate::aligned::{ensure_aligned_64, AlignedBuffer, CACHE_LINE_ALIGN};
use crate::cluster::BinaryKMeansResult;
use crate::error::{OspreyError, OspreyResult};
use crate::format::osi::{
    cast_centroids_f16, cast_centroids_f16_mut, centroids_padded_bytes, centroids_unpadded_bytes,
    cluster_block_bytes_for_bits, cluster_block_bytes_for_flags, deserialize_tombstone_footer,
    fastscan_block_bytes, serialize_tombstone_footer, OsiClusterSpan, OsiClusterView,
    OsiClusterView4Bit, OsiClusterViewProgressive4Bit, OsiHeader, RabitqEntryHeader,
    FASTSCAN_BLOCK_SIZE, OSI_FLAG_RABITQ_4BIT, OSI_FLAG_RABITQ_PROGRESSIVE,
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
pub const OSI_FLAG_UNROTATED_CENTROIDS: u8 = crate::format::osi::OSI_FLAG_UNROTATED_CENTROIDS;

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

    /// Computes exact `int4 x 1-bit` inner products across a batch of `out_dots.len()`
    /// contiguous `words_per_vec`-word vectors, amortizing query bit-plane loading/transposition.
    #[inline]
    fn rabitq_1bit_batch_dots_u64(
        &self,
        query_planes: &[[u64; 4]],
        query_signs: &[u64],
        query_abs_sum: i32,
        db_words_contiguous: &[u64],
        words_per_vec: usize,
        out_dots: &mut [i32],
    ) {
        if words_per_vec == 0 {
            out_dots.fill(0);
            return;
        }
        for (idx, dst) in out_dots.iter_mut().enumerate() {
            let start = idx * words_per_vec;
            *dst = self.rabitq_4bit_dot_u64(
                query_planes,
                query_signs,
                query_abs_sum,
                &db_words_contiguous[start..start + words_per_vec],
            );
        }
    }

    /// Evaluates one transposed 32-vector 4-bit `RaBitQ` FastScan (`vpshufb`) block against
    /// `[padded_dim * 16]` `u8` query lookup tables, writing 32 unsigned sums into `out_sums`.
    #[inline]
    fn rabitq_4bit_fastscan_block(
        &self,
        q_lut_u8: &[u8],
        padded_dim: usize,
        block_bytes: &[u8],
        out_sums: &mut [u32; FASTSCAN_BLOCK_SIZE],
    ) {
        rabitq_4bit_fastscan_block_scalar(q_lut_u8, padded_dim, block_bytes, out_sums);
    }

    /// Computes the exact `f32 x f16` inner product accumulated in `f32`.
    #[inline]
    fn dot_f32_f16(&self, a_f32: &[f32], b_f16: &[f16]) -> f32 {
        dot_f32_f16_scalar(a_f32, b_f16)
    }
}

/// Portable scalar evaluation of one transposed 32-vector 4-bit FastScan block.
#[inline]
pub fn rabitq_4bit_fastscan_block_scalar(
    q_lut_u8: &[u8],
    padded_dim: usize,
    block_bytes: &[u8],
    out_sums: &mut [u32; FASTSCAN_BLOCK_SIZE],
) {
    out_sums.fill(0);
    let needed = padded_dim.saturating_mul(16);
    if padded_dim == 0
        || !padded_dim.is_multiple_of(2)
        || q_lut_u8.len() < needed
        || block_bytes.len() < needed
    {
        return;
    }
    let num_pairs = padded_dim / 2;
    for m in 0..num_pairs {
        let lut_even = &q_lut_u8[(2 * m) * 16..(2 * m) * 16 + 16];
        let lut_odd = &q_lut_u8[(2 * m + 1) * 16..(2 * m + 1) * 16 + 16];
        let pair_bytes = &block_bytes[m * FASTSCAN_BLOCK_SIZE..(m + 1) * FASTSCAN_BLOCK_SIZE];
        for (j, &b) in pair_bytes.iter().enumerate() {
            let c_even = usize::from(b & 0x0F);
            let c_odd = usize::from(b >> 4);
            out_sums[j] += u32::from(lut_even[c_even]) + u32::from(lut_odd[c_odd]);
        }
    }
}

/// Portable unrolled scalar `f32 x f16` dot product accumulated in `f32`.
#[inline]
pub fn dot_f32_f16_scalar(a_f32: &[f32], b_f16: &[f16]) -> f32 {
    let n = a_f32.len().min(b_f16.len());
    let mut acc0 = 0.0_f32;
    let mut acc1 = 0.0_f32;
    let mut acc2 = 0.0_f32;
    let mut acc3 = 0.0_f32;
    let mut i = 0usize;
    while i + 4 <= n {
        acc0 += a_f32[i] * b_f16[i].to_f32();
        acc1 += a_f32[i + 1] * b_f16[i + 1].to_f32();
        acc2 += a_f32[i + 2] * b_f16[i + 2].to_f32();
        acc3 += a_f32[i + 3] * b_f16[i + 3].to_f32();
        i += 4;
    }
    let mut sum = (acc0 + acc1) + (acc2 + acc3);
    while i < n {
        sum += a_f32[i] * b_f16[i].to_f32();
        i += 1;
    }
    sum
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
    /// Quantizes `q_rot` symmetrically to `[-7, +7]` into caller-supplied reusable buffers,
    /// returning `(query_abs_sum, q_scale)` without heap reallocations once warmed up.
    pub fn quantize_into_slices(
        q_rot: &[f32],
        planes: &mut Vec<[u64; 4]>,
        sign_mask: &mut Vec<u64>,
    ) -> (i32, f32) {
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
        planes.clear();
        planes.reserve(n_words);
        sign_mask.clear();
        sign_mask.reserve(n_words);
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

        (query_abs_sum, q_scale)
    }

    /// Quantizes `q_rot` symmetrically to `[-7, +7]` (`q_scale = q_max / 7.0`) and transposes
    /// magnitude bit-planes and sign masks in registers without intermediate bit-matrix allocations.
    pub fn from_rotated_query(q_rot: &[f32]) -> Self {
        let mut planes = Vec::new();
        let mut sign_mask = Vec::new();
        let (query_abs_sum, q_scale) =
            Self::quantize_into_slices(q_rot, &mut planes, &mut sign_mask);
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

/// Quantizes a rotated unit residual `v` (`||v||_2 = 1`) into 4-bit unsigned codes (`0..=15`)
/// representing centered grid levels `c_bar = code - 7.5`, selecting the rescaling multiplier
/// maximizing cosine similarity `<v, c_bar> / ||c_bar||_2` and returning
/// `ip_factor = (<v, c_bar> / D)` clamped to `[RABITQ_L1_MIN_NORMAL_F32, RABITQ_F16_MAX_CLAMP]`.
pub fn quantize_4bit_unit_residual(v: &[f32], out_codes: &mut [u8]) -> OspreyResult<f16> {
    let padded_dim = v.len();
    if padded_dim < 64 || !padded_dim.is_power_of_two() {
        return Err(OspreyError::InvalidDimension { dim: padded_dim });
    }
    if out_codes.len() != padded_dim {
        return Err(OspreyError::BufferTooSmall {
            needed: padded_dim,
            actual: out_codes.len(),
        });
    }

    let d_u32 =
        u32::try_from(padded_dim).map_err(|_| OspreyError::InvalidDimension { dim: padded_dim })?;
    #[allow(clippy::cast_possible_truncation)]
    let sqrt_d = f64::from(d_u32).sqrt() as f32;

    const MULT_CANDIDATES: [f32; 5] = [2.4, 2.75, 3.1, 3.45, 3.9];
    let mut best_t = 2.98_f32 * sqrt_d;
    let mut best_cos = f32::NEG_INFINITY;

    for &mult in &MULT_CANDIDATES {
        let t = mult * sqrt_d;
        let mut ip = 0.0_f32;
        let mut norm_sq = 0.0_f32;
        for &vi in v {
            let q = (vi * t + 7.5_f32)
                .round_ties_even()
                .clamp(0.0_f32, 15.0_f32);
            let cbar = q - 7.5_f32;
            ip += vi * cbar;
            norm_sq += cbar * cbar;
        }
        let cos = if norm_sq > 0.0 {
            ip / norm_sq.sqrt()
        } else {
            f32::NEG_INFINITY
        };
        if cos > best_cos {
            best_cos = cos;
            best_t = t;
        }
    }

    let mut ip_f64 = 0.0_f64;
    for (&vi, dst) in v.iter().zip(out_codes.iter_mut()) {
        let q = (vi * best_t + 7.5_f32)
            .round_ties_even()
            .clamp(0.0_f32, 15.0_f32);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let code = q as u8;
        *dst = code;
        ip_f64 += f64::from(vi) * (f64::from(code) - 7.5_f64);
    }

    let factor = ip_f64 / f64::from(d_u32);
    #[allow(clippy::cast_possible_truncation)]
    let clamped = (factor as f32).clamp(RABITQ_L1_MIN_NORMAL_F32, RABITQ_F16_MAX_CLAMP);
    Ok(f16::from_f32(clamped))
}

/// Encodes an already FWHT-rotated vector and rotated `f16` centroid into 4-bit asymmetric
/// `RaBitQ` codes (`[u8; padded_dim]` in `0..=15`) in-place, without a second FWHT rotation.
pub fn encode_prerotated_rabitq_4bit_in_place(
    x_and_u: &mut [f32],
    centroid_rotated_f16: &[f16],
    out_codes: &mut [u8],
) -> OspreyResult<RabitqEncodedResidual> {
    let res_norm = extract_and_normalize_residual_in_place(x_and_u, centroid_rotated_f16)?;
    let l1_norm_factor = quantize_4bit_unit_residual(x_and_u, out_codes)?;
    Ok(RabitqEncodedResidual {
        res_norm,
        l1_norm_factor,
    })
}

/// Encodes a sanitized, zero-padded residual into 4-bit asymmetric `RaBitQ` (`[u8; padded_dim]`).
pub fn encode_rabitq_4bit_residual<K: RabitqFwhtKernel + ?Sized>(
    x_padded: &[f32],
    centroid_f16: &[f16],
    signs: &RademacherSigns,
    simd: &K,
    scratch_u: &mut [f32],
    out_codes: &mut [u8],
) -> OspreyResult<RabitqEncodedResidual> {
    if signs.padded_dim() != x_padded.len() {
        return Err(OspreyError::InvalidDimension {
            dim: x_padded.len(),
        });
    }
    let res_norm = extract_and_normalize_residual(x_padded, centroid_f16, scratch_u)?;
    simd.fwht_rotate_in_place(scratch_u, signs.xor_masks())?;
    let l1_norm_factor = quantize_4bit_unit_residual(scratch_u, out_codes)?;
    Ok(RabitqEncodedResidual {
        res_norm,
        l1_norm_factor,
    })
}

/// Unsigned 8-bit FastScan Lookup Table for 4-bit `RaBitQ` (`[padded_dim * 16]` bytes).
#[derive(Clone, Debug, PartialEq)]
pub struct FastScanQueryLut4Bit {
    /// Flattened `[padded_dim][16]` unsigned `u8` LUT (`g_d(c) = q_d * (c - 7.5) + 7.5 * |q_d|`).
    pub lut_u8: Vec<u8>,
    /// Global dequantization scale `(15.0 * max_abs_q) / 255.0`.
    pub lut_scale: f32,
    /// Constant shift `7.5 * sum_d |q_rot[d]|` subtracted after the LUT sum.
    pub bias_sum: f32,
}

impl FastScanQueryLut4Bit {
    /// Synthesizes `[padded_dim * 16]` `u8` LUTs into a reusable `lut_u8` buffer,
    /// returning `(lut_scale, bias_sum)` with zero heap allocations once warmed up.
    pub fn synthesize_into(q_rot: &[f32], lut_u8: &mut Vec<u8>) -> (f32, f32) {
        let mut max_abs = 0.0_f32;
        let mut sum_abs = 0.0_f64;
        for &val in q_rot {
            if val.is_finite() {
                let abs_val = val.abs();
                if abs_val > max_abs {
                    max_abs = abs_val;
                }
                sum_abs += f64::from(abs_val);
            }
        }
        let lut_scale = if max_abs > QUERY_QUANT_EPS {
            (15.0_f32 * max_abs) / 255.0_f32
        } else {
            1.0_f32
        };
        let inv_scale = if max_abs > QUERY_QUANT_EPS {
            1.0_f32 / lut_scale
        } else {
            0.0_f32
        };
        #[allow(clippy::cast_possible_truncation)]
        let bias_sum = (7.5_f64 * sum_abs) as f32;

        let needed = q_rot.len() * 16;
        lut_u8.resize(needed, 0);
        for (d, &q_val) in q_rot.iter().enumerate() {
            let q = if q_val.is_finite() { q_val } else { 0.0_f32 };
            let q_abs = q.abs();
            let row = &mut lut_u8[d * 16..(d + 1) * 16];
            for (c, slot) in row.iter_mut().enumerate() {
                #[allow(clippy::cast_precision_loss)]
                let c_centered = (c as f32) - 7.5_f32;
                let g = q * c_centered + 7.5_f32 * q_abs;
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                {
                    *slot = (g * inv_scale).round_ties_even().clamp(0.0_f32, 255.0_f32) as u8;
                }
            }
        }
        (lut_scale, bias_sum)
    }

    /// Validated constructor enforcing `padded_dim >= 64`, power-of-two dimension, and finiteness.
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
        let mut lut_u8 = Vec::new();
        let (lut_scale, bias_sum) = Self::synthesize_into(q_rot, &mut lut_u8);
        Ok(Self {
            lut_u8,
            lut_scale,
            bias_sum,
        })
    }
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

/// Cluster-grouped `.osi` (Osprey Resident 1-Bit / 4-Bit FastScan Index) containing `f16`
/// centroids, `RabitqEntryHeader` metadata, 64-byte-aligned codes, and `RoaringBitmap` tombstones.
pub struct OsiIndex {
    header: OsiHeader,
    signs: RademacherSigns,
    centroids_f16: AlignedBuffer<f16>,
    cluster_offsets: Vec<u32>,
    cluster_code_word_offsets: Vec<usize>,
    entries: Vec<RabitqEntryHeader>,
    codes: AlignedBuffer<u64>,
    tombstones: RoaringBitmap,
}

#[derive(Copy, Clone, Debug)]
struct ProgCandidate {
    score_1bit: f32,
    centroid_ip: f32,
    k_idx: u32,
    local_idx: u32,
    entry_idx: u32,
    d3: i32,
}

#[derive(Default)]
struct ScanScratch {
    q_buf: Vec<f32>,
    planes: Vec<[u64; 4]>,
    sign_mask: Vec<u64>,
    lut_u8: Vec<u8>,
    probed: Vec<ProbedCluster>,
    prog_cands: Vec<ProgCandidate>,
}

thread_local! {
    static SCAN_SCRATCH: RefCell<ScanScratch> = RefCell::new(ScanScratch::default());
}

#[inline]
fn with_scan_scratch<R>(f: impl FnOnce(&mut ScanScratch) -> R) -> R {
    let mut f_opt = Some(f);
    let tls_res = SCAN_SCRATCH.try_with(|cell| {
        if let Ok(mut guard) = cell.try_borrow_mut() {
            f_opt.take().map(|func| func(&mut guard))
        } else {
            None
        }
    });
    if let Ok(Some(val)) = tls_res {
        return val;
    }
    let mut local = ScanScratch::default();
    if let Some(func) = f_opt {
        return func(&mut local);
    }
    unreachable!()
}

enum OsiBacking {
    Mmap(Mmap),
    Aligned(AlignedBuffer<u8>),
}

impl OsiBacking {
    #[inline]
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Mmap(m) => m.as_ref(),
            Self::Aligned(buf) => buf.as_slice(),
        }
    }
}

/// Zero-copy resident `.osi` index backed directly by `memmap2::Mmap` or a single
/// 64-byte-aligned buffer, eliminating heap copies of centroids, codes, and entry headers.
pub struct OsiMappedIndex {
    backing: OsiBacking,
    header: OsiHeader,
    signs: RademacherSigns,
    centroids_start: usize,
    centroids_end: usize,
    entries_start: usize,
    cluster_offsets: Vec<u32>,
    tombstones: RoaringBitmap,
}

impl std::fmt::Debug for OsiMappedIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OsiMappedIndex")
            .field("header", &self.header)
            .field("clusters", &self.cluster_offsets.len().saturating_sub(1))
            .field("tombstones_len", &self.tombstones.len())
            .finish()
    }
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
            let block_len =
                cluster_block_bytes_for_flags(vec_count, header.padded_dim, header.flags)?;
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
            if header.is_progressive_4bit() {
                let _ = OsiClusterViewProgressive4Bit::from_cluster_bytes(
                    &entries_section[rel_off..block_end],
                    vec_count,
                    header.padded_dim,
                )?;
            } else if header.is_4bit() {
                let _ = OsiClusterView4Bit::from_cluster_bytes(
                    &entries_section[rel_off..block_end],
                    vec_count,
                    header.padded_dim,
                )?;
            } else {
                let _ = OsiClusterView::from_cluster_bytes(
                    &entries_section[rel_off..block_end],
                    vec_count,
                    header.padded_dim,
                )?;
            }
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

    /// Returns a zero-copy view over one 1-bit cluster's codes and entry headers.
    #[inline]
    pub fn cluster_view(&self, cluster_id: u32) -> OspreyResult<OsiClusterView<'a>> {
        if self.header.is_4bit() {
            return Err(OspreyError::InvalidDimension { dim: 4 });
        }
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
        let block_len = cluster_block_bytes_for_bits(vec_count, self.header.padded_dim, 1)?;
        OsiClusterView::from_cluster_bytes(
            &self.entries_section[rel_off..rel_off + block_len],
            vec_count,
            self.header.padded_dim,
        )
    }

    /// Returns a zero-copy view over one 4-bit FastScan cluster's blocks and entry headers.
    #[inline]
    pub fn cluster_view_4bit(&self, cluster_id: u32) -> OspreyResult<OsiClusterView4Bit<'a>> {
        if !self.header.is_4bit() || self.header.is_progressive_4bit() {
            return Err(OspreyError::InvalidDimension { dim: 1 });
        }
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
        let block_len = cluster_block_bytes_for_bits(vec_count, self.header.padded_dim, 4)?;
        OsiClusterView4Bit::from_cluster_bytes(
            &self.entries_section[rel_off..rel_off + block_len],
            vec_count,
            self.header.padded_dim,
        )
    }

    /// Returns a zero-copy view over one progressive bitplane 4-bit cluster.
    #[inline]
    pub fn cluster_view_progressive_4bit(
        &self,
        cluster_id: u32,
    ) -> OspreyResult<OsiClusterViewProgressive4Bit<'a>> {
        if !self.header.is_progressive_4bit() {
            return Err(OspreyError::InvalidDimension { dim: 1 });
        }
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
        let block_len =
            cluster_block_bytes_for_flags(vec_count, self.header.padded_dim, self.header.flags)?;
        OsiClusterViewProgressive4Bit::from_cluster_bytes(
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
    /// in FWHT-rotated space (`fwht_seed`), using `bits` (`1` or `4`) `RaBitQ` quantization.
    #[allow(clippy::too_many_arguments)]
    pub fn build_with_bits<K: RabitqFwhtKernel + ?Sized>(
        vectors: &[f32],
        doc_ids: &[u64],
        locations: &[VectorLocation],
        kmeans: &BinaryKMeansResult,
        orig_dim: usize,
        fwht_seed: u64,
        tombstones: RoaringBitmap,
        bits: u8,
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

        match bits {
            1 => Self::build_with_encoder(
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
            ),
            4 => Self::build_with_encoder_4bit(
                OSI_FLAG_RABITQ_4BIT,
                doc_ids,
                locations,
                kmeans,
                orig_dim,
                padded_dim,
                fwht_seed,
                tombstones,
                |scratch, src_idx, centroid_f16, out_codes| {
                    let raw = &vectors[src_idx * orig_dim..(src_idx + 1) * orig_dim];
                    prepare_and_pad_vector(raw, scratch)?;
                    simd.fwht_rotate_in_place(scratch, xor_masks)?;
                    encode_prerotated_rabitq_4bit_in_place(scratch, centroid_f16, out_codes)
                },
            ),
            other => Err(OspreyError::InvalidDimension {
                dim: usize::from(other),
            }),
        }
    }

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
        Self::build_with_bits(
            vectors, doc_ids, locations, kmeans, orig_dim, fwht_seed, tombstones, 1, simd,
        )
    }

    /// Builds a cluster-grouped `.osi` index in parallel from already `f16`-quantized vectors
    /// (`[N, orig_dim]`) using `bits` (`1` or `4`) `RaBitQ` quantization.
    #[allow(clippy::too_many_arguments)]
    pub fn build_from_f16_with_bits<K: RabitqFwhtKernel + ?Sized>(
        vectors_f16: &[f16],
        doc_ids: &[u64],
        locations: &[VectorLocation],
        kmeans: &BinaryKMeansResult,
        orig_dim: usize,
        fwht_seed: u64,
        tombstones: RoaringBitmap,
        bits: u8,
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

        match bits {
            1 => Self::build_with_encoder(
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
            ),
            4 => Self::build_with_encoder_4bit(
                OSI_FLAG_RABITQ_4BIT,
                doc_ids,
                locations,
                kmeans,
                orig_dim,
                padded_dim,
                fwht_seed,
                tombstones,
                |scratch, src_idx, centroid_f16, out_codes| {
                    let raw_f16 = &vectors_f16[src_idx * orig_dim..(src_idx + 1) * orig_dim];
                    prepare_and_pad_vector_f16(raw_f16, scratch)?;
                    simd.fwht_rotate_in_place(scratch, xor_masks)?;
                    encode_prerotated_rabitq_4bit_in_place(scratch, centroid_f16, out_codes)
                },
            ),
            other => Err(OspreyError::InvalidDimension {
                dim: usize::from(other),
            }),
        }
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
        Self::build_from_f16_with_bits(
            vectors_f16,
            doc_ids,
            locations,
            kmeans,
            orig_dim,
            fwht_seed,
            tombstones,
            1,
            simd,
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

        let words_per_vec = padded_dim / 64;
        let cluster_code_word_offsets: Vec<usize> = cluster_offsets
            .iter()
            .map(|&off| usize::try_from(off).unwrap_or(0) * words_per_vec)
            .collect();

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

        let signs = RademacherSigns::from_seed(padded_dim, fwht_seed)?;
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
            signs,
            centroids_f16,
            cluster_offsets,
            cluster_code_word_offsets,
            entries,
            codes,
            tombstones,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn build_with_encoder_4bit<F>(
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
        F: Fn(&mut [f32], usize, &[f16], &mut [u8]) -> OspreyResult<RabitqEncodedResidual>
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

        let padded_dim_u16 = u16::try_from(padded_dim)
            .map_err(|_| OspreyError::InvalidDimension { dim: padded_dim })?;
        let is_progressive = padded_dim >= 512;
        let final_flags = if is_progressive {
            flags | OSI_FLAG_RABITQ_4BIT | OSI_FLAG_RABITQ_PROGRESSIVE
        } else {
            flags | OSI_FLAG_RABITQ_4BIT
        };
        let words_per_vec = padded_dim / 64;
        let block_bytes = fastscan_block_bytes(padded_dim_u16)?;
        let words_per_block = block_bytes / size_of::<u64>();

        let mut cluster_offsets = Vec::with_capacity(num_clusters + 1);
        cluster_offsets.push(0_u32);
        let mut cluster_code_word_offsets = Vec::with_capacity(num_clusters + 1);
        cluster_code_word_offsets.push(0_usize);
        let mut running = 0_u32;
        let mut running_words = 0_usize;
        for &count in &counts {
            running += count;
            cluster_offsets.push(running);
            let cnt_usize = usize::try_from(count).unwrap_or(0);
            let cluster_words = if is_progressive {
                cnt_usize
                    .checked_mul(words_per_vec * 4)
                    .ok_or(OspreyError::AllocationFailed {
                        size: usize::MAX,
                        align: CACHE_LINE_ALIGN,
                    })?
            } else {
                let num_blocks = cnt_usize.div_ceil(FASTSCAN_BLOCK_SIZE);
                num_blocks
                    .checked_mul(words_per_block)
                    .ok_or(OspreyError::AllocationFailed {
                        size: usize::MAX,
                        align: CACHE_LINE_ALIGN,
                    })?
            };
            running_words =
                running_words
                    .checked_add(cluster_words)
                    .ok_or(OspreyError::AllocationFailed {
                        size: usize::MAX,
                        align: CACHE_LINE_ALIGN,
                    })?;
            cluster_code_word_offsets.push(running_words);
        }

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

        let mut codes = AlignedBuffer::<u64>::new(running_words)?;
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
        if n > 0 && is_progressive {
            type ProgTask<'a> = (
                usize,
                &'a [usize],
                &'a mut [RabitqEntryHeader],
                &'a mut [u64],
                &'a mut [u64],
            );
            let mut rem_entries = entries.as_mut_slice();
            let mut rem_codes = codes.as_mut_slice();
            let mut prog_tasks: Vec<ProgTask<'_>> = Vec::new();
            const CHUNK_VECS: usize = 256;

            for c in 0..num_clusters {
                let start = usize::try_from(cluster_offsets[c]).unwrap_or(0);
                let end = usize::try_from(cluster_offsets[c + 1]).unwrap_or(0);
                let count = end.saturating_sub(start);
                let word_count = cluster_code_word_offsets[c + 1] - cluster_code_word_offsets[c];
                let (c_entries, next_entries) = rem_entries.split_at_mut(count);
                let (c_codes, next_codes) = rem_codes.split_at_mut(word_count);
                rem_entries = next_entries;
                rem_codes = next_codes;

                if count == 0 {
                    continue;
                }
                let (plane3_all, low_all) = c_codes.split_at_mut(count * words_per_vec);
                let ord_c = &order[start..end];
                for (((ord_ch, ent_ch), p3_ch), low_ch) in ord_c
                    .chunks(CHUNK_VECS)
                    .zip(c_entries.chunks_mut(CHUNK_VECS))
                    .zip(plane3_all.chunks_mut(CHUNK_VECS * words_per_vec))
                    .zip(low_all.chunks_mut(CHUNK_VECS * 3 * words_per_vec))
                {
                    prog_tasks.push((c, ord_ch, ent_ch, p3_ch, low_ch));
                }
            }

            let encode_prog = |(scratch, code_buf): &mut (Vec<f32>, Vec<u8>),
                               (cluster_id, ord_ch, ent_ch, p3_ch, low_ch): (
                usize,
                &[usize],
                &mut [RabitqEntryHeader],
                &mut [u64],
                &mut [u64],
            )|
             -> OspreyResult<()> {
                let centroid =
                    &centroids_slice[cluster_id * padded_dim..(cluster_id + 1) * padded_dim];
                for (j, (&src_idx, dst_ent)) in ord_ch.iter().zip(ent_ch.iter_mut()).enumerate() {
                    let encoded = encode_fn(scratch, src_idx, centroid, code_buf)?;
                    let p3_dst = &mut p3_ch[j * words_per_vec..(j + 1) * words_per_vec];
                    let low_dst = &mut low_ch[j * 3 * words_per_vec..(j + 1) * 3 * words_per_vec];
                    for w in 0..words_per_vec {
                        let base_d = w * 64;
                        let mut b3 = 0_u64;
                        let mut b2 = 0_u64;
                        let mut b1 = 0_u64;
                        let mut b0 = 0_u64;
                        for b in 0..64 {
                            let c_val = u64::from(code_buf[base_d + b] & 0x0F);
                            b3 |= ((c_val >> 3) & 1) << b;
                            b2 |= ((c_val >> 2) & 1) << b;
                            b1 |= ((c_val >> 1) & 1) << b;
                            b0 |= (c_val & 1) << b;
                        }
                        p3_dst[w] = b3;
                        low_dst[w] = b2;
                        low_dst[words_per_vec + w] = b1;
                        low_dst[2 * words_per_vec + w] = b0;
                    }
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
            };

            #[cfg(not(miri))]
            prog_tasks.into_par_iter().try_for_each_init(
                || (vec![0.0_f32; padded_dim], vec![0_u8; padded_dim]),
                encode_prog,
            )?;
            #[cfg(miri)]
            {
                let mut state = (vec![0.0_f32; padded_dim], vec![0_u8; padded_dim]);
                for task in prog_tasks {
                    encode_prog(&mut state, task)?;
                }
            }
        } else if n > 0 {
            type BlockTask<'a> = (
                usize,
                &'a [usize],
                &'a mut [RabitqEntryHeader],
                &'a mut [u64],
            );
            let mut rem_entries = entries.as_mut_slice();
            let mut rem_codes = codes.as_mut_slice();
            let mut block_tasks: Vec<BlockTask<'_>> = Vec::new();

            for c in 0..num_clusters {
                let start = usize::try_from(cluster_offsets[c]).unwrap_or(0);
                let end = usize::try_from(cluster_offsets[c + 1]).unwrap_or(0);
                let count = end.saturating_sub(start);
                let word_count = cluster_code_word_offsets[c + 1] - cluster_code_word_offsets[c];
                let (c_entries, next_entries) = rem_entries.split_at_mut(count);
                let (c_codes, next_codes) = rem_codes.split_at_mut(word_count);
                rem_entries = next_entries;
                rem_codes = next_codes;

                if count == 0 {
                    continue;
                }
                let ord_c = &order[start..end];
                for ((ord_blk, ent_blk), code_blk) in ord_c
                    .chunks(FASTSCAN_BLOCK_SIZE)
                    .zip(c_entries.chunks_mut(FASTSCAN_BLOCK_SIZE))
                    .zip(c_codes.chunks_exact_mut(words_per_block))
                {
                    block_tasks.push((c, ord_blk, ent_blk, code_blk));
                }
            }

            let num_pairs = padded_dim / 2;
            let encode_block = |(scratch, code_buf): &mut (Vec<f32>, Vec<u8>),
                                (cluster_id, ord_blk, ent_blk, code_blk): (
                usize,
                &[usize],
                &mut [RabitqEntryHeader],
                &mut [u64],
            )|
             -> OspreyResult<()> {
                let centroid =
                    &centroids_slice[cluster_id * padded_dim..(cluster_id + 1) * padded_dim];
                let block_u8: &mut [u8] = bytemuck::cast_slice_mut(code_blk);
                block_u8.fill(0);

                for (j, (&src_idx, dst_ent)) in ord_blk.iter().zip(ent_blk.iter_mut()).enumerate() {
                    let encoded = encode_fn(scratch, src_idx, centroid, code_buf)?;
                    for m in 0..num_pairs {
                        let c_even = code_buf[2 * m] & 0x0F;
                        let c_odd = code_buf[2 * m + 1] & 0x0F;
                        block_u8[m * FASTSCAN_BLOCK_SIZE + j] = c_even | (c_odd << 4);
                    }
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
            };

            #[cfg(not(miri))]
            block_tasks.into_par_iter().try_for_each_init(
                || (vec![0.0_f32; padded_dim], vec![0_u8; padded_dim]),
                encode_block,
            )?;
            #[cfg(miri)]
            {
                let mut state = (vec![0.0_f32; padded_dim], vec![0_u8; padded_dim]);
                for task in block_tasks {
                    encode_block(&mut state, task)?;
                }
            }
        }

        let centroids_f16 = AlignedBuffer::from_slice(centroids_slice)?;
        let orig_dim_u16 =
            u16::try_from(orig_dim).map_err(|_| OspreyError::InvalidDimension { dim: orig_dim })?;
        let n_u64 = u64::try_from(n).map_err(|_| OspreyError::InvalidDimension { dim: n })?;

        let signs = RademacherSigns::from_seed(padded_dim, fwht_seed)?;
        let header = compute_sealed_osi_header(
            final_flags,
            orig_dim_u16,
            padded_dim_u16,
            k_u32,
            n_u64,
            fwht_seed,
            &cluster_offsets,
        )?;

        Ok(Self {
            header,
            signs,
            centroids_f16,
            cluster_offsets,
            cluster_code_word_offsets,
            entries,
            codes,
            tombstones,
        })
    }

    /// Returns the `RaBitQ` quantization bit-width (`1` or `4`) of this index.
    #[inline]
    pub const fn rabitq_bits(&self) -> u8 {
        self.header.rabitq_bits()
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

    /// Returns the contiguous code bytes for `cluster_id` (`1-bit` `u64` words or `4-bit` FastScan blocks).
    #[inline]
    pub fn cluster_codes_bytes(&self, cluster_id: u32) -> Option<&[u8]> {
        let k = usize::try_from(cluster_id).ok()?;
        let start_w = *self.cluster_code_word_offsets.get(k)?;
        let end_w = *self.cluster_code_word_offsets.get(k + 1)?;
        self.codes.get(start_w..end_w).map(bytemuck::cast_slice)
    }

    /// Returns the contiguous 1-bit `u64` code slice for all entries in `cluster_id`.
    #[inline]
    pub fn cluster_codes(&self, cluster_id: u32) -> Option<&[u64]> {
        if self.header.is_4bit() {
            return None;
        }
        let k = usize::try_from(cluster_id).ok()?;
        let start_idx = usize::try_from(*self.cluster_offsets.get(k)?).ok()?;
        let end_idx = usize::try_from(*self.cluster_offsets.get(k + 1)?).ok()?;
        let w = self.words_per_vec();
        self.codes
            .get(start_idx.checked_mul(w)?..end_idx.checked_mul(w)?)
    }

    /// Returns the `[u64; padded_dim / 64]` code slice for global physical index `entry_idx` (1-bit only).
    #[inline]
    pub fn entry_code(&self, entry_idx: usize) -> Option<&[u64]> {
        if self.header.is_4bit() {
            return None;
        }
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

            let block_len = cluster_block_bytes_for_flags(count, header.padded_dim, header.flags)?;
            let block_abs = ent_off + rel_cursor;
            let word_start = self.cluster_code_word_offsets[c];
            let word_end = self.cluster_code_word_offsets[c + 1];
            let src_codes_u8: &[u8] =
                bytemuck::cast_slice(&self.codes.as_slice()[word_start..word_end]);
            let codes_bytes = src_codes_u8.len();
            let entries_rel =
                checked_align_up_64(codes_bytes).ok_or(OspreyError::CorruptedFooter)?;
            let entries_bytes = count * size_of::<RabitqEntryHeader>();

            buf[block_abs..block_abs + codes_bytes].copy_from_slice(src_codes_u8);
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
        let bits = header.rabitq_bits();

        let centroids_f16 = AlignedBuffer::from_slice(view.centroids_f16())?;
        let mut cluster_offsets = Vec::with_capacity(k + 1);
        cluster_offsets.push(0_u32);
        let mut cluster_code_word_offsets = Vec::with_capacity(k + 1);
        cluster_code_word_offsets.push(0_usize);
        let mut entries = Vec::with_capacity(n);

        let total_code_words = if header.is_progressive_4bit() {
            let words_per_vec = usize::from(header.padded_dim) / 64;
            n.checked_mul(words_per_vec * 4)
                .ok_or(OspreyError::AllocationFailed {
                    size: usize::MAX,
                    align: CACHE_LINE_ALIGN,
                })?
        } else if bits == 4 {
            let block_words = fastscan_block_bytes(header.padded_dim)? / size_of::<u64>();
            let mut sum_words = 0_usize;
            for span in view.cluster_spans() {
                let cnt =
                    usize::try_from(span.vector_count).map_err(|_| OspreyError::CorruptedFooter)?;
                let blocks = cnt.div_ceil(FASTSCAN_BLOCK_SIZE);
                sum_words = sum_words
                    .checked_add(blocks.checked_mul(block_words).ok_or(
                        OspreyError::AllocationFailed {
                            size: usize::MAX,
                            align: CACHE_LINE_ALIGN,
                        },
                    )?)
                    .ok_or(OspreyError::AllocationFailed {
                        size: usize::MAX,
                        align: CACHE_LINE_ALIGN,
                    })?;
            }
            sum_words
        } else {
            let words_per_vec = usize::from(header.padded_dim) / 64;
            n.checked_mul(words_per_vec)
                .ok_or(OspreyError::AllocationFailed {
                    size: usize::MAX,
                    align: CACHE_LINE_ALIGN,
                })?
        };
        let mut codes = AlignedBuffer::<u64>::new(total_code_words)?;

        let mut running = 0_u32;
        let mut code_cursor = 0_usize;
        for c in 0..k {
            let c_u32 = u32::try_from(c).map_err(|_| OspreyError::InvalidDimension { dim: c })?;
            let ent_len = if header.is_progressive_4bit() {
                let c_view = view.cluster_view_progressive_4bit(c_u32)?;
                entries.extend_from_slice(c_view.entries);
                let p3_len = c_view.plane3.len();
                let low_len = c_view.low_planes.len();
                codes.as_mut_slice()[code_cursor..code_cursor + p3_len]
                    .copy_from_slice(c_view.plane3);
                code_cursor += p3_len;
                codes.as_mut_slice()[code_cursor..code_cursor + low_len]
                    .copy_from_slice(c_view.low_planes);
                code_cursor += low_len;
                c_view.entries.len()
            } else if bits == 4 {
                let c_view = view.cluster_view_4bit(c_u32)?;
                entries.extend_from_slice(c_view.entries);
                let byte_dst: &mut [u8] = bytemuck::cast_slice_mut(codes.as_mut_slice());
                let byte_start = code_cursor * size_of::<u64>();
                let byte_len = c_view.blocks.len();
                byte_dst[byte_start..byte_start + byte_len].copy_from_slice(c_view.blocks);
                code_cursor += byte_len / size_of::<u64>();
                c_view.entries.len()
            } else {
                let c_view = view.cluster_view(c_u32)?;
                entries.extend_from_slice(c_view.entries);
                let len_words = c_view.codes.len();
                codes.as_mut_slice()[code_cursor..code_cursor + len_words]
                    .copy_from_slice(c_view.codes);
                code_cursor += len_words;
                c_view.entries.len()
            };

            let cnt_u32 = u32::try_from(ent_len).map_err(|_| OspreyError::CorruptedFooter)?;
            running = running
                .checked_add(cnt_u32)
                .ok_or(OspreyError::CorruptedFooter)?;
            cluster_offsets.push(running);
            cluster_code_word_offsets.push(code_cursor);
        }

        let signs = RademacherSigns::from_seed(usize::from(header.padded_dim), header.fwht_seed)?;
        Ok(Self {
            header,
            signs,
            centroids_f16,
            cluster_offsets,
            cluster_code_word_offsets,
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

    /// Reconstructs the approximate Stage-1 inner product for a 4-bit FastScan entry.
    #[inline]
    pub fn reconstruct_asymmetric_score_4bit(
        centroid_dot: f32,
        entry: &RabitqEntryHeader,
        lut_scale: f32,
        bias_sum: f32,
        sum_lut_u32: u32,
        padded_dim: u16,
    ) -> f32 {
        let denom =
            f32::from(padded_dim) * entry.l1_norm_factor.to_f32().max(RABITQ_L1_MIN_NORMAL_F32);
        #[allow(clippy::cast_precision_loss)]
        let raw_dot = (sum_lut_u32 as f32) * lut_scale - bias_sum;
        centroid_dot + entry.res_norm.to_f32() * (raw_dot / denom)
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
        let mut out = Vec::new();
        probe_clusters_into(
            self.centroids_f16.as_slice(),
            self.header.num_clusters,
            self.header.padded_dim,
            q_vec,
            nprobe,
            &ScalarRabitqKernel,
            &mut out,
        )?;
        Ok(out)
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
        let cluster_accessor = |k_idx: usize| -> OspreyResult<(&[RabitqEntryHeader], &[u8], u32)> {
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
            let w_start = *self
                .cluster_code_word_offsets
                .get(k_idx)
                .ok_or(OspreyError::InvalidDimension { dim: k_idx })?;
            let w_end = *self
                .cluster_code_word_offsets
                .get(k_idx + 1)
                .ok_or(OspreyError::InvalidDimension { dim: k_idx + 1 })?;
            let base_idx = u32::try_from(start).unwrap_or(u32::MAX);
            Ok((
                &self.entries[start..end],
                bytemuck::cast_slice(&self.codes.as_slice()[w_start..w_end]),
                base_idx,
            ))
        };

        match (self.tombstones.is_empty(), allowed_docs) {
            (true, None) => search_candidates_impl(
                &self.header,
                &self.signs,
                self.centroids_f16.as_slice(),
                query,
                nprobe,
                top_m,
                simd,
                &NoFilter,
                cluster_accessor,
            ),
            (false, None) => search_candidates_impl(
                &self.header,
                &self.signs,
                self.centroids_f16.as_slice(),
                query,
                nprobe,
                top_m,
                simd,
                &TombstoneFilter(&self.tombstones),
                cluster_accessor,
            ),
            (true, Some(bm)) => search_candidates_impl(
                &self.header,
                &self.signs,
                self.centroids_f16.as_slice(),
                query,
                nprobe,
                top_m,
                simd,
                &AllowedBitmapFilter(bm),
                cluster_accessor,
            ),
            (false, Some(bm)) => search_candidates_impl(
                &self.header,
                &self.signs,
                self.centroids_f16.as_slice(),
                query,
                nprobe,
                top_m,
                simd,
                &TombstoneAndAllowedFilter(&self.tombstones, bm),
                cluster_accessor,
            ),
        }
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
        let cluster_accessor = |k_idx: usize| -> OspreyResult<(&[RabitqEntryHeader], &[u8], u32)> {
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
            let w_start = *self
                .cluster_code_word_offsets
                .get(k_idx)
                .ok_or(OspreyError::InvalidDimension { dim: k_idx })?;
            let w_end = *self
                .cluster_code_word_offsets
                .get(k_idx + 1)
                .ok_or(OspreyError::InvalidDimension { dim: k_idx + 1 })?;
            let base_idx = u32::try_from(start).unwrap_or(u32::MAX);
            Ok((
                &self.entries[start..end],
                bytemuck::cast_slice(&self.codes.as_slice()[w_start..w_end]),
                base_idx,
            ))
        };

        if self.tombstones.is_empty() {
            search_candidates_impl(
                &self.header,
                &self.signs,
                self.centroids_f16.as_slice(),
                query,
                nprobe,
                top_m,
                simd,
                &ClosureFilter(&allow_entry),
                cluster_accessor,
            )
        } else {
            search_candidates_impl(
                &self.header,
                &self.signs,
                self.centroids_f16.as_slice(),
                query,
                nprobe,
                top_m,
                simd,
                &TombstoneAndClosureFilter(&self.tombstones, &allow_entry),
                cluster_accessor,
            )
        }
    }
}

impl OsiMappedIndex {
    /// Opens and memory-maps an `.osi` file zero-copy via `memmap2::Mmap`, validating
    /// all headers, checksums, and cluster spans upfront and precomputing `RademacherSigns`.
    pub fn open_mmap<P: AsRef<Path>>(path: P) -> OspreyResult<Self> {
        let file = File::open(path)?;
        // SAFETY: `.osi` segment files are immutable for the lifetime of the read-only mapping.
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_backing(OsiBacking::Mmap(mmap))
    }

    /// Wraps a 64-byte-aligned `.osi` buffer zero-copy without copying centroids, codes,
    /// or entry headers into secondary heap allocations.
    pub fn from_aligned_buffer(buffer: AlignedBuffer<u8>) -> OspreyResult<Self> {
        Self::from_backing(OsiBacking::Aligned(buffer))
    }

    fn from_backing(backing: OsiBacking) -> OspreyResult<Self> {
        let (
            header,
            signs,
            centroids_start,
            centroids_end,
            entries_start,
            cluster_offsets,
            tombstones,
        ) = {
            let bytes = backing.as_slice();
            let view = OsiMappedView::from_bytes(bytes)?;
            let header = *view.header();
            let padded_dim = usize::from(header.padded_dim);
            let signs = RademacherSigns::from_seed(padded_dim, header.fwht_seed)?;
            let centroids_start = usize::try_from(header.centroids_offset)
                .map_err(|_| OspreyError::CorruptedFooter)?;
            let centroids_len = centroids_unpadded_bytes(header.num_clusters, header.padded_dim)?;
            let centroids_end = centroids_start
                .checked_add(centroids_len)
                .ok_or(OspreyError::CorruptedFooter)?;
            let entries_start =
                usize::try_from(header.entries_offset).map_err(|_| OspreyError::CorruptedFooter)?;

            let spans = view.cluster_spans();
            let mut cluster_offsets = Vec::with_capacity(spans.len() + 1);
            cluster_offsets.push(0_u32);
            let mut running = 0_u32;
            for span in spans {
                running = running
                    .checked_add(span.vector_count)
                    .ok_or(OspreyError::CorruptedFooter)?;
                cluster_offsets.push(running);
            }

            (
                header,
                signs,
                centroids_start,
                centroids_end,
                entries_start,
                cluster_offsets,
                view.tombstones().clone(),
            )
        };

        Ok(Self {
            backing,
            header,
            signs,
            centroids_start,
            centroids_end,
            entries_start,
            cluster_offsets,
            tombstones,
        })
    }

    /// Returns the validated `.osi` header.
    #[inline]
    pub const fn header(&self) -> &OsiHeader {
        &self.header
    }

    /// Returns the flat `K * padded_dim` `f16` centroid slice mapped zero-copy.
    #[inline]
    pub fn centroids_f16(&self) -> &[f16] {
        let slice = &self.backing.as_slice()[self.centroids_start..self.centroids_end];
        bytemuck::try_cast_slice(slice).unwrap_or(&[])
    }

    /// Returns the `RoaringBitmap` of logically deleted document IDs.
    #[inline]
    pub const fn tombstones(&self) -> &RoaringBitmap {
        &self.tombstones
    }

    /// Returns `true` if `doc_id` is marked deleted in the `.osi` tombstone bitmap.
    #[inline]
    pub fn is_deleted(&self, doc_id: u64) -> bool {
        u32::try_from(doc_id).is_ok_and(|id| self.tombstones.contains(id))
    }

    #[inline]
    fn cluster_slices(&self, k_idx: usize) -> OspreyResult<(&[RabitqEntryHeader], &[u8], u32)> {
        let bytes = self.backing.as_slice();
        let k_total = usize::try_from(self.header.num_clusters)
            .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
        if k_idx >= k_total {
            return Err(OspreyError::InvalidDimension { dim: k_idx });
        }
        let spans_bytes = k_total * size_of::<OsiClusterSpan>();
        let spans: &[OsiClusterSpan] =
            bytemuck::try_cast_slice(&bytes[self.entries_start..self.entries_start + spans_bytes])
                .map_err(|_| OspreyError::CorruptedFooter)?;
        let span = spans[k_idx];
        let rel_off =
            usize::try_from(span.cluster_offset).map_err(|_| OspreyError::CorruptedFooter)?;
        let vec_count =
            usize::try_from(span.vector_count).map_err(|_| OspreyError::CorruptedFooter)?;
        let block_len =
            cluster_block_bytes_for_flags(vec_count, self.header.padded_dim, self.header.flags)?;
        let block_start = self.entries_start + rel_off;
        let base_idx = self.cluster_offsets.get(k_idx).copied().unwrap_or(0);
        if self.header.is_progressive_4bit() {
            let cluster_view = OsiClusterViewProgressive4Bit::from_cluster_bytes(
                &bytes[block_start..block_start + block_len],
                vec_count,
                self.header.padded_dim,
            )?;
            let total_bytes =
                (cluster_view.plane3.len() + cluster_view.low_planes.len()) * size_of::<u64>();
            let code_bytes: &[u8] = if cluster_view.plane3.is_empty() {
                &[]
            } else {
                let p3_bytes: &[u8] = bytemuck::cast_slice(cluster_view.plane3);
                // SAFETY: `plane3` and `low_planes` are adjacent in the mmap/buffer (`total_bytes`).
                unsafe { std::slice::from_raw_parts(p3_bytes.as_ptr(), total_bytes) }
            };
            Ok((cluster_view.entries, code_bytes, base_idx))
        } else if self.header.is_4bit() {
            let cluster_view = OsiClusterView4Bit::from_cluster_bytes(
                &bytes[block_start..block_start + block_len],
                vec_count,
                self.header.padded_dim,
            )?;
            Ok((cluster_view.entries, cluster_view.blocks, base_idx))
        } else {
            let cluster_view = OsiClusterView::from_cluster_bytes(
                &bytes[block_start..block_start + block_len],
                vec_count,
                self.header.padded_dim,
            )?;
            Ok((
                cluster_view.entries,
                bytemuck::cast_slice(cluster_view.codes),
                base_idx,
            ))
        }
    }

    /// Executes Stage-1 asymmetric `RaBitQ` search zero-copy over the mapped `.osi`.
    pub fn search_candidates<K: RabitqScanKernel + ?Sized>(
        &self,
        query: &[f32],
        nprobe: usize,
        top_m: usize,
        simd: &K,
    ) -> OspreyResult<Vec<ScoredCandidate>> {
        self.search_candidates_filtered(query, nprobe, top_m, simd, None)
    }

    /// Executes Stage-1 asymmetric `RaBitQ` search with an optional allowed-doc bitmap filter.
    pub fn search_candidates_filtered<K: RabitqScanKernel + ?Sized>(
        &self,
        query: &[f32],
        nprobe: usize,
        top_m: usize,
        simd: &K,
        allowed_docs: Option<&RoaringBitmap>,
    ) -> OspreyResult<Vec<ScoredCandidate>> {
        let centroids = self.centroids_f16();
        let cluster_accessor = |k_idx: usize| self.cluster_slices(k_idx);
        match (self.tombstones.is_empty(), allowed_docs) {
            (true, None) => search_candidates_impl(
                &self.header,
                &self.signs,
                centroids,
                query,
                nprobe,
                top_m,
                simd,
                &NoFilter,
                cluster_accessor,
            ),
            (false, None) => search_candidates_impl(
                &self.header,
                &self.signs,
                centroids,
                query,
                nprobe,
                top_m,
                simd,
                &TombstoneFilter(&self.tombstones),
                cluster_accessor,
            ),
            (true, Some(bm)) => search_candidates_impl(
                &self.header,
                &self.signs,
                centroids,
                query,
                nprobe,
                top_m,
                simd,
                &AllowedBitmapFilter(bm),
                cluster_accessor,
            ),
            (false, Some(bm)) => search_candidates_impl(
                &self.header,
                &self.signs,
                centroids,
                query,
                nprobe,
                top_m,
                simd,
                &TombstoneAndAllowedFilter(&self.tombstones, bm),
                cluster_accessor,
            ),
        }
    }

    /// Executes Stage-1 asymmetric `RaBitQ` search with a caller-supplied predicate zero-copy.
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
        let centroids = self.centroids_f16();
        let cluster_accessor = |k_idx: usize| self.cluster_slices(k_idx);
        if self.tombstones.is_empty() {
            search_candidates_impl(
                &self.header,
                &self.signs,
                centroids,
                query,
                nprobe,
                top_m,
                simd,
                &ClosureFilter(&allow_entry),
                cluster_accessor,
            )
        } else {
            search_candidates_impl(
                &self.header,
                &self.signs,
                centroids,
                query,
                nprobe,
                top_m,
                simd,
                &TombstoneAndClosureFilter(&self.tombstones, &allow_entry),
                cluster_accessor,
            )
        }
    }
}

trait EntryFilter {
    fn allow(&self, entry_idx: u32, doc_id: u64) -> bool;
}

struct NoFilter;
impl EntryFilter for NoFilter {
    #[inline(always)]
    fn allow(&self, _entry_idx: u32, _doc_id: u64) -> bool {
        true
    }
}

struct TombstoneFilter<'a>(&'a RoaringBitmap);
impl EntryFilter for TombstoneFilter<'_> {
    #[inline(always)]
    fn allow(&self, _entry_idx: u32, doc_id: u64) -> bool {
        !u32::try_from(doc_id).is_ok_and(|id| self.0.contains(id))
    }
}

struct AllowedBitmapFilter<'a>(&'a RoaringBitmap);
impl EntryFilter for AllowedBitmapFilter<'_> {
    #[inline(always)]
    fn allow(&self, _entry_idx: u32, doc_id: u64) -> bool {
        u32::try_from(doc_id).is_ok_and(|id| self.0.contains(id))
    }
}

struct TombstoneAndAllowedFilter<'a>(&'a RoaringBitmap, &'a RoaringBitmap);
impl EntryFilter for TombstoneAndAllowedFilter<'_> {
    #[inline(always)]
    fn allow(&self, _entry_idx: u32, doc_id: u64) -> bool {
        u32::try_from(doc_id).is_ok_and(|id| !self.0.contains(id) && self.1.contains(id))
    }
}

struct ClosureFilter<'a, F>(&'a F);
impl<F: Fn(u32, u64) -> bool> EntryFilter for ClosureFilter<'_, F> {
    #[inline(always)]
    fn allow(&self, entry_idx: u32, doc_id: u64) -> bool {
        (self.0)(entry_idx, doc_id)
    }
}

struct TombstoneAndClosureFilter<'a, F>(&'a RoaringBitmap, &'a F);
impl<F: Fn(u32, u64) -> bool> EntryFilter for TombstoneAndClosureFilter<'_, F> {
    #[inline(always)]
    fn allow(&self, entry_idx: u32, doc_id: u64) -> bool {
        !u32::try_from(doc_id).is_ok_and(|id| self.0.contains(id)) && (self.1)(entry_idx, doc_id)
    }
}

fn probe_clusters_into<K: RabitqScanKernel + ?Sized>(
    centroids: &[f16],
    num_clusters: u32,
    padded_dim_u16: u16,
    q_vec: &[f32],
    nprobe: usize,
    simd: &K,
    out: &mut Vec<ProbedCluster>,
) -> OspreyResult<()> {
    let padded_dim = usize::from(padded_dim_u16);
    if q_vec.len() != padded_dim {
        return Err(OspreyError::InvalidDimension { dim: q_vec.len() });
    }
    for (index, &val) in q_vec.iter().enumerate() {
        if !val.is_finite() {
            return Err(OspreyError::NonFiniteEmbedding { index });
        }
    }
    let k_total = usize::try_from(num_clusters)
        .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
    let eff_nprobe = nprobe.min(k_total);
    out.clear();
    if eff_nprobe == 0 {
        return Ok(());
    }

    if eff_nprobe <= 64 {
        probe_clusters_cap::<64, K>(centroids, padded_dim, q_vec, k_total, eff_nprobe, simd, out)
    } else if eff_nprobe <= 256 {
        probe_clusters_cap::<256, K>(centroids, padded_dim, q_vec, k_total, eff_nprobe, simd, out)
    } else if eff_nprobe <= 512 {
        probe_clusters_cap::<512, K>(centroids, padded_dim, q_vec, k_total, eff_nprobe, simd, out)
    } else if eff_nprobe <= 4096 {
        probe_clusters_cap::<4096, K>(centroids, padded_dim, q_vec, k_total, eff_nprobe, simd, out)
    } else {
        Err(OspreyError::InvalidDimension { dim: eff_nprobe })
    }
}

fn probe_clusters_cap<const CAP: usize, K: RabitqScanKernel + ?Sized>(
    centroids: &[f16],
    padded_dim: usize,
    q_vec: &[f32],
    k_total: usize,
    eff_nprobe: usize,
    simd: &K,
    out: &mut Vec<ProbedCluster>,
) -> OspreyResult<()> {
    let mut heap = BoundedTopKHeap::<CAP>::try_new(eff_nprobe)?;
    for k in 0..k_total {
        let centroid = &centroids[k * padded_dim..(k + 1) * padded_dim];
        let dot = simd.dot_f32_f16(q_vec, centroid);
        let k_u32 = u32::try_from(k).map_err(|_| OspreyError::InvalidDimension { dim: k })?;
        heap.push(ScoredCandidate {
            score: dot,
            doc_id: u64::from(k_u32),
            entry_idx: k_u32,
            minipage_id: 0,
            slot_idx: 0,
        });
    }
    out.extend(
        heap.into_sorted_slice()
            .iter()
            .map(|c| (c.entry_idx, c.score)),
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn search_candidates_impl<'a, K, F, A>(
    header: &OsiHeader,
    signs: &RademacherSigns,
    centroids_f16: &[f16],
    query: &[f32],
    nprobe: usize,
    top_m: usize,
    simd: &K,
    filter: &F,
    cluster_accessor: A,
) -> OspreyResult<Vec<ScoredCandidate>>
where
    K: RabitqScanKernel + ?Sized,
    F: EntryFilter,
    A: Fn(usize) -> OspreyResult<(&'a [RabitqEntryHeader], &'a [u8], u32)>,
{
    let orig_dim = usize::from(header.orig_dim);
    let padded_dim = usize::from(header.padded_dim);
    if query.len() != orig_dim {
        return Err(OspreyError::InvalidDimension { dim: query.len() });
    }
    if nprobe == 0 || top_m == 0 || header.num_vectors == 0 {
        // Still validate finiteness of `query` even when returning empty.
        for (index, &val) in query.iter().enumerate() {
            if !val.is_finite() {
                return Err(OspreyError::NonFiniteEmbedding { index });
            }
        }
        return Ok(Vec::new());
    }

    with_scan_scratch(|scratch| {
        if scratch.q_buf.len() != padded_dim {
            scratch.q_buf.resize(padded_dim, 0.0_f32);
        }
        prepare_and_pad_vector(query, &mut scratch.q_buf)?;

        if (header.flags & OSI_FLAG_UNROTATED_CENTROIDS) != 0 {
            probe_clusters_into(
                centroids_f16,
                header.num_clusters,
                header.padded_dim,
                &scratch.q_buf,
                nprobe,
                simd,
                &mut scratch.probed,
            )?;
            simd.fwht_rotate_in_place(&mut scratch.q_buf, signs.xor_masks())?;
        } else {
            simd.fwht_rotate_in_place(&mut scratch.q_buf, signs.xor_masks())?;
            probe_clusters_into(
                centroids_f16,
                header.num_clusters,
                header.padded_dim,
                &scratch.q_buf,
                nprobe,
                simd,
                &mut scratch.probed,
            )?;
        }

        let probed = scratch.probed.as_slice();
        if header.is_progressive_4bit() {
            let (query_abs_sum, q_scale) = TransposedQuery4Bit::quantize_into_slices(
                &scratch.q_buf,
                &mut scratch.planes,
                &mut scratch.sign_mask,
            );
            let words_per_vec = padded_dim / 64;
            let inv_dim = 1.0_f32 / f32::from(header.padded_dim);
            let q_planes = scratch.planes.as_slice();
            let q_signs = scratch.sign_mask.as_slice();
            let prog_buf = &mut scratch.prog_cands;
            if top_m <= 512 {
                scan_probed_clusters_progressive_4bit_cap::<512, K, F, A>(
                    probed,
                    words_per_vec,
                    inv_dim,
                    q_scale,
                    query_abs_sum,
                    q_planes,
                    q_signs,
                    top_m,
                    simd,
                    filter,
                    prog_buf,
                    cluster_accessor,
                )
            } else if top_m <= 1024 {
                scan_probed_clusters_progressive_4bit_cap::<1024, K, F, A>(
                    probed,
                    words_per_vec,
                    inv_dim,
                    q_scale,
                    query_abs_sum,
                    q_planes,
                    q_signs,
                    top_m,
                    simd,
                    filter,
                    prog_buf,
                    cluster_accessor,
                )
            } else if top_m <= 2048 {
                scan_probed_clusters_progressive_4bit_cap::<2048, K, F, A>(
                    probed,
                    words_per_vec,
                    inv_dim,
                    q_scale,
                    query_abs_sum,
                    q_planes,
                    q_signs,
                    top_m,
                    simd,
                    filter,
                    prog_buf,
                    cluster_accessor,
                )
            } else if top_m <= 4096 {
                scan_probed_clusters_progressive_4bit_cap::<4096, K, F, A>(
                    probed,
                    words_per_vec,
                    inv_dim,
                    q_scale,
                    query_abs_sum,
                    q_planes,
                    q_signs,
                    top_m,
                    simd,
                    filter,
                    prog_buf,
                    cluster_accessor,
                )
            } else {
                Err(OspreyError::InvalidDimension { dim: top_m })
            }
        } else if header.is_4bit() {
            let (lut_scale, bias_sum) =
                FastScanQueryLut4Bit::synthesize_into(&scratch.q_buf, &mut scratch.lut_u8);
            let q_lut_u8 = scratch.lut_u8.as_slice();
            if top_m <= 512 {
                scan_probed_clusters_4bit_cap::<512, K, F, A>(
                    probed,
                    header.padded_dim,
                    lut_scale,
                    bias_sum,
                    q_lut_u8,
                    top_m,
                    simd,
                    filter,
                    cluster_accessor,
                )
            } else if top_m <= 1024 {
                scan_probed_clusters_4bit_cap::<1024, K, F, A>(
                    probed,
                    header.padded_dim,
                    lut_scale,
                    bias_sum,
                    q_lut_u8,
                    top_m,
                    simd,
                    filter,
                    cluster_accessor,
                )
            } else if top_m <= 2048 {
                scan_probed_clusters_4bit_cap::<2048, K, F, A>(
                    probed,
                    header.padded_dim,
                    lut_scale,
                    bias_sum,
                    q_lut_u8,
                    top_m,
                    simd,
                    filter,
                    cluster_accessor,
                )
            } else if top_m <= 4096 {
                scan_probed_clusters_4bit_cap::<4096, K, F, A>(
                    probed,
                    header.padded_dim,
                    lut_scale,
                    bias_sum,
                    q_lut_u8,
                    top_m,
                    simd,
                    filter,
                    cluster_accessor,
                )
            } else {
                Err(OspreyError::InvalidDimension { dim: top_m })
            }
        } else {
            let (query_abs_sum, q_scale) = TransposedQuery4Bit::quantize_into_slices(
                &scratch.q_buf,
                &mut scratch.planes,
                &mut scratch.sign_mask,
            );

            let words_per_vec = padded_dim / 64;
            let inv_dim = 1.0_f32 / f32::from(header.padded_dim);
            let q_planes = scratch.planes.as_slice();
            let q_signs = scratch.sign_mask.as_slice();

            if top_m <= 512 {
                scan_probed_clusters_cap::<512, K, F, A>(
                    probed,
                    words_per_vec,
                    inv_dim,
                    q_scale,
                    query_abs_sum,
                    q_planes,
                    q_signs,
                    top_m,
                    simd,
                    filter,
                    cluster_accessor,
                )
            } else if top_m <= 1024 {
                scan_probed_clusters_cap::<1024, K, F, A>(
                    probed,
                    words_per_vec,
                    inv_dim,
                    q_scale,
                    query_abs_sum,
                    q_planes,
                    q_signs,
                    top_m,
                    simd,
                    filter,
                    cluster_accessor,
                )
            } else if top_m <= 2048 {
                scan_probed_clusters_cap::<2048, K, F, A>(
                    probed,
                    words_per_vec,
                    inv_dim,
                    q_scale,
                    query_abs_sum,
                    q_planes,
                    q_signs,
                    top_m,
                    simd,
                    filter,
                    cluster_accessor,
                )
            } else if top_m <= 4096 {
                scan_probed_clusters_cap::<4096, K, F, A>(
                    probed,
                    words_per_vec,
                    inv_dim,
                    q_scale,
                    query_abs_sum,
                    q_planes,
                    q_signs,
                    top_m,
                    simd,
                    filter,
                    cluster_accessor,
                )
            } else {
                Err(OspreyError::InvalidDimension { dim: top_m })
            }
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn scan_probed_clusters_cap<'a, const CAP: usize, K, F, A>(
    probed: &[(u32, f32)],
    words_per_vec: usize,
    inv_dim: f32,
    q_scale: f32,
    query_abs_sum: i32,
    q_planes: &[[u64; 4]],
    q_signs: &[u64],
    top_m: usize,
    simd: &K,
    filter: &F,
    cluster_accessor: A,
) -> OspreyResult<Vec<ScoredCandidate>>
where
    K: RabitqScanKernel + ?Sized,
    F: EntryFilter,
    A: Fn(usize) -> OspreyResult<(&'a [RabitqEntryHeader], &'a [u8], u32)>,
{
    let mut heap = BoundedTopKHeap::<CAP>::try_new(top_m)?;
    let q_scale_inv_dim = q_scale * inv_dim;
    const PREFETCH_AHEAD: usize = 2;

    for &(cluster_id, centroid_ip) in probed {
        let k_idx = usize::try_from(cluster_id)
            .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
        let (cluster_entries, cluster_code_bytes, base_entry_idx) = cluster_accessor(k_idx)?;
        let cluster_codes: &[u64] = bytemuck::try_cast_slice(cluster_code_bytes).map_err(|_| {
            OspreyError::UnalignedMemory {
                address: cluster_code_bytes.as_ptr() as usize,
                required_align: CACHE_LINE_ALIGN,
            }
        })?;
        let count = cluster_entries.len();

        for (local_idx, entry) in cluster_entries.iter().enumerate() {
            if local_idx + PREFETCH_AHEAD < count {
                let pf_idx = local_idx + PREFETCH_AHEAD;
                let pf_code_start = pf_idx * words_per_vec;
                let pf_code_end = pf_code_start + words_per_vec;
                let pf_bytes: &[u8] =
                    bytemuck::cast_slice(&cluster_codes[pf_code_start..pf_code_end]);
                simd.prefetch_l1(pf_bytes);
                if pf_bytes.len() > 64 {
                    simd.prefetch_l1(&pf_bytes[64..]);
                }
                simd.prefetch_l1(bytemuck::bytes_of(&cluster_entries[pf_idx]));
            }

            let entry_idx_u32 =
                base_entry_idx.saturating_add(u32::try_from(local_idx).unwrap_or(u32::MAX));
            if !filter.allow(entry_idx_u32, entry.doc_id) {
                continue;
            }

            let code_start = local_idx * words_per_vec;
            let entry_bits = &cluster_codes[code_start..code_start + words_per_vec];
            let dot_int = simd.rabitq_4bit_dot_u64(q_planes, q_signs, query_abs_sum, entry_bits);

            let res_norm = entry.res_norm.to_f32();
            let l1_factor = entry.l1_norm_factor.to_f32().max(RABITQ_L1_MIN_NORMAL_F32);
            #[allow(clippy::cast_precision_loss)]
            let est_score =
                centroid_ip + res_norm * (((dot_int as f32) * q_scale_inv_dim) / l1_factor);

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

#[allow(clippy::too_many_arguments)]
fn scan_probed_clusters_4bit_cap<'a, const CAP: usize, K, F, A>(
    probed: &[(u32, f32)],
    padded_dim_u16: u16,
    lut_scale: f32,
    bias_sum: f32,
    q_lut_u8: &[u8],
    top_m: usize,
    simd: &K,
    filter: &F,
    cluster_accessor: A,
) -> OspreyResult<Vec<ScoredCandidate>>
where
    K: RabitqScanKernel + ?Sized,
    F: EntryFilter,
    A: Fn(usize) -> OspreyResult<(&'a [RabitqEntryHeader], &'a [u8], u32)>,
{
    let mut heap = BoundedTopKHeap::<CAP>::try_new(top_m)?;
    let padded_dim = usize::from(padded_dim_u16);
    let block_byte_len = fastscan_block_bytes(padded_dim_u16)?;
    let mut block_sums = [0_u32; FASTSCAN_BLOCK_SIZE];

    for &(cluster_id, centroid_ip) in probed {
        let k_idx = usize::try_from(cluster_id)
            .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
        let (cluster_entries, cluster_code_bytes, base_entry_idx) = cluster_accessor(k_idx)?;
        let count = cluster_entries.len();
        let num_blocks = count.div_ceil(FASTSCAN_BLOCK_SIZE);
        let needed_bytes =
            num_blocks
                .checked_mul(block_byte_len)
                .ok_or(OspreyError::BufferTooSmall {
                    needed: usize::MAX,
                    actual: cluster_code_bytes.len(),
                })?;
        if cluster_code_bytes.len() < needed_bytes {
            return Err(OspreyError::BufferTooSmall {
                needed: needed_bytes,
                actual: cluster_code_bytes.len(),
            });
        }

        for block_idx in 0..num_blocks {
            let block_start = block_idx * block_byte_len;
            let block = &cluster_code_bytes[block_start..block_start + block_byte_len];
            let entry_start = block_idx * FASTSCAN_BLOCK_SIZE;
            if block_idx + 1 < num_blocks {
                let next_block_start = block_start + block_byte_len;
                simd.prefetch_l1(&cluster_code_bytes[next_block_start..]);
                let next_entry = entry_start + FASTSCAN_BLOCK_SIZE;
                if next_entry < count {
                    simd.prefetch_l1(bytemuck::bytes_of(&cluster_entries[next_entry]));
                }
            }
            simd.rabitq_4bit_fastscan_block(q_lut_u8, padded_dim, block, &mut block_sums);
            let entry_end = (entry_start + FASTSCAN_BLOCK_SIZE).min(count);
            for (lane, &sum_lut_u32) in block_sums.iter().enumerate().take(entry_end - entry_start)
            {
                let local_idx = entry_start + lane;
                let entry = &cluster_entries[local_idx];
                let entry_idx_u32 =
                    base_entry_idx.saturating_add(u32::try_from(local_idx).unwrap_or(u32::MAX));
                if !filter.allow(entry_idx_u32, entry.doc_id) {
                    continue;
                }
                let est_score = OsiIndex::reconstruct_asymmetric_score_4bit(
                    centroid_ip,
                    entry,
                    lut_scale,
                    bias_sum,
                    sum_lut_u32,
                    padded_dim_u16,
                );
                heap.push(ScoredCandidate {
                    score: est_score,
                    doc_id: entry.doc_id,
                    entry_idx: entry_idx_u32,
                    minipage_id: entry.minipage_id,
                    slot_idx: entry.slot_idx,
                });
            }
        }
    }

    Ok(heap.into_sorted_slice().to_vec())
}

#[inline]
fn prog_sift_up_min(heap: &mut [ProgCandidate], mut idx: usize) {
    while idx > 0 {
        let parent = (idx - 1) >> 1;
        if heap[idx]
            .score_1bit
            .total_cmp(&heap[parent].score_1bit)
            .is_lt()
        {
            heap.swap(idx, parent);
            idx = parent;
        } else {
            break;
        }
    }
}

#[inline]
fn prog_sift_down_min(heap: &mut [ProgCandidate], mut idx: usize) {
    let len = heap.len();
    loop {
        let left = (idx << 1) + 1;
        if left >= len {
            break;
        }
        let right = left + 1;
        let smallest = if right < len
            && heap[right]
                .score_1bit
                .total_cmp(&heap[left].score_1bit)
                .is_lt()
        {
            right
        } else {
            left
        };
        if heap[smallest]
            .score_1bit
            .total_cmp(&heap[idx].score_1bit)
            .is_lt()
        {
            heap.swap(idx, smallest);
            idx = smallest;
        } else {
            break;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn scan_probed_clusters_progressive_4bit_cap<'a, const CAP: usize, K, F, A>(
    probed: &[(u32, f32)],
    words_per_vec: usize,
    inv_dim: f32,
    q_scale: f32,
    query_abs_sum: i32,
    q_planes: &[[u64; 4]],
    q_signs: &[u64],
    top_m: usize,
    simd: &K,
    filter: &F,
    prog_buf: &mut Vec<ProgCandidate>,
    cluster_accessor: A,
) -> OspreyResult<Vec<ScoredCandidate>>
where
    K: RabitqScanKernel + ?Sized,
    F: EntryFilter,
    A: Fn(usize) -> OspreyResult<(&'a [RabitqEntryHeader], &'a [u8], u32)>,
{
    let prog_k = (top_m.saturating_mul(8)).clamp(256, 1024);
    prog_buf.clear();
    if prog_buf.capacity() < prog_k {
        prog_buf.reserve(prog_k - prog_buf.capacity());
    }

    let four_q_scale_inv_dim = 4.0_f32 * q_scale * inv_dim;
    let half_q_scale_inv_dim = 0.5_f32 * q_scale * inv_dim;
    const BATCH_VECS: usize = 256;
    let mut dots_buf = [0_i32; BATCH_VECS];

    // Phase A: Stream only MSB plane `b3` (`1-bit` = 1/4 bandwidth) across all probed clusters.
    for &(cluster_id, centroid_ip) in probed {
        let k_idx = usize::try_from(cluster_id)
            .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
        let (cluster_entries, cluster_code_bytes, base_entry_idx) = cluster_accessor(k_idx)?;
        let count = cluster_entries.len();
        let plane3_words = count * words_per_vec;
        let needed_words = plane3_words * 4;
        let cluster_words: &[u64] = bytemuck::try_cast_slice(cluster_code_bytes).map_err(|_| {
            OspreyError::UnalignedMemory {
                address: cluster_code_bytes.as_ptr() as usize,
                required_align: CACHE_LINE_ALIGN,
            }
        })?;
        if cluster_words.len() < needed_words {
            return Err(OspreyError::BufferTooSmall {
                needed: needed_words * size_of::<u64>(),
                actual: cluster_code_bytes.len(),
            });
        }
        let plane3 = &cluster_words[..plane3_words];

        let mut base = 0usize;
        while base < count {
            let chunk_len = (count - base).min(BATCH_VECS);
            let w_start = base * words_per_vec;
            let w_end = (base + chunk_len) * words_per_vec;
            simd.rabitq_1bit_batch_dots_u64(
                q_planes,
                q_signs,
                query_abs_sum,
                &plane3[w_start..w_end],
                words_per_vec,
                &mut dots_buf[..chunk_len],
            );

            for (offset, &d3) in dots_buf.iter().enumerate().take(chunk_len) {
                if d3 <= 0 && prog_buf.len() == prog_k && centroid_ip <= prog_buf[0].score_1bit {
                    continue;
                }
                let local_idx = base + offset;
                let entry = &cluster_entries[local_idx];
                let entry_idx_u32 =
                    base_entry_idx.saturating_add(u32::try_from(local_idx).unwrap_or(u32::MAX));
                if !filter.allow(entry_idx_u32, entry.doc_id) {
                    continue;
                }

                let res_norm = entry.res_norm.to_f32();
                let l1_factor = entry.l1_norm_factor.to_f32().max(RABITQ_L1_MIN_NORMAL_F32);
                #[allow(clippy::cast_precision_loss)]
                let score_1bit =
                    centroid_ip + res_norm * (((d3 as f32) * four_q_scale_inv_dim) / l1_factor);
                if !score_1bit.is_finite() {
                    continue;
                }

                let local_u32 = u32::try_from(local_idx).unwrap_or(u32::MAX);
                let cand = ProgCandidate {
                    score_1bit,
                    centroid_ip,
                    k_idx: cluster_id,
                    local_idx: local_u32,
                    entry_idx: entry_idx_u32,
                    d3,
                };
                if prog_buf.len() < prog_k {
                    let pos = prog_buf.len();
                    prog_buf.push(cand);
                    prog_sift_up_min(prog_buf, pos);
                } else if score_1bit > prog_buf[0].score_1bit {
                    prog_buf[0] = cand;
                    prog_sift_down_min(prog_buf, 0);
                }
            }
            base += chunk_len;
        }
    }

    // Group survivors by cluster so `cluster_accessor` runs once per unique cluster.
    prog_buf.sort_unstable_by_key(|c| (c.k_idx, c.local_idx));

    // Phase B: Exact 4-bit reconstruction using `[b2, b1, b0]` for the `<= prog_k` survivors.
    let mut heap = BoundedTopKHeap::<CAP>::try_new(top_m)?;
    let mut cached_k: Option<usize> = None;
    let mut cached_entries: &[RabitqEntryHeader] = &[];
    let mut cached_words: &[u64] = &[];
    let mut cached_plane3_words = 0_usize;
    let mut d_low = [0_i32; 3];
    let n_surv = prog_buf.len();

    for idx in 0..n_surv {
        let cand = prog_buf[idx];
        let k_idx = usize::try_from(cand.k_idx)
            .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
        if cached_k != Some(k_idx) {
            let (entries, code_bytes, _) = cluster_accessor(k_idx)?;
            let words: &[u64] =
                bytemuck::try_cast_slice(code_bytes).map_err(|_| OspreyError::UnalignedMemory {
                    address: code_bytes.as_ptr() as usize,
                    required_align: CACHE_LINE_ALIGN,
                })?;
            cached_k = Some(k_idx);
            cached_entries = entries;
            cached_words = words;
            cached_plane3_words = entries.len() * words_per_vec;
        }
        if idx + 1 < n_surv {
            let next_c = prog_buf[idx + 1];
            if usize::try_from(next_c.k_idx).ok() == cached_k {
                if let Ok(next_local) = usize::try_from(next_c.local_idx) {
                    let next_low = cached_plane3_words + next_local * 3 * words_per_vec;
                    if next_low + 3 * words_per_vec <= cached_words.len() {
                        let pf_bytes: &[u8] = bytemuck::cast_slice(
                            &cached_words[next_low..next_low + 3 * words_per_vec],
                        );
                        simd.prefetch_l1(pf_bytes);
                        if pf_bytes.len() > 64 {
                            simd.prefetch_l1(&pf_bytes[64..]);
                        }
                        if pf_bytes.len() > 128 {
                            simd.prefetch_l1(&pf_bytes[128..]);
                        }
                        simd.prefetch_l1(bytemuck::bytes_of(&cached_entries[next_local]));
                    }
                }
            }
        }

        let local_idx = usize::try_from(cand.local_idx)
            .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
        let low_base = cached_plane3_words + local_idx * 3 * words_per_vec;
        let low = &cached_words[low_base..low_base + 3 * words_per_vec];
        simd.rabitq_1bit_batch_dots_u64(
            q_planes,
            q_signs,
            query_abs_sum,
            low,
            words_per_vec,
            &mut d_low,
        );
        let dot_4bit = 8 * cand.d3 + 4 * d_low[0] + 2 * d_low[1] + d_low[2];
        let entry = &cached_entries[local_idx];
        let res_norm = entry.res_norm.to_f32();
        let l1_factor = entry.l1_norm_factor.to_f32().max(RABITQ_L1_MIN_NORMAL_F32);
        #[allow(clippy::cast_precision_loss)]
        let est_score =
            cand.centroid_ip + res_norm * (((dot_4bit as f32) * half_q_scale_inv_dim) / l1_factor);

        heap.push(ScoredCandidate {
            score: est_score,
            doc_id: entry.doc_id,
            entry_idx: cand.entry_idx,
            minipage_id: entry.minipage_id,
            slot_idx: entry.slot_idx,
        });
    }

    Ok(heap.into_sorted_slice().to_vec())
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
        let block_len = cluster_block_bytes_for_flags(count, padded_dim, flags)?;
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

        // 3. Verify OsiMappedIndex matches OsiIndex identically (both aligned buffer & mmap)
        let osi_bytes = index.to_aligned_bytes().unwrap();
        let mapped_buf = OsiMappedIndex::from_aligned_buffer(osi_bytes).unwrap();
        assert_eq!(
            mapped_buf
                .search_candidates(query_deleted, 4, 20, &simd)
                .unwrap(),
            res_deleted
        );
        assert_eq!(
            mapped_buf
                .search_candidates(query_alive, 4, 10, &simd)
                .unwrap(),
            res_alive
        );

        let mut allowed = RoaringBitmap::new();
        allowed.insert(142);
        allowed.insert(150);
        let filtered_owned = index
            .search_candidates_filtered(query_alive, 4, 10, &simd, Some(&allowed))
            .unwrap();
        let filtered_mapped = mapped_buf
            .search_candidates_filtered(query_alive, 4, 10, &simd, Some(&allowed))
            .unwrap();
        assert_eq!(filtered_owned, filtered_mapped);
        assert_eq!(filtered_mapped[0].doc_id, 142);
    }

    #[test]
    fn four_bit_rabitq_quantize_lut_roundtrip_and_recall() {
        let orig_dim = 128;
        let padded_dim = 128;
        let n = 400;
        let k = 8;
        let fwht_seed = 0xDEAD_BEEF_CAFE_2026;
        let vecs = make_synthetic_vectors(n, orig_dim, 0x4B17_AA55_1234_9876);
        let doc_ids: Vec<u64> = (0..n as u64).map(|i| 1000 + i).collect();
        let locations: Vec<VectorLocation> = (0..n)
            .map(|i| VectorLocation {
                slab_id: 0,
                minipage_id: u32::try_from(i / 32).unwrap(),
                slot_idx: u16::try_from(i % 32).unwrap(),
            })
            .collect();

        let simd = ScalarRabitqKernel;
        let cluster_simd = crate::cluster::ScalarClusterKernel;
        let kmeans = crate::cluster::fit_binary_kmeans(
            &vecs,
            n,
            orig_dim,
            fwht_seed,
            &crate::cluster::BinaryKMeansConfig::new(k),
            &cluster_simd,
        )
        .unwrap();

        let index_4bit = OsiIndex::build_with_bits(
            &vecs,
            &doc_ids,
            &locations,
            &kmeans,
            orig_dim,
            fwht_seed,
            RoaringBitmap::new(),
            4,
            &simd,
        )
        .unwrap();
        assert_eq!(index_4bit.rabitq_bits(), 4);
        assert!(index_4bit.header().is_4bit());

        let aligned = index_4bit.to_aligned_bytes().unwrap();
        let roundtrip = OsiIndex::from_bytes(aligned.as_slice()).unwrap();
        assert_eq!(roundtrip.rabitq_bits(), 4);
        assert_eq!(roundtrip.header().num_vectors, n as u64);
        let mapped = OsiMappedIndex::from_aligned_buffer(aligned).unwrap();

        // Compare top-10 recall@10 against exact ground-truth across 20 queries.
        let mut exact_hits = 0usize;
        let mut total_gt = 0usize;
        for q_idx in 0..20 {
            let query = &vecs[q_idx * orig_dim..(q_idx + 1) * orig_dim];
            let res_owned = index_4bit.search_candidates(query, k, 20, &simd).unwrap();
            let res_rt = roundtrip.search_candidates(query, k, 20, &simd).unwrap();
            let res_mmap = mapped.search_candidates(query, k, 20, &simd).unwrap();
            assert_eq!(res_owned.len(), res_rt.len());
            assert_eq!(res_owned, res_mmap);
            for (a, b) in res_owned.iter().zip(res_rt.iter()) {
                assert_eq!(a.doc_id, b.doc_id);
                assert!((a.score - b.score).abs() < 1e-6);
            }

            let mut exact: Vec<(u64, f32)> = (0..n)
                .map(|i| {
                    let v = &vecs[i * orig_dim..(i + 1) * orig_dim];
                    let dot: f32 = query.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                    (doc_ids[i], dot)
                })
                .collect();
            exact.sort_by(|a, b| b.1.total_cmp(&a.1));
            let gt_top10: std::collections::HashSet<u64> =
                exact.iter().take(10).map(|(id, _)| *id).collect();
            let pred_top20: std::collections::HashSet<u64> =
                res_owned.iter().map(|c| c.doc_id).collect();
            exact_hits += gt_top10.intersection(&pred_top20).count();
            total_gt += gt_top10.len();

            // Self-query must always rank at #1 with near-1.0 score.
            assert_eq!(res_owned[0].doc_id, doc_ids[q_idx]);
            assert!((res_owned[0].score - 1.0).abs() < 0.05);
        }
        #[allow(clippy::cast_precision_loss)]
        let recall = exact_hits as f32 / total_gt as f32;
        assert!(
            recall >= 0.99,
            "4-bit RaBitQ top-20 recall of GT@10 was {recall:.4} (< 0.99)"
        );

        // Verify FastScan LUT & scalar block kernel directly.
        let mut unit_res = vec![0.0_f32; padded_dim];
        unit_res[0] = 1.0;
        let mut codes = vec![0_u8; padded_dim];
        let factor = quantize_4bit_unit_residual(&unit_res, &mut codes).unwrap();
        assert!(factor.to_f32() > 0.0);
        let lut = FastScanQueryLut4Bit::try_from_rotated_query(&unit_res).unwrap();
        assert_eq!(lut.lut_u8.len(), padded_dim * 16);
    }

    #[test]
    fn four_bit_progressive_high_dim_roundtrip_and_recall() {
        let orig_dim = 384;
        let n = 320;
        let k = 8;
        let fwht_seed = 0xCAFE_BABE_2026_0512;
        let vecs = make_synthetic_vectors(n, orig_dim, 0x9988_7766_5544_3322);
        let doc_ids: Vec<u64> = (0..n as u64).map(|i| 5000 + i).collect();
        let locations: Vec<VectorLocation> = (0..n)
            .map(|i| VectorLocation {
                slab_id: 0,
                minipage_id: u32::try_from(i / 32).unwrap(),
                slot_idx: u16::try_from(i % 32).unwrap(),
            })
            .collect();

        let simd = ScalarRabitqKernel;
        let cluster_simd = crate::cluster::ScalarClusterKernel;
        let kmeans = crate::cluster::fit_binary_kmeans(
            &vecs,
            n,
            orig_dim,
            fwht_seed,
            &crate::cluster::BinaryKMeansConfig::new(k),
            &cluster_simd,
        )
        .unwrap();

        let index_prog = OsiIndex::build_with_bits(
            &vecs,
            &doc_ids,
            &locations,
            &kmeans,
            orig_dim,
            fwht_seed,
            RoaringBitmap::new(),
            4,
            &simd,
        )
        .unwrap();
        assert_eq!(index_prog.rabitq_bits(), 4);
        assert!(index_prog.header().is_progressive_4bit());

        let aligned = index_prog.to_aligned_bytes().unwrap();
        let roundtrip = OsiIndex::from_bytes(aligned.as_slice()).unwrap();
        assert!(roundtrip.header().is_progressive_4bit());
        let mapped = OsiMappedIndex::from_aligned_buffer(aligned).unwrap();

        let mut exact_hits = 0usize;
        let mut total_gt = 0usize;
        for q_idx in 0..16 {
            let query = &vecs[q_idx * orig_dim..(q_idx + 1) * orig_dim];
            let res_owned = index_prog.search_candidates(query, k, 20, &simd).unwrap();
            let res_rt = roundtrip.search_candidates(query, k, 20, &simd).unwrap();
            let res_mmap = mapped.search_candidates(query, k, 20, &simd).unwrap();
            assert_eq!(res_owned, res_rt);
            assert_eq!(res_owned, res_mmap);
            assert_eq!(res_owned[0].doc_id, doc_ids[q_idx]);

            let mut exact: Vec<(u64, f32)> = (0..n)
                .map(|i| {
                    let v = &vecs[i * orig_dim..(i + 1) * orig_dim];
                    let dot: f32 = query.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                    (doc_ids[i], dot)
                })
                .collect();
            exact.sort_by(|a, b| b.1.total_cmp(&a.1));
            let gt_top10: std::collections::HashSet<u64> =
                exact.iter().take(10).map(|(id, _)| *id).collect();
            let pred_top20: std::collections::HashSet<u64> =
                res_owned.iter().map(|c| c.doc_id).collect();
            exact_hits += gt_top10.intersection(&pred_top20).count();
            total_gt += gt_top10.len();
        }
        #[allow(clippy::cast_precision_loss)]
        let recall = exact_hits as f32 / total_gt as f32;
        assert!(recall >= 0.99, "progressive 4-bit recall was {recall:.4}");
    }
}
