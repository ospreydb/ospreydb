//! `1-Bit Binary K-Means` clustering engine (`arXiv:2608.14648v1`).
//!
//! Combines FWHT orthogonal rotation, 32x-compressed 1-bit binary packing,
//! deterministic `1-Bit K-Means++` initialization, lock-free thread-local
//! majority-vote updates via `rayon`, empty cluster splitting, early stopping,
//! and exact `f64`-accumulated `f16` centroid materialization.

use std::cmp::Ordering;

use half::f16;
#[cfg(not(miri))]
use rayon::prelude::*;

use crate::aligned::AlignedBuffer;
use crate::error::{OspreyError, OspreyResult};
use crate::math::padding::NORM_EPSILON;
use crate::math::{
    padded_dimension, prepare_and_pad_vector, RademacherSigns, SplitMix64, F32_SIGN_BIT_MASK,
};

/// Hardware-accelerated or scalar kernel hooks consumed by `1-Bit Binary K-Means`.
///
/// Implemented by [`ScalarClusterKernel`] in `osprey-core` and by all `SimdBackend`
/// types (including `SimdEngine`) in `osprey-simd`, keeping the workspace crate DAG acyclic.
pub trait BinaryKMeansKernel: Send + Sync {
    /// Applies XOR sign masks and orthogonal FWHT `O(d log d)` in-place.
    fn fwht_rotate_in_place(&self, data: &mut [f32], signs_xor_mask: &[u32]) -> OspreyResult<()>;

    /// Computes Hamming distance `sum(popcnt(a[w] ^ b[w]))` over 64-bit word slices.
    fn hamming_distance_u64(&self, a: &[u64], b: &[u64]) -> u32;

    /// Finds `(best_idx, best_hamming_dist)` among `num_centroids` binary centroids.
    fn batch_nearest_binary_centroid(
        &self,
        vector_bits: &[u64],
        centroids_bits: &[u64],
        words_per_vec: usize,
        num_centroids: usize,
    ) -> (usize, u32);
}

/// Portable scalar implementation of [`BinaryKMeansKernel`] for standalone `osprey-core` usage.
#[derive(Copy, Clone, Debug, Default)]
pub struct ScalarClusterKernel;

impl BinaryKMeansKernel for ScalarClusterKernel {
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

    #[inline]
    fn hamming_distance_u64(&self, a: &[u64], b: &[u64]) -> u32 {
        let n = a.len().min(b.len());
        let mut acc0 = 0_u32;
        let mut acc1 = 0_u32;
        let mut acc2 = 0_u32;
        let mut acc3 = 0_u32;
        let mut i = 0usize;

        while i + 4 <= n {
            acc0 += (a[i] ^ b[i]).count_ones();
            acc1 += (a[i + 1] ^ b[i + 1]).count_ones();
            acc2 += (a[i + 2] ^ b[i + 2]).count_ones();
            acc3 += (a[i + 3] ^ b[i + 3]).count_ones();
            i += 4;
        }

        let mut total = acc0 + acc1 + acc2 + acc3;
        while i < n {
            total += (a[i] ^ b[i]).count_ones();
            i += 1;
        }
        total
    }

    fn batch_nearest_binary_centroid(
        &self,
        vector_bits: &[u64],
        centroids_bits: &[u64],
        words_per_vec: usize,
        num_centroids: usize,
    ) -> (usize, u32) {
        if words_per_vec == 0 || num_centroids == 0 || vector_bits.len() < words_per_vec {
            return (0, u32::MAX);
        }
        let valid_centroids = num_centroids.min(centroids_bits.len() / words_per_vec);
        if valid_centroids == 0 {
            return (0, u32::MAX);
        }

        let v = &vector_bits[..words_per_vec];
        let mut best_idx = 0usize;
        let mut best_dist = u32::MAX;
        let mut c = 0usize;

        while c + 4 <= valid_centroids {
            let base = c * words_per_vec;
            let c0 = &centroids_bits[base..base + words_per_vec];
            let c1 = &centroids_bits[base + words_per_vec..base + 2 * words_per_vec];
            let c2 = &centroids_bits[base + 2 * words_per_vec..base + 3 * words_per_vec];
            let c3 = &centroids_bits[base + 3 * words_per_vec..base + 4 * words_per_vec];

            let mut acc0 = 0_u32;
            let mut acc1 = 0_u32;
            let mut acc2 = 0_u32;
            let mut acc3 = 0_u32;

            for w in 0..words_per_vec {
                let vw = v[w];
                acc0 += (vw ^ c0[w]).count_ones();
                acc1 += (vw ^ c1[w]).count_ones();
                acc2 += (vw ^ c2[w]).count_ones();
                acc3 += (vw ^ c3[w]).count_ones();
            }

            if acc0 < best_dist {
                best_dist = acc0;
                best_idx = c;
            }
            if acc1 < best_dist {
                best_dist = acc1;
                best_idx = c + 1;
            }
            if acc2 < best_dist {
                best_dist = acc2;
                best_idx = c + 2;
            }
            if acc3 < best_dist {
                best_dist = acc3;
                best_idx = c + 3;
            }

            c += 4;
        }

        while c < valid_centroids {
            let start = c * words_per_vec;
            let dist = self.hamming_distance_u64(v, &centroids_bits[start..start + words_per_vec]);
            if dist < best_dist {
                best_dist = dist;
                best_idx = c;
            }
            c += 1;
        }

        (best_idx, best_dist)
    }
}

/// Configuration parameters for `1-Bit Binary K-Means` training.
#[derive(Clone, Debug, PartialEq)]
pub struct BinaryKMeansConfig {
    /// Number of clusters `K` to partition the dataset into.
    pub num_clusters: usize,
    /// Maximum number of 1-bit Lloyd iterations (default: `25`).
    pub max_iters: usize,
    /// Relative Hamming inertia improvement threshold for early stopping (default: `1e-4`).
    pub tolerance: f64,
    /// Deterministic `SplitMix64` seed for `1-Bit K-Means++` initialization.
    pub seed: u64,
    /// Whether to L2-normalize materialized `f16` centroids (default: `true`).
    pub spherical: bool,
}

impl BinaryKMeansConfig {
    /// Creates a `BinaryKMeansConfig` with canonical defaults for `num_clusters`.
    #[inline]
    pub const fn new(num_clusters: usize) -> Self {
        Self {
            num_clusters,
            max_iters: 25,
            tolerance: 1e-4,
            seed: 0x4F53_5052_0002,
            spherical: true,
        }
    }
}

impl Default for BinaryKMeansConfig {
    #[inline]
    fn default() -> Self {
        Self::new(64)
    }
}

/// Output of `1-Bit Binary K-Means` training, ready for `.osi` serialization and `RaBitQ` encoding.
pub struct BinaryKMeansResult {
    /// Contiguous `K * padded_dim` centroid table in `f16`, 64-byte aligned for `.osi`.
    pub centroids_f16: AlignedBuffer<f16>,
    /// Final 1-bit binary centroids (`K * words_per_vec` `u64` words).
    pub binary_centroids: Vec<u64>,
    /// Cluster label in `0..K` for each of the `N` input vectors.
    pub assignments: Vec<u32>,
    /// Total Hamming inertia (`sum_i hamming(x_i, c_{assign(i)})`) after the final step.
    pub final_hamming_inertia: u64,
    /// Number of iterations executed before convergence or `max_iters`.
    pub iterations_run: usize,
}

impl std::fmt::Debug for BinaryKMeansResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BinaryKMeansResult")
            .field("centroids_f16_len", &self.centroids_f16.len())
            .field("binary_centroids_len", &self.binary_centroids.len())
            .field("assignments_len", &self.assignments.len())
            .field("final_hamming_inertia", &self.final_hamming_inertia)
            .field("iterations_run", &self.iterations_run)
            .finish()
    }
}

/// Compile-time lookup table expanding one packed byte into 8 bit-count increments (`[u32; 8]`).
const BYTE_BIT_LUT: [[u32; 8]; 256] = {
    let mut lut = [[0_u32; 8]; 256];
    let mut byte = 0usize;
    while byte < 256 {
        let mut bit = 0usize;
        while bit < 8 {
            lut[byte][bit] = if ((byte >> bit) & 1) == 1 { 1 } else { 0 };
            bit += 1;
        }
        byte += 1;
    }
    lut
};

#[inline]
fn pack_f32_signs_le(rotated: &[f32], out_words: &mut [u64]) {
    for (w, out_word) in out_words.iter_mut().enumerate() {
        let mut word = 0_u64;
        let base = w * 64;
        for b in 0..64 {
            let val = rotated[base + b];
            // Bit 1 iff `val >= 0.0` (including +0.0 and -0.0), matching `reference/poc_reference.py`.
            if val.is_sign_positive() || val == 0.0 {
                word |= 1_u64 << b;
            }
        }
        *out_word = word;
    }
}

/// Rotates a batch of raw `f32` vectors via FWHT in parallel (`rayon`) and packs them
/// into 1-bit `[u64; padded_dim / 64]` words, reducing training memory footprint by 32x.
///
/// Uses one thread-local `Vec<f32>` scratch buffer of length `padded_dim` (zero heap
/// allocations per vector inside the parallel loop).
pub fn preprocess_and_binarize_batch<K: BinaryKMeansKernel + ?Sized>(
    vectors: &[f32],
    num_vectors: usize,
    orig_dim: usize,
    fwht_seed: u64,
    simd: &K,
) -> OspreyResult<Vec<u64>> {
    let padded_dim = padded_dimension(orig_dim)?;
    let expected_len = num_vectors
        .checked_mul(orig_dim)
        .ok_or(OspreyError::InvalidDimension { dim: num_vectors })?;
    if vectors.len() < expected_len {
        return Err(OspreyError::BufferTooSmall {
            needed: expected_len * std::mem::size_of::<f32>(),
            actual: std::mem::size_of_val(vectors),
        });
    }
    if num_vectors == 0 {
        return Ok(Vec::new());
    }

    let words_per_vec = padded_dim / 64;
    let signs = RademacherSigns::from_seed(padded_dim, fwht_seed)?;
    let xor_masks = signs.xor_masks();
    let mut packed_bits = vec![0_u64; num_vectors * words_per_vec];

    #[cfg(not(miri))]
    packed_bits
        .par_chunks_mut(words_per_vec)
        .enumerate()
        .try_for_each_init(
            || vec![0.0_f32; padded_dim],
            |scratch, (i, out_words)| -> OspreyResult<()> {
                let src = &vectors[i * orig_dim..(i + 1) * orig_dim];
                prepare_and_pad_vector(src, scratch)?;
                simd.fwht_rotate_in_place(scratch, xor_masks)?;
                pack_f32_signs_le(scratch, out_words);
                Ok(())
            },
        )?;

    #[cfg(miri)]
    {
        let mut scratch = vec![0.0_f32; padded_dim];
        for (i, out_words) in packed_bits.chunks_mut(words_per_vec).enumerate() {
            let src = &vectors[i * orig_dim..(i + 1) * orig_dim];
            prepare_and_pad_vector(src, &mut scratch)?;
            simd.fwht_rotate_in_place(&mut scratch, xor_masks)?;
            pack_f32_signs_le(&scratch, out_words);
        }
    }

    Ok(packed_bits)
}

/// Packs already-rotated `f32` vectors (`[num_vectors, padded_dim]`) into 1-bit `u64` words.
pub fn binarize_rotated_vectors(
    rotated_vectors: &[f32],
    num_vectors: usize,
    padded_dim: usize,
) -> OspreyResult<Vec<u64>> {
    if padded_dim < 64 || !padded_dim.is_power_of_two() || padded_dim > 65_536 {
        return Err(OspreyError::InvalidDimension { dim: padded_dim });
    }
    let expected_len = num_vectors
        .checked_mul(padded_dim)
        .ok_or(OspreyError::InvalidDimension { dim: num_vectors })?;
    if rotated_vectors.len() < expected_len {
        return Err(OspreyError::BufferTooSmall {
            needed: expected_len * std::mem::size_of::<f32>(),
            actual: std::mem::size_of_val(rotated_vectors),
        });
    }
    for (index, &val) in rotated_vectors[..expected_len].iter().enumerate() {
        if !val.is_finite() {
            return Err(OspreyError::NonFiniteEmbedding { index });
        }
    }
    if num_vectors == 0 {
        return Ok(Vec::new());
    }

    let words_per_vec = padded_dim / 64;
    let mut packed_bits = vec![0_u64; num_vectors * words_per_vec];
    #[cfg(not(miri))]
    packed_bits
        .par_chunks_mut(words_per_vec)
        .enumerate()
        .for_each(|(i, out_words)| {
            let src = &rotated_vectors[i * padded_dim..(i + 1) * padded_dim];
            pack_f32_signs_le(src, out_words);
        });
    #[cfg(miri)]
    for (i, out_words) in packed_bits.chunks_mut(words_per_vec).enumerate() {
        let src = &rotated_vectors[i * padded_dim..(i + 1) * padded_dim];
        pack_f32_signs_le(src, out_words);
    }

    Ok(packed_bits)
}

/// Selects `k` initial binary centroids deterministically using `1-Bit K-Means++`
/// weighted by squared Hamming distance (`D_H(x, nearest)^2`).
pub fn init_kmeans_plusplus_1bit<K: BinaryKMeansKernel + ?Sized>(
    packed_vectors: &[u64],
    num_vectors: usize,
    words_per_vec: usize,
    k: usize,
    seed: u64,
    simd: &K,
) -> OspreyResult<Vec<u64>> {
    if k == 0 || k > num_vectors {
        return Err(OspreyError::InvalidDimension { dim: k });
    }
    if words_per_vec == 0 {
        return Err(OspreyError::InvalidDimension { dim: 0 });
    }
    let expected_words = num_vectors
        .checked_mul(words_per_vec)
        .ok_or(OspreyError::InvalidDimension { dim: num_vectors })?;
    if packed_vectors.len() < expected_words {
        return Err(OspreyError::BufferTooSmall {
            needed: expected_words * std::mem::size_of::<u64>(),
            actual: std::mem::size_of_val(packed_vectors),
        });
    }

    let mut rng = SplitMix64::new(seed);
    let mut centroids = vec![0_u64; k * words_per_vec];

    let first_idx =
        usize::try_from(rng.next_u64() % u64::try_from(num_vectors).unwrap_or(1)).unwrap_or(0);
    centroids[..words_per_vec].copy_from_slice(
        &packed_vectors[first_idx * words_per_vec..(first_idx + 1) * words_per_vec],
    );

    if k == 1 {
        return Ok(centroids);
    }

    let mut min_dist_sq = vec![0_u64; num_vectors];
    {
        let first_centroid = &centroids[..words_per_vec];
        #[cfg(not(miri))]
        min_dist_sq
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, slot)| {
                let vec_bits = &packed_vectors[i * words_per_vec..(i + 1) * words_per_vec];
                let d = u64::from(simd.hamming_distance_u64(vec_bits, first_centroid));
                *slot = d * d;
            });
        #[cfg(miri)]
        for (i, slot) in min_dist_sq.iter_mut().enumerate() {
            let vec_bits = &packed_vectors[i * words_per_vec..(i + 1) * words_per_vec];
            let d = u64::from(simd.hamming_distance_u64(vec_bits, first_centroid));
            *slot = d * d;
        }
    }

    let n_u64 = u64::try_from(num_vectors).unwrap_or(u64::MAX);
    for m in 1..k {
        let total_weight: u128 = min_dist_sq.iter().map(|&w| u128::from(w)).sum();
        let chosen_idx = if total_weight == 0 {
            usize::try_from(rng.next_u64() % n_u64).unwrap_or(0)
        } else {
            let r = rng.next_u64();
            let threshold: u128 = if let Some(prod) = u128::from(r).checked_mul(total_weight) {
                prod >> 64
            } else {
                #[allow(clippy::cast_precision_loss)]
                let unit = ((r >> 11) as f64) * (1.0_f64 / ((1_u64 << 53) as f64));
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    clippy::cast_precision_loss
                )]
                {
                    ((unit * (total_weight as f64)) as u128).min(total_weight - 1)
                }
            };

            let mut prefix = 0_u128;
            let mut found = num_vectors - 1;
            for (idx, &w) in min_dist_sq.iter().enumerate() {
                prefix += u128::from(w);
                if prefix > threshold {
                    found = idx;
                    break;
                }
            }
            found
        };

        let dst_start = m * words_per_vec;
        let src_start = chosen_idx * words_per_vec;
        centroids[dst_start..dst_start + words_per_vec]
            .copy_from_slice(&packed_vectors[src_start..src_start + words_per_vec]);

        let new_centroid = &centroids[dst_start..dst_start + words_per_vec];
        #[cfg(not(miri))]
        min_dist_sq
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, slot)| {
                let vec_bits = &packed_vectors[i * words_per_vec..(i + 1) * words_per_vec];
                let d = u64::from(simd.hamming_distance_u64(vec_bits, new_centroid));
                let d_sq = d * d;
                if d_sq < *slot {
                    *slot = d_sq;
                }
            });
        #[cfg(miri)]
        for (i, slot) in min_dist_sq.iter_mut().enumerate() {
            let vec_bits = &packed_vectors[i * words_per_vec..(i + 1) * words_per_vec];
            let d = u64::from(simd.hamming_distance_u64(vec_bits, new_centroid));
            let d_sq = d * d;
            if d_sq < *slot {
                *slot = d_sq;
            }
        }
    }

    Ok(centroids)
}

#[derive(Clone)]
struct UpdateAccum {
    bit_counts: Vec<u32>,
    cluster_sizes: Vec<u32>,
    cluster_inertia: Vec<u64>,
}

impl UpdateAccum {
    fn new(k: usize, padded_dim: usize) -> Self {
        Self {
            bit_counts: vec![0_u32; k * padded_dim],
            cluster_sizes: vec![0_u32; k],
            cluster_inertia: vec![0_u64; k],
        }
    }

    #[cfg(not(miri))]
    fn merge(mut self, other: Self) -> Self {
        for (a, &b) in self.bit_counts.iter_mut().zip(other.bit_counts.iter()) {
            *a = a.saturating_add(b);
        }
        for (a, &b) in self
            .cluster_sizes
            .iter_mut()
            .zip(other.cluster_sizes.iter())
        {
            *a = a.saturating_add(b);
        }
        for (a, &b) in self
            .cluster_inertia
            .iter_mut()
            .zip(other.cluster_inertia.iter())
        {
            *a = a.saturating_add(b);
        }
        self
    }
}

#[inline]
fn accumulate_vector_bits(row: &mut [u32], vec_bits: &[u64]) {
    for (w, &word) in vec_bits.iter().enumerate() {
        if word == 0 {
            continue;
        }
        let word_base = w * 64;
        for byte_idx in 0..8 {
            let byte = usize::try_from((word >> (byte_idx * 8)) & 0xFF).unwrap_or(0);
            if byte == 0 {
                continue;
            }
            let dst = &mut row[word_base + byte_idx * 8..word_base + byte_idx * 8 + 8];
            let lut = &BYTE_BIT_LUT[byte];
            dst[0] += lut[0];
            dst[1] += lut[1];
            dst[2] += lut[2];
            dst[3] += lut[3];
            dst[4] += lut[4];
            dst[5] += lut[5];
            dst[6] += lut[6];
            dst[7] += lut[7];
        }
    }
}

#[inline]
fn update_majority_vote_centroid(c_words: &mut [u64], counts_row: &[u32], count: u32) {
    let count_u64 = u64::from(count);
    for (w, word_slot) in c_words.iter_mut().enumerate() {
        let prev_word = *word_slot;
        let mut new_word = 0_u64;
        let base = w * 64;
        for b in 0..64 {
            let twice = u64::from(counts_row[base + b]) * 2;
            let bit = match twice.cmp(&count_u64) {
                Ordering::Greater => 1_u64,
                Ordering::Less => 0_u64,
                Ordering::Equal => (prev_word >> b) & 1,
            };
            new_word |= bit << b;
        }
        *word_slot = new_word;
    }
}

fn split_empty_clusters(
    packed_vectors: &[u64],
    words_per_vec: usize,
    k: usize,
    binary_centroids: &mut [u64],
    assignments: &mut [u32],
    point_distances: &mut [u32],
    accum: &mut UpdateAccum,
) {
    let mut effective_inertia = accum.cluster_inertia.clone();

    for c_empty in 0..k {
        if accum.cluster_sizes[c_empty] != 0 {
            continue;
        }

        // Deterministically pick donor with `size >= 2` maximizing `(effective_inertia, size)`.
        let mut best_donor: Option<usize> = None;
        let mut best_inertia = 0_u64;
        let mut best_size = 0_u32;

        for (c, (&sz, &inert)) in accum
            .cluster_sizes
            .iter()
            .zip(effective_inertia.iter())
            .enumerate()
            .take(k)
        {
            if sz < 2 {
                continue;
            }
            let better = match best_donor {
                None => true,
                Some(_) => inert > best_inertia || (inert == best_inertia && sz > best_size),
            };
            if better {
                best_donor = Some(c);
                best_inertia = inert;
                best_size = sz;
            }
        }

        let Some(c_donor) = best_donor else {
            break;
        };

        let donor_u32 = u32::try_from(c_donor).unwrap_or(0);
        let empty_u32 = u32::try_from(c_empty).unwrap_or(0);

        // Locate the vector in `c_donor` farthest in Hamming distance from `c_donor`'s centroid.
        let mut far_idx: Option<usize> = None;
        let mut far_dist = 0_u32;
        for (i, (&assign, &dist)) in assignments.iter().zip(point_distances.iter()).enumerate() {
            if assign == donor_u32 && (far_idx.is_none() || dist > far_dist) {
                far_idx = Some(i);
                far_dist = dist;
            }
        }

        if let Some(v_idx) = far_idx {
            let v_start = v_idx * words_per_vec;
            let c_empty_start = c_empty * words_per_vec;
            binary_centroids[c_empty_start..c_empty_start + words_per_vec]
                .copy_from_slice(&packed_vectors[v_start..v_start + words_per_vec]);

            assignments[v_idx] = empty_u32;
            point_distances[v_idx] = 0;
            accum.cluster_sizes[c_donor] -= 1;
            accum.cluster_sizes[c_empty] = 1;
            effective_inertia[c_donor] = 0;
            accum.cluster_inertia[c_donor] = 0;
        }
    }
}

struct LoopOutput {
    binary_centroids: Vec<u64>,
    assignments: Vec<u32>,
    final_hamming_inertia: u64,
    iterations_run: usize,
}

fn run_1bit_kmeans_loop<K: BinaryKMeansKernel + ?Sized>(
    packed_vectors: &[u64],
    num_vectors: usize,
    padded_dim: usize,
    init_centroids: Vec<u64>,
    config: &BinaryKMeansConfig,
    simd: &K,
) -> OspreyResult<LoopOutput> {
    let k = config.num_clusters;
    let words_per_vec = padded_dim / 64;
    if config.max_iters == 0 {
        return Err(OspreyError::InvalidDimension { dim: 0 });
    }
    if init_centroids.len() != k * words_per_vec {
        return Err(OspreyError::BufferTooSmall {
            needed: k * words_per_vec * std::mem::size_of::<u64>(),
            actual: init_centroids.len() * std::mem::size_of::<u64>(),
        });
    }

    let mut binary_centroids = init_centroids;
    let mut assignments = vec![0_u32; num_vectors];
    let mut point_distances = vec![0_u32; num_vectors];
    let mut prev_inertia = u64::MAX;
    let mut final_inertia = 0_u64;
    let mut iterations_run = 0usize;

    for iter in 0..config.max_iters {
        iterations_run = iter + 1;

        // Parallel Assignment Step via 4x-unrolled XNOR / Hamming + POPCNT.
        #[cfg(not(miri))]
        packed_vectors
            .par_chunks(words_per_vec)
            .zip(assignments.par_iter_mut())
            .zip(point_distances.par_iter_mut())
            .for_each(|((vec_bits, assign_slot), dist_slot)| {
                let (best_idx, best_dist) = simd.batch_nearest_binary_centroid(
                    vec_bits,
                    &binary_centroids,
                    words_per_vec,
                    k,
                );
                *assign_slot = u32::try_from(best_idx).unwrap_or(0);
                *dist_slot = best_dist;
            });
        #[cfg(miri)]
        for ((vec_bits, assign_slot), dist_slot) in packed_vectors
            .chunks(words_per_vec)
            .zip(assignments.iter_mut())
            .zip(point_distances.iter_mut())
        {
            let (best_idx, best_dist) =
                simd.batch_nearest_binary_centroid(vec_bits, &binary_centroids, words_per_vec, k);
            *assign_slot = u32::try_from(best_idx).unwrap_or(0);
            *dist_slot = best_dist;
        }

        #[cfg(not(miri))]
        let current_inertia: u64 = point_distances.par_iter().map(|&d| u64::from(d)).sum();
        #[cfg(miri)]
        let current_inertia: u64 = point_distances.iter().map(|&d| u64::from(d)).sum();
        final_inertia = current_inertia;

        // Early Stopping check.
        let should_stop = if iter > 0 {
            #[allow(clippy::cast_precision_loss)]
            let rel_improvement = (prev_inertia.saturating_sub(current_inertia) as f64)
                / (prev_inertia.max(1) as f64);
            current_inertia == 0
                || (current_inertia <= prev_inertia && rel_improvement < config.tolerance)
        } else {
            current_inertia == 0
        };

        // Lock-free thread-local accumulation + Majority Vote update.
        #[cfg(not(miri))]
        let mut accum = packed_vectors
            .par_chunks(words_per_vec)
            .zip(assignments.par_iter())
            .zip(point_distances.par_iter())
            .fold(
                || UpdateAccum::new(k, padded_dim),
                |mut local, ((vec_bits, &assign), &dist)| {
                    let c = usize::try_from(assign).unwrap_or(0);
                    if c < k {
                        local.cluster_sizes[c] = local.cluster_sizes[c].saturating_add(1);
                        local.cluster_inertia[c] =
                            local.cluster_inertia[c].saturating_add(u64::from(dist));
                        let row = &mut local.bit_counts[c * padded_dim..(c + 1) * padded_dim];
                        accumulate_vector_bits(row, vec_bits);
                    }
                    local
                },
            )
            .reduce(|| UpdateAccum::new(k, padded_dim), UpdateAccum::merge);
        #[cfg(miri)]
        let mut accum = {
            let mut local = UpdateAccum::new(k, padded_dim);
            for ((vec_bits, &assign), &dist) in packed_vectors
                .chunks(words_per_vec)
                .zip(assignments.iter())
                .zip(point_distances.iter())
            {
                let c = usize::try_from(assign).unwrap_or(0);
                if c < k {
                    local.cluster_sizes[c] = local.cluster_sizes[c].saturating_add(1);
                    local.cluster_inertia[c] =
                        local.cluster_inertia[c].saturating_add(u64::from(dist));
                    let row = &mut local.bit_counts[c * padded_dim..(c + 1) * padded_dim];
                    accumulate_vector_bits(row, vec_bits);
                }
            }
            local
        };

        for c in 0..k {
            let count = accum.cluster_sizes[c];
            if count == 0 {
                continue;
            }
            let counts_row = &accum.bit_counts[c * padded_dim..(c + 1) * padded_dim];
            let c_words = &mut binary_centroids[c * words_per_vec..(c + 1) * words_per_vec];
            update_majority_vote_centroid(c_words, counts_row, count);
        }

        // Empty Cluster Splitting recovery.
        if accum.cluster_sizes.contains(&0) {
            split_empty_clusters(
                packed_vectors,
                words_per_vec,
                k,
                &mut binary_centroids,
                &mut assignments,
                &mut point_distances,
                &mut accum,
            );
        }

        if should_stop {
            break;
        }
        prev_inertia = current_inertia;
    }

    Ok(LoopOutput {
        binary_centroids,
        assignments,
        final_hamming_inertia: final_inertia,
        iterations_run,
    })
}

fn finalize_centroids_f16(
    mut sums: Vec<f64>,
    counts: &[u64],
    binary_centroids: &[u64],
    k: usize,
    padded_dim: usize,
    spherical: bool,
) -> OspreyResult<AlignedBuffer<f16>> {
    let words_per_vec = padded_dim / 64;
    let mut centroids_f16 = AlignedBuffer::<f16>::new(k * padded_dim)?;
    #[allow(clippy::cast_precision_loss)]
    let fallback_scale = 1.0_f64 / (padded_dim as f64).sqrt();

    for c in 0..k {
        let count = counts[c];
        let row = &mut sums[c * padded_dim..(c + 1) * padded_dim];

        if count > 0 {
            #[allow(clippy::cast_precision_loss)]
            let inv_count = 1.0_f64 / (count as f64);
            for val in row.iter_mut() {
                *val *= inv_count;
            }
        } else {
            // Defensive fallback if an impossible 0-member cluster occurred.
            let c_words = &binary_centroids[c * words_per_vec..(c + 1) * words_per_vec];
            for (w, &word) in c_words.iter().enumerate() {
                for b in 0..64 {
                    row[w * 64 + b] = if ((word >> b) & 1) == 1 {
                        fallback_scale
                    } else {
                        -fallback_scale
                    };
                }
            }
        }

        if spherical {
            let sq_norm: f64 = row.iter().map(|&x| x * x).sum();
            let norm = sq_norm.sqrt();
            if norm > NORM_EPSILON && norm.is_finite() {
                let inv_norm = 1.0_f64 / norm;
                for val in row.iter_mut() {
                    *val *= inv_norm;
                }
            } else {
                let c_words = &binary_centroids[c * words_per_vec..(c + 1) * words_per_vec];
                for (w, &word) in c_words.iter().enumerate() {
                    for b in 0..64 {
                        row[w * 64 + b] = if ((word >> b) & 1) == 1 {
                            fallback_scale
                        } else {
                            -fallback_scale
                        };
                    }
                }
            }
        }

        let out_row = &mut centroids_f16[c * padded_dim..(c + 1) * padded_dim];
        for (dst, &val) in out_row.iter_mut().zip(row.iter()) {
            #[allow(clippy::cast_possible_truncation)]
            let clamped = if val.is_finite() {
                (val as f32).clamp(-65_504.0_f32, 65_504.0_f32)
            } else {
                0.0_f32
            };
            let converted = f16::from_f32(clamped);
            *dst = if converted.is_finite() {
                converted
            } else {
                f16::ZERO
            };
        }
    }

    Ok(centroids_f16)
}

fn validate_fit_inputs(num_vectors: usize, k: usize) -> OspreyResult<()> {
    if num_vectors == 0 {
        return Err(OspreyError::InvalidDimension { dim: 0 });
    }
    if k == 0 || k > num_vectors || u32::try_from(k).is_err() {
        return Err(OspreyError::InvalidDimension { dim: k });
    }
    Ok(())
}

/// Runs the complete `1-Bit Binary K-Means` training pipeline over raw `f32` vectors:
/// 1. Parallel FWHT rotation + 1-bit packing (`32x` training RAM reduction);
/// 2. Deterministic `1-Bit K-Means++` initialization;
/// 3. Parallel 1-bit assignment, thread-local majority-vote update, empty cluster splitting,
///    and early stopping;
/// 4. Final exact `f64`-accumulated `f16` centroid materialization in a 64B-aligned buffer.
pub fn fit_binary_kmeans<K: BinaryKMeansKernel + ?Sized>(
    vectors: &[f32],
    num_vectors: usize,
    orig_dim: usize,
    fwht_seed: u64,
    config: &BinaryKMeansConfig,
    simd: &K,
) -> OspreyResult<BinaryKMeansResult> {
    let k = config.num_clusters;
    validate_fit_inputs(num_vectors, k)?;
    let padded_dim = padded_dimension(orig_dim)?;
    let words_per_vec = padded_dim / 64;

    // Rotate + binarize batch into 1-bit `[u64; words_per_vec]` (32x RAM savings).
    let packed_bits =
        preprocess_and_binarize_batch(vectors, num_vectors, orig_dim, fwht_seed, simd)?;

    // Deterministic 1-Bit K-Means++.
    let init_centroids = init_kmeans_plusplus_1bit(
        &packed_bits,
        num_vectors,
        words_per_vec,
        k,
        config.seed,
        simd,
    )?;

    // Iterative 1-bit clustering loop.
    let loop_out = run_1bit_kmeans_loop(
        &packed_bits,
        num_vectors,
        padded_dim,
        init_centroids,
        config,
        simd,
    )?;

    // Single final parallel pass over raw vectors -> exact `f64` sums -> `f16` centroids.
    let signs = RademacherSigns::from_seed(padded_dim, fwht_seed)?;
    let xor_masks = signs.xor_masks();

    struct MatState {
        sums: Vec<f64>,
        counts: Vec<u64>,
        scratch: Vec<f32>,
    }

    #[cfg(not(miri))]
    let final_acc = loop_out
        .assignments
        .par_iter()
        .enumerate()
        .try_fold(
            || MatState {
                sums: vec![0.0_f64; k * padded_dim],
                counts: vec![0_u64; k],
                scratch: vec![0.0_f32; padded_dim],
            },
            |mut st, (i, &assign)| -> OspreyResult<MatState> {
                let c = usize::try_from(assign).unwrap_or(0);
                let src = &vectors[i * orig_dim..(i + 1) * orig_dim];
                prepare_and_pad_vector(src, &mut st.scratch)?;
                simd.fwht_rotate_in_place(&mut st.scratch, xor_masks)?;
                st.counts[c] += 1;
                let dst_row = &mut st.sums[c * padded_dim..(c + 1) * padded_dim];
                for (dst, &val) in dst_row.iter_mut().zip(st.scratch.iter()) {
                    *dst += f64::from(val);
                }
                Ok(st)
            },
        )
        .try_reduce(
            || MatState {
                sums: vec![0.0_f64; k * padded_dim],
                counts: vec![0_u64; k],
                scratch: Vec::new(),
            },
            |mut a, b| -> OspreyResult<MatState> {
                for (sa, &sb) in a.sums.iter_mut().zip(b.sums.iter()) {
                    *sa += sb;
                }
                for (ca, &cb) in a.counts.iter_mut().zip(b.counts.iter()) {
                    *ca += cb;
                }
                Ok(a)
            },
        )?;
    #[cfg(miri)]
    let final_acc = {
        let mut st = MatState {
            sums: vec![0.0_f64; k * padded_dim],
            counts: vec![0_u64; k],
            scratch: vec![0.0_f32; padded_dim],
        };
        for (i, &assign) in loop_out.assignments.iter().enumerate() {
            let c = usize::try_from(assign).unwrap_or(0);
            let src = &vectors[i * orig_dim..(i + 1) * orig_dim];
            prepare_and_pad_vector(src, &mut st.scratch)?;
            simd.fwht_rotate_in_place(&mut st.scratch, xor_masks)?;
            st.counts[c] += 1;
            let dst_row = &mut st.sums[c * padded_dim..(c + 1) * padded_dim];
            for (dst, &val) in dst_row.iter_mut().zip(st.scratch.iter()) {
                *dst += f64::from(val);
            }
        }
        st
    };

    let centroids_f16 = finalize_centroids_f16(
        final_acc.sums,
        &final_acc.counts,
        &loop_out.binary_centroids,
        k,
        padded_dim,
        config.spherical,
    )?;

    Ok(BinaryKMeansResult {
        centroids_f16,
        binary_centroids: loop_out.binary_centroids,
        assignments: loop_out.assignments,
        final_hamming_inertia: loop_out.final_hamming_inertia,
        iterations_run: loop_out.iterations_run,
    })
}

/// Runs `1-Bit Binary K-Means` on already FWHT-rotated `f32` vectors (`[num_vectors, padded_dim]`)
/// using deterministic `1-Bit K-Means++` initialization.
pub fn fit_binary_kmeans_from_rotated<K: BinaryKMeansKernel + ?Sized>(
    rotated_vectors: &[f32],
    num_vectors: usize,
    padded_dim: usize,
    config: &BinaryKMeansConfig,
    simd: &K,
) -> OspreyResult<BinaryKMeansResult> {
    let k = config.num_clusters;
    validate_fit_inputs(num_vectors, k)?;
    let packed_bits = binarize_rotated_vectors(rotated_vectors, num_vectors, padded_dim)?;
    let words_per_vec = padded_dim / 64;
    let init_centroids = init_kmeans_plusplus_1bit(
        &packed_bits,
        num_vectors,
        words_per_vec,
        k,
        config.seed,
        simd,
    )?;
    fit_binary_kmeans_from_rotated_with_init(
        rotated_vectors,
        &packed_bits,
        num_vectors,
        padded_dim,
        init_centroids,
        config,
        simd,
    )
}

/// Runs `1-Bit Binary K-Means` on already-rotated `f32` vectors and pre-packed `1-bit` vectors
/// starting from explicit initial binary centroids (useful for golden-vector validation).
pub fn fit_binary_kmeans_from_rotated_with_init<K: BinaryKMeansKernel + ?Sized>(
    rotated_vectors: &[f32],
    packed_bits: &[u64],
    num_vectors: usize,
    padded_dim: usize,
    init_centroids: Vec<u64>,
    config: &BinaryKMeansConfig,
    simd: &K,
) -> OspreyResult<BinaryKMeansResult> {
    let k = config.num_clusters;
    validate_fit_inputs(num_vectors, k)?;
    if padded_dim < 64 || !padded_dim.is_power_of_two() || padded_dim > 65_536 {
        return Err(OspreyError::InvalidDimension { dim: padded_dim });
    }
    let expected_floats = num_vectors
        .checked_mul(padded_dim)
        .ok_or(OspreyError::InvalidDimension { dim: num_vectors })?;
    if rotated_vectors.len() < expected_floats {
        return Err(OspreyError::BufferTooSmall {
            needed: expected_floats * std::mem::size_of::<f32>(),
            actual: std::mem::size_of_val(rotated_vectors),
        });
    }

    let loop_out = run_1bit_kmeans_loop(
        packed_bits,
        num_vectors,
        padded_dim,
        init_centroids,
        config,
        simd,
    )?;

    #[cfg(not(miri))]
    let (sums, counts) = loop_out
        .assignments
        .par_iter()
        .enumerate()
        .fold(
            || (vec![0.0_f64; k * padded_dim], vec![0_u64; k]),
            |(mut sums, mut counts), (i, &assign)| {
                let c = usize::try_from(assign).unwrap_or(0);
                counts[c] += 1;
                let src = &rotated_vectors[i * padded_dim..(i + 1) * padded_dim];
                let dst_row = &mut sums[c * padded_dim..(c + 1) * padded_dim];
                for (dst, &val) in dst_row.iter_mut().zip(src.iter()) {
                    *dst += f64::from(val);
                }
                (sums, counts)
            },
        )
        .reduce(
            || (vec![0.0_f64; k * padded_dim], vec![0_u64; k]),
            |(mut sa, mut ca), (sb, cb)| {
                for (a, &b) in sa.iter_mut().zip(sb.iter()) {
                    *a += b;
                }
                for (a, &b) in ca.iter_mut().zip(cb.iter()) {
                    *a += b;
                }
                (sa, ca)
            },
        );
    #[cfg(miri)]
    let (sums, counts) = {
        let mut sums = vec![0.0_f64; k * padded_dim];
        let mut counts = vec![0_u64; k];
        for (i, &assign) in loop_out.assignments.iter().enumerate() {
            let c = usize::try_from(assign).unwrap_or(0);
            counts[c] += 1;
            let src = &rotated_vectors[i * padded_dim..(i + 1) * padded_dim];
            let dst_row = &mut sums[c * padded_dim..(c + 1) * padded_dim];
            for (dst, &val) in dst_row.iter_mut().zip(src.iter()) {
                *dst += f64::from(val);
            }
        }
        (sums, counts)
    };

    let centroids_f16 = finalize_centroids_f16(
        sums,
        &counts,
        &loop_out.binary_centroids,
        k,
        padded_dim,
        config.spherical,
    )?;

    Ok(BinaryKMeansResult {
        centroids_f16,
        binary_centroids: loop_out.binary_centroids,
        assignments: loop_out.assignments,
        final_hamming_inertia: loop_out.final_hamming_inertia,
        iterations_run: loop_out.iterations_run,
    })
}

/// Ensures `F32_SIGN_BIT_MASK` link stays coherent with `fwht` / `rademacher`.
#[allow(dead_code)]
const _: u32 = F32_SIGN_BIT_MASK;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::aligned::CACHE_LINE_ALIGN;

    fn make_synthetic_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = SplitMix64::new(seed);
        let mut out = vec![0.0_f32; n * dim];
        for val in out.iter_mut() {
            let bits = rng.next_u64() >> 40;
            #[allow(clippy::cast_precision_loss)]
            let unit = (bits as f32) / ((1_u32 << 24) as f32);
            *val = unit * 2.0 - 1.0;
        }
        out
    }

    #[test]
    fn preprocess_and_binarize_achieves_32x_memory_reduction_and_sign_parity() {
        let n = if cfg!(miri) { 4usize } else { 64usize };
        let orig_dim = if cfg!(miri) { 64usize } else { 384usize };
        let padded_dim = padded_dimension(orig_dim).unwrap();
        let vectors = make_synthetic_vectors(n, orig_dim, 0x1234_5678);
        let kernel = ScalarClusterKernel;

        let packed =
            preprocess_and_binarize_batch(&vectors, n, orig_dim, 0x4F53_5052_0001, &kernel)
                .unwrap();

        let raw_padded_bytes = n * padded_dim * std::mem::size_of::<f32>();
        let packed_bytes = packed.len() * std::mem::size_of::<u64>();
        assert_eq!(raw_padded_bytes / packed_bytes, 32);

        // Verify -0.0 and +0.0 both pack to bit 1, matching `x_rot >= 0.0`.
        let rot_edge = vec![-0.0_f32, 0.0_f32, -1e-6_f32, 1e-6_f32];
        let mut padded_edge = vec![0.0_f32; 64];
        padded_edge[..4].copy_from_slice(&rot_edge);
        let bits = binarize_rotated_vectors(&padded_edge, 1, 64).unwrap();
        assert_eq!(bits[0] & 0b1111, 0b1011);
    }

    #[test]
    fn init_kmeans_plusplus_1bit_is_deterministic_and_handles_duplicates() {
        let n = if cfg!(miri) { 8usize } else { 64usize };
        let orig_dim = 128usize;
        let vectors = make_synthetic_vectors(n, orig_dim, 0xABCD_EF01);
        let kernel = ScalarClusterKernel;
        let packed =
            preprocess_and_binarize_batch(&vectors, n, orig_dim, 0x4F53_5052_0002, &kernel)
                .unwrap();

        let words_per_vec = 2usize;
        let k = 8usize;
        let c1 = init_kmeans_plusplus_1bit(&packed, n, words_per_vec, k, 0x9999, &kernel).unwrap();
        let c2 = init_kmeans_plusplus_1bit(&packed, n, words_per_vec, k, 0x9999, &kernel).unwrap();
        assert_eq!(c1, c2);
        assert_eq!(c1.len(), k * words_per_vec);

        // Duplicate dataset (total_weight == 0 branch) must not panic or fail.
        let dup_packed = vec![0xAAAA_5555_AAAA_5555_u64; 16 * 2];
        let dup_c = init_kmeans_plusplus_1bit(&dup_packed, 16, 2, 4, 42, &kernel).unwrap();
        assert_eq!(dup_c.len(), 4 * 2);
    }

    #[test]
    fn fit_binary_kmeans_recovers_empty_clusters_and_early_stops_without_nans() {
        // Degenerate dataset: vectors concentrated at only 3 distinct points, clustered with K = 8.
        let n = if cfg!(miri) { 16usize } else { 100usize };
        let dim = 64usize;
        let k = 8usize;
        let mut vectors = vec![0.0_f32; n * dim];
        for i in 0..n {
            let pattern = i % 3;
            let row = &mut vectors[i * dim..(i + 1) * dim];
            match pattern {
                0 => row[0] = 1.0,
                1 => row[1] = 1.0,
                _ => row[2] = -1.0,
            }
        }

        let mut cfg = BinaryKMeansConfig::new(k);
        cfg.max_iters = 20;
        cfg.seed = 0xDEAD_BEEF;
        let kernel = ScalarClusterKernel;

        let res = fit_binary_kmeans(&vectors, n, dim, 0x4F53_5052_0005, &cfg, &kernel).unwrap();
        assert_eq!(res.assignments.len(), n);
        assert_eq!(res.centroids_f16.len(), k * dim);
        assert_eq!((res.centroids_f16.as_ptr() as usize) % CACHE_LINE_ALIGN, 0);
        assert!(
            res.iterations_run < cfg.max_iters,
            "expected early stopping"
        );

        // All K=8 clusters must be non-empty thanks to Empty Cluster Splitting.
        let mut counts = vec![0usize; k];
        for &a in &res.assignments {
            counts[usize::try_from(a).unwrap()] += 1;
        }
        assert!(
            counts.iter().all(|&c| c >= 1),
            "found empty cluster in {counts:?}"
        );

        // No NaNs or Infs in materialized f16 centroids.
        assert!(res.centroids_f16.iter().all(|x| x.is_finite()));
    }
}
