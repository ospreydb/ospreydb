//! Integration benchmark comparing `1-Bit Binary K-Means` against classical
//! `f32` Lloyd K-Means (`N=10_000, d=512, K=64, max_iters=20`):
//! - 32x training working-memory reduction
//! - Distortion parity within < 1.0%
//! - Loop speedup > 15x in `--release`

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::Instant;

use osprey_core::cluster::{
    fit_binary_kmeans_from_rotated_with_init, init_kmeans_plusplus_1bit,
    preprocess_and_binarize_batch, BinaryKMeansConfig,
};
use osprey_core::math::{prepare_and_pad_vector, RademacherSigns, SplitMix64};
use osprey_simd::{SimdBackend, SimdEngine};

fn splitmix_unit_normal(rng: &mut SplitMix64) -> f32 {
    // Sum of 12 uniforms - 6.0 (Irwin-Hall normal approximation) for deterministic clustered data.
    let mut sum = 0.0_f32;
    for _ in 0..12 {
        let u = ((rng.next_u64() >> 40) as f32) / ((1_u32 << 24) as f32);
        sum += u;
    }
    sum - 6.0
}

fn generate_clustered_dataset(n: usize, dim: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let mut centers = vec![0.0_f32; k * dim];
    for c in 0..k {
        let row = &mut centers[c * dim..(c + 1) * dim];
        let mut norm_sq = 0.0_f64;
        for val in row.iter_mut() {
            let x = splitmix_unit_normal(&mut rng);
            *val = x;
            norm_sq += f64::from(x) * f64::from(x);
        }
        #[allow(clippy::cast_possible_truncation)]
        let inv = (1.0_f64 / norm_sq.sqrt()) as f32;
        for val in row.iter_mut() {
            *val *= inv;
        }
    }

    let noise_scale = 0.45_f32 / (dim as f32).sqrt();
    let mut data = vec![0.0_f32; n * dim];
    for i in 0..n {
        let c = usize::try_from(rng.next_u64() % (k as u64)).unwrap();
        let center = &centers[c * dim..(c + 1) * dim];
        let dst = &mut data[i * dim..(i + 1) * dim];
        let mut norm_sq = 0.0_f64;
        for j in 0..dim {
            let x = center[j] + noise_scale * splitmix_unit_normal(&mut rng);
            dst[j] = x;
            norm_sq += f64::from(x) * f64::from(x);
        }
        #[allow(clippy::cast_possible_truncation)]
        let inv = (1.0_f64 / norm_sq.sqrt()) as f32;
        for val in dst.iter_mut() {
            *val *= inv;
        }
    }
    data
}

/// Classical pure-`f32` Lloyd K-Means baseline evaluated on rotated `f32` vectors.
fn lloyd_f32_kmeans(
    rotated: &[f32],
    n: usize,
    dim: usize,
    k: usize,
    max_iters: usize,
    init_indices: &[usize],
) -> (Vec<f32>, Vec<u32>) {
    let mut centroids = vec![0.0_f32; k * dim];
    for (c, &idx) in init_indices.iter().enumerate() {
        centroids[c * dim..(c + 1) * dim].copy_from_slice(&rotated[idx * dim..(idx + 1) * dim]);
    }

    let mut assignments = vec![0_u32; n];
    let mut sums = vec![0.0_f32; k * dim];
    let mut counts = vec![0_u32; k];

    for _iter in 0..max_iters {
        // Assignment step: exact f32 L2 distance across all N * K * d elements.
        for i in 0..n {
            let x = &rotated[i * dim..(i + 1) * dim];
            let mut best_c = 0usize;
            let mut best_dist = f32::MAX;
            for c in 0..k {
                let cent = &centroids[c * dim..(c + 1) * dim];
                let mut dist_sq = 0.0_f32;
                for j in 0..dim {
                    let d = x[j] - cent[j];
                    dist_sq += d * d;
                }
                if dist_sq < best_dist {
                    best_dist = dist_sq;
                    best_c = c;
                }
            }
            assignments[i] = u32::try_from(best_c).unwrap();
        }

        // Update step: accumulate f32 means.
        sums.fill(0.0_f32);
        counts.fill(0);
        for i in 0..n {
            let c = usize::try_from(assignments[i]).unwrap();
            counts[c] += 1;
            let x = &rotated[i * dim..(i + 1) * dim];
            let dst = &mut sums[c * dim..(c + 1) * dim];
            for j in 0..dim {
                dst[j] += x[j];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0_f32 / (counts[c] as f32);
                let dst = &mut centroids[c * dim..(c + 1) * dim];
                let src = &sums[c * dim..(c + 1) * dim];
                for j in 0..dim {
                    dst[j] = src[j] * inv;
                }
            }
        }
    }

    (centroids, assignments)
}

#[test]
#[cfg_attr(miri, ignore)]
fn binary_kmeans_vs_lloyd_f32_quality_within_1pct_and_15x_speedup() {
    const N: usize = 10_000;
    const DIM: usize = 512;
    const K: usize = 64;
    const MAX_ITERS: usize = 20;
    const FWHT_SEED: u64 = 0x4F53_5052_0005;
    const INIT_SEED: u64 = 0x5EED_0077_0001;

    let engine = SimdEngine::new_auto();
    let raw_vectors = generate_clustered_dataset(N, DIM, K, 0x2026_1116);

    // Prepare rotated f32 dataset for the Lloyd f32 baseline and distortion measurement.
    let signs = RademacherSigns::from_seed(DIM, FWHT_SEED).unwrap();
    let mut rotated_f32 = vec![0.0_f32; N * DIM];
    for i in 0..N {
        let src = &raw_vectors[i * DIM..(i + 1) * DIM];
        let dst = &mut rotated_f32[i * DIM..(i + 1) * DIM];
        prepare_and_pad_vector(src, dst).unwrap();
        engine.fwht_rotate_in_place(dst, signs.xor_masks()).unwrap();
    }

    // Shared initial centroid indices for both algorithms (same starting centroids).
    let mut init_rng = SplitMix64::new(INIT_SEED);
    let mut init_indices = Vec::with_capacity(K);
    for _ in 0..K {
        init_indices.push(usize::try_from(init_rng.next_u64() % (N as u64)).unwrap());
    }

    // Pre-binarize 1-bit training buffer (32x smaller than rotated_f32).
    let packed_bits =
        preprocess_and_binarize_batch(&raw_vectors, N, DIM, FWHT_SEED, &engine).unwrap();
    let words_per_vec = DIM / 64;
    let mut init_bits = Vec::with_capacity(K * words_per_vec);
    for &idx in &init_indices {
        init_bits.extend_from_slice(&packed_bits[idx * words_per_vec..(idx + 1) * words_per_vec]);
    }
    // Also exercise deterministic 1-Bit K-Means++ initialization on the 10k corpus.
    let pp_bits =
        init_kmeans_plusplus_1bit(&packed_bits, N, words_per_vec, K, INIT_SEED, &engine).unwrap();
    assert_eq!(pp_bits.len(), K * words_per_vec);

    let lloyd_mem_bytes = N * DIM * std::mem::size_of::<f32>();
    let binary_kmeans_mem_bytes = packed_bits.len() * std::mem::size_of::<u64>();
    let mem_reduction = lloyd_mem_bytes / binary_kmeans_mem_bytes;
    assert_eq!(mem_reduction, 32);

    // 1. Time classical Lloyd f32 K-Means (20 iterations).
    let t_lloyd_start = Instant::now();
    let (lloyd_centroids, lloyd_assignments) =
        lloyd_f32_kmeans(&rotated_f32, N, DIM, K, MAX_ITERS, &init_indices);
    let lloyd_elapsed = t_lloyd_start.elapsed();

    // 2. Time 1-Bit Binary K-Means (20 iterations without early stop so both run the full loop,
    //    or with standard early stop — note early stop makes 1-bit even faster, so running with
    //    tolerance = 0.0 gives a conservative apples-to-apples 20-iteration loop speedup).
    let mut cfg = BinaryKMeansConfig::new(K);
    cfg.max_iters = MAX_ITERS;
    cfg.tolerance = 0.0;
    cfg.seed = INIT_SEED;
    cfg.spherical = false;

    let t_1bit_start = Instant::now();
    let skm_res = fit_binary_kmeans_from_rotated_with_init(
        &rotated_f32,
        &packed_bits,
        N,
        DIM,
        init_bits,
        &cfg,
        &engine,
    )
    .unwrap();
    let skm_elapsed = t_1bit_start.elapsed();

    // Compute mean squared Euclidean distortion and mean intra-cluster cosine similarity.
    let mut lloyd_distortion = 0.0_f64;
    let mut lloyd_cos_sum = 0.0_f64;
    let mut skm_distortion = 0.0_f64;
    let mut skm_cos_sum = 0.0_f64;

    for i in 0..N {
        let x = &rotated_f32[i * DIM..(i + 1) * DIM];

        let cl = usize::try_from(lloyd_assignments[i]).unwrap();
        let lc = &lloyd_centroids[cl * DIM..(cl + 1) * DIM];
        let mut l_sq = 0.0_f64;
        let mut l_dot = 0.0_f64;
        let mut lc_norm_sq = 0.0_f64;
        for j in 0..DIM {
            let xj = f64::from(x[j]);
            let cj = f64::from(lc[j]);
            let d = xj - cj;
            l_sq += d * d;
            l_dot += xj * cj;
            lc_norm_sq += cj * cj;
        }
        lloyd_distortion += l_sq;
        lloyd_cos_sum += l_dot / lc_norm_sq.sqrt().max(1e-12);

        let cs = usize::try_from(skm_res.assignments[i]).unwrap();
        let sc = &skm_res.centroids_f16[cs * DIM..(cs + 1) * DIM];
        let mut s_sq = 0.0_f64;
        let mut s_dot = 0.0_f64;
        let mut sc_norm_sq = 0.0_f64;
        for j in 0..DIM {
            let xj = f64::from(x[j]);
            let cj = f64::from(sc[j].to_f32());
            let d = xj - cj;
            s_sq += d * d;
            s_dot += xj * cj;
            sc_norm_sq += cj * cj;
        }
        skm_distortion += s_sq;
        skm_cos_sum += s_dot / sc_norm_sq.sqrt().max(1e-12);
    }

    lloyd_distortion /= N as f64;
    lloyd_cos_sum /= N as f64;
    skm_distortion /= N as f64;
    skm_cos_sum /= N as f64;

    let distortion_rel_diff = (skm_distortion - lloyd_distortion) / lloyd_distortion;
    let speedup = lloyd_elapsed.as_secs_f64() / skm_elapsed.as_secs_f64().max(1e-9);

    println!(
        "\n=== 1-Bit Binary K-Means vs Lloyd f32 (N={N}, d={DIM}, K={K}, iters={MAX_ITERS}, simd={:?}) ===",
        engine.active_level()
    );
    println!(
        "Working memory: Lloyd f32 = {} KiB | 1-Bit Binary K-Means = {} KiB ({}x reduction)",
        lloyd_mem_bytes / 1024,
        binary_kmeans_mem_bytes / 1024,
        mem_reduction
    );
    println!(
        "Distortion MSE: Lloyd f32 = {lloyd_distortion:.6} | 1-Bit Binary K-Means = {skm_distortion:.6} (rel diff = {:.4}%)",
        distortion_rel_diff * 100.0
    );
    println!(
        "Mean Cosine   : Lloyd f32 = {lloyd_cos_sum:.6} | 1-Bit Binary K-Means = {skm_cos_sum:.6}"
    );
    println!(
        "Wall Time     : Lloyd f32 = {:.2} ms | 1-Bit Binary K-Means = {:.2} ms (speedup = {speedup:.2}x)",
        lloyd_elapsed.as_secs_f64() * 1e3,
        skm_elapsed.as_secs_f64() * 1e3
    );

    assert!(
        distortion_rel_diff <= 0.01,
        "1-Bit Binary K-Means distortion exceeded Lloyd f32 by > 1%: {distortion_rel_diff:.6}"
    );
    assert!(
        speedup > 15.0,
        "expected > 15x speedup over Lloyd f32, got {speedup:.2}x"
    );
}
