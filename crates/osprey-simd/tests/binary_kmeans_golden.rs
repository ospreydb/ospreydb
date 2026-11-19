//! Cross-validation of `1-Bit Binary K-Means` across all SIMD backends
//! against `reference/golden/binary_kmeans_golden.bin` (`N=512, d=128, K=16`).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use half::f16;
use osprey_core::cluster::{
    binarize_rotated_vectors, fit_binary_kmeans_from_rotated_with_init, BinaryKMeansConfig,
};
use osprey_simd::{is_level_supported, SimdBackend, SimdEngine, SimdLevel};

const GOLDEN_BYTES: &[u8] = include_bytes!("../../../reference/golden/binary_kmeans_golden.bin");
const GOLDEN_MANIFEST: &str = include_str!("../../../reference/golden/golden_manifest.json");
const N_VECS: usize = 512;
const DIM: usize = 128;
const NUM_CLUSTERS: usize = 16;
const WORDS_PER_VEC: usize = DIM / 64;

fn read_u32_le(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().unwrap())
}

fn read_u64_le(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().unwrap())
}

fn read_f32_le(bytes: &[u8]) -> f32 {
    f32::from_le_bytes(bytes.try_into().unwrap())
}

fn read_f16_le(bytes: &[u8]) -> f16 {
    f16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap()))
}

fn mean_squared_distortion(vectors: &[f32], centroids_f16: &[f16], assignments: &[u32]) -> f64 {
    let mut total_sq = 0.0_f64;
    for (i, &assign) in assignments.iter().enumerate() {
        let c = usize::try_from(assign).unwrap();
        let v_row = &vectors[i * DIM..(i + 1) * DIM];
        let c_row = &centroids_f16[c * DIM..(c + 1) * DIM];
        let mut d_sq = 0.0_f64;
        for j in 0..DIM {
            let diff = f64::from(v_row[j]) - f64::from(c_row[j].to_f32());
            d_sq += diff * diff;
        }
        total_sq += d_sq;
    }
    total_sq / (vectors.len() / DIM) as f64
}

fn hamming_inertia_u64(
    packed_vecs: &[u64],
    centroids_bits: &[u64],
    assignments: &[u32],
    engine: &SimdEngine,
) -> u64 {
    let mut sum = 0_u64;
    for (i, &assign) in assignments.iter().enumerate() {
        let c = usize::try_from(assign).unwrap();
        let v = &packed_vecs[i * WORDS_PER_VEC..(i + 1) * WORDS_PER_VEC];
        let cent = &centroids_bits[c * WORDS_PER_VEC..(c + 1) * WORDS_PER_VEC];
        sum += u64::from(engine.hamming_distance_u64(v, cent));
    }
    sum
}

#[test]
#[cfg_attr(miri, ignore)]
fn binary_kmeans_golden_parity_across_all_simd_backends() {
    // Offsets from `reference/golden/golden_manifest.json` (`binary_kmeans_golden.bin`).
    assert!(GOLDEN_MANIFEST.contains("\"file\": \"binary_kmeans_golden.bin\""));
    assert!(GOLDEN_MANIFEST.contains("\"total_bytes\": 268800"));
    assert_eq!(GOLDEN_BYTES.len(), 268_800);

    let mut input_vectors = Vec::with_capacity(N_VECS * DIM);
    for i in 0..(N_VECS * DIM) {
        input_vectors.push(read_f32_le(&GOLDEN_BYTES[i * 4..(i + 1) * 4]));
    }

    let mut init_centroids_bits = Vec::with_capacity(NUM_CLUSTERS * WORDS_PER_VEC);
    let init_off = 262_144;
    for i in 0..(NUM_CLUSTERS * WORDS_PER_VEC) {
        init_centroids_bits.push(read_u64_le(
            &GOLDEN_BYTES[init_off + i * 8..init_off + (i + 1) * 8],
        ));
    }

    let mut golden_assignments = Vec::with_capacity(N_VECS);
    let assign_off = 262_400;
    for i in 0..N_VECS {
        golden_assignments.push(read_u32_le(
            &GOLDEN_BYTES[assign_off + i * 4..assign_off + (i + 1) * 4],
        ));
    }

    let mut golden_final_bits = Vec::with_capacity(NUM_CLUSTERS * WORDS_PER_VEC);
    let bits_off = 264_448;
    for i in 0..(NUM_CLUSTERS * WORDS_PER_VEC) {
        golden_final_bits.push(read_u64_le(
            &GOLDEN_BYTES[bits_off + i * 8..bits_off + (i + 1) * 8],
        ));
    }

    let mut golden_final_f16 = Vec::with_capacity(NUM_CLUSTERS * DIM);
    let f16_off = 264_704;
    for i in 0..(NUM_CLUSTERS * DIM) {
        golden_final_f16.push(read_f16_le(
            &GOLDEN_BYTES[f16_off + i * 2..f16_off + (i + 1) * 2],
        ));
    }

    // 1. Binarization bit-exact parity with Python `pack_bits_u64(x_rot >= 0.0)`.
    let packed_vecs = binarize_rotated_vectors(&input_vectors, N_VECS, DIM).unwrap();
    let f16_as_f32: Vec<f32> = golden_final_f16.iter().map(|x| x.to_f32()).collect();
    let repacked_golden_centroids =
        binarize_rotated_vectors(&f16_as_f32, NUM_CLUSTERS, DIM).unwrap();
    assert_eq!(repacked_golden_centroids, golden_final_bits);

    // 2. Run 1-Bit Binary K-Means across all supported SIMD backends from golden initial centroids.
    let mut cfg = BinaryKMeansConfig::new(NUM_CLUSTERS);
    cfg.max_iters = 12;
    cfg.tolerance = 0.0;
    cfg.spherical = false;

    let golden_distortion =
        mean_squared_distortion(&input_vectors, &golden_final_f16, &golden_assignments);

    struct GoldenRunSnapshot {
        assignments: Vec<u32>,
        binary_centroids: Vec<u64>,
        f16_bits: Vec<u16>,
        inertia: u64,
    }
    let mut scalar_ref: Option<GoldenRunSnapshot> = None;

    for level in [
        SimdLevel::Scalar,
        SimdLevel::Avx2,
        SimdLevel::Avx512,
        SimdLevel::Neon,
    ] {
        if !is_level_supported(level) {
            continue;
        }
        let engine = SimdEngine::with_level(level);
        let res = fit_binary_kmeans_from_rotated_with_init(
            &input_vectors,
            &packed_vecs,
            N_VECS,
            DIM,
            init_centroids_bits.clone(),
            &cfg,
            &engine,
        )
        .unwrap();

        let rust_f16_bits: Vec<u16> = res.centroids_f16.iter().map(|x| x.to_bits()).collect();
        if let Some(ref exp) = scalar_ref {
            assert_eq!(
                res.assignments, exp.assignments,
                "assignments differ on {level:?}"
            );
            assert_eq!(
                res.binary_centroids, exp.binary_centroids,
                "binary centroids differ on {level:?}"
            );
            assert_eq!(
                rust_f16_bits, exp.f16_bits,
                "f16 centroids differ on {level:?}"
            );
            assert_eq!(
                res.final_hamming_inertia, exp.inertia,
                "inertia differs on {level:?}"
            );
        } else {
            scalar_ref = Some(GoldenRunSnapshot {
                assignments: res.assignments.clone(),
                binary_centroids: res.binary_centroids.clone(),
                f16_bits: rust_f16_bits,
                inertia: res.final_hamming_inertia,
            });
        }

        let golden_inertia = hamming_inertia_u64(
            &packed_vecs,
            &golden_final_bits,
            &golden_assignments,
            &engine,
        );

        // Golden vectors (`generate_golden_vectors.py` lines 188-194) perform one final
        // exact `f16` L2 assignment pass after centroid materialization; evaluate distortion
        // under that exact same rule to verify parity against `golden_final_f16`.
        let mut f16_reassigned = vec![0_u32; N_VECS];
        for (i, slot) in f16_reassigned.iter_mut().enumerate() {
            let v_row = &input_vectors[i * DIM..(i + 1) * DIM];
            let mut best_c = 0usize;
            let mut best_sq = f64::MAX;
            for c in 0..NUM_CLUSTERS {
                let c_row = &res.centroids_f16[c * DIM..(c + 1) * DIM];
                let mut d_sq = 0.0_f64;
                for j in 0..DIM {
                    let diff = f64::from(v_row[j]) - f64::from(c_row[j].to_f32());
                    d_sq += diff * diff;
                }
                if d_sq < best_sq {
                    best_sq = d_sq;
                    best_c = c;
                }
            }
            *slot = u32::try_from(best_c).unwrap();
        }

        let rust_distortion =
            mean_squared_distortion(&input_vectors, &res.centroids_f16, &f16_reassigned);
        let rel_distortion_diff = ((rust_distortion - golden_distortion) / golden_distortion).abs();

        assert!(
            rel_distortion_diff < 1e-3,
            "level={level:?}: rust_distortion={rust_distortion}, golden_distortion={golden_distortion}, rel_diff={rel_distortion_diff}"
        );
        assert!(
            res.final_hamming_inertia <= golden_inertia + (N_VECS as u64 / 2),
            "level={level:?}: rust_inertia={}, golden_inertia={golden_inertia}",
            res.final_hamming_inertia
        );
    }
}
