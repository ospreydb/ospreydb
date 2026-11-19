//! Cross-architecture SIMD parity validation against
//! `reference/golden/rabitq_golden.bin` (`Scalar`, `Avx2`, `Avx512`, `Neon`).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use half::f16;
use osprey_core::cluster::BinaryKMeansResult;
use osprey_core::{
    AlignedBuffer, OsiIndex, RabitqEntryHeader, TransposedQuery4Bit, VectorLocation,
};
use osprey_simd::{SimdBackend, SimdEngine, SimdLevel};
use roaring::RoaringBitmap;

const SKM_GOLDEN_BYTES: &[u8] =
    include_bytes!("../../../reference/golden/binary_kmeans_golden.bin");
const RAB_GOLDEN_BYTES: &[u8] = include_bytes!("../../../reference/golden/rabitq_golden.bin");

const N_VECS: usize = 512;
const DIM: usize = 128;
const NUM_CLUSTERS: usize = 16;
const N_QUERIES: usize = 16;
const WORDS_PER_VEC: usize = DIM / 64;
const DB_SEED: u64 = 0x4F53_5052_0002;

#[test]
#[cfg_attr(miri, ignore)]
fn rabitq_golden_parity_across_all_simd_backends() {
    let mut backends = vec![SimdEngine::with_level(SimdLevel::Scalar)];
    for level in [SimdLevel::Avx2, SimdLevel::Avx512, SimdLevel::Neon] {
        if let Some(engine) = SimdEngine::try_with_level(level) {
            backends.push(engine);
        }
    }

    let mut db_rot = vec![0.0_f32; N_VECS * DIM];
    for (i, dst) in db_rot.iter_mut().enumerate() {
        let off = i * 4;
        *dst = f32::from_le_bytes(SKM_GOLDEN_BYTES[off..off + 4].try_into().unwrap());
    }

    let mut cluster_ids = vec![0_u32; N_VECS];
    for (i, dst) in cluster_ids.iter_mut().enumerate() {
        let off = i * 4;
        *dst = u32::from_le_bytes(RAB_GOLDEN_BYTES[off..off + 4].try_into().unwrap());
    }

    let mut golden_res_norm = vec![f16::ZERO; N_VECS];
    for (i, dst) in golden_res_norm.iter_mut().enumerate() {
        let off = 2048 + i * 2;
        *dst = f16::from_le_bytes(RAB_GOLDEN_BYTES[off..off + 2].try_into().unwrap());
    }

    let mut golden_l1 = vec![f16::ZERO; N_VECS];
    for (i, dst) in golden_l1.iter_mut().enumerate() {
        let off = 3072 + i * 2;
        *dst = f16::from_le_bytes(RAB_GOLDEN_BYTES[off..off + 2].try_into().unwrap());
    }

    let mut golden_codes = vec![0_u64; N_VECS * WORDS_PER_VEC];
    for (i, dst) in golden_codes.iter_mut().enumerate() {
        let off = 4096 + i * 8;
        *dst = u64::from_le_bytes(RAB_GOLDEN_BYTES[off..off + 8].try_into().unwrap());
    }

    let mut centroids_f16 = AlignedBuffer::<f16>::new(NUM_CLUSTERS * DIM).unwrap();
    for (i, dst) in centroids_f16.iter_mut().enumerate() {
        let off = 143_360 + i * 2;
        *dst = f16::from_le_bytes(RAB_GOLDEN_BYTES[off..off + 2].try_into().unwrap());
    }

    for engine in &backends {
        for qi in 0..N_QUERIES {
            let mut q_rot = vec![0.0_f32; DIM];
            for (d, dst) in q_rot.iter_mut().enumerate() {
                let off = 151_552 + (qi * DIM + d) * 4;
                *dst = f32::from_le_bytes(RAB_GOLDEN_BYTES[off..off + 4].try_into().unwrap());
            }
            let tq = TransposedQuery4Bit::try_from_rotated_query(&q_rot).unwrap();

            let mut c_dots = [0.0_f32; NUM_CLUSTERS];
            for (c, c_dot) in c_dots.iter_mut().enumerate() {
                let mut acc = 0.0_f64;
                for d in 0..DIM {
                    acc += f64::from(centroids_f16[c * DIM + d].to_f32()) * f64::from(q_rot[d]);
                }
                #[allow(clippy::cast_possible_truncation)]
                {
                    *c_dot = acc as f32;
                }
            }

            for vi in 0..N_VECS {
                let db_slice = &golden_codes[vi * WORDS_PER_VEC..(vi + 1) * WORDS_PER_VEC];
                let dot_int = engine.rabitq_4bit_dot_u64(
                    &tq.planes,
                    &tq.sign_mask,
                    tq.query_abs_sum,
                    db_slice,
                );
                let entry = RabitqEntryHeader {
                    doc_id: u64::try_from(vi).unwrap(),
                    minipage_id: 0,
                    slab_id: 0,
                    slot_idx: 0,
                    res_norm: golden_res_norm[vi],
                    l1_norm_factor: golden_l1[vi],
                    _reserved: 0,
                };
                let est = OsiIndex::reconstruct_asymmetric_score(
                    c_dots[usize::try_from(cluster_ids[vi]).unwrap()],
                    &entry,
                    tq.q_scale,
                    dot_int,
                    u16::try_from(DIM).unwrap(),
                );
                let asym_off = 162_880 + (qi * N_VECS + vi) * 4;
                let golden_asym = f32::from_le_bytes(
                    RAB_GOLDEN_BYTES[asym_off..asym_off + 4].try_into().unwrap(),
                );
                assert!(
                    (est - golden_asym).abs() < 1e-3,
                    "backend={:?} q={qi} v={vi}: est={est} golden={golden_asym}",
                    engine.active_level()
                );
            }
        }

        // Also validate full OsiIndex::search_candidates + tombstone filtering per SIMD backend
        let doc_ids: Vec<u64> = (0..N_VECS).map(|i| u64::try_from(i).unwrap()).collect();
        let locations = vec![VectorLocation::default(); N_VECS];
        let kmeans = BinaryKMeansResult {
            centroids_f16: AlignedBuffer::from_slice(&centroids_f16).unwrap(),
            binary_centroids: vec![0_u64; NUM_CLUSTERS * WORDS_PER_VEC],
            assignments: cluster_ids.clone(),
            final_hamming_inertia: 0,
            iterations_run: 12,
        };
        let mut tombstones = RoaringBitmap::new();
        tombstones.insert(0);
        let osi = OsiIndex::build_from_rotated(
            &db_rot, &doc_ids, &locations, &kmeans, DIM, DB_SEED, tombstones,
        )
        .unwrap();

        let mut q0_raw = vec![0.0_f32; DIM];
        for (d, dst) in q0_raw.iter_mut().enumerate() {
            let off = 147_456 + d * 2;
            *dst = f16::from_le_bytes(RAB_GOLDEN_BYTES[off..off + 2].try_into().unwrap()).to_f32();
        }
        let hits = osi
            .search_candidates(&q0_raw, NUM_CLUSTERS, 32, engine)
            .unwrap();
        assert_eq!(hits.len(), 32);
        assert!(hits.iter().all(|c| c.doc_id != 0));
    }
}
