//! Golden-vector parity validation against `reference/golden/rabitq_golden.bin`
//! (`N=512, d=128, K=16, N_q=16`) plus Miri-compatible mini-index & tombstone checks.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use half::f16;
use osprey_core::cluster::{fit_binary_kmeans, BinaryKMeansConfig, BinaryKMeansResult};
use osprey_core::math::{
    fwht_scalar_in_place, prepare_and_pad_vector, RademacherSigns, SplitMix64,
};
use osprey_core::{
    encode_prerotated_rabitq_residual, AlignedBuffer, BoundedTopKHeap, OsiIndex, RabitqEntryHeader,
    RabitqScanKernel, ScalarRabitqKernel, ScoredCandidate, TransposedQuery4Bit, VectorLocation,
};
use roaring::RoaringBitmap;

const SKM_GOLDEN_BYTES: &[u8] =
    include_bytes!("../../../reference/golden/binary_kmeans_golden.bin");
const RAB_GOLDEN_BYTES: &[u8] = include_bytes!("../../../reference/golden/rabitq_golden.bin");
const GOLDEN_MANIFEST: &str = include_str!("../../../reference/golden/golden_manifest.json");

const N_VECS: usize = 512;
const DIM: usize = 128;
const NUM_CLUSTERS: usize = 16;
const N_QUERIES: usize = 16;
const WORDS_PER_VEC: usize = DIM / 64;
const DB_SEED: u64 = 0x4F53_5052_0002;

#[test]
#[cfg_attr(miri, ignore)]
fn rabitq_golden_parity_and_tombstone_serialization() {
    assert!(GOLDEN_MANIFEST.contains("\"file\": \"rabitq_golden.bin\""));
    assert_eq!(SKM_GOLDEN_BYTES.len(), 268_800);
    assert_eq!(RAB_GOLDEN_BYTES.len(), 196_288);

    // 1. Load rotated database vectors from binary_kmeans_golden.bin
    let mut db_rot = vec![0.0_f32; N_VECS * DIM];
    for (i, dst) in db_rot.iter_mut().enumerate() {
        let off = i * 4;
        *dst = f32::from_le_bytes(SKM_GOLDEN_BYTES[off..off + 4].try_into().unwrap());
    }

    // 2. Load rabitq_golden.bin sections (offsets from golden_manifest.json)
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

    // 3. Verify 1-bit codes, res_norm, and l1_norm_factor match golden fixture
    let mut scratch = vec![0.0_f32; DIM];
    let mut out_words = vec![0_u64; WORDS_PER_VEC];
    for i in 0..N_VECS {
        let c = usize::try_from(cluster_ids[i]).unwrap();
        let encoded = encode_prerotated_rabitq_residual(
            &db_rot[i * DIM..(i + 1) * DIM],
            &centroids_f16[c * DIM..(c + 1) * DIM],
            &mut scratch,
            &mut out_words,
        )
        .unwrap();

        assert!(
            (encoded.res_norm.to_f32() - golden_res_norm[i].to_f32()).abs() <= 1e-3,
            "res_norm mismatch at {i}"
        );
        assert!(
            (encoded.l1_norm_factor.to_f32() - golden_l1[i].to_f32()).abs() <= 1e-3,
            "l1_norm_factor mismatch at {i}"
        );
        assert_eq!(
            &out_words[..],
            &golden_codes[i * WORDS_PER_VEC..(i + 1) * WORDS_PER_VEC],
            "1-bit code mismatch at {i}"
        );
    }

    // 4. Verify TransposedQuery4Bit bit-planes and 16 x 512 estimated score matrix (< 1e-3 error)
    let kernel = ScalarRabitqKernel;
    for qi in 0..N_QUERIES {
        let mut q_rot = vec![0.0_f32; DIM];
        for (d, dst) in q_rot.iter_mut().enumerate() {
            let off = 151_552 + (qi * DIM + d) * 4;
            *dst = f32::from_le_bytes(RAB_GOLDEN_BYTES[off..off + 4].try_into().unwrap());
        }
        let scale_off = 159_744 + qi * 4;
        let golden_scale = f32::from_le_bytes(
            RAB_GOLDEN_BYTES[scale_off..scale_off + 4]
                .try_into()
                .unwrap(),
        );
        let tq = TransposedQuery4Bit::try_from_rotated_query(&q_rot).unwrap();
        assert!((tq.q_scale - golden_scale).abs() <= 1e-6);

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
            let dot_int =
                kernel.rabitq_4bit_dot_u64(&tq.planes, &tq.sign_mask, tq.query_abs_sum, db_slice);
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
            let golden_asym =
                f32::from_le_bytes(RAB_GOLDEN_BYTES[asym_off..asym_off + 4].try_into().unwrap());
            assert!(
                (est - golden_asym).abs() < 1e-3,
                "score mismatch q={qi} v={vi}: est={est}, golden={golden_asym}"
            );
        }
    }

    // 5. Build OsiIndex with RoaringBitmap tombstones, serialize/reopen, and verify O(1) filtering
    let doc_ids: Vec<u64> = (0..N_VECS).map(|i| u64::try_from(i).unwrap()).collect();
    let locations = vec![VectorLocation::default(); N_VECS];
    let kmeans = BinaryKMeansResult {
        centroids_f16,
        binary_centroids: vec![0_u64; NUM_CLUSTERS * WORDS_PER_VEC],
        assignments: cluster_ids,
        final_hamming_inertia: 0,
        iterations_run: 12,
    };

    // Unfiltered search for query 0 (using q_raw_f16 from offset 147456)
    let mut q0_raw = vec![0.0_f32; DIM];
    for (d, dst) in q0_raw.iter_mut().enumerate() {
        let off = 147_456 + d * 2;
        *dst = f16::from_le_bytes(RAB_GOLDEN_BYTES[off..off + 2].try_into().unwrap()).to_f32();
    }

    let base_osi = OsiIndex::build_from_rotated(
        &db_rot,
        &doc_ids,
        &locations,
        &kmeans,
        DIM,
        DB_SEED,
        RoaringBitmap::new(),
    )
    .unwrap();
    let base_top = base_osi
        .search_candidates(&q0_raw, NUM_CLUSTERS, 16, &kernel)
        .unwrap();
    let top1_doc = base_top[0].doc_id;
    let top2_doc = base_top[1].doc_id;

    let mut tombstones = RoaringBitmap::new();
    tombstones.insert(u32::try_from(top1_doc).unwrap());
    tombstones.insert(u32::try_from(top2_doc).unwrap());

    let tomb_osi = OsiIndex::build_from_rotated(
        &db_rot, &doc_ids, &locations, &kmeans, DIM, DB_SEED, tombstones,
    )
    .unwrap();
    let bytes = tomb_osi.to_aligned_bytes().unwrap();
    let reopened = OsiIndex::from_bytes(&bytes).unwrap();

    assert!(reopened.is_deleted(top1_doc));
    assert!(reopened.is_deleted(top2_doc));
    let filtered_top = reopened
        .search_candidates(&q0_raw, NUM_CLUSTERS, 16, &kernel)
        .unwrap();
    assert_eq!(filtered_top.len(), 16);
    assert!(filtered_top
        .iter()
        .all(|c| c.doc_id != top1_doc && c.doc_id != top2_doc));
}

#[test]
fn miri_mini_osi_index_tombstones_heap_and_roundtrip() {
    let n = 64usize;
    let orig_dim = 64usize;
    let k = 4usize;
    let seed = 0x4F53_5052_A11C;
    let mut rng = SplitMix64::new(seed);

    let mut vectors = Vec::with_capacity(n * orig_dim);
    for _ in 0..n {
        let mut sq = 0.0_f64;
        let start = vectors.len();
        for _ in 0..orig_dim {
            #[allow(clippy::cast_precision_loss)]
            let v = ((rng.next_u64() >> 11) as f64) / ((1_u64 << 53) as f64) * 2.0 - 1.0;
            sq += v * v;
            #[allow(clippy::cast_possible_truncation)]
            vectors.push(v as f32);
        }
        let inv = 1.0_f64 / sq.sqrt();
        for v in &mut vectors[start..] {
            #[allow(clippy::cast_possible_truncation)]
            {
                *v = (f64::from(*v) * inv) as f32;
            }
        }
    }

    let kernel = ScalarRabitqKernel;
    let skm_cfg = BinaryKMeansConfig {
        num_clusters: k,
        max_iters: 4,
        tolerance: 1e-4,
        seed,
        spherical: true,
    };
    let skm = fit_binary_kmeans(
        &vectors,
        n,
        orig_dim,
        seed,
        &skm_cfg,
        &osprey_core::cluster::ScalarClusterKernel,
    )
    .unwrap();

    let doc_ids: Vec<u64> = (0..n).map(|i| u64::try_from(i + 500).unwrap()).collect();
    let locations = vec![VectorLocation::default(); n];
    let mut tombstones = RoaringBitmap::new();
    tombstones.insert(510);

    let index = OsiIndex::build(
        &vectors, &doc_ids, &locations, &skm, orig_dim, seed, tombstones, &kernel,
    )
    .unwrap();

    let serialized = index.to_aligned_bytes().unwrap();
    let reopened = OsiIndex::from_bytes(&serialized).unwrap();
    assert!(reopened.is_deleted(510));

    let q = &vectors[10 * orig_dim..11 * orig_dim];
    let hits = reopened.search_candidates(q, k, 8, &kernel).unwrap();
    assert_eq!(hits.len(), 8);
    assert!(hits.iter().all(|c| c.doc_id != 510));

    // Exercise BoundedTopKHeap directly under Miri
    let mut heap = BoundedTopKHeap::<16>::new(4);
    for (idx, h) in hits.iter().enumerate() {
        heap.push(ScoredCandidate {
            score: h.score,
            doc_id: h.doc_id,
            entry_idx: u32::try_from(idx).unwrap(),
            minipage_id: 0,
            slot_idx: 0,
        });
    }
    let top = heap.into_sorted_slice();
    assert_eq!(top.len(), 4);
    assert!(top.windows(2).all(|w| w[0].score >= w[1].score));

    // Exercise FWHT + padding invariants
    let signs = RademacherSigns::from_seed(orig_dim, seed).unwrap();
    let mut buf = vec![0.0_f32; orig_dim];
    prepare_and_pad_vector(q, &mut buf).unwrap();
    fwht_scalar_in_place(&mut buf, &signs).unwrap();
    let norm: f64 = buf
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        .sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
}
