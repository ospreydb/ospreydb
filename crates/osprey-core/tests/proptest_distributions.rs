//! Comprehensive multi-distribution `proptest` suite for `osprey-core`.
//!
//! Validates `1-Bit Binary K-Means`, `RaBitQ`, `LUT-Rank`, and zero-copy `.osi`/`.osr`
//! serialization/parsing across:
//! - Dimensions $d \in \{64, 96, 128, 256, 384, 512, 768, 1024, 1536\}$ (power-of-two & padded);
//! - Batch sizes $N \in [1, 257]$ and cluster counts $K \in [1, 32]$;
//! - 4 distinct mathematical distributions: Uniform $[-1, 1]^d$, Standard Gaussian,
//!   Extreme Anisotropic ($1000\times$ coordinates), and Sparse/Degenerate (95% zeros,
//!   single non-zero coordinate, and exact-centroid zero-residual vectors).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::mem::size_of;

use half::f16;
use osprey_core::math::{
    pad_vector_preserve_scale, padded_dimension, prepare_and_pad_vector, RademacherSigns,
    SplitMix64,
};
use osprey_core::{
    encode_prerotated_rabitq_residual, encode_prerotated_token_codes, encode_rabitq_residual,
    fit_binary_kmeans, pack_nibbles, unpack_nibbles, BinaryKMeansConfig, LutRankKernel,
    NestedLatticeCodebooks, OsiIndex, OsiMappedView, OspreyError, OsrDocumentTokens, OsrStore,
    QuantizedQueryLut, RabitqFwhtKernel, ScalarClusterKernel, ScalarRabitqKernel, VectorLocation,
    CACHE_LINE_ALIGN, DEFAULT_LUTRANK_GAMMA, LUTRANK_TOKEN_DIM, RABITQ_L1_MIN_NORMAL_F32,
};
use proptest::prelude::*;
use roaring::RoaringBitmap;
use xxhash_rust::xxh64::xxh64;

const TEST_DIMS: [usize; 9] = [64, 96, 128, 256, 384, 512, 768, 1024, 1536];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DistributionKind {
    Uniform,
    StandardGaussian,
    ExtremeAnisotropic,
    Sparse95Zero,
    SingleNonZeroCoord,
    IdenticalVectors,
}

const ALL_DISTRIBUTIONS: [DistributionKind; 6] = [
    DistributionKind::Uniform,
    DistributionKind::StandardGaussian,
    DistributionKind::ExtremeAnisotropic,
    DistributionKind::Sparse95Zero,
    DistributionKind::SingleNonZeroCoord,
    DistributionKind::IdenticalVectors,
];

#[inline]
fn next_unit_open01(rng: &mut SplitMix64) -> f64 {
    // Convert top 53 bits to (0, 1], avoiding exact 0.0 for Box-Muller log().
    let mant = (rng.next_u64() >> 11).max(1);
    (mant as f64) / ((1_u64 << 53) as f64)
}

#[inline]
fn next_uniform_sym(rng: &mut SplitMix64) -> f32 {
    let u = ((rng.next_u64() >> 11) as f64) / ((1_u64 << 53) as f64);
    #[allow(clippy::cast_possible_truncation)]
    {
        (u * 2.0 - 1.0) as f32
    }
}

#[inline]
fn next_gaussian(rng: &mut SplitMix64) -> f32 {
    let u1 = next_unit_open01(rng);
    let u2 = next_unit_open01(rng);
    let mag = (-2.0_f64 * u1.ln()).sqrt();
    let z0 = mag * (2.0_f64 * std::f64::consts::PI * u2).cos();
    #[allow(clippy::cast_possible_truncation)]
    {
        z0 as f32
    }
}

fn generate_vector(dist: DistributionKind, dim: usize, rng: &mut SplitMix64) -> Vec<f32> {
    match dist {
        DistributionKind::Uniform => (0..dim).map(|_| next_uniform_sym(rng)).collect(),
        DistributionKind::StandardGaussian => (0..dim).map(|_| next_gaussian(rng)).collect(),
        DistributionKind::ExtremeAnisotropic => {
            let mut v: Vec<f32> = (0..dim).map(|_| next_uniform_sym(rng)).collect();
            let c0 = usize::try_from(rng.next_u64() % (dim as u64)).unwrap_or(0);
            let mut c1 = usize::try_from(rng.next_u64() % (dim as u64)).unwrap_or(1 % dim);
            if dim > 1 && c1 == c0 {
                c1 = (c0 + 1) % dim;
            }
            let s0 = if (rng.next_u64() & 1) == 0 {
                1000.0_f32
            } else {
                -1000.0_f32
            };
            let s1 = if (rng.next_u64() & 1) == 0 {
                1000.0_f32
            } else {
                -1000.0_f32
            };
            v[c0] = s0 * (0.5 + next_uniform_sym(rng).abs());
            v[c1] = s1 * (0.5 + next_uniform_sym(rng).abs());
            v
        }
        DistributionKind::Sparse95Zero => {
            let mut v = vec![0.0_f32; dim];
            for val in &mut v {
                if (rng.next_u64() % 100) < 5 {
                    *val = next_gaussian(rng);
                }
            }
            v
        }
        DistributionKind::SingleNonZeroCoord => {
            let mut v = vec![0.0_f32; dim];
            let idx = usize::try_from(rng.next_u64() % (dim as u64)).unwrap_or(0);
            let sign = if (rng.next_u64() & 1) == 0 {
                1.0_f32
            } else {
                -1.0_f32
            };
            v[idx] = sign * (0.25 + next_uniform_sym(rng).abs() * 10.0);
            v
        }
        DistributionKind::IdenticalVectors => {
            // Identical coordinate pattern so cluster centroids coincide with input vectors.
            let mut v = vec![0.0_f32; dim];
            for (idx, slot) in v.iter_mut().enumerate() {
                *slot = if idx % 7 == 0 { 1.0 } else { 0.0 };
            }
            v
        }
    }
}

fn generate_batch(dist: DistributionKind, n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    if dist == DistributionKind::IdenticalVectors {
        let base = generate_vector(dist, dim, &mut rng);
        let mut out = Vec::with_capacity(n * dim);
        for _ in 0..n {
            out.extend_from_slice(&base);
        }
        return out;
    }

    let mut out = Vec::with_capacity(n * dim);
    for _ in 0..n {
        out.extend(generate_vector(dist, dim, &mut rng));
    }
    out
}

fn dist_strategy() -> impl Strategy<Value = DistributionKind> {
    prop::sample::select(ALL_DISTRIBUTIONS.to_vec())
}

fn dim_strategy() -> impl Strategy<Value = usize> {
    prop::sample::select(TEST_DIMS.to_vec())
}

fn assert_osi_binary_contracts(bytes: &[u8], expected_dim: usize, expected_n: usize, k: usize) {
    assert_eq!(
        (bytes.as_ptr() as usize) % CACHE_LINE_ALIGN,
        0,
        ".osi buffer base pointer is not 64-byte aligned"
    );
    assert!(bytes.len() >= 64 + size_of::<u64>());

    let view = OsiMappedView::from_bytes(bytes).expect("OsiMappedView::from_bytes must succeed");
    let hdr = view.header();
    assert_eq!(usize::from(hdr.orig_dim), expected_dim);
    assert_eq!(
        usize::from(hdr.padded_dim),
        padded_dimension(expected_dim).unwrap()
    );
    assert_eq!(usize::try_from(hdr.num_vectors).unwrap(), expected_n);
    assert_eq!(usize::try_from(hdr.num_clusters).unwrap(), k);

    assert_eq!(hdr.centroids_offset % 64, 0);
    assert_eq!(hdr.entries_offset % 64, 0);
    assert_eq!(hdr.tombstones_offset % 64, 0);
    assert_eq!(hdr.header_xxh64, hdr.compute_header_xxh64());

    let payload_end = bytes.len() - size_of::<u64>();
    let stored_file_hash = u64::from_le_bytes(bytes[payload_end..].try_into().unwrap());
    assert_eq!(stored_file_hash, xxh64(&bytes[..payload_end], 0));

    for (c, span) in view.cluster_spans().iter().enumerate() {
        assert_eq!(span.cluster_offset % 64, 0);
        assert_eq!((hdr.entries_offset + span.cluster_offset) % 64, 0);
        let c_view = view
            .cluster_view(u32::try_from(c).unwrap())
            .expect("cluster_view must succeed");
        if !c_view.codes.is_empty() {
            assert_eq!(
                (c_view.codes.as_ptr() as usize) % CACHE_LINE_ALIGN,
                0,
                "cluster {c} 1-bit codes slice is not 64-byte aligned"
            );
        }
        for entry in c_view.entries {
            let rn = entry.res_norm.to_f32();
            let l1 = entry.l1_norm_factor.to_f32();
            assert!(
                rn.is_finite() && !rn.is_nan() && rn >= 0.0,
                "invalid res_norm={rn}"
            );
            assert!(
                l1.is_finite() && !l1.is_nan() && l1 >= RABITQ_L1_MIN_NORMAL_F32 * 0.99,
                "invalid l1_norm_factor={l1}"
            );
        }
    }
}

fn assert_osr_binary_contracts(bytes: &[u8], expected_docs: usize) {
    assert_eq!(
        (bytes.as_ptr() as usize) % CACHE_LINE_ALIGN,
        0,
        ".osr buffer base pointer is not 64-byte aligned"
    );
    let store = OsrStore::from_bytes(bytes).expect("OsrStore::from_bytes must succeed");
    let hdr = store.header();
    assert_eq!(hdr.doc_table_offset % 64, 0);
    assert_eq!(hdr.tensors_offset % 64, 0);
    assert_eq!(hdr.header_xxh64, hdr.compute_header_xxh64());
    assert_eq!(usize::try_from(hdr.num_docs).unwrap(), expected_docs);

    let payload_end = bytes.len() - size_of::<u64>();
    let stored_file_hash = u64::from_le_bytes(bytes[payload_end..].try_into().unwrap());
    assert_eq!(stored_file_hash, xxh64(&bytes[..payload_end], 0));

    for range in store.doc_table() {
        let tok_bytes = store.token_bytes(range).expect("token_bytes must succeed");
        if !tok_bytes.is_empty() {
            assert_eq!(
                (tok_bytes.as_ptr() as usize) % CACHE_LINE_ALIGN,
                0,
                "doc {} token slice is not 64-byte aligned",
                range.doc_id
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: if cfg!(miri) { 2 } else { 36 },
        failure_persistence: None,
        .. ProptestConfig::default()
    })]

    #[test]
    #[cfg_attr(miri, ignore)]
    fn prop_osi_multidistribution_binary_kmeans_rabitq_and_roundtrip(
        dim in dim_strategy(),
        n in 1usize..=257,
        raw_k in 1usize..=32,
        dist in dist_strategy(),
        data_seed in any::<u64>(),
        fwht_seed in any::<u64>(),
        kmeans_seed in any::<u64>(),
        tomb_seed in any::<u64>(),
    ) {
        let k = raw_k.min(n);
        let vectors = generate_batch(dist, n, dim, data_seed);

        let cfg = BinaryKMeansConfig {
            num_clusters: k,
            max_iters: 10,
            tolerance: 1e-4,
            seed: kmeans_seed,
            spherical: true,
        };

        let kmeans = fit_binary_kmeans(
            &vectors,
            n,
            dim,
            fwht_seed,
            &cfg,
            &ScalarClusterKernel,
        )
        .expect("fit_binary_kmeans must not fail on finite multi-distribution batches");

        for &c_f16 in kmeans.centroids_f16.as_slice() {
            let c = c_f16.to_f32();
            prop_assert!(c.is_finite() && !c.is_nan(), "non-finite centroid f16={c}");
        }

        let doc_ids: Vec<u64> = (0..n).map(|i| 1_000 + i as u64).collect();
        let locations: Vec<VectorLocation> = (0..n)
            .map(|i| VectorLocation {
                slab_id: 0,
                minipage_id: u32::try_from(i / 16).unwrap(),
                slot_idx: u16::try_from(i % 16).unwrap(),
            })
            .collect();

        let mut tombstones = RoaringBitmap::new();
        let mut tomb_rng = SplitMix64::new(tomb_seed);
        if n > 2 && (tomb_rng.next_u64() & 1) == 0 {
            let deleted_id = doc_ids[usize::try_from(tomb_rng.next_u64() % (n as u64)).unwrap()];
            tombstones.insert(u32::try_from(deleted_id).unwrap());
        }

        let index = OsiIndex::build(
            &vectors,
            &doc_ids,
            &locations,
            &kmeans,
            dim,
            fwht_seed,
            tombstones.clone(),
            &ScalarRabitqKernel,
        )
        .expect("OsiIndex::build must not fail or produce NaN/Inf");

        for entry in index.entries() {
            let rn = entry.res_norm.to_f32();
            let l1 = entry.l1_norm_factor.to_f32();
            prop_assert!(
                rn.is_finite() && !rn.is_nan() && rn >= 0.0,
                "dist={dist:?}, dim={dim}, invalid res_norm={rn}"
            );
            prop_assert!(
                l1.is_finite() && !l1.is_nan() && l1 >= RABITQ_L1_MIN_NORMAL_F32 * 0.99,
                "dist={dist:?}, dim={dim}, invalid l1_norm_factor={l1}"
            );
        }

        let bytes = index.to_aligned_bytes().expect("to_aligned_bytes must succeed");
        assert_osi_binary_contracts(bytes.as_slice(), dim, n, k);

        let decoded = OsiIndex::from_bytes(bytes.as_slice()).expect("OsiIndex::from_bytes must succeed");
        let re_bytes = decoded.to_aligned_bytes().expect("re-serialization must succeed");
        prop_assert_eq!(bytes.as_slice(), re_bytes.as_slice());

        // Search with a query drawn from the same distribution.
        let mut q_rng = SplitMix64::new(data_seed ^ 0xDEAD_BEEF_CAFE_BABE);
        let query = generate_vector(dist, dim, &mut q_rng);
        let nprobe = k.clamp(1, 8);
        let top_c = n.clamp(1, 32);
        let hits = decoded
            .search_candidates(&query, nprobe, top_c, &ScalarRabitqKernel)
            .expect("search_candidates must succeed");

        for hit in &hits {
            prop_assert!(hit.score.is_finite() && !hit.score.is_nan());
            prop_assert!(!tombstones.contains(u32::try_from(hit.doc_id).unwrap()));
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: if cfg!(miri) { 4 } else { 64 },
        failure_persistence: None,
        .. ProptestConfig::default()
    })]

    #[test]
    #[cfg_attr(miri, ignore)]
    fn prop_rabitq_exact_zero_residual_and_degenerate_inputs(
        dim in dim_strategy(),
        dist in dist_strategy(),
        seed in any::<u64>(),
        fwht_seed in any::<u64>(),
    ) {
        let p_dim = padded_dimension(dim).unwrap();
        let signs = RademacherSigns::from_seed(p_dim, fwht_seed).unwrap();
        let mut rng = SplitMix64::new(seed);
        let raw = generate_vector(dist, dim, &mut rng);

        let mut x_padded = vec![0.0_f32; p_dim];
        prepare_and_pad_vector(&raw, &mut x_padded).unwrap();

        // Case 1: Vector identical to its centroid -> residual r = x - c_k == 0.
        let centroid_exact_f16: Vec<f16> = x_padded.iter().map(|&v| f16::from_f32(v)).collect();
        let x_exact_from_f16: Vec<f32> = centroid_exact_f16.iter().map(|c| c.to_f32()).collect();
        let mut scratch = vec![0.0_f32; p_dim];
        let mut out_words = vec![0_u64; p_dim / 64];

        let enc_unrot_zero = encode_rabitq_residual(
            &x_exact_from_f16,
            &centroid_exact_f16,
            &signs,
            &ScalarRabitqKernel,
            &mut scratch,
            &mut out_words,
        )
        .expect("exact zero residual encode_rabitq_residual must succeed");

        prop_assert!(enc_unrot_zero.res_norm.to_f32().is_finite());
        prop_assert_eq!(enc_unrot_zero.res_norm.to_f32(), 0.0_f32);
        prop_assert!(
            enc_unrot_zero.l1_norm_factor.to_f32().is_finite()
                && enc_unrot_zero.l1_norm_factor.to_f32() >= RABITQ_L1_MIN_NORMAL_F32 * 0.99
        );

        let enc_prerot_zero = encode_prerotated_rabitq_residual(
            &x_exact_from_f16,
            &centroid_exact_f16,
            &mut scratch,
            &mut out_words,
        )
        .expect("exact zero residual encode_prerotated_rabitq_residual must succeed");
        prop_assert!(enc_prerot_zero.res_norm.to_f32().is_finite());
        prop_assert_eq!(enc_prerot_zero.res_norm.to_f32(), 0.0_f32);
        prop_assert!(
            enc_prerot_zero.l1_norm_factor.to_f32().is_finite()
                && enc_prerot_zero.l1_norm_factor.to_f32() >= RABITQ_L1_MIN_NORMAL_F32 * 0.99
        );

        // Case 2: All-zero input vector with all-zero centroid.
        let zeros_f32 = vec![0.0_f32; p_dim];
        let zeros_f16 = vec![f16::ZERO; p_dim];
        let enc_all_zero = encode_rabitq_residual(
            &zeros_f32,
            &zeros_f16,
            &signs,
            &ScalarRabitqKernel,
            &mut scratch,
            &mut out_words,
        )
        .expect("all-zero residual must not divide by zero");
        prop_assert_eq!(enc_all_zero.res_norm.to_f32(), 0.0_f32);
        prop_assert!(
            enc_all_zero.l1_norm_factor.to_f32().is_finite()
                && enc_all_zero.l1_norm_factor.to_f32() >= RABITQ_L1_MIN_NORMAL_F32 * 0.99
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn prop_lutrank_all_dims_and_osr_store_multidistribution(
        dim in dim_strategy(),
        num_docs in 1usize..=48,
        tokens_per_doc in 1usize..=6,
        num_q_tokens in 1usize..=6,
        dist in dist_strategy(),
        seed in any::<u64>(),
        fwht_seed in any::<u64>(),
    ) {
        let p_dim = padded_dimension(dim).unwrap();
        let signs = RademacherSigns::from_seed(p_dim, fwht_seed).unwrap();
        let codebooks = NestedLatticeCodebooks::new(p_dim, DEFAULT_LUTRANK_GAMMA).unwrap();
        let n_sub = p_dim / 2;

        // 1. Multi-dimension LUT-Rank (all 9 dimensions, padded to power of 2).
        let mut rng = SplitMix64::new(seed);
        let mut q_rot_all = Vec::with_capacity(num_q_tokens * p_dim);
        for _ in 0..num_q_tokens {
            let raw_q = generate_vector(dist, dim, &mut rng);
            let mut buf = vec![0.0_f32; p_dim];
            prepare_and_pad_vector(&raw_q, &mut buf).unwrap();
            ScalarRabitqKernel
                .fwht_rotate_in_place(&mut buf, signs.xor_masks())
                .unwrap();
            q_rot_all.extend_from_slice(&buf);
        }

        let q_lut = QuantizedQueryLut::synthesize_from_prerotated(
            &q_rot_all,
            num_q_tokens,
            p_dim,
            DEFAULT_LUTRANK_GAMMA,
        )
        .expect("synthesize_from_prerotated must succeed for all distributions");

        prop_assert!(q_lut.scale1.is_finite() && !q_lut.scale1.is_nan() && q_lut.scale1 > 0.0);
        prop_assert!(q_lut.scale2.is_finite() && !q_lut.scale2.is_nan() && q_lut.scale2 > 0.0);

        let mut c1 = vec![0_u8; n_sub];
        let mut c2 = vec![0_u8; n_sub];
        let mut unpacked1 = vec![0_u8; n_sub];
        let mut doc_packed = Vec::with_capacity(tokens_per_doc * n_sub);

        for _ in 0..tokens_per_doc {
            let raw_t = generate_vector(dist, dim, &mut rng);
            let mut t_rot = vec![0.0_f32; p_dim];
            pad_vector_preserve_scale(&raw_t, &mut t_rot).unwrap();
            ScalarRabitqKernel
                .fwht_rotate_in_place(&mut t_rot, signs.xor_masks())
                .unwrap();
            encode_prerotated_token_codes(&t_rot, &codebooks, &mut c1, &mut c2).unwrap();

            let mut l1_bytes = vec![0_u8; n_sub / 2];
            let mut l2_bytes = vec![0_u8; n_sub / 2];
            pack_nibbles(&c1, &mut l1_bytes).unwrap();
            pack_nibbles(&c2, &mut l2_bytes).unwrap();
            unpack_nibbles(&l1_bytes, &mut unpacked1).unwrap();
            prop_assert_eq!(&c1, &unpacked1);

            doc_packed.extend_from_slice(&l1_bytes);
            doc_packed.extend_from_slice(&l2_bytes);
        }

        let maxsim = ScalarRabitqKernel.lutrank_doc_maxsim(
            &doc_packed,
            tokens_per_doc,
            n_sub,
            q_lut.lut1_i8.as_slice(),
            q_lut.lut2_i8.as_slice(),
            num_q_tokens,
            q_lut.scale1,
            q_lut.scale2,
        );
        prop_assert!(maxsim.is_finite() && !maxsim.is_nan(), "maxsim={maxsim}");

        // 2. Canonical `.osr` OsrStore build, 64B alignment, xxHash64 & roundtrip (128-dim tokens).
        let mut doc_token_vecs = Vec::with_capacity(num_docs);
        for d_idx in 0..num_docs {
            let mut toks = Vec::with_capacity(tokens_per_doc * LUTRANK_TOKEN_DIM);
            for _ in 0..tokens_per_doc {
                let raw_128 = generate_vector(dist, LUTRANK_TOKEN_DIM, &mut rng);
                toks.extend_from_slice(&raw_128);
            }
            doc_token_vecs.push((5_000 + d_idx as u64, toks));
        }

        let doc_descs: Vec<OsrDocumentTokens<'_>> = doc_token_vecs
            .iter()
            .map(|(doc_id, toks)| OsrDocumentTokens {
                doc_id: *doc_id,
                tokens: toks,
                num_tokens: tokens_per_doc,
            })
            .collect();

        let store = OsrStore::build(
            &doc_descs,
            LUTRANK_TOKEN_DIM,
            DEFAULT_LUTRANK_GAMMA,
            fwht_seed,
            &ScalarRabitqKernel,
        )
        .expect("OsrStore::build must succeed across all distributions");

        assert_osr_binary_contracts(store.as_bytes(), num_docs);

        let re_buf = store.to_aligned_bytes().unwrap();
        let reopened = OsrStore::from_bytes(re_buf.as_slice()).unwrap();
        prop_assert_eq!(reopened.as_bytes(), store.as_bytes());

        let mut q_128 = Vec::with_capacity(num_q_tokens * LUTRANK_TOKEN_DIM);
        for _ in 0..num_q_tokens {
            q_128.extend(generate_vector(dist, LUTRANK_TOKEN_DIM, &mut rng));
        }
        let osr_qlut = QuantizedQueryLut::synthesize(
            &q_128,
            num_q_tokens,
            LUTRANK_TOKEN_DIM,
            DEFAULT_LUTRANK_GAMMA,
            fwht_seed,
            &ScalarRabitqKernel,
        )
        .expect("QuantizedQueryLut::synthesize must succeed");
        prop_assert!(osr_qlut.scale1.is_finite() && osr_qlut.scale1 > 0.0);
        prop_assert!(osr_qlut.scale2.is_finite() && osr_qlut.scale2 > 0.0);

        for (doc_id, _) in &doc_token_vecs {
            let score = reopened
                .score_doc_id_maxsim(*doc_id, &osr_qlut, &ScalarRabitqKernel)
                .unwrap()
                .expect("doc_id must exist in OsrStore");
            prop_assert!(score.is_finite() && !score.is_nan());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn prop_corrupted_osi_and_osr_reject_safely_without_panic(
        dim in dim_strategy(),
        n in 4usize..=32,
        dist in dist_strategy(),
        seed in any::<u64>(),
        corrupt_pos_seed in any::<usize>(),
        corrupt_bit in 0u8..8,
    ) {
        let k = 2.min(n);
        let vectors = generate_batch(dist, n, dim, seed);
        let kmeans = fit_binary_kmeans(
            &vectors,
            n,
            dim,
            seed,
            &BinaryKMeansConfig::new(k),
            &ScalarClusterKernel,
        )
        .unwrap();
        let doc_ids: Vec<u64> = (0..n).map(|i| i as u64).collect();
        let locs: Vec<VectorLocation> = (0..n)
            .map(|i| VectorLocation {
                slab_id: 0,
                minipage_id: 0,
                slot_idx: u16::try_from(i).unwrap(),
            })
            .collect();
        let osi = OsiIndex::build(
            &vectors,
            &doc_ids,
            &locs,
            &kmeans,
            dim,
            seed,
            RoaringBitmap::new(),
            &ScalarRabitqKernel,
        )
        .unwrap();

        let mut osi_bytes = osi.to_aligned_bytes().unwrap();
        let osi_unaligned = matches!(
            OsiMappedView::from_bytes(&osi_bytes.as_slice()[1..]),
            Err(OspreyError::UnalignedMemory { .. })
        );
        prop_assert!(osi_unaligned);
        let osi_idx = corrupt_pos_seed % osi_bytes.len();
        osi_bytes.as_mut_slice()[osi_idx] ^= 1_u8 << corrupt_bit;
        prop_assert!(OsiMappedView::from_bytes(osi_bytes.as_slice()).is_err());
        prop_assert!(OsiIndex::from_bytes(osi_bytes.as_slice()).is_err());

        let mut rng = SplitMix64::new(seed);
        let tok = generate_vector(dist, LUTRANK_TOKEN_DIM, &mut rng);
        let docs = [OsrDocumentTokens {
            doc_id: 1,
            tokens: &tok,
            num_tokens: 1,
        }];
        let osr = OsrStore::build(
            &docs,
            LUTRANK_TOKEN_DIM,
            DEFAULT_LUTRANK_GAMMA,
            seed,
            &ScalarRabitqKernel,
        )
        .unwrap();
        let mut osr_bytes = osr.to_aligned_bytes().unwrap();
        let osr_unaligned = matches!(
            OsrStore::from_bytes(&osr_bytes.as_slice()[1..]),
            Err(OspreyError::UnalignedMemory { .. })
        );
        prop_assert!(osr_unaligned);
        let osr_idx = corrupt_pos_seed % osr_bytes.len();
        osr_bytes.as_mut_slice()[osr_idx] ^= 1_u8 << corrupt_bit;
        prop_assert!(OsrStore::from_bytes(osr_bytes.as_slice()).is_err());
    }
}
