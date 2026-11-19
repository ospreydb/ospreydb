//! Property-based isometric invariants for FWHT using `proptest`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use osprey_core::math::{pad_vector_preserve_scale, padded_dimension, RademacherSigns};
use osprey_core::AlignedBuffer;
use osprey_simd::{is_level_supported, SimdBackend, SimdEngine, SimdLevel};
use proptest::prelude::*;

const TEST_DIMS: [usize; 7] = [64, 128, 384, 512, 768, 1024, 1536];

fn non_zero_vec_strategy() -> impl Strategy<Value = (usize, u64, Vec<f32>)> {
    prop::sample::select(TEST_DIMS.to_vec()).prop_flat_map(|dim| {
        (
            Just(dim),
            any::<u64>(),
            prop::collection::vec(-10.0_f32..=10.0_f32, dim).prop_filter("must be non-zero", |v| {
                v.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>() > 1e-3
            }),
        )
    })
}

fn normalized_pair_strategy() -> impl Strategy<Value = (usize, u64, Vec<f32>, Vec<f32>)> {
    prop::sample::select(TEST_DIMS.to_vec()).prop_flat_map(|dim| {
        let vec_strat = prop::collection::vec(-10.0_f32..=10.0_f32, dim)
            .prop_filter("must be normalizable", |v| {
                v.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>() > 1e-3
            });
        (Just(dim), any::<u64>(), vec_strat.clone(), vec_strat)
    })
}

fn normalize_f64_to_f32(v: &[f32]) -> Vec<f32> {
    let norm = v
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    let inv = 1.0_f64 / norm;
    #[allow(clippy::cast_possible_truncation)]
    v.iter().map(|&x| (f64::from(x) * inv) as f32).collect()
}

fn pos_signs_vec_strategy() -> impl Strategy<Value = (usize, Vec<f32>)> {
    prop::sample::select(vec![64usize, 128, 256, 512, 1024, 2048])
        .prop_flat_map(|dim| (Just(dim), prop::collection::vec(-10.0_f32..=10.0_f32, dim)))
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: if cfg!(miri) { 4 } else { 256 },
        failure_persistence: None,
        .. ProptestConfig::default()
    })]

    #[test]
    #[cfg_attr(miri, ignore)]
    fn prop1_l2_norm_preservation((dim, seed, x) in non_zero_vec_strategy()) {
        let p_dim = padded_dimension(dim).unwrap();
        let signs = RademacherSigns::from_seed(p_dim, seed).unwrap();
        let engine = SimdEngine::new_auto();

        let mut buf = AlignedBuffer::<f32>::new(p_dim).unwrap();
        let orig_norm = f64::from(pad_vector_preserve_scale(&x, &mut buf).unwrap());
        engine.fwht_rotate_in_place(&mut buf, signs.xor_masks()).unwrap();

        let rot_norm = buf.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>().sqrt();
        let rel_err = (rot_norm - orig_norm).abs();
        prop_assert!(
            rel_err < 1e-5 * orig_norm,
            "dim={dim}: rot_norm={rot_norm}, orig_norm={orig_norm}, err={rel_err}"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn prop2_inner_product_isometry((dim, seed, raw_x, raw_y) in normalized_pair_strategy()) {
        let x = normalize_f64_to_f32(&raw_x);
        let y = normalize_f64_to_f32(&raw_y);
        let p_dim = padded_dimension(dim).unwrap();
        let signs = RademacherSigns::from_seed(p_dim, seed).unwrap();
        let engine = SimdEngine::new_auto();

        let exact_dot: f64 = x.iter().zip(y.iter()).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum();

        let mut px = AlignedBuffer::<f32>::new(p_dim).unwrap();
        let mut py = AlignedBuffer::<f32>::new(p_dim).unwrap();
        pad_vector_preserve_scale(&x, &mut px).unwrap();
        pad_vector_preserve_scale(&y, &mut py).unwrap();

        engine.fwht_rotate_in_place(&mut px, signs.xor_masks()).unwrap();
        engine.fwht_rotate_in_place(&mut py, signs.xor_masks()).unwrap();

        let rot_dot: f64 = px.iter().zip(py.iter()).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum();
        let err = (rot_dot - exact_dot).abs();
        prop_assert!(err < 1e-5, "dim={dim}: rot_dot={rot_dot}, exact_dot={exact_dot}, err={err}");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn prop3_hadamard_involution_positive_signs((dim, x) in pos_signs_vec_strategy()) {
        // All bits == 1 => all Rademacher signs == +1.0 (XOR mask 0x0000_0000)
        let pos_signs = RademacherSigns::from_words(dim, vec![u64::MAX; dim / 64]).unwrap();
        let engine = SimdEngine::new_auto();

        let mut buf = AlignedBuffer::from_slice(&x).unwrap();
        engine.fwht_rotate_in_place(&mut buf, pos_signs.xor_masks()).unwrap();
        engine.fwht_rotate_in_place(&mut buf, pos_signs.xor_masks()).unwrap();

        let max_err = buf.iter().zip(x.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        prop_assert!(max_err < 1e-5, "dim={dim}: involution max_err={max_err}");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn prop4_simd_matches_scalar_within_1e6((dim, seed, x) in non_zero_vec_strategy()) {
        let p_dim = padded_dimension(dim).unwrap();
        let signs = RademacherSigns::from_seed(p_dim, seed).unwrap();
        let scalar = SimdEngine::with_level(SimdLevel::Scalar);

        let mut expected = AlignedBuffer::<f32>::new(p_dim).unwrap();
        pad_vector_preserve_scale(&x, &mut expected).unwrap();
        scalar.fwht_rotate_in_place(&mut expected, signs.xor_masks()).unwrap();

        for level in [SimdLevel::Avx2, SimdLevel::Avx512, SimdLevel::Neon] {
            if !is_level_supported(level) {
                continue;
            }
            let simd = SimdEngine::with_level(level);
            let mut actual = AlignedBuffer::<f32>::new(p_dim).unwrap();
            pad_vector_preserve_scale(&x, &mut actual).unwrap();
            simd.fwht_rotate_in_place(&mut actual, signs.xor_masks()).unwrap();

            let max_diff = actual
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            prop_assert!(max_diff < 1e-6, "dim={dim} level={level:?}: max_diff={max_diff}");
        }
    }
}
