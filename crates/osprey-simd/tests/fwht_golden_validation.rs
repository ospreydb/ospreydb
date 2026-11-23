//! Cross-validation of all FWHT backends against Python golden vectors (`fwht_d*.bin`).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use osprey_core::math::{padded_dimension, prepare_and_pad_vector, RademacherSigns};
use osprey_core::AlignedBuffer;
use osprey_simd::{is_level_supported, SimdBackend, SimdEngine, SimdLevel};

const SEED_FWHT: u64 = 0x4F53_5052_0001;
const GOLDEN_NUM_VECS: usize = 16;

fn golden_bytes_for_dim(orig_dim: usize) -> &'static [u8] {
    match orig_dim {
        64 => include_bytes!("../../../reference/golden/fwht_d64.bin"),
        128 => include_bytes!("../../../reference/golden/fwht_d128.bin"),
        256 => include_bytes!("../../../reference/golden/fwht_d256.bin"),
        512 => include_bytes!("../../../reference/golden/fwht_d512.bin"),
        768 => include_bytes!("../../../reference/golden/fwht_d768.bin"),
        1024 => include_bytes!("../../../reference/golden/fwht_d1024.bin"),
        1536 => include_bytes!("../../../reference/golden/fwht_d1536.bin"),
        _ => unreachable!("unsupported golden dimension"),
    }
}

fn read_u64_le(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().unwrap())
}

fn read_f32_le(bytes: &[u8]) -> f32 {
    f32::from_le_bytes(bytes.try_into().unwrap())
}

fn verify_golden_dim(orig_dim: usize, expected_padded: usize) {
    let p_dim = padded_dimension(orig_dim).unwrap();
    assert_eq!(p_dim, expected_padded);

    let raw = golden_bytes_for_dim(orig_dim);
    let n_words = p_dim / 64;
    let words_bytes = n_words * 8;
    let in_bytes = GOLDEN_NUM_VECS * orig_dim * 4;
    let out_bytes = GOLDEN_NUM_VECS * p_dim * 4;
    assert_eq!(raw.len(), words_bytes + in_bytes + out_bytes);

    let signs = RademacherSigns::from_seed(p_dim, SEED_FWHT).unwrap();
    for w in 0..n_words {
        let golden_word = read_u64_le(&raw[w * 8..(w + 1) * 8]);
        assert_eq!(
            signs.words()[w],
            golden_word,
            "SplitMix64 word mismatch at dim={orig_dim}, word={w}"
        );
    }

    let levels = [
        SimdLevel::Scalar,
        SimdLevel::Avx2,
        SimdLevel::Avx512,
        SimdLevel::Neon,
    ];

    for level in levels {
        if !is_level_supported(level) {
            continue;
        }
        let engine = SimdEngine::with_level(level);
        assert_eq!(engine.active_level(), level);

        let mut work = AlignedBuffer::<f32>::new(p_dim).unwrap();
        let mut input_buf = vec![0.0_f32; orig_dim];

        for vec_idx in 0..GOLDEN_NUM_VECS {
            let in_base = words_bytes + vec_idx * orig_dim * 4;
            for j in 0..orig_dim {
                input_buf[j] = read_f32_le(&raw[in_base + j * 4..in_base + (j + 1) * 4]);
            }

            prepare_and_pad_vector(&input_buf, &mut work).unwrap();
            engine
                .fwht_rotate_in_place(&mut work, signs.xor_masks())
                .unwrap();

            let out_base = words_bytes + in_bytes + vec_idx * p_dim * 4;
            let mut max_abs_err = 0.0_f32;
            for j in 0..p_dim {
                let python_golden = read_f32_le(&raw[out_base + j * 4..out_base + (j + 1) * 4]);
                let abs_err = (work[j] - python_golden).abs();
                if abs_err > max_abs_err {
                    max_abs_err = abs_err;
                }
            }
            assert!(
                max_abs_err < 1e-6,
                "FWHT golden parity failed: orig_dim={orig_dim}, vec={vec_idx}, level={level:?}, max_abs_err={max_abs_err}"
            );
        }
    }
}

#[test]
fn golden_fwht_d64_parity() {
    verify_golden_dim(64, 64);
}

#[test]
#[cfg_attr(miri, ignore)]
fn golden_fwht_all_dimensions_parity() {
    for (orig_dim, padded) in [
        (64usize, 64usize),
        (128, 128),
        (256, 256),
        (512, 512),
        (768, 1024),
        (1024, 1024),
        (1536, 2048),
    ] {
        verify_golden_dim(orig_dim, padded);
    }
}
