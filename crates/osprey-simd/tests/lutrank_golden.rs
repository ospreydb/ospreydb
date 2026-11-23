//! Cross-architecture SIMD parity validation against
//! `reference/golden/lutrank_golden.bin` (`Scalar`, `Avx2`, `Avx512`, `Neon`).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use osprey_core::{
    OsrDocumentTokens, OsrStore, QuantizedQueryLut, DEFAULT_LUTRANK_GAMMA, LUTRANK_SUBSPACES,
};
use osprey_simd::{SimdBackend, SimdEngine, SimdLevel};

const LUTRANK_GOLDEN_BYTES: &[u8] = include_bytes!("../../../reference/golden/lutrank_golden.bin");

const N_DOCS: usize = 16;
const L_D: usize = 32;
const L_Q: usize = 8;
const DIM: usize = 128;
const SEED: u64 = 0x4F53_5052_0004;

fn read_f32_le(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

#[test]
#[cfg_attr(miri, ignore)]
fn lutrank_golden_parity_across_all_simd_backends() {
    let mut backends = vec![SimdEngine::with_level(SimdLevel::Scalar)];
    for level in [SimdLevel::Avx2, SimdLevel::Avx512, SimdLevel::Neon] {
        if let Some(engine) = SimdEngine::try_with_level(level) {
            backends.push(engine);
        }
    }

    let mut doc_tokens_raw = vec![0.0_f32; N_DOCS * L_D * DIM];
    for (i, dst) in doc_tokens_raw.iter_mut().enumerate() {
        *dst = read_f32_le(LUTRANK_GOLDEN_BYTES, 256 + i * 4);
    }
    let mut q_tokens_raw = vec![0.0_f32; L_Q * DIM];
    for (i, dst) in q_tokens_raw.iter_mut().enumerate() {
        *dst = read_f32_le(LUTRANK_GOLDEN_BYTES, 557_312 + i * 4);
    }

    let scalar = SimdEngine::with_level(SimdLevel::Scalar);
    let docs: Vec<OsrDocumentTokens<'_>> = (0..N_DOCS)
        .map(|d| OsrDocumentTokens {
            doc_id: u64::try_from(d + 1).unwrap(),
            tokens: &doc_tokens_raw[d * L_D * DIM..(d + 1) * L_D * DIM],
            num_tokens: L_D,
        })
        .collect();
    let store = OsrStore::build(&docs, DIM, DEFAULT_LUTRANK_GAMMA, SEED, &scalar).unwrap();

    // Load exact golden LUTs + scales per query token to verify exact backend parity (< 1e-4)
    let golden_l1: Vec<i8> = (0..L_Q * LUTRANK_SUBSPACES * 16)
        .map(|i| i8::from_le_bytes([LUTRANK_GOLDEN_BYTES[565_504 + i]]))
        .collect();
    let golden_l2: Vec<i8> = (0..L_Q * LUTRANK_SUBSPACES * 16)
        .map(|i| i8::from_le_bytes([LUTRANK_GOLDEN_BYTES[573_696 + i]]))
        .collect();
    let golden_s1: Vec<f32> = (0..L_Q)
        .map(|q| read_f32_le(LUTRANK_GOLDEN_BYTES, 581_888 + q * 4))
        .collect();
    let golden_s2: Vec<f32> = (0..L_Q)
        .map(|q| read_f32_le(LUTRANK_GOLDEN_BYTES, 581_920 + q * 4))
        .collect();

    for engine in &backends {
        let q_lut = QuantizedQueryLut::synthesize(
            &q_tokens_raw,
            L_Q,
            DIM,
            DEFAULT_LUTRANK_GAMMA,
            SEED,
            engine,
        )
        .unwrap();

        for d in 0..N_DOCS {
            let range = store.doc_table()[d];
            let doc_bytes = store.token_bytes(&range).unwrap();

            // 1. Exact golden LUT evaluation per query token against `maxsim_lut_f32`
            let mut golden_eval = 0.0_f32;
            for qi in 0..L_Q {
                let s = engine.lutrank_doc_maxsim(
                    doc_bytes,
                    L_D,
                    LUTRANK_SUBSPACES,
                    &golden_l1[qi * 1024..(qi + 1) * 1024],
                    &golden_l2[qi * 1024..(qi + 1) * 1024],
                    1,
                    golden_s1[qi],
                    golden_s2[qi],
                );
                golden_eval += s;
            }
            let expected_golden = read_f32_le(LUTRANK_GOLDEN_BYTES, 582_016 + d * 4);
            assert!(
                (golden_eval - expected_golden).abs() <= 1e-4,
                "backend={:?} doc={d}: got={golden_eval}, expected={expected_golden}",
                engine.active_level()
            );

            // 2. End-to-end QuantizedQueryLut evaluation parity across all backends
            let simd_score = store.score_document_maxsim(d, &q_lut, engine);
            let scalar_score = store.score_document_maxsim(d, &q_lut, &scalar);
            assert!(
                (simd_score - scalar_score).abs() < 1e-5,
                "backend={:?} vs scalar doc={d}: simd={simd_score}, scalar={scalar_score}",
                engine.active_level()
            );
        }
    }
}
