//! Golden-vector parity validation against
//! `reference/golden/lutrank_golden.bin` (`N_docs=16, L_d=32, L_q=8, D=128, gamma=0.28`)
//! plus Miri-compatible mini `.osr` roundtrip and corruption checks.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use osprey_core::math::{RademacherSigns, SplitMix64};
use osprey_core::{
    build_nested_lattice_codebooks, encode_prerotated_token, encode_token,
    synthesize_query_token_luts, AlignedBuffer, LutRankKernel, NestedLatticeCodebooks, OspreyError,
    OsrDocumentTokens, OsrStore, QuantizedQueryLut, QueryTokenLuts, ScalarRabitqKernel,
    DEFAULT_LUTRANK_GAMMA, LUTRANK_SUBSPACES, LUTRANK_TOKEN_DIM,
};

const LUTRANK_GOLDEN_BYTES: &[u8] = include_bytes!("../../../reference/golden/lutrank_golden.bin");
const GOLDEN_MANIFEST: &str = include_str!("../../../reference/golden/golden_manifest.json");

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
fn lutrank_golden_parity_codebooks_nibbles_luts_and_maxsim() {
    assert!(GOLDEN_MANIFEST.contains("\"file\": \"lutrank_golden.bin\""));
    assert_eq!(LUTRANK_GOLDEN_BYTES.len(), 582_080);

    // 1. Verify CB_LAYER1 and CB_LAYER2 match golden fixture
    let (cb1, cb2) = build_nested_lattice_codebooks(DIM, DEFAULT_LUTRANK_GAMMA).unwrap();
    for c in 0..16 {
        let off = c * 8;
        let gx1 = read_f32_le(LUTRANK_GOLDEN_BYTES, off);
        let gy1 = read_f32_le(LUTRANK_GOLDEN_BYTES, off + 4);
        let gx2 = read_f32_le(LUTRANK_GOLDEN_BYTES, 128 + off);
        let gy2 = read_f32_le(LUTRANK_GOLDEN_BYTES, 128 + off + 4);

        assert!((cb1[c][0] - gx1).abs() <= 1e-6);
        assert!((cb1[c][1] - gy1).abs() <= 1e-6);
        assert!((cb2[c][0] - gx2).abs() <= 1e-6);
        assert!((cb2[c][1] - gy2).abs() <= 1e-6);
    }

    let codebooks = NestedLatticeCodebooks::canonical().unwrap();
    let signs = RademacherSigns::from_seed(DIM, SEED).unwrap();
    let simd = ScalarRabitqKernel;

    let mut doc_tokens_raw = vec![0.0_f32; N_DOCS * L_D * DIM];
    for (i, dst) in doc_tokens_raw.iter_mut().enumerate() {
        *dst = read_f32_le(LUTRANK_GOLDEN_BYTES, 256 + i * 4);
    }
    let mut doc_tokens_rot = vec![0.0_f32; N_DOCS * L_D * DIM];
    for (i, dst) in doc_tokens_rot.iter_mut().enumerate() {
        *dst = read_f32_le(LUTRANK_GOLDEN_BYTES, 262_400 + i * 4);
    }

    // 2. Verify bit-exact packed nibbles for all 16 x 32 document tokens
    let golden_nibbles = &LUTRANK_GOLDEN_BYTES[524_544..524_544 + N_DOCS * L_D * 64];
    let mut scratch = vec![0.0_f32; DIM];
    for d in 0..N_DOCS {
        for t in 0..L_D {
            let off = (d * L_D + t) * DIM;
            let nib_off = (d * L_D + t) * 64;
            let expected = &golden_nibbles[nib_off..nib_off + 64];

            let c_rot =
                encode_prerotated_token(&doc_tokens_rot[off..off + DIM], &codebooks).unwrap();
            assert_eq!(bytemuck::bytes_of(&c_rot), expected);

            let c_raw = encode_token(
                &doc_tokens_raw[off..off + DIM],
                &signs,
                &codebooks,
                &simd,
                &mut scratch,
            )
            .unwrap();
            assert_eq!(bytemuck::bytes_of(&c_raw), expected);
        }
    }

    // 3. Verify q_lut1_i8, q_lut2_i8, scale1, scale2 against golden fixture
    let mut q_tokens_raw = vec![0.0_f32; L_Q * DIM];
    for (i, dst) in q_tokens_raw.iter_mut().enumerate() {
        *dst = read_f32_le(LUTRANK_GOLDEN_BYTES, 557_312 + i * 4);
    }
    let mut q_tokens_rot = vec![0.0_f32; L_Q * DIM];
    for (i, dst) in q_tokens_rot.iter_mut().enumerate() {
        *dst = read_f32_le(LUTRANK_GOLDEN_BYTES, 561_408 + i * 4);
    }

    let mut q_luts_per_tok = vec![QueryTokenLuts::default(); L_Q];
    for qi in 0..L_Q {
        synthesize_query_token_luts(
            &q_tokens_rot[qi * DIM..(qi + 1) * DIM],
            &codebooks,
            &mut q_luts_per_tok[qi],
        )
        .unwrap();

        let gs1 = read_f32_le(LUTRANK_GOLDEN_BYTES, 581_888 + qi * 4);
        let gs2 = read_f32_le(LUTRANK_GOLDEN_BYTES, 581_920 + qi * 4);
        assert!((q_luts_per_tok[qi].scale1 - gs1).abs() <= 1e-6);
        assert!((q_luts_per_tok[qi].scale2 - gs2).abs() <= 1e-6);

        // Also check single-token QuantizedQueryLut::synthesize parity
        let single_qlut = QuantizedQueryLut::synthesize(
            &q_tokens_raw[qi * DIM..(qi + 1) * DIM],
            1,
            DIM,
            DEFAULT_LUTRANK_GAMMA,
            SEED,
            &simd,
        )
        .unwrap();
        assert!((single_qlut.scale1 - gs1).abs() <= 1e-5);
        assert!((single_qlut.scale2 - gs2).abs() <= 1e-5);

        for s in 0..LUTRANK_SUBSPACES {
            for c in 0..16 {
                let g1 =
                    i8::from_le_bytes([LUTRANK_GOLDEN_BYTES[565_504 + (qi * 64 + s) * 16 + c]]);
                let g2 =
                    i8::from_le_bytes([LUTRANK_GOLDEN_BYTES[573_696 + (qi * 64 + s) * 16 + c]]);
                assert!((i16::from(q_luts_per_tok[qi].lut1[s][c]) - i16::from(g1)).abs() <= 1);
                assert!((i16::from(q_luts_per_tok[qi].lut2[s][c]) - i16::from(g2)).abs() <= 1);
                assert!((i16::from(single_qlut.lut1_i8[s * 16 + c]) - i16::from(g1)).abs() <= 1);
                assert!((i16::from(single_qlut.lut2_i8[s * 16 + c]) - i16::from(g2)).abs() <= 1);
            }
        }
    }

    // 4. Build OsrStore, reopen zero-copy, and validate MaxSim against golden maxsim_lut_f32
    let docs: Vec<OsrDocumentTokens<'_>> = (0..N_DOCS)
        .map(|d| OsrDocumentTokens {
            doc_id: u64::try_from(d + 1).unwrap(),
            tokens: &doc_tokens_raw[d * L_D * DIM..(d + 1) * L_D * DIM],
            num_tokens: L_D,
        })
        .collect();
    let store = OsrStore::build(&docs, DIM, DEFAULT_LUTRANK_GAMMA, SEED, &simd).unwrap();
    let aligned_bytes = store.to_aligned_bytes().unwrap();
    let reopened = OsrStore::from_bytes(&aligned_bytes).unwrap();

    // Verify per-token golden LUT MaxSim (< 1e-4) and QuantizedQueryLut batch MaxSim
    let batch_qlut =
        QuantizedQueryLut::synthesize(&q_tokens_raw, L_Q, DIM, DEFAULT_LUTRANK_GAMMA, SEED, &simd)
            .unwrap();

    for d in 0..N_DOCS {
        let range = reopened.doc_table()[d];
        let tok_bytes = reopened.token_bytes(&range).unwrap();
        let mut exact_lut_maxsim = 0.0_f32;
        for (qi, qlut) in q_luts_per_tok.iter().enumerate() {
            let mut best = f32::NEG_INFINITY;
            for t in 0..L_D {
                let off = t * 64;
                let g_l1: Vec<i8> = (0..LUTRANK_SUBSPACES * 16)
                    .map(|idx| i8::from_le_bytes([LUTRANK_GOLDEN_BYTES[565_504 + qi * 1024 + idx]]))
                    .collect();
                let g_l2: Vec<i8> = (0..LUTRANK_SUBSPACES * 16)
                    .map(|idx| i8::from_le_bytes([LUTRANK_GOLDEN_BYTES[573_696 + qi * 1024 + idx]]))
                    .collect();
                let sim = simd.lutrank_token_sim_i32(
                    &tok_bytes[off..off + 32],
                    &tok_bytes[off + 32..off + 64],
                    &g_l1,
                    &g_l2,
                    qlut.scale1,
                    qlut.scale2,
                );
                if sim > best {
                    best = sim;
                }
            }
            exact_lut_maxsim += best;
        }
        let golden_lut = read_f32_le(LUTRANK_GOLDEN_BYTES, 582_016 + d * 4);
        assert!(
            (exact_lut_maxsim - golden_lut).abs() <= 1e-4,
            "doc={d} exact_lut_maxsim={exact_lut_maxsim}, golden={golden_lut}"
        );

        let batch_score = reopened.score_document_maxsim(d, &batch_qlut, &simd);
        assert!(
            (batch_score - golden_lut).abs() <= 0.05,
            "doc={d} batch_score={batch_score}, golden={golden_lut}"
        );
    }
}

#[test]
fn miri_mini_osr_store_quantized_lut_and_corruption_checks() {
    let seed = 0x4F53_5052_7777_u64;
    let simd = ScalarRabitqKernel;
    let mut rng = SplitMix64::new(seed);

    let mut make_tokens = |n: usize| -> Vec<f32> {
        let mut v = Vec::with_capacity(n * LUTRANK_TOKEN_DIM);
        for _ in 0..n {
            let mut sq = 0.0_f64;
            let start = v.len();
            for _ in 0..LUTRANK_TOKEN_DIM {
                #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
                let x =
                    (((rng.next_u64() >> 11) as f64) / ((1_u64 << 53) as f64) * 2.0 - 1.0) as f32;
                sq += f64::from(x) * f64::from(x);
                v.push(x);
            }
            let inv = 1.0_f64 / sq.sqrt();
            for x in &mut v[start..] {
                #[allow(clippy::cast_possible_truncation)]
                {
                    *x = (f64::from(*x) * inv) as f32;
                }
            }
        }
        v
    };

    let doc0 = make_tokens(4);
    let doc1 = make_tokens(3);
    let q_toks = make_tokens(2);

    let docs = [
        OsrDocumentTokens {
            doc_id: 20,
            tokens: &doc1,
            num_tokens: 3,
        },
        OsrDocumentTokens {
            doc_id: 10,
            tokens: &doc0,
            num_tokens: 4,
        },
    ];
    let store =
        OsrStore::build(&docs, LUTRANK_TOKEN_DIM, DEFAULT_LUTRANK_GAMMA, seed, &simd).unwrap();
    let bytes = store.to_aligned_bytes().unwrap();
    let view = OsrStore::from_bytes(&bytes).unwrap();

    let q_lut = QuantizedQueryLut::synthesize(
        &q_toks,
        2,
        LUTRANK_TOKEN_DIM,
        DEFAULT_LUTRANK_GAMMA,
        seed,
        &simd,
    )
    .unwrap();

    let s0 = view.score_document_maxsim(0, &q_lut, &simd);
    let s1 = view.score_document_maxsim(1, &q_lut, &simd);
    assert!(s0.is_finite());
    assert!(s1.is_finite());
    assert_eq!(
        view.score_doc_id_maxsim(10, &q_lut, &simd).unwrap(),
        Some(s0)
    );
    assert_eq!(view.score_doc_id_maxsim(99, &q_lut, &simd).unwrap(), None);

    // Corruption detection
    let mut bad = AlignedBuffer::from_slice(&bytes).unwrap();
    bad[12] ^= 0x01;
    assert!(matches!(
        OsrStore::from_bytes(&bad).unwrap_err(),
        OspreyError::ChecksumMismatch { .. }
    ));
}
