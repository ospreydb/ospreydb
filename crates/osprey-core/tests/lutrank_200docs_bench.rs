//! 200-document ColBERT anisotropic benchmark in `osprey-core`
//! (scalar baseline: Pearson > 0.99, Recall@10 >= 0.95, 8x memory compression).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashSet;
use std::time::Instant;

use osprey_core::math::SplitMix64;
use osprey_core::{
    OsrDocumentTokens, OsrStore, QuantizedQueryLut, ScalarRabitqKernel, DEFAULT_LUTRANK_GAMMA,
    LUTRANK_TOKEN_DIM, OSR_TOKEN_BYTES,
};

const N_DOCS: usize = 200;
const L_D: usize = 64;
const L_Q: usize = 16;
const DIM: usize = LUTRANK_TOKEN_DIM;
const SEED: u64 = 0x4F53_5052_0108;

fn next_standard_normal(rng: &mut SplitMix64) -> f64 {
    let u1 = {
        #[allow(clippy::cast_precision_loss)]
        let raw = ((rng.next_u64() >> 11) as f64) / ((1_u64 << 53) as f64);
        raw.max(1e-12)
    };
    let u2 = {
        #[allow(clippy::cast_precision_loss)]
        let raw = ((rng.next_u64() >> 11) as f64) / ((1_u64 << 53) as f64);
        raw
    };
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn normalize_l2_in_place(v: &mut [f32]) {
    let sq: f64 = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let inv = 1.0_f64 / sq.sqrt().max(1e-12);
    for x in v.iter_mut() {
        #[allow(clippy::cast_possible_truncation)]
        {
            *x = (f64::from(*x) * inv) as f32;
        }
    }
}

fn sample_anisotropic_tokens(rng: &mut SplitMix64, count: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; count * DIM];
    for t in 0..count {
        let row = &mut out[t * DIM..(t + 1) * DIM];
        for (d, slot) in row.iter_mut().enumerate() {
            let scale = if d < 4 {
                12.0_f64
            } else {
                #[allow(clippy::cast_precision_loss)]
                {
                    1.0_f64 / (1.0 + (d as f64) * 0.15).sqrt()
                }
            };
            #[allow(clippy::cast_possible_truncation)]
            {
                *slot = (next_standard_normal(rng) * scale) as f32;
            }
        }
        normalize_l2_in_place(row);
    }
    out
}

fn pearson_r(x: &[f32], y: &[f32]) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let n = x.len() as f64;
    let mean_x: f64 = x.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let mean_y: f64 = y.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let mut num = 0.0_f64;
    let mut den_x = 0.0_f64;
    let mut den_y = 0.0_f64;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = f64::from(xi) - mean_x;
        let dy = f64::from(yi) - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    num / (den_x * den_y).sqrt().max(1e-12)
}

#[test]
#[cfg_attr(miri, ignore)]
fn lutrank_200docs_anisotropic_quality_compression_and_scalar_baseline() {
    let mut rng = SplitMix64::new(0x2026_1116_0108);
    let scalar = ScalarRabitqKernel;

    let q_tokens = sample_anisotropic_tokens(&mut rng, L_Q);

    // Build 200 candidate docs with graded semantic token alignment (shuffled order).
    let mut tiers: Vec<usize> = (0..N_DOCS).collect();
    for i in (1..N_DOCS).rev() {
        let j = usize::try_from(rng.next_u64() % u64::try_from(i + 1).unwrap()).unwrap();
        tiers.swap(i, j);
    }

    let mut doc_buffers = Vec::with_capacity(N_DOCS);
    let mut exact_scores = Vec::with_capacity(N_DOCS);

    for &tier in &tiers {
        let overlap = if tier < 10 {
            #[allow(clippy::cast_precision_loss)]
            {
                0.88 - 0.02 * (tier as f64)
            }
        } else if tier < 50 {
            #[allow(clippy::cast_precision_loss)]
            {
                0.52 - 0.005 * ((tier - 10) as f64)
            }
        } else {
            #[allow(clippy::cast_precision_loss)]
            {
                (0.28 - 0.0015 * ((tier - 50) as f64)).max(0.05)
            }
        };

        let mut d_toks = sample_anisotropic_tokens(&mut rng, L_D);
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let n_aligned = ((L_Q as f64) * overlap).max(1.0) as usize;
        for k in 0..n_aligned {
            let iq = k % L_Q;
            let id_tok = (k * 3) % L_D;
            let q_slice = &q_tokens[iq * DIM..(iq + 1) * DIM];
            let d_slice = &mut d_toks[id_tok * DIM..(id_tok + 1) * DIM];
            for (d_val, &q_val) in d_slice.iter_mut().zip(q_slice.iter()) {
                #[allow(clippy::cast_possible_truncation)]
                {
                    *d_val =
                        (overlap * f64::from(q_val) + (1.0 - overlap) * f64::from(*d_val)) as f32;
                }
            }
            normalize_l2_in_place(d_slice);
        }

        // Exact FP32 MaxSim
        let mut exact_maxsim = 0.0_f64;
        for qi in 0..L_Q {
            let q_slice = &q_tokens[qi * DIM..(qi + 1) * DIM];
            let mut best = f64::NEG_INFINITY;
            for ti in 0..L_D {
                let d_slice = &d_toks[ti * DIM..(ti + 1) * DIM];
                let dot: f64 = q_slice
                    .iter()
                    .zip(d_slice.iter())
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                if dot > best {
                    best = dot;
                }
            }
            exact_maxsim += best;
        }
        #[allow(clippy::cast_possible_truncation)]
        exact_scores.push(exact_maxsim as f32);
        doc_buffers.push(d_toks);
    }

    let doc_inputs: Vec<OsrDocumentTokens<'_>> = doc_buffers
        .iter()
        .enumerate()
        .map(|(i, buf)| OsrDocumentTokens {
            doc_id: u64::try_from(i).unwrap(),
            tokens: buf,
            num_tokens: L_D,
        })
        .collect();

    let store = OsrStore::build(&doc_inputs, DIM, DEFAULT_LUTRANK_GAMMA, SEED, &scalar).unwrap();
    let aligned_bytes = store.to_aligned_bytes().unwrap();
    let view = OsrStore::from_bytes(&aligned_bytes).unwrap();

    // Verify 8x compression: 512B/token (FP32) -> 64B/token (.osr)
    let raw_bytes_per_token = DIM * std::mem::size_of::<f32>();
    assert_eq!(raw_bytes_per_token, 512);
    assert_eq!(OSR_TOKEN_BYTES, 64);
    assert_eq!(raw_bytes_per_token / OSR_TOKEN_BYTES, 8);

    // Warm-up + 50 iterations timing
    for _ in 0..5 {
        let ql = QuantizedQueryLut::synthesize(
            &q_tokens,
            L_Q,
            DIM,
            DEFAULT_LUTRANK_GAMMA,
            SEED,
            &scalar,
        )
        .unwrap();
        let _ = view.score_document_maxsim(0, &ql, &scalar);
    }

    let iters = 50usize;
    let t_syn = Instant::now();
    let mut last_ql = None;
    for _ in 0..iters {
        last_ql = Some(
            QuantizedQueryLut::synthesize(
                &q_tokens,
                L_Q,
                DIM,
                DEFAULT_LUTRANK_GAMMA,
                SEED,
                &scalar,
            )
            .unwrap(),
        );
    }
    let syn_us = t_syn.elapsed().as_secs_f64() * 1e6 / (iters as f64);
    let q_lut = last_ql.unwrap();

    let mut lut_scores = vec![0.0_f32; N_DOCS];
    let t_rerank = Instant::now();
    for _ in 0..iters {
        for (d_idx, slot) in lut_scores.iter_mut().enumerate() {
            *slot = view.score_document_maxsim(d_idx, &q_lut, &scalar);
        }
    }
    let rerank_ms = t_rerank.elapsed().as_secs_f64() * 1e3 / (iters as f64);

    let r = pearson_r(&exact_scores, &lut_scores);

    let mut exact_order: Vec<usize> = (0..N_DOCS).collect();
    exact_order.sort_by(|&a, &b| exact_scores[b].total_cmp(&exact_scores[a]));
    let mut lut_order: Vec<usize> = (0..N_DOCS).collect();
    lut_order.sort_by(|&a, &b| lut_scores[b].total_cmp(&lut_scores[a]));

    let gt_top10: HashSet<usize> = exact_order.iter().copied().take(10).collect();
    let pred_top10: HashSet<usize> = lut_order.iter().copied().take(10).collect();
    #[allow(clippy::cast_precision_loss)]
    let recall_at_10 = (gt_top10.intersection(&pred_top10).count() as f64) / 10.0;

    println!(
        "[osprey-core Scalar] 200 docs x 64 toks | 16 query toks | D=128\n\
         \tCompression       : {}B -> {}B ({}x)\n\
         \tLUT synth (scalar): {:.2} us\n\
         \tRerank 200 docs   : {:.3} ms\n\
         \tPearson r         : {:.6}\n\
         \tRecall@10         : {:.2}%",
        raw_bytes_per_token,
        OSR_TOKEN_BYTES,
        raw_bytes_per_token / OSR_TOKEN_BYTES,
        syn_us,
        rerank_ms,
        r,
        recall_at_10 * 100.0
    );

    assert!(r > 0.99, "Pearson r {r:.6} <= 0.99");
    assert!(recall_at_10 >= 0.95, "Recall@10 {recall_at_10:.4} < 0.95");
}
