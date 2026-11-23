//! Low-dimensional (`d = 128`, SIFT1M-like) integration test for
//! [`SearchOptions::resolve`]: adaptive `dim <= 256` oversampling must widen the Stage-1
//! pool enough to recover `Recall@10 >= 0.95` after exact `f16` re-scoring.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashSet;

use half::f16;
use osprey_core::cluster::{fit_binary_kmeans, BinaryKMeansConfig, ScalarClusterKernel};
use osprey_core::math::SplitMix64;
use osprey_core::{OsiIndex, ScalarRabitqKernel, SearchOptions, VectorLocation};
use roaring::RoaringBitmap;

const N_DOCS: usize = 4_000;
const DIM: usize = 128;
const NUM_CLUSTERS: usize = 32;
const N_QUERIES: usize = 50;
const BASE_NPROBE: usize = 6;
const BASE_CANDIDATES: usize = 48;
const TOP_K: usize = 10;
const FWHT_SEED: u64 = 0x4F53_5052_3132_3844;

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

fn normalize_l2_in_place(vec: &mut [f32]) {
    let sq_sum: f64 = vec.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let inv = 1.0_f64 / sq_sum.sqrt().max(1e-12);
    for v in vec.iter_mut() {
        #[allow(clippy::cast_possible_truncation)]
        {
            *v = (f64::from(*v) * inv) as f32;
        }
    }
}

fn build_dataset() -> (Vec<f32>, Vec<f16>, Vec<u64>, Vec<f32>) {
    let mut rng = SplitMix64::new(0x2026_1117_0128);

    let mut centers = vec![0.0_f32; NUM_CLUSTERS * DIM];
    for c in 0..NUM_CLUSTERS {
        let row = &mut centers[c * DIM..(c + 1) * DIM];
        for (d, slot) in row.iter_mut().enumerate() {
            let scale = if d < 8 {
                12.0_f64
            } else {
                1.0_f64 / (1.0 + (d as f64) * 0.20).sqrt()
            };
            #[allow(clippy::cast_possible_truncation)]
            {
                *slot = (next_standard_normal(&mut rng) * scale) as f32;
            }
        }
        normalize_l2_in_place(row);
    }

    let mut db_f32 = vec![0.0_f32; N_DOCS * DIM];
    let mut db_f16 = vec![f16::ZERO; N_DOCS * DIM];
    for i in 0..N_DOCS {
        let c = usize::try_from(rng.next_u64() % u64::try_from(NUM_CLUSTERS).unwrap()).unwrap();
        let center = &centers[c * DIM..(c + 1) * DIM];
        let row = &mut db_f32[i * DIM..(i + 1) * DIM];
        for (d, slot) in row.iter_mut().enumerate() {
            let noise_scale = if d < 8 {
                0.30_f64
            } else {
                0.30_f64 / (1.0 + (d as f64) * 0.20).sqrt()
            };
            #[allow(clippy::cast_possible_truncation)]
            {
                *slot = center[d] + (next_standard_normal(&mut rng) * noise_scale) as f32;
            }
        }
        normalize_l2_in_place(row);
        for (d, &val) in row.iter().enumerate() {
            db_f16[i * DIM + d] = f16::from_f32(val);
        }
    }

    let mut queries = vec![0.0_f32; N_QUERIES * DIM];
    for q in 0..N_QUERIES {
        let c = usize::try_from(rng.next_u64() % u64::try_from(NUM_CLUSTERS).unwrap()).unwrap();
        let center = &centers[c * DIM..(c + 1) * DIM];
        let row = &mut queries[q * DIM..(q + 1) * DIM];
        for (d, slot) in row.iter_mut().enumerate() {
            let noise_scale = if d < 8 {
                0.26_f64
            } else {
                0.26_f64 / (1.0 + (d as f64) * 0.20).sqrt()
            };
            #[allow(clippy::cast_possible_truncation)]
            {
                *slot = center[d] + (next_standard_normal(&mut rng) * noise_scale) as f32;
            }
        }
        normalize_l2_in_place(row);
    }

    let doc_ids: Vec<u64> = (0..N_DOCS).map(|i| u64::try_from(i).unwrap()).collect();
    (db_f32, db_f16, doc_ids, queries)
}

fn ground_truth(db_f32: &[f32], queries: &[f32]) -> Vec<HashSet<u64>> {
    (0..N_QUERIES)
        .map(|qi| {
            let q = &queries[qi * DIM..(qi + 1) * DIM];
            let mut scores: Vec<(usize, f32)> = (0..N_DOCS)
                .map(|vi| {
                    let v = &db_f32[vi * DIM..(vi + 1) * DIM];
                    let mut dot = 0.0_f32;
                    for (&qv, &dv) in q.iter().zip(v.iter()) {
                        dot += qv * dv;
                    }
                    (vi, dot)
                })
                .collect();
            scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            scores[..TOP_K]
                .iter()
                .map(|&(id, _)| u64::try_from(id).unwrap())
                .collect()
        })
        .collect()
}

fn mean_recall_at_10(
    osi: &OsiIndex,
    db_f16: &[f16],
    queries: &[f32],
    truth: &[HashSet<u64>],
    options: SearchOptions,
) -> f64 {
    let kernel = ScalarRabitqKernel;
    let plan = options.resolve(u16::try_from(DIM).unwrap());
    let mut recall_sum = 0.0_f64;
    for qi in 0..N_QUERIES {
        let q = &queries[qi * DIM..(qi + 1) * DIM];
        let stage1 = osi
            .search_candidates(q, plan.effective_nprobe, plan.stage1_limit, &kernel)
            .unwrap();

        let mut refined: Vec<(u64, f32)> = stage1
            .iter()
            .map(|cand| {
                let vi = usize::try_from(cand.doc_id).unwrap();
                let v16 = &db_f16[vi * DIM..(vi + 1) * DIM];
                let mut dot = 0.0_f32;
                for (&qv, &dv16) in q.iter().zip(v16.iter()) {
                    dot += qv * dv16.to_f32();
                }
                (cand.doc_id, dot)
            })
            .collect();
        refined.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let hits = refined[..plan.refine_limit.min(refined.len())]
            .iter()
            .take(TOP_K)
            .filter(|&&(doc_id, _)| truth[qi].contains(&doc_id))
            .count();
        recall_sum += (hits as f64) / (TOP_K as f64);
    }
    recall_sum / (N_QUERIES as f64)
}

#[test]
#[cfg_attr(miri, ignore)]
fn adaptive_low_dim_oversampling_recovers_recall_at_10() {
    let (db_f32, db_f16, doc_ids, queries) = build_dataset();
    let truth = ground_truth(&db_f32, &queries);

    let skm_cfg = BinaryKMeansConfig {
        num_clusters: NUM_CLUSTERS,
        max_iters: 20,
        tolerance: 1e-4,
        seed: 0x4F53_5052_0128,
        spherical: true,
    };
    let cluster_kernel = ScalarClusterKernel;
    let skm =
        fit_binary_kmeans(&db_f32, N_DOCS, DIM, FWHT_SEED, &skm_cfg, &cluster_kernel).unwrap();

    let locations = vec![VectorLocation::default(); N_DOCS];
    let kernel = ScalarRabitqKernel;
    let osi = OsiIndex::build(
        &db_f32,
        &doc_ids,
        &locations,
        &skm,
        DIM,
        FWHT_SEED,
        RoaringBitmap::new(),
        &kernel,
    )
    .unwrap();

    let base = SearchOptions {
        refine_multiplier: 1,
        ..SearchOptions::new(BASE_NPROBE, BASE_CANDIDATES, TOP_K)
    };
    let adaptive = SearchOptions::new(BASE_NPROBE, BASE_CANDIDATES, TOP_K);

    let base_plan = base.resolve(u16::try_from(DIM).unwrap());
    let adaptive_plan = adaptive.resolve(u16::try_from(DIM).unwrap());
    assert_eq!(base_plan.refine_multiplier, 1);
    assert_eq!(adaptive_plan.refine_multiplier, 2);
    assert!(adaptive_plan.low_dim_oversample);
    assert!(adaptive_plan.stage1_limit > base_plan.stage1_limit);

    let base_recall = mean_recall_at_10(&osi, &db_f16, &queries, &truth, base);
    let adaptive_recall = mean_recall_at_10(&osi, &db_f16, &queries, &truth, adaptive);

    println!(
        "\n=== Adaptive low-dim (d=128, K=32, nprobe={BASE_NPROBE}, candidates={BASE_CANDIDATES}) ==="
    );
    println!(
        "  Base     : nprobe={} pool={} -> Recall@10 {:.2}%",
        base_plan.effective_nprobe,
        base_plan.stage1_limit,
        base_recall * 100.0
    );
    println!(
        "  Adaptive : nprobe={} pool={} -> Recall@10 {:.2}%",
        adaptive_plan.effective_nprobe,
        adaptive_plan.stage1_limit,
        adaptive_recall * 100.0
    );

    assert!(
        adaptive_recall >= 0.95,
        "adaptive Recall@10 {:.4} is below 0.95",
        adaptive_recall
    );
    assert!(
        adaptive_recall >= base_recall,
        "adaptive Recall@10 {:.4} regressed vs base {:.4}",
        adaptive_recall,
        base_recall
    );
}
