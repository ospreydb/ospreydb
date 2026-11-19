//! Large-scale anisotropic 20,000-vector (d=512) integration test
//! verifying `Recall@10 >= 96.5%` after `f16` refinement of Top-150 Stage-1 candidates.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashSet;
use std::time::Instant;

use half::f16;
use osprey_core::cluster::{fit_binary_kmeans, BinaryKMeansConfig, ScalarClusterKernel};
use osprey_core::math::SplitMix64;
use osprey_core::{OsiIndex, ScalarRabitqKernel, VectorLocation};
use roaring::RoaringBitmap;

const N_DOCS: usize = 20_000;
const DIM: usize = 512;
const NUM_CLUSTERS: usize = 64;
const N_QUERIES: usize = 100;
const NPROBE: usize = 12;
const TOP_M: usize = 150;
const TOP_K: usize = 10;
const FWHT_SEED: u64 = 0x4F53_5052_5333_5636;

/// Deterministic Box-Muller standard normal sampler backed by `SplitMix64`.
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

#[test]
#[cfg_attr(miri, ignore)]
fn rabitq_anisotropic_20k_recall_at_10_above_965() {
    let mut rng = SplitMix64::new(0x2026_1116_0092);

    // Build 64 strongly anisotropic cluster centroids (15.0x on first 8 dims)
    let mut centers = vec![0.0_f32; NUM_CLUSTERS * DIM];
    for c in 0..NUM_CLUSTERS {
        let row = &mut centers[c * DIM..(c + 1) * DIM];
        for (d, slot) in row.iter_mut().enumerate() {
            let scale = if d < 8 {
                15.0_f64
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

    // Generate 20,000 anisotropic clustered vectors (8.0x noise on first 8 dims)
    let mut db_f32 = vec![0.0_f32; N_DOCS * DIM];
    let mut db_f16 = vec![f16::ZERO; N_DOCS * DIM];
    for i in 0..N_DOCS {
        let c = usize::try_from(rng.next_u64() % u64::try_from(NUM_CLUSTERS).unwrap()).unwrap();
        let center = &centers[c * DIM..(c + 1) * DIM];
        let row = &mut db_f32[i * DIM..(i + 1) * DIM];
        for (d, slot) in row.iter_mut().enumerate() {
            let noise_scale = if d < 8 {
                0.28_f64 * (8.0_f64 / 15.0_f64)
            } else {
                0.28_f64 / (1.0 + (d as f64) * 0.20).sqrt()
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

    // Generate 100 queries drawn from cluster topics with anisotropic perturbation
    let mut queries = vec![0.0_f32; N_QUERIES * DIM];
    for q in 0..N_QUERIES {
        let c = usize::try_from(rng.next_u64() % u64::try_from(NUM_CLUSTERS).unwrap()).unwrap();
        let center = &centers[c * DIM..(c + 1) * DIM];
        let row = &mut queries[q * DIM..(q + 1) * DIM];
        for (d, slot) in row.iter_mut().enumerate() {
            let noise_scale = if d < 8 {
                0.24_f64 * (8.0_f64 / 15.0_f64)
            } else {
                0.24_f64 / (1.0 + (d as f64) * 0.20).sqrt()
            };
            #[allow(clippy::cast_possible_truncation)]
            {
                *slot = center[d] + (next_standard_normal(&mut rng) * noise_scale) as f32;
            }
        }
        normalize_l2_in_place(row);
    }

    // Train 1-Bit Binary K-Means (K=64) and build cluster-grouped .osi index
    let skm_cfg = BinaryKMeansConfig {
        num_clusters: NUM_CLUSTERS,
        max_iters: 20,
        tolerance: 1e-4,
        seed: 0x4F53_5052_0002,
        spherical: true,
    };
    let cluster_kernel = ScalarClusterKernel;
    let t_build = Instant::now();
    let skm =
        fit_binary_kmeans(&db_f32, N_DOCS, DIM, FWHT_SEED, &skm_cfg, &cluster_kernel).unwrap();

    let doc_ids: Vec<u64> = (0..N_DOCS).map(|i| u64::try_from(i).unwrap()).collect();
    let locations = vec![VectorLocation::default(); N_DOCS];
    let scan_kernel = ScalarRabitqKernel;
    let osi = OsiIndex::build(
        &db_f32,
        &doc_ids,
        &locations,
        &skm,
        DIM,
        FWHT_SEED,
        RoaringBitmap::new(),
        &scan_kernel,
    )
    .unwrap();
    let osi_bytes = osi.to_aligned_bytes().unwrap();
    let build_ms = t_build.elapsed().as_secs_f64() * 1000.0;

    let raw_f32_bytes_per_vec = (DIM * std::mem::size_of::<f32>()) as f64;
    let osi_bytes_per_vec = (osi_bytes.len() as f64) / (N_DOCS as f64);

    // Evaluate all 100 queries: brute-force FP32 ground truth vs Stage-1 .osi + Top-150 f16 refine
    let mut recall_sum = 0.0_f64;
    let mut total_scan_us = 0.0_f64;

    for qi in 0..N_QUERIES {
        let q = &queries[qi * DIM..(qi + 1) * DIM];

        // Exact FP32 ground-truth Top-10
        let mut gt_scores: Vec<(usize, f32)> = (0..N_DOCS)
            .map(|vi| {
                let v = &db_f32[vi * DIM..(vi + 1) * DIM];
                let mut dot = 0.0_f32;
                for (&qv, &dv) in q.iter().zip(v.iter()) {
                    dot += qv * dv;
                }
                (vi, dot)
            })
            .collect();
        gt_scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let gt_top10: HashSet<u64> = gt_scores[..TOP_K]
            .iter()
            .map(|&(id, _)| u64::try_from(id).unwrap())
            .collect();

        // Stage-1 .osi scan (nprobe=12, top_m=150)
        let t0 = Instant::now();
        let stage1 = osi
            .search_candidates(q, NPROBE, TOP_M, &scan_kernel)
            .unwrap();
        total_scan_us += t0.elapsed().as_secs_f64() * 1_000_000.0;

        // Exact f16 refinement on surviving Top-150 candidates
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

        let hits = refined[..TOP_K.min(refined.len())]
            .iter()
            .filter(|&&(doc_id, _)| gt_top10.contains(&doc_id))
            .count();
        recall_sum += (hits as f64) / (TOP_K as f64);
    }

    let recall_at_10 = recall_sum / (N_QUERIES as f64);
    let avg_scan_us = total_scan_us / (N_QUERIES as f64);
    let avg_scan_ms = avg_scan_us / 1000.0;

    println!("\n=== Anisotropic 20k (d=512, K=64, nprobe=12, Top-150 -> f16 Top-10) ===");
    println!("  Build + .osi serialize time : {build_ms:.2} ms");
    println!(
        "  .osi size per vector        : {osi_bytes_per_vec:.2} B/vec (vs {raw_f32_bytes_per_vec:.0} B/vec raw f32, {:.2}x smaller)",
        raw_f32_bytes_per_vec / osi_bytes_per_vec
    );
    println!("  Avg Stage-1 .osi scan       : {avg_scan_ms:.3} ms ({avg_scan_us:.1} us/query)");
    println!(
        "  Mean Recall@10 (100 queries): {:.2}% (target >= 96.50%)",
        recall_at_10 * 100.0
    );

    assert!(
        recall_at_10 >= 0.965,
        "Recall@10 {:.4} is below required 0.965 threshold",
        recall_at_10
    );
}
