//! Integration test validating TSP 2-opt cluster locality improvement and
//! zero-copy `memmap2` `.osv` slab reading, Zstd decompression, exact `f16`
//! inner product refinement, and 1-bit `page_xxh64` corruption detection.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;

use half::f16;
use osprey_core::{OspreyError, OspreyResult};
use osprey_storage::{
    compute_centroid_distance_matrix, tsp_nearest_neighbor, tsp_two_opt_refine,
    ClusterLocalityPermutation, OsvSlabReader, OsvVectorRecord, OsvWriter, OsvWriterConfig,
};

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_unit_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32) / ((1_u32 << 24) as f32)
    }
}

fn temp_test_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("osprey_phase8_test_{}", std::process::id()));
    let _ = fs::create_dir_all(&dir);
    dir
}

fn measure_top_clusters_dispersion(
    reader: &OsvSlabReader,
    target_physical_clusters: &[u32],
) -> (u64, u64) {
    let set: HashSet<u32> = target_physical_clusters.iter().copied().collect();
    let mut entries: Vec<(u64, u64)> = reader
        .directory()
        .iter()
        .filter(|e| set.contains(&e.cluster_id))
        .map(|e| (e.byte_offset, e.byte_offset + u64::from(e.compressed_size)))
        .collect();
    entries.sort_unstable_by_key(|&(start, _)| start);

    if entries.is_empty() {
        return (0, 0);
    }

    let min_offset = entries[0].0;
    let max_offset = entries[entries.len() - 1].0;
    let byte_span = max_offset - min_offset;

    let mut gap_sum = 0_u64;
    for pair in entries.windows(2) {
        let prev_end = pair[0].1;
        let next_start = pair[1].0;
        if next_start > prev_end {
            gap_sum += next_start - prev_end;
        }
    }

    (byte_span, gap_sum)
}

#[test]
fn tsp_reordering_reduces_top8_offset_dispersion_and_validates_mmap_roundtrip() -> OspreyResult<()>
{
    const K: usize = 64;
    const D: usize = 128;
    const VECS_PER_CLUSTER: usize = 40;
    const N: usize = K * VECS_PER_CLUSTER; // 2,560 vectors
    const NUM_QUERIES: usize = 100;
    const TOP_PROBE: usize = 8;

    let mut rng = SplitMix64(0x4F53_5052_2026_0008);

    // 1. Build 64 centroids along a smooth 3D curve embedded in R^128, then shuffle IDs 0..63.
    let mut curve_centroids = Vec::with_capacity(K);
    for m in 0..K {
        let t = (m as f32) / ((K - 1) as f32);
        let mut c = vec![0.0_f32; D];
        for (coord, val) in c.iter_mut().enumerate() {
            let block = coord % 3;
            let amp = 1.0_f32 / (1.0 + (coord / 3) as f32 * 0.05);
            *val = match block {
                0 => (1.0 - t) * 4.0 * amp,
                1 => t * 4.0 * amp,
                _ => (std::f32::consts::PI * t).sin() * 0.8 * amp,
            };
        }
        curve_centroids.push(c);
    }

    // Fisher-Yates shuffle into logical cluster IDs 0..63.
    let mut shuffled_order: Vec<usize> = (0..K).collect();
    for i in (1..K).rev() {
        let j = usize::try_from(rng.next_u64() % u64::try_from(i + 1).unwrap_or(1)).unwrap_or(0);
        shuffled_order.swap(i, j);
    }

    let mut centroids_f32 = vec![0.0_f32; K * D];
    let mut centroids_f16 = vec![f16::ZERO; K * D];
    for (logical_id, &curve_idx) in shuffled_order.iter().enumerate() {
        for d in 0..D {
            let v = curve_centroids[curve_idx][d];
            centroids_f32[logical_id * D + d] = v;
            centroids_f16[logical_id * D + d] = f16::from_f32(v);
        }
    }

    // 2. Generate N = 2,560 vectors (40 per cluster) and realistic JSON payloads.
    let mut all_vectors_f16: Vec<Vec<f16>> = Vec::with_capacity(N);
    let mut all_payloads: Vec<Vec<u8>> = Vec::with_capacity(N);
    let mut logical_clusters: Vec<u32> = Vec::with_capacity(N);
    let mut doc_ids: Vec<u64> = Vec::with_capacity(N);

    for logical_id in 0..K {
        for slot in 0..VECS_PER_CLUSTER {
            let doc_id = u64::try_from(logical_id * VECS_PER_CLUSTER + slot + 10_000).unwrap_or(0);
            let mut vec_f16 = Vec::with_capacity(D);
            for d in 0..D {
                let jitter = (rng.next_unit_f32() - 0.5) * 0.04;
                vec_f16.push(f16::from_f32(centroids_f32[logical_id * D + d] + jitter));
            }
            let payload = format!(
                r#"{{"doc_id":{doc_id},"logical_cluster":{logical_id},"slot":{slot},"tenant":"osprey-bench","region":"us-east-1","tags":["ann","rabitq","osv","tsp"],"score":{:.4}}}"#,
                rng.next_unit_f32()
            )
            .into_bytes();

            all_vectors_f16.push(vec_f16);
            all_payloads.push(payload);
            logical_clusters.push(u32::try_from(logical_id).unwrap_or(0));
            doc_ids.push(doc_id);
        }
    }

    let records: Vec<OsvVectorRecord<'_>> = (0..N)
        .map(|i| OsvVectorRecord {
            doc_id: doc_ids[i],
            cluster_id: logical_clusters[i],
            vector_f16: &all_vectors_f16[i],
            payload: &all_payloads[i],
        })
        .collect();

    // 3. Compute TSP ordering (Nearest-Neighbor + 2-opt) starting from the most peripheral
    //    centroid (furthest from the global centroid mean) so the open path traces the manifold end-to-end.
    let dist_matrix = compute_centroid_distance_matrix(&centroids_f16, K, D)?;
    let mut mean_centroid = vec![0.0_f32; D];
    for c in 0..K {
        for d in 0..D {
            mean_centroid[d] += centroids_f16[c * D + d].to_f32();
        }
    }
    for val in &mut mean_centroid {
        *val /= K as f32;
    }
    let start_cluster = (0..K)
        .max_by(|&a, &b| {
            let da: f32 = (0..D)
                .map(|d| {
                    let diff = centroids_f16[a * D + d].to_f32() - mean_centroid[d];
                    diff * diff
                })
                .sum();
            let db: f32 = (0..D)
                .map(|d| {
                    let diff = centroids_f16[b * D + d].to_f32() - mean_centroid[d];
                    diff * diff
                })
                .sum();
            da.total_cmp(&db)
        })
        .unwrap_or(0);
    let mut tour = tsp_nearest_neighbor(&dist_matrix, K, start_cluster);
    let _final_tour_len = tsp_two_opt_refine(&mut tour, &dist_matrix, 50);
    let perm = ClusterLocalityPermutation::from_tour(tour)?;

    // 4. Write both (A) Random-order .osv and (B) TSP-order .osv to disk, then open via memmap2.
    let tmp_dir = temp_test_dir();
    let random_path = tmp_dir.join("random_order.osv");
    let tsp_path = tmp_dir.join("tsp_order.osv");
    let corrupted_path = tmp_dir.join("tsp_corrupted.osv");

    let writer = OsvWriter::new(OsvWriterConfig::default());

    let mut random_file = fs::File::create(&random_path)?;
    let _random_report = writer.write_slab(&mut random_file, &records, D, None)?;
    drop(random_file);

    let mut tsp_file = fs::File::create(&tsp_path)?;
    let tsp_report = writer.write_slab(&mut tsp_file, &records, D, Some(&perm))?;
    drop(tsp_file);

    let random_reader = OsvSlabReader::open_mmap(&random_path)?;
    let tsp_reader = OsvSlabReader::open_mmap(&tsp_path)?;

    // 5. Measure Top-8 cluster byte_span and gap_sum across 100 random queries.
    let mut total_random_span = 0_u64;
    let mut total_tsp_span = 0_u64;
    let mut total_random_gaps = 0_u64;
    let mut total_tsp_gaps = 0_u64;

    for _q in 0..NUM_QUERIES {
        let t = rng.next_unit_f32();
        let mut query = vec![0.0_f32; D];
        for (coord, val) in query.iter_mut().enumerate() {
            let block = coord % 3;
            let amp = 1.0_f32 / (1.0 + (coord / 3) as f32 * 0.05);
            let base = match block {
                0 => (1.0 - t) * 4.0 * amp,
                1 => t * 4.0 * amp,
                _ => (std::f32::consts::PI * t).sin() * 0.8 * amp,
            };
            *val = base + (rng.next_unit_f32() - 0.5) * 0.05;
        }

        let mut cluster_dists: Vec<(u32, f32)> = (0..K)
            .map(|c| {
                let mut sq = 0.0_f32;
                for d in 0..D {
                    let diff = centroids_f16[c * D + d].to_f32() - query[d];
                    sq += diff * diff;
                }
                (u32::try_from(c).unwrap_or(0), sq)
            })
            .collect();
        cluster_dists.sort_by(|a, b| a.1.total_cmp(&b.1));

        let top8_logical: Vec<u32> = cluster_dists
            .iter()
            .take(TOP_PROBE)
            .map(|&(c, _)| c)
            .collect();
        let mut top8_physical = Vec::with_capacity(TOP_PROBE);
        for &logical_id in &top8_logical {
            top8_physical.push(perm.to_physical(logical_id)?);
        }

        let (r_span, r_gaps) = measure_top_clusters_dispersion(&random_reader, &top8_logical);
        let (t_span, t_gaps) = measure_top_clusters_dispersion(&tsp_reader, &top8_physical);

        total_random_span += r_span;
        total_tsp_span += t_span;
        total_random_gaps += r_gaps;
        total_tsp_gaps += t_gaps;
    }

    let avg_random_span = (total_random_span as f64) / (NUM_QUERIES as f64);
    let avg_tsp_span = (total_tsp_span as f64) / (NUM_QUERIES as f64);
    let span_reduction = avg_random_span / avg_tsp_span.max(1.0);

    let avg_random_gaps = (total_random_gaps as f64) / (NUM_QUERIES as f64);
    let avg_tsp_gaps = (total_tsp_gaps as f64) / (NUM_QUERIES as f64);
    let gap_reduction = avg_random_gaps / avg_tsp_gaps.max(1.0);

    println!(
        "TSP Locality Benchmark (K={K}, N={N}, Top-{TOP_PROBE} over {NUM_QUERIES} queries):\n\
         - Random Avg Byte Span: {avg_random_span:.1} B | TSP Avg Byte Span: {avg_tsp_span:.1} B (Reduction: {span_reduction:.2}x)\n\
         - Random Avg Gap Sum:   {avg_random_gaps:.1} B | TSP Avg Gap Sum:   {avg_tsp_gaps:.1} B (Reduction: {gap_reduction:.2}x)"
    );

    assert!(
        span_reduction >= 4.0,
        "expected TSP to reduce byte span by >= 4.0x, got {span_reduction:.2}x"
    );
    assert!(
        gap_reduction >= 4.0,
        "expected TSP to reduce byte gaps by >= 4.0x, got {gap_reduction:.2}x"
    );

    // 6. Validate exact memmap2 roundtrip for all 2,560 vectors, JSON payloads, and f16 dot product.
    let mut decoded_pages = HashMap::new();
    let test_query: Vec<f32> = (0..D).map(|d| (d as f32) * 0.01 - 0.5).collect();

    for i in 0..N {
        let doc_id = doc_ids[i];
        let (minipage_id, slot_idx) = tsp_report
            .location_for_doc(doc_id)
            .ok_or(OspreyError::CorruptedFooter)?;

        if let std::collections::hash_map::Entry::Vacant(entry) = decoded_pages.entry(minipage_id) {
            let page = tsp_reader.decompress_minipage(minipage_id)?;
            entry.insert(page);
        }
        let page = &decoded_pages[&minipage_id];

        assert_eq!(page.doc_id(slot_idx)?, doc_id);
        assert_eq!(page.vector_f16(slot_idx)?, all_vectors_f16[i].as_slice());
        assert_eq!(page.payload_bytes(slot_idx)?, all_payloads[i].as_slice());
        let json_str = page.payload_str(slot_idx)?;
        let parsed: serde_json::Value = serde_json::from_str(json_str).map_err(|err| {
            OspreyError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, err))
        })?;
        assert_eq!(parsed["doc_id"].as_u64(), Some(doc_id));

        let dot = page.compute_exact_f16_dot(slot_idx, &test_query);
        let expected_dot: f32 = all_vectors_f16[i]
            .iter()
            .zip(test_query.iter())
            .map(|(a, &b)| a.to_f32() * b)
            .sum();
        assert!(
            (dot - expected_dot).abs() <= 1e-4,
            "dot mismatch for doc {doc_id}: got {dot}, expected {expected_dot}"
        );
    }

    // 7. Flip 1 bit inside compressed minipage 0 on disk and assert ChecksumMismatch via memmap2.
    let mut corrupted_bytes = fs::read(&tsp_path)?;
    let target_byte = usize::try_from(tsp_reader.directory()[0].byte_offset).unwrap_or(64);
    corrupted_bytes[target_byte] ^= 0x01;
    fs::write(&corrupted_path, &corrupted_bytes)?;

    let corrupted_reader = OsvSlabReader::open_mmap(&corrupted_path)?;
    let err = corrupted_reader.decompress_minipage(0);
    assert!(matches!(err, Err(OspreyError::ChecksumMismatch { .. })));

    let _ = fs::remove_file(&random_path);
    let _ = fs::remove_file(&tsp_path);
    let _ = fs::remove_file(&corrupted_path);
    let _ = fs::remove_dir(&tmp_dir);

    Ok(())
}
