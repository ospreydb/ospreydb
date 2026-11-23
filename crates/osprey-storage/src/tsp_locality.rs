//! TSP centroid locality reordering for `.osi` and `.osv` slab layouts.
//!
//! Computes symmetric squared-L2 distances across `f16` centroids, builds an
//! initial deterministic Nearest-Neighbor open path, refines it via 2-opt local
//! search, and provides bidirectional cluster permutation helpers to reorder
//! centroids and remap vector `cluster_id`s before writing `.osv` slabs.

use std::cmp::Ordering;
use std::mem::{size_of, size_of_val};

use half::f16;
use osprey_core::{AlignedBuffer, OspreyError, OspreyResult};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Cluster count `K` threshold at or above which distance matrix computation uses `rayon`.
pub const TSP_PARALLEL_CLUSTER_THRESHOLD: usize = 64;

/// Floating-point improvement threshold for accepting a 2-opt edge swap.
const TWO_OPT_EPSILON: f32 = 1e-6;

/// Bidirectional permutation between logical cluster IDs and TSP-ordered physical positions.
///
/// Physical position `p` in `.osi`/`.osv` stores logical cluster `physical_to_logical[p]`,
/// while logical cluster `c` is placed at physical index `logical_to_physical[c]`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClusterLocalityPermutation {
    /// Optimal TSP tour where `physical_to_logical[physical_id] == logical_id`.
    pub physical_to_logical: Vec<u32>,
    /// Inverse permutation where `logical_to_physical[logical_id] == physical_id`.
    pub logical_to_physical: Vec<u32>,
}

impl ClusterLocalityPermutation {
    /// Builds a validated [`ClusterLocalityPermutation`] from a `physical_to_logical` TSP tour.
    ///
    /// Returns an error if `physical_to_logical` is not a bijection over `0..num_clusters`.
    pub fn from_tour(physical_to_logical: Vec<u32>) -> OspreyResult<Self> {
        let num_clusters = physical_to_logical.len();
        if num_clusters > 0 && u32::try_from(num_clusters - 1).is_err() {
            return Err(OspreyError::InvalidDimension { dim: num_clusters });
        }

        let mut logical_to_physical = vec![u32::MAX; num_clusters];
        for (physical_idx, &logical_id) in physical_to_logical.iter().enumerate() {
            let logical = idx_from_u32(logical_id);
            if logical >= num_clusters || logical_to_physical[logical] != u32::MAX {
                return Err(OspreyError::InvalidDimension { dim: logical });
            }
            let physical_u32 = u32::try_from(physical_idx)
                .map_err(|_| OspreyError::InvalidDimension { dim: physical_idx })?;
            logical_to_physical[logical] = physical_u32;
        }

        Ok(Self {
            physical_to_logical,
            logical_to_physical,
        })
    }

    /// Creates the identity permutation for `num_clusters` clusters.
    pub fn identity(num_clusters: usize) -> OspreyResult<Self> {
        if num_clusters > 0 && u32::try_from(num_clusters - 1).is_err() {
            return Err(OspreyError::InvalidDimension { dim: num_clusters });
        }
        let mut ids = Vec::with_capacity(num_clusters);
        for idx in 0..num_clusters {
            let id = u32::try_from(idx).map_err(|_| OspreyError::InvalidDimension { dim: idx })?;
            ids.push(id);
        }
        Ok(Self {
            physical_to_logical: ids.clone(),
            logical_to_physical: ids,
        })
    }

    /// Returns the number of clusters `K` covered by this permutation.
    #[inline]
    pub fn num_clusters(&self) -> usize {
        self.physical_to_logical.len()
    }

    /// Maps a logical cluster ID to its TSP-ordered physical cluster ID.
    #[inline]
    pub fn to_physical(&self, logical_cluster_id: u32) -> OspreyResult<u32> {
        let idx = idx_from_u32(logical_cluster_id);
        self.logical_to_physical
            .get(idx)
            .copied()
            .ok_or(OspreyError::InvalidDimension { dim: idx })
    }

    /// Maps a physical cluster ID back to its original logical cluster ID.
    #[inline]
    pub fn to_logical(&self, physical_cluster_id: u32) -> OspreyResult<u32> {
        let idx = idx_from_u32(physical_cluster_id);
        self.physical_to_logical
            .get(idx)
            .copied()
            .ok_or(OspreyError::InvalidDimension { dim: idx })
    }

    /// Reorders an `[K, padded_dim]` `f16` centroid table into physical TSP order.
    pub fn reorder_centroids_f16(
        &self,
        centroids_f16: &[f16],
        padded_dim: usize,
    ) -> OspreyResult<Vec<f16>> {
        let expected_len =
            validate_centroid_table_shape(centroids_f16, self.num_clusters(), padded_dim)?;
        let mut reordered = vec![f16::ZERO; expected_len];
        copy_reordered_centroids(
            &self.physical_to_logical,
            centroids_f16,
            padded_dim,
            &mut reordered,
        );
        Ok(reordered)
    }

    /// Reorders an `[K, padded_dim]` `f16` centroid table into a 64-byte-aligned buffer.
    pub fn reorder_centroids_f16_aligned(
        &self,
        centroids_f16: &[f16],
        padded_dim: usize,
    ) -> OspreyResult<AlignedBuffer<f16>> {
        let expected_len =
            validate_centroid_table_shape(centroids_f16, self.num_clusters(), padded_dim)?;
        let mut reordered = AlignedBuffer::<f16>::new(expected_len)?;
        copy_reordered_centroids(
            &self.physical_to_logical,
            centroids_f16,
            padded_dim,
            reordered.as_mut_slice(),
        );
        Ok(reordered)
    }

    /// Remaps vector `cluster_id`s from logical IDs to TSP physical IDs in a new `Vec<u32>`.
    pub fn remap_cluster_ids(&self, logical_cluster_ids: &[u32]) -> OspreyResult<Vec<u32>> {
        let mut remapped = Vec::with_capacity(logical_cluster_ids.len());
        for &logical_id in logical_cluster_ids {
            remapped.push(self.to_physical(logical_id)?);
        }
        Ok(remapped)
    }

    /// Remaps vector `cluster_id`s in-place from logical IDs to TSP physical IDs.
    ///
    /// Validates all IDs before mutating so an out-of-range ID leaves `cluster_ids` unchanged.
    pub fn remap_cluster_ids_in_place(&self, cluster_ids: &mut [u32]) -> OspreyResult<()> {
        for &logical_id in cluster_ids.iter() {
            let _ = self.to_physical(logical_id)?;
        }
        for cluster_id in cluster_ids.iter_mut() {
            *cluster_id = self.logical_to_physical[idx_from_u32(*cluster_id)];
        }
        Ok(())
    }

    /// Reorders per-cluster cardinalities `[K]` from logical order into physical TSP order.
    pub fn reorder_cluster_counts(&self, logical_cluster_counts: &[u64]) -> OspreyResult<Vec<u64>> {
        let k = self.num_clusters();
        if logical_cluster_counts.len() < k {
            return Err(OspreyError::BufferTooSmall {
                needed: k * size_of::<u64>(),
                actual: size_of_val(logical_cluster_counts),
            });
        }
        if logical_cluster_counts.len() != k {
            return Err(OspreyError::InvalidDimension {
                dim: logical_cluster_counts.len(),
            });
        }
        let mut reordered = Vec::with_capacity(k);
        for &logical_id in &self.physical_to_logical {
            reordered.push(logical_cluster_counts[idx_from_u32(logical_id)]);
        }
        Ok(reordered)
    }
}

/// Computes the symmetric `K x K` squared Euclidean distance matrix over `f16` centroids.
///
/// Promotes each `f16` coordinate to `f32` and computes
/// `dist_matrix[i * num_clusters + j] = sum_d (c_i[d] - c_j[d])^2`.
/// Uses `rayon` parallelization when `num_clusters >= 64`.
pub fn compute_centroid_distance_matrix(
    centroids_f16: &[f16],
    num_clusters: usize,
    padded_dim: usize,
) -> OspreyResult<Vec<f32>> {
    let expected_len = validate_centroid_table_shape(centroids_f16, num_clusters, padded_dim)?;
    if num_clusters == 0 {
        return Ok(Vec::new());
    }

    let matrix_len = num_clusters
        .checked_mul(num_clusters)
        .ok_or(OspreyError::InvalidDimension { dim: num_clusters })?;

    let mut centroids_f32 = Vec::with_capacity(expected_len);
    for (index, &val) in centroids_f16.iter().enumerate() {
        let promoted = val.to_f32();
        if !promoted.is_finite() {
            return Err(OspreyError::NonFiniteEmbedding { index });
        }
        centroids_f32.push(promoted);
    }

    let mut dist_matrix = vec![0.0_f32; matrix_len];
    let fill_row = |(i, row): (usize, &mut [f32])| {
        let ci = &centroids_f32[i * padded_dim..(i + 1) * padded_dim];
        for (j, cell) in row.iter_mut().enumerate() {
            if i == j {
                *cell = 0.0_f32;
            } else {
                let cj = &centroids_f32[j * padded_dim..(j + 1) * padded_dim];
                *cell = squared_l2_f32(ci, cj);
            }
        }
    };

    if num_clusters >= TSP_PARALLEL_CLUSTER_THRESHOLD {
        dist_matrix
            .par_chunks_mut(num_clusters)
            .enumerate()
            .for_each(fill_row);
    } else {
        dist_matrix
            .chunks_mut(num_clusters)
            .enumerate()
            .for_each(fill_row);
    }

    Ok(dist_matrix)
}

/// Builds a deterministic Nearest-Neighbor open TSP tour starting at `start_cluster`.
///
/// At each step, chooses the unvisited cluster with minimum distance in `dist_matrix`,
/// breaking exact ties strictly by lowest cluster index. Returns an empty `Vec` if
/// `num_clusters == 0`, `start_cluster >= num_clusters`, or `dist_matrix` is too small.
pub fn tsp_nearest_neighbor(
    dist_matrix: &[f32],
    num_clusters: usize,
    start_cluster: usize,
) -> Vec<u32> {
    if num_clusters == 0 || start_cluster >= num_clusters {
        return Vec::new();
    }
    let Some(expected_len) = num_clusters.checked_mul(num_clusters) else {
        return Vec::new();
    };
    if dist_matrix.len() < expected_len {
        return Vec::new();
    }
    if u32::try_from(num_clusters - 1).is_err() {
        return Vec::new();
    }

    let mut visited = vec![false; num_clusters];
    let mut tour = Vec::with_capacity(num_clusters);

    let mut current = start_cluster;
    visited[current] = true;
    let Ok(start_u32) = u32::try_from(current) else {
        return Vec::new();
    };
    tour.push(start_u32);

    for _ in 1..num_clusters {
        let row = &dist_matrix[current * num_clusters..(current + 1) * num_clusters];
        let mut best_next: Option<usize> = None;
        let mut best_dist = f32::INFINITY;

        for (candidate, &dist) in row.iter().enumerate() {
            if visited[candidate] {
                continue;
            }
            let is_better = match best_next {
                None => true,
                Some(_) => (!dist.is_nan() && best_dist.is_nan()) || dist < best_dist,
            };
            if is_better {
                best_dist = dist;
                best_next = Some(candidate);
            }
        }

        let Some(next) = best_next else {
            break;
        };
        visited[next] = true;
        let Ok(next_u32) = u32::try_from(next) else {
            return Vec::new();
        };
        tour.push(next_u32);
        current = next;
    }

    tour
}

/// Refines an open TSP path in-place with 2-opt local search and returns the final path length.
///
/// Evaluates edge pairs `(tour[i - 1], tour[i])` and `(tour[j], tour[j + 1])` with
/// `1 <= i < j < tour.len() - 1`, reversing `tour[i..=j]` whenever the swap reduces
/// length by more than `1e-6`. Stops when a pass yields zero improvements or after
/// `max_passes` iterations.
pub fn tsp_two_opt_refine(tour: &mut [u32], dist_matrix: &[f32], max_passes: usize) -> f32 {
    let Some(stride) = matrix_stride_for_tour(dist_matrix, tour) else {
        return 0.0_f32;
    };

    let k = tour.len();
    if k >= 4 && max_passes > 0 {
        for _ in 0..max_passes {
            let mut improved = false;
            for i in 1..(k - 2) {
                for j in (i + 1)..(k - 1) {
                    let a = idx_from_u32(tour[i - 1]);
                    let b = idx_from_u32(tour[i]);
                    let c = idx_from_u32(tour[j]);
                    let d = idx_from_u32(tour[j + 1]);

                    let before = dist_matrix[a * stride + b] + dist_matrix[c * stride + d];
                    let after = dist_matrix[a * stride + c] + dist_matrix[b * stride + d];

                    if after + TWO_OPT_EPSILON < before {
                        tour[i..=j].reverse();
                        improved = true;
                    }
                }
            }
            if !improved {
                break;
            }
        }
    }

    open_path_length_with_stride(tour, dist_matrix, stride)
}

/// Computes the total open-path length `sum_{m=0}^{K-2} dist(tour[m], tour[m+1])`.
pub fn open_tour_length(tour: &[u32], dist_matrix: &[f32]) -> f32 {
    let Some(stride) = matrix_stride_for_tour(dist_matrix, tour) else {
        return 0.0_f32;
    };
    open_path_length_with_stride(tour, dist_matrix, stride)
}

/// Solves centroid locality reordering (distance matrix + Nearest-Neighbor + 2-opt)
/// starting from `start_cluster`.
pub fn solve_tsp_locality(
    centroids_f16: &[f16],
    num_clusters: usize,
    padded_dim: usize,
    start_cluster: usize,
    max_2opt_passes: usize,
) -> OspreyResult<ClusterLocalityPermutation> {
    let dist_matrix = compute_centroid_distance_matrix(centroids_f16, num_clusters, padded_dim)?;
    if num_clusters == 0 {
        return ClusterLocalityPermutation::identity(0);
    }
    if start_cluster >= num_clusters {
        return Err(OspreyError::InvalidDimension { dim: start_cluster });
    }
    let mut tour = tsp_nearest_neighbor(&dist_matrix, num_clusters, start_cluster);
    let _ = tsp_two_opt_refine(&mut tour, &dist_matrix, max_2opt_passes);
    ClusterLocalityPermutation::from_tour(tour)
}

/// Solves centroid locality reordering starting from the highest-cardinality cluster
/// (breaking cardinality ties by lowest `cluster_id`), matching `.osv` build spec.
pub fn solve_tsp_locality_largest_first(
    centroids_f16: &[f16],
    num_clusters: usize,
    padded_dim: usize,
    cluster_counts: &[u64],
    max_2opt_passes: usize,
) -> OspreyResult<ClusterLocalityPermutation> {
    if cluster_counts.len() < num_clusters {
        return Err(OspreyError::BufferTooSmall {
            needed: num_clusters * size_of::<u64>(),
            actual: size_of_val(cluster_counts),
        });
    }
    if cluster_counts.len() != num_clusters {
        return Err(OspreyError::InvalidDimension {
            dim: cluster_counts.len(),
        });
    }
    let start_cluster = cluster_counts
        .iter()
        .enumerate()
        .fold((0usize, 0u64), |(best_idx, best_cnt), (idx, &cnt)| {
            if idx == 0 || cnt > best_cnt {
                (idx, cnt)
            } else {
                (best_idx, best_cnt)
            }
        })
        .0;

    solve_tsp_locality(
        centroids_f16,
        num_clusters,
        padded_dim,
        start_cluster,
        max_2opt_passes,
    )
}

fn validate_centroid_table_shape(
    centroids_f16: &[f16],
    num_clusters: usize,
    padded_dim: usize,
) -> OspreyResult<usize> {
    if padded_dim == 0 {
        return Err(OspreyError::InvalidDimension { dim: 0 });
    }
    let expected_len = num_clusters
        .checked_mul(padded_dim)
        .ok_or(OspreyError::InvalidDimension { dim: num_clusters })?;
    if centroids_f16.len() < expected_len {
        return Err(OspreyError::BufferTooSmall {
            needed: expected_len * size_of::<f16>(),
            actual: size_of_val(centroids_f16),
        });
    }
    if centroids_f16.len() != expected_len {
        return Err(OspreyError::InvalidDimension {
            dim: centroids_f16.len(),
        });
    }
    Ok(expected_len)
}

fn copy_reordered_centroids(
    physical_to_logical: &[u32],
    centroids_f16: &[f16],
    padded_dim: usize,
    out: &mut [f16],
) {
    for (physical_idx, &logical_id) in physical_to_logical.iter().enumerate() {
        let logical_idx = idx_from_u32(logical_id);
        let src = &centroids_f16[logical_idx * padded_dim..(logical_idx + 1) * padded_dim];
        let dst = &mut out[physical_idx * padded_dim..(physical_idx + 1) * padded_dim];
        dst.copy_from_slice(src);
    }
}

#[inline]
fn idx_from_u32(v: u32) -> usize {
    usize::try_from(v).unwrap_or(usize::MAX)
}

#[inline]
fn squared_l2_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc0 = 0.0_f32;
    let mut acc1 = 0.0_f32;
    let mut acc2 = 0.0_f32;
    let mut acc3 = 0.0_f32;
    let mut d = 0usize;

    while d + 4 <= n {
        let d0 = a[d] - b[d];
        let d1 = a[d + 1] - b[d + 1];
        let d2 = a[d + 2] - b[d + 2];
        let d3 = a[d + 3] - b[d + 3];
        acc0 += d0 * d0;
        acc1 += d1 * d1;
        acc2 += d2 * d2;
        acc3 += d3 * d3;
        d += 4;
    }

    let mut sum = (acc0 + acc1) + (acc2 + acc3);
    while d < n {
        let diff = a[d] - b[d];
        sum += diff * diff;
        d += 1;
    }
    sum
}

#[inline]
fn isqrt_exact(n: usize) -> Option<usize> {
    if n == 0 {
        return Some(0);
    }
    let mut lo = 1usize;
    let mut hi = n.min(1usize << (usize::BITS / 2));
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        let sq = mid.saturating_mul(mid);
        match sq.cmp(&n) {
            Ordering::Equal => return Some(mid),
            Ordering::Less => lo = mid + 1,
            Ordering::Greater => hi = mid - 1,
        }
    }
    None
}

fn matrix_stride_for_tour(dist_matrix: &[f32], tour: &[u32]) -> Option<usize> {
    if tour.len() <= 1 {
        return None;
    }
    let stride = isqrt_exact(dist_matrix.len())?;
    if stride == 0 {
        return None;
    }
    for &node in tour {
        if idx_from_u32(node) >= stride {
            return None;
        }
    }
    Some(stride)
}

fn open_path_length_with_stride(tour: &[u32], dist_matrix: &[f32], stride: usize) -> f32 {
    let mut total = 0.0_f32;
    for pair in tour.windows(2) {
        let u = idx_from_u32(pair[0]);
        let v = idx_from_u32(pair[1]);
        total += dist_matrix[u * stride + v];
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_f16_vec(vals: &[f32]) -> Vec<f16> {
        vals.iter().copied().map(f16::from_f32).collect()
    }

    fn sequential_reference_dist_matrix(
        centroids_f16: &[f16],
        num_clusters: usize,
        padded_dim: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0_f32; num_clusters * num_clusters];
        for i in 0..num_clusters {
            for j in 0..num_clusters {
                if i == j {
                    continue;
                }
                let mut sum = 0.0_f32;
                for d in 0..padded_dim {
                    let diff = centroids_f16[i * padded_dim + d].to_f32()
                        - centroids_f16[j * padded_dim + d].to_f32();
                    sum += diff * diff;
                }
                out[i * num_clusters + j] = sum;
            }
        }
        out
    }

    #[test]
    fn distance_matrix_validates_dimensions_and_finiteness() -> OspreyResult<()> {
        let zero_dim = compute_centroid_distance_matrix(&[], 0, 0);
        assert!(matches!(
            zero_dim,
            Err(OspreyError::InvalidDimension { dim: 0 })
        ));

        let too_small = compute_centroid_distance_matrix(&[f16::ONE; 3], 2, 2);
        assert!(matches!(too_small, Err(OspreyError::BufferTooSmall { .. })));

        let too_large = compute_centroid_distance_matrix(&[f16::ONE; 5], 2, 2);
        assert!(matches!(
            too_large,
            Err(OspreyError::InvalidDimension { dim: 5 })
        ));

        let non_finite =
            compute_centroid_distance_matrix(&[f16::ONE, f16::NAN, f16::ZERO, f16::ONE], 2, 2);
        assert!(matches!(
            non_finite,
            Err(OspreyError::NonFiniteEmbedding { index: 1 })
        ));

        let empty = compute_centroid_distance_matrix(&[], 0, 64)?;
        assert!(empty.is_empty());
        Ok(())
    }

    #[test]
    fn distance_matrix_is_symmetric_and_matches_exact_squared_l2() -> OspreyResult<()> {
        let raw = [
            1.0_f32, 2.0, -1.0, 0.5, 0.0, // c0
            4.0, 2.0, -1.0, 0.5, 0.0, // c1 -> diff 3^2 = 9
            1.0, 6.0, -1.0, 0.5, 4.0, // c2 -> diff 4^2 + 4^2 = 32
        ];
        let centroids = to_f16_vec(&raw);
        let dist = compute_centroid_distance_matrix(&centroids, 3, 5)?;
        assert_eq!(dist.len(), 9);

        for i in 0..3 {
            assert_eq!(dist[i * 3 + i], 0.0);
            for j in 0..3 {
                assert_eq!(dist[i * 3 + j].to_bits(), dist[j * 3 + i].to_bits());
            }
        }

        assert!((dist[1] - 9.0).abs() < 1e-6);
        assert!((dist[2] - 32.0).abs() < 1e-6);
        assert!((dist[3 + 2] - 41.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Exercises rayon threadpool / crossbeam-epoch
    fn distance_matrix_parallel_path_matches_sequential_reference() -> OspreyResult<()> {
        let num_clusters = 80usize;
        let padded_dim = 64usize;
        let mut raw = Vec::with_capacity(num_clusters * padded_dim);
        for c in 0..num_clusters {
            for d in 0..padded_dim {
                let val = (((c * 31 + d * 17) % 97) as f32 - 48.0) * 0.125;
                raw.push(val);
            }
        }
        let centroids = to_f16_vec(&raw);
        let par_dist = compute_centroid_distance_matrix(&centroids, num_clusters, padded_dim)?;
        let seq_dist = sequential_reference_dist_matrix(&centroids, num_clusters, padded_dim);

        assert_eq!(par_dist.len(), seq_dist.len());
        for (i, (&p, &s)) in par_dist.iter().zip(seq_dist.iter()).enumerate() {
            assert!(
                (p - s).abs() <= 1e-4,
                "mismatch at {i}: parallel={p}, sequential={s}"
            );
        }
        Ok(())
    }

    #[test]
    fn nearest_neighbor_breaks_ties_deterministically_by_lowest_index() {
        // Cluster 0 has equal distance 1.0 to clusters 1, 2, 3 -> must pick 1 first.
        let dist = vec![
            0.0, 1.0, 1.0, 1.0, // 0
            1.0, 0.0, 5.0, 2.0, // 1 -> closest unvisited is 3 (2.0 < 5.0)
            1.0, 5.0, 0.0, 4.0, // 2
            1.0, 2.0, 4.0, 0.0, // 3 -> remaining unvisited is 2
        ];
        assert_eq!(tsp_nearest_neighbor(&dist, 4, 0), vec![0, 1, 3, 2]);
        assert_eq!(tsp_nearest_neighbor(&dist, 4, 2), vec![2, 0, 1, 3]);
        assert!(tsp_nearest_neighbor(&dist, 0, 0).is_empty());
        assert!(tsp_nearest_neighbor(&dist, 4, 4).is_empty());
        assert!(tsp_nearest_neighbor(&dist[..10], 4, 0).is_empty());
    }

    #[test]
    fn two_opt_uncrosses_bow_tie_and_reduces_open_path_length() -> OspreyResult<()> {
        // Points on a 2D square:
        // 0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (0, 1)
        let centroids = to_f16_vec(&[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        let dist = compute_centroid_distance_matrix(&centroids, 4, 2)?;

        // Crossed open tour: 0 -> 2 -> 1 -> 3
        // Edges: d(0,2)=2 + d(2,1)=1 + d(1,3)=2 => total = 5.0
        // Swapping (0,2) and (1,3) reverses [2, 1] -> [0, 1, 2, 3]:
        // Edges: d(0,1)=1 + d(1,2)=1 + d(2,3)=1 => total = 3.0
        let mut tour = vec![0, 2, 1, 3];
        let initial_len = open_tour_length(&tour, &dist);
        assert!((initial_len - 5.0).abs() < 1e-6);

        let mut zero_pass_tour = tour.clone();
        let zero_pass_len = tsp_two_opt_refine(&mut zero_pass_tour, &dist, 0);
        assert_eq!(zero_pass_tour, vec![0, 2, 1, 3]);
        assert!((zero_pass_len - 5.0).abs() < 1e-6);

        let final_len = tsp_two_opt_refine(&mut tour, &dist, 50);
        assert_eq!(tour, vec![0, 1, 2, 3]);
        assert!((final_len - 3.0).abs() < 1e-6);
        assert!((final_len - open_tour_length(&tour, &dist)).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn two_opt_strictly_improves_scrambled_line_and_handles_small_k() -> OspreyResult<()> {
        // 12 points along the x-axis with scrambled cluster indices.
        let coords: Vec<f32> = vec![0.0, 8.0, 3.0, 11.0, 1.0, 6.0, 10.0, 4.0, 2.0, 9.0, 5.0, 7.0];
        let centroids = to_f16_vec(&coords);
        let k = coords.len();
        let dist = compute_centroid_distance_matrix(&centroids, k, 1)?;

        let mut tour: Vec<u32> = (0..12).collect();
        let before = open_tour_length(&tour, &dist);
        let after = tsp_two_opt_refine(&mut tour, &dist, 50);
        assert!(after < before);
        assert!((after - open_tour_length(&tour, &dist)).abs() < 1e-5);

        // Verify tour remains a valid permutation.
        let perm = ClusterLocalityPermutation::from_tour(tour)?;
        assert_eq!(perm.num_clusters(), k);

        // K <= 3 edge cases
        let small_dist = vec![0.0; 9];
        let mut t3 = vec![0, 2, 1];
        assert_eq!(tsp_two_opt_refine(&mut t3, &small_dist, 10), 0.0);
        let mut t1 = vec![0];
        assert_eq!(tsp_two_opt_refine(&mut t1, &[0.0], 10), 0.0);
        Ok(())
    }

    #[test]
    fn permutation_table_reorders_centroids_and_remaps_cluster_ids() -> OspreyResult<()> {
        // Logical clusters arranged out-of-order on a 1D line:
        // logical 0 at x=10, logical 1 at x=0, logical 2 at x=2, logical 3 at x=8
        let centroids = to_f16_vec(&[
            10.0, 0.0, // logical 0
            0.0, 0.0, // logical 1
            2.0, 0.0, // logical 2
            8.0, 0.0, // logical 3
        ]);
        let counts = [5_u64, 20, 10, 3]; // logical 1 has highest count (20) -> start at 1

        let perm = solve_tsp_locality_largest_first(&centroids, 4, 2, &counts, 50)?;
        // Starting at logical 1 (x=0): 1 (x=0) -> 2 (x=2) -> 3 (x=8) -> 0 (x=10)
        assert_eq!(perm.physical_to_logical, vec![1, 2, 3, 0]);
        assert_eq!(perm.logical_to_physical, vec![3, 0, 1, 2]);

        for p in 0..4u32 {
            let l = perm.to_logical(p)?;
            assert_eq!(perm.to_physical(l)?, p);
        }

        let reordered = perm.reorder_centroids_f16(&centroids, 2)?;
        let expected = to_f16_vec(&[0.0, 0.0, 2.0, 0.0, 8.0, 0.0, 10.0, 0.0]);
        assert_eq!(reordered, expected);

        let aligned = perm.reorder_centroids_f16_aligned(&centroids, 2)?;
        assert_eq!(aligned.as_ptr() as usize % 64, 0);
        assert_eq!(aligned.as_slice(), expected.as_slice());

        let logical_vec_clusters = vec![0, 1, 2, 3, 1, 0];
        let remapped = perm.remap_cluster_ids(&logical_vec_clusters)?;
        assert_eq!(remapped, vec![3, 0, 1, 2, 0, 3]);

        let mut in_place = logical_vec_clusters;
        perm.remap_cluster_ids_in_place(&mut in_place)?;
        assert_eq!(in_place, remapped);

        // Atomic failure on invalid cluster ID in remap_cluster_ids_in_place.
        let mut with_invalid = vec![0, 1, 99, 2];
        let err = perm.remap_cluster_ids_in_place(&mut with_invalid);
        assert!(matches!(
            err,
            Err(OspreyError::InvalidDimension { dim: 99 })
        ));
        assert_eq!(with_invalid, vec![0, 1, 99, 2]);

        let reordered_counts = perm.reorder_cluster_counts(&counts)?;
        assert_eq!(reordered_counts, vec![20, 10, 3, 5]);
        Ok(())
    }

    #[test]
    fn permutation_validation_rejects_duplicates_and_oob() {
        assert!(matches!(
            ClusterLocalityPermutation::from_tour(vec![0, 1, 1]),
            Err(OspreyError::InvalidDimension { dim: 1 })
        ));
        assert!(matches!(
            ClusterLocalityPermutation::from_tour(vec![0, 3, 1]),
            Err(OspreyError::InvalidDimension { dim: 3 })
        ));
    }
}
