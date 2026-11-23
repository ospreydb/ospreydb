//! Stack-allocated fixed-capacity `BoundedTopKHeap` for zero-allocation Stage-1 scanning.

use std::cmp::Ordering;

use crate::error::{OspreyError, OspreyResult};

/// Candidate record produced by Stage-1 cluster probing or `.osi` `RaBitQ` scanning.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ScoredCandidate {
    /// Estimated or exact similarity score (higher is better).
    pub score: f32,
    /// Global logical document identifier.
    pub doc_id: u64,
    /// Physical entry index inside the `.osi` cluster-grouped arrays (or `cluster_id` during probing).
    pub entry_idx: u32,
    /// Target `.osv` minipage identifier.
    pub minipage_id: u32,
    /// Vector slot index inside the target minipage.
    pub slot_idx: u16,
}

/// Fixed-capacity Min-Heap allocated entirely on the stack (`[ScoredCandidate; MAX_CAP]`).
///
/// Maintains the top `k_limit` candidates by highest `score` (with deterministic tie-breaking
/// preferring smaller `doc_id`), achieving O(1) rejection for candidates below the current
/// kth-best threshold without any heap (`Vec`/`Box`) allocations.
#[derive(Clone, Debug)]
pub struct BoundedTopKHeap<const MAX_CAP: usize = 512> {
    storage: [ScoredCandidate; MAX_CAP],
    len: usize,
    k_limit: usize,
}

impl<const MAX_CAP: usize> BoundedTopKHeap<MAX_CAP> {
    /// Creates a new stack-allocated `BoundedTopKHeap` retaining at most `k_limit` best items.
    ///
    /// Clamps `k_limit` to `MAX_CAP` without panicking.
    #[inline]
    pub fn new(k_limit: usize) -> Self {
        Self {
            storage: [ScoredCandidate::default(); MAX_CAP],
            len: 0,
            k_limit: k_limit.min(MAX_CAP),
        }
    }

    /// Fallible constructor returning [`OspreyError::InvalidDimension`] if `k_limit > MAX_CAP`.
    #[inline]
    pub fn try_new(k_limit: usize) -> OspreyResult<Self> {
        if k_limit > MAX_CAP {
            return Err(OspreyError::InvalidDimension { dim: k_limit });
        }
        Ok(Self::new(k_limit))
    }

    /// Returns the number of candidates currently stored in the heap.
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the heap contains zero candidates.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the configured maximum number of retained candidates (`<= MAX_CAP`).
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.k_limit
    }

    /// Returns the current minimum (eviction-threshold) score when the heap is full.
    #[inline]
    pub fn threshold_score(&self) -> Option<f32> {
        (self.len == self.k_limit && self.len > 0).then_some(self.storage[0].score)
    }

    /// Resets the heap length to zero without zeroing or reallocating backing storage.
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Evaluates `candidate` against the bounded min-heap in O(1) fast-reject / O(log K) insert.
    ///
    /// Non-finite (`NaN`) scores are silently ignored to protect heap ordering invariants.
    #[inline(always)]
    pub fn push(&mut self, candidate: ScoredCandidate) {
        if self.k_limit == 0 || candidate.score.is_nan() {
            return;
        }

        if self.len < self.k_limit {
            let idx = self.len;
            self.storage[idx] = candidate;
            self.len += 1;
            self.sift_up(idx);
            return;
        }

        // Fast O(1) rejection path (~99% of candidates after warmup).
        let root = self.storage[0];
        if candidate.score < root.score
            || (candidate.score == root.score && candidate.doc_id >= root.doc_id)
        {
            return;
        }

        self.storage[0] = candidate;
        self.sift_down(0);
    }

    /// Sorts the retained candidates in-place by descending `score` (ties broken by ascending
    /// `doc_id`) and returns the sorted slice.
    #[inline]
    pub fn into_sorted_slice(&mut self) -> &[ScoredCandidate] {
        self.storage[..self.len].sort_unstable_by(|a, b| match b.score.total_cmp(&a.score) {
            Ordering::Equal => a.doc_id.cmp(&b.doc_id),
            ord => ord,
        });
        &self.storage[..self.len]
    }

    /// Returns `true` iff `a` is strictly *worse* than `b` (i.e., `a` sits higher in the min-heap).
    #[inline(always)]
    fn is_worse(a: &ScoredCandidate, b: &ScoredCandidate) -> bool {
        a.score < b.score || (a.score == b.score && a.doc_id > b.doc_id)
    }

    #[inline(always)]
    fn sift_up(&mut self, mut idx: usize) {
        let item = self.storage[idx];
        while idx > 0 {
            let parent = (idx - 1) >> 1;
            if Self::is_worse(&item, &self.storage[parent]) {
                self.storage[idx] = self.storage[parent];
                idx = parent;
            } else {
                break;
            }
        }
        self.storage[idx] = item;
    }

    #[inline(always)]
    fn sift_down(&mut self, mut idx: usize) {
        let item = self.storage[idx];
        let half = self.len >> 1;
        while idx < half {
            let left = (idx << 1) + 1;
            let right = left + 1;
            let best_child =
                if right < self.len && Self::is_worse(&self.storage[right], &self.storage[left]) {
                    right
                } else {
                    left
                };

            if Self::is_worse(&self.storage[best_child], &item) {
                self.storage[idx] = self.storage[best_child];
                idx = best_child;
            } else {
                break;
            }
        }
        self.storage[idx] = item;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::math::SplitMix64;

    #[test]
    fn bounded_topk_heap_matches_full_sort_with_tie_breaking_and_nan_safety() {
        let mut rng = SplitMix64::new(0x4F53_5052_0091);
        let mut all = Vec::with_capacity(2000);
        let mut heap = BoundedTopKHeap::<512>::new(37);

        for i in 0..2000u64 {
            // Quantize scores coarsely to force frequent exact score ties.
            #[allow(clippy::cast_precision_loss)]
            let bucket = ((rng.next_u64() % 200) as f32) * 0.05 - 5.0;
            let cand = ScoredCandidate {
                score: bucket,
                doc_id: i ^ 0x555,
                entry_idx: u32::try_from(i).unwrap(),
                minipage_id: u32::try_from(i % 16).unwrap(),
                slot_idx: u16::try_from(i % 64).unwrap(),
            };
            all.push(cand);
            heap.push(cand);
        }

        // NaNs must be ignored cleanly
        heap.push(ScoredCandidate {
            score: f32::NAN,
            doc_id: 999_999,
            ..Default::default()
        });

        all.sort_unstable_by(|a, b| match b.score.total_cmp(&a.score) {
            Ordering::Equal => a.doc_id.cmp(&b.doc_id),
            ord => ord,
        });

        let got = heap.into_sorted_slice();
        assert_eq!(got.len(), 37);
        assert_eq!(got, &all[..37]);
    }

    #[test]
    fn bounded_topk_heap_handles_zero_and_oversized_k() {
        let mut zero_heap = BoundedTopKHeap::<16>::new(0);
        zero_heap.push(ScoredCandidate {
            score: 10.0,
            doc_id: 1,
            ..Default::default()
        });
        assert!(zero_heap.is_empty());
        assert_eq!(zero_heap.into_sorted_slice().len(), 0);

        assert!(BoundedTopKHeap::<16>::try_new(17).is_err());
        let clamped = BoundedTopKHeap::<16>::new(32);
        assert_eq!(clamped.capacity(), 16);
    }
}
