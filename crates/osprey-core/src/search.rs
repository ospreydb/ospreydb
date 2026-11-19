//! Query-plan sizing shared by the CLI, benchmark harnesses, and external runners.
//!
//! Stage-1 `.osi` uses a 1-bit asymmetric `RaBitQ` estimate, which is noisier for
//! low-dimensional embeddings (e.g. `SIFT1M`, `d = 128`). To keep `Recall@10 >= 0.95`
//! without raising the latency budget for high-dimensional workloads, the exact-`f16`
//! re-score pool is oversampled only when the unpadded dimensionality is small.
//!
//! [`SearchOptions::resolve`] is the single source of truth for this policy. Callers must
//! pass the returned [`ResolvedSearchPlan`] verbatim to `OsiIndex::search_candidates`
//! (`effective_nprobe`, `stage1_limit`) and to the `f16` refinement stage (`refine_limit`).
//!
//! Callers without a tuned candidate budget can pass `candidates: 0`; the resolver then
//! selects a dimension-aware default ([`LOW_DIM_DEFAULT_CANDIDATES`] for `orig_dim <= 256`,
//! [`HIGH_DIM_DEFAULT_CANDIDATES`] otherwise).

/// Unpadded vector dimensionality at or below which adaptive oversampling engages.
pub const LOW_DIM_OVERSAMPLE_THRESHOLD: u16 = 256;

/// Oversampling multiplier applied to `nprobe`/`candidates` for low-dimensional data.
pub const DEFAULT_LOW_DIM_OVERSAMPLE_MULTIPLIER: usize = 2;

/// Minimum exact-`f16` re-score pool for low-dimensional data (`SIFT1M`-like recall).
pub const LOW_DIM_REFINE_POOL: usize = 128;

/// Default Stage-1 candidate pool when the caller requests auto-sizing on low-dimensional data.
///
/// SIFT1M measurements show that the 1-bit Stage-1 ranking needs a ~2000-candidate request
/// (`4000` after the adaptive `2x`) to recover `Recall@10 >= 0.95`.
pub const LOW_DIM_DEFAULT_CANDIDATES: usize = 2000;

/// Default Stage-1 candidate pool when the caller requests auto-sizing on high-dimensional data.
pub const HIGH_DIM_DEFAULT_CANDIDATES: usize = 300;

/// User-facing Stage-1 search knobs before adaptive resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchOptions {
    /// Requested number of TSP clusters to probe in `.osi`.
    pub nprobe: usize,
    /// Requested number of Stage-1 `RaBitQ` candidates.
    ///
    /// `0` selects the policy default: [`LOW_DIM_DEFAULT_CANDIDATES`] for `orig_dim <= 256`,
    /// [`HIGH_DIM_DEFAULT_CANDIDATES`] otherwise.
    pub candidates: usize,
    /// Final number of results to return.
    pub top_k: usize,
    /// Stage-3 survivors passed to `.osr` reranking (`0` = Stage-4 disabled).
    pub rerank_top_m: usize,
    /// `0` = adaptive (`2x` when `dim <= 256`), `1` = disabled, `N > 1` = forced `Nx`.
    pub refine_multiplier: usize,
}

impl SearchOptions {
    /// Creates options with Stage-4 reranking disabled.
    pub const fn new(nprobe: usize, candidates: usize, top_k: usize) -> Self {
        Self {
            nprobe,
            candidates,
            top_k,
            rerank_top_m: 0,
            refine_multiplier: 0,
        }
    }

    /// Resolves the adaptive oversampling policy for an unpadded `orig_dim`.
    ///
    /// `candidates == 0` selects the policy default ([`LOW_DIM_DEFAULT_CANDIDATES`] for
    /// low-dimensional data, [`HIGH_DIM_DEFAULT_CANDIDATES`] otherwise). The candidate-pool
    /// floor ([`LOW_DIM_REFINE_POOL`]) only applies while oversampling is active, so
    /// `refine_multiplier = 1` fully disables the low-dimensional adaptation.
    pub fn resolve(&self, orig_dim: u16) -> ResolvedSearchPlan {
        let low_dim = orig_dim <= LOW_DIM_OVERSAMPLE_THRESHOLD;
        let refine_multiplier = if self.refine_multiplier == 0 {
            if low_dim {
                DEFAULT_LOW_DIM_OVERSAMPLE_MULTIPLIER
            } else {
                1
            }
        } else {
            self.refine_multiplier.max(1)
        };

        let auto_candidates = self.candidates == 0;
        let base_candidates = if auto_candidates {
            if low_dim {
                LOW_DIM_DEFAULT_CANDIDATES
            } else {
                HIGH_DIM_DEFAULT_CANDIDATES
            }
        } else {
            self.candidates
        };

        let low_dim_oversample = low_dim && refine_multiplier > 1;
        let refine_floor = if low_dim_oversample {
            LOW_DIM_REFINE_POOL
        } else {
            0
        };
        let base_refine_limit = if self.rerank_top_m > 0 {
            self.rerank_top_m.max(self.top_k)
        } else {
            self.top_k
        };
        let refine_limit = base_refine_limit.max(refine_floor);
        let effective_candidates = base_candidates.saturating_mul(refine_multiplier);
        let stage1_limit = effective_candidates.max(refine_limit);
        let effective_nprobe = self.nprobe.saturating_mul(refine_multiplier);

        ResolvedSearchPlan {
            refine_multiplier,
            effective_nprobe,
            effective_candidates,
            refine_limit,
            stage1_limit,
            low_dim_oversample,
            auto_candidates,
        }
    }
}

/// Fully resolved Stage-1/Stage-3 sizing consumed by the search pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedSearchPlan {
    /// Resolved oversampling multiplier (`0` was never resolved).
    pub refine_multiplier: usize,
    /// Clusters to probe in `.osi`.
    pub effective_nprobe: usize,
    /// Stage-1 `RaBitQ` candidates after oversampling.
    pub effective_candidates: usize,
    /// Candidates scored exactly in `f16` before float ranking/reranking.
    pub refine_limit: usize,
    /// Upper bound passed to `OsiIndex::search_candidates` (`>= refine_limit`).
    pub stage1_limit: usize,
    /// Whether the low-dimensional adaptive rule engaged.
    pub low_dim_oversample: bool,
    /// Whether [`SearchOptions::candidates`] was `0` and resolved to the policy default.
    pub auto_candidates: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_dim_adaptive_doubles_pool_and_applies_floor() {
        let plan = SearchOptions::new(16, 200, 10).resolve(128);
        assert_eq!(plan.refine_multiplier, 2);
        assert_eq!(plan.effective_nprobe, 32);
        assert_eq!(plan.effective_candidates, 400);
        assert_eq!(plan.refine_limit, LOW_DIM_REFINE_POOL);
        assert_eq!(plan.stage1_limit, 400);
        assert!(plan.low_dim_oversample);
    }

    #[test]
    fn low_dim_floor_lifts_small_explicit_pools() {
        let plan = SearchOptions::new(8, 64, 10).resolve(128);
        assert_eq!(plan.effective_candidates, 128);
        assert_eq!(plan.refine_limit, LOW_DIM_REFINE_POOL);
        assert_eq!(plan.stage1_limit, LOW_DIM_REFINE_POOL);
    }

    #[test]
    fn threshold_boundary_is_inclusive() {
        assert!(
            SearchOptions::new(16, 200, 10)
                .resolve(256)
                .low_dim_oversample
        );
        assert!(
            !SearchOptions::new(16, 200, 10)
                .resolve(257)
                .low_dim_oversample
        );
    }

    #[test]
    fn explicit_one_disables_adaptive_and_floor() {
        let options = SearchOptions {
            refine_multiplier: 1,
            ..SearchOptions::new(16, 50, 10)
        };
        let plan = options.resolve(128);
        assert_eq!(plan.refine_multiplier, 1);
        assert_eq!(plan.effective_nprobe, 16);
        assert_eq!(plan.effective_candidates, 50);
        assert_eq!(plan.refine_limit, 10);
        assert_eq!(plan.stage1_limit, 50);
        assert!(!plan.low_dim_oversample);
    }

    #[test]
    fn forced_multiplier_still_applies_floor() {
        let options = SearchOptions {
            refine_multiplier: 3,
            ..SearchOptions::new(4, 64, 10)
        };
        let plan = options.resolve(128);
        assert_eq!(plan.refine_multiplier, 3);
        assert_eq!(plan.effective_nprobe, 12);
        assert_eq!(plan.effective_candidates, 192);
        assert_eq!(plan.refine_limit, LOW_DIM_REFINE_POOL);
        assert_eq!(plan.stage1_limit, 192);
        assert!(plan.low_dim_oversample);
    }

    #[test]
    fn high_dim_is_untouched() {
        let options = SearchOptions {
            rerank_top_m: 200,
            ..SearchOptions::new(16, 200, 10)
        };
        let plan = options.resolve(1536);
        assert_eq!(plan.refine_multiplier, 1);
        assert_eq!(plan.effective_nprobe, 16);
        assert_eq!(plan.effective_candidates, 200);
        assert_eq!(plan.refine_limit, 200);
        assert_eq!(plan.stage1_limit, 200);
        assert!(!plan.low_dim_oversample);
        assert!(!plan.auto_candidates);
    }

    #[test]
    fn auto_low_dim_candidates_use_recall_default() {
        let plan = SearchOptions::new(32, 0, 10).resolve(128);
        assert!(plan.auto_candidates);
        assert_eq!(plan.refine_multiplier, 2);
        assert_eq!(plan.effective_nprobe, 64);
        assert_eq!(plan.effective_candidates, 4000);
        assert_eq!(plan.refine_limit, LOW_DIM_REFINE_POOL);
        assert_eq!(plan.stage1_limit, 4000);
        assert!(plan.low_dim_oversample);
    }

    #[test]
    fn auto_high_dim_candidates_use_default() {
        let plan = SearchOptions::new(32, 0, 10).resolve(768);
        assert!(plan.auto_candidates);
        assert_eq!(plan.refine_multiplier, 1);
        assert_eq!(plan.effective_nprobe, 32);
        assert_eq!(plan.effective_candidates, HIGH_DIM_DEFAULT_CANDIDATES);
        assert_eq!(plan.stage1_limit, HIGH_DIM_DEFAULT_CANDIDATES);
        assert!(!plan.low_dim_oversample);
    }

    #[test]
    fn explicit_candidates_are_never_auto_scaled() {
        let options = SearchOptions {
            refine_multiplier: 1,
            ..SearchOptions::new(8, 64, 10)
        };
        let plan = options.resolve(128);
        assert!(!plan.auto_candidates);
        assert_eq!(plan.effective_candidates, 64);
        assert_eq!(plan.stage1_limit, 64);
    }

    #[test]
    fn stage1_limit_never_below_refine_limit() {
        let options = SearchOptions {
            rerank_top_m: 300,
            ..SearchOptions::new(8, 64, 10)
        };
        let plan = options.resolve(128);
        assert_eq!(plan.refine_limit, 300);
        assert_eq!(plan.stage1_limit, 300);
    }
}
