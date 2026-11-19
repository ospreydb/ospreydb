//! Zero-allocation Dynamic Programming range planner for coalescing object-storage `Range-GET`s.
//!
//! Converts Stage-1 surviving `Top-M` candidate page references into sorted, deduplicated
//! byte intervals and solves the exact Bellman coalescing recurrence in $O(k)$ worst-case
//! time using only fixed-size stack arrays (zero heap allocations).

use osprey_core::{MinipageDirectoryEntry, OspreyError, OspreyResult};

/// Default fixed request overhead cost (`256 KiB` equivalent bandwidth) per HTTP `Range-GET`.
pub const DEFAULT_COALESCE_TAU_REQ: f64 = 262_144.0;

/// Default marginal cost per byte transferred (`1.0`).
pub const DEFAULT_COALESCE_TAU_BYTE: f64 = 1.0;

/// Default maximum byte span for a single coalesced `Range-GET` request (`8 MiB`).
pub const DEFAULT_MAX_COALESCE_BYTES: u64 = 8 * 1024 * 1024;

/// Default stack capacity `MAX_K` for fixed-size DP range planner buffers.
pub const DEFAULT_COALESCE_MAX_K: usize = 512;

/// Cost model configuration for the DP range coalescer.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RangePlannerConfig {
    /// Fixed latency/request overhead cost per isolated HTTP `Range-GET` request.
    pub tau_req: f64,
    /// Marginal cost per byte transferred over object storage.
    pub tau_byte: f64,
    /// Hard upper bound in bytes on any single coalesced `Range-GET` window.
    pub max_coalesce_bytes: u64,
}

impl Default for RangePlannerConfig {
    #[inline]
    fn default() -> Self {
        Self {
            tau_req: DEFAULT_COALESCE_TAU_REQ,
            tau_byte: DEFAULT_COALESCE_TAU_BYTE,
            max_coalesce_bytes: DEFAULT_MAX_COALESCE_BYTES,
        }
    }
}

/// Stage-1 candidate reference pointing into an `.osv` minipage and vector slot.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CandidatePageRef {
    /// Global logical document identifier.
    pub doc_id: u64,
    /// Zero-based minipage index in the `.osv` `MinipageDirectoryEntry` table.
    pub minipage_id: u32,
    /// Zero-based vector slot index within the target minipage.
    pub slot_idx: u16,
    /// IEEE-754 `f32::to_bits()` encoding of the Stage-1 approximate score.
    pub approx_score_bits: u32,
}

impl From<osprey_core::ScoredCandidate> for CandidatePageRef {
    #[inline]
    fn from(candidate: osprey_core::ScoredCandidate) -> Self {
        Self {
            doc_id: candidate.doc_id,
            minipage_id: candidate.minipage_id,
            slot_idx: candidate.slot_idx,
            approx_score_bits: candidate.score.to_bits(),
        }
    }
}

impl From<&osprey_core::ScoredCandidate> for CandidatePageRef {
    #[inline]
    fn from(candidate: &osprey_core::ScoredCandidate) -> Self {
        Self::from(*candidate)
    }
}

/// Half-open byte interval `[start, end)` of a unique compressed `.osv` minipage.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct PageByteInterval {
    /// Inclusive starting byte offset within the `.osv` object.
    pub start: u64,
    /// Exclusive ending byte offset within the `.osv` object.
    pub end: u64,
    /// Index of the referenced minipage in the `.osv` directory.
    pub minipage_id: u32,
}

impl PageByteInterval {
    /// Returns the compressed byte length (`end.saturating_sub(start)`).
    #[inline]
    pub const fn byte_len(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }
}

/// Coalesced byte range ready for batched `object_store::get_ranges` execution.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CoalescedRange {
    /// Inclusive starting byte offset of the merged `Range-GET` request.
    pub start: u64,
    /// Exclusive ending byte offset of the merged `Range-GET` request.
    pub end: u64,
    /// Starting index in the sorted unique [`PageByteInterval`] slice covered by this request.
    pub first_interval_idx: u16,
    /// Number of consecutive intervals in the sorted unique slice covered by this request.
    pub interval_count: u16,
}

impl CoalescedRange {
    /// Returns the total byte span (`end.saturating_sub(start)`) of this coalesced request.
    #[inline]
    pub const fn byte_len(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }
}

/// Translates Stage-1 `Top-M` candidates into sorted, deduplicated `.osv` byte intervals on stack.
///
/// Validates `candidate.minipage_id < directory.len()`, extracts `[byte_offset, byte_offset + compressed_size)`,
/// sorts `&mut out_intervals[..candidates.len()]` in-place by `(start, minipage_id)`, and deduplicates
/// repeated minipages in-place, returning the unique interval count `k`.
pub fn extract_unique_page_intervals<const MAX_M: usize>(
    candidates: &[CandidatePageRef],
    directory: &[MinipageDirectoryEntry],
    out_intervals: &mut [PageByteInterval; MAX_M],
) -> OspreyResult<usize> {
    if candidates.len() > MAX_M {
        return Err(OspreyError::BufferTooSmall {
            needed: candidates.len(),
            actual: MAX_M,
        });
    }
    if candidates.is_empty() {
        return Ok(0);
    }

    let len = candidates.len();
    for (idx, candidate) in candidates.iter().enumerate() {
        let page_idx = usize::try_from(candidate.minipage_id)
            .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
        let entry = directory
            .get(page_idx)
            .ok_or(OspreyError::InvalidDimension { dim: page_idx })?;

        let start = entry.byte_offset;
        let end = start
            .checked_add(u64::from(entry.compressed_size))
            .ok_or(OspreyError::CorruptedFooter)?;

        out_intervals[idx] = PageByteInterval {
            start,
            end,
            minipage_id: candidate.minipage_id,
        };
    }

    out_intervals[..len].sort_unstable_by_key(|interval| (interval.start, interval.minipage_id));

    let mut write_idx = 1usize;
    for read_idx in 1..len {
        if out_intervals[read_idx].minipage_id != out_intervals[write_idx - 1].minipage_id {
            out_intervals[write_idx] = out_intervals[read_idx];
            write_idx += 1;
        }
    }

    Ok(write_idx)
}

/// Solves the DP range planner Bellman DP coalescing recurrence in $O(k)$ time using 100% stack memory.
///
/// Uses fixed-size stack arrays (`dp_cost: [f64; MAX_K]`, `dp_prev: [u16; MAX_K]`, and a
/// monotonic candidate queue `[u16; MAX_K]`) with zero heap allocations, applying both:
/// - **Exact gap pruning:** if `tau_byte * (a_m - b_{m-1}) >= tau_req`, no optimal window crosses
///   the gap between `m - 1` and `m`, resetting the candidate queue in $O(1)$.
/// - **Sliding window bound:** candidate `j` is evicted from the front once
///   `intervals[i].end - intervals[j].start > config.max_coalesce_bytes`.
///
/// Backtracks through `dp_prev` and fills `out_ranges[..num_coalesced]` in ascending byte offset order.
pub fn plan_coalesced_ranges_stack<const MAX_K: usize>(
    intervals: &[PageByteInterval],
    k: usize,
    config: &RangePlannerConfig,
    out_ranges: &mut [CoalescedRange; MAX_K],
) -> usize {
    let n = k.min(intervals.len()).min(MAX_K).min(usize::from(u16::MAX));
    if n == 0 {
        return 0;
    }

    let tau_req = if config.tau_req.is_nan() || config.tau_req < 0.0 {
        0.0
    } else {
        config.tau_req
    };
    let tau_byte = if config.tau_byte.is_finite() && config.tau_byte > 0.0 {
        config.tau_byte
    } else {
        0.0
    };

    let mut dp_cost = [0.0_f64; MAX_K];
    let mut dp_prev = [0_u16; MAX_K];
    // Monotonic queue of candidate start indices `j` ordered strictly by increasing cost
    // for all future endpoints `i >= current`. Since each index `0..n` is pushed at most once,
    // `dq_tail` never exceeds `n <= MAX_K`.
    let mut mono_deque = [0_u16; MAX_K];
    let mut dq_head = 0usize;
    let mut dq_tail = 0usize;

    for i in 0..n {
        if i > 0 {
            let gap = intervals[i].start.saturating_sub(intervals[i - 1].end);
            if tau_byte * (gap as f64) >= tau_req {
                // Exact Bellman cut (Theorem 4): no optimal window can cross this gap.
                dq_head = dq_tail;
            }
        }

        let i_u16 = u16::try_from(i).unwrap_or(u16::MAX);
        let c_prev_i = if i == 0 { 0.0 } else { dp_cost[i - 1] };

        // Maintain monotonicity: pop back candidates `j1` dominated by `j2 = i`
        // (i.e., `C(j2 - 1) <= C(j1 - 1) + tau_byte * (a_{j2} - a_{j1})`).
        // Computing `a_{j2} - a_{j1}` in `u64` avoids precision loss at large file offsets.
        while dq_tail > dq_head {
            let back_idx = usize::from(mono_deque[dq_tail - 1]);
            let c_prev_back = if back_idx == 0 {
                0.0
            } else {
                dp_cost[back_idx - 1]
            };
            let start_delta = intervals[i].start.saturating_sub(intervals[back_idx].start);
            let back_adjusted = c_prev_back + tau_byte * (start_delta as f64);
            if c_prev_i <= back_adjusted {
                dq_tail -= 1;
            } else {
                break;
            }
        }

        mono_deque[dq_tail] = i_u16;
        dq_tail += 1;

        // Evict front candidates that exceed `max_coalesce_bytes` when merged up to `intervals[i].end`.
        let end_i = intervals[i].end;
        while dq_head < dq_tail {
            let front_idx = usize::from(mono_deque[dq_head]);
            let span = end_i.saturating_sub(intervals[front_idx].start);
            if span > config.max_coalesce_bytes {
                dq_head += 1;
            } else {
                break;
            }
        }

        // Fallback to singleton `[intervals[i].start, intervals[i].end]` if a single oversized
        // minipage itself exceeds `max_coalesce_bytes`.
        let best_j = if dq_head < dq_tail {
            usize::from(mono_deque[dq_head])
        } else {
            i
        };

        let c_prev_best = if best_j == 0 {
            0.0
        } else {
            dp_cost[best_j - 1]
        };
        let span = end_i.saturating_sub(intervals[best_j].start);
        dp_cost[i] = c_prev_best + tau_req + tau_byte * (span as f64);
        dp_prev[i] = u16::try_from(best_j).unwrap_or(u16::MAX);
    }

    // Backtrack from `n - 1` down to `0` and reverse to emit ascending byte offsets.
    let mut num_coalesced = 0usize;
    let mut cur_end_exclusive = n;
    while cur_end_exclusive > 0 {
        let end_idx = cur_end_exclusive - 1;
        let start_idx = usize::from(dp_prev[end_idx]).min(end_idx);

        out_ranges[num_coalesced] = CoalescedRange {
            start: intervals[start_idx].start,
            end: intervals[end_idx].end,
            first_interval_idx: u16::try_from(start_idx).unwrap_or(u16::MAX),
            interval_count: u16::try_from(cur_end_exclusive - start_idx).unwrap_or(u16::MAX),
        };
        num_coalesced += 1;
        cur_end_exclusive = start_idx;
    }

    out_ranges[..num_coalesced].reverse();
    num_coalesced
}

#[cfg(test)]
mod tests {
    use std::hint::black_box;
    #[cfg(not(miri))]
    use std::time::Duration;
    use std::time::Instant;

    use super::*;

    fn make_dir_entry(byte_offset: u64, compressed_size: u32) -> MinipageDirectoryEntry {
        MinipageDirectoryEntry {
            byte_offset,
            min_doc_id: 0,
            max_doc_id: 0,
            page_xxh64: 0,
            compressed_size,
            uncompressed_size: compressed_size.saturating_mul(2),
            cluster_id: 0,
            vector_count: 1,
            flags: 0,
        }
    }

    /// Reference full $O(k^2)$ unpruned Bellman DP without gap cutting or sliding-window queue.
    fn reference_unpruned_bellman_dp(
        intervals: &[PageByteInterval],
        k: usize,
        config: &RangePlannerConfig,
    ) -> (Vec<CoalescedRange>, f64) {
        if k == 0 {
            return (Vec::new(), 0.0);
        }
        let mut dp_cost = vec![f64::INFINITY; k];
        let mut dp_prev = vec![0usize; k];

        for i in 0..k {
            let mut best_cost = f64::INFINITY;
            let mut best_j = i;
            let end_i = intervals[i].end;

            for j in (0..=i).rev() {
                let span = end_i.saturating_sub(intervals[j].start);
                if span > config.max_coalesce_bytes && j < i {
                    continue;
                }
                if span > config.max_coalesce_bytes && j == i && best_cost.is_finite() {
                    continue;
                }
                let prev = if j == 0 { 0.0 } else { dp_cost[j - 1] };
                let candidate_cost = prev + config.tau_req + config.tau_byte * (span as f64);
                if candidate_cost < best_cost {
                    best_cost = candidate_cost;
                    best_j = j;
                }
            }

            dp_cost[i] = best_cost;
            dp_prev[i] = best_j;
        }

        let mut ranges = Vec::new();
        let mut cur_end = k;
        while cur_end > 0 {
            let end_idx = cur_end - 1;
            let start_idx = dp_prev[end_idx];
            ranges.push(CoalescedRange {
                start: intervals[start_idx].start,
                end: intervals[end_idx].end,
                first_interval_idx: u16::try_from(start_idx).unwrap_or(u16::MAX),
                interval_count: u16::try_from(cur_end - start_idx).unwrap_or(u16::MAX),
            });
            cur_end = start_idx;
        }
        ranges.reverse();
        (ranges, dp_cost[k - 1])
    }

    fn plan_cost(ranges: &[CoalescedRange], config: &RangePlannerConfig) -> f64 {
        ranges
            .iter()
            .map(|r| config.tau_req + config.tau_byte * (r.byte_len() as f64))
            .sum()
    }

    #[test]
    fn range_planner_config_defaults_match_specification() {
        let cfg = RangePlannerConfig::default();
        assert_eq!(cfg.tau_req, 262_144.0);
        assert_eq!(cfg.tau_byte, 1.0);
        assert_eq!(cfg.max_coalesce_bytes, 8 * 1024 * 1024);
    }

    #[test]
    fn extract_unique_page_intervals_sorts_dedups_and_validates() -> OspreyResult<()> {
        let directory = [
            make_dir_entry(64, 4_000),      // page 0: [64, 4064)
            make_dir_entry(4_064, 6_000),   // page 1: [4064, 10064)
            make_dir_entry(10_064, 5_000),  // page 2: [10064, 15064)
            make_dir_entry(300_000, 7_500), // page 3: [300000, 307500)
        ];

        let mut out = [PageByteInterval::default(); 16];

        // Empty candidates
        assert_eq!(extract_unique_page_intervals(&[], &directory, &mut out)?, 0);

        // Multiple Top-M candidates falling onto the same minipages, unsorted
        let candidates = [
            CandidatePageRef {
                doc_id: 10,
                minipage_id: 3,
                slot_idx: 2,
                approx_score_bits: 0x3f80_0000,
            },
            CandidatePageRef {
                doc_id: 11,
                minipage_id: 1,
                slot_idx: 0,
                approx_score_bits: 0x3f40_0000,
            },
            CandidatePageRef {
                doc_id: 12,
                minipage_id: 3,
                slot_idx: 5,
                approx_score_bits: 0x3f20_0000,
            },
            CandidatePageRef {
                doc_id: 13,
                minipage_id: 0,
                slot_idx: 1,
                approx_score_bits: 0x3f00_0000,
            },
            CandidatePageRef {
                doc_id: 14,
                minipage_id: 1,
                slot_idx: 3,
                approx_score_bits: 0x3e80_0000,
            },
        ];

        let k = extract_unique_page_intervals(&candidates, &directory, &mut out)?;
        assert_eq!(k, 3);
        assert_eq!(
            &out[..k],
            &[
                PageByteInterval {
                    start: 64,
                    end: 4_064,
                    minipage_id: 0,
                },
                PageByteInterval {
                    start: 4_064,
                    end: 10_064,
                    minipage_id: 1,
                },
                PageByteInterval {
                    start: 300_000,
                    end: 307_500,
                    minipage_id: 3,
                },
            ]
        );

        // Out-of-bounds minipage_id must return error
        let bad_candidate = [CandidatePageRef {
            doc_id: 99,
            minipage_id: 4,
            slot_idx: 0,
            approx_score_bits: 0,
        }];
        assert!(matches!(
            extract_unique_page_intervals(&bad_candidate, &directory, &mut out),
            Err(OspreyError::InvalidDimension { dim: 4 })
        ));

        // Exceeding MAX_M must return BufferTooSmall
        let mut small_out = [PageByteInterval::default(); 2];
        assert!(matches!(
            extract_unique_page_intervals(&candidates, &directory, &mut small_out),
            Err(OspreyError::BufferTooSmall {
                needed: 5,
                actual: 2
            })
        ));

        // Byte offset overflow must return CorruptedFooter
        let overflow_dir = [make_dir_entry(u64::MAX - 10, 64)];
        let overflow_cand = [CandidatePageRef {
            doc_id: 1,
            minipage_id: 0,
            slot_idx: 0,
            approx_score_bits: 0,
        }];
        assert!(matches!(
            extract_unique_page_intervals(&overflow_cand, &overflow_dir, &mut out),
            Err(OspreyError::CorruptedFooter)
        ));

        Ok(())
    }

    #[test]
    fn plan_coalesced_ranges_handles_edge_cases_gaps_and_max_coalesce_cap() {
        let cfg = RangePlannerConfig::default();
        let mut out = [CoalescedRange::default(); 32];

        // Empty
        assert_eq!(plan_coalesced_ranges_stack(&[], 0, &cfg, &mut out), 0);

        // Gap exactly equal to tau_req / tau_byte (262_144) triggers cut; smaller gap merges.
        let intervals = [
            PageByteInterval {
                start: 0,
                end: 10_000,
                minipage_id: 0,
            },
            PageByteInterval {
                start: 110_000, // gap = 100_000 < 262_144 -> merge
                end: 120_000,
                minipage_id: 1,
            },
            PageByteInterval {
                start: 120_000 + 262_144, // gap == 262_144 -> split
                end: 120_000 + 262_144 + 8_000,
                minipage_id: 2,
            },
        ];
        let count = plan_coalesced_ranges_stack(&intervals, intervals.len(), &cfg, &mut out);
        assert_eq!(count, 2);
        assert_eq!(
            &out[..count],
            &[
                CoalescedRange {
                    start: 0,
                    end: 120_000,
                    first_interval_idx: 0,
                    interval_count: 2,
                },
                CoalescedRange {
                    start: 382_144,
                    end: 390_144,
                    first_interval_idx: 2,
                    interval_count: 1,
                },
            ]
        );

        // Oversized single minipage exceeding max_coalesce_bytes must still emit singleton range
        let tight_cfg = RangePlannerConfig {
            tau_req: 1_000.0,
            tau_byte: 1.0,
            max_coalesce_bytes: 500,
        };
        let oversized = [
            PageByteInterval {
                start: 0,
                end: 200,
                minipage_id: 0,
            },
            PageByteInterval {
                start: 200,
                end: 1_000, // span 800 > 500
                minipage_id: 1,
            },
            PageByteInterval {
                start: 1_000,
                end: 1_200,
                minipage_id: 2,
            },
        ];
        let c_over = plan_coalesced_ranges_stack(&oversized, oversized.len(), &tight_cfg, &mut out);
        assert_eq!(c_over, 3);
        assert_eq!(out[1].start, 200);
        assert_eq!(out[1].end, 1_000);
        assert_eq!(out[1].interval_count, 1);
    }

    #[test]
    fn plan_coalesced_ranges_matches_unpruned_dp_across_varied_patterns() {
        let cfg = RangePlannerConfig {
            tau_req: 262_144.0,
            tau_byte: 1.0,
            max_coalesce_bytes: 64 * 1024,
        };

        // Pseudo-random deterministic gaps crossing both `max_coalesce_bytes` and `tau_req`.
        let mut intervals = [PageByteInterval::default(); 128];
        let mut offset = 64_u64;
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        for (idx, slot) in intervals.iter_mut().enumerate() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let gap = match seed % 7 {
                0 | 1 => 0,
                2 | 3 => (seed % 16_384) + 128,
                4 => 150_000,
                5 => 262_144, // exact tie boundary
                _ => 400_000,
            };
            offset += gap;
            let size = 4_096 + (seed % 8_192);
            *slot = PageByteInterval {
                start: offset,
                end: offset + size,
                minipage_id: u32::try_from(idx).unwrap_or(u32::MAX),
            };
            offset += size;
        }

        let mut out = [CoalescedRange::default(); 128];
        let k = intervals.len();
        let num_coalesced = plan_coalesced_ranges_stack(&intervals, k, &cfg, &mut out);
        let (ref_ranges, ref_cost) = reference_unpruned_bellman_dp(&intervals, k, &cfg);

        assert_eq!(&out[..num_coalesced], ref_ranges.as_slice());
        assert!((plan_cost(&out[..num_coalesced], &cfg) - ref_cost).abs() < 1e-9);
    }

    #[test]
    fn plan_coalesced_ranges_k500_under_200us_and_exact_equivalence_to_unpruned_dp() {
        const MAX_K: usize = 512;
        const K: usize = 500;

        let config = RangePlannerConfig::default();
        let mut intervals = [PageByteInterval::default(); MAX_K];

        // Construct 500 realistic TSP-ordered `.osv` minipages with mixed small gaps,
        // dense coalescing chains hitting the 8 MiB cap, and occasional large gaps > 256 KiB.
        let mut offset = 64_u64;
        let mut state = 0x1234_5678_9abc_def0_u64;
        for (idx, slot) in intervals[..K].iter_mut().enumerate() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let selector = (state >> 32) % 20;
            let gap = if selector == 0 {
                // Large gap (> 256 KiB) triggering Bellman cut
                300_000 + ((state >> 16) % 200_000)
            } else if selector <= 4 {
                // Medium gap (< 256 KiB) that tests DP trade-off
                32_768 + ((state >> 16) % 120_000)
            } else {
                // Contiguous or near-contiguous TSP locality run
                (state >> 16) % 4_096
            };

            offset += gap;
            let size = 8_192 + ((state >> 48) % 8_192);
            *slot = PageByteInterval {
                start: offset,
                end: offset + size,
                minipage_id: u32::try_from(idx).unwrap_or(u32::MAX),
            };
            offset += size;
        }

        let mut out_ranges = [CoalescedRange::default(); MAX_K];

        // Warm-up call to avoid cold instruction-cache skew.
        let _ = plan_coalesced_ranges_stack(black_box(&intervals), K, &config, &mut out_ranges);

        let start = Instant::now();
        let num_coalesced =
            plan_coalesced_ranges_stack(black_box(&intervals), K, &config, &mut out_ranges);
        let elapsed = start.elapsed();

        assert!(num_coalesced > 0 && num_coalesced <= K);
        #[cfg(not(miri))]
        assert!(
            elapsed < Duration::from_micros(200),
            "DP range stack planning for k={K} took {:?}, exceeding 200us budget",
            elapsed
        );
        let _ = elapsed;

        // Verify contiguous coverage of all 500 intervals and strict `max_coalesce_bytes` adherence.
        let mut covered = 0usize;
        for (range_idx, range) in out_ranges[..num_coalesced].iter().enumerate() {
            assert_eq!(usize::from(range.first_interval_idx), covered);
            assert!(range.interval_count > 0);
            let last_idx = covered + usize::from(range.interval_count) - 1;
            assert_eq!(range.start, intervals[covered].start);
            assert_eq!(range.end, intervals[last_idx].end);
            assert!(range.byte_len() <= config.max_coalesce_bytes);
            if range_idx > 0 {
                assert!(range.start >= out_ranges[range_idx - 1].end);
            }
            covered += usize::from(range.interval_count);
        }
        assert_eq!(covered, K);

        // Exact match against full unpruned O(k^2) Bellman DP recurrence.
        let (ref_ranges, ref_cost) = reference_unpruned_bellman_dp(&intervals, K, &config);
        assert_eq!(&out_ranges[..num_coalesced], ref_ranges.as_slice());
        assert_eq!(plan_cost(&out_ranges[..num_coalesced], &config), ref_cost);
    }

    #[test]
    fn plan_coalesced_ranges_handles_extreme_offsets_and_config_bounds() {
        let near_max = u64::MAX - 20_000;
        let intervals = [
            PageByteInterval {
                start: near_max,
                end: near_max + 4_096,
                minipage_id: 0,
            },
            PageByteInterval {
                start: near_max + 8_192,
                end: near_max + 12_288,
                minipage_id: 1,
            },
        ];
        let mut out = [CoalescedRange::default(); 8];

        let max_cfg = RangePlannerConfig {
            tau_req: f64::MAX,
            tau_byte: 1.0,
            max_coalesce_bytes: u64::MAX,
        };
        let c1 = plan_coalesced_ranges_stack(&intervals, intervals.len(), &max_cfg, &mut out);
        assert_eq!(c1, 1);
        assert_eq!(out[0].start, near_max);
        assert_eq!(out[0].end, near_max + 12_288);

        let nan_cfg = RangePlannerConfig {
            tau_req: f64::NAN,
            tau_byte: -5.0,
            max_coalesce_bytes: u64::MAX,
        };
        let c2 = plan_coalesced_ranges_stack(&intervals, intervals.len(), &nan_cfg, &mut out);
        assert_eq!(c2, 2);
    }
}
