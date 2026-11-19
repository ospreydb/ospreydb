//! Prometheus observability registry exporting P50/P95/P99 latency histograms for
//! Stage-1 RaBitQ scan, Stage-2 DP range planner, S3 calls saved, and Stage-4 `LUT-Rank` throughput.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Histogram upper bounds in microseconds for CPU stage latencies.
const LATENCY_BUCKETS_US: &[u64] = &[
    10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000,
];

/// Bounded sliding window capacity per metric for exact P50/P95/P99 percentile calculation.
const QUANTILE_WINDOW_CAP: usize = 4_096;

#[derive(Debug)]
struct LatencyHistogram {
    buckets: Vec<AtomicU64>,
    overflow: AtomicU64,
    sum_us: AtomicU64,
    count: AtomicU64,
    ring: Mutex<RingSamples>,
}

#[derive(Debug, Default)]
struct RingSamples {
    buf: Vec<u64>,
    cursor: usize,
}

impl RingSamples {
    fn push(&mut self, value: u64) {
        if self.buf.len() < QUANTILE_WINDOW_CAP {
            self.buf.push(value);
        } else {
            self.buf[self.cursor] = value;
            self.cursor = (self.cursor + 1) % QUANTILE_WINDOW_CAP;
        }
    }

    fn quantiles(&self) -> (f64, f64, f64) {
        if self.buf.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let mut sorted = self.buf.clone();
        sorted.sort_unstable();
        let n = sorted.len();
        let p50 = sorted[(n - 1) * 50 / 100];
        let p95 = sorted[(n - 1) * 95 / 100];
        let p99 = sorted[(n - 1) * 99 / 100];
        #[allow(clippy::cast_precision_loss)]
        (p50 as f64, p95 as f64, p99 as f64)
    }
}

impl LatencyHistogram {
    fn new() -> Self {
        let mut buckets = Vec::with_capacity(LATENCY_BUCKETS_US.len());
        for _ in LATENCY_BUCKETS_US {
            buckets.push(AtomicU64::new(0));
        }
        Self {
            buckets,
            overflow: AtomicU64::new(0),
            sum_us: AtomicU64::new(0),
            count: AtomicU64::new(0),
            ring: Mutex::new(RingSamples {
                buf: Vec::with_capacity(256),
                cursor: 0,
            }),
        }
    }

    fn observe(&self, latency_us: u64) {
        let mut matched = false;
        for (idx, &bound) in LATENCY_BUCKETS_US.iter().enumerate() {
            if latency_us <= bound {
                self.buckets[idx].fetch_add(1, Ordering::Relaxed);
                matched = true;
                break;
            }
        }
        if !matched {
            self.overflow.fetch_add(1, Ordering::Relaxed);
        }
        self.sum_us.fetch_add(latency_us, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);

        if let Ok(mut guard) = self.ring.try_lock() {
            guard.push(latency_us);
        }
    }

    fn quantiles(&self) -> (f64, f64, f64) {
        self.ring
            .lock()
            .map_or((0.0, 0.0, 0.0), |guard| guard.quantiles())
    }

    fn write_prometheus(&self, out: &mut String, metric_base: &str, help: &str) {
        let _ = writeln!(out, "# HELP {metric_base} {help}");
        let _ = writeln!(out, "# TYPE {metric_base} histogram");

        let mut cumulative = 0_u64;
        for (idx, &bound) in LATENCY_BUCKETS_US.iter().enumerate() {
            cumulative = cumulative.saturating_add(self.buckets[idx].load(Ordering::Relaxed));
            let _ = writeln!(out, "{metric_base}_bucket{{le=\"{bound}\"}} {cumulative}");
        }
        let total = cumulative.saturating_add(self.overflow.load(Ordering::Relaxed));
        let _ = writeln!(out, "{metric_base}_bucket{{le=\"+Inf\"}} {total}");
        let sum = self.sum_us.load(Ordering::Relaxed);
        let count = self.count.load(Ordering::Relaxed);
        let _ = writeln!(out, "{metric_base}_sum {sum}");
        let _ = writeln!(out, "{metric_base}_count {count}");

        let (p50, p95, p99) = self.quantiles();
        let _ = writeln!(
            out,
            "# HELP {metric_base}_p50_microseconds P50 latency in microseconds"
        );
        let _ = writeln!(out, "# TYPE {metric_base}_p50_microseconds gauge");
        let _ = writeln!(out, "{metric_base}_p50_microseconds {p50:.2}");
        let _ = writeln!(
            out,
            "# HELP {metric_base}_p95_microseconds P95 latency in microseconds"
        );
        let _ = writeln!(out, "# TYPE {metric_base}_p95_microseconds gauge");
        let _ = writeln!(out, "{metric_base}_p95_microseconds {p95:.2}");
        let _ = writeln!(
            out,
            "# HELP {metric_base}_p99_microseconds P99 latency in microseconds"
        );
        let _ = writeln!(out, "# TYPE {metric_base}_p99_microseconds gauge");
        let _ = writeln!(out, "{metric_base}_p99_microseconds {p99:.2}");
    }
}

/// Snapshot summary of [`ProMetricsRegistry`] counters and quantiles.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MetricsSnapshot {
    /// Total RaBitQ scans recorded.
    pub rabitq_scan_count: u64,
    /// RaBitQ scan P50/P95/P99 latency in microseconds.
    pub rabitq_p50_p95_p99_us: (f64, f64, f64),
    /// Total DP range plans recorded.
    pub range_planner_count: u64,
    /// Range planner P50/P95/P99 latency in microseconds.
    pub range_planner_p50_p95_p99_us: (f64, f64, f64),
    /// Total S3 `Range-GET` round trips avoided via DP range coalescing and LRU cache hits.
    pub s3_calls_saved_total: u64,
    /// Total raw `.osv` minipage requests before coalescing/caching.
    pub s3_raw_page_requests_total: u64,
    /// Total coalesced S3 `Range-GET` calls issued.
    pub s3_coalesced_range_gets_total: u64,
    /// Total minipages served from `MinipageLruCache`.
    pub s3_cache_hits_total: u64,
    /// Total LUT-Rank tokens scored.
    pub lutrank_tokens_total: u64,
    /// Observed LUT-Rank throughput (tokens/sec).
    pub lutrank_tokens_per_sec: f64,
    /// LUT-Rank P50/P95/P99 latency in microseconds.
    pub lutrank_p50_p95_p99_us: (f64, f64, f64),
}

/// Lock-free atomic histogram + sum/count/max tracker for per-stage breakdown telemetry.
#[derive(Debug)]
struct AtomicStageHistogram {
    buckets: [AtomicU64; 13],
    overflow: AtomicU64,
    sum_us: AtomicU64,
    count: AtomicU64,
    max_us: AtomicU64,
}

impl AtomicStageHistogram {
    fn new() -> Self {
        Self {
            buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            overflow: AtomicU64::new(0),
            sum_us: AtomicU64::new(0),
            count: AtomicU64::new(0),
            max_us: AtomicU64::new(0),
        }
    }

    #[inline]
    fn observe(&self, latency_us: u64) {
        let mut matched = false;
        for (idx, &bound) in LATENCY_BUCKETS_US.iter().enumerate() {
            if latency_us <= bound {
                self.buckets[idx].fetch_add(1, Ordering::Relaxed);
                matched = true;
                break;
            }
        }
        if !matched {
            self.overflow.fetch_add(1, Ordering::Relaxed);
        }
        self.sum_us.fetch_add(latency_us, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        self.max_us.fetch_max(latency_us, Ordering::Relaxed);
    }

    fn write_prometheus(&self, out: &mut String, metric_base: &str, help: &str) {
        let _ = writeln!(out, "# HELP {metric_base} {help}");
        let _ = writeln!(out, "# TYPE {metric_base} histogram");
        let mut cumulative = 0_u64;
        for (idx, &bound) in LATENCY_BUCKETS_US.iter().enumerate() {
            cumulative = cumulative.saturating_add(self.buckets[idx].load(Ordering::Relaxed));
            let _ = writeln!(out, "{metric_base}_bucket{{le=\"{bound}\"}} {cumulative}");
        }
        let total = cumulative.saturating_add(self.overflow.load(Ordering::Relaxed));
        let _ = writeln!(out, "{metric_base}_bucket{{le=\"+Inf\"}} {total}");
        let sum = self.sum_us.load(Ordering::Relaxed);
        let count = self.count.load(Ordering::Relaxed);
        let max = self.max_us.load(Ordering::Relaxed);
        let _ = writeln!(out, "{metric_base}_sum {sum}");
        let _ = writeln!(out, "{metric_base}_count {count}");
        let _ = writeln!(out, "# TYPE {metric_base}_max_microseconds gauge");
        let _ = writeln!(out, "{metric_base}_max_microseconds {max}");
    }
}

/// Lock-free Prometheus metrics collector for `osprey` servers and query pipelines.
#[derive(Debug)]
pub struct ProMetricsRegistry {
    rabitq_scan_us: LatencyHistogram,
    range_planner_us: LatencyHistogram,
    lutrank_us: LatencyHistogram,
    stage1_us: AtomicStageHistogram,
    cache_lookup_us: AtomicStageHistogram,
    disk_read_us: AtomicStageHistogram,
    zstd_decode_us: AtomicStageHistogram,
    f16_score_us: AtomicStageHistogram,
    s3_raw_page_requests: AtomicU64,
    s3_coalesced_range_gets: AtomicU64,
    s3_cache_hits: AtomicU64,
    s3_calls_saved: AtomicU64,
    lutrank_docs_total: AtomicU64,
    lutrank_tokens_total: AtomicU64,
    lutrank_total_elapsed_us: AtomicU64,
    wal_batches_total: AtomicU64,
    delta_ingested_docs_total: AtomicU64,
    compactions_total: AtomicU64,
    gc_tombstones_reclaimed_total: AtomicU64,
}

impl Default for ProMetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ProMetricsRegistry {
    /// Creates an empty [`ProMetricsRegistry`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            rabitq_scan_us: LatencyHistogram::new(),
            range_planner_us: LatencyHistogram::new(),
            lutrank_us: LatencyHistogram::new(),
            stage1_us: AtomicStageHistogram::new(),
            cache_lookup_us: AtomicStageHistogram::new(),
            disk_read_us: AtomicStageHistogram::new(),
            zstd_decode_us: AtomicStageHistogram::new(),
            f16_score_us: AtomicStageHistogram::new(),
            s3_raw_page_requests: AtomicU64::new(0),
            s3_coalesced_range_gets: AtomicU64::new(0),
            s3_cache_hits: AtomicU64::new(0),
            s3_calls_saved: AtomicU64::new(0),
            lutrank_docs_total: AtomicU64::new(0),
            lutrank_tokens_total: AtomicU64::new(0),
            lutrank_total_elapsed_us: AtomicU64::new(0),
            wal_batches_total: AtomicU64::new(0),
            delta_ingested_docs_total: AtomicU64::new(0),
            compactions_total: AtomicU64::new(0),
            gc_tombstones_reclaimed_total: AtomicU64::new(0),
        }
    }

    /// Records a Stage-1 RaBitQ scan latency sample in microseconds.
    pub fn observe_rabitq_scan(&self, latency_us: u64) {
        self.rabitq_scan_us.observe(latency_us);
        self.stage1_us.observe(latency_us);
    }

    /// Records Stage-2/Stage-3 breakdown timings in microseconds (lock-free).
    pub fn observe_refine_breakdown(
        &self,
        cache_lookup_us: u64,
        disk_read_us: u64,
        zstd_decode_us: u64,
        f16_score_us: u64,
    ) {
        self.cache_lookup_us.observe(cache_lookup_us);
        self.disk_read_us.observe(disk_read_us);
        self.zstd_decode_us.observe(zstd_decode_us);
        self.f16_score_us.observe(f16_score_us);
    }

    /// Records Stage-2 DP range planning latency and S3 request savings.
    pub fn observe_range_plan_and_s3(
        &self,
        range_plan_time_us: u64,
        raw_page_requests: usize,
        coalesced_range_gets: usize,
        cache_hits: usize,
    ) {
        self.range_planner_us.observe(range_plan_time_us);
        let raw_u64 = u64::try_from(raw_page_requests).unwrap_or(u64::MAX);
        let coalesced_u64 = u64::try_from(coalesced_range_gets).unwrap_or(u64::MAX);
        let hits_u64 = u64::try_from(cache_hits).unwrap_or(u64::MAX);
        let saved = raw_u64.saturating_sub(coalesced_u64);

        self.s3_raw_page_requests
            .fetch_add(raw_u64, Ordering::Relaxed);
        self.s3_coalesced_range_gets
            .fetch_add(coalesced_u64, Ordering::Relaxed);
        self.s3_cache_hits.fetch_add(hits_u64, Ordering::Relaxed);
        self.s3_calls_saved.fetch_add(saved, Ordering::Relaxed);
    }

    /// Records a `LUT-Rank` rerank execution (`docs`, `tokens`, and duration in microseconds).
    pub fn observe_lutrank(&self, docs: usize, tokens: usize, duration_us: u64) {
        self.lutrank_us.observe(duration_us);
        let docs_u64 = u64::try_from(docs).unwrap_or(u64::MAX);
        let tokens_u64 = u64::try_from(tokens).unwrap_or(u64::MAX);
        self.lutrank_docs_total
            .fetch_add(docs_u64, Ordering::Relaxed);
        self.lutrank_tokens_total
            .fetch_add(tokens_u64, Ordering::Relaxed);
        self.lutrank_total_elapsed_us
            .fetch_add(duration_us.max(1), Ordering::Relaxed);
    }

    /// Records a WAL + delta ingestion batch.
    pub fn observe_wal_ingest(&self, num_docs: usize) {
        self.wal_batches_total.fetch_add(1, Ordering::Relaxed);
        self.delta_ingested_docs_total.fetch_add(
            u64::try_from(num_docs).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
    }

    /// Records a completed CoW compaction cycle.
    pub fn observe_compaction(&self) {
        self.compactions_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Records reclaimed tombstones from `TombstoneGarbageCollector`.
    pub fn observe_gc_reclaimed(&self, tombstones: u64) {
        self.gc_tombstones_reclaimed_total
            .fetch_add(tombstones, Ordering::Relaxed);
    }

    /// Returns a point-in-time [`MetricsSnapshot`].
    #[must_use]
    pub fn snapshot(&self) -> MetricsSnapshot {
        let tokens = self.lutrank_tokens_total.load(Ordering::Relaxed);
        let elapsed_us = self.lutrank_total_elapsed_us.load(Ordering::Relaxed);
        #[allow(clippy::cast_precision_loss)]
        let lutrank_tokens_per_sec = if elapsed_us == 0 {
            0.0
        } else {
            (tokens as f64) * 1_000_000.0 / (elapsed_us as f64)
        };

        MetricsSnapshot {
            rabitq_scan_count: self.rabitq_scan_us.count.load(Ordering::Relaxed),
            rabitq_p50_p95_p99_us: self.rabitq_scan_us.quantiles(),
            range_planner_count: self.range_planner_us.count.load(Ordering::Relaxed),
            range_planner_p50_p95_p99_us: self.range_planner_us.quantiles(),
            s3_calls_saved_total: self.s3_calls_saved.load(Ordering::Relaxed),
            s3_raw_page_requests_total: self.s3_raw_page_requests.load(Ordering::Relaxed),
            s3_coalesced_range_gets_total: self.s3_coalesced_range_gets.load(Ordering::Relaxed),
            s3_cache_hits_total: self.s3_cache_hits.load(Ordering::Relaxed),
            lutrank_tokens_total: tokens,
            lutrank_tokens_per_sec,
            lutrank_p50_p95_p99_us: self.lutrank_us.quantiles(),
        }
    }

    /// Renders all metrics in Prometheus text exposition format (`0.0.4`).
    #[must_use]
    pub fn render_prometheus(&self) -> String {
        let mut out = String::with_capacity(4096);

        self.rabitq_scan_us.write_prometheus(
            &mut out,
            "osprey_pro_rabitq_scan_duration_microseconds",
            "Stage-1 1-bit RaBitQ scan latency in microseconds",
        );
        self.range_planner_us.write_prometheus(
            &mut out,
            "osprey_pro_range_planner_duration_microseconds",
            "Stage-2 DP range coalescing planner latency in microseconds",
        );
        self.lutrank_us.write_prometheus(
            &mut out,
            "osprey_pro_lutrank_duration_microseconds",
            "Stage-4 LUT-Rank MaxSim reranking latency in microseconds",
        );
        self.stage1_us.write_prometheus(
            &mut out,
            "osprey_pro_stage1_duration_microseconds",
            "Stage-1 RaBitQ scan latency in microseconds",
        );
        self.cache_lookup_us.write_prometheus(
            &mut out,
            "osprey_pro_cache_lookup_duration_microseconds",
            "Stage-2 minipage LRU cache lookup latency in microseconds",
        );
        self.disk_read_us.write_prometheus(
            &mut out,
            "osprey_pro_disk_read_duration_microseconds",
            "Stage-2 object-store or local pread latency in microseconds",
        );
        self.zstd_decode_us.write_prometheus(
            &mut out,
            "osprey_pro_zstd_decode_duration_microseconds",
            "Stage-2 Zstd decompression and checksum verification latency in microseconds",
        );
        self.f16_score_us.write_prometheus(
            &mut out,
            "osprey_pro_f16_score_duration_microseconds",
            "Stage-3 exact f16 scoring and top-K selection latency in microseconds",
        );

        let snap = self.snapshot();
        let _ = writeln!(
            out,
            "# HELP osprey_pro_s3_calls_saved_total Total S3 Range-GET calls saved by DP range coalescing and LRU cache"
        );
        let _ = writeln!(out, "# TYPE osprey_pro_s3_calls_saved_total counter");
        let _ = writeln!(
            out,
            "osprey_pro_s3_calls_saved_total {}",
            snap.s3_calls_saved_total
        );

        let _ = writeln!(
            out,
            "# HELP osprey_pro_s3_raw_page_requests_total Total minipage requests before DP range coalescing"
        );
        let _ = writeln!(out, "# TYPE osprey_pro_s3_raw_page_requests_total counter");
        let _ = writeln!(
            out,
            "osprey_pro_s3_raw_page_requests_total {}",
            snap.s3_raw_page_requests_total
        );

        let _ = writeln!(
            out,
            "# HELP osprey_pro_s3_coalesced_range_gets_total Total coalesced S3 Range-GET calls executed"
        );
        let _ = writeln!(
            out,
            "# TYPE osprey_pro_s3_coalesced_range_gets_total counter"
        );
        let _ = writeln!(
            out,
            "osprey_pro_s3_coalesced_range_gets_total {}",
            snap.s3_coalesced_range_gets_total
        );

        let _ = writeln!(
            out,
            "# HELP osprey_pro_s3_cache_hits_total Total .osv minipage cache hits"
        );
        let _ = writeln!(out, "# TYPE osprey_pro_s3_cache_hits_total counter");
        let _ = writeln!(
            out,
            "osprey_pro_s3_cache_hits_total {}",
            snap.s3_cache_hits_total
        );

        let _ = writeln!(
            out,
            "# HELP osprey_pro_lutrank_tokens_total Total ColBERT tokens evaluated by LUT-Rank"
        );
        let _ = writeln!(out, "# TYPE osprey_pro_lutrank_tokens_total counter");
        let _ = writeln!(
            out,
            "osprey_pro_lutrank_tokens_total {}",
            snap.lutrank_tokens_total
        );

        let _ = writeln!(
            out,
            "# HELP osprey_pro_lutrank_docs_total Total candidate documents scored by LUT-Rank"
        );
        let _ = writeln!(out, "# TYPE osprey_pro_lutrank_docs_total counter");
        let _ = writeln!(
            out,
            "osprey_pro_lutrank_docs_total {}",
            self.lutrank_docs_total.load(Ordering::Relaxed)
        );

        let _ = writeln!(
            out,
            "# HELP osprey_pro_lutrank_tokens_per_sec Aggregate LUT-Rank throughput in tokens per second"
        );
        let _ = writeln!(out, "# TYPE osprey_pro_lutrank_tokens_per_sec gauge");
        let _ = writeln!(
            out,
            "osprey_pro_lutrank_tokens_per_sec {:.2}",
            snap.lutrank_tokens_per_sec
        );

        let _ = writeln!(
            out,
            "# HELP osprey_pro_wal_batches_total Total WAL batch commits"
        );
        let _ = writeln!(out, "# TYPE osprey_pro_wal_batches_total counter");
        let _ = writeln!(
            out,
            "osprey_pro_wal_batches_total {}",
            self.wal_batches_total.load(Ordering::Relaxed)
        );

        let _ = writeln!(
            out,
            "# HELP osprey_pro_delta_ingested_docs_total Total documents ingested into LSM Delta Buffer"
        );
        let _ = writeln!(out, "# TYPE osprey_pro_delta_ingested_docs_total counter");
        let _ = writeln!(
            out,
            "osprey_pro_delta_ingested_docs_total {}",
            self.delta_ingested_docs_total.load(Ordering::Relaxed)
        );

        let _ = writeln!(
            out,
            "# HELP osprey_pro_compactions_total Total CoW slab compactions completed"
        );
        let _ = writeln!(out, "# TYPE osprey_pro_compactions_total counter");
        let _ = writeln!(
            out,
            "osprey_pro_compactions_total {}",
            self.compactions_total.load(Ordering::Relaxed)
        );

        let _ = writeln!(
            out,
            "# HELP osprey_pro_gc_tombstones_reclaimed_total Total tombstones reclaimed by Tombstone GC"
        );
        let _ = writeln!(
            out,
            "# TYPE osprey_pro_gc_tombstones_reclaimed_total counter"
        );
        let _ = writeln!(
            out,
            "osprey_pro_gc_tombstones_reclaimed_total {}",
            self.gc_tombstones_reclaimed_total.load(Ordering::Relaxed)
        );

        out
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn prometheus_exposition_renders_histograms_quantiles_and_savings() {
        let reg = ProMetricsRegistry::new();
        for us in [20, 40, 60, 120, 450, 900] {
            reg.observe_rabitq_scan(us);
        }
        reg.observe_range_plan_and_s3(180, 16, 3, 5);
        reg.observe_lutrank(32, 4096, 500);
        reg.observe_refine_breakdown(15, 35, 10, 45);

        let text = reg.render_prometheus();
        assert!(text.contains("osprey_pro_rabitq_scan_duration_microseconds_bucket{le=\"50\"}"));
        assert!(text.contains("osprey_pro_rabitq_scan_duration_microseconds_p50_microseconds"));
        assert!(text.contains("osprey_pro_rabitq_scan_duration_microseconds_p95_microseconds"));
        assert!(text.contains("osprey_pro_rabitq_scan_duration_microseconds_p99_microseconds"));
        assert!(text.contains("osprey_pro_range_planner_duration_microseconds_count 1"));
        assert!(text.contains("osprey_pro_s3_calls_saved_total 13"));
        assert!(text.contains("osprey_pro_lutrank_tokens_total 4096"));
        assert!(text.contains("osprey_pro_lutrank_tokens_per_sec"));
        assert!(text.contains("osprey_pro_stage1_duration_microseconds_count 6"));
        assert!(text.contains("osprey_pro_cache_lookup_duration_microseconds_count 1"));
        assert!(text.contains("osprey_pro_disk_read_duration_microseconds_count 1"));
        assert!(text.contains("osprey_pro_zstd_decode_duration_microseconds_count 1"));
        assert!(text.contains("osprey_pro_f16_score_duration_microseconds_count 1"));
    }
}
