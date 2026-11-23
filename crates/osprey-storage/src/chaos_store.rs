//! Deterministic S3/Object-Storage chaos-engineering wrapper.
//!
//! [`ChaosObjectStore`] wraps any `object_store::ObjectStore` and intercepts
//! `get_opts`, `get_range`, and `get_ranges` to deterministically inject:
//! - Transient HTTP 500 / 503 errors (`object_store::Error::Generic`) on the first `N`
//!   reads or at a configurable probability via [`SplitMix64`];
//! - Tail-latency stalls / timeouts (e.g. `120 ms`) to exercise speculative P99 hedged requests;
//! - Partial (truncated) responses and single-byte in-transit corruptions to validate
//!   `page_xxh64` integrity checks and bounds guards without panicking.

use std::fmt;
use std::future::Future;
use std::ops::Range;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use bytes::Bytes;
use futures::stream::BoxStream;
use object_store::path::Path as ObjectPath;
use object_store::{
    GetOptions, GetResult, GetResultPayload, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOpts, PutOptions, PutPayload, PutResult, Result as ObjectStoreResult,
};
use osprey_core::math::SplitMix64;

type BoxFut<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Fault-injection profile for [`ChaosObjectStore`].
#[derive(Clone, Debug, PartialEq)]
pub struct ChaosConfig {
    /// Deterministic `SplitMix64` seed driving per-call pseudo-random fault decisions.
    pub seed: u64,
    /// Unconditionally fail the first `fail_first_n` read calls with HTTP 500/503.
    pub fail_first_n: usize,
    /// Explicit 0-based read call indices that must fail with HTTP 500/503.
    pub http_error_calls: Vec<usize>,
    /// Probability in `[0.0, 1.0]` of injecting an HTTP 500/503 error on a read call.
    pub http_error_rate: f64,
    /// Explicit 0-based read call indices that must incur `tail_latency_ms`.
    pub tail_latency_calls: Vec<usize>,
    /// Probability in `[0.0, 1.0]` of injecting `tail_latency_ms` on a read call.
    pub tail_latency_rate: f64,
    /// Artificial tail-latency spike duration in milliseconds (e.g. `120`).
    pub tail_latency_ms: u64,
    /// Baseline network TTFB latency applied to read calls in milliseconds.
    pub base_latency_ms: u64,
    /// Explicit 0-based read call indices that must return a truncated (partial) response.
    pub partial_response_calls: Vec<usize>,
    /// Probability in `[0.0, 1.0]` of returning a truncated (partial) response.
    pub partial_response_rate: f64,
    /// Explicit 0-based read call indices that must corrupt one byte in transit.
    pub corrupt_bytes_calls: Vec<usize>,
    /// Probability in `[0.0, 1.0]` of flipping a byte in the returned response.
    pub corrupt_bytes_rate: f64,
}

impl Default for ChaosConfig {
    fn default() -> Self {
        Self {
            seed: 0x4348_414F_535F_0143,
            fail_first_n: 0,
            http_error_calls: Vec::new(),
            http_error_rate: 0.0,
            tail_latency_calls: Vec::new(),
            tail_latency_rate: 0.0,
            tail_latency_ms: 0,
            base_latency_ms: 0,
            partial_response_calls: Vec::new(),
            partial_response_rate: 0.0,
            corrupt_bytes_calls: Vec::new(),
            corrupt_bytes_rate: 0.0,
        }
    }
}

/// Snapshot of fault-injection and I/O telemetry counters recorded by [`ChaosObjectStore`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ChaosStatsSnapshot {
    /// Total read calls (`get_opts`, `get_range`, `get_ranges`) intercepted.
    pub total_reads: usize,
    /// Total HTTP 500 / 503 `Error::Generic` faults injected.
    pub http_errors_injected: usize,
    /// Total tail-latency stalls injected.
    pub tail_spikes_injected: usize,
    /// Total truncated (partial) responses injected.
    pub partial_responses_injected: usize,
    /// Total byte-corrupted responses injected.
    pub corruptions_injected: usize,
    /// Total payload bytes returned by successful or corrupted reads.
    pub bytes_served: u64,
}

#[derive(Debug, Default)]
struct ChaosStats {
    total_reads: AtomicUsize,
    http_errors_injected: AtomicUsize,
    tail_spikes_injected: AtomicUsize,
    partial_responses_injected: AtomicUsize,
    corruptions_injected: AtomicUsize,
    bytes_served: AtomicU64,
}

impl ChaosStats {
    fn snapshot(&self) -> ChaosStatsSnapshot {
        ChaosStatsSnapshot {
            total_reads: self.total_reads.load(Ordering::Relaxed),
            http_errors_injected: self.http_errors_injected.load(Ordering::Relaxed),
            tail_spikes_injected: self.tail_spikes_injected.load(Ordering::Relaxed),
            partial_responses_injected: self.partial_responses_injected.load(Ordering::Relaxed),
            corruptions_injected: self.corruptions_injected.load(Ordering::Relaxed),
            bytes_served: self.bytes_served.load(Ordering::Relaxed),
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct ChaosDecision {
    call_idx: usize,
    inject_http_error: bool,
    sleep_ms: u64,
    inject_tail_spike: bool,
    inject_partial: bool,
    inject_corrupt: bool,
    corrupt_offset_seed: u64,
}

/// Deterministic fault-injecting `object_store::ObjectStore` wrapper over an inner store `T`.
///
/// Intercepts `get_opts`, `get_range`, and `get_ranges` to simulate real-world S3 transient
/// HTTP 500/503 responses, P99 tail-latency stalls, truncated payloads, and bit flips.
pub struct ChaosObjectStore<T: ObjectStore> {
    inner: T,
    config: ChaosConfig,
    stats: ChaosStats,
}

impl<T: ObjectStore> fmt::Debug for ChaosObjectStore<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChaosObjectStore")
            .field("inner", &self.inner)
            .field("config", &self.config)
            .field("stats", &self.stats.snapshot())
            .finish()
    }
}

impl<T: ObjectStore> fmt::Display for ChaosObjectStore<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ChaosObjectStore({})", self.inner)
    }
}

#[inline]
fn word_to_unit_f64(word: u64) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    {
        ((word >> 11) as f64) / ((1_u64 << 53) as f64)
    }
}

impl<T: ObjectStore> ChaosObjectStore<T> {
    /// Wraps `inner` with the given deterministic [`ChaosConfig`].
    pub fn new(inner: T, config: ChaosConfig) -> Self {
        Self {
            inner,
            config,
            stats: ChaosStats::default(),
        }
    }

    /// Returns a reference to the wrapped inner `ObjectStore`.
    #[inline]
    pub const fn inner(&self) -> &T {
        &self.inner
    }

    /// Returns a reference to the active [`ChaosConfig`].
    #[inline]
    pub const fn config(&self) -> &ChaosConfig {
        &self.config
    }

    /// Returns a point-in-time snapshot of fault-injection telemetry counters.
    #[inline]
    pub fn stats(&self) -> ChaosStatsSnapshot {
        self.stats.snapshot()
    }

    fn decide_fault(&self) -> ChaosDecision {
        let call_idx = self.stats.total_reads.fetch_add(1, Ordering::SeqCst);
        let mixed_seed = self
            .config
            .seed
            .wrapping_add((call_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut rng = SplitMix64::new(mixed_seed);

        let u_http = word_to_unit_f64(rng.next_u64());
        let u_tail = word_to_unit_f64(rng.next_u64());
        let u_partial = word_to_unit_f64(rng.next_u64());
        let u_corrupt = word_to_unit_f64(rng.next_u64());
        let corrupt_offset_seed = rng.next_u64();

        let inject_http_error = call_idx < self.config.fail_first_n
            || self.config.http_error_calls.contains(&call_idx)
            || (self.config.http_error_rate > 0.0 && u_http < self.config.http_error_rate);

        let inject_tail_spike = self.config.tail_latency_ms > 0
            && (self.config.tail_latency_calls.contains(&call_idx)
                || (self.config.tail_latency_rate > 0.0 && u_tail < self.config.tail_latency_rate));

        let mut sleep_ms = self.config.base_latency_ms;
        if inject_tail_spike {
            sleep_ms = sleep_ms.saturating_add(self.config.tail_latency_ms);
        }

        let inject_partial = self.config.partial_response_calls.contains(&call_idx)
            || (self.config.partial_response_rate > 0.0
                && u_partial < self.config.partial_response_rate);

        let inject_corrupt = self.config.corrupt_bytes_calls.contains(&call_idx)
            || (self.config.corrupt_bytes_rate > 0.0 && u_corrupt < self.config.corrupt_bytes_rate);

        ChaosDecision {
            call_idx,
            inject_http_error,
            sleep_ms,
            inject_tail_spike,
            inject_partial,
            inject_corrupt,
            corrupt_offset_seed,
        }
    }

    fn make_http_error(call_idx: usize) -> object_store::Error {
        let (status, msg) = if call_idx.is_multiple_of(2) {
            (
                503_u16,
                "HTTP 503 Service Unavailable (simulated S3 transient chaos; connection reset)",
            )
        } else {
            (
                500_u16,
                "HTTP 500 Internal Server Error (simulated S3 transient chaos)",
            )
        };
        object_store::Error::Generic {
            store: "ChaosObjectStore",
            source: Box::new(std::io::Error::new(
                std::io::ErrorKind::ConnectionReset,
                format!("status={status} call_idx={call_idx}: {msg}"),
            )),
        }
    }

    async fn apply_pre_read_chaos(&self, decision: &ChaosDecision) -> ObjectStoreResult<()> {
        if decision.inject_tail_spike {
            self.stats
                .tail_spikes_injected
                .fetch_add(1, Ordering::Relaxed);
        }
        if decision.sleep_ms > 0 {
            tokio::time::sleep(Duration::from_millis(decision.sleep_ms)).await;
        }
        if decision.inject_http_error {
            self.stats
                .http_errors_injected
                .fetch_add(1, Ordering::Relaxed);
            return Err(Self::make_http_error(decision.call_idx));
        }
        Ok(())
    }

    fn apply_payload_chaos(&self, mut bytes: Bytes, decision: &ChaosDecision) -> Bytes {
        if decision.inject_partial && !bytes.is_empty() {
            let truncated_len = (bytes.len() / 2).min(bytes.len().saturating_sub(1));
            bytes = bytes.slice(..truncated_len);
            self.stats
                .partial_responses_injected
                .fetch_add(1, Ordering::Relaxed);
        }

        if decision.inject_corrupt && !bytes.is_empty() {
            let mut owned = bytes.to_vec();
            let offset =
                usize::try_from(decision.corrupt_offset_seed % (owned.len() as u64)).unwrap_or(0);
            owned[offset] ^= 0x5A;
            bytes = Bytes::from(owned);
            self.stats
                .corruptions_injected
                .fetch_add(1, Ordering::Relaxed);
        }

        self.stats.bytes_served.fetch_add(
            u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        bytes
    }
}

impl<T: ObjectStore> ObjectStore for ChaosObjectStore<T> {
    fn put_opts<'life0, 'life1, 'async_trait>(
        &'life0 self,
        location: &'life1 ObjectPath,
        payload: PutPayload,
        opts: PutOptions,
    ) -> BoxFut<'async_trait, ObjectStoreResult<PutResult>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move { self.inner.put_opts(location, payload, opts).await })
    }

    fn put_multipart_opts<'life0, 'life1, 'async_trait>(
        &'life0 self,
        location: &'life1 ObjectPath,
        opts: PutMultipartOpts,
    ) -> BoxFut<'async_trait, ObjectStoreResult<Box<dyn MultipartUpload>>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move { self.inner.put_multipart_opts(location, opts).await })
    }

    fn get_opts<'life0, 'life1, 'async_trait>(
        &'life0 self,
        location: &'life1 ObjectPath,
        options: GetOptions,
    ) -> BoxFut<'async_trait, ObjectStoreResult<GetResult>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move {
            let is_head = options.head;
            let decision = self.decide_fault();
            self.apply_pre_read_chaos(&decision).await?;

            let result = self.inner.get_opts(location, options).await?;
            if is_head || (!decision.inject_partial && !decision.inject_corrupt) {
                if !is_head {
                    let expected =
                        u64::try_from(result.range.end.saturating_sub(result.range.start))
                            .unwrap_or(u64::MAX);
                    self.stats
                        .bytes_served
                        .fetch_add(expected, Ordering::Relaxed);
                }
                return Ok(result);
            }

            let orig_range = result.range.clone();
            let meta = result.meta.clone();
            let attributes = result.attributes.clone();
            let raw = result.bytes().await?;
            let modified = self.apply_payload_chaos(raw, &decision);
            let modified_end = orig_range.start.saturating_add(modified.len());

            Ok(GetResult {
                payload: GetResultPayload::Stream(Box::pin(futures::stream::once(
                    futures::future::ready(Ok(modified)),
                ))),
                meta,
                range: orig_range.start..modified_end,
                attributes,
            })
        })
    }

    fn get_range<'life0, 'life1, 'async_trait>(
        &'life0 self,
        location: &'life1 ObjectPath,
        range: Range<usize>,
    ) -> BoxFut<'async_trait, ObjectStoreResult<Bytes>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move {
            let decision = self.decide_fault();
            self.apply_pre_read_chaos(&decision).await?;
            let bytes = self.inner.get_range(location, range).await?;
            Ok(self.apply_payload_chaos(bytes, &decision))
        })
    }

    fn get_ranges<'life0, 'life1, 'life2, 'async_trait>(
        &'life0 self,
        location: &'life1 ObjectPath,
        ranges: &'life2 [Range<usize>],
    ) -> BoxFut<'async_trait, ObjectStoreResult<Vec<Bytes>>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        'life2: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move {
            let decision = self.decide_fault();
            self.apply_pre_read_chaos(&decision).await?;
            let mut bufs = self.inner.get_ranges(location, ranges).await?;
            let mut applied_fault = false;
            for buf in &mut bufs {
                let raw = std::mem::take(buf);
                if !applied_fault && !raw.is_empty() {
                    *buf = self.apply_payload_chaos(raw, &decision);
                    applied_fault = true;
                } else {
                    self.stats.bytes_served.fetch_add(
                        u64::try_from(raw.len()).unwrap_or(u64::MAX),
                        Ordering::Relaxed,
                    );
                    *buf = raw;
                }
            }
            Ok(bufs)
        })
    }

    fn delete<'life0, 'life1, 'async_trait>(
        &'life0 self,
        location: &'life1 ObjectPath,
    ) -> BoxFut<'async_trait, ObjectStoreResult<()>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move { self.inner.delete(location).await })
    }

    fn list<'life0>(
        &'life0 self,
        prefix: Option<&ObjectPath>,
    ) -> BoxStream<'life0, ObjectStoreResult<ObjectMeta>> {
        self.inner.list(prefix)
    }

    fn list_with_offset<'life0, 'life1>(
        &'life0 self,
        prefix: Option<&'life1 ObjectPath>,
        offset: &'life1 ObjectPath,
    ) -> BoxStream<'life0, ObjectStoreResult<ObjectMeta>> {
        self.inner.list_with_offset(prefix, offset)
    }

    fn list_with_delimiter<'life0, 'life1, 'async_trait>(
        &'life0 self,
        prefix: Option<&'life1 ObjectPath>,
    ) -> BoxFut<'async_trait, ObjectStoreResult<ListResult>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move { self.inner.list_with_delimiter(prefix).await })
    }

    fn copy<'life0, 'life1, 'life2, 'async_trait>(
        &'life0 self,
        from: &'life1 ObjectPath,
        to: &'life2 ObjectPath,
    ) -> BoxFut<'async_trait, ObjectStoreResult<()>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        'life2: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move { self.inner.copy(from, to).await })
    }

    fn copy_if_not_exists<'life0, 'life1, 'life2, 'async_trait>(
        &'life0 self,
        from: &'life1 ObjectPath,
        to: &'life2 ObjectPath,
    ) -> BoxFut<'async_trait, ObjectStoreResult<()>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        'life2: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move { self.inner.copy_if_not_exists(from, to).await })
    }
}
