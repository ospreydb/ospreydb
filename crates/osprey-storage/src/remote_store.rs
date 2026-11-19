//! Multi-cloud `object_store` connector factory, HTTP/2 pool tuning, resilient
//! `tokio` coalesced range fetching (hedged requests + full-jitter exponential retry),
//! zero-copy `Bytes::slice` splitting, and parallel Zstd minipage decompression.

use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use futures::future::{try_join_all, BoxFuture};
use object_store::aws::AmazonS3Builder;
use object_store::azure::MicrosoftAzureBuilder;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::http::HttpBuilder;
use object_store::local::LocalFileSystem;
use object_store::memory::InMemory;
use object_store::path::Path as ObjectPath;
use object_store::{ClientOptions, ObjectStore, ObjectStoreScheme, RetryConfig};
use osprey_core::{MinipageDirectoryEntry, OspreyError, OspreyResult};
use rayon::prelude::*;
use tokio::sync::Semaphore;
use url::Url;
use zstd::dict::DecoderDictionary;

use crate::osv_slab::{verify_and_decompress_minipage_with, DecodedMinipage};
use crate::range_planner::{CoalescedRange, PageByteInterval};

/// Default maximum concurrent HTTP/2 streams for object-storage `Range-GET` requests.
pub const DEFAULT_MAX_CONCURRENT_STREAMS: usize = 64;

/// Default maximum idle connections per host in the HTTP/2 connection pool.
pub const DEFAULT_POOL_MAX_IDLE_PER_HOST: usize = 32;

/// Default HTTP/2 keep-alive interval in seconds.
pub const DEFAULT_HTTP2_KEEP_ALIVE_SECS: u64 = 30;

/// Default per-request timeout in milliseconds.
pub const DEFAULT_REQUEST_TIMEOUT_MS: u64 = 2_000;

/// Default speculative hedged-request delay threshold (P99 latency guard) in milliseconds.
pub const DEFAULT_HEDGED_REQUEST_DELAY_MS: u64 = 35;

/// Default maximum retry attempts for transient object-storage errors.
pub const DEFAULT_MAX_RETRIES: usize = 3;

/// Base backoff window in milliseconds for full-jitter exponential retries.
const RETRY_BASE_BACKOFF_MS: u64 = 15;

/// Tuning parameters for multi-cloud HTTP/2 object-storage clients, hedged requests, and retries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteStoreConfig {
    /// Maximum number of concurrent `Range-GET` streams (`default: 64`).
    pub max_concurrent_streams: usize,
    /// Maximum idle HTTP connections kept per host (`default: 32`).
    pub pool_max_idle_per_host: usize,
    /// HTTP/2 keep-alive probe interval in seconds (`default: 30`).
    pub http2_keep_alive_secs: u64,
    /// Request timeout in milliseconds (`default: 2_000`).
    pub request_timeout_ms: u64,
    /// P99 threshold in milliseconds before firing a speculative hedged request (`default: 35`).
    pub hedged_request_delay_ms: u64,
    /// Maximum full-jitter exponential backoff retries (`default: 3`).
    pub max_retries: usize,
    /// Optional custom endpoint URL (for MinIO, Cloudflare R2, or Ceph RGW).
    pub custom_endpoint: Option<String>,
}

impl Default for RemoteStoreConfig {
    #[inline]
    fn default() -> Self {
        Self {
            max_concurrent_streams: DEFAULT_MAX_CONCURRENT_STREAMS,
            pool_max_idle_per_host: DEFAULT_POOL_MAX_IDLE_PER_HOST,
            http2_keep_alive_secs: DEFAULT_HTTP2_KEEP_ALIVE_SECS,
            request_timeout_ms: DEFAULT_REQUEST_TIMEOUT_MS,
            hedged_request_delay_ms: DEFAULT_HEDGED_REQUEST_DELAY_MS,
            max_retries: DEFAULT_MAX_RETRIES,
            custom_endpoint: None,
        }
    }
}

/// Abstract asynchronous byte-range transport enabling both `object_store::ObjectStore`
/// backends and lightweight offline/mock implementations.
pub trait AsyncRangeTransport: Send + Sync {
    /// Fetches the specified byte ranges from `path` in order.
    fn get_ranges<'a>(
        &'a self,
        path: &'a ObjectPath,
        ranges: &'a [Range<usize>],
    ) -> BoxFuture<'a, OspreyResult<Vec<Bytes>>>;
}

impl<T: ObjectStore + ?Sized> AsyncRangeTransport for T {
    fn get_ranges<'a>(
        &'a self,
        path: &'a ObjectPath,
        ranges: &'a [Range<usize>],
    ) -> BoxFuture<'a, OspreyResult<Vec<Bytes>>> {
        Box::pin(async move {
            ObjectStore::get_ranges(self, path, ranges)
                .await
                .map_err(map_object_store_err)
        })
    }
}

/// Maps an `object_store::Error` into `OspreyError::Io`, preserving non-transient vs transient kinds.
fn map_object_store_err(err: object_store::Error) -> OspreyError {
    let kind = match &err {
        object_store::Error::NotFound { .. } => std::io::ErrorKind::NotFound,
        object_store::Error::InvalidPath { .. } => std::io::ErrorKind::InvalidInput,
        object_store::Error::PermissionDenied { .. }
        | object_store::Error::Unauthenticated { .. } => std::io::ErrorKind::PermissionDenied,
        object_store::Error::AlreadyExists { .. } => std::io::ErrorKind::AlreadyExists,
        object_store::Error::NotSupported { .. } => std::io::ErrorKind::Unsupported,
        _ => {
            let msg = err.to_string().to_ascii_lowercase();
            if msg.contains("timeout") || msg.contains("timed out") {
                std::io::ErrorKind::TimedOut
            } else if msg.contains("reset") || msg.contains("connection") {
                std::io::ErrorKind::ConnectionReset
            } else if msg.contains("429")
                || msg.contains("500")
                || msg.contains("502")
                || msg.contains("503")
                || msg.contains("504")
                || msg.contains("too many requests")
                || msg.contains("slowdown")
                || msg.contains("service unavailable")
                || msg.contains("internal server error")
            {
                std::io::ErrorKind::TimedOut
            } else {
                std::io::ErrorKind::Other
            }
        }
    };
    OspreyError::Io(std::io::Error::new(kind, err))
}

#[inline]
fn is_transient_osprey_error(err: &OspreyError) -> bool {
    match err {
        OspreyError::Io(io_err) => matches!(
            io_err.kind(),
            std::io::ErrorKind::TimedOut
                | std::io::ErrorKind::ConnectionReset
                | std::io::ErrorKind::ConnectionAborted
                | std::io::ErrorKind::Interrupted
                | std::io::ErrorKind::WouldBlock
                | std::io::ErrorKind::Other
        ),
        _ => false,
    }
}

/// Deterministic SplitMix64 PRNG used for full-jitter exponential retry backoff.
#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    #[inline]
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// Builds `object_store::RetryConfig` for transient HTTP `429`/`500`/`502`/`503`/`504`
/// and network errors using exponential backoff with jitter.
pub fn build_retry_config(config: &RemoteStoreConfig) -> RetryConfig {
    let mut retry = RetryConfig {
        max_retries: config.max_retries,
        ..RetryConfig::default()
    };
    if config.request_timeout_ms > 0 {
        let per_retry = Duration::from_millis(config.request_timeout_ms);
        let attempts = u32::try_from(config.max_retries.saturating_add(1)).unwrap_or(16);
        retry.retry_timeout = per_retry
            .saturating_mul(attempts)
            .max(Duration::from_secs(10));
    }
    retry
}

/// Builds `object_store::ClientOptions` tuned for HTTP/2 pooling, keep-alive, and strict timeouts.
pub fn build_client_options(config: &RemoteStoreConfig, allow_http: bool) -> ClientOptions {
    let mut opts = ClientOptions::new()
        .with_allow_http(allow_http)
        .with_allow_http2()
        .with_pool_max_idle_per_host(config.pool_max_idle_per_host)
        .with_http2_keep_alive_while_idle();

    if config.http2_keep_alive_secs > 0 {
        opts =
            opts.with_http2_keep_alive_interval(Duration::from_secs(config.http2_keep_alive_secs));
    }
    if config.request_timeout_ms > 0 {
        opts = opts.with_timeout(Duration::from_millis(config.request_timeout_ms));
    }
    opts
}

/// Constructs a multi-cloud `Arc<dyn ObjectStore>` and normalized object [`ObjectPath`] from `uri`.
///
/// Supports `s3://`, `s3a://`, `gs://`, `az://` / `abfs://` / `abfss://`, `http://` / `https://`
/// (including MinIO / Cloudflare R2 / Ceph via `config.custom_endpoint`), `memory://`, and `file://`.
pub fn build_object_store(
    uri: &str,
    config: &RemoteStoreConfig,
) -> OspreyResult<(Arc<dyn ObjectStore>, ObjectPath)> {
    let url = Url::parse(uri).map_err(|err| {
        OspreyError::Io(std::io::Error::new(std::io::ErrorKind::InvalidInput, err))
    })?;

    let (scheme, raw_path) = ObjectStoreScheme::parse(&url)
        .map_err(|err| map_object_store_err(object_store::Error::from(err)))?;
    let object_path = ObjectPath::parse(raw_path)
        .map_err(|err| map_object_store_err(object_store::Error::from(err)))?;

    let env_allow_http = std::env::var("AWS_ALLOW_HTTP")
        .ok()
        .is_some_and(|v| v.eq_ignore_ascii_case("true") || v == "1")
        || std::env::var("AWS_ENDPOINT_URL")
            .ok()
            .is_some_and(|ep| ep.starts_with("http://"));
    let allow_http = env_allow_http
        || url.scheme().eq_ignore_ascii_case("http")
        || config
            .custom_endpoint
            .as_deref()
            .is_some_and(|ep| ep.starts_with("http://"));
    let client_options = build_client_options(config, allow_http);
    let retry_config = build_retry_config(config);

    // If a custom S3-compatible endpoint is provided alongside http(s)://, route through AmazonS3Builder.
    if matches!(scheme, ObjectStoreScheme::Http) {
        if let Some(endpoint) = &config.custom_endpoint {
            let mut segments = url
                .path_segments()
                .ok_or_else(|| {
                    OspreyError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "missing bucket segment in custom-endpoint URL",
                    ))
                })?
                .filter(|s| !s.is_empty());
            let bucket = segments.next().ok_or_else(|| {
                OspreyError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "missing bucket segment in custom-endpoint URL",
                ))
            })?;
            let remaining: Vec<&str> = segments.collect();
            let key_path = ObjectPath::from_iter(remaining);

            let s3 = AmazonS3Builder::from_env()
                .with_endpoint(endpoint)
                .with_bucket_name(bucket)
                .with_allow_http(allow_http)
                .with_client_options(client_options)
                .with_retry(retry_config)
                .build()
                .map_err(map_object_store_err)?;
            return Ok((Arc::new(s3), key_path));
        }
    }

    let store: Arc<dyn ObjectStore> = match scheme {
        ObjectStoreScheme::Local => Arc::new(LocalFileSystem::new()),
        ObjectStoreScheme::Memory => Arc::new(InMemory::new()),
        ObjectStoreScheme::AmazonS3 => {
            let mut builder = AmazonS3Builder::from_env()
                .with_url(uri)
                .with_client_options(client_options)
                .with_retry(retry_config);
            if let Some(endpoint) = &config.custom_endpoint {
                builder = builder.with_endpoint(endpoint).with_allow_http(allow_http);
            }
            Arc::new(builder.build().map_err(map_object_store_err)?)
        }
        ObjectStoreScheme::GoogleCloudStorage => {
            let gcs = GoogleCloudStorageBuilder::from_env()
                .with_url(uri)
                .with_client_options(client_options)
                .with_retry(retry_config)
                .build()
                .map_err(map_object_store_err)?;
            Arc::new(gcs)
        }
        ObjectStoreScheme::MicrosoftAzure => {
            let mut builder = MicrosoftAzureBuilder::from_env()
                .with_url(uri)
                .with_client_options(client_options)
                .with_retry(retry_config);
            if let Some(endpoint) = &config.custom_endpoint {
                builder = builder.with_endpoint(endpoint.clone());
            }
            Arc::new(builder.build().map_err(map_object_store_err)?)
        }
        ObjectStoreScheme::Http => {
            let base_url = &url[..url::Position::BeforePath];
            let http = HttpBuilder::new()
                .with_url(base_url)
                .with_client_options(client_options)
                .with_retry(retry_config)
                .build()
                .map_err(map_object_store_err)?;
            Arc::new(http)
        }
        other => {
            return Err(OspreyError::Io(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                format!("unsupported object_store scheme: {other:?}"),
            )));
        }
    };

    Ok((store, object_path))
}

async fn execute_single_range_with_timeout<T: AsyncRangeTransport + ?Sized>(
    transport: &T,
    path: &ObjectPath,
    range: &Range<usize>,
    timeout_ms: u64,
) -> OspreyResult<Bytes> {
    let range_slice = std::slice::from_ref(range);
    let mut bufs = if timeout_ms > 0 {
        tokio::time::timeout(
            Duration::from_millis(timeout_ms),
            transport.get_ranges(path, range_slice),
        )
        .await
        .map_err(|_| {
            OspreyError::Io(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "object_store range-get timed out",
            ))
        })??
    } else {
        transport.get_ranges(path, range_slice).await?
    };

    if bufs.len() != 1 {
        return Err(OspreyError::BufferTooSmall {
            needed: 1,
            actual: bufs.len(),
        });
    }
    bufs.pop().ok_or(OspreyError::CorruptedFooter)
}

async fn execute_hedged_range_once<T: AsyncRangeTransport + ?Sized>(
    transport: &T,
    path: &ObjectPath,
    range: &Range<usize>,
    config: &RemoteStoreConfig,
) -> OspreyResult<Bytes> {
    if config.hedged_request_delay_ms == 0 {
        return execute_single_range_with_timeout(
            transport,
            path,
            range,
            config.request_timeout_ms,
        )
        .await;
    }

    let primary =
        execute_single_range_with_timeout(transport, path, range, config.request_timeout_ms);
    tokio::pin!(primary);

    let hedge_timer = tokio::time::sleep(Duration::from_millis(config.hedged_request_delay_ms));
    tokio::pin!(hedge_timer);

    tokio::select! {
        res = &mut primary => res,
        () = &mut hedge_timer => {
            // Primary exceeded P99 latency guard: launch speculative secondary request in parallel.
            let secondary = execute_single_range_with_timeout(
                transport,
                path,
                range,
                config.request_timeout_ms,
            );
            tokio::pin!(secondary);

            tokio::select! {
                res = &mut primary => match res {
                    Ok(bytes) => Ok(bytes),
                    Err(_) => secondary.await,
                },
                res = &mut secondary => match res {
                    Ok(bytes) => Ok(bytes),
                    Err(_) => primary.await,
                },
            }
        }
    }
}

async fn fetch_single_range_resilient<T: AsyncRangeTransport + ?Sized>(
    transport: &T,
    path: &ObjectPath,
    range: &Range<usize>,
    config: &RemoteStoreConfig,
    seed: u64,
) -> OspreyResult<Bytes> {
    let mut rng = SplitMix64::new(seed);
    let mut attempt = 0usize;

    loop {
        match execute_hedged_range_once(transport, path, range, config).await {
            Ok(bytes) => return Ok(bytes),
            Err(err) => {
                if !is_transient_osprey_error(&err) || attempt >= config.max_retries {
                    return Err(err);
                }
                let exp_cap = RETRY_BASE_BACKOFF_MS.saturating_mul(1_u64 << attempt.min(10));
                let jitter_ms = rng.next_u64() % (exp_cap + 1);
                if jitter_ms > 0 {
                    tokio::time::sleep(Duration::from_millis(jitter_ms)).await;
                }
                attempt += 1;
            }
        }
    }
}

/// Fetches all DP-coalesced byte windows concurrently over `tokio` with semaphore stream
/// bounding, speculative P99 hedged requests, and full-jitter exponential backoff retries.
pub async fn fetch_coalesced_ranges_resilient(
    store: &Arc<dyn ObjectStore>,
    path: &ObjectPath,
    coalesced: &[CoalescedRange],
    config: &RemoteStoreConfig,
) -> OspreyResult<Vec<Bytes>> {
    fetch_coalesced_ranges_with_transport(store, path, coalesced, config).await
}

/// Generic transport variant of [`fetch_coalesced_ranges_resilient`] for any [`AsyncRangeTransport`].
pub async fn fetch_coalesced_ranges_with_transport<T: AsyncRangeTransport + ?Sized>(
    transport: &T,
    path: &ObjectPath,
    coalesced: &[CoalescedRange],
    config: &RemoteStoreConfig,
) -> OspreyResult<Vec<Bytes>> {
    if coalesced.is_empty() {
        return Ok(Vec::new());
    }

    let mut byte_ranges = Vec::with_capacity(coalesced.len());
    for range in coalesced {
        let start = usize::try_from(range.start).map_err(|_| OspreyError::CorruptedFooter)?;
        let end = usize::try_from(range.end).map_err(|_| OspreyError::CorruptedFooter)?;
        byte_ranges.push(start..end);
    }

    let semaphore = Arc::new(Semaphore::new(config.max_concurrent_streams.max(1)));
    let futs = byte_ranges.iter().enumerate().map(|(idx, range)| {
        let sem = Arc::clone(&semaphore);
        let seed = 0x9e37_79b9_7f4a_7c15_u64
            ^ (idx as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
            ^ (range.start as u64);
        async move {
            let _permit = sem.acquire().await.map_err(|err| {
                OspreyError::Io(std::io::Error::new(std::io::ErrorKind::BrokenPipe, err))
            })?;
            fetch_single_range_resilient(transport, path, range, config, seed).await
        }
    });

    try_join_all(futs).await
}

/// Zero-copy slices compressed minipages out of coalesced [`Bytes`] buffers and
/// decompresses them in parallel via `rayon`, validating `page_xxh64` before Zstd.
///
/// For each [`CoalescedRange`] `r` (with `base_offset = r.start` and `buf = &coalesced_buffers[idx]`),
/// every covered interval `p` in `page_intervals[r.first_interval_idx .. +r.interval_count]`
/// is extracted via `buf.slice(rel_start..rel_end)` in $O(1)$ without copying intermediate gaps.
pub fn slice_and_decompress_coalesced(
    coalesced_ranges: &[CoalescedRange],
    coalesced_buffers: &[Bytes],
    page_intervals: &[PageByteInterval],
    directory: &[MinipageDirectoryEntry],
    zstd_dict: Option<&[u8]>,
) -> OspreyResult<Vec<(u32, Arc<DecodedMinipage>)>> {
    if coalesced_ranges.len() != coalesced_buffers.len() {
        return Err(OspreyError::BufferTooSmall {
            needed: coalesced_ranges.len(),
            actual: coalesced_buffers.len(),
        });
    }
    if coalesced_ranges.is_empty() {
        return Ok(Vec::new());
    }

    let total_pages: usize = coalesced_ranges
        .iter()
        .map(|r| usize::from(r.interval_count))
        .sum();
    let mut sliced_tasks: Vec<(u32, &MinipageDirectoryEntry, Bytes)> =
        Vec::with_capacity(total_pages);

    for (range, buf) in coalesced_ranges.iter().zip(coalesced_buffers.iter()) {
        let expected_len =
            usize::try_from(range.byte_len()).map_err(|_| OspreyError::CorruptedFooter)?;
        if buf.len() < expected_len {
            return Err(OspreyError::BufferTooSmall {
                needed: expected_len,
                actual: buf.len(),
            });
        }

        let first = usize::from(range.first_interval_idx);
        let count = usize::from(range.interval_count);
        let end_idx = first
            .checked_add(count)
            .ok_or(OspreyError::InvalidDimension { dim: first })?;
        let covered = page_intervals
            .get(first..end_idx)
            .ok_or(OspreyError::InvalidDimension { dim: end_idx })?;

        let base_offset = range.start;
        for interval in covered {
            if interval.start < base_offset
                || interval.end > range.end
                || interval.start > interval.end
            {
                return Err(OspreyError::CorruptedFooter);
            }

            let rel_start = usize::try_from(interval.start - base_offset)
                .map_err(|_| OspreyError::CorruptedFooter)?;
            let rel_end = usize::try_from(interval.end - base_offset)
                .map_err(|_| OspreyError::CorruptedFooter)?;

            let dir_idx = usize::try_from(interval.minipage_id)
                .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
            let entry = directory
                .get(dir_idx)
                .ok_or(OspreyError::InvalidDimension { dim: dir_idx })?;

            // O(1) zero-copy refcounted sub-slice; intermediate DP range gaps are discarded.
            let compressed_slice = buf.slice(rel_start..rel_end);
            sliced_tasks.push((interval.minipage_id, entry, compressed_slice));
        }
    }

    let prepared_dict = zstd_dict.map(DecoderDictionary::copy);

    let decoded_pages: OspreyResult<Vec<(u32, Arc<DecodedMinipage>)>> = sliced_tasks
        .into_par_iter()
        .map_init(
            || match &prepared_dict {
                Some(dict) => zstd::bulk::Decompressor::with_prepared_dictionary(dict),
                None => zstd::bulk::Decompressor::new(),
            },
            |decompressor_res, (minipage_id, entry, compressed_slice)| {
                let decompressor = decompressor_res.as_mut().map_err(|err| {
                    OspreyError::Io(std::io::Error::new(err.kind(), err.to_string()))
                })?;
                let decoded = verify_and_decompress_minipage_with(
                    compressed_slice.as_ref(),
                    entry,
                    None,
                    decompressor,
                )?;
                Ok((minipage_id, Arc::new(decoded)))
            },
        )
        .collect();

    decoded_pages
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;

    use half::f16;
    use object_store::PutPayload;

    use super::*;
    use crate::osv_slab::{OsvSlabReader, OsvVectorRecord, OsvWriter, OsvWriterConfig};
    use crate::range_planner::{
        extract_unique_page_intervals, plan_coalesced_ranges_stack, CandidatePageRef,
        RangePlannerConfig,
    };

    #[test]
    fn remote_store_config_defaults_match_specification() {
        let cfg = RemoteStoreConfig::default();
        assert_eq!(cfg.max_concurrent_streams, 64);
        assert_eq!(cfg.pool_max_idle_per_host, 32);
        assert_eq!(cfg.http2_keep_alive_secs, 30);
        assert_eq!(cfg.request_timeout_ms, 2_000);
        assert_eq!(cfg.hedged_request_delay_ms, 35);
        assert_eq!(cfg.max_retries, 3);
        assert_eq!(cfg.custom_endpoint, None);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Builds object_store HTTP/cloud client connectors
    fn build_object_store_parses_multi_cloud_and_local_uris() -> OspreyResult<()> {
        let cfg = RemoteStoreConfig::default();

        let (mem_store, mem_path) = build_object_store("memory:///osprey/seg0.osv", &cfg)?;
        assert_eq!(mem_path.as_ref(), "osprey/seg0.osv");
        let _ = mem_store;

        let (_local_store, local_path) = build_object_store("file:///tmp/osprey/seg0.osv", &cfg)?;
        assert_eq!(local_path.as_ref(), "tmp/osprey/seg0.osv");

        let (_s3_store, s3_path) = build_object_store("s3://my-bucket/idx/seg0.osv", &cfg)?;
        assert_eq!(s3_path.as_ref(), "idx/seg0.osv");

        let (_gs_store, gs_path) = build_object_store("gs://my-gcs-bucket/idx/seg0.osv", &cfg)?;
        assert_eq!(gs_path.as_ref(), "idx/seg0.osv");

        let (_az_store, az_path) = build_object_store(
            "abfss://my-container@myaccount.dfs.core.windows.net/idx/seg0.osv",
            &cfg,
        )?;
        assert_eq!(az_path.as_ref(), "idx/seg0.osv");

        let minio_cfg = RemoteStoreConfig {
            custom_endpoint: Some("http://127.0.0.1:9000".to_string()),
            ..RemoteStoreConfig::default()
        };
        let (_minio_store, minio_path) = build_object_store(
            "http://127.0.0.1:9000/osprey-bucket/idx/seg0.osv",
            &minio_cfg,
        )?;
        assert_eq!(minio_path.as_ref(), "idx/seg0.osv");

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(miri, ignore)] // Uses tokio multi-thread runtime
    async fn hedged_requests_and_full_jitter_retry_overcome_tail_spikes_and_transient_errors(
    ) -> OspreyResult<()> {
        struct FlakyTailTransport {
            inner: InMemory,
            calls: AtomicUsize,
            stall_first_ms: u64,
            fail_first_n: usize,
        }

        impl AsyncRangeTransport for FlakyTailTransport {
            fn get_ranges<'a>(
                &'a self,
                path: &'a ObjectPath,
                ranges: &'a [Range<usize>],
            ) -> BoxFuture<'a, OspreyResult<Vec<Bytes>>> {
                Box::pin(async move {
                    let call_idx = self.calls.fetch_add(1, Ordering::SeqCst);
                    if call_idx < self.fail_first_n {
                        return Err(OspreyError::Io(std::io::Error::new(
                            std::io::ErrorKind::ConnectionReset,
                            "simulated HTTP 503 connection reset",
                        )));
                    }
                    if call_idx == self.fail_first_n && self.stall_first_ms > 0 {
                        tokio::time::sleep(Duration::from_millis(self.stall_first_ms)).await;
                    }
                    ObjectStore::get_ranges(&self.inner, path, ranges)
                        .await
                        .map_err(map_object_store_err)
                })
            }
        }

        let mem = InMemory::new();
        let path = ObjectPath::from("seg0.osv");
        let data = Bytes::from_static(b"osprey-hedged-range-payload-0123456789");
        mem.put(&path, PutPayload::from_bytes(data.clone()))
            .await
            .map_err(map_object_store_err)?;

        // Case 1: Primary stalls 80 ms, hedge delay is 15 ms -> secondary hedge wins in < 45 ms.
        let transport_hedge = FlakyTailTransport {
            inner: mem,
            calls: AtomicUsize::new(0),
            stall_first_ms: 80,
            fail_first_n: 0,
        };
        let cfg_hedge = RemoteStoreConfig {
            hedged_request_delay_ms: 15,
            max_retries: 1,
            request_timeout_ms: 500,
            ..RemoteStoreConfig::default()
        };
        let ranges = [CoalescedRange {
            start: 7,
            end: 25,
            first_interval_idx: 0,
            interval_count: 1,
        }];

        let t0 = Instant::now();
        let fetched =
            fetch_coalesced_ranges_with_transport(&transport_hedge, &path, &ranges, &cfg_hedge)
                .await?;
        let elapsed = t0.elapsed();

        assert_eq!(fetched.len(), 1);
        assert_eq!(fetched[0], data.slice(7..25));
        assert_eq!(transport_hedge.calls.load(Ordering::SeqCst), 2);
        assert!(
            elapsed < Duration::from_millis(55),
            "Hedged request took {:?}, failing to beat 80ms primary tail stall",
            elapsed
        );

        // Case 2: 2 transient ConnectionReset failures recover cleanly via full-jitter retry.
        let mem2 = InMemory::new();
        mem2.put(&path, PutPayload::from_bytes(data.clone()))
            .await
            .map_err(map_object_store_err)?;
        let transport_retry = FlakyTailTransport {
            inner: mem2,
            calls: AtomicUsize::new(0),
            stall_first_ms: 0,
            fail_first_n: 2,
        };
        let cfg_retry = RemoteStoreConfig {
            hedged_request_delay_ms: 0,
            max_retries: 3,
            ..RemoteStoreConfig::default()
        };
        let fetched2 =
            fetch_coalesced_ranges_with_transport(&transport_retry, &path, &ranges, &cfg_retry)
                .await?;
        assert_eq!(fetched2[0], data.slice(7..25));
        assert_eq!(transport_retry.calls.load(Ordering::SeqCst), 3);

        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Invokes zstd-sys C FFI and rayon threadpool
    fn slice_and_decompress_coalesced_discards_gaps_zero_copy_and_validates_checksum(
    ) -> OspreyResult<()> {
        let orig_dim = 8usize;
        let mut records = Vec::new();
        let mut vecs = Vec::new();
        let mut payloads = Vec::new();

        for i in 0..32usize {
            let v: Vec<f16> = (0..orig_dim)
                .map(|d| f16::from_f32(((i * orig_dim + d) as f32) * 0.01))
                .collect();
            let p = format!(
                "{{\"doc\":{i},\"category\":\"cat_{}\",\"region\":\"us-east\"}}",
                i % 4
            )
            .into_bytes();
            vecs.push(v);
            payloads.push(p);
        }
        for i in 0..32usize {
            records.push(OsvVectorRecord {
                doc_id: 1_000 + i as u64,
                cluster_id: u32::try_from(i / 4).unwrap_or(0),
                vector_f16: &vecs[i],
                payload: &payloads[i],
            });
        }

        let cfg = OsvWriterConfig {
            minipage_target_bytes: 512,
            zstd_level: 3,
            enable_zstd_dict: true,
            max_dict_bytes: 16 * 1024,
        };
        let mut raw_slab = Vec::new();
        let report = OsvWriter::new(cfg).write_slab(&mut raw_slab, &records, orig_dim, None)?;
        assert!(report.directory.len() >= 6);

        let reader = OsvSlabReader::from_slice(&raw_slab)?;
        let directory = reader.directory();
        let zstd_dict = if reader.header().zstd_dict_len > 0 {
            let dict_start = usize::try_from(reader.header().zstd_dict_offset)
                .map_err(|_| OspreyError::CorruptedFooter)?;
            let dict_len = usize::try_from(reader.header().zstd_dict_len)
                .map_err(|_| OspreyError::CorruptedFooter)?;
            Some(&raw_slab[dict_start..dict_start + dict_len])
        } else {
            None
        };

        // Select non-contiguous minipages (skipping page 1 and page 3 to force DP range gaps).
        let candidates = [
            CandidatePageRef {
                doc_id: 1_000,
                minipage_id: 0,
                slot_idx: 0,
                approx_score_bits: 0,
            },
            CandidatePageRef {
                doc_id: 1_008,
                minipage_id: 2,
                slot_idx: 0,
                approx_score_bits: 0,
            },
            CandidatePageRef {
                doc_id: 1_016,
                minipage_id: 4,
                slot_idx: 0,
                approx_score_bits: 0,
            },
            CandidatePageRef {
                doc_id: 1_020,
                minipage_id: 5,
                slot_idx: 0,
                approx_score_bits: 0,
            },
        ];

        let mut intervals = [PageByteInterval::default(); 16];
        let k = extract_unique_page_intervals(&candidates, directory, &mut intervals)?;
        assert_eq!(k, 4);

        let mut coalesced = [CoalescedRange::default(); 16];
        let num_ranges = plan_coalesced_ranges_stack(
            &intervals,
            k,
            &RangePlannerConfig::default(),
            &mut coalesced,
        );
        assert_eq!(num_ranges, 1);
        assert!(coalesced[0].byte_len() > intervals[0].byte_len() + intervals[1].byte_len());

        // Simulate one HTTP Range-GET buffer containing both requested pages AND skipped gap bytes.
        let r_start =
            usize::try_from(coalesced[0].start).map_err(|_| OspreyError::CorruptedFooter)?;
        let r_end = usize::try_from(coalesced[0].end).map_err(|_| OspreyError::CorruptedFooter)?;
        let coalesced_bufs = vec![Bytes::copy_from_slice(&raw_slab[r_start..r_end])];

        let decoded = slice_and_decompress_coalesced(
            &coalesced[..num_ranges],
            &coalesced_bufs,
            &intervals[..k],
            directory,
            zstd_dict,
        )?;
        assert_eq!(decoded.len(), 4);

        for (minipage_id, page) in &decoded {
            let expected = reader.decompress_minipage(*minipage_id)?;
            assert_eq!(page.doc_ids(), expected.doc_ids());
            assert_eq!(page.cluster_id(), expected.cluster_id());
        }

        // Corrupt a single byte in the first minipage inside the coalesced buffer -> ChecksumMismatch.
        let mut corrupted = raw_slab[r_start..r_end].to_vec();
        corrupted[0] ^= 0x01;
        let err = slice_and_decompress_coalesced(
            &coalesced[..num_ranges],
            &[Bytes::from(corrupted)],
            &intervals[..k],
            directory,
            zstd_dict,
        );
        assert!(matches!(err, Err(OspreyError::ChecksumMismatch { .. })));
        Ok(())
    }

    #[test]
    fn transient_http_429_500_503_errors_are_classified_for_retry() {
        for status_msg in [
            "HTTP 429 Too Many Requests",
            "HTTP 500 Internal Server Error",
            "HTTP 503 Service Unavailable",
        ] {
            let obj_err = object_store::Error::Generic {
                store: "S3",
                source: Box::new(std::io::Error::other(status_msg)),
            };
            let mapped = map_object_store_err(obj_err);
            assert!(
                is_transient_osprey_error(&mapped),
                "expected '{status_msg}' to be retried as transient"
            );
        }

        let cfg = RemoteStoreConfig::default();
        let retry = build_retry_config(&cfg);
        assert_eq!(retry.max_retries, cfg.max_retries);
    }
}
