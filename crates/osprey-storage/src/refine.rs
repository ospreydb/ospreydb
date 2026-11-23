//! Exact `f16` refinement stage combining `MinipageLruCache`, DP range range coalescing,
//! resilient async `object_store` fetching, zero-copy `Bytes::slice` + parallel Zstd
//! decompression, and `compute_exact_f16_dot`.

use std::collections::{HashMap, HashSet};
use std::fs::File;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use memmap2::Mmap;
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;
use osprey_core::format::osv::OSV_FLAG_ZSTD_ENABLED;
use osprey_core::{AlignedBuffer, MinipageDirectoryEntry, OspreyError, OspreyResult, OsvHeader};
use xxhash_rust::xxh64::xxh64;
use zstd::dict::DecoderDictionary;

use crate::osv_slab::{
    parse_minipage, verify_and_decompress_minipage_with, DecodedMinipage, MinipageView,
};
use crate::page_cache::MinipageLruCache;
use crate::range_planner::{
    extract_unique_page_intervals, plan_coalesced_ranges_stack, CandidatePageRef, CoalescedRange,
    PageByteInterval, RangePlannerConfig, DEFAULT_COALESCE_MAX_K,
};
use crate::remote_store::{
    fetch_coalesced_ranges_with_transport, slice_and_decompress_coalesced, AsyncRangeTransport,
    RemoteStoreConfig,
};

/// Final exact-refined search hit returned by [`RefineStage`].
#[derive(Clone, Debug, PartialEq)]
pub struct RefinedSearchResult {
    /// Global logical document identifier.
    pub doc_id: u64,
    /// Exact `f32`-accumulated inner product $\langle q_{\text{f32}}, x_{\text{f16}} \rangle$.
    pub exact_score: f32,
    /// `.osv` minipage index containing the document vector and payload.
    pub minipage_id: u32,
    /// Zero-based slot index inside the minipage.
    pub slot_idx: u16,
    /// UTF-8 JSON/metadata payload bytes extracted only for `Top-K` survivors.
    pub payload_json: Vec<u8>,
}

/// Telemetry emitted per query by [`RefineStage`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct RefineStats {
    /// Total unique minipages requested by the `Top-M` candidate set.
    pub raw_page_requests: usize,
    /// Number of coalesced `Range-GET` requests issued for cache-missed minipages.
    pub coalesced_range_gets: usize,
    /// Total compressed + gap bytes fetched across coalesced `Range-GET` windows.
    pub bytes_fetched: u64,
    /// Number of unique minipages satisfied directly from [`MinipageLruCache`].
    pub cache_hits: usize,
    /// Total time spent in DP range interval extraction and DP planning (microseconds).
    pub range_plan_time_us: u64,
    /// Time spent looking up minipages in [`MinipageLruCache`] (microseconds).
    pub cache_lookup_us: u64,
    /// Time spent reading ranges from object storage or local `pread` (microseconds).
    pub disk_read_us: u64,
    /// Time spent verifying checksums and decompressing Zstd frames (microseconds).
    pub zstd_decode_us: u64,
    /// Time spent computing exact `f16` dot products and selecting `Top-K` (microseconds).
    pub f16_score_us: u64,
}

/// Direct local filesystem backing for `file://` `.osv` slabs, bypassing async `LocalFileSystem`.
#[derive(Clone, Debug)]
pub enum LocalOsvBacking {
    /// Uncompressed `.osv` mapped via `memmap2::Mmap` for zero-copy direct `f16` scoring.
    UncompressedMmap(Arc<Mmap>),
    /// Zstd-compressed `.osv` file descriptor for direct synchronous `pread` (`FileExt::read_exact_at`).
    CompressedFile(Arc<File>),
}

/// Stage-3 exact `f16` refinement coordinator over cached and object-stored `.osv` minipages.
#[derive(Clone)]
pub struct RefineStage {
    slab_id: u16,
    range_planner_config: RangePlannerConfig,
    zstd_dict: Option<Vec<u8>>,
    prepared_dict: Option<Arc<DecoderDictionary<'static>>>,
    zstd_enabled: bool,
    local_backing: Option<LocalOsvBacking>,
}

impl std::fmt::Debug for RefineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RefineStage")
            .field("slab_id", &self.slab_id)
            .field("range_planner_config", &self.range_planner_config)
            .field("has_zstd_dict", &self.prepared_dict.is_some())
            .field("zstd_enabled", &self.zstd_enabled)
            .field("local_backing", &self.local_backing.is_some())
            .finish()
    }
}

impl RefineStage {
    /// Creates a [`RefineStage`] targeting `slab_id` with `range_planner_config` and optional Zstd dict.
    pub fn new(
        slab_id: u16,
        range_planner_config: RangePlannerConfig,
        zstd_dict: Option<Vec<u8>>,
    ) -> Self {
        let prepared_dict = zstd_dict
            .as_deref()
            .map(|d| Arc::new(DecoderDictionary::copy(d)));
        Self {
            slab_id,
            range_planner_config,
            zstd_dict,
            prepared_dict,
            zstd_enabled: true,
            local_backing: None,
        }
    }

    /// Configures whether the backing `.osv` uses Zstd compression (`OSV_FLAG_ZSTD_ENABLED`).
    #[must_use]
    pub const fn with_zstd_enabled(mut self, enabled: bool) -> Self {
        self.zstd_enabled = enabled;
        self
    }

    /// Attaches a direct local `file://` backing (`mmap` or `pread`) to bypass async `LocalFileSystem`.
    #[must_use]
    pub fn with_local_backing(mut self, backing: LocalOsvBacking) -> Self {
        self.local_backing = Some(backing);
        self
    }

    /// Opens a local `.osv` file with the optimal fast-path (`UncompressedMmap` when
    /// `OSV_FLAG_ZSTD_ENABLED` is unset, `CompressedFile` for `pread` when Zstd is enabled).
    pub fn open_local_path(
        slab_id: u16,
        range_planner_config: RangePlannerConfig,
        osv_header: &OsvHeader,
        zstd_dict: Option<Vec<u8>>,
        path: &Path,
    ) -> OspreyResult<Self> {
        let zstd_enabled = (osv_header.flags & OSV_FLAG_ZSTD_ENABLED) != 0;
        let file = File::open(path)?;
        let backing = if zstd_enabled {
            LocalOsvBacking::CompressedFile(Arc::new(file))
        } else {
            // SAFETY: `.osv` segment files are immutable for the lifetime of the read-only mapping.
            let mmap = unsafe { Mmap::map(&file)? };
            LocalOsvBacking::UncompressedMmap(Arc::new(mmap))
        };
        Ok(Self::new(slab_id, range_planner_config, zstd_dict)
            .with_zstd_enabled(zstd_enabled)
            .with_local_backing(backing))
    }

    /// Returns the slab identifier used to namespace [`MinipageLruCache`] lookups.
    #[inline]
    pub const fn slab_id(&self) -> u16 {
        self.slab_id
    }

    /// Returns the DP range DP cost configuration.
    #[inline]
    pub const fn range_planner_config(&self) -> &RangePlannerConfig {
        &self.range_planner_config
    }

    /// Asynchronous Stage-3 refinement backed by `object_store::ObjectStore` and
    /// [`fetch_coalesced_ranges_with_transport`] (hedged requests + full-jitter retries),
    /// or direct local `mmap`/`pread` fast-path when `local_backing` is configured.
    #[allow(clippy::too_many_arguments)]
    pub async fn refine_async(
        &self,
        query_f32: &[f32],
        candidates: &[CandidatePageRef],
        top_k: usize,
        directory: &[MinipageDirectoryEntry],
        cache: &MinipageLruCache,
        store: &Arc<dyn ObjectStore>,
        object_path: &ObjectPath,
        remote_config: &RemoteStoreConfig,
    ) -> OspreyResult<(Vec<RefinedSearchResult>, RefineStats)> {
        if candidates.is_empty() || top_k == 0 {
            return Ok((Vec::new(), RefineStats::default()));
        }

        match &self.local_backing {
            Some(LocalOsvBacking::UncompressedMmap(mmap)) => {
                return refine_local_uncompressed_mmap(
                    query_f32,
                    candidates,
                    top_k,
                    directory,
                    mmap.as_ref(),
                );
            }
            Some(LocalOsvBacking::CompressedFile(file)) => {
                return refine_local_compressed_file(
                    self.slab_id,
                    query_f32,
                    candidates,
                    top_k,
                    directory,
                    &self.range_planner_config,
                    self.prepared_dict.as_deref(),
                    cache,
                    file.as_ref(),
                );
            }
            None => {}
        }

        self.refine_async_with_transport(
            query_f32,
            candidates,
            top_k,
            directory,
            cache,
            store,
            object_path,
            remote_config,
        )
        .await
    }

    /// Generic async refinement over any [`AsyncRangeTransport`].
    #[allow(clippy::too_many_arguments)]
    pub async fn refine_async_with_transport<T: AsyncRangeTransport + ?Sized>(
        &self,
        query_f32: &[f32],
        candidates: &[CandidatePageRef],
        top_k: usize,
        directory: &[MinipageDirectoryEntry],
        cache: &MinipageLruCache,
        transport: &T,
        object_path: &ObjectPath,
        remote_config: &RemoteStoreConfig,
    ) -> OspreyResult<(Vec<RefinedSearchResult>, RefineStats)> {
        if candidates.is_empty() || top_k == 0 {
            return Ok((Vec::new(), RefineStats::default()));
        }

        let t_cache = Instant::now();
        let (mut resolved_pages, miss_candidates, mut stats) =
            partition_cache_hits_and_misses(self.slab_id, candidates, cache);
        stats.cache_lookup_us = u64::try_from(t_cache.elapsed().as_micros()).unwrap_or(u64::MAX);

        if !miss_candidates.is_empty() {
            for chunk in miss_candidates.chunks(DEFAULT_COALESCE_MAX_K) {
                let mut intervals = [PageByteInterval::default(); DEFAULT_COALESCE_MAX_K];
                let mut coalesced = [CoalescedRange::default(); DEFAULT_COALESCE_MAX_K];
                let (unique_k, num_coalesced) = plan_coalesced_chunk(
                    chunk,
                    directory,
                    &self.range_planner_config,
                    &mut intervals,
                    &mut coalesced,
                    &mut stats,
                )?;
                if num_coalesced == 0 {
                    continue;
                }

                let active_ranges = &coalesced[..num_coalesced];
                let t_disk = Instant::now();
                let buffers = fetch_coalesced_ranges_with_transport(
                    transport,
                    object_path,
                    active_ranges,
                    remote_config,
                )
                .await?;
                stats.disk_read_us = stats.disk_read_us.saturating_add(
                    u64::try_from(t_disk.elapsed().as_micros()).unwrap_or(u64::MAX),
                );

                let t_decode = Instant::now();
                decode_and_cache_chunk(
                    self.slab_id,
                    active_ranges,
                    &buffers,
                    &intervals[..unique_k],
                    directory,
                    self.zstd_enabled,
                    self.zstd_dict.as_deref(),
                    cache,
                    &mut resolved_pages,
                )?;
                stats.zstd_decode_us = stats.zstd_decode_us.saturating_add(
                    u64::try_from(t_decode.elapsed().as_micros()).unwrap_or(u64::MAX),
                );
            }
        }

        let t_score = Instant::now();
        let results = score_and_extract_topk(query_f32, candidates, top_k, &resolved_pages)?;
        stats.f16_score_us = u64::try_from(t_score.elapsed().as_micros()).unwrap_or(u64::MAX);
        Ok((results, stats))
    }

    /// Synchronous Stage-3 refinement using a caller-supplied range fetcher callback.
    pub fn refine_with_fetcher<F>(
        &self,
        query_f32: &[f32],
        candidates: &[CandidatePageRef],
        top_k: usize,
        directory: &[MinipageDirectoryEntry],
        cache: &MinipageLruCache,
        mut fetch_ranges: F,
    ) -> OspreyResult<(Vec<RefinedSearchResult>, RefineStats)>
    where
        F: FnMut(&[CoalescedRange]) -> OspreyResult<Vec<Bytes>>,
    {
        if candidates.is_empty() || top_k == 0 {
            return Ok((Vec::new(), RefineStats::default()));
        }

        let t_cache = Instant::now();
        let (mut resolved_pages, miss_candidates, mut stats) =
            partition_cache_hits_and_misses(self.slab_id, candidates, cache);
        stats.cache_lookup_us = u64::try_from(t_cache.elapsed().as_micros()).unwrap_or(u64::MAX);

        if !miss_candidates.is_empty() {
            for chunk in miss_candidates.chunks(DEFAULT_COALESCE_MAX_K) {
                let mut intervals = [PageByteInterval::default(); DEFAULT_COALESCE_MAX_K];
                let mut coalesced = [CoalescedRange::default(); DEFAULT_COALESCE_MAX_K];
                let (unique_k, num_coalesced) = plan_coalesced_chunk(
                    chunk,
                    directory,
                    &self.range_planner_config,
                    &mut intervals,
                    &mut coalesced,
                    &mut stats,
                )?;
                if num_coalesced == 0 {
                    continue;
                }

                let active_ranges = &coalesced[..num_coalesced];
                let t_disk = Instant::now();
                let buffers = fetch_ranges(active_ranges)?;
                stats.disk_read_us = stats.disk_read_us.saturating_add(
                    u64::try_from(t_disk.elapsed().as_micros()).unwrap_or(u64::MAX),
                );

                let t_decode = Instant::now();
                decode_and_cache_chunk(
                    self.slab_id,
                    active_ranges,
                    &buffers,
                    &intervals[..unique_k],
                    directory,
                    self.zstd_enabled,
                    self.zstd_dict.as_deref(),
                    cache,
                    &mut resolved_pages,
                )?;
                stats.zstd_decode_us = stats.zstd_decode_us.saturating_add(
                    u64::try_from(t_decode.elapsed().as_micros()).unwrap_or(u64::MAX),
                );
            }
        }

        let t_score = Instant::now();
        let results = score_and_extract_topk(query_f32, candidates, top_k, &resolved_pages)?;
        stats.f16_score_us = u64::try_from(t_score.elapsed().as_micros()).unwrap_or(u64::MAX);
        Ok((results, stats))
    }

    /// Convenience wrapper over [`Self::refine_async`] accepting Stage-1
    /// [`osprey_core::ScoredCandidate`] slices directly via `From<ScoredCandidate> for CandidatePageRef`.
    #[allow(clippy::too_many_arguments)]
    pub async fn refine_scored_async(
        &self,
        query_f32: &[f32],
        candidates: &[osprey_core::ScoredCandidate],
        top_k: usize,
        directory: &[MinipageDirectoryEntry],
        cache: &MinipageLruCache,
        store: &Arc<dyn ObjectStore>,
        object_path: &ObjectPath,
        remote_config: &RemoteStoreConfig,
    ) -> OspreyResult<(Vec<RefinedSearchResult>, RefineStats)> {
        let page_refs: Vec<CandidatePageRef> = {
            let mut refs = Vec::with_capacity(candidates.len());
            refs.extend(candidates.iter().map(CandidatePageRef::from));
            refs
        };
        self.refine_async(
            query_f32,
            &page_refs,
            top_k,
            directory,
            cache,
            store,
            object_path,
            remote_config,
        )
        .await
    }

    /// Convenience wrapper over [`Self::refine_with_fetcher`] accepting Stage-1
    /// [`osprey_core::ScoredCandidate`] slices directly via `From<ScoredCandidate> for CandidatePageRef`.
    pub fn refine_scored_with_fetcher<F>(
        &self,
        query_f32: &[f32],
        candidates: &[osprey_core::ScoredCandidate],
        top_k: usize,
        directory: &[MinipageDirectoryEntry],
        cache: &MinipageLruCache,
        fetch_ranges: F,
    ) -> OspreyResult<(Vec<RefinedSearchResult>, RefineStats)>
    where
        F: FnMut(&[CoalescedRange]) -> OspreyResult<Vec<Bytes>>,
    {
        let page_refs: Vec<CandidatePageRef> = {
            let mut refs = Vec::with_capacity(candidates.len());
            refs.extend(candidates.iter().map(CandidatePageRef::from));
            refs
        };
        self.refine_with_fetcher(query_f32, &page_refs, top_k, directory, cache, fetch_ranges)
    }

    /// Stage-4 `LUT-Rank` ColBERT reranker over Stage-3 [`RefinedSearchResult`] survivors,
    /// returning `(result, maxsim_score)` sorted descending by `maxsim_score` and truncated to `top_k`.
    pub fn rerank_with_osr<K: osprey_core::LutRankKernel + ?Sized>(
        refined: &[RefinedSearchResult],
        osr: &osprey_core::OsrStore<'_>,
        query_lut: &osprey_core::QuantizedQueryLut,
        top_k: usize,
        simd: &K,
    ) -> OspreyResult<Vec<(RefinedSearchResult, f32)>> {
        if refined.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }
        let mut scored = Vec::with_capacity(refined.len());
        for (idx, item) in refined.iter().enumerate() {
            if let Some(next_item) = refined.get(idx + 1) {
                osr.prefetch_doc_id(next_item.doc_id, simd);
            }
            let maxsim = osr
                .score_doc_id_maxsim(item.doc_id, query_lut, simd)?
                .unwrap_or(f32::NEG_INFINITY);
            scored.push((item.clone(), maxsim));
        }
        scored.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| a.0.doc_id.cmp(&b.0.doc_id))
        });
        scored.truncate(top_k);
        Ok(scored)
    }
}

fn partition_cache_hits_and_misses(
    slab_id: u16,
    candidates: &[CandidatePageRef],
    cache: &MinipageLruCache,
) -> (
    HashMap<u32, Arc<DecodedMinipage>>,
    Vec<CandidatePageRef>,
    RefineStats,
) {
    let mut stats = RefineStats::default();
    let mut resolved_pages: HashMap<u32, Arc<DecodedMinipage>> =
        HashMap::with_capacity(candidates.len());
    let mut miss_candidates: Vec<CandidatePageRef> = Vec::with_capacity(candidates.len());
    let mut seen_misses: HashSet<u32> = HashSet::with_capacity(candidates.len());

    for &cand in candidates {
        if resolved_pages.contains_key(&cand.minipage_id) || seen_misses.contains(&cand.minipage_id)
        {
            continue;
        }
        stats.raw_page_requests += 1;

        if let Some(cached_page) = cache.get(slab_id, cand.minipage_id) {
            stats.cache_hits += 1;
            resolved_pages.insert(cand.minipage_id, cached_page);
        } else {
            seen_misses.insert(cand.minipage_id);
            miss_candidates.push(cand);
        }
    }

    (resolved_pages, miss_candidates, stats)
}

fn plan_coalesced_chunk(
    chunk: &[CandidatePageRef],
    directory: &[MinipageDirectoryEntry],
    range_planner_config: &RangePlannerConfig,
    intervals: &mut [PageByteInterval; DEFAULT_COALESCE_MAX_K],
    coalesced: &mut [CoalescedRange; DEFAULT_COALESCE_MAX_K],
    stats: &mut RefineStats,
) -> OspreyResult<(usize, usize)> {
    let plan_start = Instant::now();
    let unique_k = extract_unique_page_intervals(chunk, directory, intervals)?;
    let num_coalesced =
        plan_coalesced_ranges_stack(intervals, unique_k, range_planner_config, coalesced);
    let plan_us = u64::try_from(plan_start.elapsed().as_micros()).unwrap_or(u64::MAX);
    stats.range_plan_time_us = stats.range_plan_time_us.saturating_add(plan_us);

    if num_coalesced > 0 {
        stats.coalesced_range_gets += num_coalesced;
        for range in &coalesced[..num_coalesced] {
            stats.bytes_fetched = stats.bytes_fetched.saturating_add(range.byte_len());
        }
    }

    Ok((unique_k, num_coalesced))
}

#[allow(clippy::too_many_arguments)]
fn decode_and_cache_chunk(
    slab_id: u16,
    active_ranges: &[CoalescedRange],
    buffers: &[Bytes],
    page_intervals: &[PageByteInterval],
    directory: &[MinipageDirectoryEntry],
    zstd_enabled: bool,
    zstd_dict: Option<&[u8]>,
    cache: &MinipageLruCache,
    resolved_pages: &mut HashMap<u32, Arc<DecodedMinipage>>,
) -> OspreyResult<()> {
    let decoded_pages = if zstd_enabled {
        slice_and_decompress_coalesced(
            active_ranges,
            buffers,
            page_intervals,
            directory,
            zstd_dict,
        )?
    } else {
        decode_uncompressed_coalesced(active_ranges, buffers, page_intervals, directory)?
    };

    for (minipage_id, page_arc) in decoded_pages {
        cache.insert(slab_id, minipage_id, Arc::clone(&page_arc));
        resolved_pages.insert(minipage_id, page_arc);
    }
    Ok(())
}

fn decode_uncompressed_coalesced(
    coalesced_ranges: &[CoalescedRange],
    coalesced_buffers: &[Bytes],
    page_intervals: &[PageByteInterval],
    directory: &[MinipageDirectoryEntry],
) -> OspreyResult<Vec<(u32, Arc<DecodedMinipage>)>> {
    if coalesced_ranges.len() != coalesced_buffers.len() {
        return Err(OspreyError::BufferTooSmall {
            needed: coalesced_ranges.len(),
            actual: coalesced_buffers.len(),
        });
    }
    let mut out = Vec::new();
    for (range, buf) in coalesced_ranges.iter().zip(coalesced_buffers.iter()) {
        let first = usize::from(range.first_interval_idx);
        let count = usize::from(range.interval_count);
        let covered = page_intervals
            .get(first..first + count)
            .ok_or(OspreyError::InvalidDimension { dim: first + count })?;
        for interval in covered {
            let rel_start = usize::try_from(interval.start - range.start)
                .map_err(|_| OspreyError::CorruptedFooter)?;
            let rel_end = usize::try_from(interval.end - range.start)
                .map_err(|_| OspreyError::CorruptedFooter)?;
            let dir_idx = usize::try_from(interval.minipage_id)
                .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
            let entry = directory
                .get(dir_idx)
                .ok_or(OspreyError::InvalidDimension { dim: dir_idx })?;
            let raw = buf
                .get(rel_start..rel_end)
                .ok_or(OspreyError::CorruptedFooter)?;
            let computed = xxh64(raw, 0);
            if computed != entry.page_xxh64 {
                return Err(OspreyError::ChecksumMismatch {
                    expected: entry.page_xxh64,
                    computed,
                });
            }
            let aligned = AlignedBuffer::<u8>::from_slice(raw)?;
            let decoded = DecodedMinipage::from_aligned_buffer(aligned)?;
            if decoded.cluster_id() != entry.cluster_id
                || decoded.vector_count() != entry.vector_count
            {
                return Err(OspreyError::CorruptedFooter);
            }
            out.push((interval.minipage_id, Arc::new(decoded)));
        }
    }
    Ok(out)
}

fn refine_local_uncompressed_mmap(
    query_f32: &[f32],
    candidates: &[CandidatePageRef],
    top_k: usize,
    directory: &[MinipageDirectoryEntry],
    mmap: &[u8],
) -> OspreyResult<(Vec<RefinedSearchResult>, RefineStats)> {
    let mut stats = RefineStats {
        raw_page_requests: candidates.len(),
        ..RefineStats::default()
    };
    let t_score = Instant::now();

    struct LocalScored<'a> {
        doc_id: u64,
        exact_score: f32,
        minipage_id: u32,
        slot_idx: u16,
        view: MinipageView<'a>,
    }

    let mut scored = Vec::with_capacity(candidates.len());
    for cand in candidates {
        let dir_idx = usize::try_from(cand.minipage_id)
            .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
        let entry = directory
            .get(dir_idx)
            .ok_or(OspreyError::InvalidDimension { dim: dir_idx })?;
        let start = usize::try_from(entry.byte_offset).map_err(|_| OspreyError::CorruptedFooter)?;
        let len =
            usize::try_from(entry.compressed_size).map_err(|_| OspreyError::CorruptedFooter)?;
        let end = start.checked_add(len).ok_or(OspreyError::CorruptedFooter)?;
        let page_bytes = mmap.get(start..end).ok_or(OspreyError::BufferTooSmall {
            needed: end,
            actual: mmap.len(),
        })?;
        let view = parse_minipage(page_bytes)?;
        let exact_score = view.compute_exact_f16_dot(usize::from(cand.slot_idx), query_f32)?;
        scored.push(LocalScored {
            doc_id: cand.doc_id,
            exact_score,
            minipage_id: cand.minipage_id,
            slot_idx: cand.slot_idx,
            view,
        });
    }

    let cmp = |a: &LocalScored<'_>, b: &LocalScored<'_>| {
        b.exact_score
            .total_cmp(&a.exact_score)
            .then_with(|| a.doc_id.cmp(&b.doc_id))
    };
    let final_len = top_k.min(scored.len());
    if scored.len() > final_len && final_len > 0 {
        scored.select_nth_unstable_by(final_len, cmp);
        scored.truncate(final_len);
    }
    scored.sort_unstable_by(cmp);

    let mut results = Vec::with_capacity(final_len);
    for item in &scored[..final_len] {
        let payload_json = item.view.payload(usize::from(item.slot_idx))?.to_vec();
        results.push(RefinedSearchResult {
            doc_id: item.doc_id,
            exact_score: item.exact_score,
            minipage_id: item.minipage_id,
            slot_idx: item.slot_idx,
            payload_json,
        });
    }

    stats.f16_score_us = u64::try_from(t_score.elapsed().as_micros()).unwrap_or(u64::MAX);
    Ok((results, stats))
}

#[allow(clippy::too_many_arguments)]
fn refine_local_compressed_file(
    slab_id: u16,
    query_f32: &[f32],
    candidates: &[CandidatePageRef],
    top_k: usize,
    directory: &[MinipageDirectoryEntry],
    range_planner_config: &RangePlannerConfig,
    prepared_dict: Option<&DecoderDictionary<'static>>,
    cache: &MinipageLruCache,
    file: &File,
) -> OspreyResult<(Vec<RefinedSearchResult>, RefineStats)> {
    let t_cache = Instant::now();
    let (mut resolved_pages, miss_candidates, mut stats) =
        partition_cache_hits_and_misses(slab_id, candidates, cache);
    stats.cache_lookup_us = u64::try_from(t_cache.elapsed().as_micros()).unwrap_or(u64::MAX);

    if !miss_candidates.is_empty() {
        let mut read_buf: Vec<u8> = Vec::new();
        let mut decompressor = match prepared_dict {
            Some(dict) => zstd::bulk::Decompressor::with_prepared_dictionary(dict)?,
            None => zstd::bulk::Decompressor::new()?,
        };

        for chunk in miss_candidates.chunks(DEFAULT_COALESCE_MAX_K) {
            let mut intervals = [PageByteInterval::default(); DEFAULT_COALESCE_MAX_K];
            let mut coalesced = [CoalescedRange::default(); DEFAULT_COALESCE_MAX_K];
            let (unique_k, num_coalesced) = plan_coalesced_chunk(
                chunk,
                directory,
                range_planner_config,
                &mut intervals,
                &mut coalesced,
                &mut stats,
            )?;
            if num_coalesced == 0 {
                continue;
            }

            let active_ranges = &coalesced[..num_coalesced];
            let page_intervals = &intervals[..unique_k];

            for range in active_ranges {
                let range_len =
                    usize::try_from(range.byte_len()).map_err(|_| OspreyError::CorruptedFooter)?;
                if read_buf.len() < range_len {
                    read_buf.resize(range_len, 0);
                }
                let t_disk = Instant::now();
                read_exact_at_compat(file, &mut read_buf[..range_len], range.start)?;
                stats.disk_read_us = stats.disk_read_us.saturating_add(
                    u64::try_from(t_disk.elapsed().as_micros()).unwrap_or(u64::MAX),
                );

                let t_decode = Instant::now();
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

                    let decoded = verify_and_decompress_minipage_with(
                        &read_buf[rel_start..rel_end],
                        entry,
                        None,
                        &mut decompressor,
                    )?;
                    let page_arc = Arc::new(decoded);
                    cache.insert(slab_id, interval.minipage_id, Arc::clone(&page_arc));
                    resolved_pages.insert(interval.minipage_id, page_arc);
                }
                stats.zstd_decode_us = stats.zstd_decode_us.saturating_add(
                    u64::try_from(t_decode.elapsed().as_micros()).unwrap_or(u64::MAX),
                );
            }
        }
    }

    let t_score = Instant::now();
    let results = score_and_extract_topk(query_f32, candidates, top_k, &resolved_pages)?;
    stats.f16_score_us = u64::try_from(t_score.elapsed().as_micros()).unwrap_or(u64::MAX);
    Ok((results, stats))
}

#[cfg(unix)]
fn read_exact_at_compat(file: &File, buf: &mut [u8], offset: u64) -> OspreyResult<()> {
    file.read_exact_at(buf, offset)?;
    Ok(())
}

#[cfg(not(unix))]
fn read_exact_at_compat(file: &File, buf: &mut [u8], offset: u64) -> OspreyResult<()> {
    use std::io::{Read, Seek, SeekFrom};
    let mut cloned = file.try_clone()?;
    cloned.seek(SeekFrom::Start(offset))?;
    cloned.read_exact(buf)?;
    Ok(())
}

fn score_and_extract_topk(
    query_f32: &[f32],
    candidates: &[CandidatePageRef],
    top_k: usize,
    resolved_pages: &HashMap<u32, Arc<DecodedMinipage>>,
) -> OspreyResult<Vec<RefinedSearchResult>> {
    struct ScoredCandidate<'a> {
        doc_id: u64,
        exact_score: f32,
        minipage_id: u32,
        slot_idx: u16,
        page: &'a Arc<DecodedMinipage>,
    }

    let mut scored: Vec<ScoredCandidate<'_>> = Vec::with_capacity(candidates.len());
    for cand in candidates {
        let page =
            resolved_pages
                .get(&cand.minipage_id)
                .ok_or_else(|| OspreyError::InvalidDimension {
                    dim: usize::try_from(cand.minipage_id).unwrap_or(usize::MAX),
                })?;

        if cand.slot_idx >= page.vector_count() {
            return Err(OspreyError::InvalidDimension {
                dim: usize::from(cand.slot_idx),
            });
        }
        if query_f32.len() != usize::from(page.orig_dim()) {
            return Err(OspreyError::InvalidDimension {
                dim: query_f32.len(),
            });
        }

        let vec_f16 = page.vector_f16(cand.slot_idx)?;
        let exact_score = osprey_simd::dot_f32_f16(query_f32, vec_f16);
        scored.push(ScoredCandidate {
            doc_id: cand.doc_id,
            exact_score,
            minipage_id: cand.minipage_id,
            slot_idx: cand.slot_idx,
            page,
        });
    }

    let cmp = |a: &ScoredCandidate<'_>, b: &ScoredCandidate<'_>| {
        b.exact_score
            .total_cmp(&a.exact_score)
            .then_with(|| a.doc_id.cmp(&b.doc_id))
    };
    let final_len = top_k.min(scored.len());
    if scored.len() > final_len && final_len > 0 {
        scored.select_nth_unstable_by(final_len, cmp);
        scored.truncate(final_len);
    }
    scored.sort_unstable_by(cmp);

    let mut results = Vec::with_capacity(final_len);
    for item in &scored[..final_len] {
        let payload_json = item.page.payload_bytes(item.slot_idx)?.to_vec();
        results.push(RefinedSearchResult {
            doc_id: item.doc_id,
            exact_score: item.exact_score,
            minipage_id: item.minipage_id,
            slot_idx: item.slot_idx,
            payload_json,
        });
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use half::f16;
    use object_store::memory::InMemory;
    use object_store::PutPayload;

    use super::*;
    use crate::osv_slab::{OsvSlabReader, OsvVectorRecord, OsvWriter, OsvWriterConfig};

    #[test]
    #[cfg_attr(miri, ignore)] // Invokes zstd-sys C FFI and rayon threadpool
    fn refine_stage_end_to_end_cold_then_warm_cache_and_exact_ranking() -> OspreyResult<()> {
        let orig_dim = 4usize;
        // 6 clusters -> 5 distinct minipages with controlled vectors along [1, 0, 0, 0].
        let vectors_f32: Vec<Vec<f32>> = vec![
            vec![0.10, 0.20, 0.00, 0.00], // page 0, doc 100
            vec![0.95, 0.05, 0.00, 0.00], // page 0, doc 101 (highest dot with [1,0,0,0])
            vec![0.40, 0.00, 0.50, 0.00], // page 1, doc 102
            vec![0.80, 0.10, 0.00, 0.00], // page 2, doc 103 (2nd highest)
            vec![0.60, 0.00, 0.00, 0.30], // page 3, doc 104 (3rd highest)
            vec![0.25, 0.00, 0.00, 0.00], // page 4, doc 105
        ];

        let vecs_f16: Vec<Vec<f16>> = vectors_f32
            .iter()
            .map(|v| v.iter().copied().map(f16::from_f32).collect())
            .collect();
        let payloads: Vec<Vec<u8>> = (0..6usize)
            .map(|i| format!("{{\"doc_id\":{},\"rank_hint\":{i}}}", 100 + i).into_bytes())
            .collect();

        let cluster_map = [0_u32, 0, 1, 2, 3, 4];
        let records: Vec<OsvVectorRecord<'_>> = (0..6usize)
            .map(|i| OsvVectorRecord {
                doc_id: 100 + i as u64,
                cluster_id: cluster_map[i],
                vector_f16: &vecs_f16[i],
                payload: &payloads[i],
            })
            .collect();

        let writer_cfg = OsvWriterConfig {
            minipage_target_bytes: 16_384,
            zstd_level: 3,
            enable_zstd_dict: false,
            max_dict_bytes: 0,
        };
        let mut raw_slab = Vec::new();
        let report =
            OsvWriter::new(writer_cfg).write_slab(&mut raw_slab, &records, orig_dim, None)?;
        assert_eq!(report.directory.len(), 5);

        let reader = OsvSlabReader::from_slice(&raw_slab)?;
        let directory = reader.directory();

        let candidates: Vec<CandidatePageRef> = (0..6usize)
            .filter_map(|i| {
                let doc_id = 100 + i as u64;
                report
                    .location_for_doc(doc_id)
                    .map(|(minipage_id, slot_idx)| CandidatePageRef {
                        doc_id,
                        minipage_id,
                        slot_idx,
                        approx_score_bits: 0,
                    })
            })
            .collect();
        assert_eq!(candidates.len(), 6);

        let query = [1.0_f32, 0.0, 0.0, 0.0];
        let cache = MinipageLruCache::new(1024 * 1024);
        let stage = RefineStage::new(7, RangePlannerConfig::default(), None);

        // Cold run: 0 cache hits, 5 unique pages coalesced into 1 HTTP Range-GET.
        let mut fetch_calls = 0usize;
        let (cold_results, cold_stats) =
            stage.refine_with_fetcher(&query, &candidates, 3, directory, &cache, |ranges| {
                fetch_calls += 1;
                ranges
                    .iter()
                    .map(|r| {
                        let s =
                            usize::try_from(r.start).map_err(|_| OspreyError::CorruptedFooter)?;
                        let e = usize::try_from(r.end).map_err(|_| OspreyError::CorruptedFooter)?;
                        Ok(Bytes::copy_from_slice(&raw_slab[s..e]))
                    })
                    .collect()
            })?;

        assert_eq!(fetch_calls, 1);
        assert_eq!(cold_stats.raw_page_requests, 5);
        assert_eq!(cold_stats.cache_hits, 0);
        assert_eq!(cold_stats.coalesced_range_gets, 1);
        assert!(cold_stats.bytes_fetched > 0);

        assert_eq!(cold_results.len(), 3);
        assert_eq!(cold_results[0].doc_id, 101);
        assert_eq!(cold_results[1].doc_id, 103);
        assert_eq!(cold_results[2].doc_id, 104);
        assert!((cold_results[0].exact_score - f16::from_f32(0.95).to_f32()).abs() < 1e-5);
        assert_eq!(
            cold_results[0].payload_json,
            br#"{"doc_id":101,"rank_hint":1}"#
        );

        // Warm run: all 5 unique minipages served from MinipageLruCache, 0 remote fetches!
        let mut warm_fetch_calls = 0usize;
        let (warm_results, warm_stats) =
            stage.refine_with_fetcher(&query, &candidates, 3, directory, &cache, |_ranges| {
                warm_fetch_calls += 1;
                Ok(Vec::new())
            })?;

        assert_eq!(warm_fetch_calls, 0);
        assert_eq!(warm_stats.raw_page_requests, 5);
        assert_eq!(warm_stats.cache_hits, 5);
        assert_eq!(warm_stats.coalesced_range_gets, 0);
        assert_eq!(warm_stats.bytes_fetched, 0);
        assert_eq!(warm_results, cold_results);
        assert_eq!(cache.hits(), 5);
        assert_eq!(cache.misses(), 5);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(miri, ignore)] // Uses tokio runtime + zstd-sys C FFI
    async fn refine_async_executes_against_object_store_in_memory() -> OspreyResult<()> {
        let orig_dim = 4usize;
        let v0 = [f16::from_f32(0.2), f16::from_f32(0.0), f16::ZERO, f16::ZERO];
        let v1 = [f16::from_f32(0.9), f16::from_f32(0.0), f16::ZERO, f16::ZERO];
        let records = [
            OsvVectorRecord {
                doc_id: 1,
                cluster_id: 0,
                vector_f16: &v0,
                payload: br#"{"id":1}"#,
            },
            OsvVectorRecord {
                doc_id: 2,
                cluster_id: 1,
                vector_f16: &v1,
                payload: br#"{"id":2}"#,
            },
        ];

        let mut raw_slab = Vec::new();
        let report = OsvWriter::new(OsvWriterConfig {
            enable_zstd_dict: false,
            ..OsvWriterConfig::default()
        })
        .write_slab(&mut raw_slab, &records, orig_dim, None)?;

        let reader = OsvSlabReader::from_slice(&raw_slab)?;
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let obj_path = ObjectPath::from("seg0.osv");
        store
            .put(&obj_path, PutPayload::from_bytes(Bytes::from(raw_slab)))
            .await
            .map_err(|e| OspreyError::Io(std::io::Error::other(e)))?;

        let candidates: Vec<CandidatePageRef> = [1_u64, 2_u64]
            .into_iter()
            .filter_map(|doc_id| {
                report
                    .location_for_doc(doc_id)
                    .map(|(minipage_id, slot_idx)| CandidatePageRef {
                        doc_id,
                        minipage_id,
                        slot_idx,
                        approx_score_bits: 0,
                    })
            })
            .collect();

        let cache = MinipageLruCache::new(64 * 1024);
        let stage = RefineStage::new(0, RangePlannerConfig::default(), None);
        let (results, stats) = stage
            .refine_async(
                &[1.0_f32, 0.0, 0.0, 0.0],
                &candidates,
                1,
                reader.directory(),
                &cache,
                &store,
                &obj_path,
                &RemoteStoreConfig::default(),
            )
            .await?;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, 2);
        assert_eq!(results[0].payload_json, br#"{"id":2}"#);
        assert_eq!(stats.coalesced_range_gets, 1);
        Ok(())
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn refine_local_fast_path_supports_uncompressed_mmap_and_compressed_pread(
    ) -> OspreyResult<()> {
        let orig_dim = 8usize;
        let v0 = [f16::from_f32(0.1); 8];
        let v1 = [f16::from_f32(0.8); 8];
        let v2 = [f16::from_f32(0.4); 8];
        let records = [
            OsvVectorRecord {
                doc_id: 10,
                cluster_id: 0,
                vector_f16: &v0,
                payload: br#"{"k":10}"#,
            },
            OsvVectorRecord {
                doc_id: 20,
                cluster_id: 1,
                vector_f16: &v1,
                payload: br#"{"k":20}"#,
            },
            OsvVectorRecord {
                doc_id: 30,
                cluster_id: 2,
                vector_f16: &v2,
                payload: br#"{"k":30}"#,
            },
        ];

        let tmp_dir =
            std::env::temp_dir().join(format!("osprey_refine_local_test_{}", std::process::id()));
        std::fs::create_dir_all(&tmp_dir)?;

        let dummy_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let dummy_path = ObjectPath::from("unused.osv");
        let query = [1.0_f32; 8];

        // 1. Uncompressed local .osv -> mmap zero-copy path (cache remains untouched)
        let uncomp_path = tmp_dir.join("uncompressed.osv");
        let (uncomp_bytes, uncomp_report) = OsvWriter::new(OsvWriterConfig::uncompressed())
            .write_slab_to_bytes(&records, orig_dim, None)?;
        std::fs::write(&uncomp_path, &uncomp_bytes)?;
        let uncomp_reader = OsvSlabReader::open_mmap(&uncomp_path)?;
        let uncomp_cands: Vec<CandidatePageRef> = [10_u64, 20, 30]
            .into_iter()
            .filter_map(|doc_id| {
                uncomp_report
                    .location_for_doc(doc_id)
                    .map(|(minipage_id, slot_idx)| CandidatePageRef {
                        doc_id,
                        minipage_id,
                        slot_idx,
                        approx_score_bits: 0,
                    })
            })
            .collect();
        let cache_uncomp = MinipageLruCache::new(1024 * 1024);
        let stage_uncomp = RefineStage::open_local_path(
            0,
            RangePlannerConfig::default(),
            uncomp_reader.header(),
            None,
            &uncomp_path,
        )?;
        let (res_u, stats_u) = stage_uncomp
            .refine_async(
                &query,
                &uncomp_cands,
                2,
                uncomp_reader.directory(),
                &cache_uncomp,
                &dummy_store,
                &dummy_path,
                &RemoteStoreConfig::default(),
            )
            .await?;
        assert_eq!(res_u.len(), 2);
        assert_eq!(res_u[0].doc_id, 20);
        assert_eq!(res_u[1].doc_id, 30);
        assert_eq!(stats_u.coalesced_range_gets, 0);
        assert_eq!(cache_uncomp.len(), 0);

        // 2. Zstd-compressed local .osv -> pread + cache path (works even in current_thread runtime)
        let comp_path = tmp_dir.join("compressed.osv");
        let (comp_bytes, comp_report) = OsvWriter::new(OsvWriterConfig {
            enable_zstd_dict: false,
            ..OsvWriterConfig::default()
        })
        .write_slab_to_bytes(&records, orig_dim, None)?;
        std::fs::write(&comp_path, &comp_bytes)?;
        let comp_reader = OsvSlabReader::open_mmap(&comp_path)?;
        let comp_cands: Vec<CandidatePageRef> = [10_u64, 20, 30]
            .into_iter()
            .filter_map(|doc_id| {
                comp_report
                    .location_for_doc(doc_id)
                    .map(|(minipage_id, slot_idx)| CandidatePageRef {
                        doc_id,
                        minipage_id,
                        slot_idx,
                        approx_score_bits: 0,
                    })
            })
            .collect();
        let cache_comp = MinipageLruCache::new(1024 * 1024);
        let stage_comp = RefineStage::open_local_path(
            0,
            RangePlannerConfig::default(),
            comp_reader.header(),
            None,
            &comp_path,
        )?;
        let (res_c_cold, stats_c_cold) = stage_comp
            .refine_async(
                &query,
                &comp_cands,
                2,
                comp_reader.directory(),
                &cache_comp,
                &dummy_store,
                &dummy_path,
                &RemoteStoreConfig::default(),
            )
            .await?;
        assert_eq!(res_c_cold.len(), 2);
        assert_eq!(res_c_cold[0].doc_id, 20);
        assert_eq!(res_c_cold[1].doc_id, 30);
        assert_eq!(stats_c_cold.cache_hits, 0);
        assert!(cache_comp.len() == 3);

        let (res_c_warm, stats_c_warm) = stage_comp
            .refine_async(
                &query,
                &comp_cands,
                2,
                comp_reader.directory(),
                &cache_comp,
                &dummy_store,
                &dummy_path,
                &RemoteStoreConfig::default(),
            )
            .await?;
        assert_eq!(res_c_warm, res_c_cold);
        assert_eq!(stats_c_warm.cache_hits, 3);
        assert_eq!(stats_c_warm.coalesced_range_gets, 0);

        let _ = std::fs::remove_dir_all(&tmp_dir);
        Ok(())
    }
}
