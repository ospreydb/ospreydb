//! `osprey bench`: auditable anisotropic synthetic dataset generator,
//! end-to-end Vector Search + DP range coalescing + `LUT-Rank` latency/QPS/Recall@10 benchmark, comparative cloud
//! TCO simulator (Pinecone vs Qdrant Cloud vs Legacy HNSW vs OspreyDB), and Markdown/JSON
//! executive report exporter.

use std::collections::HashSet;
use std::fs::File;
use std::io::BufWriter;
use std::ops::Range;
use std::path::Path;
use std::time::Instant;

use bytes::Bytes;
use comfy_table::{Cell, CellAlignment};
use futures::future::BoxFuture;
use half::f16;
use memmap2::MmapOptions;
use object_store::path::Path as ObjectPath;
use osprey_core::math::{padded_dimension, SplitMix64};
use osprey_core::{
    fit_binary_kmeans, AlignedBuffer, BinaryKMeansConfig, BinaryKMeansResult, OsiIndex,
    OsiMappedView, OspreyError, OspreyResult, OsrDocumentTokens, OsrStore, QuantizedQueryLut,
    SearchOptions, VectorLocation, DEFAULT_LUTRANK_GAMMA, LUTRANK_TOKEN_DIM,
};
use osprey_simd::{SimdBackend, SimdEngine};
use osprey_storage::{
    solve_tsp_locality_largest_first, AsyncRangeTransport, CandidatePageRef, MinipageLruCache,
    OsvSlabReader, OsvVectorRecord, OsvWriter, OsvWriterConfig, RangePlannerConfig, RefineStage,
    RemoteStoreConfig,
};
use rayon::prelude::*;
use roaring::RoaringBitmap;
use serde::Serialize;

use crate::cli::{BenchArgs, RerankCostMode};
use crate::commands::delete::atomic_write_file;
use crate::output::{
    current_dir_short, format_bytes, format_usd_per_mo, new_rounded_table, paint_accent, paint_dim,
    paint_pass, print_cli_header, print_indented_table, print_json, print_kv, print_section,
    CliContext, CliError, CliResult,
};

// Cloud & Infrastructure Pricing Constants (AWS us-east-1 & managed vector SaaS baselines)
pub const HNSW_GRAPH_OVERHEAD: f64 = 1.65;
pub const R6I_8XL_RAM_GB: f64 = 256.0;
pub const R6I_8XL_MONTHLY_USD: f64 = 1_472.0; // r6i.8xlarge (~$2.016/h * 730h)
pub const R6I_16XL_RAM_GB: f64 = 512.0;
pub const R6I_16XL_MONTHLY_USD: f64 = 2_944.0; // r6i.16xlarge (~$4.032/h * 730h)
pub const G5_XLARGE_MONTHLY_USD: f64 = 734.0; // g5.xlarge dedicated GPU reranker
pub const RERANK_API_PER_1K_USD: f64 = 2.00; // $2.00 / 1,000 /v1/rerank API calls
pub const S3_STANDARD_PER_GB_USD: f64 = 0.023; // $0.023 / GB / month
pub const S3_GET_PER_1K_USD: f64 = 0.0004; // $0.0004 / 1,000 Range-GETs
pub const C7I_2XL_MONTHLY_USD: f64 = 245.0; // c7i.2xlarge (16 GB RAM, AVX-512)

// Pinecone (Dedicated Read Nodes / Standard On-Demand) pricing constants
const PINECONE_STORAGE_PER_GB_USD: f64 = 0.33; // $0.33 / GB / month
const PINECONE_READ_NODE_SHARD_GB: f64 = 100.0; // 1 dedicated read node shard per 100 GB
const PINECONE_READ_NODE_MONTHLY_USD: f64 = 292.0; // ~$0.40/h * 730h per dedicated read node
const PINECONE_RU_PER_1M_USD: f64 = 8.25; // $8.25 / 1M Read Units
const PINECONE_WU_PER_1M_USD: f64 = 2.00; // $2.00 / 1M Write Units

// Qdrant Cloud (Managed SSD Offload + TurboQuant 1-Bit + HA Replication) pricing constants
const QDRANT_HA_REPLICATION_FACTOR: f64 = 2.0; // 2x HA replication
const QDRANT_MANAGED_SSD_PER_GB_USD: f64 = 0.25; // Managed NVMe/gp3 SSD + snapshots
const QDRANT_COMPUTE_NODE_RAM_GB: f64 = 8.0; // 8 GB RAM + 2 vCPU unit
const QDRANT_COMPUTE_NODE_MONTHLY_USD: f64 = 128.0; // Managed Qdrant Cloud 8 GB unit

/// In-memory byte-slice transport for benchmarking DP range coalescing + `RefineStage` without S3 network jitter.
struct InMemorySlabTransport {
    bytes: Bytes,
}

impl AsyncRangeTransport for InMemorySlabTransport {
    fn get_ranges<'a>(
        &'a self,
        _path: &'a ObjectPath,
        ranges: &'a [Range<usize>],
    ) -> BoxFuture<'a, OspreyResult<Vec<Bytes>>> {
        Box::pin(async move {
            let mut out = Vec::with_capacity(ranges.len());
            for r in ranges {
                if r.end > self.bytes.len() || r.start > r.end {
                    return Err(OspreyError::BufferTooSmall {
                        needed: r.end,
                        actual: self.bytes.len(),
                    });
                }
                out.push(self.bytes.slice(r.clone()));
            }
            Ok(out)
        })
    }
}

#[derive(Debug, Serialize)]
pub struct LocalBenchMetrics {
    pub simd_tier: String,
    pub vectors: usize,
    pub orig_dim: usize,
    pub padded_dim: usize,
    pub queries: usize,
    pub clusters: usize,
    pub requested_nprobe: usize,
    pub requested_candidates: usize,
    pub nprobe: usize,
    pub candidates: usize,
    pub refine_multiplier: usize,
    pub low_dim_oversample: bool,
    pub build_time_ms: f64,
    pub osi_bytes: u64,
    pub osv_bytes: u64,
    pub osv_compression_ratio: f64,
    pub legacy_fp32_hnsw_bytes: u64,
    pub ram_pruned_pct: f64,
    pub rss_before_mb: Option<f64>,
    pub rss_after_build_mb: Option<f64>,
    pub rss_mmap_mb: Option<f64>,
    pub range_planner_avg_ms: f64,
    pub coalesce_factor: f64,
    pub avg_coalesced_gets_per_query: f64,
    pub ann_p50_ms: f64,
    pub ann_p95_ms: f64,
    pub ann_p99_ms: f64,
    pub ann_qps: f64,
    pub ann_recall_at_10_pct: f64,
    pub lutrank: Option<LutRankBenchMetrics>,
}

#[derive(Debug, Serialize)]
pub struct LutRankBenchMetrics {
    pub doc_tokens_ld: usize,
    pub query_tokens_lq: usize,
    pub osr_bytes: u64,
    pub lut_synthesis_avg_us: f64,
    pub rerank_200_docs_p50_ms: f64,
    pub rerank_200_docs_p99_ms: f64,
    pub pearson_correlation: f64,
    pub maxsim_recall_at_10_pct: f64,
}

#[derive(Debug, Serialize)]
pub struct ProviderTcoBreakdown {
    pub provider: String,
    pub architecture_summary: String,
    pub storage_gb: f64,
    pub ram_gb: f64,
    pub storage_monthly_usd: f64,
    pub compute_monthly_usd: f64,
    pub io_and_tolls_monthly_usd: f64,
    pub rerank_monthly_usd: f64,
    pub total_monthly_usd: f64,
    pub total_annual_usd: f64,
    pub osprey_savings_pct: f64,
}

#[derive(Debug, Serialize)]
pub struct CloudTcoProjection {
    pub project_vectors: u64,
    pub project_dim: u64,
    pub project_qpm: u64,
    pub projected_sustained_qps: f64,
    pub rerank_mode: String,
    pub raw_dataset_gb: f64,
    pub osprey_osi_ram_gb: f64,
    pub osprey_osv_s3_gb: f64,
    pub avg_coalesced_gets_per_query: f64,
    pub pinecone: ProviderTcoBreakdown,
    pub qdrant: ProviderTcoBreakdown,
    pub legacy_hnsw: ProviderTcoBreakdown,
    pub ospreydb: ProviderTcoBreakdown,
}

#[derive(Debug, Serialize)]
pub struct BenchReport {
    pub local: LocalBenchMetrics,
    pub tco: CloudTcoProjection,
}

/// Executes `osprey bench`.
pub fn run(args: &BenchArgs, ctx: &CliContext) -> CliResult<()> {
    validate_bench_args(args)?;

    ctx.log_info(format!(
        "Running OspreyDB scientific benchmark (N={}, D={}, Q={}, K={}, nprobe={}, with_lutrank={})...",
        args.vectors, args.dim, args.queries, args.clusters, args.nprobe, args.with_lutrank
    ));

    let local = run_local_benchmark(args, ctx)?;
    let tco = compute_cloud_tco_projection(args, &local);
    let report = BenchReport { local, tco };

    if let Some(export_path) = &args.export_report {
        export_report_file(export_path, &report)?;
        ctx.log_info(format!(
            "Exported executive benchmark & TCO report to '{}'",
            export_path.display()
        ));
    }

    if ctx.json {
        print_json(&report)?;
    } else {
        print_bench_human(&report, args.export_report.as_deref());
    }

    Ok(())
}

fn validate_bench_args(args: &BenchArgs) -> CliResult<()> {
    if args.vectors == 0 || args.dim == 0 || args.queries == 0 || args.clusters == 0 {
        return Err(CliError::InvalidArgs(
            "--vectors, --dim, --queries, and --clusters must all be > 0".to_string(),
        ));
    }
    if args.nprobe == 0 || args.candidates == 0 {
        return Err(CliError::InvalidArgs(
            "--nprobe and --candidates must be > 0".to_string(),
        ));
    }
    if args.with_lutrank && (args.doc_tokens == 0 || args.query_tokens == 0) {
        return Err(CliError::InvalidArgs(
            "--doc-tokens and --query-tokens must be > 0 when --with-lutrank is enabled"
                .to_string(),
        ));
    }
    if args.project_vectors == 0 || args.project_dim == 0 {
        return Err(CliError::InvalidArgs(
            "--project-vectors and --project-dim must be > 0".to_string(),
        ));
    }
    Ok(())
}

fn run_local_benchmark(args: &BenchArgs, ctx: &CliContext) -> CliResult<LocalBenchMetrics> {
    let simd = SimdEngine::new_auto();
    let n = args.vectors;
    let dim = args.dim;
    let q_count = args.queries;
    let k = usize::try_from(args.clusters).unwrap_or(128).min(n).max(1);
    let top_k = 10.min(n);
    let orig_dim_u16 = u16::try_from(dim).unwrap_or(u16::MAX);
    let plan = SearchOptions {
        nprobe: args.nprobe,
        candidates: args.candidates,
        top_k,
        rerank_top_m: 0,
        refine_multiplier: args.refine_multiplier,
    }
    .resolve(orig_dim_u16);
    let nprobe = plan.effective_nprobe.min(k);
    let candidates_k = plan.effective_candidates.min(n).max(top_k);

    let rss_before_mb = read_process_rss_mb();

    // 1. Auditable anisotropic clustered dataset + queries
    let (db_f32, db_f16, doc_ids, queries) =
        generate_anisotropic_dataset(n, dim, q_count, k, args.seed);

    // 2. Exact FP32 Ground-Truth Top-10
    let gt_top10: Vec<HashSet<u64>> = (0..q_count)
        .into_par_iter()
        .map(|qi| {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let mut scores: Vec<(u64, f32)> = (0..n)
                .map(|di| {
                    let v = &db_f32[di * dim..(di + 1) * dim];
                    let dot: f32 = q.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();
                    (doc_ids[di], dot)
                })
                .collect();
            scores.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            scores.into_iter().take(10).map(|(id, _)| id).collect()
        })
        .collect();

    // 3. Build 1-Bit Binary K-Means + TSP locality ordering + .osv + .osi
    let t_build = Instant::now();
    let skm_cfg = BinaryKMeansConfig {
        num_clusters: k,
        max_iters: 20,
        ..BinaryKMeansConfig::new(k)
    };
    let km = fit_binary_kmeans(&db_f32, n, dim, args.seed, &skm_cfg, &simd)?;
    let padded_dim = padded_dimension(dim)?;

    let mut cluster_counts = vec![0_u64; k];
    for &cid in &km.assignments {
        if let Some(slot) = cluster_counts.get_mut(usize::try_from(cid).unwrap_or(usize::MAX)) {
            *slot += 1;
        }
    }
    let perm = solve_tsp_locality_largest_first(
        km.centroids_f16.as_slice(),
        k,
        padded_dim,
        &cluster_counts,
        20,
    )?;
    let tsp_centroids = perm.reorder_centroids_f16(km.centroids_f16.as_slice(), padded_dim)?;
    let mut aligned_cents = AlignedBuffer::<f16>::new(tsp_centroids.len())?;
    aligned_cents.as_mut_slice().copy_from_slice(&tsp_centroids);

    let words_per_vec = padded_dim / 64;
    let mut binary_centroids = vec![0_u64; k * words_per_vec];
    for c in 0..k {
        let cent_slice = &tsp_centroids[c * padded_dim..(c + 1) * padded_dim];
        for (w, out_word) in binary_centroids[c * words_per_vec..(c + 1) * words_per_vec]
            .iter_mut()
            .enumerate()
        {
            let mut word = 0_u64;
            for b in 0..64 {
                if cent_slice[w * 64 + b].to_f32() >= 0.0 {
                    word |= 1_u64 << b;
                }
            }
            *out_word = word;
        }
    }

    let mut tsp_assignments = Vec::with_capacity(n);
    for &logical_cid in &km.assignments {
        tsp_assignments.push(perm.to_physical(logical_cid)?);
    }

    let tmp_dir = tempfile::tempdir()?;
    let osv_path = tmp_dir.path().join("bench.osv");
    let osi_path = tmp_dir.path().join("bench.osi");

    let empty_payload = b"{}";
    let osv_records: Vec<OsvVectorRecord<'_>> = (0..n)
        .map(|i| OsvVectorRecord {
            doc_id: doc_ids[i],
            cluster_id: tsp_assignments[i],
            vector_f16: &db_f16[i * dim..(i + 1) * dim],
            payload: empty_payload,
        })
        .collect();
    let osv_report = OsvWriter::new(OsvWriterConfig::default()).write_slab(
        &mut BufWriter::new(File::create(&osv_path)?),
        &osv_records,
        dim,
        None,
    )?;

    let locations: Vec<VectorLocation> = doc_ids
        .iter()
        .map(|&id| {
            let (minipage_id, slot_idx) = osv_report.location_for_doc(id).unwrap_or((0, 0));
            VectorLocation {
                minipage_id,
                slot_idx,
                slab_id: 0,
            }
        })
        .collect();

    let km_res = BinaryKMeansResult {
        centroids_f16: aligned_cents,
        binary_centroids,
        assignments: tsp_assignments,
        final_hamming_inertia: km.final_hamming_inertia,
        iterations_run: km.iterations_run,
    };
    let osi_built = OsiIndex::build(
        &db_f32,
        &doc_ids,
        &locations,
        &km_res,
        dim,
        args.seed,
        RoaringBitmap::new(),
        &simd,
    )?;
    let osi_aligned = osi_built.to_aligned_bytes()?;
    atomic_write_file(&osi_path, osi_aligned.as_slice())?;
    let build_time_ms = t_build.elapsed().as_secs_f64() * 1_000.0;
    let rss_after_build_mb = read_process_rss_mb();

    let osi_bytes = u64::try_from(osi_aligned.len()).unwrap_or(u64::MAX);
    let osv_file_bytes = std::fs::metadata(&osv_path)?.len();
    #[allow(clippy::cast_precision_loss)]
    let osv_compression_ratio = if osv_report.total_compressed_bytes > 0 {
        (osv_report.total_uncompressed_bytes as f64) / (osv_report.total_compressed_bytes as f64)
    } else {
        1.0
    };

    // Release raw build buffers before mmap query benchmark so RSS reflects query footprint.
    drop(osi_built);
    drop(osi_aligned);
    drop(db_f32);
    drop(db_f16);

    let osi_file = File::open(&osi_path)?;
    // SAFETY: `osi_path` is immutable for the duration of the benchmark and checksum-verified.
    // `populate()` pre-faults the resident `.osi` index, removing p99 tail page faults.
    let osi_mmap = unsafe { MmapOptions::new().populate().map(&osi_file)? };
    let _osi_view = OsiMappedView::from_bytes(&osi_mmap)?;
    let osi = OsiIndex::from_bytes(&osi_mmap)?;
    let rss_mmap_mb = read_process_rss_mb();

    let osv_raw = std::fs::read(&osv_path)?;
    let osv_reader = OsvSlabReader::from_slice(&osv_raw)?;
    let directory = osv_reader.directory().to_vec();
    let zstd_dict = if osv_reader.header().zstd_dict_len > 0 {
        let start = usize::try_from(osv_reader.header().zstd_dict_offset)
            .map_err(|_| CliError::Core(OspreyError::CorruptedFooter))?;
        let len = usize::try_from(osv_reader.header().zstd_dict_len)
            .map_err(|_| CliError::Core(OspreyError::CorruptedFooter))?;
        Some(osv_raw[start..start + len].to_vec())
    } else {
        None
    };

    let transport = InMemorySlabTransport {
        bytes: Bytes::from(osv_raw),
    };
    let refine_stage = RefineStage::new(0, RangePlannerConfig::default(), zstd_dict);
    let dummy_obj = ObjectPath::from("bench.osv");
    let remote_cfg = RemoteStoreConfig::default();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| CliError::Io(format!("failed to create tokio runtime: {err}")))?;

    let mut query_latencies_ms = Vec::with_capacity(q_count);
    let mut total_range_plan_us = 0_u64;
    let mut total_raw_pages = 0_usize;
    let mut total_coalesced_gets = 0_usize;
    let mut total_recall_hits = 0_usize;

    let t_all_queries = Instant::now();
    for qi in 0..q_count {
        let q = &queries[qi * dim..(qi + 1) * dim];
        // Fresh empty cache per query to measure cold DP range coalescing & Zstd refinement accurately.
        let cache = MinipageLruCache::new(4 * 1024 * 1024);

        let t_q = Instant::now();
        let stage1 = osi.search_candidates(q, nprobe, candidates_k, &simd)?;
        let page_refs: Vec<CandidatePageRef> = stage1.iter().map(CandidatePageRef::from).collect();
        let (refined, stats) = rt.block_on(refine_stage.refine_async_with_transport(
            q,
            &page_refs,
            10,
            &directory,
            &cache,
            &transport,
            &dummy_obj,
            &remote_cfg,
        ))?;
        query_latencies_ms.push(t_q.elapsed().as_secs_f64() * 1_000.0);

        total_range_plan_us += stats.range_plan_time_us;
        total_raw_pages += stats.raw_page_requests;
        total_coalesced_gets += stats.coalesced_range_gets;

        let got_top10: HashSet<u64> = refined.iter().take(10).map(|r| r.doc_id).collect();
        total_recall_hits += gt_top10[qi].intersection(&got_top10).count();
    }
    let total_query_secs = t_all_queries.elapsed().as_secs_f64().max(1e-9);

    query_latencies_ms.sort_by(f64::total_cmp);
    #[allow(clippy::cast_precision_loss)]
    let ann_qps = (q_count as f64) / total_query_secs;
    #[allow(clippy::cast_precision_loss)]
    let ann_recall_at_10_pct =
        ((total_recall_hits as f64) / ((q_count * 10.min(n)) as f64)) * 100.0;
    #[allow(clippy::cast_precision_loss)]
    let range_planner_avg_ms = (total_range_plan_us as f64) / (q_count as f64) / 1_000.0;
    #[allow(clippy::cast_precision_loss)]
    let coalesce_factor = if total_coalesced_gets > 0 {
        (total_raw_pages as f64) / (total_coalesced_gets as f64)
    } else {
        1.0
    };
    #[allow(clippy::cast_precision_loss)]
    let avg_coalesced_gets_per_query = (total_coalesced_gets as f64) / (q_count as f64);

    let legacy_fp32_hnsw_bytes = u64::try_from(n)
        .unwrap_or(u64::MAX)
        .saturating_mul(u64::try_from(dim).unwrap_or(u64::MAX))
        .saturating_mul(4)
        .saturating_mul(165)
        / 100;
    #[allow(clippy::cast_precision_loss)]
    let ram_pruned_pct = if legacy_fp32_hnsw_bytes > 0 {
        (1.0 - (osi_bytes as f64) / (legacy_fp32_hnsw_bytes as f64)) * 100.0
    } else {
        0.0
    };

    let lutrank = if args.with_lutrank {
        ctx.log_info("Running Stage-4 LUT-Rank ColBERT 200-doc SIMD benchmark...");
        Some(run_lutrank_benchmark(args, &simd)?)
    } else {
        None
    };

    Ok(LocalBenchMetrics {
        simd_tier: format!("{:?}", simd.level()),
        vectors: n,
        orig_dim: dim,
        padded_dim,
        queries: q_count,
        clusters: k,
        requested_nprobe: args.nprobe,
        requested_candidates: args.candidates,
        nprobe,
        candidates: candidates_k,
        refine_multiplier: plan.refine_multiplier,
        low_dim_oversample: plan.low_dim_oversample,
        build_time_ms,
        osi_bytes,
        osv_bytes: osv_file_bytes,
        osv_compression_ratio,
        legacy_fp32_hnsw_bytes,
        ram_pruned_pct,
        rss_before_mb,
        rss_after_build_mb,
        rss_mmap_mb,
        range_planner_avg_ms,
        coalesce_factor,
        avg_coalesced_gets_per_query,
        ann_p50_ms: percentile_f64(&query_latencies_ms, 50),
        ann_p95_ms: percentile_f64(&query_latencies_ms, 95),
        ann_p99_ms: percentile_f64(&query_latencies_ms, 99),
        ann_qps,
        ann_recall_at_10_pct,
        lutrank,
    })
}

fn run_lutrank_benchmark(args: &BenchArgs, simd: &SimdEngine) -> CliResult<LutRankBenchMetrics> {
    const RERANK_DOCS: usize = 200;
    let l_d = args.doc_tokens;
    let l_q = args.query_tokens;
    let tok_dim = LUTRANK_TOKEN_DIM;

    let mut rng = SplitMix64::new(args.seed ^ 0xA5A5_5A5A_1234_9876);

    // Build 200 tiered anisotropic ColBERT documents for scientifically calibrated correlation/recall.
    let q_tokens = sample_anisotropic_tokens(&mut rng, l_q, tok_dim);
    let mut tiers: Vec<usize> = (0..RERANK_DOCS).collect();
    for i in (1..RERANK_DOCS).rev() {
        let j = usize::try_from(rng.next_u64() % u64::try_from(i + 1).unwrap_or(1)).unwrap_or(0);
        tiers.swap(i, j);
    }

    let mut doc_buffers: Vec<Vec<f32>> = Vec::with_capacity(RERANK_DOCS);
    for &tier in &tiers {
        let mut toks = sample_anisotropic_tokens(&mut rng, l_d, tok_dim);
        // Blend progressive similarity to query tokens so ground-truth ranking spans a realistic spectrum.
        #[allow(clippy::cast_precision_loss)]
        let relevance = 1.0 - (tier as f32) / (RERANK_DOCS as f32);
        let match_tokens = l_q.min(l_d);
        for qt in 0..match_tokens {
            let dst = &mut toks[qt * tok_dim..(qt + 1) * tok_dim];
            let src = &q_tokens[qt * tok_dim..(qt + 1) * tok_dim];
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                *d = *d * (1.0 - relevance * 0.75) + *s * (relevance * 0.75);
            }
            normalize_l2_in_place(dst);
        }
        doc_buffers.push(toks);
    }

    let osr_docs: Vec<OsrDocumentTokens<'_>> = (0..RERANK_DOCS)
        .map(|i| OsrDocumentTokens {
            doc_id: u64::try_from(i).unwrap_or(0),
            tokens: &doc_buffers[i],
            num_tokens: l_d,
        })
        .collect();

    let osr = OsrStore::build(&osr_docs, tok_dim, DEFAULT_LUTRANK_GAMMA, args.seed, simd)?;
    let osr_bytes = u64::try_from(osr.as_bytes().len()).unwrap_or(u64::MAX);

    // Exact FP32 MaxSim ground truth
    let mut exact_scores = Vec::with_capacity(RERANK_DOCS);
    for doc_toks in &doc_buffers {
        let mut total = 0.0_f32;
        for qi in 0..l_q {
            let qv = &q_tokens[qi * tok_dim..(qi + 1) * tok_dim];
            let mut best = f32::NEG_INFINITY;
            for di in 0..l_d {
                let dv = &doc_toks[di * tok_dim..(di + 1) * tok_dim];
                let dot: f32 = qv.iter().zip(dv.iter()).map(|(&a, &b)| a * b).sum();
                if dot > best {
                    best = dot;
                }
            }
            total += best;
        }
        exact_scores.push(total);
    }

    let bench_iters = args.queries.clamp(10, 50);
    let mut lut_us_samples = Vec::with_capacity(bench_iters);
    let mut rerank_ms_samples = Vec::with_capacity(bench_iters);
    let mut lut_scores = vec![0.0_f32; RERANK_DOCS];

    for _ in 0..bench_iters {
        let t_lut = Instant::now();
        let q_lut = QuantizedQueryLut::synthesize(
            &q_tokens,
            l_q,
            tok_dim,
            DEFAULT_LUTRANK_GAMMA,
            args.seed,
            simd,
        )?;
        lut_us_samples.push(t_lut.elapsed().as_secs_f64() * 1_000_000.0);

        let t_rr = Instant::now();
        for (doc_idx, dst_score) in lut_scores.iter_mut().enumerate() {
            if doc_idx + 1 < RERANK_DOCS {
                osr.prefetch_doc_idx(doc_idx + 1, simd);
            }
            *dst_score = osr.score_document_maxsim(doc_idx, &q_lut, simd);
        }
        rerank_ms_samples.push(t_rr.elapsed().as_secs_f64() * 1_000.0);
    }

    let pearson = pearson_r(&exact_scores, &lut_scores);
    let mut exact_order: Vec<(usize, f32)> = exact_scores.iter().copied().enumerate().collect();
    exact_order.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let gt10: HashSet<usize> = exact_order.iter().take(10).map(|&(idx, _)| idx).collect();

    let mut lut_order: Vec<(usize, f32)> = lut_scores.iter().copied().enumerate().collect();
    lut_order.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let got10: HashSet<usize> = lut_order.iter().take(10).map(|&(idx, _)| idx).collect();
    #[allow(clippy::cast_precision_loss)]
    let maxsim_recall_at_10_pct = ((gt10.intersection(&got10).count() as f64) / 10.0) * 100.0;

    #[allow(clippy::cast_precision_loss)]
    let lut_synthesis_avg_us = lut_us_samples.iter().sum::<f64>() / (lut_us_samples.len() as f64);
    rerank_ms_samples.sort_by(f64::total_cmp);

    Ok(LutRankBenchMetrics {
        doc_tokens_ld: l_d,
        query_tokens_lq: l_q,
        osr_bytes,
        lut_synthesis_avg_us,
        rerank_200_docs_p50_ms: percentile_f64(&rerank_ms_samples, 50),
        rerank_200_docs_p99_ms: percentile_f64(&rerank_ms_samples, 99),
        pearson_correlation: pearson,
        maxsim_recall_at_10_pct,
    })
}

fn compute_cloud_tco_projection(args: &BenchArgs, local: &LocalBenchMetrics) -> CloudTcoProjection {
    #[allow(clippy::cast_precision_loss)]
    let n = args.project_vectors as f64;
    #[allow(clippy::cast_precision_loss)]
    let d = args.project_dim as f64;
    #[allow(clippy::cast_precision_loss)]
    let qpm = args.project_qpm as f64;

    let projected_sustained_qps = qpm / (30.0 * 86_400.0);
    let avg_coalesced_gets = local.avg_coalesced_gets_per_query.max(1.0);

    // 512 B metadata alongside FP32 dense vectors => 6,656 B/vec at D=1536 => 332.8 GB (~333 GB) for 50M vectors.
    let raw_bytes_per_vec = d * 4.0 + 512.0;
    let raw_dataset_gb = (n * raw_bytes_per_vec) / 1e9;

    // OspreyDB .osv stores f16 vectors + Zstd-compressed full JSON payloads + minipage headers.
    // Scaled with the raw dataset volume (~332.8 GB for 50M x 1536 => ~$7.65/mo S3 Standard).
    let osv_s3_gb = raw_dataset_gb;

    // OspreyDB .osi resident RAM: ceil(D / 8) 1-bit code + 4B factors/header overhead => ~9.8 GB for 50M x 1536.
    let osi_bytes_per_vec = (d / 8.0).ceil() + 4.0;
    let osprey_osi_ram_gb = (n * osi_bytes_per_vec) / 1e9;

    // Reranking baseline cost for Legacy / Pinecone / Qdrant
    let rerank_cost_usd = match args.rerank_mode {
        RerankCostMode::Gpu => {
            let gpu_nodes = (projected_sustained_qps / 50.0).ceil().max(1.0);
            gpu_nodes * G5_XLARGE_MONTHLY_USD
        }
        RerankCostMode::Api => (qpm / 1_000.0) * RERANK_API_PER_1K_USD,
        RerankCostMode::None => 0.0,
    };
    let rerank_label = match args.rerank_mode {
        RerankCostMode::Gpu => "GPU (g5.xlarge)",
        RerankCostMode::Api => "API (/v1/rerank)",
        RerankCostMode::None => "None",
    };

    // (C) OspreyDB (Open-Core Apache-2.0)
    let osprey_nodes = (osprey_osi_ram_gb / 14.0).ceil().max(1.0);
    let osprey_compute_usd = osprey_nodes * C7I_2XL_MONTHLY_USD;
    let osprey_storage_usd = osv_s3_gb * S3_STANDARD_PER_GB_USD;
    let osprey_s3_gets_usd = ((qpm * avg_coalesced_gets) / 1_000.0) * S3_GET_PER_1K_USD;
    let osprey_total_usd = osprey_compute_usd + osprey_storage_usd + osprey_s3_gets_usd;

    // (A) Pinecone (Dedicated Read Nodes / On-Demand)
    let pinecone_storage_usd = raw_dataset_gb * PINECONE_STORAGE_PER_GB_USD;
    let pinecone_read_nodes = (raw_dataset_gb / PINECONE_READ_NODE_SHARD_GB)
        .ceil()
        .max(1.0);
    let pinecone_compute_usd = pinecone_read_nodes * PINECONE_READ_NODE_MONTHLY_USD;
    // Monthly Read Units (scaled by index footprint) + 10% monthly upsert/update Write Units toll
    let read_units_m = (qpm / 1_000_000.0) * (raw_dataset_gb / 100.0).max(1.0);
    let write_units_m = ((n * 0.10) / 1_000_000.0) * (d / 384.0).max(1.0);
    let pinecone_tolls_usd =
        read_units_m * PINECONE_RU_PER_1M_USD + write_units_m * PINECONE_WU_PER_1M_USD;
    let pinecone_total_usd =
        pinecone_storage_usd + pinecone_compute_usd + pinecone_tolls_usd + rerank_cost_usd;

    // (B) Qdrant Cloud (Managed SSD Offload + TurboQuant 1-Bit + 2x HA Replication)
    let qdrant_ram_gb = (n * ((d / 8.0).ceil() + 48.0) * QDRANT_HA_REPLICATION_FACTOR) / 1e9;
    let qdrant_nodes = (qdrant_ram_gb / QDRANT_COMPUTE_NODE_RAM_GB).ceil().max(1.0);
    let qdrant_compute_usd = qdrant_nodes * QDRANT_COMPUTE_NODE_MONTHLY_USD;
    let qdrant_storage_gb = raw_dataset_gb * QDRANT_HA_REPLICATION_FACTOR;
    let qdrant_storage_usd = qdrant_storage_gb * QDRANT_MANAGED_SSD_PER_GB_USD;
    let qdrant_total_usd = qdrant_compute_usd + qdrant_storage_usd + rerank_cost_usd;

    // Legacy Self-Managed HNSW in RAM (r6i.16xlarge / r6i.8xlarge + GPU/API Rerank)
    let legacy_ram_gb = (n * d * 4.0 * HNSW_GRAPH_OVERHEAD) / 1e9;
    let legacy_compute_usd = compute_r6i_ram_cost(legacy_ram_gb);
    let legacy_storage_usd = raw_dataset_gb * 0.08; // EBS gp3 snapshot/volume baseline
    let legacy_total_usd = legacy_compute_usd + legacy_storage_usd + rerank_cost_usd;

    let savings_vs = |other_total: f64| -> f64 {
        if other_total > 0.0 {
            ((other_total - osprey_total_usd) / other_total) * 100.0
        } else {
            0.0
        }
    };

    CloudTcoProjection {
        project_vectors: args.project_vectors,
        project_dim: args.project_dim,
        project_qpm: args.project_qpm,
        projected_sustained_qps,
        rerank_mode: rerank_label.to_string(),
        raw_dataset_gb,
        osprey_osi_ram_gb,
        osprey_osv_s3_gb: osv_s3_gb,
        avg_coalesced_gets_per_query: avg_coalesced_gets,
        pinecone: ProviderTcoBreakdown {
            provider: "Pinecone (Dedicated Read + On-Demand)".to_string(),
            architecture_summary: format!(
                "{pinecone_read_nodes:.0} read node(s) + {raw_dataset_gb:.1} GB managed + RU/WU tolls",
            ),
            storage_gb: raw_dataset_gb,
            ram_gb: raw_dataset_gb,
            storage_monthly_usd: pinecone_storage_usd,
            compute_monthly_usd: pinecone_compute_usd,
            io_and_tolls_monthly_usd: pinecone_tolls_usd,
            rerank_monthly_usd: rerank_cost_usd,
            total_monthly_usd: pinecone_total_usd,
            total_annual_usd: pinecone_total_usd * 12.0,
            osprey_savings_pct: savings_vs(pinecone_total_usd),
        },
        qdrant: ProviderTcoBreakdown {
            provider: "Qdrant Cloud (TurboQuant 1-Bit + SSD)".to_string(),
            architecture_summary: format!(
                "{qdrant_nodes:.0} x 8GB nodes ({qdrant_ram_gb:.1} GB 1-bit HA RAM) + {qdrant_storage_gb:.1} GB SSD (2x HA)",
            ),
            storage_gb: qdrant_storage_gb,
            ram_gb: qdrant_ram_gb,
            storage_monthly_usd: qdrant_storage_usd,
            compute_monthly_usd: qdrant_compute_usd,
            io_and_tolls_monthly_usd: 0.0,
            rerank_monthly_usd: rerank_cost_usd,
            total_monthly_usd: qdrant_total_usd,
            total_annual_usd: qdrant_total_usd * 12.0,
            osprey_savings_pct: savings_vs(qdrant_total_usd),
        },
        legacy_hnsw: ProviderTcoBreakdown {
            provider: "Legacy Self-Managed HNSW (AWS r6i)".to_string(),
            architecture_summary: format!(
                "{legacy_ram_gb:.1} GB HNSW RAM + EBS gp3 + {rerank_label}",
            ),
            storage_gb: raw_dataset_gb,
            ram_gb: legacy_ram_gb,
            storage_monthly_usd: legacy_storage_usd,
            compute_monthly_usd: legacy_compute_usd,
            io_and_tolls_monthly_usd: 0.0,
            rerank_monthly_usd: rerank_cost_usd,
            total_monthly_usd: legacy_total_usd,
            total_annual_usd: legacy_total_usd * 12.0,
            osprey_savings_pct: savings_vs(legacy_total_usd),
        },
        ospreydb: ProviderTcoBreakdown {
            provider: "OspreyDB (Open-Core Apache-2.0)".to_string(),
            architecture_summary: format!(
                "{osprey_nodes:.0}x c7i.2xlarge ({osprey_osi_ram_gb:.1} GB .osi) + {osv_s3_gb:.1} GB S3 + CPU LUT-Rank",
            ),
            storage_gb: osv_s3_gb,
            ram_gb: osprey_osi_ram_gb,
            storage_monthly_usd: osprey_storage_usd,
            compute_monthly_usd: osprey_compute_usd,
            io_and_tolls_monthly_usd: osprey_s3_gets_usd,
            rerank_monthly_usd: 0.0,
            total_monthly_usd: osprey_total_usd,
            total_annual_usd: osprey_total_usd * 12.0,
            osprey_savings_pct: 0.0,
        },
    }
}

fn compute_r6i_ram_cost(required_ram_gb: f64) -> f64 {
    if required_ram_gb <= R6I_8XL_RAM_GB {
        R6I_8XL_MONTHLY_USD
    } else {
        let count_16xl = (required_ram_gb / R6I_16XL_RAM_GB).floor();
        let rem = required_ram_gb - count_16xl * R6I_16XL_RAM_GB;
        let extra = if rem <= 0.0 {
            0.0
        } else if rem <= R6I_8XL_RAM_GB {
            R6I_8XL_MONTHLY_USD
        } else {
            R6I_16XL_MONTHLY_USD
        };
        count_16xl * R6I_16XL_MONTHLY_USD + extra
    }
}

fn export_report_file(path: &Path, report: &BenchReport) -> CliResult<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let is_json = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("json"));
    if is_json {
        let json = serde_json::to_string_pretty(report)
            .map_err(|err| CliError::Io(format!("JSON export serialization failed: {err}")))?;
        atomic_write_file(path, json.as_bytes())?;
    } else {
        let md = render_markdown_report(report);
        atomic_write_file(path, md.as_bytes())?;
    }
    Ok(())
}

fn render_markdown_report(r: &BenchReport) -> String {
    let l = &r.local;
    let t = &r.tco;

    let mut md = String::new();
    md.push_str("# OspreyDB Executive Benchmark & Cloud TCO Report\n\n");
    md.push_str(&format!(
        "- **SIMD Backend:** `{}`\n- **Local Corpus:** `N = {}` vectors, `D = {}` (`padded = {}`), `K = {}` clusters, `nprobe = {}`, `queries = {}`\n- **Cloud Projection Scale:** `{} vectors x {} dims`, `{} QPM` (`{:.2} sustained QPS`), rerank baseline: `{}`\n\n",
        l.simd_tier,
        l.vectors,
        l.orig_dim,
        l.padded_dim,
        l.clusters,
        l.nprobe,
        l.queries,
        format_u64_commas(t.project_vectors),
        t.project_dim,
        format_u64_commas(t.project_qpm),
        t.projected_sustained_qps,
        t.rerank_mode,
    ));

    md.push_str("## 1. Local Scientific Benchmark (Measured)\n\n");
    md.push_str("| Metric | Measured Value |\n| :--- | :--- |\n");
    md.push_str(&format!(
        "| Build Time (1-Bit Binary K-Means + TSP + .osv/.osi) | `{:.2} ms` |\n",
        l.build_time_ms
    ));
    md.push_str(&format!(
        "| Index Size (`.osi`) | `{}` (`{:.2}%` RAM pruned vs FP32 HNSW `{}`) |\n",
        format_bytes(l.osi_bytes),
        l.ram_pruned_pct,
        format_bytes(l.legacy_fp32_hnsw_bytes)
    ));
    md.push_str(&format!(
        "| Slab Storage (`.osv`) | `{}` (`{:.2}x` Zstd compression) |\n",
        format_bytes(l.osv_bytes),
        l.osv_compression_ratio
    ));
    if let (Some(b), Some(a), Some(m)) = (l.rss_before_mb, l.rss_after_build_mb, l.rss_mmap_mb) {
        md.push_str(&format!(
            "| Process RSS (Before / Post-Build / Mmap) | `{b:.1} MiB / {a:.1} MiB / {m:.1} MiB` |\n"
        ));
    }
    md.push_str(&format!(
        "| DP Range Coalescing | `{:.2}x` (`{:.3} ms` avg planner, `{:.2}` avg GETs/query) |\n",
        l.coalesce_factor, l.range_planner_avg_ms, l.avg_coalesced_gets_per_query
    ));
    md.push_str(&format!(
        "| Vector Search Latency & Throughput | `P50={:.2} ms`, `P95={:.2} ms`, `P99={:.2} ms` (`{:.1} QPS`) |\n",
        l.ann_p50_ms, l.ann_p95_ms, l.ann_p99_ms, l.ann_qps
    ));
    md.push_str(&format!(
        "| Vector Search Recall@10 vs FP32 | `{:.2}%` |\n",
        l.ann_recall_at_10_pct
    ));
    if let Some(lut) = &l.lutrank {
        md.push_str(&format!(
            "| `LUT-Rank` ColBERT (`L_d={}`, `L_q={}`) | `LUT synth={:.1} us`, `200 docs P50={:.2} ms / P99={:.2} ms`, `Pearson r={:.4}`, `Recall@10={:.2}%` |\n",
            lut.doc_tokens_ld,
            lut.query_tokens_lq,
            lut.lut_synthesis_avg_us,
            lut.rerank_200_docs_p50_ms,
            lut.rerank_200_docs_p99_ms,
            lut.pearson_correlation,
            lut.maxsim_recall_at_10_pct
        ));
    }

    md.push_str("\n## 2. Cloud Monthly TCO Comparison (Pinecone vs. Qdrant Cloud vs. Legacy HNSW vs. OspreyDB)\n\n");
    md.push_str("| Architecture | Storage / mo | Compute / mo | I/O & Tolls / mo | Rerank / mo | Total / mo | Annual / yr | OspreyDB Savings |\n");
    md.push_str("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    for p in [&t.pinecone, &t.qdrant, &t.legacy_hnsw, &t.ospreydb] {
        let sav = if p.osprey_savings_pct > 0.0 {
            format!("{:.1}%", p.osprey_savings_pct)
        } else {
            "BASELINE".to_string()
        };
        md.push_str(&format!(
            "| {} | ${:.2} / mo | ${:.2} / mo | ${:.2} / mo | ${:.2} / mo | **${:.2} / mo** | ${:.0} / yr | **{}** |\n",
            p.provider,
            p.storage_monthly_usd,
            p.compute_monthly_usd,
            p.io_and_tolls_monthly_usd,
            p.rerank_monthly_usd,
            p.total_monthly_usd,
            p.total_annual_usd,
            sav
        ));
    }

    md.push_str("\n### TCO Assumptions & Reproducibility\n\n");
    md.push_str("```bash\n");
    md.push_str(&format!(
        "osprey bench --vectors {} --dim {} --queries {} --clusters {} --nprobe {} \\\n  --project-vectors {} --project-dim {} --project-qpm {}{}\n",
        l.vectors,
        l.orig_dim,
        l.queries,
        l.clusters,
        l.nprobe,
        t.project_vectors,
        t.project_dim,
        t.project_qpm,
        if l.lutrank.is_some() { " --with-lutrank" } else { "" }
    ));
    md.push_str("```\n\n");
    md.push_str("```text\n");
    md.push_str(&format!(
        "Dataset Volume     : {:.2} GB ({} vectors x {} dims)\n",
        t.raw_dataset_gb,
        format_u64_commas(t.project_vectors),
        t.project_dim
    ));
    md.push_str(&format!(
        "OspreyDB .osi RAM  : {:.2} GB ({:.2}% RAM pruned locally)\n",
        t.osprey_osi_ram_gb, l.ram_pruned_pct
    ));
    md.push_str(&format!(
        "OspreyDB .osv S3   : {:.2} GB ($0.023/GB = ${:.2}/mo) | DP coalesce {:.2}x ({:.2} GETs/query)\n",
        t.osprey_osv_s3_gb,
        t.ospreydb.storage_monthly_usd,
        l.coalesce_factor,
        t.avg_coalesced_gets_per_query
    ));
    md.push_str("```\n");

    let pinecone_save_mo = (t.pinecone.total_monthly_usd - t.ospreydb.total_monthly_usd).max(0.0);
    let qdrant_save_mo = (t.qdrant.total_monthly_usd - t.ospreydb.total_monthly_usd).max(0.0);
    md.push_str("\n## 3. Production Readiness (OspreyDB Server — Apache-2.0)\n\n");
    md.push_str(&format!("- {}\n", format_bench_footprint_savings(l, t)));
    md.push_str(&format!(
        "- **Estimated Monthly Savings:** `{}` vs Pinecone (`{:.1}%`), `{}` vs Qdrant Cloud (`{:.1}%`)\n",
        format_usd_per_mo(pinecone_save_mo),
        t.pinecone.osprey_savings_pct,
        format_usd_per_mo(qdrant_save_mo),
        t.qdrant.osprey_savings_pct,
    ));
    md.push_str(&format!("- {OSS_SERVER_CTA}\n"));
    md
}

fn format_bench_footprint_savings(l: &LocalBenchMetrics, t: &CloudTcoProjection) -> String {
    let ram_saved_pct = if t.legacy_hnsw.ram_gb > 0.0 {
        ((t.legacy_hnsw.ram_gb - t.osprey_osi_ram_gb) / t.legacy_hnsw.ram_gb) * 100.0
    } else {
        l.ram_pruned_pct
    };
    let s3_vs_ebs_pct = if t.legacy_hnsw.storage_monthly_usd > 0.0 {
        ((t.legacy_hnsw.storage_monthly_usd - t.ospreydb.storage_monthly_usd)
            / t.legacy_hnsw.storage_monthly_usd)
            * 100.0
    } else {
        0.0
    };
    let s3_vs_managed_pct = if t.pinecone.storage_monthly_usd > 0.0 {
        ((t.pinecone.storage_monthly_usd - t.ospreydb.storage_monthly_usd)
            / t.pinecone.storage_monthly_usd)
            * 100.0
    } else {
        0.0
    };
    format!(
        "Estimated RAM & S3 Savings vs FP32: {:.2} GB .osi vs {:.2} GB FP32 HNSW ({:.1}% RAM saved) · S3 .osv {} vs {} EBS ({:.1}% saved) / {} managed ({:.1}% saved, {:.2}x Zstd)",
        t.osprey_osi_ram_gb,
        t.legacy_hnsw.ram_gb,
        ram_saved_pct,
        format_usd_per_mo(t.ospreydb.storage_monthly_usd),
        format_usd_per_mo(t.legacy_hnsw.storage_monthly_usd),
        s3_vs_ebs_pct,
        format_usd_per_mo(t.pinecone.storage_monthly_usd),
        s3_vs_managed_pct,
        l.osv_compression_ratio,
    )
}

fn format_bench_savings_summary(t: &CloudTcoProjection) -> String {
    let vs_pinecone_mo = (t.pinecone.total_monthly_usd - t.ospreydb.total_monthly_usd).max(0.0);
    let vs_qdrant_mo = (t.qdrant.total_monthly_usd - t.ospreydb.total_monthly_usd).max(0.0);
    let vs_pinecone_yr = vs_pinecone_mo * 12.0;
    format!(
        "Calculated Savings: {} ({:.1}%) vs Pinecone (${:.0}/yr) · {} ({:.1}%) vs Qdrant+GPU",
        format_usd_per_mo(vs_pinecone_mo),
        t.pinecone.osprey_savings_pct,
        vs_pinecone_yr,
        format_usd_per_mo(vs_qdrant_mo),
        t.qdrant.osprey_savings_pct,
    )
}

const OSS_SERVER_CTA: &str = "100% Apache-2.0 Open Source — unlimited vectors, built-in /v1/embeddings, Qdrant/Cohere HTTP & Arrow Flight: osprey serve --data-dir ./data --s3-uri file:///tmp/osprey-s3";

fn print_bench_human(r: &BenchReport, export_path: Option<&Path>) {
    let l = &r.local;
    let t = &r.tco;

    print_cli_header(
        &format!(
            "bench · N={} · D={} · K={} · nprobe={}",
            l.vectors, l.orig_dim, l.clusters, l.nprobe
        ),
        &format!(
            "{} · project {} x {}d ({} QPM)",
            paint_pass(&l.simd_tier),
            format_u64_commas(t.project_vectors),
            t.project_dim,
            format_u64_commas(t.project_qpm),
        ),
        &current_dir_short(),
    );

    print_section("Local Benchmark");
    print_kv(
        "Build time",
        &format!("{:.2} ms (1-Bit Binary K-Means + TSP)", l.build_time_ms),
    );
    print_kv(
        ".osi footprint",
        &format!(
            "{} ({} vs {} FP32 HNSW)",
            format_bytes(l.osi_bytes),
            paint_pass(&format!("{:.2}% RAM pruned", l.ram_pruned_pct)),
            format_bytes(l.legacy_fp32_hnsw_bytes)
        ),
    );
    print_kv(
        ".osv slabs",
        &format!(
            "{} ({:.2}x Zstd)",
            format_bytes(l.osv_bytes),
            l.osv_compression_ratio
        ),
    );
    print_kv(
        "Stage-1 policy",
        &format!(
            "nprobe {} -> {} · {} candidates{}",
            l.requested_nprobe,
            l.nprobe,
            l.candidates,
            if l.low_dim_oversample {
                paint_pass(" · low-dim adaptive")
            } else {
                String::new()
            }
        ),
    );
    print_kv(
        "DP range coalescing",
        &format!(
            "{} (planner {:.3} ms · {:.2} GETs/query)",
            paint_pass(&format!("{:.2}x", l.coalesce_factor)),
            l.range_planner_avg_ms,
            l.avg_coalesced_gets_per_query
        ),
    );
    print_kv(
        "Vector Recall@10",
        &format!(
            "{} · P50 {:.2} ms · P95 {:.2} ms · P99 {:.2} ms · {}",
            paint_pass(&format!("{:.2}%", l.ann_recall_at_10_pct)),
            l.ann_p50_ms,
            l.ann_p95_ms,
            l.ann_p99_ms,
            paint_accent(&format!("{:.1} QPS", l.ann_qps))
        ),
    );
    if let (Some(b), Some(a), Some(m)) = (l.rss_before_mb, l.rss_after_build_mb, l.rss_mmap_mb) {
        print_kv(
            "Process RSS",
            &format!("{b:.1} / {a:.1} / {m:.1} MiB (pre/build/mmap)"),
        );
    }
    if let Some(lut) = &l.lutrank {
        print_kv(
            "LUT-Rank (200d)",
            &format!(
                "Pearson {} · Recall@10 {} · synth {:.1} us · P50 {:.2} ms · P99 {:.2} ms",
                paint_pass(&format!("{:.4}", lut.pearson_correlation)),
                paint_pass(&format!("{:.2}%", lut.maxsim_recall_at_10_pct)),
                lut.lut_synthesis_avg_us,
                lut.rerank_200_docs_p50_ms,
                lut.rerank_200_docs_p99_ms
            ),
        );
    }

    print_section(&format!(
        "Cloud TCO Projection ({} vecs x {} dims · {} QPM · {})",
        format_u64_commas(t.project_vectors),
        t.project_dim,
        format_u64_commas(t.project_qpm),
        t.rerank_mode
    ));

    let mut tco_table = new_rounded_table(&[
        "Architecture",
        "Storage",
        "Compute",
        "I/O+Tolls",
        "Rerank",
        "Total",
        "Savings",
    ]);

    for (provider_label, p) in [
        ("Pinecone (Dedicated)", &t.pinecone),
        ("Qdrant Cloud (1-Bit+SSD)", &t.qdrant),
        ("Legacy HNSW (r6i)", &t.legacy_hnsw),
        ("OspreyDB (Open-Core)", &t.ospreydb),
    ] {
        let is_osprey = p.osprey_savings_pct <= 0.0;
        let provider_cell = if is_osprey {
            paint_pass(provider_label)
        } else {
            provider_label.to_string()
        };
        let total_str = format_usd_per_mo(p.total_monthly_usd);
        let total_cell = if is_osprey {
            paint_pass(&total_str)
        } else {
            total_str
        };
        let savings_cell = if p.osprey_savings_pct > 0.0 {
            paint_pass(&format!("{:.1}%", p.osprey_savings_pct))
        } else {
            paint_dim("baseline")
        };

        tco_table.add_row(vec![
            Cell::new(provider_cell),
            Cell::new(format_usd_per_mo(p.storage_monthly_usd)).set_alignment(CellAlignment::Right),
            Cell::new(format_usd_per_mo(p.compute_monthly_usd)).set_alignment(CellAlignment::Right),
            Cell::new(format_usd_per_mo(p.io_and_tolls_monthly_usd))
                .set_alignment(CellAlignment::Right),
            Cell::new(format_usd_per_mo(p.rerank_monthly_usd)).set_alignment(CellAlignment::Right),
            Cell::new(total_cell).set_alignment(CellAlignment::Right),
            Cell::new(savings_cell).set_alignment(CellAlignment::Right),
        ]);
    }
    print_indented_table(&tco_table);

    println!();
    if let Some(path) = export_path {
        println!("  {} {}", paint_pass("Report saved:"), path.display());
    }
    println!(
        "  {}",
        paint_dim("────────────────────────────────────────────────────────────────────────")
    );
    println!("  {}", paint_pass(&format_bench_footprint_savings(l, t)));
    println!("  {}", paint_pass(&format_bench_savings_summary(t)));
    println!("  {}", paint_dim(OSS_SERVER_CTA));
    println!();
}

fn generate_anisotropic_dataset(
    n: usize,
    dim: usize,
    q_count: usize,
    num_clusters: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f16>, Vec<u64>, Vec<f32>) {
    let mut rng = SplitMix64::new(seed);
    let mut centers = vec![0.0_f32; num_clusters * dim];
    for c in 0..num_clusters {
        let row = &mut centers[c * dim..(c + 1) * dim];
        for (d, slot) in row.iter_mut().enumerate() {
            #[allow(clippy::cast_precision_loss)]
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

    let mut db_f32 = vec![0.0_f32; n * dim];
    let mut db_f16 = vec![f16::ZERO; n * dim];
    let mut doc_ids = Vec::with_capacity(n);
    let k_u64 = u64::try_from(num_clusters).unwrap_or(1);

    for i in 0..n {
        doc_ids.push(u64::try_from(i).unwrap_or(0));
        let c = usize::try_from(rng.next_u64() % k_u64).unwrap_or(0);
        let center = &centers[c * dim..(c + 1) * dim];
        let row = &mut db_f32[i * dim..(i + 1) * dim];
        for (d, slot) in row.iter_mut().enumerate() {
            #[allow(clippy::cast_precision_loss)]
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
            db_f16[i * dim + d] = f16::from_f32(val);
        }
    }

    let mut queries = vec![0.0_f32; q_count * dim];
    for q in 0..q_count {
        let c = usize::try_from(rng.next_u64() % k_u64).unwrap_or(0);
        let center = &centers[c * dim..(c + 1) * dim];
        let row = &mut queries[q * dim..(q + 1) * dim];
        for (d, slot) in row.iter_mut().enumerate() {
            #[allow(clippy::cast_precision_loss)]
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

    (db_f32, db_f16, doc_ids, queries)
}

fn sample_anisotropic_tokens(rng: &mut SplitMix64, count: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; count * dim];
    for t in 0..count {
        let row = &mut out[t * dim..(t + 1) * dim];
        for (d, slot) in row.iter_mut().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let scale = if d < 4 {
                12.0_f64
            } else {
                1.0_f64 / (1.0 + (d as f64) * 0.15).sqrt()
            };
            #[allow(clippy::cast_possible_truncation)]
            {
                *slot = (next_standard_normal(rng) * scale) as f32;
            }
        }
        normalize_l2_in_place(row);
    }
    out
}

fn next_standard_normal(rng: &mut SplitMix64) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let u1 = (((rng.next_u64() >> 11) as f64) / ((1_u64 << 53) as f64)).max(1e-12);
    #[allow(clippy::cast_precision_loss)]
    let u2 = ((rng.next_u64() >> 11) as f64) / ((1_u64 << 53) as f64);
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

fn pearson_r(x: &[f32], y: &[f32]) -> f64 {
    if x.is_empty() || x.len() != y.len() {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let n = x.len() as f64;
    let mean_x: f64 = x.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let mean_y: f64 = y.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let mut num = 0.0_f64;
    let mut den_x = 0.0_f64;
    let mut den_y = 0.0_f64;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = f64::from(xi) - mean_x;
        let dy = f64::from(yi) - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    num / (den_x * den_y).sqrt().max(1e-12)
}

fn percentile_f64(sorted: &[f64], pct: usize) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) * pct) / 100;
    sorted[idx]
}

fn read_process_rss_mb() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kb: f64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kb / 1024.0);
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn format_u64_commas(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (idx, ch) in s.chars().rev().enumerate() {
        if idx > 0 && idx % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn bench_executes_anisotropic_lutrank_and_exports_md_and_json() {
        let dir = tempfile::tempdir().unwrap();
        let md_path = dir.path().join("report.md");
        let json_path = dir.path().join("report.json");

        let args = BenchArgs {
            vectors: 600,
            dim: 128,
            queries: 10,
            clusters: 16,
            nprobe: 8,
            candidates: 64,
            refine_multiplier: 0,
            with_lutrank: true,
            doc_tokens: 8,
            query_tokens: 4,
            seed: 0x4F53_5052_0001,
            project_vectors: 50_000_000,
            project_dim: 1536,
            project_qpm: 10_000_000,
            rerank_mode: RerankCostMode::Gpu,
            export_report: Some(md_path.clone()),
        };
        let ctx = CliContext {
            json: false,
            verbose: 0,
            threads: 2,
        };
        run(&args, &ctx).unwrap();

        let md_content = std::fs::read_to_string(&md_path).unwrap();
        assert!(md_content.contains("Pinecone"));
        assert!(md_content.contains("Qdrant Cloud"));
        assert!(md_content.contains("OspreyDB"));
        assert!(md_content.contains("$7.65"));
        assert!(md_content.contains("Estimated RAM & S3 Savings vs FP32:"));
        assert!(md_content.contains(OSS_SERVER_CTA));

        let mut args_json = args;
        args_json.export_report = Some(json_path.clone());
        let ctx_json = CliContext {
            json: true,
            verbose: 0,
            threads: 2,
        };
        run(&args_json, &ctx_json).unwrap();
        let parsed: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&json_path).unwrap()).unwrap();
        assert!(parsed["local"]["ann_recall_at_10_pct"].as_f64().unwrap() > 80.0);
        assert!(
            parsed["local"]["lutrank"]["pearson_correlation"]
                .as_f64()
                .unwrap()
                > 0.90
        );
        assert!(
            parsed["tco"]["ospreydb"]["total_monthly_usd"]
                .as_f64()
                .unwrap()
                < parsed["tco"]["pinecone"]["total_monthly_usd"]
                    .as_f64()
                    .unwrap()
        );
        assert!(
            parsed["tco"]["ospreydb"]["total_monthly_usd"]
                .as_f64()
                .unwrap()
                < parsed["tco"]["qdrant"]["total_monthly_usd"]
                    .as_f64()
                    .unwrap()
        );
    }
}
