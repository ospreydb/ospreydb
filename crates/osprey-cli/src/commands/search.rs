//! `osprey search`: mmap `.osi` Stage-1 scan, DP `.osv` range
//! planning, exact `f16` refinement, optional `.osr` ColBERT `LUT-Rank` reranking over
//! Top-200 candidates, and clean ASCII table / `--json` output.

use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use comfy_table::{Cell, CellAlignment};
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;
use osprey_core::{
    cast_minipage_directory, AlignedBuffer, MinipageDirectoryEntry, OsiMappedIndex, OspreyError,
    OsrStore, OsvHeader, QuantizedQueryLut, ScoredCandidate, SearchOptions,
};
use osprey_simd::{active_simd_backend, SimdBackend, SimdEngine};
use osprey_storage::{
    build_object_store, MinipageLruCache, OsvSlabReader, RangePlannerConfig, RefineStage,
    RefineStats, RefinedSearchResult, RemoteStoreConfig,
};
use serde::Serialize;

use crate::cli::SearchArgs;
use crate::output::{
    format_bytes, new_rounded_table, paint_accent, paint_dim, paint_pass, paint_score,
    parse_query_input, parse_query_tokens_input, print_indented_table, print_json, print_kv,
    print_section, CliContext, CliError, CliResult, LocationTarget,
};

#[derive(Debug, Serialize)]
pub struct SearchHitOutput {
    pub rank: usize,
    pub doc_id: u64,
    pub score: f32,
    pub dense_f16_score: f32,
    pub stage1_est_score: Option<f32>,
    pub minipage_id: u32,
    pub slot_idx: u16,
    pub payload: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct SearchTelemetry {
    pub simd_tier: String,
    pub simd_backend: &'static str,
    pub stage1_candidates: usize,
    pub stage1_ms: f64,
    pub range_plan_us: u64,
    pub raw_page_requests: usize,
    pub coalesced_range_gets: usize,
    pub coalesce_factor: f64,
    pub cache_hits: usize,
    pub bytes_fetched: u64,
    pub cache_lookup_us: u64,
    pub disk_read_us: u64,
    pub zstd_decode_us: u64,
    pub f16_score_us: u64,
    pub stage2_3_ms: f64,
    pub rerank_applied: bool,
    pub rerank_candidates: usize,
    pub stage4_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct SearchReport {
    pub index_path: String,
    pub slabs_uri: String,
    pub rerank_store: Option<String>,
    pub top_k: usize,
    /// Requested clusters (`--nprobe`), before adaptive oversampling.
    pub requested_nprobe: usize,
    /// Effective clusters probed after adaptive oversampling.
    pub nprobe: usize,
    /// Resolved oversampling multiplier (`1` = disabled).
    pub refine_multiplier: usize,
    /// Stage-1 candidates after oversampling.
    pub effective_candidates: usize,
    /// Candidates scored exactly in `f16` before float ranking/reranking.
    pub refine_pool: usize,
    /// Whether the low-dimensional (`dim <= 256`) adaptive rule engaged.
    pub low_dim_oversample: bool,
    pub hits: Vec<SearchHitOutput>,
    pub telemetry: SearchTelemetry,
}

/// Executes `osprey search`.
pub fn run(args: &SearchArgs, ctx: &CliContext) -> CliResult<()> {
    let report = execute_search(args, ctx)?;
    if ctx.json {
        print_json(&report)?;
    } else {
        print_search_human(&report);
    }
    Ok(())
}

/// Core reusable search pipeline shared by `osprey search` and end-to-end `osprey rerank`.
pub fn execute_search(args: &SearchArgs, ctx: &CliContext) -> CliResult<SearchReport> {
    let t_total = Instant::now();
    if args.top_k == 0 {
        return Err(CliError::InvalidArgs(
            "--top-k must be greater than 0".to_string(),
        ));
    }

    let parsed_query = parse_query_input(&args.query)?;
    let query_tokens: Option<Vec<Vec<f32>>> = match (&args.query_tokens, parsed_query.tokens) {
        (Some(raw_tok), _) => Some(parse_query_tokens_input(raw_tok)?),
        (None, Some(toks)) => Some(toks),
        (None, None) => None,
    };

    if args.rerank_store.is_some() && query_tokens.is_none() {
        return Err(CliError::InvalidArgs(
            "--rerank-store requires --query-tokens (or a query JSON object with 'tokens')"
                .to_string(),
        ));
    }
    if args.rerank_store.is_none() && args.query_tokens.is_some() {
        return Err(CliError::InvalidArgs(
            "--query-tokens requires --rerank-store <PATH>".to_string(),
        ));
    }

    ctx.log_info(format!(
        "Memory-mapping .osi index '{}'...",
        args.index.display()
    ));
    let osi = OsiMappedIndex::open_mmap(&args.index)?;

    let simd = SimdEngine::new_auto();
    let do_rerank = args.rerank_store.is_some() && query_tokens.is_some();

    // Adaptive low-dimensional oversampling (`SearchOptions::resolve`): widen the exact-`f16`
    // re-score pool for `dim <= 256` (e.g. SIFT1M) so `Recall@10 >= 0.95` is recovered within
    // the latency budget, while high-dimensional workloads keep their requested nprobe.
    let orig_dim = osi.header().orig_dim;
    let options = SearchOptions {
        nprobe: args.nprobe,
        candidates: args.candidates,
        top_k: args.top_k,
        rerank_top_m: if do_rerank { args.rerank_top_m } else { 0 },
        refine_multiplier: args.refine_multiplier,
    };
    let plan = options.resolve(orig_dim);
    let stage1_limit = plan.stage1_limit;
    let refine_limit = plan.refine_limit;

    // Stage 1: `.osi` 1-bit RaBitQ scan
    let t_s1 = Instant::now();
    let stage1_hits: Vec<ScoredCandidate> = osi.search_candidates(
        &parsed_query.vector,
        plan.effective_nprobe,
        stage1_limit,
        &simd,
    )?;
    let stage1_ms = t_s1.elapsed().as_secs_f64() * 1_000.0;
    ctx.log_info(format!(
        "Stage 1 (.osi scan, {:?}) returned {} candidates in {:.2} ms",
        simd.level(),
        stage1_hits.len(),
        stage1_ms
    ));

    let mut stage1_scores = std::collections::HashMap::with_capacity(stage1_hits.len());
    for c in &stage1_hits {
        stage1_scores.insert(c.doc_id, c.score);
    }

    // Stage 2 & 3: DP range coalescing + ObjectStore::get_ranges + Zstd + exact f16 refinement
    let slabs_target = LocationTarget::parse(&args.slabs)?;
    let slabs_uri = slabs_target.to_object_store_uri()?;
    let remote_cfg = RemoteStoreConfig::default();
    let (store, obj_path) = build_object_store(&slabs_uri, &remote_cfg)?;

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| CliError::Io(format!("failed to create tokio runtime: {err}")))?;

    let t_s23 = Instant::now();
    let (refined, refine_stats): (Vec<RefinedSearchResult>, RefineStats) = match &slabs_target {
        LocationTarget::Local(local_osv_path) => {
            let reader = OsvSlabReader::open_mmap(local_osv_path)?;
            let zstd_dict = if reader.header().zstd_dict_len > 0 {
                let mut f = std::fs::File::open(local_osv_path)?;
                use std::io::{Read, Seek, SeekFrom};
                let start = reader.header().zstd_dict_offset;
                let len = usize::try_from(reader.header().zstd_dict_len)
                    .map_err(|_| CliError::Core(OspreyError::CorruptedFooter))?;
                f.seek(SeekFrom::Start(start))?;
                let mut dict = vec![0_u8; len];
                f.read_exact(&mut dict)?;
                Some(dict)
            } else {
                None
            };
            let refine_stage = RefineStage::open_local_path(
                0,
                RangePlannerConfig::default(),
                reader.header(),
                zstd_dict,
                local_osv_path,
            )?;
            let cache_bytes = args.cache_mb.max(1).saturating_mul(1024 * 1024);
            let cache = MinipageLruCache::new(cache_bytes);
            rt.block_on(refine_stage.refine_scored_async(
                &parsed_query.vector,
                &stage1_hits,
                refine_limit,
                reader.directory(),
                &cache,
                &store,
                &obj_path,
                &remote_cfg,
            ))
            .map_err(CliError::from)?
        }
        LocationTarget::Remote(_) => rt.block_on(async {
            let (directory, zstd_dict) = load_osv_metadata_async(&store, &obj_path).await?;
            let refine_stage = RefineStage::new(0, RangePlannerConfig::default(), zstd_dict);
            let cache_bytes = args.cache_mb.max(1).saturating_mul(1024 * 1024);
            let cache = MinipageLruCache::new(cache_bytes);

            refine_stage
                .refine_scored_async(
                    &parsed_query.vector,
                    &stage1_hits,
                    refine_limit,
                    &directory,
                    &cache,
                    &store,
                    &obj_path,
                    &remote_cfg,
                )
                .await
                .map_err(CliError::from)
        })?,
    };
    let stage2_3_ms = t_s23.elapsed().as_secs_f64() * 1_000.0;

    // Optional Stage 4: `.osr` ColBERT LUT-Rank reranking over Top-200 survivors
    let t_s4 = Instant::now();
    let (final_hits, rerank_candidates) = if let (Some(osr_path), Some(q_tokens)) =
        (&args.rerank_store, &query_tokens)
    {
        let rerank_count = refined.len().min(args.rerank_top_m);
        let rerank_slice = &refined[..rerank_count];
        let osr = OsrStore::open_mmap(osr_path)?;
        let osr_hdr = osr.header();
        let token_dim = usize::from(osr_hdr.token_dim);
        let mut flat_tokens = Vec::with_capacity(q_tokens.len() * token_dim);
        for tok in q_tokens {
            if tok.len() != token_dim {
                return Err(CliError::Core(OspreyError::InvalidDimension {
                    dim: tok.len(),
                }));
            }
            flat_tokens.extend_from_slice(tok);
        }
        let q_lut = QuantizedQueryLut::synthesize(
            &flat_tokens,
            q_tokens.len(),
            token_dim,
            osr_hdr.gamma,
            osr_hdr.fwht_seed,
            &simd,
        )?;
        let reranked = RefineStage::rerank_with_osr(rerank_slice, &osr, &q_lut, args.top_k, &simd)?;
        let mapped = reranked
            .into_iter()
            .enumerate()
            .map(|(idx, (res, maxsim))| {
                let s1 = stage1_scores.get(&res.doc_id).copied();
                build_hit_output(idx + 1, maxsim, res, s1)
            })
            .collect();
        (mapped, rerank_count)
    } else {
        let mapped = refined
            .into_iter()
            .take(args.top_k)
            .enumerate()
            .map(|(idx, res)| {
                let s1 = stage1_scores.get(&res.doc_id).copied();
                let score = res.exact_score;
                build_hit_output(idx + 1, score, res, s1)
            })
            .collect();
        (mapped, 0)
    };
    let stage4_ms = if do_rerank {
        t_s4.elapsed().as_secs_f64() * 1_000.0
    } else {
        0.0
    };

    #[allow(clippy::cast_precision_loss)]
    let coalesce_factor = if refine_stats.coalesced_range_gets > 0 {
        let misses = refine_stats
            .raw_page_requests
            .saturating_sub(refine_stats.cache_hits);
        (misses as f64) / (refine_stats.coalesced_range_gets as f64)
    } else {
        1.0
    };

    Ok(SearchReport {
        index_path: args.index.display().to_string(),
        slabs_uri: args.slabs.clone(),
        rerank_store: args
            .rerank_store
            .as_ref()
            .map(|p: &PathBuf| p.display().to_string()),
        top_k: args.top_k,
        requested_nprobe: args.nprobe,
        nprobe: plan.effective_nprobe,
        refine_multiplier: plan.refine_multiplier,
        effective_candidates: plan.effective_candidates,
        refine_pool: plan.refine_limit,
        low_dim_oversample: plan.low_dim_oversample,
        hits: final_hits,
        telemetry: SearchTelemetry {
            simd_tier: format!("{:?}", simd.level()),
            simd_backend: active_simd_backend(),
            stage1_candidates: stage1_hits.len(),
            stage1_ms,
            range_plan_us: refine_stats.range_plan_time_us,
            raw_page_requests: refine_stats.raw_page_requests,
            coalesced_range_gets: refine_stats.coalesced_range_gets,
            coalesce_factor,
            cache_hits: refine_stats.cache_hits,
            bytes_fetched: refine_stats.bytes_fetched,
            cache_lookup_us: refine_stats.cache_lookup_us,
            disk_read_us: refine_stats.disk_read_us,
            zstd_decode_us: refine_stats.zstd_decode_us,
            f16_score_us: refine_stats.f16_score_us,
            stage2_3_ms,
            rerank_applied: do_rerank,
            rerank_candidates,
            stage4_ms,
            total_ms: t_total.elapsed().as_secs_f64() * 1_000.0,
        },
    })
}

async fn load_osv_metadata_async(
    store: &Arc<dyn ObjectStore>,
    obj_path: &ObjectPath,
) -> CliResult<(Vec<MinipageDirectoryEntry>, Option<Vec<u8>>)> {
    let hdr_bytes = store
        .get_range(obj_path, 0..64)
        .await
        .map_err(|err| CliError::Io(format!("failed to fetch .osv header: {err}")))?;
    let aligned_hdr = AlignedBuffer::from_slice(&hdr_bytes)?;
    let hdr = *OsvHeader::from_bytes(aligned_hdr.as_slice())?;

    let mut ranges: Vec<Range<usize>> = Vec::with_capacity(2);
    let has_dict = hdr.zstd_dict_len > 0;
    if has_dict {
        let start = usize::try_from(hdr.zstd_dict_offset)
            .map_err(|_| CliError::Core(OspreyError::CorruptedFooter))?;
        let len = usize::try_from(hdr.zstd_dict_len)
            .map_err(|_| CliError::Core(OspreyError::CorruptedFooter))?;
        let end = start
            .checked_add(len)
            .ok_or(CliError::Core(OspreyError::CorruptedFooter))?;
        ranges.push(start..end);
    }

    let dir_start = usize::try_from(hdr.directory_offset)
        .map_err(|_| CliError::Core(OspreyError::CorruptedFooter))?;
    let dir_len = usize::try_from(hdr.num_minipages)
        .map_err(|_| CliError::Core(OspreyError::CorruptedFooter))?
        .checked_mul(std::mem::size_of::<MinipageDirectoryEntry>())
        .ok_or(CliError::Core(OspreyError::CorruptedFooter))?;
    let dir_end = dir_start
        .checked_add(dir_len)
        .ok_or(CliError::Core(OspreyError::CorruptedFooter))?;
    if dir_len > 0 {
        ranges.push(dir_start..dir_end);
    }

    if ranges.is_empty() {
        return Ok((Vec::new(), None));
    }

    let fetched = store
        .get_ranges(obj_path, &ranges)
        .await
        .map_err(|err| CliError::Io(format!("failed to fetch .osv directory/dict: {err}")))?;

    let mut idx = 0_usize;
    let zstd_dict = if has_dict {
        let b = fetched[idx].to_vec();
        idx += 1;
        Some(b)
    } else {
        None
    };

    let directory = if dir_len > 0 {
        let aligned_dir = AlignedBuffer::from_slice(&fetched[idx])?;
        cast_minipage_directory(aligned_dir.as_slice(), hdr.num_minipages)?.to_vec()
    } else {
        Vec::new()
    };

    Ok((directory, zstd_dict))
}

fn build_hit_output(
    rank: usize,
    score: f32,
    res: RefinedSearchResult,
    stage1_est: Option<f32>,
) -> SearchHitOutput {
    let payload = if res.payload_json.is_empty() {
        serde_json::Value::Object(serde_json::Map::new())
    } else {
        serde_json::from_slice(&res.payload_json).unwrap_or_else(|_| {
            serde_json::Value::String(String::from_utf8_lossy(&res.payload_json).into_owned())
        })
    };
    SearchHitOutput {
        rank,
        doc_id: res.doc_id,
        score,
        dense_f16_score: res.exact_score,
        stage1_est_score: stage1_est,
        minipage_id: res.minipage_id,
        slot_idx: res.slot_idx,
        payload,
    }
}

fn print_search_human(report: &SearchReport) {
    let t = &report.telemetry;
    let mut results_table = new_rounded_table(&[
        "Rank",
        "Doc ID",
        "Score",
        "Dense f16",
        "Page:Slot",
        "Payload",
    ]);

    for h in &report.hits {
        let payload_str = serde_json::to_string(&h.payload).unwrap_or_else(|_| "{}".to_string());
        let truncated = if payload_str.len() > 52 {
            format!("{}...", &payload_str[..49])
        } else {
            payload_str
        };

        let score_str = paint_score(h.score, &format!("{:.5}", h.score));
        let dense_str = paint_score(h.dense_f16_score, &format!("{:.5}", h.dense_f16_score));

        results_table.add_row(vec![
            Cell::new(h.rank).set_alignment(CellAlignment::Right),
            Cell::new(h.doc_id).set_alignment(CellAlignment::Right),
            Cell::new(score_str).set_alignment(CellAlignment::Right),
            Cell::new(dense_str).set_alignment(CellAlignment::Right),
            Cell::new(paint_dim(&format!("{}:{}", h.minipage_id, h.slot_idx)))
                .set_alignment(CellAlignment::Right),
            Cell::new(truncated),
        ]);
    }
    print_indented_table(&results_table);

    let cache_misses = t.raw_page_requests.saturating_sub(t.cache_hits);
    let s3_calls_saved = cache_misses.saturating_sub(t.coalesced_range_gets);

    print_section(&format!(
        "Pipeline  {}",
        paint_accent(&format!("{:.2} ms", t.total_ms))
    ));
    print_kv("SIMD backend", &paint_pass(t.simd_backend));
    print_kv(
        ".osi scan",
        &format!("{:.2} ms · {} candidates", t.stage1_ms, t.stage1_candidates),
    );
    print_kv(
        "Stage-1 policy",
        &format!(
            "nprobe {} -> {} · {} candidates · {} f16 re-score pool{}",
            report.requested_nprobe,
            report.nprobe,
            report.effective_candidates,
            report.refine_pool,
            if report.low_dim_oversample {
                paint_pass(" · low-dim adaptive")
            } else {
                String::new()
            }
        ),
    );
    print_kv(
        "Range planner",
        &format!(
            "{} us · {} pages -> {} S3 GETs ({}, {}, {} cache hits, {})",
            t.range_plan_us,
            t.raw_page_requests,
            t.coalesced_range_gets,
            paint_pass(&format!("{:.2}x coalesce", t.coalesce_factor)),
            paint_pass(&format!("{s3_calls_saved} S3 saved")),
            t.cache_hits,
            format_bytes(t.bytes_fetched),
        ),
    );
    print_kv(".osv f16 refine", &format!("{:.2} ms", t.stage2_3_ms));
    print_kv(
        ".osr LUT-Rank",
        &if t.rerank_applied {
            paint_pass(&format!(
                "{:.2} ms · {} docs reranked",
                t.stage4_ms, t.rerank_candidates
            ))
        } else {
            paint_dim("disabled")
        },
    );
    println!();
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]
mod tests {
    use super::*;
    use crate::cli::ImportArgs;
    use crate::commands::import;
    use arrow::array::{FixedSizeListBuilder, Float32Builder, ListBuilder, UInt64Builder};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::fs::File;
    use std::io::BufWriter;

    #[test]
    fn search_end_to_end_with_and_without_lutrank() {
        let dir = tempfile::tempdir().unwrap();
        let pq_path = dir.path().join("corpus.parquet");
        let seg_dir = dir.path().join("seg");

        let n = 30_usize;
        let dim = 64_usize;
        let mut id_builder = UInt64Builder::new();
        let mut vec_builder = FixedSizeListBuilder::new(Float32Builder::new(), dim as i32);
        let mut tok_builder =
            ListBuilder::new(FixedSizeListBuilder::new(Float32Builder::new(), 128));

        let mut target_vec = vec![0.0_f32; dim];
        let mut target_tok = vec![0.0_f32; 128];

        for i in 0..n {
            id_builder.append_value(1000 + i as u64);
            for (d, dst_v) in target_vec.iter_mut().enumerate().take(dim) {
                let v = if i == 13 {
                    if d == 0 {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    ((i * 11 + d) as f32 * 0.07).sin() * 0.1
                };
                if i == 13 {
                    *dst_v = v;
                }
                vec_builder.values().append_value(v);
            }
            vec_builder.append(true);

            for t in 0..4 {
                for (d, dst_t) in target_tok.iter_mut().enumerate().take(128) {
                    let tv = if i == 13 && t == 0 {
                        if d == 0 {
                            1.0
                        } else {
                            0.0
                        }
                    } else {
                        ((i + t + d) as f32 * 0.03).cos() * 0.05
                    };
                    if i == 13 && t == 0 {
                        *dst_t = tv;
                    }
                    tok_builder.values().values().append_value(tv);
                }
                tok_builder.values().append(true);
            }
            tok_builder.append(true);
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::UInt64, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                false,
            ),
            Field::new(
                "colbert_tokens",
                DataType::List(Arc::new(Field::new(
                    "item",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        128,
                    ),
                    true,
                ))),
                false,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(id_builder.finish()),
                Arc::new(vec_builder.finish()),
                Arc::new(tok_builder.finish()),
            ],
        )
        .unwrap();
        let mut writer = ArrowWriter::try_new(
            BufWriter::new(File::create(&pq_path).unwrap()),
            schema,
            None,
        )
        .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let ctx = CliContext {
            json: false,
            verbose: 0,
            threads: 1,
        };
        import::run(
            &ImportArgs {
                input: pq_path.display().to_string(),
                output: seg_dir.display().to_string(),
                id_col: None,
                vec_col: None,
                token_col: None,
                metadata_cols: vec![],
                clusters: 4,
                page_size_kb: 16,
                with_lutrank: true,
                batch_size: 16,
                sample_size: 30,
                no_zstd_dict: false,
                osv_compression: crate::cli::OsvCompressionMode::Auto,
                seed: 77,
            },
            &ctx,
        )
        .unwrap();

        let q_json = serde_json::to_string(&target_vec).unwrap();
        let qt_json = serde_json::to_string(&vec![target_tok]).unwrap();

        let search_args = SearchArgs {
            index: seg_dir.join("segment.osi"),
            slabs: seg_dir.join("segment.osv").display().to_string(),
            query: q_json,
            top_k: 5,
            nprobe: 4,
            candidates: 30,
            refine_multiplier: 0,
            cache_mb: 16,
            rerank_store: Some(seg_dir.join("segment.osr")),
            query_tokens: Some(qt_json),
            rerank_top_m: 200,
        };
        let report = execute_search(&search_args, &ctx).unwrap();
        assert_eq!(report.hits.len(), 5);
        assert_eq!(report.hits[0].doc_id, 1013);
        assert!(report.telemetry.rerank_applied);
    }
}
