//! `osprey import`: ingests local (`file://`, `.parquet`, `.jsonl`, `.npy`)
//! or remote datasets using the out-of-core streaming pipeline, with `indicatif` progress
//! bars, generating `.osi`, `.osv`, and optional `.osr` artifacts.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor, Seek, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{
    Array, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, LargeStringArray,
    StringArray, TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
    TimestampSecondArray, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, TimeUnit};
use arrow::record_batch::RecordBatch;
use futures::StreamExt;
use half::f16;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use object_store::ObjectStore;
use osprey_core::math::SplitMix64;
use osprey_core::{
    AlignedBuffer, BinaryKMeansResult, OsiIndex, OspreyError, OsrDocumentTokens, OsrStore,
    VectorLocation, DEFAULT_LUTRANK_GAMMA, LUTRANK_TOKEN_DIM,
};
use osprey_simd::SimdEngine;
use osprey_storage::{
    build_object_store, extract_batch_colbert_tokens_f32, extract_batch_doc_ids,
    extract_batch_vectors_f32, IngestColumnHints, JsonlRecordReader, NpyMatrixReader,
    OsvVectorRecord, OsvWriteReport, OsvWriterConfig, OutOfCoreIngestConfig,
    OutOfCoreIngestPipeline, ParquetBatchReader, ParquetBatchStream, RemoteStoreConfig,
    StreamingOsvWriter,
};
use roaring::RoaringBitmap;
use serde::Serialize;

use crate::cli::{ImportArgs, OsvCompressionMode};
use crate::commands::delete::atomic_write_file;
use crate::output::{
    format_bytes, paint_accent, paint_pass, print_json, print_kv, print_section,
    validate_finite_vec, CliContext, CliError, CliResult, LocationTarget,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputFormat {
    Parquet,
    Jsonl,
    Npy,
}

#[derive(Debug, Serialize)]
struct ImportReport {
    input: String,
    input_format: String,
    output: String,
    vectors_indexed: u64,
    orig_dim: usize,
    padded_dim: usize,
    clusters: usize,
    page_size_kb: usize,
    minipages: usize,
    zstd_dict_bytes: usize,
    osi_path: String,
    osi_bytes: u64,
    osv_path: String,
    osv_bytes: u64,
    osv_compression_ratio: f64,
    osr_path: Option<String>,
    osr_bytes: Option<u64>,
    osr_docs: Option<u64>,
    osr_tokens: Option<u64>,
    elapsed_ms: f64,
}

struct IngestedCorpus {
    orig_dim: usize,
    doc_ids: Vec<u64>,
    vectors_f16: Vec<f16>,
    tsp_assignments: Vec<u32>,
    locations: Vec<VectorLocation>,
    colbert_tokens: Vec<Vec<f32>>,
    colbert_token_counts: Vec<usize>,
    has_colbert: bool,
    codebook: osprey_storage::TrainedLakeCodebook,
    osv_report: OsvWriteReport,
}

/// Executes `osprey import`.
pub fn run(args: &ImportArgs, ctx: &CliContext) -> CliResult<()> {
    let t_start = Instant::now();
    if args.clusters == 0 {
        return Err(CliError::InvalidArgs(
            "--clusters must be greater than 0".to_string(),
        ));
    }
    if args.page_size_kb == 0 {
        return Err(CliError::InvalidArgs(
            "--page-size-kb must be greater than 0".to_string(),
        ));
    }
    if args.batch_size == 0 {
        return Err(CliError::InvalidArgs(
            "--batch-size must be greater than 0".to_string(),
        ));
    }
    if args.bits != 1 && args.bits != 4 {
        return Err(CliError::InvalidArgs("--bits must be 1 or 4".to_string()));
    }

    let input_target = LocationTarget::parse(&args.input)?;
    let output_target = LocationTarget::parse(&args.output)?;
    let format = detect_input_format(&args.input)?;
    let want_lutrank = args.with_lutrank || args.token_col.is_some();

    if want_lutrank && format == InputFormat::Npy {
        return Err(CliError::InvalidArgs(
            "--with-lutrank is unsupported for .npy 2D dense matrices; use .parquet or .jsonl with ColBERT tokens"
                .to_string(),
        ));
    }

    let pb = create_progress_bar(ctx.json);
    pb.set_message("[1/4] Sampling and inspecting dataset...");

    let hints = IngestColumnHints {
        id_column: args.id_col.clone(),
        vector_column: args.vec_col.clone(),
        colbert_column: args.token_col.clone(),
        metadata_columns: args.metadata_cols.clone(),
        expected_vector_dim: None,
        expected_colbert_dim: want_lutrank.then_some(LUTRANK_TOKEN_DIM),
    };

    let use_zstd = match args.osv_compression {
        OsvCompressionMode::Auto => matches!(output_target, LocationTarget::Remote(_)),
        OsvCompressionMode::None => false,
        OsvCompressionMode::Zstd => true,
    };

    let osv_config = if use_zstd {
        OsvWriterConfig {
            minipage_target_bytes: args.page_size_kb * 1024,
            enable_zstd_dict: !args.no_zstd_dict,
            ..Default::default()
        }
    } else {
        OsvWriterConfig {
            minipage_target_bytes: args.page_size_kb * 1024,
            ..OsvWriterConfig::uncompressed()
        }
    };

    let k_clusters = usize::try_from(args.clusters).unwrap_or(256);
    let effective_sample_size = if args.sample_size == 0 {
        osprey_core::kmeans_subsample_target(usize::MAX, k_clusters)
    } else {
        args.sample_size.max(1)
    };
    let pipeline = OutOfCoreIngestPipeline::new(OutOfCoreIngestConfig {
        max_training_sample_vectors: effective_sample_size,
        batch_size_rows: args.batch_size,
        num_clusters: k_clusters,
        fwht_seed: args.seed,
        reservoir_seed: args.seed ^ 0x9E37_79B9_7F4A_7C15,
        hints: hints.clone(),
        osv_config: osv_config.clone(),
        ..Default::default()
    });

    // Prepare staging dir (either final local output dir or temp dir before remote upload).
    let (stage_dir, _temp_guard) = match &output_target {
        LocationTarget::Local(dir) => {
            std::fs::create_dir_all(dir)?;
            (dir.clone(), None)
        }
        LocationTarget::Remote(_) => {
            let tmp = tempfile::tempdir()?;
            let path = tmp.path().to_path_buf();
            (path, Some(tmp))
        }
    };

    let osi_local = stage_dir.join("segment.osi");
    let osv_local = stage_dir.join("segment.osv");
    let osr_local = stage_dir.join("segment.osr");

    // Stream Pass 1 + BinaryKMeans training + streaming Pass 2 (`.osv` blocks written per chunk).
    let corpus = match (&input_target, format) {
        (LocationTarget::Local(path), InputFormat::Parquet) => ingest_local_parquet(
            path,
            &pipeline,
            &hints,
            want_lutrank,
            &osv_local,
            &osv_config,
            &pb,
            ctx,
        )?,
        (LocationTarget::Remote(uri), InputFormat::Parquet) => ingest_remote_parquet(
            uri,
            &pipeline,
            &hints,
            want_lutrank,
            &osv_local,
            &osv_config,
            &pb,
            ctx,
        )?,
        (LocationTarget::Local(path), InputFormat::Jsonl) => ingest_local_jsonl(
            path,
            &pipeline,
            want_lutrank,
            &osv_local,
            &osv_config,
            &pb,
            ctx,
        )?,
        (LocationTarget::Remote(uri), InputFormat::Jsonl) => {
            let raw_bytes = fetch_remote_bytes(uri)?;
            ingest_jsonl_from_bytes(
                &raw_bytes,
                &pipeline,
                want_lutrank,
                &osv_local,
                &osv_config,
                &pb,
                ctx,
            )?
        }
        (LocationTarget::Local(path), InputFormat::Npy) => {
            ingest_local_npy(path, &pipeline, &osv_local, &osv_config, &pb, ctx)?
        }
        (LocationTarget::Remote(uri), InputFormat::Npy) => {
            let raw_bytes = fetch_remote_bytes(uri)?;
            ingest_npy_from_bytes(&raw_bytes, &pipeline, &osv_local, &osv_config, &pb, ctx)?
        }
    };

    if want_lutrank && !corpus.has_colbert {
        pb.finish_and_clear();
        return Err(CliError::InvalidArgs(
            "--with-lutrank requested, but the input dataset contained no ColBERT multi-vector tokens"
                .to_string(),
        ));
    }

    let n = corpus.doc_ids.len();
    if n == 0 {
        pb.finish_and_clear();
        return Err(CliError::InvalidArgs(
            "input dataset contained zero indexable vectors".to_string(),
        ));
    }

    let osv_report = &corpus.osv_report;
    let osv_bytes = std::fs::metadata(&osv_local)?.len();

    // Step 4a: Build & atomically seal `.osi` index from the retained `f16` corpus (no `f32`).
    pb.set_message("[4/4] Sealing artifacts and xxHash64 checksums (.osi/.osv/.osr)...");
    let simd = SimdEngine::new_auto();
    let padded_dim = corpus.codebook.padded_dim;
    let k = corpus.codebook.num_clusters;

    let mut aligned_centroids = AlignedBuffer::<f16>::new(corpus.codebook.centroids_f16.len())?;
    aligned_centroids
        .as_mut_slice()
        .copy_from_slice(&corpus.codebook.centroids_f16);
    let words_per_vec = padded_dim / 64;
    let mut binary_centroids = vec![0_u64; k * words_per_vec];
    for c in 0..k {
        let cent_slice = &corpus.codebook.centroids_f16[c * padded_dim..(c + 1) * padded_dim];
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

    let km_result = BinaryKMeansResult {
        centroids_f16: aligned_centroids,
        binary_centroids,
        assignments: corpus.tsp_assignments.clone(),
        final_hamming_inertia: 0,
        iterations_run: 1,
    };

    let osi_index = OsiIndex::build_from_f16_with_bits(
        &corpus.vectors_f16,
        &corpus.doc_ids,
        &corpus.locations,
        &km_result,
        corpus.orig_dim,
        args.seed,
        RoaringBitmap::new(),
        args.bits,
        &simd,
    )?;
    let osi_aligned = osi_index.to_aligned_bytes()?;
    atomic_write_file(&osi_local, osi_aligned.as_slice())?;
    let osi_bytes = u64::try_from(osi_aligned.len()).unwrap_or(u64::MAX);

    // Step 4b: Optionally build & seal `.osr` LUT-Rank store.
    let (osr_out_path, osr_bytes, osr_docs, osr_tokens) = if want_lutrank && corpus.has_colbert {
        pb.set_message("[4/4] Sealing .osr LUT-Rank store and xxHash64 checksum...");
        let mut osr_inputs = Vec::with_capacity(n);
        for i in 0..n {
            let num_tokens = corpus.colbert_token_counts[i];
            if num_tokens > 0 {
                osr_inputs.push(OsrDocumentTokens {
                    doc_id: corpus.doc_ids[i],
                    tokens: &corpus.colbert_tokens[i],
                    num_tokens,
                });
            }
        }
        let osr_built = OsrStore::build(
            &osr_inputs,
            LUTRANK_TOKEN_DIM,
            DEFAULT_LUTRANK_GAMMA,
            args.seed,
            &simd,
        )?;
        let osr_aligned = osr_built.to_aligned_bytes()?;
        atomic_write_file(&osr_local, osr_aligned.as_slice())?;
        let osr_len = u64::try_from(osr_aligned.len()).unwrap_or(u64::MAX);
        let hdr = osr_built.header();
        (
            Some(osr_local.display().to_string()),
            Some(osr_len),
            Some(hdr.num_docs),
            Some(hdr.total_tokens),
        )
    } else {
        (None, None, None, None)
    };

    // Upload artifacts if `--output` is a remote Object Storage URI.
    let (final_osi_path, final_osv_path, final_osr_path) = match &output_target {
        LocationTarget::Local(_) => (
            osi_local.display().to_string(),
            osv_local.display().to_string(),
            osr_out_path,
        ),
        LocationTarget::Remote(base_uri) => {
            pb.set_message(format!("Uploading segments to {base_uri}..."));
            let osi_uri = join_remote_segment_uri(base_uri, "segment.osi");
            let osv_uri = join_remote_segment_uri(base_uri, "segment.osv");
            upload_file_to_remote(&osi_local, &osi_uri)?;
            upload_file_to_remote(&osv_local, &osv_uri)?;
            let osr_uri = if osr_out_path.is_some() {
                let u = join_remote_segment_uri(base_uri, "segment.osr");
                upload_file_to_remote(&osr_local, &u)?;
                Some(u)
            } else {
                None
            };
            (osi_uri, osv_uri, osr_uri)
        }
    };

    pb.finish_and_clear();

    #[allow(clippy::cast_precision_loss)]
    let osv_compression_ratio = if osv_report.total_compressed_bytes > 0 {
        (osv_report.total_uncompressed_bytes as f64) / (osv_report.total_compressed_bytes as f64)
    } else {
        1.0
    };

    let report = ImportReport {
        input: args.input.clone(),
        input_format: match format {
            InputFormat::Parquet => "parquet".to_string(),
            InputFormat::Jsonl => "jsonl".to_string(),
            InputFormat::Npy => "npy".to_string(),
        },
        output: args.output.clone(),
        vectors_indexed: u64::try_from(n).unwrap_or(u64::MAX),
        orig_dim: corpus.orig_dim,
        padded_dim,
        clusters: k,
        page_size_kb: args.page_size_kb,
        minipages: osv_report.directory.len(),
        zstd_dict_bytes: osv_report.zstd_dict_bytes,
        osi_path: final_osi_path,
        osi_bytes,
        osv_path: final_osv_path,
        osv_bytes,
        osv_compression_ratio,
        osr_path: final_osr_path,
        osr_bytes,
        osr_docs,
        osr_tokens,
        elapsed_ms: t_start.elapsed().as_secs_f64() * 1_000.0,
    };

    if ctx.json {
        print_json(&report)?;
    } else {
        print_import_human(&report);
    }

    Ok(())
}

fn create_progress_bar(hidden: bool) -> ProgressBar {
    let pb = if hidden {
        ProgressBar::hidden()
    } else {
        ProgressBar::with_draw_target(None, ProgressDrawTarget::stderr())
    };
    if let Ok(style) = ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} chunks ({eta}) {msg}",
    ) {
        pb.set_style(style.progress_chars("=>-"));
    }
    pb.enable_steady_tick(std::time::Duration::from_millis(120));
    pb
}

fn detect_input_format(raw: &str) -> CliResult<InputFormat> {
    let clean = raw.split('?').next().unwrap_or(raw).to_ascii_lowercase();
    if clean.ends_with(".parquet") || clean.ends_with(".pq") {
        Ok(InputFormat::Parquet)
    } else if clean.ends_with(".jsonl") || clean.ends_with(".ndjson") {
        Ok(InputFormat::Jsonl)
    } else if clean.ends_with(".npy") {
        Ok(InputFormat::Npy)
    } else {
        Err(CliError::InvalidArgs(format!(
            "cannot detect format from '{raw}': expected .parquet, .jsonl, or .npy"
        )))
    }
}

#[allow(clippy::too_many_arguments)]
fn ingest_local_parquet(
    path: &Path,
    pipeline: &OutOfCoreIngestPipeline,
    hints: &IngestColumnHints,
    want_lutrank: bool,
    osv_path: &Path,
    osv_config: &OsvWriterConfig,
    pb: &ProgressBar,
    ctx: &CliContext,
) -> CliResult<IngestedCorpus> {
    pb.set_message("[1/4] Sampling and inspecting dataset (.parquet)...");
    let (detected_schema, reservoir_f32, sampled_count, total_seen) =
        pipeline.run_pass1_reservoir_sampling(File::open(path)?)?;
    ctx.log_info(format!(
        "Parquet Pass 1 complete: seen={total_seen}, sampled={sampled_count}, dim={}",
        detected_schema.orig_dim
    ));

    pb.set_message(format!(
        "[2/4] Training 1-Bit Binary K-Means (K={}) + TSP...",
        pipeline.config.num_clusters.min(sampled_count)
    ));
    let codebook =
        pipeline.train_codebook(&reservoir_f32, sampled_count, detected_schema.orig_dim)?;
    drop(reservoir_f32);

    let batch_rows_u64 = u64::try_from(pipeline.config.batch_size_rows.max(1)).unwrap_or(1);
    pb.set_length(total_seen.div_ceil(batch_rows_u64).max(1));
    pb.set_position(0);
    pb.set_message("[3/4] Writing Zstd minipage batches (.parquet)...");

    let mut reader = ParquetBatchReader::open(path, hints, pipeline.config.batch_size_rows)?;
    let col_map = reader.projected_map().clone();
    let projected_schema = reader.projected_schema().clone();
    let total_cap = usize::try_from(total_seen).unwrap_or(0);

    let mut writer = create_streaming_osv_writer(osv_path, osv_config, codebook.orig_dim)?;
    let mut acc = StreamingAccumulator::new(total_cap, codebook.orig_dim, codebook.padded_dim);
    for batch_res in &mut reader {
        let batch = batch_res?;
        acc.push_arrow_batch(
            &batch,
            &col_map,
            projected_schema.as_ref(),
            pipeline,
            &codebook,
            want_lutrank,
            &mut writer,
        )?;
        pb.inc(1);
    }
    let osv_report = writer.finish()?;

    acc.into_corpus(codebook, osv_report)
}

#[allow(clippy::too_many_arguments)]
fn ingest_remote_parquet(
    uri: &str,
    pipeline: &OutOfCoreIngestPipeline,
    hints: &IngestColumnHints,
    want_lutrank: bool,
    osv_path: &Path,
    osv_config: &OsvWriterConfig,
    pb: &ProgressBar,
    ctx: &CliContext,
) -> CliResult<IngestedCorpus> {
    let remote_cfg = RemoteStoreConfig::default();
    let (store, obj_path) = build_object_store(uri, &remote_cfg)?;
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| CliError::Io(format!("failed to create tokio runtime: {err}")))?;

    rt.block_on(async {
        pb.set_message("[1/4] Pass-1 Remote Parquet Reservoir Sampling...");
        let vec_only_hints = IngestColumnHints {
            id_column: hints.id_column.clone(),
            vector_column: hints.vector_column.clone(),
            colbert_column: None,
            metadata_columns: Vec::new(),
            expected_vector_dim: hints.expected_vector_dim,
            expected_colbert_dim: None,
        };
        let mut stream1 = ParquetBatchStream::from_object_store(
            Arc::clone(&store),
            &obj_path,
            &vec_only_hints,
            pipeline.config.batch_size_rows,
        )
        .await?;

        let cap = pipeline.config.max_training_sample_vectors.max(1);
        let mut rng = SplitMix64::new(pipeline.config.reservoir_seed);
        let mut inferred_dim =
            (stream1.detected_schema().orig_dim > 0).then_some(stream1.detected_schema().orig_dim);
        let vec_batch_idx = stream1.projected_map().vector_batch_idx;
        let mut reservoir: Vec<f32> = Vec::new();
        let mut sampled_count = 0_usize;
        let mut total_seen = 0_u64;

        while let Some(batch_res) = stream1.next().await {
            let batch = batch_res?;
            if batch.num_rows() == 0 {
                continue;
            }
            let (batch_dim, batch_vecs) =
                extract_batch_vectors_f32(&batch, vec_batch_idx, inferred_dim)?;
            if inferred_dim.is_none() {
                inferred_dim = Some(batch_dim);
                reservoir.reserve(cap.saturating_mul(batch_dim));
            }
            for r in 0..batch.num_rows() {
                let slice = &batch_vecs[r * batch_dim..(r + 1) * batch_dim];
                if sampled_count < cap {
                    reservoir.extend_from_slice(slice);
                    sampled_count += 1;
                } else {
                    let j_u64 = rng.next_u64() % (total_seen + 1);
                    let j = usize::try_from(j_u64).unwrap_or(usize::MAX);
                    if j < cap {
                        reservoir[j * batch_dim..(j + 1) * batch_dim].copy_from_slice(slice);
                    }
                }
                total_seen += 1;
            }
        }
        let orig_dim = inferred_dim.ok_or_else(|| {
            CliError::InvalidArgs("remote Parquet dataset contained zero vectors".to_string())
        })?;
        ctx.log_info(format!(
            "Remote Parquet Pass 1 complete: seen={total_seen}, sampled={sampled_count}, dim={orig_dim}"
        ));

        pb.set_message("[2/4] Training 1-Bit Binary K-Means + TSP...");
        let codebook = pipeline.train_codebook(&reservoir, sampled_count, orig_dim)?;
        drop(reservoir);

        let batch_rows_u64 = u64::try_from(pipeline.config.batch_size_rows.max(1)).unwrap_or(1);
        pb.set_length(total_seen.div_ceil(batch_rows_u64).max(1));
        pb.set_position(0);
        pb.set_message("[3/4] Writing Zstd minipage batches (remote)...");
        let mut stream2 = ParquetBatchStream::from_object_store(
            store,
            &obj_path,
            hints,
            pipeline.config.batch_size_rows,
        )
        .await?;
        let col_map = stream2.projected_map().clone();
        let projected_schema = stream2.projected_schema().clone();

        let mut writer = create_streaming_osv_writer(osv_path, osv_config, codebook.orig_dim)?;
        let mut acc = StreamingAccumulator::new(
            usize::try_from(total_seen).unwrap_or(0),
            codebook.orig_dim,
            codebook.padded_dim,
        );
        while let Some(batch_res) = stream2.next().await {
            let batch = batch_res?;
            acc.push_arrow_batch(
                &batch,
                &col_map,
                projected_schema.as_ref(),
                pipeline,
                &codebook,
                want_lutrank,
                &mut writer,
            )?;
            pb.inc(1);
        }
        let osv_report = writer.finish()?;

        acc.into_corpus(codebook, osv_report)
    })
}

fn ingest_local_jsonl(
    path: &Path,
    pipeline: &OutOfCoreIngestPipeline,
    want_lutrank: bool,
    osv_path: &Path,
    osv_config: &OsvWriterConfig,
    pb: &ProgressBar,
    ctx: &CliContext,
) -> CliResult<IngestedCorpus> {
    pb.set_message("[1/4] Sampling and inspecting dataset (.jsonl)...");
    let (reservoir, sampled_count, total_seen, orig_dim) = reservoir_sample_jsonl(
        || JsonlRecordReader::open(path).map_err(CliError::from),
        pipeline,
    )?;
    ctx.log_info(format!(
        "JSONL Pass 1 complete: seen={total_seen}, sampled={sampled_count}, dim={orig_dim}"
    ));

    pb.set_message("[2/4] Training 1-Bit Binary K-Means + TSP...");
    let codebook = pipeline.train_codebook(&reservoir, sampled_count, orig_dim)?;
    drop(reservoir);

    let batch_rows_u64 = u64::try_from(pipeline.config.batch_size_rows.max(1)).unwrap_or(1);
    pb.set_length(total_seen.div_ceil(batch_rows_u64).max(1));
    pb.set_position(0);
    pb.set_message("[3/4] Writing Zstd minipage batches (.jsonl)...");
    let reader = JsonlRecordReader::open(path)?;
    consume_jsonl_pass2(
        reader,
        pipeline,
        &codebook,
        total_seen,
        want_lutrank,
        osv_path,
        osv_config,
        pb,
    )
}

fn ingest_jsonl_from_bytes(
    raw: &[u8],
    pipeline: &OutOfCoreIngestPipeline,
    want_lutrank: bool,
    osv_path: &Path,
    osv_config: &OsvWriterConfig,
    pb: &ProgressBar,
    ctx: &CliContext,
) -> CliResult<IngestedCorpus> {
    pb.set_message("[1/4] Sampling and inspecting dataset (.jsonl)...");
    let (reservoir, sampled_count, total_seen, orig_dim) = reservoir_sample_jsonl(
        || Ok(JsonlRecordReader::new(BufReader::new(Cursor::new(raw)))),
        pipeline,
    )?;
    ctx.log_info(format!(
        "Remote JSONL Pass 1 complete: seen={total_seen}, sampled={sampled_count}, dim={orig_dim}"
    ));

    pb.set_message("[2/4] Training 1-Bit Binary K-Means + TSP...");
    let codebook = pipeline.train_codebook(&reservoir, sampled_count, orig_dim)?;
    drop(reservoir);

    let batch_rows_u64 = u64::try_from(pipeline.config.batch_size_rows.max(1)).unwrap_or(1);
    pb.set_length(total_seen.div_ceil(batch_rows_u64).max(1));
    pb.set_position(0);
    pb.set_message("[3/4] Writing Zstd minipage batches (.jsonl)...");
    let reader = JsonlRecordReader::new(BufReader::new(Cursor::new(raw)));
    consume_jsonl_pass2(
        reader,
        pipeline,
        &codebook,
        total_seen,
        want_lutrank,
        osv_path,
        osv_config,
        pb,
    )
}

fn reservoir_sample_jsonl<R, F>(
    make_reader: F,
    pipeline: &OutOfCoreIngestPipeline,
) -> CliResult<(Vec<f32>, usize, u64, usize)>
where
    R: std::io::BufRead,
    F: Fn() -> CliResult<JsonlRecordReader<R>>,
{
    let mut reader = make_reader()?;
    let cap = pipeline.config.max_training_sample_vectors.max(1);
    let mut rng = SplitMix64::new(pipeline.config.reservoir_seed);
    let mut reservoir = Vec::new();
    let mut sampled_count = 0_usize;
    let mut total_seen = 0_u64;
    let mut orig_dim: Option<usize> = None;

    while let Some(rec) = reader.next_record()? {
        validate_finite_vec(&rec.vector)?;
        let dim = match orig_dim {
            Some(d) if d == rec.vector.len() => d,
            Some(_) => {
                return Err(CliError::Core(OspreyError::InvalidDimension {
                    dim: rec.vector.len(),
                }));
            }
            None => {
                orig_dim = Some(rec.vector.len());
                reservoir.reserve(cap * rec.vector.len());
                rec.vector.len()
            }
        };
        if sampled_count < cap {
            reservoir.extend_from_slice(&rec.vector);
            sampled_count += 1;
        } else {
            let j_u64 = rng.next_u64() % (total_seen + 1);
            let j = usize::try_from(j_u64).unwrap_or(usize::MAX);
            if j < cap {
                reservoir[j * dim..(j + 1) * dim].copy_from_slice(&rec.vector);
            }
        }
        total_seen += 1;
    }

    let dim = orig_dim
        .ok_or_else(|| CliError::InvalidArgs("JSONL dataset contained zero records".to_string()))?;
    Ok((reservoir, sampled_count, total_seen, dim))
}

#[allow(clippy::too_many_arguments)]
fn consume_jsonl_pass2<R: std::io::BufRead>(
    mut reader: JsonlRecordReader<R>,
    pipeline: &OutOfCoreIngestPipeline,
    codebook: &osprey_storage::TrainedLakeCodebook,
    total_seen: u64,
    want_lutrank: bool,
    osv_path: &Path,
    osv_config: &OsvWriterConfig,
    pb: &ProgressBar,
) -> CliResult<IngestedCorpus> {
    let cap = usize::try_from(total_seen).unwrap_or(0);
    let mut writer = create_streaming_osv_writer(osv_path, osv_config, codebook.orig_dim)?;
    let mut acc = StreamingAccumulator::new(cap, codebook.orig_dim, codebook.padded_dim);

    let chunk_size = pipeline.config.batch_size_rows.max(1);
    let mut chunk_doc_ids = Vec::with_capacity(chunk_size);
    let mut chunk_vecs_f32 = Vec::with_capacity(chunk_size.saturating_mul(codebook.orig_dim));
    let mut chunk_cluster_ids = Vec::with_capacity(chunk_size);
    let mut chunk_f16 = Vec::with_capacity(chunk_size.saturating_mul(codebook.orig_dim));
    let mut chunk_payloads: Vec<Vec<u8>> = Vec::with_capacity(chunk_size);

    while let Some(rec) = reader.next_record()? {
        if rec.vector.len() != codebook.orig_dim {
            return Err(CliError::Core(OspreyError::InvalidDimension {
                dim: rec.vector.len(),
            }));
        }
        chunk_doc_ids.push(rec.doc_id);
        chunk_vecs_f32.extend_from_slice(&rec.vector);
        chunk_payloads.push(if rec.metadata_payload.is_empty() {
            b"{}".to_vec()
        } else {
            rec.metadata_payload
        });

        if want_lutrank {
            if let Some(toks) = rec.colbert_tokens {
                acc.has_colbert = true;
                let count = toks.len();
                let mut flat = Vec::with_capacity(count.saturating_mul(LUTRANK_TOKEN_DIM));
                for tok in toks {
                    if tok.len() != LUTRANK_TOKEN_DIM {
                        return Err(CliError::Core(OspreyError::InvalidDimension {
                            dim: tok.len(),
                        }));
                    }
                    validate_finite_vec(&tok)?;
                    flat.extend_from_slice(&tok);
                }
                acc.colbert_tokens.push(flat);
                acc.colbert_token_counts.push(count);
            } else {
                acc.colbert_tokens.push(Vec::new());
                acc.colbert_token_counts.push(0);
            }
        }

        if chunk_doc_ids.len() >= chunk_size {
            pipeline.assign_tsp_clusters_batch(
                codebook,
                &chunk_vecs_f32,
                chunk_doc_ids.len(),
                &mut chunk_cluster_ids,
                &mut chunk_f16,
            )?;
            acc.write_chunk(
                &mut writer,
                &chunk_doc_ids,
                &chunk_cluster_ids,
                &chunk_f16,
                &chunk_payloads,
            )?;
            chunk_doc_ids.clear();
            chunk_vecs_f32.clear();
            chunk_cluster_ids.clear();
            chunk_f16.clear();
            chunk_payloads.clear();
            pb.inc(1);
        }
    }
    if !chunk_doc_ids.is_empty() {
        pipeline.assign_tsp_clusters_batch(
            codebook,
            &chunk_vecs_f32,
            chunk_doc_ids.len(),
            &mut chunk_cluster_ids,
            &mut chunk_f16,
        )?;
        acc.write_chunk(
            &mut writer,
            &chunk_doc_ids,
            &chunk_cluster_ids,
            &chunk_f16,
            &chunk_payloads,
        )?;
        pb.inc(1);
    }

    let osv_report = writer.finish()?;
    acc.into_corpus(codebook.clone(), osv_report)
}

fn ingest_local_npy(
    path: &Path,
    pipeline: &OutOfCoreIngestPipeline,
    osv_path: &Path,
    osv_config: &OsvWriterConfig,
    pb: &ProgressBar,
    ctx: &CliContext,
) -> CliResult<IngestedCorpus> {
    pb.set_message("[1/4] Sampling and inspecting dataset (.npy)...");
    let (reservoir, sampled_count, total_seen, orig_dim) = reservoir_sample_npy(
        || NpyMatrixReader::open(path).map_err(CliError::from),
        pipeline,
    )?;
    ctx.log_info(format!(
        "NPY Pass 1 complete: rows={total_seen}, sampled={sampled_count}, dim={orig_dim}"
    ));

    pb.set_message("[2/4] Training 1-Bit Binary K-Means + TSP...");
    let codebook = pipeline.train_codebook(&reservoir, sampled_count, orig_dim)?;
    drop(reservoir);

    let batch_rows_u64 = u64::try_from(pipeline.config.batch_size_rows.max(1)).unwrap_or(1);
    pb.set_length(total_seen.div_ceil(batch_rows_u64).max(1));
    pb.set_position(0);
    pb.set_message("[3/4] Writing Zstd minipage batches (.npy)...");
    let reader = NpyMatrixReader::open(path)?;
    consume_npy_pass2(
        reader, pipeline, &codebook, total_seen, osv_path, osv_config, pb,
    )
}

fn ingest_npy_from_bytes(
    raw: &[u8],
    pipeline: &OutOfCoreIngestPipeline,
    osv_path: &Path,
    osv_config: &OsvWriterConfig,
    pb: &ProgressBar,
    ctx: &CliContext,
) -> CliResult<IngestedCorpus> {
    pb.set_message("[1/4] Sampling and inspecting dataset (.npy)...");
    let (reservoir, sampled_count, total_seen, orig_dim) = reservoir_sample_npy(
        || NpyMatrixReader::new(Cursor::new(raw)).map_err(CliError::from),
        pipeline,
    )?;
    ctx.log_info(format!(
        "Remote NPY Pass 1 complete: rows={total_seen}, sampled={sampled_count}, dim={orig_dim}"
    ));

    pb.set_message("[2/4] Training 1-Bit Binary K-Means + TSP...");
    let codebook = pipeline.train_codebook(&reservoir, sampled_count, orig_dim)?;
    drop(reservoir);

    let batch_rows_u64 = u64::try_from(pipeline.config.batch_size_rows.max(1)).unwrap_or(1);
    pb.set_length(total_seen.div_ceil(batch_rows_u64).max(1));
    pb.set_position(0);
    pb.set_message("[3/4] Writing Zstd minipage batches (.npy)...");
    let reader = NpyMatrixReader::new(Cursor::new(raw))?;
    consume_npy_pass2(
        reader, pipeline, &codebook, total_seen, osv_path, osv_config, pb,
    )
}

fn reservoir_sample_npy<R, F>(
    make_reader: F,
    pipeline: &OutOfCoreIngestPipeline,
) -> CliResult<(Vec<f32>, usize, u64, usize)>
where
    R: std::io::Read,
    F: Fn() -> CliResult<NpyMatrixReader<R>>,
{
    let mut reader = make_reader()?;
    let orig_dim = reader.dim();
    if reader.num_rows() == 0 || orig_dim == 0 {
        return Err(CliError::InvalidArgs(
            ".npy file contained zero vectors".to_string(),
        ));
    }

    let cap = pipeline.config.max_training_sample_vectors.max(1);
    let mut rng = SplitMix64::new(pipeline.config.reservoir_seed);
    let mut reservoir = Vec::with_capacity(cap.min(reader.num_rows()) * orig_dim);
    let mut sampled_count = 0_usize;
    let mut total_seen = 0_u64;

    while let Some(batch) = reader.read_batch_f32(pipeline.config.batch_size_rows)? {
        for r in 0..batch.doc_ids.len() {
            let slice = &batch.vectors_f32[r * orig_dim..(r + 1) * orig_dim];
            if sampled_count < cap {
                reservoir.extend_from_slice(slice);
                sampled_count += 1;
            } else {
                let j_u64 = rng.next_u64() % (total_seen + 1);
                let j = usize::try_from(j_u64).unwrap_or(usize::MAX);
                if j < cap {
                    reservoir[j * orig_dim..(j + 1) * orig_dim].copy_from_slice(slice);
                }
            }
            total_seen += 1;
        }
    }

    Ok((reservoir, sampled_count, total_seen, orig_dim))
}

fn consume_npy_pass2<R: std::io::Read>(
    mut reader: NpyMatrixReader<R>,
    pipeline: &OutOfCoreIngestPipeline,
    codebook: &osprey_storage::TrainedLakeCodebook,
    total_seen: u64,
    osv_path: &Path,
    osv_config: &OsvWriterConfig,
    pb: &ProgressBar,
) -> CliResult<IngestedCorpus> {
    let cap = usize::try_from(total_seen).unwrap_or(0);
    let mut writer = create_streaming_osv_writer(osv_path, osv_config, codebook.orig_dim)?;
    let mut acc = StreamingAccumulator::new(cap, codebook.orig_dim, codebook.padded_dim);

    while let Some(batch) = reader.read_batch_f32(pipeline.config.batch_size_rows)? {
        let n = batch.doc_ids.len();
        let mut cluster_ids = Vec::with_capacity(n);
        let mut vectors_f16 = Vec::with_capacity(n.saturating_mul(codebook.orig_dim));
        pipeline.assign_tsp_clusters_batch(
            codebook,
            &batch.vectors_f32,
            n,
            &mut cluster_ids,
            &mut vectors_f16,
        )?;
        let payloads = vec![b"{}".to_vec(); n];
        acc.write_chunk(
            &mut writer,
            &batch.doc_ids,
            &cluster_ids,
            &vectors_f16,
            &payloads,
        )?;
        pb.inc(1);
    }

    let osv_report = writer.finish()?;
    acc.into_corpus(codebook.clone(), osv_report)
}

fn create_streaming_osv_writer(
    osv_path: &Path,
    config: &OsvWriterConfig,
    orig_dim: usize,
) -> CliResult<StreamingOsvWriter<BufWriter<File>>> {
    let file = File::create(osv_path)?;
    StreamingOsvWriter::new(BufWriter::new(file), config.clone(), orig_dim, Vec::new())
        .map_err(CliError::from)
}

/// Streaming Pass-2 accumulator: retains only the `f16` corpus (for the `.osi` build) plus the
/// per-doc `.osv` locations, while `.osv` minipage blocks are compressed and written per chunk.
/// The raw `f32` corpus is never retained, bounding peak RSS for large datasets.
struct StreamingAccumulator {
    orig_dim: usize,
    doc_ids: Vec<u64>,
    vectors_f16: Vec<f16>,
    tsp_assignments: Vec<u32>,
    locations_by_doc: HashMap<u64, (u32, u16)>,
    colbert_tokens: Vec<Vec<f32>>,
    colbert_token_counts: Vec<usize>,
    has_colbert: bool,
}

impl StreamingAccumulator {
    fn new(cap: usize, orig_dim: usize, _padded_dim: usize) -> Self {
        Self {
            orig_dim,
            doc_ids: Vec::with_capacity(cap),
            vectors_f16: Vec::with_capacity(cap.saturating_mul(orig_dim)),
            tsp_assignments: Vec::with_capacity(cap),
            locations_by_doc: HashMap::with_capacity(cap),
            colbert_tokens: Vec::with_capacity(cap),
            colbert_token_counts: Vec::with_capacity(cap),
            has_colbert: false,
        }
    }

    fn write_chunk<W: Write + Seek>(
        &mut self,
        writer: &mut StreamingOsvWriter<W>,
        doc_ids: &[u64],
        cluster_ids: &[u32],
        vectors_f16: &[f16],
        payloads: &[Vec<u8>],
    ) -> CliResult<()> {
        let records: Vec<OsvVectorRecord<'_>> = (0..doc_ids.len())
            .map(|i| OsvVectorRecord {
                doc_id: doc_ids[i],
                cluster_id: cluster_ids[i],
                vector_f16: &vectors_f16[i * self.orig_dim..(i + 1) * self.orig_dim],
                payload: &payloads[i],
            })
            .collect();
        let locations = writer.write_chunk(&records, None)?;
        for (doc_id, loc) in locations {
            self.locations_by_doc
                .insert(doc_id, (loc.minipage_id, loc.slot_idx));
        }
        self.doc_ids.extend_from_slice(doc_ids);
        self.tsp_assignments.extend_from_slice(cluster_ids);
        self.vectors_f16.extend_from_slice(vectors_f16);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn push_arrow_batch<W: Write + Seek>(
        &mut self,
        batch: &RecordBatch,
        col_map: &osprey_storage::ProjectedColumnMap,
        projected_schema: &arrow::datatypes::Schema,
        pipeline: &OutOfCoreIngestPipeline,
        codebook: &osprey_storage::TrainedLakeCodebook,
        want_lutrank: bool,
        writer: &mut StreamingOsvWriter<W>,
    ) -> CliResult<()> {
        let n = batch.num_rows();
        if n == 0 {
            return Ok(());
        }
        let doc_ids = extract_batch_doc_ids(batch, col_map.id_batch_idx)?;
        let (dim, vecs_f32) =
            extract_batch_vectors_f32(batch, col_map.vector_batch_idx, Some(self.orig_dim))?;
        if dim != self.orig_dim {
            return Err(CliError::Core(OspreyError::InvalidDimension { dim }));
        }
        let batch_payloads =
            build_arrow_metadata_json(batch, projected_schema, &col_map.metadata_batch_indices)?;

        let mut cluster_ids = Vec::with_capacity(n);
        let mut vectors_f16 = Vec::with_capacity(n.saturating_mul(dim));
        pipeline.assign_tsp_clusters_batch(
            codebook,
            &vecs_f32,
            n,
            &mut cluster_ids,
            &mut vectors_f16,
        )?;
        // Release the raw `f32` batch before compressing + writing the `.osv` blocks.
        drop(vecs_f32);

        self.write_chunk(
            writer,
            &doc_ids,
            &cluster_ids,
            &vectors_f16,
            &batch_payloads,
        )?;

        if want_lutrank {
            if let Some(cb_idx) = col_map.colbert_batch_idx {
                self.has_colbert = true;
                let (tok_dim, toks) =
                    extract_batch_colbert_tokens_f32(batch, cb_idx, Some(LUTRANK_TOKEN_DIM))?;
                if tok_dim != LUTRANK_TOKEN_DIM {
                    return Err(CliError::Core(OspreyError::InvalidDimension {
                        dim: tok_dim,
                    }));
                }
                for flat in toks {
                    let count = flat.len() / tok_dim;
                    self.colbert_token_counts.push(count);
                    self.colbert_tokens.push(flat);
                }
            }
        }

        Ok(())
    }

    fn into_corpus(
        self,
        codebook: osprey_storage::TrainedLakeCodebook,
        osv_report: OsvWriteReport,
    ) -> CliResult<IngestedCorpus> {
        let mut locations = Vec::with_capacity(self.doc_ids.len());
        for &doc_id in &self.doc_ids {
            let (minipage_id, slot_idx) =
                self.locations_by_doc.get(&doc_id).copied().ok_or_else(|| {
                    CliError::Core(OspreyError::InvalidDimension {
                        dim: usize::try_from(doc_id).unwrap_or(usize::MAX),
                    })
                })?;
            locations.push(VectorLocation {
                minipage_id,
                slot_idx,
                slab_id: 0,
            });
        }
        Ok(IngestedCorpus {
            orig_dim: self.orig_dim,
            doc_ids: self.doc_ids,
            vectors_f16: self.vectors_f16,
            tsp_assignments: self.tsp_assignments,
            locations,
            colbert_tokens: self.colbert_tokens,
            colbert_token_counts: self.colbert_token_counts,
            has_colbert: self.has_colbert,
            codebook,
            osv_report,
        })
    }
}

fn build_arrow_metadata_json(
    batch: &RecordBatch,
    schema: &arrow::datatypes::Schema,
    metadata_batch_indices: &[(usize, usize)],
) -> CliResult<Vec<Vec<u8>>> {
    let n = batch.num_rows();
    if metadata_batch_indices.is_empty() {
        return Ok(vec![b"{}".to_vec(); n]);
    }

    let mut payloads = Vec::with_capacity(n);
    for row in 0..n {
        let mut obj = serde_json::Map::with_capacity(metadata_batch_indices.len());
        for &(_, batch_col_idx) in metadata_batch_indices {
            let field_name = schema.field(batch_col_idx).name();
            let col = batch.column(batch_col_idx);
            if col.is_null(row) {
                obj.insert(field_name.clone(), serde_json::Value::Null);
                continue;
            }
            let val = match col.data_type() {
                DataType::UInt32 => col
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .map(|a| serde_json::Value::from(a.value(row)))
                    .unwrap_or(serde_json::Value::Null),
                DataType::Int32 => col
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .map(|a| serde_json::Value::from(a.value(row)))
                    .unwrap_or(serde_json::Value::Null),
                DataType::UInt64 => col
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .map(|a| serde_json::Value::from(a.value(row)))
                    .unwrap_or(serde_json::Value::Null),
                DataType::Int64 => col
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .map(|a| serde_json::Value::from(a.value(row)))
                    .unwrap_or(serde_json::Value::Null),
                DataType::Float32 => col
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .and_then(|a| serde_json::Number::from_f64(f64::from(a.value(row))))
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null),
                DataType::Float64 => col
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .and_then(|a| serde_json::Number::from_f64(a.value(row)))
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null),
                DataType::Boolean => col
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .map(|a| serde_json::Value::from(a.value(row)))
                    .unwrap_or(serde_json::Value::Null),
                DataType::Utf8 => col
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .map(|a| serde_json::Value::from(a.value(row)))
                    .unwrap_or(serde_json::Value::Null),
                DataType::LargeUtf8 => col
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .map(|a| serde_json::Value::from(a.value(row)))
                    .unwrap_or(serde_json::Value::Null),
                DataType::Timestamp(unit, _) => {
                    let secs = match unit {
                        TimeUnit::Second => col
                            .as_any()
                            .downcast_ref::<TimestampSecondArray>()
                            .map(|a| a.value(row)),
                        TimeUnit::Millisecond => col
                            .as_any()
                            .downcast_ref::<TimestampMillisecondArray>()
                            .map(|a| a.value(row) / 1_000),
                        TimeUnit::Microsecond => col
                            .as_any()
                            .downcast_ref::<TimestampMicrosecondArray>()
                            .map(|a| a.value(row) / 1_000_000),
                        TimeUnit::Nanosecond => col
                            .as_any()
                            .downcast_ref::<TimestampNanosecondArray>()
                            .map(|a| a.value(row) / 1_000_000_000),
                    };
                    secs.map(serde_json::Value::from)
                        .unwrap_or(serde_json::Value::Null)
                }
                _ => serde_json::Value::Null,
            };
            obj.insert(field_name.clone(), val);
        }
        let encoded = serde_json::to_vec(&obj)
            .map_err(|err| CliError::Io(format!("JSON metadata encode failed: {err}")))?;
        payloads.push(encoded);
    }
    Ok(payloads)
}

fn fetch_remote_bytes(uri: &str) -> CliResult<Vec<u8>> {
    let remote_cfg = RemoteStoreConfig::default();
    let (store, obj_path) = build_object_store(uri, &remote_cfg)?;
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| CliError::Io(format!("failed to create tokio runtime: {err}")))?;
    rt.block_on(async {
        let res = store
            .get(&obj_path)
            .await
            .map_err(|err| CliError::Io(format!("object_store get failed for '{uri}': {err}")))?;
        let bytes = res
            .bytes()
            .await
            .map_err(|err| CliError::Io(format!("object_store read failed for '{uri}': {err}")))?;
        Ok(bytes.to_vec())
    })
}

fn join_remote_segment_uri(base: &str, filename: &str) -> String {
    let trimmed = base.trim_end_matches('/');
    format!("{trimmed}/{filename}")
}

fn upload_file_to_remote(local_path: &Path, remote_uri: &str) -> CliResult<()> {
    let bytes = std::fs::read(local_path)?;
    let remote_cfg = RemoteStoreConfig::default();
    let (store, obj_path) = build_object_store(remote_uri, &remote_cfg)?;
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| CliError::Io(format!("failed to create tokio runtime: {err}")))?;
    rt.block_on(async {
        store
            .put(&obj_path, bytes.into())
            .await
            .map_err(|err| CliError::Io(format!("failed to upload '{remote_uri}': {err}")))?;
        Ok(())
    })
}

fn print_import_human(r: &ImportReport) {
    print_section(&format!(
        "Imported {} vectors ({} -> {})",
        paint_accent(&r.vectors_indexed.to_string()),
        r.input_format,
        r.output
    ));
    print_kv("Vectors", &paint_accent(&r.vectors_indexed.to_string()));
    print_kv(
        "Dimensions",
        &format!("{} -> padded {}", r.orig_dim, r.padded_dim),
    );
    print_kv("Clusters (TSP)", &r.clusters.to_string());
    print_kv(
        "Minipages",
        &format!("{} KiB target ({} pages)", r.page_size_kb, r.minipages),
    );

    print_section("Artifacts");
    print_kv(
        ".osi index",
        &format!("{} ({})", r.osi_path, format_bytes(r.osi_bytes)),
    );
    print_kv(
        ".osv slabs",
        &format!(
            "{} ({}, {}, dict={})",
            r.osv_path,
            format_bytes(r.osv_bytes),
            paint_pass(&format!("{:.2}x Zstd", r.osv_compression_ratio)),
            format_bytes(u64::try_from(r.zstd_dict_bytes).unwrap_or(0))
        ),
    );
    if let Some(osr_path) = &r.osr_path {
        print_kv(
            ".osr LUT-Rank",
            &format!(
                "{} ({}, {} docs, {} tokens)",
                osr_path,
                format_bytes(r.osr_bytes.unwrap_or(0)),
                r.osr_docs.unwrap_or(0),
                r.osr_tokens.unwrap_or(0)
            ),
        );
    }
    print_kv("Elapsed", &paint_pass(&format!("{:.2} ms", r.elapsed_ms)));
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
    use arrow::array::{FixedSizeListBuilder, Float32Builder, ListBuilder, StringBuilder};
    use arrow::datatypes::{Field, Schema};
    use parquet::arrow::ArrowWriter;

    fn write_test_parquet(path: &Path, n: usize, dim: usize, with_tokens: bool) {
        let mut id_builder = StringBuilder::new();
        let mut vec_builder = FixedSizeListBuilder::new(Float32Builder::new(), dim as i32);
        let mut tok_builder =
            ListBuilder::new(FixedSizeListBuilder::new(Float32Builder::new(), 128));

        for i in 0..n {
            id_builder.append_value(format!("doc_{i:04}"));
            for d in 0..dim {
                vec_builder
                    .values()
                    .append_value(((i + d) as f32 * 0.05).sin());
            }
            vec_builder.append(true);

            if with_tokens {
                for t in 0..4 {
                    for d in 0..128 {
                        tok_builder
                            .values()
                            .values()
                            .append_value(((i + t + d) as f32 * 0.02).cos());
                    }
                    tok_builder.values().append(true);
                }
                tok_builder.append(true);
            }
        }

        let mut fields = vec![
            Field::new("doc_id", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                false,
            ),
        ];
        let mut arrays: Vec<arrow::array::ArrayRef> = vec![
            Arc::new(id_builder.finish()),
            Arc::new(vec_builder.finish()),
        ];

        if with_tokens {
            fields.push(Field::new(
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
            ));
            arrays.push(Arc::new(tok_builder.finish()));
        }

        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema.clone(), arrays).unwrap();
        let mut writer =
            ArrowWriter::try_new(BufWriter::new(File::create(path).unwrap()), schema, None)
                .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    #[test]
    fn import_parquet_with_lutrank_generates_osi_osv_osr() {
        let dir = tempfile::tempdir().unwrap();
        let pq_path = dir.path().join("docs.parquet");
        let out_dir = dir.path().join("out");
        write_test_parquet(&pq_path, 40, 64, true);

        let args = ImportArgs {
            input: url::Url::from_file_path(&pq_path).unwrap().to_string(),
            output: out_dir.display().to_string(),
            id_col: Some("doc_id".to_string()),
            vec_col: Some("embedding".to_string()),
            token_col: None,
            metadata_cols: vec![],
            clusters: 4,
            page_size_kb: 16,
            with_lutrank: true,
            batch_size: 16,
            sample_size: 32,
            bits: 1,
            no_zstd_dict: false,
            osv_compression: crate::cli::OsvCompressionMode::Auto,
            seed: 0x1234,
        };
        let ctx = CliContext {
            json: false,
            verbose: 0,
            threads: 1,
        };
        run(&args, &ctx).unwrap();

        assert!(out_dir.join("segment.osi").exists());
        assert!(out_dir.join("segment.osv").exists());
        assert!(out_dir.join("segment.osr").exists());
    }

    #[test]
    fn import_jsonl_and_npy_succeed() {
        let dir = tempfile::tempdir().unwrap();
        let jsonl_path = dir.path().join("docs.jsonl");
        let jsonl_out = dir.path().join("out_jsonl");
        let mut lines = String::new();
        for i in 0..16 {
            let vals: Vec<String> = (0..64)
                .map(|d| format!("{:.4}", ((i + d) as f32).sin()))
                .collect();
            lines.push_str(&format!(
                "{{\"id\":{i},\"vector\":[{}],\"tag\":\"v{i}\"}}\n",
                vals.join(",")
            ));
        }
        std::fs::write(&jsonl_path, lines).unwrap();

        let ctx = CliContext {
            json: true,
            verbose: 0,
            threads: 1,
        };
        let args_jsonl = ImportArgs {
            input: jsonl_path.display().to_string(),
            output: jsonl_out.display().to_string(),
            id_col: None,
            vec_col: None,
            token_col: None,
            metadata_cols: vec![],
            clusters: 2,
            page_size_kb: 16,
            with_lutrank: false,
            batch_size: 8,
            sample_size: 16,
            bits: 4,
            no_zstd_dict: true,
            osv_compression: crate::cli::OsvCompressionMode::Auto,
            seed: 99,
        };
        run(&args_jsonl, &ctx).unwrap();
        assert!(jsonl_out.join("segment.osi").exists());
        assert!(jsonl_out.join("segment.osv").exists());
    }
}
