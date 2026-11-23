//! Unified `osprey` binary (100% Apache-2.0):
//! - Production daemon (`osprey serve`): Apache Arrow Flight (`:8815`),
//!   Drop-In Qdrant/Cohere HTTP & Prometheus `/metrics` (`:6333`), built-in
//!   `/v1/embeddings` (OpenAI-compatible), WAL/LSM Delta Buffer, and CoW Compactor.
//! - Engine & Storage CLI (`import`, `search`, `rerank`, `delete`, `inspect`,
//!   `bench`, `doctor`, `completions`, `wal-stress`).

use std::io::Write as _;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::Duration;

use clap::{Args, CommandFactory, FromArgMatches, Parser, Subcommand, ValueEnum};
use object_store::local::LocalFileSystem;
use object_store::memory::InMemory;
use object_store::ObjectStore;
use osprey_cli::cli::{
    BenchArgs, CompletionsArgs, DeleteArgs, DoctorArgs, ImportArgs, InspectArgs,
    OsvCompressionMode, RerankArgs, SearchArgs,
};
use osprey_cli::commands;
use osprey_cli::output::{self, CliContext, CliError, CliResult};
use osprey_core::math::padded_dimension;
use osprey_core::RawVectorRecord;
use osprey_simd::{active_simd_backend, detect_best_simd_level, SimdLevel};
use osprey_storage::{
    build_object_store, MinipageLruCache, OsvWriterConfig, RangePlannerConfig, RemoteStoreConfig,
};
use ospreydb::{
    serve_arrow_flight, serve_http, CompactorConfig, CowSlabCompactor, Embedder, EmbedderConfig,
    HttpServerState, HybridSearchEngine, LsmDeltaBuffer, LutRankReranker, ManifestStore,
    OspreyFlightService, OspreyWal, ProEngineError, ProMetricsRegistry, ProResult,
    TombstoneGarbageCollector, TombstoneGcConfig, WalConfig, WalRecoveryMode, WalSyncPolicy,
    DEFAULT_ARROW_FLIGHT_ADDR, DEFAULT_HTTP_ADDR, DEFAULT_LUTRANK_SEED, DEFAULT_MANIFEST_KEY,
};
use serde::Serialize;
use tokio::sync::Notify;
use yansi::{Condition, Paint};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
enum NodeRole {
    /// Single-node unified writer + reader + compactor.
    #[default]
    Unified,
    /// Dedicated distributed writer + compactor node.
    Writer,
    /// Stateless distributed reader node polling S3 manifest.json.
    Reader,
}

#[derive(Debug, Parser)]
#[command(
    name = "osprey",
    about = "OspreyDB: 1-Bit RAM Index + Zstd S3 Slabs + LUT-Rank Vector Database (Server & CLI)",
    version,
    after_help = "Examples:\n  osprey serve --data-dir ./data --s3-uri file:///tmp/osprey-s3\n  osprey import --input corpus.parquet --output ./out --with-lutrank\n  osprey search --index ./out/segment.osi --slabs ./out/segment.osv --query '[0.1, -0.2, ...]' -k 10\n  osprey doctor\n  osprey bench --vectors 20000 --dim 384 --with-lutrank"
)]
struct Cli {
    /// Emit structured JSON output to stdout (for CLI subcommands).
    #[arg(long, global = true)]
    json: bool,

    /// Rayon worker thread pool size (`0` = auto-detect logical CPU cores).
    #[arg(long, global = true, default_value_t = 0)]
    threads: usize,

    /// Diagnostic verbosity level on stderr (`-v`, `-vv`).
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Option<Command>,

    #[command(flatten)]
    serve: ServeArgs,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Start the OspreyDB HTTP (:6333) + Arrow Flight (:8815) + WAL/LSM + S3 CAS server.
    Serve(Box<ServeArgs>),
    /// Ingest local or remote Parquet/JSONL/NPY datasets into `.osi`, `.osv`, and optional `.osr` segments.
    Import(ImportArgs),
    /// Run vector search (Stage-1 `.osi` mmap + Stage-2 DP-coalesced `.osv` + Stage-3 exact f16 + optional `.osr` LUT-Rank).
    Search(SearchArgs),
    /// Rerank candidate document IDs strictly on CPU using `.osr` LUT-Rank SIMD kernels.
    Rerank(RerankArgs),
    /// Atomically add tombstones for deleted `doc_id`s into an existing `.osi` index.
    Delete(DeleteArgs),
    /// Inspect headers, compression ratios, minipage distributions, tombstones, and xxHash64 integrity.
    Inspect(InspectArgs),
    /// Run reproducible scientific benchmark and AWS cloud FinOps TCO projection.
    Bench(BenchArgs),
    /// Inspect host CPU SIMD capabilities, 64-byte alignment, memory bandwidth, and S3 connectivity.
    Doctor(DoctorArgs),
    /// Generate shell completion scripts (`bash`, `zsh`, `fish`, `elvish`, `powershell`).
    Completions(CompletionsArgs),
    /// Run WAL durability stress test for crash-recovery validation.
    WalStress(WalStressArgs),
}

#[derive(Debug, Args)]
struct ServeArgs {
    #[arg(long, value_enum, default_value_t = NodeRole::Unified, env = "OSPREY_ROLE")]
    role: NodeRole,

    #[arg(long, default_value = "default")]
    collection: String,

    #[arg(long, default_value_t = 128)]
    dim: usize,

    #[arg(long, default_value_t = 0x4F53_5052_0001)]
    fwht_seed: u64,

    #[arg(long, env = "OSPREY_DATA_DIR")]
    data_dir: Option<PathBuf>,

    #[arg(long, env = "OSPREY_WAL_PATH")]
    wal_path: Option<PathBuf>,

    #[arg(
        long,
        visible_aliases = ["s3-uri", "s3-manifest"],
        env = "OSPREY_OBJECT_STORE_URI"
    )]
    object_store_uri: Option<String>,

    #[arg(long, default_value_t = 5, env = "OSPREY_MANIFEST_POLL_SECS")]
    manifest_poll_secs: u64,

    #[arg(long, default_value = DEFAULT_ARROW_FLIGHT_ADDR)]
    flight_addr: SocketAddr,

    #[arg(long, default_value = DEFAULT_HTTP_ADDR)]
    http_addr: SocketAddr,

    #[arg(long, env = "OSPREY_API_KEY")]
    api_key: Option<String>,

    #[arg(long, default_value_t = ospreydb::DEFAULT_DELTA_FLUSH_THRESHOLD)]
    delta_flush_threshold: usize,

    /// Maximum in-memory minipage LRU cache capacity in MiB (`0` disables caching).
    #[arg(long, default_value_t = 256, env = "OSPREY_CACHE_MB")]
    cache_mb: usize,

    /// `.osv` slab compression mode for compacted segments (`auto` = `none` on `file://`, `zstd` on S3/cloud).
    #[arg(long, value_enum, default_value_t = OsvCompressionMode::Auto, env = "OSPREY_OSV_COMPRESSION")]
    osv_compression: OsvCompressionMode,

    // -- Embedder configuration (OpenAI-compatible /v1/embeddings) --
    #[arg(long, visible_alias = "embeddings-url", env = "OSPREY_EMBED_URL")]
    embed_url: Option<String>,

    #[arg(
        long,
        env = "OSPREY_EMBED_MODEL",
        default_value = "text-embedding-3-small"
    )]
    embed_model: String,

    #[arg(long, env = "OSPREY_EMBED_API_KEY")]
    embed_api_key: Option<String>,

    #[arg(long, default_value_t = 32)]
    embed_batch_size: usize,
}

#[derive(Debug, Args)]
struct WalStressArgs {
    #[arg(long)]
    wal_path: PathBuf,

    #[arg(long, default_value_t = 64)]
    dim: usize,

    #[arg(long, default_value_t = 0)]
    batches: usize,

    #[arg(long, default_value_t = 50)]
    batch_size: usize,
}

#[derive(Debug, Serialize)]
struct JsonErrorPayload {
    error: String,
    exit_code: u8,
}

fn build_cli_version() -> &'static str {
    match detect_best_simd_level() {
        SimdLevel::Avx512 => concat!(env!("CARGO_PKG_VERSION"), " (simd: avx512_vpopcntdq)"),
        SimdLevel::Avx2 => concat!(env!("CARGO_PKG_VERSION"), " (simd: avx2)"),
        SimdLevel::Neon => concat!(env!("CARGO_PKG_VERSION"), " (simd: neon)"),
        SimdLevel::Scalar => concat!(env!("CARGO_PKG_VERSION"), " (simd: scalar)"),
    }
}

fn build_cli_command() -> clap::Command {
    let ver = build_cli_version();
    Cli::command().version(ver).long_version(ver)
}

fn resolve_wal_path(data_dir: Option<&PathBuf>, wal_path: Option<&PathBuf>) -> PathBuf {
    if let Some(explicit) = wal_path {
        return explicit.clone();
    }
    if let Some(dir) = data_dir {
        return dir.join("osprey.wal");
    }
    PathBuf::from("./osprey.wal")
}

async fn shutdown_signal() -> ProResult<()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut sigterm = signal(SignalKind::terminate()).map_err(ProEngineError::Io)?;
        tokio::select! {
            res = tokio::signal::ctrl_c() => {
                res.map_err(ProEngineError::Io)?;
            }
            _ = sigterm.recv() => {}
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c().await.map_err(ProEngineError::Io)
    }
}

fn run_serve_blocking(args: ServeArgs) -> ProResult<()> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(ProEngineError::Io)?;
    rt.block_on(run_serve(args))
}

async fn run_serve(args: ServeArgs) -> ProResult<()> {
    let embedder_config = EmbedderConfig {
        embed_url: args.embed_url.unwrap_or_default(),
        embed_model: args.embed_model,
        api_key: args.embed_api_key,
        batch_size: args.embed_batch_size,
    };
    let embedder = Arc::new(Embedder::new(embedder_config));

    if let Some(ref data_dir) = args.data_dir {
        std::fs::create_dir_all(data_dir)?;
    }

    let padded_dim = padded_dimension(args.dim)?;
    let (wal, delta) = if args.role == NodeRole::Reader {
        let delta = Arc::new(LsmDeltaBuffer::new(
            args.dim,
            padded_dim,
            args.fwht_seed,
            Vec::new(),
            None,
        )?);
        (None, delta)
    } else {
        let wal_path = resolve_wal_path(args.data_dir.as_ref(), args.wal_path.as_ref());
        let (wal, replayed) = OspreyWal::open(&wal_path, WalConfig::default())?;
        let wal = Arc::new(wal);
        let delta = Arc::new(LsmDeltaBuffer::new(
            args.dim,
            padded_dim,
            args.fwht_seed,
            Vec::new(),
            Some(Arc::clone(&wal)),
        )?);
        if !replayed.is_empty() {
            let count = delta.recover_from_wal(&replayed)?;
            tracing::info!(
                replayed_records = count,
                "recovered WAL state into delta buffer"
            );
        }
        (Some(wal), delta)
    };

    let (store, manifest_key, local_store_root): (Arc<dyn ObjectStore>, String, Option<PathBuf>) =
        match &args.object_store_uri {
            Some(uri) if uri.starts_with("file://") => {
                let local_path = uri.strip_prefix("file://").unwrap_or(uri.as_str());
                std::fs::create_dir_all(local_path)?;
                (
                    Arc::new(LocalFileSystem::new_with_prefix(local_path)?),
                    DEFAULT_MANIFEST_KEY.to_owned(),
                    Some(PathBuf::from(local_path)),
                )
            }
            Some(uri) => {
                let (s, path) = build_object_store(uri, &RemoteStoreConfig::default())?;
                let path_str = path.to_string();
                let key = if path_str.is_empty() {
                    DEFAULT_MANIFEST_KEY.to_owned()
                } else if path_str.ends_with(".json") {
                    path_str
                } else {
                    format!(
                        "{}/{}",
                        path_str.trim_end_matches('/'),
                        DEFAULT_MANIFEST_KEY
                    )
                };
                (s, key, None)
            }
            None => (
                Arc::new(InMemory::new()),
                DEFAULT_MANIFEST_KEY.to_owned(),
                None,
            ),
        };

    let manifest_store = ManifestStore::new(Arc::clone(&store), manifest_key);
    let metrics = Arc::new(ProMetricsRegistry::new());
    let mut engine = HybridSearchEngine::with_metrics(
        Arc::clone(&delta),
        Arc::clone(&store),
        manifest_store.clone(),
        Arc::new(MinipageLruCache::from_mb(args.cache_mb)),
        RemoteStoreConfig::default(),
        RangePlannerConfig::default(),
        Arc::clone(&metrics),
    );
    if let Some(ref local_root) = local_store_root {
        engine = engine.with_local_store_root(local_root);
    }
    let engine = Arc::new(engine);
    let _ = engine.refresh_from_manifest().await?;

    let use_zstd = match args.osv_compression {
        OsvCompressionMode::Auto => local_store_root.is_none() && args.object_store_uri.is_some(),
        OsvCompressionMode::None => false,
        OsvCompressionMode::Zstd => true,
    };
    let osv_writer_config = if use_zstd {
        OsvWriterConfig::default()
    } else {
        OsvWriterConfig::uncompressed()
    };

    let compactor_handle = if matches!(args.role, NodeRole::Unified | NodeRole::Writer) {
        let compactor = CowSlabCompactor::new(
            CompactorConfig {
                delta_flush_threshold: args.delta_flush_threshold,
                fwht_seed: args.fwht_seed,
                poll_interval: Duration::from_secs(2),
                osv_writer_config,
                ..CompactorConfig::default()
            },
            Arc::clone(&store),
            manifest_store.clone(),
        );
        let tombstone_gc = TombstoneGarbageCollector::new(
            TombstoneGcConfig::default(),
            Arc::clone(&store),
            manifest_store,
        )?;
        Some(compactor.spawn_background(
            Arc::clone(&delta),
            wal.clone(),
            Arc::clone(&engine),
            Some(tombstone_gc),
        ))
    } else {
        None
    };

    let reranker = Arc::new(LutRankReranker::new(DEFAULT_LUTRANK_SEED)?);
    let initial_vectors = engine.total_live_vector_count()?;
    let vector_counter = Arc::new(AtomicU64::new(initial_vectors));

    let is_reader = args.role == NodeRole::Reader;
    let http_state = HttpServerState::with_vector_counter(
        Arc::clone(&reranker),
        embedder,
        Arc::clone(&metrics),
        Arc::clone(&vector_counter),
    )
    .with_api_key(args.api_key)
    .with_read_only(is_reader);
    http_state.register_collection(&args.collection, Arc::clone(&engine))?;

    let flight_service =
        OspreyFlightService::with_vector_counter(Arc::clone(&engine), reranker, vector_counter)
            .with_read_only(is_reader);

    let shutdown_notify = Arc::new(Notify::new());
    let flight_shutdown = Arc::clone(&shutdown_notify);
    let http_shutdown = Arc::clone(&shutdown_notify);

    let manifest_poll_task = if matches!(args.role, NodeRole::Reader | NodeRole::Unified)
        && args.manifest_poll_secs > 0
    {
        let poll_engine = Arc::clone(&engine);
        let poll_shutdown = Arc::clone(&shutdown_notify);
        let poll_interval = Duration::from_secs(args.manifest_poll_secs);
        Some(tokio::spawn(async move {
            loop {
                tokio::select! {
                    () = poll_shutdown.notified() => break,
                    () = tokio::time::sleep(poll_interval) => {
                        match poll_engine.refresh_from_manifest().await {
                            Ok(true) => {
                                tracing::info!("reloaded cluster manifest from object store");
                            }
                            Ok(false) => {}
                            Err(err) => {
                                tracing::warn!(error = %err, "manifest polling refresh failed");
                            }
                        }
                    }
                }
            }
        }))
    } else {
        None
    };

    let flight_task = tokio::spawn(async move {
        serve_arrow_flight(args.flight_addr, flight_service, async move {
            flight_shutdown.notified().await;
        })
        .await
    });
    let http_task = tokio::spawn(async move {
        serve_http(args.http_addr, http_state, async move {
            http_shutdown.notified().await;
        })
        .await
    });

    shutdown_signal().await?;
    tracing::info!("shutdown signal received; draining servers and flushing WAL");
    shutdown_notify.notify_waiters();

    let _ = flight_task.await;
    let _ = http_task.await;
    if let Some(handle) = manifest_poll_task {
        let _ = handle.await;
    }
    if let Some(handle) = compactor_handle {
        handle.shutdown().await;
    }
    if let Some(wal_ref) = wal {
        wal_ref.sync()?;
    }
    Ok(())
}

fn run_wal_stress(args: WalStressArgs) -> ProResult<()> {
    let (wal, replayed) = OspreyWal::open(
        &args.wal_path,
        WalConfig {
            recovery_mode: WalRecoveryMode::TruncateTornTail,
            sync_policy: WalSyncPolicy::SyncData,
        },
    )?;
    let wal = Arc::new(wal);
    let padded_dim = padded_dimension(args.dim)?;
    let delta = LsmDeltaBuffer::new(
        args.dim,
        padded_dim,
        0x4F53_5052_0001,
        Vec::new(),
        Some(Arc::clone(&wal)),
    )?;
    if !replayed.is_empty() {
        let _ = delta.recover_from_wal(&replayed)?;
    }

    let mut next_doc_id = replayed
        .iter()
        .map(|r| r.doc_id)
        .max()
        .unwrap_or(0)
        .saturating_add(1);

    let mut batch_idx = 0_usize;
    let mut stdout = std::io::stdout().lock();

    while args.batches == 0 || batch_idx < args.batches {
        let mut vecs = Vec::with_capacity(args.batch_size);
        let mut payloads = Vec::with_capacity(args.batch_size);
        let mut doc_ids = Vec::with_capacity(args.batch_size);

        for _ in 0..args.batch_size {
            let doc_id = next_doc_id;
            next_doc_id = next_doc_id.saturating_add(1);
            let mut v = vec![0.0_f32; args.dim];
            let hot = usize::try_from(doc_id).unwrap_or(0) % args.dim;
            v[hot] = 1.0;
            v[(hot + 1) % args.dim] = 0.25;
            vecs.push(v);
            payloads.push(format!(r#"{{"doc_id":{doc_id},"batch":{batch_idx}}}"#).into_bytes());
            doc_ids.push(doc_id);
        }

        let records: Vec<RawVectorRecord<'_>> = (0..args.batch_size)
            .map(|i| RawVectorRecord {
                doc_id: doc_ids[i],
                vector: &vecs[i],
                payload_json: &payloads[i],
            })
            .collect();

        let seq = delta.ingest_batch(&records)?;
        if let Some(&last_doc) = doc_ids.last() {
            writeln!(stdout, "COMMITTED {seq} {last_doc}")?;
            stdout.flush()?;
        }
        batch_idx += 1;
    }

    Ok(())
}

fn dispatch_cli_command(command: Command, ctx: &CliContext) -> CliResult<()> {
    if ctx.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(ctx.threads)
            .build_global()
            .map_err(|err| {
                CliError::Io(format!("failed to initialize rayon thread pool: {err}"))
            })?;
    }

    ctx.log_debug(format!(
        "osprey initialized (simd={}, json={}, verbose={}, threads={})",
        active_simd_backend(),
        ctx.json,
        ctx.verbose,
        ctx.threads
    ));

    match command {
        Command::Doctor(args) => commands::doctor::run(&args, ctx),
        Command::Import(args) => commands::import::run(&args, ctx),
        Command::Search(args) => commands::search::run(&args, ctx),
        Command::Rerank(args) => commands::rerank::run(&args, ctx),
        Command::Delete(args) => commands::delete::run(&args, ctx),
        Command::Inspect(args) => commands::inspect::run(&args, ctx),
        Command::Bench(args) => commands::bench::run(&args, ctx),
        Command::Completions(args) => {
            let mut cmd = build_cli_command();
            clap_complete::generate(args.shell, &mut cmd, "osprey", &mut std::io::stdout());
            Ok(())
        }
        Command::Serve(_) | Command::WalStress(_) => unreachable!("handled directly in main"),
    }
}

fn main() -> ExitCode {
    yansi::whenever(Condition::TTY_AND_COLOR);
    let matches = build_cli_command().get_matches();
    let cli = Cli::from_arg_matches(&matches).unwrap_or_else(|err| err.exit());
    let ctx = CliContext {
        json: cli.json,
        verbose: cli.verbose,
        threads: cli.threads,
    };

    match cli.command {
        Some(Command::Serve(args)) => match run_serve_blocking(*args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("osprey error: {err}");
                ExitCode::FAILURE
            }
        },
        None => match run_serve_blocking(cli.serve) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("osprey error: {err}");
                ExitCode::FAILURE
            }
        },
        Some(Command::WalStress(args)) => match run_wal_stress(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("osprey error: {err}");
                ExitCode::FAILURE
            }
        },
        Some(other) => {
            if let Err(err) = dispatch_cli_command(other, &ctx) {
                if ctx.json {
                    let payload = JsonErrorPayload {
                        error: err.to_string(),
                        exit_code: err.numeric_code(),
                    };
                    if let Ok(json) = serde_json::to_string_pretty(&payload) {
                        eprintln!("{json}");
                    } else {
                        eprintln!(
                            "{{\"error\":\"{err}\",\"exit_code\":{}}}",
                            err.numeric_code()
                        );
                    }
                } else {
                    eprintln!("{} {err}", "error:".red().bright().bold());
                }
                return err.exit_code();
            }
            ExitCode::from(output::EXIT_OK)
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn cli_parses_all_subcommands_including_serve_and_engine_commands() {
        let srv = Cli::try_parse_from([
            "osprey",
            "serve",
            "--data-dir",
            "/tmp/osprey-data",
            "--s3-uri",
            "file:///tmp/osprey-s3",
            "--embeddings-url",
            "https://api.openai.com",
            "--embed-api-key",
            "sk-123",
            "--api-key",
            "secret-token",
            "--role",
            "reader",
            "--manifest-poll-secs",
            "3",
        ])
        .unwrap();
        let Some(Command::Serve(serve)) = srv.command else {
            panic!("expected Serve subcommand");
        };
        assert_eq!(serve.role, NodeRole::Reader);
        assert_eq!(serve.manifest_poll_secs, 3);
        assert_eq!(
            serve.object_store_uri.as_deref(),
            Some("file:///tmp/osprey-s3")
        );
        assert_eq!(serve.embed_url.as_deref(), Some("https://api.openai.com"));
        assert_eq!(serve.embed_api_key.as_deref(), Some("sk-123"));
        assert_eq!(serve.api_key.as_deref(), Some("secret-token"));
        assert_eq!(
            resolve_wal_path(serve.data_dir.as_ref(), serve.wal_path.as_ref()),
            PathBuf::from("/tmp/osprey-data/osprey.wal")
        );

        let writer = Cli::try_parse_from(["osprey", "serve", "--role", "writer"]).unwrap();
        assert!(matches!(
            writer.command,
            Some(Command::Serve(ref s)) if s.role == NodeRole::Writer
        ));

        let fallback = Cli::try_parse_from(["osprey", "--data-dir", "/tmp/fallback"]).unwrap();
        assert!(fallback.command.is_none());
        assert_eq!(
            fallback.serve.data_dir.as_deref(),
            Some(std::path::Path::new("/tmp/fallback"))
        );

        let doc = Cli::try_parse_from(["osprey", "--json", "-vv", "doctor"]).unwrap();
        assert!(doc.json);
        assert_eq!(doc.verbose, 2);
        assert!(matches!(doc.command, Some(Command::Doctor(_))));

        let imp = Cli::try_parse_from([
            "osprey",
            "import",
            "--input",
            "data.parquet",
            "--output",
            "./out",
            "--with-lutrank",
        ])
        .unwrap();
        assert!(matches!(imp.command, Some(Command::Import(_))));

        let srch = Cli::try_parse_from([
            "osprey",
            "search",
            "--index",
            "seg.osi",
            "--slabs",
            "seg.osv",
            "--query",
            "[0.1,0.2]",
        ])
        .unwrap();
        assert!(matches!(srch.command, Some(Command::Search(_))));

        let rrk = Cli::try_parse_from([
            "osprey",
            "rerank",
            "--rerank-store",
            "seg.osr",
            "--query-tokens",
            "[[0.1]]",
            "--doc-ids",
            "1,2,3",
        ])
        .unwrap();
        assert!(matches!(rrk.command, Some(Command::Rerank(_))));

        let del = Cli::try_parse_from(["osprey", "delete", "--index", "seg.osi", "--ids", "10,20"])
            .unwrap();
        assert!(matches!(del.command, Some(Command::Delete(_))));

        let insp = Cli::try_parse_from(["osprey", "inspect", "-f", "seg.osi", "seg.osv"]).unwrap();
        assert!(matches!(insp.command, Some(Command::Inspect(_))));

        let bnc = Cli::try_parse_from(["osprey", "bench", "--with-lutrank"]).unwrap();
        assert!(matches!(bnc.command, Some(Command::Bench(_))));

        let comp = Cli::try_parse_from(["osprey", "completions", "bash"]).unwrap();
        assert!(matches!(comp.command, Some(Command::Completions(_))));

        let wal = Cli::try_parse_from(["osprey", "wal-stress", "--wal-path", "test.wal"]).unwrap();
        assert!(matches!(wal.command, Some(Command::WalStress(_))));

        let ver = build_cli_version();
        assert!(ver.contains(env!("CARGO_PKG_VERSION")));
        assert!(ver.contains(active_simd_backend()));
    }
}
