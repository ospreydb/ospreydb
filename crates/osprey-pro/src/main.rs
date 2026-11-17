//! `osprey-pro` production daemon hosting Apache Arrow Flight (`:8815`),
//! Drop-In Qdrant/Cohere HTTP & Prometheus `/metrics` (`:6333`), built-in
//! `/v1/embeddings` (OpenAI-compatible), offline Ed25519 license verification,
//! WAL/LSM Delta Buffer, and CoW Compactor.

use std::io::Write as _;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use clap::{Args, Parser, Subcommand};
use object_store::local::LocalFileSystem;
use object_store::memory::InMemory;
use object_store::ObjectStore;
use osprey_core::math::padded_dimension;
use osprey_core::{ProCapability, RawVectorRecord};
use osprey_pro::{
    serve_arrow_flight, serve_http, CompactorConfig, CowSlabCompactor, Embedder, EmbedderConfig,
    HttpServerState, HybridSearchEngine, LsmDeltaBuffer, LutRankReranker, ManifestStore,
    OfflineLicenseVerifier, OspreyFlightService, OspreyWal, ProEngineError, ProMetricsRegistry,
    ProResult, WalConfig, WalRecoveryMode, WalSyncPolicy, DEFAULT_ARROW_FLIGHT_ADDR,
    DEFAULT_HTTP_ADDR, DEFAULT_LUTRANK_SEED, DEFAULT_MANIFEST_KEY,
};
use osprey_storage::{
    build_object_store, MinipageLruCache, RemoteStoreConfig, WarpConfig,
    DEFAULT_MINIPAGE_CACHE_BYTES,
};
use tokio::sync::Notify;

#[derive(Debug, Parser)]
#[command(
    name = "osprey-pro",
    about = "OspreyDB Commercial Engine (Arrow Flight, Qdrant/Cohere Drop-In, Auto-Embeddings, WAL/LSM & S3 CAS)",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    #[command(flatten)]
    serve: ServeArgs,
}

#[derive(Debug, Subcommand)]
enum Command {
    Serve(Box<ServeArgs>),
    WalStress(WalStressArgs),
}

#[derive(Debug, Args)]
struct ServeArgs {
    #[arg(long, env = "OSPREY_PRO_LICENSE")]
    license: Option<String>,

    #[arg(long, default_value = "default")]
    collection: String,

    #[arg(long, default_value_t = 128)]
    dim: usize,

    #[arg(long, default_value_t = 0x4F53_5052_0001)]
    fwht_seed: u64,

    #[arg(long, default_value = "./osprey-pro.wal")]
    wal_path: PathBuf,

    #[arg(long)]
    object_store_uri: Option<String>,

    #[arg(long, default_value = DEFAULT_ARROW_FLIGHT_ADDR)]
    flight_addr: SocketAddr,

    #[arg(long, default_value = DEFAULT_HTTP_ADDR)]
    http_addr: SocketAddr,

    #[arg(long, default_value_t = osprey_pro::DEFAULT_DELTA_FLUSH_THRESHOLD)]
    delta_flush_threshold: usize,

    // -- Embedder configuration (OpenAI-compatible /v1/embeddings) --
    #[arg(long, env = "OSPREY_EMBED_URL")]
    embed_url: Option<String>,

    #[arg(long, env = "OSPREY_EMBED_MODEL", default_value = "text-embedding-3-small")]
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

async fn run_serve(args: ServeArgs) -> ProResult<()> {
    let embedder_config = EmbedderConfig {
        embed_url: args.embed_url.unwrap_or_default(),
        embed_model: args.embed_model,
        api_key: args.embed_api_key,
        batch_size: args.embed_batch_size,
    };
    let embedder = Arc::new(Embedder::new(embedder_config));

    let verifier = OfflineLicenseVerifier::new()?;
    let active_license = if let Some(token) = &args.license {
        let lic = verifier.verify_now(token)?;
        let _ = verifier.verify_and_activate(token, lic.issued_at)?;
        tracing::info!(
            org = %lic.customer_org,
            tier = %lic.tier,
            max_nodes = lic.max_nodes,
            max_vectors = lic.max_vectors,
            unlimited_vectors = lic.is_unlimited_vectors(),
            arrow_flight = lic.has_capability(ProCapability::ArrowFlight),
            "verified offline Ed25519 commercial license"
        );
        Some(lic)
    } else {
        tracing::info!(
            max_vectors = osprey_pro::COMMUNITY_MAX_VECTORS,
            "Single-Node Community Tier — no license key supplied (5M vector limit, local storage)"
        );
        None
    };

    let padded_dim = padded_dimension(args.dim)?;
    let (wal, replayed) = OspreyWal::open(&args.wal_path, WalConfig::default())?;
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

    let store: Arc<dyn ObjectStore> = match &args.object_store_uri {
        Some(uri) if uri.starts_with("file://") => {
            let local_path = uri.trim_start_matches("file://");
            std::fs::create_dir_all(local_path)?;
            Arc::new(LocalFileSystem::new_with_prefix(local_path)?)
        }
        Some(uri) => {
            let (s, _path) = build_object_store(uri, &RemoteStoreConfig::default())?;
            s
        }
        None => Arc::new(InMemory::new()),
    };

    let manifest_store = ManifestStore::new(Arc::clone(&store), DEFAULT_MANIFEST_KEY);
    let metrics = Arc::new(ProMetricsRegistry::new());
    let engine = Arc::new(HybridSearchEngine::with_metrics(
        Arc::clone(&delta),
        Arc::clone(&store),
        manifest_store.clone(),
        Arc::new(MinipageLruCache::new(DEFAULT_MINIPAGE_CACHE_BYTES)),
        RemoteStoreConfig::default(),
        WarpConfig::default(),
        Arc::clone(&metrics),
    ));
    let _ = engine.refresh_from_manifest().await?;

    let compactor = CowSlabCompactor::new(
        CompactorConfig {
            delta_flush_threshold: args.delta_flush_threshold,
            fwht_seed: args.fwht_seed,
            poll_interval: Duration::from_secs(2),
            ..CompactorConfig::default()
        },
        store,
        manifest_store,
    );
    let compactor_handle = compactor.spawn_background(Arc::clone(&delta), Some(wal));

    let reranker = Arc::new(LutRankReranker::new(DEFAULT_LUTRANK_SEED)?);
    let http_state = HttpServerState::new(Arc::clone(&reranker), embedder, Arc::clone(&metrics));
    if let Some(lic) = active_license {
        http_state.activate_license(lic);
    }
    http_state.register_collection(&args.collection, Arc::clone(&engine))?;

    let flight_service = OspreyFlightService::new(engine, reranker);

    let shutdown_notify = Arc::new(Notify::new());
    let flight_shutdown = Arc::clone(&shutdown_notify);
    let http_shutdown = Arc::clone(&shutdown_notify);

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

    tokio::signal::ctrl_c().await.map_err(ProEngineError::Io)?;
    shutdown_notify.notify_waiters();

    let _ = flight_task.await;
    let _ = http_task.await;
    compactor_handle.shutdown().await;
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

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    let res = match cli.command {
        Some(Command::Serve(args)) => run_serve(*args).await,
        Some(Command::WalStress(args)) => run_wal_stress(args),
        None => run_serve(cli.serve).await,
    };
    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("osprey-pro error: {err}");
            ExitCode::FAILURE
        }
    }
}