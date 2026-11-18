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
    current_unix_secs, serve_arrow_flight, serve_http, CompactorConfig, CowSlabCompactor, Embedder,
    EmbedderConfig, HttpServerState, HybridSearchEngine, LsmDeltaBuffer, LutRankReranker,
    ManifestStore, OfflineLicenseVerifier, OspreyFlightService, OspreyWal, ProEngineError,
    ProMetricsRegistry, ProResult, VectorQuotaState, WalConfig, WalRecoveryMode, WalSyncPolicy,
    DEFAULT_ARROW_FLIGHT_ADDR, DEFAULT_HTTP_ADDR, DEFAULT_LUTRANK_SEED, DEFAULT_MANIFEST_KEY,
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
    Activate(ActivateArgs),
    WalStress(WalStressArgs),
}

#[derive(Debug, Args)]
struct ActivateArgs {
    #[arg(long)]
    license: String,

    #[arg(long)]
    output: Option<PathBuf>,
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

    #[arg(long, alias = "s3-manifest", env = "OSPREY_OBJECT_STORE_URI")]
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

fn default_license_path() -> ProResult<PathBuf> {
    let home = std::env::var_os("HOME").ok_or_else(|| {
        ProEngineError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "HOME environment variable is not set",
        ))
    })?;
    Ok(PathBuf::from(home).join(".osprey").join("license.key"))
}

fn load_license_token(explicit: Option<&str>) -> ProResult<Option<String>> {
    if let Some(tok) = explicit.map(str::trim).filter(|s| !s.is_empty()) {
        return Ok(Some(tok.to_owned()));
    }
    if let Ok(path) = default_license_path() {
        if path.exists() {
            let contents = std::fs::read_to_string(&path)?;
            let trimmed = contents.trim();
            if !trimmed.is_empty() {
                tracing::info!(path = %path.display(), "loaded persisted license key");
                return Ok(Some(trimmed.to_owned()));
            }
        }
    }
    Ok(None)
}

fn run_activate(args: ActivateArgs) -> ProResult<()> {
    let token = args.license.trim();
    let verifier = OfflineLicenseVerifier::new()?;
    let lic = verifier.verify_now(token)?;
    let now_secs = current_unix_secs()?;

    let out_path = match args.output {
        Some(p) => p,
        None => default_license_path()?,
    };
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out_path, format!("{token}\n"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(&out_path, std::fs::Permissions::from_mode(0o600))?;
    }

    if lic.is_in_grace_period_at(now_secs) {
        tracing::warn!(
            org = %lic.customer_org,
            tier = %lic.tier,
            expires_at = lic.expires_at,
            grace_days_remaining = lic.grace_days_remaining_at(now_secs),
            "activated license is expired and inside the 30-day renewal grace period"
        );
    }
    println!(
        "Activated OspreyDB Pro license for '{}' (tier={}, max_vectors={}, saved to {})",
        lic.customer_org,
        lic.tier,
        if lic.is_unlimited_vectors() {
            "unlimited".to_owned()
        } else {
            lic.max_vectors.to_string()
        },
        out_path.display()
    );
    Ok(())
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
    let license_token = load_license_token(args.license.as_deref())?;
    let active_license = if let Some(token) = &license_token {
        let now_secs = current_unix_secs()?;
        let lic = verifier.verify_now(token)?;
        let _ = verifier.verify_and_activate(token, now_secs)?;
        if lic.is_in_grace_period_at(now_secs) {
            let days_left = lic.grace_days_remaining_at(now_secs);
            tracing::warn!(
                org = %lic.customer_org,
                tier = %lic.tier,
                expires_at = lic.expires_at,
                grace_deadline = lic.grace_deadline(),
                grace_days_remaining = days_left,
                "OspreyDB Pro license expired; running inside 30-day renewal grace period ({days_left} days remaining before Pro startup is blocked)"
            );
        } else {
            tracing::info!(
                org = %lic.customer_org,
                tier = %lic.tier,
                max_nodes = lic.max_nodes,
                max_vectors = lic.max_vectors,
                unlimited_vectors = lic.is_unlimited_vectors(),
                arrow_flight = lic.has_capability(ProCapability::ArrowFlight),
                "verified offline Ed25519 commercial license"
            );
        }
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

    let (store, manifest_key): (Arc<dyn ObjectStore>, String) = match &args.object_store_uri {
        Some(uri) if uri.starts_with("file://") => {
            let local_path = uri.trim_start_matches("file://");
            std::fs::create_dir_all(local_path)?;
            (
                Arc::new(LocalFileSystem::new_with_prefix(local_path)?),
                DEFAULT_MANIFEST_KEY.to_owned(),
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
            (s, key)
        }
        None => (Arc::new(InMemory::new()), DEFAULT_MANIFEST_KEY.to_owned()),
    };

    let manifest_store = ManifestStore::new(Arc::clone(&store), manifest_key);
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
    let initial_vectors = engine.total_live_vector_count()?;
    let quota = VectorQuotaState::with_initial_vectors(initial_vectors);
    if let Some(lic) = active_license {
        quota.activate_license(lic);
    } else if initial_vectors >= osprey_pro::COMMUNITY_MAX_VECTORS {
        tracing::warn!(
            total_vectors = initial_vectors,
            max_vectors = osprey_pro::COMMUNITY_MAX_VECTORS,
            "Community Tier 5M vector limit reached; node remains online in read/search-only mode for vectors"
        );
    }

    let http_state = HttpServerState::with_quota(
        Arc::clone(&reranker),
        embedder,
        Arc::clone(&metrics),
        quota.clone(),
    );
    http_state.register_collection(&args.collection, Arc::clone(&engine))?;

    let flight_service = OspreyFlightService::with_quota(engine, reranker, quota);

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
        Some(Command::Activate(args)) => run_activate(args),
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
