//! `clap` command tree definitions for OspreyDB CLI subcommands.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};
use osprey_storage::DEFAULT_MAX_TRAINING_SAMPLE_VECTORS;

/// Root CLI parser for `osprey` engine commands.
#[derive(Debug, Parser)]
#[command(
    name = "osprey",
    version,
    about = "OspreyDB: Zero-Copy Vector Search + LUT-Rank Engine CLI",
    after_help = "Examples:\n  osprey doctor\n  osprey import --input corpus.parquet --output ./data --with-lutrank\n  osprey search --index ./data/segment.osi --slabs ./data/segment.osv --query '[0.1, -0.2, ...]' -k 10\n  osprey bench --vectors 20000 --dim 384 --with-lutrank"
)]
pub struct OspreyCli {
    /// Emit structured JSON output to stdout.
    #[arg(long, global = true)]
    pub json: bool,

    /// Rayon worker thread pool size (`0` = auto-detect logical CPU cores).
    #[arg(long, global = true, default_value_t = 0)]
    pub threads: usize,

    /// Diagnostic verbosity level on stderr (`-v`, `-vv`).
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Active subcommand.
    #[command(subcommand)]
    pub command: OspreyCommand,
}

/// Top-level `osprey` subcommands.
#[derive(Debug, Subcommand)]
pub enum OspreyCommand {
    /// Inspect host CPU SIMD capabilities, 64-byte alignment, memory bandwidth, and S3 connectivity.
    Doctor(DoctorArgs),
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
    /// Generate shell completion scripts (`bash`, `zsh`, `fish`, `elvish`, `powershell`).
    Completions(CompletionsArgs),
}

/// Arguments for `osprey completions`.
#[derive(Debug, Args)]
pub struct CompletionsArgs {
    /// Target shell to generate completion script for.
    #[arg(value_enum)]
    pub shell: clap_complete::Shell,
}

/// Arguments for `osprey doctor`.
#[derive(Debug, Args)]
pub struct DoctorArgs {
    /// Optional Object Storage URI (`s3://...`, `file://...`) to test Head / Range-GET connectivity.
    #[arg(long)]
    pub s3_uri: Option<String>,

    /// Memory bandwidth micro-benchmark buffer size in MiB.
    #[arg(long, default_value_t = 64)]
    pub bench_mb: usize,
}

/// Arguments for `osprey import`.
#[derive(Debug, Args)]
pub struct ImportArgs {
    /// Input dataset path or URI (`.parquet`, `.jsonl`, `.npy`; local path, `file://`, or remote URI).
    #[arg(short, long)]
    pub input: String,

    /// Output directory path or remote URI prefix for `segment.{osi,osv,osr}`.
    #[arg(short, long)]
    pub output: String,

    /// Document ID column name (auto-detected when omitted, e.g., `doc_id` / `id`).
    #[arg(long)]
    pub id_col: Option<String>,

    /// Dense embedding column name (auto-detected when omitted, e.g., `embedding` / `vector`).
    #[arg(long)]
    pub vec_col: Option<String>,

    /// Optional ColBERT multi-vector token column name (`dim=128`) for `.osr` generation.
    #[arg(long)]
    pub token_col: Option<String>,

    /// Additional scalar metadata / payload column names to project from Parquet.
    #[arg(long, value_delimiter = ',')]
    pub metadata_cols: Vec<String>,

    /// Number of `1-Bit Binary K-Means` clusters `K`.
    #[arg(long, default_value_t = 256)]
    pub clusters: u32,

    /// Target uncompressed `.osv` minipage size in KiB (default: 16 KB).
    #[arg(long, default_value_t = 16)]
    pub page_size_kb: usize,

    /// Generate `.osr` LUT-Rank token store alongside `.osi` and `.osv`.
    #[arg(long)]
    pub with_lutrank: bool,

    /// Streaming batch size in rows.
    #[arg(long, default_value_t = 4096)]
    pub batch_size: usize,

    /// Maximum Pass-1 reservoir sample size for `1-Bit Binary K-Means` training.
    #[arg(long, default_value_t = DEFAULT_MAX_TRAINING_SAMPLE_VECTORS)]
    pub sample_size: usize,

    /// Disable Zstd payload dictionary training on `.osv` slabs.
    #[arg(long)]
    pub no_zstd_dict: bool,

    /// Deterministic SplitMix64 seed for FWHT rotation and Binary K-Means.
    #[arg(long, default_value_t = 0x4F53_5052_0001)]
    pub seed: u64,
}

/// Arguments for `osprey search`.
#[derive(Debug, Args)]
pub struct SearchArgs {
    /// Path to the `.osi` 1-bit index file (opened via `memmap2`).
    #[arg(long)]
    pub index: PathBuf,

    /// Path or Object Storage URI (`file://`, `s3://`, etc.) to the `.osv` slab file.
    #[arg(long)]
    pub slabs: String,

    /// Query vector in JSON (`[0.1, -0.2, ...]`), CSV floats, or path to `.json`/`.npy` file.
    #[arg(long)]
    pub query: String,

    /// Final number of results to return.
    #[arg(short = 'k', long, default_value_t = 10)]
    pub top_k: usize,

    /// Number of nearest TSP clusters to probe in `.osi`.
    #[arg(long, default_value_t = 16)]
    pub nprobe: usize,

    /// Number of Stage-1 `RaBitQ` candidates passed to DP range coalescing + exact `f16` refinement.
    #[arg(long, default_value_t = 200)]
    pub candidates: usize,

    /// Oversampling multiplier for `nprobe`/`candidates` (`0` = adaptive: `2x` when `dim <= 256`).
    #[arg(long, default_value_t = 0)]
    pub refine_multiplier: usize,

    /// Maximum local minipage LRU cache capacity in MiB.
    #[arg(long, default_value_t = 256)]
    pub cache_mb: usize,

    /// Optional `.osr` store path to apply Stage-4 ColBERT `LUT-Rank` reranking.
    #[arg(long)]
    pub rerank_store: Option<PathBuf>,

    /// Optional query token matrix (`[[...128...], ...]`) in JSON or `.json`/`.npy` file for `.osr` reranking.
    #[arg(long)]
    pub query_tokens: Option<String>,

    /// Number of Stage-3 survivors passed into `.osr` ColBERT reranking (default: Top-200).
    #[arg(long, default_value_t = 200)]
    pub rerank_top_m: usize,
}

/// Arguments for `osprey rerank`.
#[derive(Debug, Args)]
pub struct RerankArgs {
    /// Path to the `.osr` nibble-packed ColBERT token store (opened via `memmap2`).
    #[arg(long)]
    pub rerank_store: PathBuf,

    /// Query token matrix (`[[...128...], ...]`) as inline JSON or path to `.json`/`.npy` file.
    #[arg(long)]
    pub query_tokens: String,

    /// Comma-separated candidate `doc_id`s for standalone CPU LUT-Rank reranking.
    #[arg(long, value_delimiter = ',')]
    pub doc_ids: Vec<u64>,

    /// Path to a JSON array or newline-separated file of candidate `doc_id`s.
    #[arg(long)]
    pub candidates_file: Option<PathBuf>,

    /// Number of top reranked documents to return.
    #[arg(short = 'k', long, default_value_t = 10)]
    pub top_k: usize,

    /// Optional `.osi` index for end-to-end search+rerank mode when `--doc-ids` are not provided.
    #[arg(long)]
    pub index: Option<PathBuf>,

    /// Optional `.osv` slabs path/URI for end-to-end search+rerank mode.
    #[arg(long)]
    pub slabs: Option<String>,

    /// Optional dense query vector for end-to-end search+rerank mode.
    #[arg(long)]
    pub query: Option<String>,

    /// Clusters to probe in end-to-end search+rerank mode.
    #[arg(long, default_value_t = 16)]
    pub nprobe: usize,

    /// Stage-1 candidate count in end-to-end search+rerank mode.
    #[arg(long, default_value_t = 200)]
    pub candidates: usize,

    /// Oversampling multiplier for `nprobe`/`candidates` (`0` = adaptive: `2x` when `dim <= 256`).
    #[arg(long, default_value_t = 0)]
    pub refine_multiplier: usize,

    /// Local minipage LRU cache size in MiB for end-to-end search+rerank mode.
    #[arg(long, default_value_t = 256)]
    pub cache_mb: usize,

    /// Stage-3 survivors passed to `.osr` reranking in end-to-end mode.
    #[arg(long, default_value_t = 200)]
    pub rerank_top_m: usize,
}

/// Arguments for `osprey delete`.
#[derive(Debug, Args)]
pub struct DeleteArgs {
    /// Path to the existing `.osi` index file to update atomically.
    #[arg(long)]
    pub index: PathBuf,

    /// Comma-separated list of `doc_id`s to tombstone.
    #[arg(long, value_delimiter = ',')]
    pub ids: Vec<u64>,

    /// Path to a text/JSON file containing `doc_id`s to tombstone.
    #[arg(long)]
    pub ids_file: Option<PathBuf>,
}

/// Arguments for `osprey inspect`.
#[derive(Debug, Args)]
pub struct InspectArgs {
    /// One or more `.osi`, `.osv`, or `.osr` artifact paths to inspect.
    #[arg(short, long, required = true, num_args = 1..)]
    pub file: Vec<PathBuf>,

    /// Perform exhaustive `xxHash64` checksum verification across all minipages/footers.
    #[arg(long)]
    pub verify_checksums: bool,

    /// Dump minipage directory entries (`.osv`) or cluster spans (`.osi`) or doc entries (`.osr`).
    #[arg(long)]
    pub dump_directory: bool,
}

/// Cloud reranking baseline cost mode for `osprey bench` TCO projection.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum RerankCostMode {
    Gpu,
    Api,
    None,
}

/// Arguments for `osprey bench`.
#[derive(Debug, Args)]
pub struct BenchArgs {
    /// Number of synthetic vectors to benchmark locally.
    #[arg(long, default_value_t = 20_000)]
    pub vectors: usize,
    /// Vector dimensionality.
    #[arg(long, default_value_t = 384)]
    pub dim: usize,
    /// Number of queries to execute.
    #[arg(long, default_value_t = 100)]
    pub queries: usize,
    /// Number of `1-Bit Binary K-Means` clusters.
    #[arg(long, default_value_t = 128)]
    pub clusters: u32,
    /// Number of clusters probed per query.
    #[arg(long, default_value_t = 24)]
    pub nprobe: usize,
    /// Number of Stage-1 candidates refined via DP range coalescing + exact `f16`.
    #[arg(long, default_value_t = 150)]
    pub candidates: usize,
    /// Oversampling multiplier for `nprobe`/`candidates` (`0` = adaptive: `2x` when `dim <= 256`).
    #[arg(long, default_value_t = 0)]
    pub refine_multiplier: usize,
    /// Include Stage-4 `LUT-Rank` ColBERT benchmark.
    #[arg(long)]
    pub with_lutrank: bool,
    /// Number of ColBERT tokens per document ($L_d$) when `--with-lutrank` is active.
    #[arg(long, default_value_t = 16)]
    pub doc_tokens: usize,
    /// Number of ColBERT tokens per query ($L_q$) when `--with-lutrank` is active.
    #[arg(long, default_value_t = 4)]
    pub query_tokens: usize,
    /// Deterministic SplitMix64 seed for dataset generation and FWHT rotation.
    #[arg(long, default_value_t = 0x4F53_5052_0001)]
    pub seed: u64,
    /// Target scale in vectors for monthly cloud TCO projection (e.g., 50000000 or 500000000).
    #[arg(long, default_value_t = 50_000_000)]
    pub project_vectors: u64,
    /// Target vector dimensionality for monthly cloud TCO projection (e.g., 1536).
    #[arg(long, default_value_t = 1536)]
    pub project_dim: u64,
    /// Projected monthly query volume (QPM).
    #[arg(long, default_value_t = 10_000_000)]
    pub project_qpm: u64,
    /// Legacy reranking baseline mode.
    #[arg(long, value_enum, default_value_t = RerankCostMode::Gpu)]
    pub rerank_mode: RerankCostMode,
    /// Optional path to export an executive report (`.md` Markdown or `.json`).
    #[arg(long)]
    pub export_report: Option<PathBuf>,
}
