//! Shared CLI context, POSIX exit code mapping, input parsers, and ASCII/JSON formatters.

use std::fmt;
use std::fs::File;
use std::io::{BufReader, IsTerminal};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use comfy_table::presets::NOTHING;
use comfy_table::{Cell, ContentArrangement, Table, TableComponent};
use osprey_core::{OspreyError, LUTRANK_TOKEN_DIM};
use osprey_storage::NpyMatrixReader;
use serde::Serialize;
use url::Url;
use yansi::Paint;

/// POSIX exit code `0`: full success.
pub const EXIT_OK: u8 = 0;
/// POSIX exit code `2`: invalid CLI syntax or arguments.
pub const EXIT_INVALID_ARGS: u8 = 2;
/// POSIX exit code `3`: structural, alignment, checksum, or numerical error.
pub const EXIT_INTEGRITY_OR_FORMAT: u8 = 3;
/// POSIX exit code `4`: local filesystem or Object Storage I/O failure.
pub const EXIT_IO_OR_STORAGE: u8 = 4;
/// POSIX exit code `5`: diagnostic or benchmark gate failure.
pub const EXIT_GATE_FAILED: u8 = 5;

/// Fallible result alias used across `osprey-cli` commands.
pub type CliResult<T> = Result<T, CliError>;

/// Global execution context passed to every subcommand.
#[derive(Clone, Copy, Debug)]
pub struct CliContext {
    /// Emit structured JSON to `stdout` instead of human-readable ASCII tables.
    pub json: bool,
    /// Verbosity level (`0` = quiet, `1` = info, `2+` = debug) written to `stderr`.
    pub verbose: u8,
    /// Configured Rayon worker threads (`0` = automatic physical/logical cores).
    pub threads: usize,
}

impl CliContext {
    /// Logs an informational message to `stderr` when `--verbose >= 1`.
    pub fn log_info(&self, msg: impl fmt::Display) {
        if self.verbose >= 1 {
            eprintln!("[osprey info] {msg}");
        }
    }

    /// Logs a debug message to `stderr` when `--verbose >= 2`.
    pub fn log_debug(&self, msg: impl fmt::Display) {
        if self.verbose >= 2 {
            eprintln!("[osprey debug] {msg}");
        }
    }
}

/// Structured CLI error mapped deterministically to POSIX exit codes (`2..=5`).
#[derive(Debug)]
pub enum CliError {
    /// Invalid CLI arguments or query format (`exit 2`).
    InvalidArgs(String),
    /// Core binary format, alignment, checksum, or math invariant error.
    Core(OspreyError),
    /// Filesystem, parsing, or Object Storage I/O failure (`exit 4`).
    Io(String),
    /// Diagnostic gate or benchmark threshold failure (`exit 5`).
    GateFailed(String),
}

impl CliError {
    /// Returns the POSIX [`ExitCode`] corresponding to this error category.
    pub fn exit_code(&self) -> ExitCode {
        let code = self.numeric_code();
        ExitCode::from(code)
    }

    /// Returns the numeric POSIX exit code (`2`, `3`, `4`, or `5`).
    pub fn numeric_code(&self) -> u8 {
        match self {
            Self::InvalidArgs(_) => EXIT_INVALID_ARGS,
            Self::Io(_) => EXIT_IO_OR_STORAGE,
            Self::GateFailed(_) => EXIT_GATE_FAILED,
            Self::Core(err) => match err {
                OspreyError::InvalidMagic { .. }
                | OspreyError::UnsupportedVersion { .. }
                | OspreyError::InvalidFormatKind { .. }
                | OspreyError::ChecksumMismatch { .. }
                | OspreyError::UnalignedMemory { .. }
                | OspreyError::InvalidDimension { .. }
                | OspreyError::NonFiniteEmbedding { .. }
                | OspreyError::BufferTooSmall { .. }
                | OspreyError::CorruptedFooter => EXIT_INTEGRITY_OR_FORMAT,
                OspreyError::AllocationFailed { .. } | OspreyError::Io(_) => EXIT_IO_OR_STORAGE,
            },
        }
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArgs(msg) => write!(f, "invalid arguments: {msg}"),
            Self::Core(err) => write!(f, "{err}"),
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
            Self::GateFailed(msg) => write!(f, "gate check failed: {msg}"),
        }
    }
}

impl std::error::Error for CliError {}

impl From<OspreyError> for CliError {
    fn from(err: OspreyError) -> Self {
        Self::Core(err)
    }
}

impl From<std::io::Error> for CliError {
    fn from(err: std::io::Error) -> Self {
        Self::Core(OspreyError::Io(err))
    }
}

/// Resolved input/output location: either a local filesystem path or a remote Object Storage URI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LocationTarget {
    /// Local filesystem path (including decoded `file://` URIs).
    Local(PathBuf),
    /// Remote Object Storage URI (`s3://`, `gs://`, `az://`, `http://`, `https://`, `memory://`).
    Remote(String),
}

impl LocationTarget {
    /// Parses a raw CLI path or URI string into a [`LocationTarget`].
    pub fn parse(raw: &str) -> CliResult<Self> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(CliError::InvalidArgs("empty path or URI".to_string()));
        }
        if let Some(rest) = trimmed.strip_prefix("file://") {
            if let Ok(url) = Url::parse(trimmed) {
                if let Ok(path) = url.to_file_path() {
                    return Ok(Self::Local(path));
                }
            }
            return Ok(Self::Local(PathBuf::from(rest)));
        }
        if trimmed.contains("://") {
            let url = Url::parse(trimmed)
                .map_err(|err| CliError::InvalidArgs(format!("invalid URI '{trimmed}': {err}")))?;
            if url.scheme().eq_ignore_ascii_case("file") {
                let path = url.to_file_path().map_err(|()| {
                    CliError::InvalidArgs(format!("invalid file:// URI '{trimmed}'"))
                })?;
                return Ok(Self::Local(path));
            }
            return Ok(Self::Remote(trimmed.to_string()));
        }
        Ok(Self::Local(PathBuf::from(trimmed)))
    }

    /// Converts a local path or remote URI into a URI string suitable for `build_object_store`.
    pub fn to_object_store_uri(&self) -> CliResult<String> {
        match self {
            Self::Remote(uri) => Ok(uri.clone()),
            Self::Local(path) => {
                let abs = if path.is_absolute() {
                    path.clone()
                } else {
                    std::env::current_dir()?.join(path)
                };
                Url::from_file_path(&abs)
                    .map(|u| u.to_string())
                    .map_err(|()| {
                        CliError::InvalidArgs(format!(
                            "failed to convert local path '{}' to file:// URI",
                            abs.display()
                        ))
                    })
            }
        }
    }
}

/// Prints a serializable value as pretty JSON to `stdout`.
pub fn print_json<T: Serialize>(value: &T) -> CliResult<()> {
    let serialized = serde_json::to_string_pretty(value)
        .map_err(|err| CliError::Io(format!("JSON serialization failed: {err}")))?;
    println!("{serialized}");
    Ok(())
}

/// Formats a byte count using binary IEC units (`B`, `KiB`, `MiB`, `GiB`).
pub fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    #[allow(clippy::cast_precision_loss)]
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.2} GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.2} MiB", b / MIB)
    } else if b >= KIB {
        format!("{:.2} KiB", b / KIB)
    } else {
        format!("{bytes} B")
    }
}

/// Paints a passing status or supported capability in bright green.
pub fn paint_pass(text: &str) -> String {
    text.green().bright().bold().to_string()
}

/// Paints a warning or intermediate value in yellow.
pub fn paint_warn(text: &str) -> String {
    text.yellow().bold().to_string()
}

/// Paints an error or failed gate in bright red.
pub fn paint_fail(text: &str) -> String {
    text.red().bright().bold().to_string()
}

/// Paints a muted/absent feature or secondary text using terminal-native dim styling.
pub fn paint_dim(text: &str) -> String {
    text.dim().to_string()
}

/// Paints a primary metric highlight in bright cyan.
pub fn paint_accent(text: &str) -> String {
    text.cyan().bright().bold().to_string()
}

/// Paints a bold white heading.
pub fn paint_heading(text: &str) -> String {
    text.white().bright().bold().to_string()
}

/// Applies semantic similarity-score coloring:
/// `score > 0.7` in bright green, `0.3..=0.7` in yellow, `< 0.3` in dim gray.
pub fn paint_score(score: f32, formatted: &str) -> String {
    if score > 0.7 {
        formatted.green().bright().bold().to_string()
    } else if score >= 0.3 {
        formatted.yellow().to_string()
    } else {
        formatted.dim().to_string()
    }
}

/// Shortens `$HOME` prefix to `~` for clean CLI header paths.
pub fn shorten_home_path(raw: &str) -> String {
    if let Ok(home) = std::env::var("HOME") {
        if !home.is_empty() {
            if let Some(rest) = raw.strip_prefix(&home) {
                if rest.is_empty() {
                    return "~".to_string();
                }
                if rest.starts_with('/') {
                    return format!("~{rest}");
                }
            }
        }
    }
    raw.to_string()
}

/// Returns the current working directory shortened with `~`.
pub fn current_dir_short() -> String {
    std::env::current_dir()
        .map(|p| shorten_home_path(&p.display().to_string()))
        .unwrap_or_else(|_| ".".to_string())
}

/// Prints the minimalist OspreyDB typographic header + hairline divider on interactive TTYs.
pub fn print_cli_header(subtitle: &str, detail: &str, location: &str) {
    if !std::io::stdout().is_terminal() {
        return;
    }
    let version = env!("CARGO_PKG_VERSION");
    let title = format!("{} {}", paint_heading("OspreyDB"), paint_dim(version));
    let bar = "│".cyan().bright().to_string();
    let lines = [
        title,
        subtitle.to_string(),
        detail.to_string(),
        paint_dim(&shorten_home_path(location)),
    ];

    println!();
    for line in &lines {
        println!("  {bar} {line}");
    }
    println!();
    println!(
        "{}",
        paint_dim("────────────────────────────────────────────────────────────────────────")
    );
}

/// Prints a clean indented section heading.
pub fn print_section(title: &str) {
    println!();
    println!("  {}", paint_heading(title));
}

/// Prints an aligned indented key-value row.
pub fn print_kv(label: &str, value: &str) {
    println!("    {} {}", paint_dim(&format!("{label:<20}")), value);
}

/// Creates a clean borderless `comfy_table::Table` with only a single hairline rule under headers.
pub fn new_rounded_table(headers: &[&str]) -> Table {
    let mut table = Table::new();
    table
        .load_preset(NOTHING)
        .set_style(TableComponent::HeaderLines, '─')
        .set_style(TableComponent::MiddleHeaderIntersections, '─')
        .set_content_arrangement(ContentArrangement::Disabled);
    if !headers.is_empty() {
        table.set_header(
            headers
                .iter()
                .map(|h| Cell::new(h.bold().to_string()))
                .collect::<Vec<_>>(),
        );
    }
    table
}

/// Prints a `comfy_table::Table` indented cleanly by 2 spaces.
pub fn print_indented_table(table: &Table) {
    println!();
    for line in table.to_string().lines() {
        println!(" {line}");
    }
}

/// Formats a USD/month amount with comma thousands separators (`$2,326.38/mo`).
pub fn format_usd_per_mo(amount: f64) -> String {
    let formatted = format!("{amount:.2}");
    let (int_part, frac_part) = formatted.split_once('.').unwrap_or((&formatted, "00"));
    let mut out = String::with_capacity(int_part.len() + int_part.len() / 3);
    for (idx, ch) in int_part.chars().rev().enumerate() {
        if idx > 0 && idx % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    let int_commas: String = out.chars().rev().collect();
    format!("${int_commas}.{frac_part}/mo")
}

/// Parsed dense query vector alongside optional ColBERT query tokens if provided in a JSON object.
#[derive(Clone, Debug)]
pub struct ParsedQueryInput {
    /// Dense `f32` query vector.
    pub vector: Vec<f32>,
    /// Optional ColBERT multi-vector token matrix (`[M, token_dim]`).
    pub tokens: Option<Vec<Vec<f32>>>,
}

/// Parses a dense query input from:
/// 1. Inline JSON array (`"[0.1, -0.2, ...]"`) or JSON object (`{"vector": [...], "tokens": [[...]]}`),
/// 2. Local `.json` or `.npy` file path,
/// 3. Comma-separated float list (`"0.1,-0.2,0.3"`).
pub fn parse_query_input(raw: &str) -> CliResult<ParsedQueryInput> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(CliError::InvalidArgs("query cannot be empty".to_string()));
    }

    if trimmed.starts_with('[') || trimmed.starts_with('{') {
        return parse_query_json_str(trimmed);
    }

    let candidate_path = Path::new(trimmed);
    if candidate_path.exists() && candidate_path.is_file() {
        if candidate_path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("npy"))
        {
            let mut reader = NpyMatrixReader::open(candidate_path)?;
            let (_, vec) = reader.next_row_f32()?.ok_or_else(|| {
                CliError::InvalidArgs(format!(
                    ".npy query file '{}' contains zero rows",
                    candidate_path.display()
                ))
            })?;
            validate_finite_vec(&vec)?;
            return Ok(ParsedQueryInput {
                vector: vec,
                tokens: None,
            });
        }
        let content = std::fs::read_to_string(candidate_path)?;
        return parse_query_json_str(content.trim());
    }

    // Comma-separated float fallback: "0.1, -0.2, 0.3"
    if trimmed.contains(',') {
        let mut vec = Vec::new();
        for part in trimmed.split(',') {
            let val: f32 = part.trim().parse().map_err(|err| {
                CliError::InvalidArgs(format!(
                    "failed to parse query coordinate '{part}' as f32: {err}"
                ))
            })?;
            vec.push(val);
        }
        validate_finite_vec(&vec)?;
        return Ok(ParsedQueryInput {
            vector: vec,
            tokens: None,
        });
    }

    Err(CliError::InvalidArgs(format!(
        "unrecognized query input '{trimmed}': pass a JSON vector array `[f32, ...]`, comma-separated floats, or a path to a `.json`/`.npy` query file"
    )))
}

fn parse_query_json_str(raw: &str) -> CliResult<ParsedQueryInput> {
    let val: serde_json::Value = serde_json::from_str(raw)
        .map_err(|err| CliError::InvalidArgs(format!("invalid query JSON: {err}")))?;
    match val {
        serde_json::Value::Array(_) => {
            let vector: Vec<f32> = serde_json::from_value(val).map_err(|err| {
                CliError::InvalidArgs(format!("expected JSON array of f32 floats: {err}"))
            })?;
            validate_finite_vec(&vector)?;
            Ok(ParsedQueryInput {
                vector,
                tokens: None,
            })
        }
        serde_json::Value::Object(mut obj) => {
            let vec_val = obj
                .remove("vector")
                .or_else(|| obj.remove("embedding"))
                .or_else(|| obj.remove("query"))
                .ok_or_else(|| {
                    CliError::InvalidArgs(
                        "query JSON object must contain a 'vector', 'embedding', or 'query' array"
                            .to_string(),
                    )
                })?;
            let vector: Vec<f32> = serde_json::from_value(vec_val).map_err(|err| {
                CliError::InvalidArgs(format!("invalid 'vector' array in query JSON: {err}"))
            })?;
            validate_finite_vec(&vector)?;

            let tokens = match obj
                .remove("tokens")
                .or_else(|| obj.remove("query_tokens"))
                .or_else(|| obj.remove("colbert"))
            {
                Some(serde_json::Value::Null) | None => None,
                Some(tok_val) => Some(parse_tokens_value(tok_val)?),
            };
            Ok(ParsedQueryInput { vector, tokens })
        }
        _ => Err(CliError::InvalidArgs(
            "query JSON must be a float array or object with 'vector'".to_string(),
        )),
    }
}

/// Parses ColBERT query tokens (`[M, 128]`) from an inline JSON matrix, JSON file, or `.npy` file.
pub fn parse_query_tokens_input(raw: &str) -> CliResult<Vec<Vec<f32>>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(CliError::InvalidArgs(
            "query tokens cannot be empty".to_string(),
        ));
    }

    if trimmed.starts_with('[') || trimmed.starts_with('{') {
        let val: serde_json::Value = serde_json::from_str(trimmed)
            .map_err(|err| CliError::InvalidArgs(format!("invalid query tokens JSON: {err}")))?;
        return parse_tokens_value(val);
    }

    let path = Path::new(trimmed);
    if !path.exists() || !path.is_file() {
        return Err(CliError::InvalidArgs(format!(
            "query tokens input '{trimmed}' is neither valid JSON nor an existing file"
        )));
    }

    if path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("npy"))
    {
        let mut reader = NpyMatrixReader::open(path)?;
        let dim = reader.dim();
        if dim != LUTRANK_TOKEN_DIM {
            return Err(CliError::Core(OspreyError::InvalidDimension { dim }));
        }
        let mut tokens = Vec::with_capacity(reader.num_rows());
        while let Some((_, row)) = reader.next_row_f32()? {
            validate_finite_vec(&row)?;
            tokens.push(row);
        }
        if tokens.is_empty() {
            return Err(CliError::InvalidArgs(
                ".npy query tokens file contains zero rows".to_string(),
            ));
        }
        return Ok(tokens);
    }

    let file = File::open(path)?;
    let val: serde_json::Value = serde_json::from_reader(BufReader::new(file))
        .map_err(|err| CliError::InvalidArgs(format!("invalid query tokens JSON file: {err}")))?;
    parse_tokens_value(val)
}

fn parse_tokens_value(val: serde_json::Value) -> CliResult<Vec<Vec<f32>>> {
    let matrix_val = match val {
        serde_json::Value::Array(_) => val,
        serde_json::Value::Object(mut obj) => obj
            .remove("tokens")
            .or_else(|| obj.remove("query_tokens"))
            .or_else(|| obj.remove("colbert"))
            .ok_or_else(|| {
                CliError::InvalidArgs(
                    "query tokens JSON object must contain 'tokens' or 'query_tokens'".to_string(),
                )
            })?,
        _ => {
            return Err(CliError::InvalidArgs(
                "query tokens must be a 2D JSON array [[f32; 128], ...]".to_string(),
            ));
        }
    };

    let tokens: Vec<Vec<f32>> = serde_json::from_value(matrix_val).map_err(|err| {
        CliError::InvalidArgs(format!("expected 2D array of f32 token vectors: {err}"))
    })?;
    if tokens.is_empty() {
        return Err(CliError::InvalidArgs(
            "query tokens matrix must contain at least 1 token".to_string(),
        ));
    }
    for tok in &tokens {
        if tok.len() != LUTRANK_TOKEN_DIM {
            return Err(CliError::Core(OspreyError::InvalidDimension {
                dim: tok.len(),
            }));
        }
        validate_finite_vec(tok)?;
    }
    Ok(tokens)
}

/// Validates that `vec` is non-empty and every coordinate is finite (`!NaN && !Infinite`).
pub fn validate_finite_vec(vec: &[f32]) -> CliResult<()> {
    if vec.is_empty() {
        return Err(CliError::Core(OspreyError::InvalidDimension { dim: 0 }));
    }
    for (idx, &v) in vec.iter().enumerate() {
        if !v.is_finite() {
            return Err(CliError::Core(OspreyError::NonFiniteEmbedding {
                index: idx,
            }));
        }
    }
    Ok(())
}

/// Parses an explicit JSON array, JSON object list, or newline-delimited list of `u64` document IDs.
pub fn parse_doc_ids_file(path: &Path) -> CliResult<Vec<u64>> {
    let raw = std::fs::read_to_string(path)?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    if trimmed.starts_with('[') {
        if let Ok(ids) = serde_json::from_str::<Vec<u64>>(trimmed) {
            return Ok(ids);
        }
        if let Ok(vals) = serde_json::from_str::<Vec<serde_json::Value>>(trimmed) {
            let mut ids = Vec::with_capacity(vals.len());
            for (idx, v) in vals.into_iter().enumerate() {
                let id = v
                    .get("doc_id")
                    .or_else(|| v.get("id"))
                    .and_then(|x| x.as_u64())
                    .ok_or_else(|| {
                        CliError::InvalidArgs(format!(
                            "candidate entry #{idx} in '{}' missing u64 'doc_id'/'id'",
                            path.display()
                        ))
                    })?;
                ids.push(id);
            }
            return Ok(ids);
        }
    }

    let mut ids = Vec::new();
    for (line_idx, line) in trimmed.lines().enumerate() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        for tok in l.split(',') {
            let tok = tok.trim();
            if tok.is_empty() {
                continue;
            }
            let id: u64 = tok.parse().map_err(|err| {
                CliError::InvalidArgs(format!(
                    "invalid doc_id '{tok}' on line {} of '{}': {err}",
                    line_idx + 1,
                    path.display()
                ))
            })?;
            ids.push(id);
        }
    }
    Ok(ids)
}
