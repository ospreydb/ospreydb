//! Private developer CLI (`osprey-keygen`) for generating Ed25519 keypairs and
//! issuing `OSPREY-PRO-v1.<base64url_payload>.<base64url_sig>` offline licenses
//! for all three commercial waves:
//!   - `--tier beta --days 90`  (free Design Partners)
//!   - `--tier founding --days 365`  (discounted Early Access)
//!   - `--tier enterprise --days 365`  (unlimited scale, multi-node S3, Arrow Flight)

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use base64::Engine as _;
use clap::{Args, Parser, Subcommand};
use ed25519_dalek::{Signer as _, SigningKey};
use osprey_pro::{
    OfflineLicenseVerifier, OspreyLicense, ProLicenseError, COMMUNITY_MAX_VECTORS,
    LICENSE_TOKEN_PREFIX,
};
use rand::rngs::OsRng;
use thiserror::Error;

const ALL_PRO_FEATURES: &[&str] = &[
    "wal_streaming",
    "cow_compactor",
    "tombstone_gc",
    "s3_cas_manifest",
    "arrow_flight",
    "rerank_http",
    "qdrant_compat",
];

/// Feature set for Beta / Design Partner tier (90-day free).
const BETA_FEATURES: &[&str] = &[
    "wal_streaming",
    "cow_compactor",
    "qdrant_compat",
];

/// Feature set for Founding tier (discounted Early Access, 1-year).
const FOUNDING_FEATURES: &[&str] = &[
    "wal_streaming",
    "cow_compactor",
    "tombstone_gc",
    "s3_cas_manifest",
    "arrow_flight",
    "rerank_http",
    "qdrant_compat",
];

/// Feature set for Enterprise tier (unlimited scale, multi-node, Arrow Flight).
const ENTERPRISE_FEATURES: &[&str] = &[
    "wal_streaming",
    "cow_compactor",
    "tombstone_gc",
    "s3_cas_manifest",
    "arrow_flight",
    "rerank_http",
    "qdrant_compat",
];

#[derive(Debug, Error)]
enum KeygenError {
    #[error("missing required argument: --org / --customer-org")]
    MissingOrg,

    #[error(
        "missing Ed25519 private key: pass --private-key, --private-key-file, or set OSPREY_PRO_SIGNING_KEY"
    )]
    MissingPrivateKey,

    #[error("invalid Ed25519 private key encoding: {0}")]
    InvalidPrivateKey(String),

    #[error("invalid Ed25519 public key encoding: {0}")]
    InvalidPublicKey(String),

    #[error("expires_at ({expires_at}) must be strictly greater than issued_at ({issued_at})")]
    InvalidValidityWindow { issued_at: u64, expires_at: u64 },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("license verification error: {0}")]
    Verify(#[from] ProLicenseError),

    #[error("system clock error")]
    Clock,
}

#[derive(Debug, Parser)]
#[command(
    name = "osprey-keygen",
    about = "Private developer tool for issuing Ed25519-signed OSPREY-PRO-v1 offline licenses",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    #[command(flatten)]
    sign: SignArgs,
}

#[derive(Debug, Subcommand)]
enum Command {
    Sign(SignArgs),
    Keygen(KeygenArgs),
    Verify(VerifyArgs),
}

#[derive(Debug, Args)]
struct SignArgs {
    #[arg(long = "org", visible_alias = "customer-org")]
    customer_org: Option<String>,

    /// Commercial plan tier: `beta`, `founding`, `enterprise`, or custom.
    #[arg(long, default_value = "enterprise")]
    tier: String,

    /// Maximum concurrent nodes (`0` = unlimited).
    #[arg(long, default_value_t = 0)]
    max_nodes: u32,

    /// Maximum vector count (`0` = unlimited, `5_000_000` = community default).
    #[arg(long, default_value_t = 0)]
    max_vectors: u64,

    #[arg(long)]
    issued_at: Option<u64>,

    #[arg(long, conflicts_with = "ttl_days")]
    expires_at: Option<u64>,

    /// Validity period in days from `issued_at` (used when `--expires-at` is omitted).
    ///   90 days = Beta Design Partners, 365 days = Founding/Enterprise.
    #[arg(long, default_value_t = 365)]
    ttl_days: u64,

    #[arg(long = "features", value_delimiter = ',')]
    features: Vec<String>,

    #[arg(long)]
    all_features: bool,

    #[arg(
        long,
        env = "OSPREY_PRO_SIGNING_KEY",
        conflicts_with = "private_key_file"
    )]
    private_key: Option<String>,

    #[arg(long)]
    private_key_file: Option<PathBuf>,

    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct KeygenArgs {
    #[arg(long)]
    out_private: Option<PathBuf>,

    #[arg(long)]
    out_public: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct VerifyArgs {
    #[arg(long)]
    token: String,

    #[arg(long, conflicts_with = "public_key_file")]
    public_key: Option<String>,

    #[arg(long)]
    public_key_file: Option<PathBuf>,

    #[arg(long)]
    now: Option<u64>,
}

fn current_unix_secs() -> Result<u64, KeygenError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|dur| dur.as_secs())
        .map_err(|_| KeygenError::Clock)
}

fn decode_32_bytes_text(raw: &str) -> Result<[u8; 32], String> {
    let trimmed = raw.trim();
    if trimmed.len() == 64 && trimmed.bytes().all(|b| b.is_ascii_hexdigit()) {
        let mut out = [0_u8; 32];
        let (chunks, _) = trimmed.as_bytes().as_chunks::<2>();
        for (idx, chunk) in chunks.iter().enumerate() {
            let hex_pair =
                std::str::from_utf8(chunk).map_err(|e| format!("invalid UTF-8 hex byte: {e}"))?;
            out[idx] = u8::from_str_radix(hex_pair, 16)
                .map_err(|e| format!("invalid hex byte '{hex_pair}': {e}"))?;
        }
        return Ok(out);
    }

    let decoded = URL_SAFE_NO_PAD
        .decode(trimmed)
        .or_else(|_| STANDARD.decode(trimmed))
        .map_err(|e| format!("expected 64-char hex or base64 32-byte key: {e}"))?;

    if decoded.len() == 32 || decoded.len() == 64 {
        let mut out = [0_u8; 32];
        out.copy_from_slice(&decoded[..32]);
        return Ok(out);
    }

    Err(format!(
        "decoded key has {} bytes; expected 32 bytes (or 64-byte keypair)",
        decoded.len()
    ))
}

fn load_private_key(args: &SignArgs) -> Result<SigningKey, KeygenError> {
    let seed = if let Some(path) = &args.private_key_file {
        let bytes = fs::read(path)?;
        if bytes.len() == 32 {
            let mut out = [0_u8; 32];
            out.copy_from_slice(&bytes);
            out
        } else {
            let text = String::from_utf8(bytes)
                .map_err(|e| KeygenError::InvalidPrivateKey(e.to_string()))?;
            decode_32_bytes_text(&text).map_err(KeygenError::InvalidPrivateKey)?
        }
    } else if let Some(raw) = &args.private_key {
        decode_32_bytes_text(raw).map_err(KeygenError::InvalidPrivateKey)?
    } else {
        return Err(KeygenError::MissingPrivateKey);
    };

    Ok(SigningKey::from_bytes(&seed))
}

fn load_public_key_bytes(
    inline: Option<&str>,
    file: Option<&Path>,
) -> Result<Option<[u8; 32]>, KeygenError> {
    if let Some(path) = file {
        let bytes = fs::read(path)?;
        if bytes.len() == 32 {
            let mut out = [0_u8; 32];
            out.copy_from_slice(&bytes);
            return Ok(Some(out));
        }
        let text =
            String::from_utf8(bytes).map_err(|e| KeygenError::InvalidPublicKey(e.to_string()))?;
        let parsed = decode_32_bytes_text(&text).map_err(KeygenError::InvalidPublicKey)?;
        return Ok(Some(parsed));
    }
    if let Some(raw) = inline {
        let parsed = decode_32_bytes_text(raw).map_err(KeygenError::InvalidPublicKey)?;
        return Ok(Some(parsed));
    }
    Ok(None)
}

fn encode_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(out, "{b:02x}");
    }
    out
}

fn sign_license_token(
    signing_key: &SigningKey,
    license: &OspreyLicense,
) -> Result<String, KeygenError> {
    let payload_json = serde_json::to_vec(license)?;
    let payload_b64 = URL_SAFE_NO_PAD.encode(payload_json);
    let signed_message = format!("{LICENSE_TOKEN_PREFIX}.{payload_b64}");
    let signature = signing_key.sign(signed_message.as_bytes());
    let sig_b64 = URL_SAFE_NO_PAD.encode(signature.to_bytes());
    Ok(format!("{signed_message}.{sig_b64}"))
}

/// Resolves the feature list based on tier and user-provided flags.
fn resolve_features(tier: &str, provided: &[String], all: bool) -> Vec<String> {
    if all {
        let mut feats: Vec<String> = ALL_PRO_FEATURES.iter().map(|s| (*s).to_owned()).collect();
        for p in provided {
            if !feats.contains(p) {
                feats.push(p.clone());
            }
        }
        return feats;
    }

    if !provided.is_empty() {
        return provided.to_vec();
    }

    match tier {
        "beta" => BETA_FEATURES.iter().map(|s| (*s).to_owned()).collect(),
        "founding" => FOUNDING_FEATURES.iter().map(|s| (*s).to_owned()).collect(),
        "enterprise" => ENTERPRISE_FEATURES.iter().map(|s| (*s).to_owned()).collect(),
        _ => ALL_PRO_FEATURES.iter().map(|s| (*s).to_owned()).collect(),
    }
}

/// Resolves max_vectors based on tier: founding/enterprise are unlimited (0),
/// beta defaults to 100k, others keep user-provided value.
fn resolve_max_vectors(tier: &str, user_value: u64) -> u64 {
    if user_value != 0 {
        return user_value;
    }
    match tier {
        "founding" | "enterprise" => 0,
        "beta" => 100_000,
        _ => COMMUNITY_MAX_VECTORS,
    }
}

fn run_sign(args: SignArgs) -> Result<(), KeygenError> {
    let customer_org = args
        .customer_org
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or(KeygenError::MissingOrg)?
        .to_owned();

    let signing_key = load_private_key(&args)?;
    let issued_at = match args.issued_at {
        Some(ts) => ts,
        None => current_unix_secs()?,
    };
    let expires_at = match args.expires_at {
        Some(ts) => ts,
        None => issued_at.saturating_add(args.ttl_days.saturating_mul(86_400)),
    };

    if expires_at <= issued_at {
        return Err(KeygenError::InvalidValidityWindow {
            issued_at,
            expires_at,
        });
    }

    let features = resolve_features(&args.tier, &args.features, args.all_features);
    let max_vectors = resolve_max_vectors(&args.tier, args.max_vectors);

    let license = OspreyLicense {
        customer_org,
        tier: args.tier,
        max_nodes: args.max_nodes,
        max_vectors,
        issued_at,
        expires_at,
        features,
    };

    let token = sign_license_token(&signing_key, &license)?;
    if args.json {
        let output = serde_json::json!({
            "token": token,
            "license": license,
            "public_key_hex": encode_hex(&signing_key.verifying_key().to_bytes()),
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("{token}");
    }
    Ok(())
}

fn write_private_key_file(path: &Path, bytes: &[u8; 32]) -> Result<(), KeygenError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(path, bytes)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let perms = fs::Permissions::from_mode(0o600);
        fs::set_permissions(path, perms)?;
    }
    Ok(())
}

fn run_keygen(args: KeygenArgs) -> Result<(), KeygenError> {
    let signing_key = SigningKey::generate(&mut OsRng);
    let private_bytes = signing_key.to_bytes();
    let public_bytes = signing_key.verifying_key().to_bytes();

    if let Some(path) = &args.out_private {
        write_private_key_file(path, &private_bytes)?;
    }
    if let Some(path) = &args.out_public {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, public_bytes)?;
    }

    let summary = serde_json::json!({
        "public_key_hex": encode_hex(&public_bytes),
        "public_key_b64": URL_SAFE_NO_PAD.encode(public_bytes),
        "private_key_written_to": args.out_private,
        "public_key_written_to": args.out_public,
        "private_key_hex": if args.out_private.is_none() {
            Some(encode_hex(&private_bytes))
        } else {
            None
        }
    });
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn run_verify(args: VerifyArgs) -> Result<(), KeygenError> {
    let verifier =
        match load_public_key_bytes(args.public_key.as_deref(), args.public_key_file.as_deref())? {
            Some(pk) => OfflineLicenseVerifier::from_public_key_bytes(&pk)?,
            None => OfflineLicenseVerifier::new()?,
        };

    let now = match args.now {
        Some(ts) => ts,
        None => current_unix_secs()?,
    };
    let license = verifier.verify(&args.token, now)?;
    println!("{}", serde_json::to_string_pretty(&license)?);
    Ok(())
}

fn dispatch(cli: Cli) -> Result<(), KeygenError> {
    match cli.command {
        Some(Command::Sign(args)) => run_sign(args),
        Some(Command::Keygen(args)) => run_keygen(args),
        Some(Command::Verify(args)) => run_verify(args),
        None => run_sign(cli.sign),
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match dispatch(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn keygen_sign_and_verify_round_trip() {
        let signing_key = SigningKey::generate(&mut OsRng);
        let pk = signing_key.verifying_key().to_bytes();

        let now = 1_794_856_000_u64;
        let license = OspreyLicense {
            customer_org: "fintech-prod".to_owned(),
            tier: "enterprise".to_owned(),
            max_nodes: 8,
            max_vectors: 0,
            issued_at: now - 10,
            expires_at: now + 86_400 * 90,
            features: vec!["wal_streaming".to_owned(), "s3_cas_manifest".to_owned()],
        };

        let token = sign_license_token(&signing_key, &license).unwrap();
        assert!(token.starts_with("OSPREY-PRO-v1."));
        let verifier = OfflineLicenseVerifier::from_public_key_bytes(&pk).unwrap();
        let decoded = verifier.verify(&token, now).unwrap();
        assert_eq!(decoded, license);
    }

    #[test]
    fn hex_and_base64_private_key_decoding() {
        let signing_key = SigningKey::generate(&mut OsRng);
        let raw = signing_key.to_bytes();

        let hex_str = encode_hex(&raw);
        let b64_str = URL_SAFE_NO_PAD.encode(raw);

        assert_eq!(decode_32_bytes_text(&hex_str).unwrap(), raw);
        assert_eq!(decode_32_bytes_text(&b64_str).unwrap(), raw);
    }

    #[test]
    fn tier_resolves_correct_features_and_vector_limits() {
        assert_eq!(
            resolve_features("beta", &[], false),
            BETA_FEATURES.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>()
        );
        assert_eq!(
            resolve_features("founding", &[], false),
            FOUNDING_FEATURES.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>()
        );
        assert_eq!(
            resolve_features("enterprise", &[], false),
            ENTERPRISE_FEATURES.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>()
        );

        assert_eq!(resolve_max_vectors("founding", 0), 0);
        assert_eq!(resolve_max_vectors("enterprise", 0), 0);
        assert_eq!(resolve_max_vectors("beta", 0), 100_000);
        assert_eq!(resolve_max_vectors("beta", 5000), 5000);
        assert_eq!(resolve_max_vectors("unknown", 0), COMMUNITY_MAX_VECTORS);
    }
}