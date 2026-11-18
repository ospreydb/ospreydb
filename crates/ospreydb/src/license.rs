//! 100% offline `Ed25519` cryptographic license verification (`< 50 us`, zero phone-home).
//!
//! Canonical license format (`ADR-003` §4):
//! `OSPREY-PRO-v1.<base64url_payload>.<base64url_ed25519_sig>`
//!
//! The 64-byte Ed25519 signature is computed over the exact ASCII bytes of:
//! `"OSPREY-PRO-v1." || <base64url_payload>`

use std::io::{Error as IoError, ErrorKind};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use ed25519_dalek::{Signature, VerifyingKey};
use osprey_core::{LicenseClaims, LicenseGuard, OspreyError, OspreyResult, ProCapability};
use serde::{Deserialize, Serialize};

use crate::error::{ProEngineError, ProLicenseError, ProResult};

/// Canonical token version prefix.
pub const LICENSE_TOKEN_PREFIX: &str = "OSPREY-PRO-v1";

/// Maximum accepted license token length in bytes (prevents unbounded decoding).
pub const MAX_LICENSE_TOKEN_BYTES: usize = 16 * 1024;

/// Maximum forward clock skew tolerated on `issued_at` (5 minutes).
pub const MAX_CLOCK_SKEW_SECS: u64 = 300;

/// Stack buffer capacity used for base64url payload decoding without heap allocation.
const STACK_PAYLOAD_BUF_BYTES: usize = 4096;

/// Stack buffer capacity for 64-byte Ed25519 signature decoding.
const STACK_SIG_BUF_BYTES: usize = 96;

/// Statically embedded 32-byte Ed25519 public verifying key (`zero phone-home`).
pub const EMBEDDED_PUBLIC_KEY: &[u8; 32] = include_bytes!("osprey_pro_ed25519.pub");

/// Community tier hard limit: 5,000,000 vectors without a Pro license key.
pub const COMMUNITY_MAX_VECTORS: u64 = 5_000_000;

/// Renewal grace period after `expires_at` before Pro mode is blocked (30 days in seconds).
pub const LICENSE_GRACE_PERIOD_SECS: u64 = 30 * 86_400;

/// Commercial license claims decoded from an `OSPREY-PRO-v1.<payload>.<sig>` token.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OspreyLicense {
    /// Licensed customer organization identifier.
    #[serde(alias = "org_name")]
    pub customer_org: String,
    /// Commercial plan tier (`"personal"`, `"founding"`, `"enterprise"`, `"beta"`).
    #[serde(alias = "plan_tier")]
    pub tier: String,
    /// Maximum concurrent cluster nodes authorized (`0` means unlimited).
    pub max_nodes: u32,
    /// Maximum vector count (`0` means unlimited, i.e. no hard cap).
    #[serde(default)]
    pub max_vectors: u64,
    /// Issuance timestamp in Unix seconds (UTC).
    pub issued_at: u64,
    /// Expiration timestamp in Unix seconds (UTC).
    pub expires_at: u64,
    /// Enabled commercial feature scopes.
    #[serde(alias = "feature_flags")]
    pub features: Vec<String>,
}

impl OspreyLicense {
    /// Returns `true` if `feature` is explicitly listed in [`Self::features`].
    #[must_use]
    pub fn has_feature(&self, feature: &str) -> bool {
        self.features.iter().any(|entry| entry == feature)
    }

    /// Ensures `feature` is enabled, returning [`ProLicenseError::MissingFeature`] otherwise.
    pub fn require_feature(&self, feature: &str) -> Result<(), ProLicenseError> {
        if self.has_feature(feature) {
            Ok(())
        } else {
            Err(ProLicenseError::MissingFeature {
                feature: feature.to_owned(),
            })
        }
    }

    /// Returns `true` if the license authorizes the given [`ProCapability`].
    #[must_use]
    pub fn has_capability(&self, capability: ProCapability) -> bool {
        self.has_feature(capability_name(capability))
    }

    /// Returns `true` if the license is past its nominal `expires_at` timestamp at `now_unix_secs`.
    #[must_use]
    pub const fn is_expired_at(&self, now_unix_secs: u64) -> bool {
        now_unix_secs >= self.expires_at
    }

    /// Returns the hard cutoff Unix timestamp (`expires_at + 30 days`) after which Pro mode is blocked.
    #[must_use]
    pub const fn grace_deadline(&self) -> u64 {
        self.expires_at.saturating_add(LICENSE_GRACE_PERIOD_SECS)
    }

    /// Returns `true` if the license is past `expires_at` but still within the 30-day renewal grace period.
    #[must_use]
    pub const fn is_in_grace_period_at(&self, now_unix_secs: u64) -> bool {
        now_unix_secs >= self.expires_at && now_unix_secs <= self.grace_deadline()
    }

    /// Returns `true` if `now_unix_secs` is strictly past the 30-day renewal grace period.
    #[must_use]
    pub const fn is_past_grace_period_at(&self, now_unix_secs: u64) -> bool {
        now_unix_secs > self.grace_deadline()
    }

    /// Returns the remaining whole days in the 30-day grace period, or `0` if not in grace period.
    #[must_use]
    pub const fn grace_days_remaining_at(&self, now_unix_secs: u64) -> u64 {
        if self.is_in_grace_period_at(now_unix_secs) {
            self.grace_deadline().saturating_sub(now_unix_secs) / 86_400
        } else {
            0
        }
    }

    /// Returns `true` if `active_nodes` does not exceed [`Self::max_nodes`] (`0` = unlimited).
    #[must_use]
    pub const fn allows_nodes(&self, active_nodes: u32) -> bool {
        self.max_nodes == 0 || active_nodes <= self.max_nodes
    }

    /// Returns `true` if `vector_count` does not exceed [`Self::max_vectors`] (`0` = unlimited).
    #[must_use]
    pub const fn allows_vectors(&self, vector_count: u64) -> bool {
        self.max_vectors == 0 || vector_count <= self.max_vectors
    }

    /// Returns `true` if this is a `founding` or `enterprise` license that unlocks unlimited vectors.
    #[must_use]
    pub fn is_unlimited_vectors(&self) -> bool {
        self.max_vectors == 0
    }

    /// Converts this license payload into the open-core [`LicenseClaims`] projection.
    #[must_use]
    pub fn to_license_claims(&self) -> LicenseClaims {
        let capabilities = self
            .features
            .iter()
            .filter_map(|name| capability_from_str(name))
            .collect();

        LicenseClaims {
            org_name: self.customer_org.clone(),
            plan_tier: self.tier.clone(),
            max_nodes: self.max_nodes,
            issued_at: self.issued_at,
            expires_at: self.expires_at,
            capabilities,
        }
    }
}

/// Maps a canonical feature string to [`ProCapability`].
#[must_use]
pub fn capability_from_str(name: &str) -> Option<ProCapability> {
    match name {
        "wal_streaming" => Some(ProCapability::WalStreaming),
        "cow_compactor" => Some(ProCapability::CowCompactor),
        "tombstone_gc" => Some(ProCapability::TombstoneGc),
        "s3_cas_manifest" => Some(ProCapability::S3CasManifest),
        "arrow_flight" => Some(ProCapability::ArrowFlight),
        "rerank_http" => Some(ProCapability::RerankHttp),
        "qdrant_compat" => Some(ProCapability::QdrantCompat),
        _ => None,
    }
}

/// Maps a [`ProCapability`] to its canonical feature string identifier.
#[must_use]
pub const fn capability_name(capability: ProCapability) -> &'static str {
    match capability {
        ProCapability::WalStreaming => "wal_streaming",
        ProCapability::CowCompactor => "cow_compactor",
        ProCapability::TombstoneGc => "tombstone_gc",
        ProCapability::S3CasManifest => "s3_cas_manifest",
        ProCapability::ArrowFlight => "arrow_flight",
        ProCapability::RerankHttp => "rerank_http",
        ProCapability::QdrantCompat => "qdrant_compat",
    }
}

const fn capability_bit(capability: ProCapability) -> u8 {
    match capability {
        ProCapability::WalStreaming => 1 << 0,
        ProCapability::CowCompactor => 1 << 1,
        ProCapability::TombstoneGc => 1 << 2,
        ProCapability::S3CasManifest => 1 << 3,
        ProCapability::ArrowFlight => 1 << 4,
        ProCapability::RerankHttp => 1 << 5,
        ProCapability::QdrantCompat => 1 << 6,
    }
}

/// Offline `Ed25519` license verifier (`< 50 us`, zero HTTP/network calls).
#[derive(Debug)]
pub struct OfflineLicenseVerifier {
    verifying_key: VerifyingKey,
    active_capabilities: AtomicU8,
}

/// Compatibility alias for [`OfflineLicenseVerifier`].
pub type Ed25519LicenseVerifier = OfflineLicenseVerifier;

impl OfflineLicenseVerifier {
    /// Creates a verifier backed by the statically embedded 32-byte Ed25519 public key.
    pub fn new() -> Result<Self, ProLicenseError> {
        Self::from_public_key_bytes(EMBEDDED_PUBLIC_KEY)
    }

    /// Creates a verifier from an explicit 32-byte Ed25519 public verifying key.
    pub fn from_public_key_bytes(bytes: &[u8; 32]) -> Result<Self, ProLicenseError> {
        let verifying_key =
            VerifyingKey::from_bytes(bytes).map_err(|_| ProLicenseError::InvalidPublicKey)?;
        Ok(Self {
            verifying_key,
            active_capabilities: AtomicU8::new(0),
        })
    }

    /// Returns the raw 32-byte Ed25519 public key configured in this verifier.
    #[must_use]
    pub fn public_key_bytes(&self) -> [u8; 32] {
        self.verifying_key.to_bytes()
    }

    /// Clears cached runtime capability flags.
    pub fn deactivate(&self) {
        self.active_capabilities.store(0, Ordering::Release);
    }

    /// Verifies an `OSPREY-PRO-v1.<base64url_payload>.<base64url_sig>` token 100% offline.
    ///
    /// Signature verification runs strictly over `"OSPREY-PRO-v1." || <base64url_payload>`
    /// before JSON deserialization, and checks expiration and `issued_at` clock skew against
    /// `now_unix_secs`.
    pub fn verify(
        &self,
        token: &str,
        now_unix_secs: u64,
    ) -> Result<OspreyLicense, ProLicenseError> {
        let token = token.trim();
        if token.len() > MAX_LICENSE_TOKEN_BYTES {
            return Err(ProLicenseError::TokenTooLarge {
                len: token.len(),
                max: MAX_LICENSE_TOKEN_BYTES,
            });
        }

        let (signed_message, sig_b64) = token
            .rsplit_once('.')
            .ok_or(ProLicenseError::InvalidFormat)?;
        let (version_prefix, payload_b64) = signed_message
            .split_once('.')
            .ok_or(ProLicenseError::InvalidFormat)?;

        if version_prefix != LICENSE_TOKEN_PREFIX {
            if version_prefix.starts_with("OSPREY-PRO-") {
                return Err(ProLicenseError::UnsupportedVersion {
                    found: version_prefix.to_owned(),
                });
            }
            return Err(ProLicenseError::InvalidFormat);
        }

        if payload_b64.is_empty() || sig_b64.is_empty() || payload_b64.contains('.') {
            return Err(ProLicenseError::InvalidFormat);
        }

        // Decode 64-byte Ed25519 signature onto stack without heap allocation.
        if sig_b64.len() > 128 {
            return Err(ProLicenseError::InvalidSignatureLength {
                expected: 64,
                actual: sig_b64.len(),
            });
        }
        let mut sig_buf = [0_u8; STACK_SIG_BUF_BYTES];
        let sig_len = URL_SAFE_NO_PAD.decode_slice(sig_b64.as_bytes(), &mut sig_buf)?;
        if sig_len != 64 {
            return Err(ProLicenseError::InvalidSignatureLength {
                expected: 64,
                actual: sig_len,
            });
        }
        let signature = Signature::from_slice(&sig_buf[..64]).map_err(|_| {
            ProLicenseError::InvalidSignatureLength {
                expected: 64,
                actual: sig_len,
            }
        })?;

        // Verify cryptographic signature BEFORE parsing JSON payload.
        self.verifying_key
            .verify_strict(signed_message.as_bytes(), &signature)
            .map_err(|_| ProLicenseError::InvalidSignature)?;

        let license = decode_license_payload(payload_b64)?;
        validate_claims(&license, now_unix_secs)?;
        Ok(license)
    }

    /// Verifies `token` using the current system UTC clock.
    pub fn verify_now(&self, token: &str) -> Result<OspreyLicense, ProLicenseError> {
        self.verify(token, current_unix_secs()?)
    }

    /// Verifies `token` offline and ensures `required_feature` is granted.
    pub fn verify_for_feature(
        &self,
        token: &str,
        now_unix_secs: u64,
        required_feature: &str,
    ) -> Result<OspreyLicense, ProLicenseError> {
        let license = self.verify(token, now_unix_secs)?;
        license.require_feature(required_feature)?;
        Ok(license)
    }

    /// Verifies `token` offline, checking optional feature scope and node count limits.
    pub fn verify_for_cluster(
        &self,
        token: &str,
        now_unix_secs: u64,
        required_feature: Option<&str>,
        active_nodes: u32,
    ) -> Result<OspreyLicense, ProLicenseError> {
        let license = self.verify(token, now_unix_secs)?;
        if let Some(feature) = required_feature {
            license.require_feature(feature)?;
        }
        if !license.allows_nodes(active_nodes) {
            return Err(ProLicenseError::NodeLimitExceeded {
                max_nodes: license.max_nodes,
                active_nodes,
            });
        }
        Ok(license)
    }

    /// Verifies `token`, caches authorized capabilities for [`LicenseGuard::is_enabled`],
    /// and returns the verified [`OspreyLicense`].
    pub fn verify_and_activate(
        &self,
        token: &str,
        now_unix_secs: u64,
    ) -> Result<OspreyLicense, ProLicenseError> {
        let license = self.verify(token, now_unix_secs)?;
        let mut mask = 0_u8;
        for feature in &license.features {
            if let Some(cap) = capability_from_str(feature) {
                mask |= capability_bit(cap);
            }
        }
        self.active_capabilities.store(mask, Ordering::Release);
        Ok(license)
    }
}

impl LicenseGuard for OfflineLicenseVerifier {
    fn verify_token(&self, token: &str, now_unix_secs: u64) -> OspreyResult<LicenseClaims> {
        let license = self
            .verify_and_activate(token, now_unix_secs)
            .map_err(|err| OspreyError::Io(IoError::new(ErrorKind::PermissionDenied, err)))?;
        Ok(license.to_license_claims())
    }

    fn is_enabled(&self, capability: ProCapability) -> bool {
        (self.active_capabilities.load(Ordering::Acquire) & capability_bit(capability)) != 0
    }
}

fn decode_license_payload(payload_b64: &str) -> Result<OspreyLicense, ProLicenseError> {
    if payload_b64.len() <= STACK_PAYLOAD_BUF_BYTES {
        let mut stack_buf = [0_u8; STACK_PAYLOAD_BUF_BYTES];
        let decoded_len = URL_SAFE_NO_PAD.decode_slice(payload_b64.as_bytes(), &mut stack_buf)?;
        let license = serde_json::from_slice(&stack_buf[..decoded_len])?;
        Ok(license)
    } else {
        let decoded = URL_SAFE_NO_PAD.decode(payload_b64.as_bytes())?;
        let license = serde_json::from_slice(&decoded)?;
        Ok(license)
    }
}

fn validate_claims(license: &OspreyLicense, now_unix_secs: u64) -> Result<(), ProLicenseError> {
    if license.customer_org.trim().is_empty() {
        return Err(ProLicenseError::InvalidClaim {
            reason: "customer_org must not be empty",
        });
    }
    if license.tier.trim().is_empty() {
        return Err(ProLicenseError::InvalidClaim {
            reason: "tier must not be empty",
        });
    }
    if license.expires_at <= license.issued_at {
        return Err(ProLicenseError::InvalidClaim {
            reason: "expires_at must be strictly greater than issued_at",
        });
    }
    if license.issued_at > now_unix_secs.saturating_add(MAX_CLOCK_SKEW_SECS) {
        return Err(ProLicenseError::IssuedInFuture {
            issued_at: license.issued_at,
            now_unix_secs,
        });
    }
    if license.is_past_grace_period_at(now_unix_secs) {
        return Err(ProLicenseError::LicenseExpired {
            expires_at: license.expires_at,
            now_unix_secs,
        });
    }
    Ok(())
}

/// Returns the current UTC Unix timestamp in seconds.
pub fn current_unix_secs() -> Result<u64, ProLicenseError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .map_err(|_| ProLicenseError::SystemClockError)
}

/// Shared thread-safe vector quota tracker enforced across HTTP and Arrow Flight ingestion.
///
/// When the active limit (`5_000_000` in Community Tier) is reached, read/search/rerank
/// operations remain fully operational while new vector insertions are rejected.
#[derive(Clone, Debug, Default)]
pub struct VectorQuotaState {
    total_vectors: Arc<AtomicU64>,
    active_license: Arc<RwLock<Option<OspreyLicense>>>,
}

impl VectorQuotaState {
    /// Creates a new [`VectorQuotaState`] starting at zero vectors with no Pro license.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a [`VectorQuotaState`] initialized with `initial_vectors` recovered from storage.
    #[must_use]
    pub fn with_initial_vectors(initial_vectors: u64) -> Self {
        Self {
            total_vectors: Arc::new(AtomicU64::new(initial_vectors)),
            active_license: Arc::new(RwLock::new(None)),
        }
    }

    /// Activates a verified commercial [`OspreyLicense`], updating the vector ceiling.
    pub fn activate_license(&self, license: OspreyLicense) {
        if let Ok(mut guard) = self.active_license.write() {
            *guard = Some(license);
        }
    }

    /// Returns a snapshot of the currently active license, if any.
    #[must_use]
    pub fn active_license_snapshot(&self) -> Option<OspreyLicense> {
        self.active_license
            .read()
            .ok()
            .and_then(|guard| guard.clone())
    }

    /// Returns the tracked total vector count across all collections.
    #[must_use]
    pub fn total_vector_count(&self) -> u64 {
        self.total_vectors.load(Ordering::Acquire)
    }

    /// Sets the tracked total vector count (e.g. after WAL and manifest recovery).
    pub fn set_vector_count(&self, count: u64) {
        self.total_vectors.store(count, Ordering::Release);
    }

    /// Increments the tracked total vector count by `delta` after successful ingestion.
    pub fn increment_vector_count(&self, delta: u64) {
        self.total_vectors.fetch_add(delta, Ordering::Release);
    }

    /// Returns the effective maximum vector limit (`0` = unlimited).
    pub fn max_vectors(&self) -> ProResult<u64> {
        let guard = self
            .active_license
            .read()
            .map_err(|_| ProEngineError::Io(IoError::other("license RwLock poisoned")))?;
        Ok(guard
            .as_ref()
            .map_or(COMMUNITY_MAX_VECTORS, |lic| lic.max_vectors))
    }

    /// Returns `true` if the vector cap has been reached (insertions blocked, reads allowed).
    #[must_use]
    pub fn is_limit_reached(&self) -> bool {
        match self.max_vectors() {
            Ok(0) | Err(_) => false,
            Ok(limit) => self.total_vector_count() >= limit,
        }
    }

    /// Checks whether `additional` new vectors can be ingested under the active license.
    pub fn check_vector_limit(&self, additional: u64) -> ProResult<()> {
        let max_vectors = self.max_vectors()?;
        if max_vectors == 0 {
            return Ok(());
        }
        let current = self.total_vector_count();
        if additional > 0 && current.saturating_add(additional) > max_vectors {
            return Err(ProEngineError::CommunityTierLimitReached { limit: max_vectors });
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::time::Instant;

    use ed25519_dalek::{Signer as _, SigningKey};
    use osprey_core::LicenseVerifier;
    use rand::rngs::OsRng;

    use super::*;

    fn sign_test_token(signing_key: &SigningKey, license: &OspreyLicense) -> String {
        let payload_json = serde_json::to_vec(license).unwrap();
        let payload_b64 = URL_SAFE_NO_PAD.encode(payload_json);
        let signed_message = format!("{LICENSE_TOKEN_PREFIX}.{payload_b64}");
        let signature = signing_key.sign(signed_message.as_bytes());
        let sig_b64 = URL_SAFE_NO_PAD.encode(signature.to_bytes());
        format!("{signed_message}.{sig_b64}")
    }

    fn sample_license(issued_at: u64, expires_at: u64) -> OspreyLicense {
        OspreyLicense {
            customer_org: "acme-corp-datalake".to_owned(),
            tier: "enterprise".to_owned(),
            max_nodes: 16,
            max_vectors: 0,
            issued_at,
            expires_at,
            features: vec![
                "wal_streaming".to_owned(),
                "cow_compactor".to_owned(),
                "s3_cas_manifest".to_owned(),
                "arrow_flight".to_owned(),
            ],
        }
    }

    #[test]
    fn embedded_public_key_loads_cleanly() {
        let verifier = OfflineLicenseVerifier::new().unwrap();
        assert_eq!(&verifier.public_key_bytes(), EMBEDDED_PUBLIC_KEY);
        assert!(!verifier.is_enabled(ProCapability::WalStreaming));
    }

    #[test]
    fn offline_sign_and_verify_round_trip() {
        let signing_key = SigningKey::generate(&mut OsRng);
        let pk = signing_key.verifying_key().to_bytes();
        let verifier = OfflineLicenseVerifier::from_public_key_bytes(&pk).unwrap();

        let now = 1_794_856_000_u64;
        let license = sample_license(now - 60, now + 86_400 * 365);
        let token = sign_test_token(&signing_key, &license);

        let verified = verifier.verify(&token, now).unwrap();
        assert_eq!(verified, license);

        let claims = verifier.verify_token(&token, now).unwrap();
        assert_eq!(claims.org_name, "acme-corp-datalake");
        assert_eq!(claims.plan_tier, "enterprise");
        assert_eq!(claims.max_nodes, 16);
        assert!(verifier.is_enabled(ProCapability::WalStreaming));
        assert!(verifier.is_enabled(ProCapability::CowCompactor));
        assert!(verifier.is_enabled(ProCapability::S3CasManifest));
        assert!(verifier.is_enabled(ProCapability::ArrowFlight));
        assert!(!verifier.is_enabled(ProCapability::QdrantCompat));
    }

    #[test]
    fn rejects_tampered_payload_or_signature() {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifier =
            OfflineLicenseVerifier::from_public_key_bytes(&signing_key.verifying_key().to_bytes())
                .unwrap();

        let now = 1_794_856_000_u64;
        let token = sign_test_token(&signing_key, &sample_license(now, now + 3600));

        // Tamper one character in payload
        let parts: Vec<&str> = token.split('.').collect();
        assert_eq!(parts.len(), 3);
        let mut tampered_payload = parts[1].as_bytes().to_vec();
        tampered_payload[0] = if tampered_payload[0] == b'A' {
            b'B'
        } else {
            b'A'
        };
        let tampered = format!(
            "{}.{}.{}",
            parts[0],
            std::str::from_utf8(&tampered_payload).unwrap(),
            parts[2]
        );
        assert!(matches!(
            verifier.verify(&tampered, now),
            Err(ProLicenseError::InvalidSignature)
        ));

        // Tamper signature with another key
        let wrong_key = SigningKey::generate(&mut OsRng);
        let wrong_token = sign_test_token(&wrong_key, &sample_license(now, now + 3600));
        assert!(matches!(
            verifier.verify(&wrong_token, now),
            Err(ProLicenseError::InvalidSignature)
        ));
    }

    #[test]
    fn rejects_expired_and_future_tokens() {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifier =
            OfflineLicenseVerifier::from_public_key_bytes(&signing_key.verifying_key().to_bytes())
                .unwrap();

        let issued_at = 1_794_856_000_u64;
        let expires_at = issued_at + 3600;
        let token = sign_test_token(&signing_key, &sample_license(issued_at, expires_at));

        // Within 30-day grace period succeeds with grace metadata flagged
        let grace_lic = verifier.verify(&token, expires_at).unwrap();
        assert!(grace_lic.is_in_grace_period_at(expires_at));
        assert_eq!(grace_lic.grace_days_remaining_at(expires_at), 30);
        assert!(verifier
            .verify(&token, expires_at + LICENSE_GRACE_PERIOD_SECS)
            .is_ok());

        // Strictly past 30-day grace period fails with LicenseExpired
        assert!(matches!(
            verifier.verify(&token, expires_at + LICENSE_GRACE_PERIOD_SECS + 1),
            Err(ProLicenseError::LicenseExpired { .. })
        ));
        assert!(matches!(
            verifier.verify(&token, issued_at - MAX_CLOCK_SKEW_SECS - 1),
            Err(ProLicenseError::IssuedInFuture { .. })
        ));
        // Within skew tolerance succeeds
        assert!(verifier.verify(&token, issued_at - 60).is_ok());
    }

    #[test]
    fn enforces_feature_scope_and_node_limits() {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifier =
            OfflineLicenseVerifier::from_public_key_bytes(&signing_key.verifying_key().to_bytes())
                .unwrap();

        let now = 1_794_856_000_u64;
        let token = sign_test_token(&signing_key, &sample_license(now, now + 3600));

        assert!(verifier
            .verify_for_feature(&token, now, "wal_streaming")
            .is_ok());
        assert!(matches!(
            verifier.verify_for_feature(&token, now, "qdrant_compat"),
            Err(ProLicenseError::MissingFeature { .. })
        ));

        assert!(verifier
            .verify_for_cluster(&token, now, Some("cow_compactor"), 16)
            .is_ok());
        assert!(matches!(
            verifier.verify_for_cluster(&token, now, Some("cow_compactor"), 17),
            Err(ProLicenseError::NodeLimitExceeded {
                max_nodes: 16,
                active_nodes: 17
            })
        ));
    }

    #[test]
    fn accepts_adr003_field_aliases() {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifier =
            OfflineLicenseVerifier::from_public_key_bytes(&signing_key.verifying_key().to_bytes())
                .unwrap();

        let now = 1_794_856_000_u64;
        let adr_json = format!(
            r#"{{"org_name":"bank-ai","plan_tier":"enterprise","max_nodes":0,"issued_at":{},"expires_at":{},"feature_flags":["wal_streaming","rerank_http"]}}"#,
            now,
            now + 86400
        );
        let payload_b64 = URL_SAFE_NO_PAD.encode(adr_json.as_bytes());
        let signed_msg = format!("{LICENSE_TOKEN_PREFIX}.{payload_b64}");
        let sig_b64 = URL_SAFE_NO_PAD.encode(signing_key.sign(signed_msg.as_bytes()).to_bytes());
        let token = format!("{signed_msg}.{sig_b64}");

        let lic = verifier.verify(&token, now).unwrap();
        assert_eq!(lic.customer_org, "bank-ai");
        assert_eq!(lic.tier, "enterprise");
        assert_eq!(lic.max_nodes, 0);
        assert!(lic.allows_nodes(1024));
        assert!(lic.has_capability(ProCapability::RerankHttp));
    }

    #[test]
    fn verify_latency_under_50_microseconds_in_release() {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifier =
            OfflineLicenseVerifier::from_public_key_bytes(&signing_key.verifying_key().to_bytes())
                .unwrap();

        let now = 1_794_856_000_u64;
        let token = sign_test_token(&signing_key, &sample_license(now, now + 86_400));

        // Warmup
        for _ in 0..50 {
            let _ = verifier.verify(&token, now).unwrap();
        }

        let iters = 1_000_u32;
        let start = Instant::now();
        for _ in 0..iters {
            let _ =
                std::hint::black_box(verifier.verify(std::hint::black_box(&token), now).unwrap());
        }
        let elapsed = start.elapsed();
        let avg_us = elapsed.as_secs_f64() * 1_000_000.0 / f64::from(iters);
        eprintln!("offline Ed25519 verify avg latency: {avg_us:.2} us");

        // Enforce strict < 50 us SLA under optimized release builds.
        if !cfg!(debug_assertions) {
            assert!(
                avg_us < 50.0,
                "expected < 50 us verification latency, observed {avg_us:.2} us"
            );
        }
    }
}
