#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::Instant;

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use ed25519_dalek::{Signer as _, SigningKey};
use osprey_core::{LicenseVerifier, ProCapability};
use osprey_pro::{
    OfflineLicenseVerifier, OspreyLicense, ProLicenseError, LICENSE_TOKEN_PREFIX,
    MAX_CLOCK_SKEW_SECS, MAX_LICENSE_TOKEN_BYTES,
};
use rand::rngs::OsRng;

fn issue_test_token(key: &SigningKey, license: &OspreyLicense) -> String {
    let payload_json = serde_json::to_vec(license).unwrap();
    let payload_b64 = URL_SAFE_NO_PAD.encode(payload_json);
    let signed = format!("{LICENSE_TOKEN_PREFIX}.{payload_b64}");
    let sig_b64 = URL_SAFE_NO_PAD.encode(key.sign(signed.as_bytes()).to_bytes());
    format!("{signed}.{sig_b64}")
}

#[test]
fn rejects_malformed_tokens_before_signature_check() {
    let verifier = OfflineLicenseVerifier::new().unwrap();
    let now = 1_794_856_000_u64;

    assert!(matches!(
        verifier.verify("not-a-token", now),
        Err(ProLicenseError::InvalidFormat)
    ));
    assert!(matches!(
        verifier.verify("OSPREY-PRO-v2.abc.def", now),
        Err(ProLicenseError::UnsupportedVersion { .. })
    ));
    assert!(matches!(
        verifier.verify("OSPREY-PRO-v1..def", now),
        Err(ProLicenseError::InvalidFormat)
    ));
    assert!(matches!(
        verifier.verify("OSPREY-PRO-v1.abc.", now),
        Err(ProLicenseError::InvalidFormat)
    ));
    assert!(matches!(
        verifier.verify("OSPREY-PRO-v1.abc.def.ghi", now),
        Err(ProLicenseError::InvalidFormat)
    ));

    let oversized = format!("OSPREY-PRO-v1.{}.abc", "A".repeat(MAX_LICENSE_TOKEN_BYTES));
    assert!(matches!(
        verifier.verify(&oversized, now),
        Err(ProLicenseError::TokenTooLarge { .. })
    ));
}

#[test]
fn rejects_wrong_signature_length_and_invalid_claims() {
    let key = SigningKey::generate(&mut OsRng);
    let verifier =
        OfflineLicenseVerifier::from_public_key_bytes(&key.verifying_key().to_bytes()).unwrap();
    let now = 1_794_856_000_u64;

    // Valid format, short signature
    let short_sig = URL_SAFE_NO_PAD.encode([1_u8; 32]);
    let payload = URL_SAFE_NO_PAD.encode(b"{}");
    let bad_len = format!("{LICENSE_TOKEN_PREFIX}.{payload}.{short_sig}");
    assert!(matches!(
        verifier.verify(&bad_len, now),
        Err(ProLicenseError::InvalidSignatureLength {
            expected: 64,
            actual: 32
        })
    ));

    // Empty org
    let empty_org = OspreyLicense {
        customer_org: "   ".to_owned(),
        tier: "pro".to_owned(),
        max_nodes: 4,
        issued_at: now - 10,
        expires_at: now + 3600,
        features: vec!["wal_streaming".to_owned()],
    };
    let token_empty_org = issue_test_token(&key, &empty_org);
    assert!(matches!(
        verifier.verify(&token_empty_org, now),
        Err(ProLicenseError::InvalidClaim { .. })
    ));

    // Invalid validity window (expires_at <= issued_at)
    let bad_window = OspreyLicense {
        customer_org: "acme".to_owned(),
        tier: "pro".to_owned(),
        max_nodes: 4,
        issued_at: now,
        expires_at: now,
        features: vec![],
    };
    let token_bad_window = issue_test_token(&key, &bad_window);
    assert!(matches!(
        verifier.verify(&token_bad_window, now - 1),
        Err(ProLicenseError::InvalidClaim { .. })
    ));
}

#[test]
fn license_verifier_trait_and_all_capabilities_work() {
    let key = SigningKey::generate(&mut OsRng);
    let verifier =
        OfflineLicenseVerifier::from_public_key_bytes(&key.verifying_key().to_bytes()).unwrap();
    let now = 1_794_856_000_u64;

    let license = OspreyLicense {
        customer_org: "global-quant".to_owned(),
        tier: "enterprise".to_owned(),
        max_nodes: 32,
        issued_at: now - 100,
        expires_at: now + 86_400 * 30,
        features: vec![
            "wal_streaming".to_owned(),
            "cow_compactor".to_owned(),
            "tombstone_gc".to_owned(),
            "s3_cas_manifest".to_owned(),
            "arrow_flight".to_owned(),
            "rerank_http".to_owned(),
            "qdrant_compat".to_owned(),
        ],
    };
    let token = issue_test_token(&key, &license);

    // Invoke via open-core LicenseVerifier trait object
    let trait_ref: &dyn LicenseVerifier = &verifier;
    let claims = trait_ref.verify_token(&token, now).unwrap();
    assert_eq!(claims.org_name, "global-quant");
    assert_eq!(claims.plan_tier, "enterprise");
    assert_eq!(claims.capabilities.len(), 7);

    for cap in [
        ProCapability::WalStreaming,
        ProCapability::CowCompactor,
        ProCapability::TombstoneGc,
        ProCapability::S3CasManifest,
        ProCapability::ArrowFlight,
        ProCapability::RerankHttp,
        ProCapability::QdrantCompat,
    ] {
        assert!(trait_ref.is_enabled(cap));
    }

    verifier.deactivate();
    assert!(!trait_ref.is_enabled(ProCapability::WalStreaming));
}

#[test]
fn boundary_timestamps_and_skew_tolerance() {
    let key = SigningKey::generate(&mut OsRng);
    let verifier =
        OfflineLicenseVerifier::from_public_key_bytes(&key.verifying_key().to_bytes()).unwrap();

    let issued_at = 1_794_856_000_u64;
    let expires_at = issued_at + 1000;
    let license = OspreyLicense {
        customer_org: "edge-ai".to_owned(),
        tier: "pro".to_owned(),
        max_nodes: 2,
        issued_at,
        expires_at,
        features: vec!["wal_streaming".to_owned()],
    };
    let token = issue_test_token(&key, &license);

    // Exact boundary: 1 second before expiry passes; at expiry fails.
    assert!(verifier.verify(&token, expires_at - 1).is_ok());
    assert!(matches!(
        verifier.verify(&token, expires_at),
        Err(ProLicenseError::LicenseExpired { .. })
    ));

    // Exact clock-skew boundary
    assert!(verifier
        .verify(&token, issued_at - MAX_CLOCK_SKEW_SECS)
        .is_ok());
    assert!(matches!(
        verifier.verify(&token, issued_at - MAX_CLOCK_SKEW_SECS - 1),
        Err(ProLicenseError::IssuedInFuture { .. })
    ));
}

#[test]
fn release_offline_verification_meets_50_microsecond_sla() {
    let key = SigningKey::generate(&mut OsRng);
    let verifier =
        OfflineLicenseVerifier::from_public_key_bytes(&key.verifying_key().to_bytes()).unwrap();

    let now = 1_794_856_000_u64;
    let license = OspreyLicense {
        customer_org: "perf-corp".to_owned(),
        tier: "enterprise".to_owned(),
        max_nodes: 64,
        issued_at: now - 60,
        expires_at: now + 86_400 * 365,
        features: vec![
            "wal_streaming".to_owned(),
            "cow_compactor".to_owned(),
            "s3_cas_manifest".to_owned(),
        ],
    };
    let token = issue_test_token(&key, &license);

    for _ in 0..100 {
        let _ = verifier.verify(&token, now).unwrap();
    }

    let iters = 2_000_u32;
    let start = Instant::now();
    for _ in 0..iters {
        let lic = std::hint::black_box(verifier.verify(std::hint::black_box(&token), now).unwrap());
        assert!(lic.has_feature("wal_streaming"));
    }
    let avg_us = start.elapsed().as_secs_f64() * 1_000_000.0 / f64::from(iters);
    eprintln!("integration test avg verify latency: {avg_us:.2} us");

    if !cfg!(debug_assertions) {
        assert!(
            avg_us < 50.0,
            "verification exceeded 50 us budget: {avg_us:.2} us"
        );
    }
}
