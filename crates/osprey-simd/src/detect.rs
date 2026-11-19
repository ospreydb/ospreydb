//! Runtime CPU SIMD capability detection with zero-overhead `OnceLock` caching.

use std::sync::OnceLock;

/// Supported SIMD execution tiers in descending order of vector width.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SimdLevel {
    /// x86_64 AVX-512F + AVX-512BW + AVX-512VPOPCNTDQ (512-bit ZMM registers).
    Avx512,
    /// x86 / x86_64 AVX2 + FMA + POPCNT (256-bit YMM registers).
    Avx2,
    /// ARM64 / AArch64 NEON (128-bit Q registers + `vtbl`).
    Neon,
    /// Portable scalar fallback (also forced under Miri).
    Scalar,
}

impl SimdLevel {
    /// Returns the canonical lowercase backend identifier string for this SIMD tier.
    #[inline]
    pub const fn as_backend_str(self) -> &'static str {
        match self {
            Self::Avx512 => "avx512_vpopcntdq",
            Self::Avx2 => "avx2",
            Self::Neon => "neon",
            Self::Scalar => "scalar",
        }
    }
}

static DETECTED_SIMD_LEVEL: OnceLock<SimdLevel> = OnceLock::new();

/// Detects and caches the highest supported SIMD tier on the host CPU.
#[inline]
pub fn detect_best_simd_level() -> SimdLevel {
    *DETECTED_SIMD_LEVEL.get_or_init(detect_uncached)
}

/// Returns the active SIMD backend identifier detected on the host CPU:
/// `"avx512_vpopcntdq"`, `"avx2"`, `"neon"`, or `"scalar"`.
#[inline]
pub fn active_simd_backend() -> &'static str {
    detect_best_simd_level().as_backend_str()
}

/// Returns `true` if `level` can execute safely without `SIGILL` on the current host.
#[inline]
pub fn is_level_supported(level: SimdLevel) -> bool {
    let best = detect_best_simd_level();
    match level {
        SimdLevel::Scalar => true,
        SimdLevel::Avx2 => matches!(best, SimdLevel::Avx2 | SimdLevel::Avx512),
        SimdLevel::Avx512 => matches!(best, SimdLevel::Avx512),
        SimdLevel::Neon => matches!(best, SimdLevel::Neon),
    }
}

fn detect_uncached() -> SimdLevel {
    #[cfg(miri)]
    {
        SimdLevel::Scalar
    }

    #[cfg(all(not(miri), any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vpopcntdq")
        {
            return SimdLevel::Avx512;
        }
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("popcnt")
        {
            return SimdLevel::Avx2;
        }
        SimdLevel::Scalar
    }

    #[cfg(all(not(miri), target_arch = "aarch64"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return SimdLevel::Neon;
        }
        SimdLevel::Scalar
    }

    #[cfg(all(
        not(miri),
        not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))
    ))]
    {
        SimdLevel::Scalar
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn detect_caches_deterministically_and_scalar_always_supported() {
        let first = detect_best_simd_level();
        let second = detect_best_simd_level();
        assert_eq!(first, second);
        assert!(is_level_supported(SimdLevel::Scalar));
        assert!(is_level_supported(first));
        assert_eq!(active_simd_backend(), first.as_backend_str());
        assert!(matches!(
            active_simd_backend(),
            "avx512_vpopcntdq" | "avx2" | "neon" | "scalar"
        ));
    }
}
