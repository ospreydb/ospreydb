//! Runtime-dispatched SIMD kernels (`AVX-512`, `AVX2`, `ARM64 NEON`, scalar fallback)
//! for FWHT, `1-Bit Binary K-Means`, asymmetric `RaBitQ`, `f16` refinement, and `LUT-Rank`.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx512;
pub mod backend;
pub mod detect;
#[cfg(target_arch = "aarch64")]
pub mod neon;

pub use backend::{ScalarBackend, SimdBackend, SimdEngine};
pub use detect::{
    active_simd_backend, detect_best_simd_level, is_f16c_supported, is_level_supported, SimdLevel,
};
pub use osprey_core::{OspreyError, OspreyResult};

/// Computes the exact `f32 x f16` inner product accumulated in `f32` using the
/// best runtime-detected SIMD backend (`AVX-512F+F16C+FMA`, `AVX2+F16C+FMA`, or scalar).
#[inline]
pub fn dot_f32_f16(a_f32: &[f32], b_f16: &[half::f16]) -> f32 {
    SimdEngine::new_auto().dot_f32_f16(a_f32, b_f16)
}
