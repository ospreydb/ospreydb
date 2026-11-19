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
pub use detect::{active_simd_backend, detect_best_simd_level, is_level_supported, SimdLevel};
pub use osprey_core::{OspreyError, OspreyResult};
