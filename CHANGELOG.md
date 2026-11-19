# Changelog

All notable changes to **OspreyDB** (`ospreydb/ospreydb`: `osprey-core`, `osprey-simd`, `osprey-storage`, `osprey-cli`, `ospreydb`) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-11-18

### Added
- Unified all 5 crates (`osprey-core`, `osprey-simd`, `osprey-storage`, `osprey-cli`, `ospreydb`) into a single 100% Apache-2.0 workspace and `osprey` binary (`serve`, `import`, `search`, `rerank`, `delete`, `inspect`, `bench`, `doctor`, `completions`, `wal-stress`).
- Exposed `osprey_simd::active_simd_backend() -> &'static str` (`"avx512_vpopcntdq"`, `"avx2"`, `"neon"`, `"scalar"`) and surfaced the active SIMD backend in `osprey --version`, `search`, and `inspect`.
- Added `osprey completions <SHELL>` (`bash`, `zsh`, `fish`, `elvish`, `powershell`) powered by `clap_complete` for Homebrew and shell integration.
- Added compile-time little-endian target guard (`#[cfg(target_endian = "big")] compile_error!(...)`) in `osprey-core`.
- Added `CITATION.cff` and `NOTICE` alongside the canonical `Apache-2.0` `LICENSE`.
- Added self-contained 30-second quickstart, S3 environment variable reference table, and SIFT1M non-negative binary quantization explanation in `README.md`.
- Added architectural comparison with DiskANN/Vamana and IVF-PQ object-storage engines in `ARCHITECTURE.md`.

### Changed
- Unified workspace version to `0.1.1` under `Apache-2.0` (`osprey-core`, `osprey-simd`, `osprey-storage`, `osprey-cli`, `ospreydb`) with unified `osprey` binary naming.
- Updated workspace `documentation` URL to `https://ospreydb.dev/architecture` and inherited `readme` / `categories` in `[workspace.package]`.
- Reconciled `MinipageDirectoryEntry` (48 bytes) and `.osi` `RabitqEntryHeader` (24 bytes + $D/8$ binary code) documentation across `ARCHITECTURE.md`, `rfcs/`, and `adr/ADR-001`.
- Configured explicit `object_store::RetryConfig` and transient HTTP `429`/`500`/`502`/`503`/`504` retry classification in `osprey-storage`.
- Hardened `RangePlanner` boundary checks and added extreme-value overflow tests.

### Removed
- Removed all commercial paywalls, Ed25519 license verification, 5M vector cap, `--role writer`/`--role reader` restrictions, commercial license agreement, and keygen crate.
- Removed internal development artifact `BRAND_AND_PROFILES.md` and internal phase/task labels from fuzz targets and CI workflows.

## [0.1.0] - 2026-11-16

### Added
- Initial release of `osprey-core`, `osprey-simd`, `osprey-storage`, and `osprey-cli`:
  - 64-byte aligned zero-copy binary formats (`.osi`, `.osv`, `.osr`) with `xxHash64` header/page/footer verification.
  - FWHT orthogonal rotation, `1-Bit Binary K-Means`, asymmetric `RaBitQ` (`int4` query bit-planes), TSP slab locality ordering, and $O(k)$ DP `RangePlanner`.
  - Runtime-dispatched SIMD kernels (`AVX-512 VPOPCNTDQ`/`BW`, `AVX2`, `ARM64 NEON`, scalar fallback) including `LUT-Rank` nibble-packed `MaxSim` reranking.
  - CLI subcommands (`doctor`, `import`, `search`, `rerank`, `delete`, `inspect`, `bench`).
