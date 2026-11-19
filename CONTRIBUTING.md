# Contributing to OspreyDB

Thank you for contributing to **OspreyDB (`ospreydb/ospreydb`)**, licensed under **Apache-2.0**. Before writing code, read [`AGENTS.md`](./AGENTS.md) and [`ARCHITECTURE.md`](./ARCHITECTURE.md) — both documents define the binary layout invariants, SIMD rules, and crate topology.

## Workspace Architecture

Dependencies between crates are strictly acyclic:

```text
ospreydb  ──▶  osprey-cli  ──▶  osprey-storage  ──▶  osprey-simd  ──▶  osprey-core
```

- `crates/osprey-core`: 64-byte aligned binary headers/layouts (`.osi`, `.osv`, `.osr`), FWHT, `1-Bit Binary K-Means`, asymmetric `RaBitQ`, DP range planner primitives, `LUT-Rank` math, `xxHash64`, and extension traits (`DeltaBufferExt` / `StorageEngineExt`). No network or filesystem I/O.
- `crates/osprey-simd`: Runtime-dispatched SIMD kernels (`AVX-512 VPOPCNTDQ`/`BW`, `AVX2`, `ARM64 NEON`) with a bit-exact scalar fallback.
- `crates/osprey-storage`: `memmap2` zero-copy readers, TSP slab packing, DP range coalescing, Zstd minipage codecs, `RoaringBitmap` tombstones, and async `object_store` S3 Range-GET transport.
- `crates/osprey-cli`: CLI subcommand library (`doctor`, `import`, `search`, `rerank`, `delete`, `inspect`, `bench`, `completions`).
- `crates/ospreydb`: Unified `osprey` binary and daemon (`serve` HTTP `:6333`, Arrow Flight `:8815`, WAL, `LsmDeltaBuffer`, `CowSlabCompactor`, `TombstoneGarbageCollector`, S3 `manifest.json` CAS).

## Development Setup & Running Tests

Requirements: Rust toolchain pinned in [`rust-toolchain.toml`](./rust-toolchain.toml) (`1.99.0` with `clippy` and `rustfmt`).

```bash
# Format check
cargo fmt --all -- --check

# Strict lint gate (denies unwrap/expect/panic in production crates)
cargo clippy --workspace --all-targets --all-features -- -D warnings

# Full release test suite (default target flags)
cargo test --workspace --release

# Native-CPU SIMD test run
RUSTFLAGS="-C target-cpu=native" cargo test --workspace --release
```

If you change `unsafe` pointer/alignment code or `bytemuck` casts, also run Miri on the core/SIMD/storage suites:

```bash
cargo +nightly miri test -p osprey-core -p osprey-simd -p osprey-storage --lib
cargo +nightly miri test -p osprey-core --test binary_contracts --test lutrank_golden --test rabitq_golden
cargo +nightly miri test -p osprey-simd --test fwht_golden_validation
```

## Engineering Invariants

1. **64-byte alignment:** Headers (`OsiHeader`, `OsvHeader`, `OsrHeader`) are 64 bytes (`#[repr(C, align(64))]`) with explicit padding fields and `bytemuck::Pod` / `bytemuck::Zeroable`. Tabular sections start at `offset % 64 == 0`.
2. **Error handling:** Never call `.unwrap()`, `.expect()`, or `panic!()` outside `#[cfg(test)]`.
3. **SIMD hot loops:** Zero heap allocations (`Vec`, `Box`, `String`, `format!`) inside `osprey-simd` kernels.
4. **Checksum integrity:** Verify `header_xxh64` before trusting offsets or allocating buffers, and `page_xxh64` before Zstd decompression.

## DCO & Licensing Policy

All contributions to `ospreydb/ospreydb` are licensed under **Apache-2.0**. We use the Developer Certificate of Origin (DCO) rather than a separate heavyweight CLA:

1. Sign off every commit with `git commit -s` (adds `Signed-off-by: Your Name <you@example.com>`).
2. By adding that line, you certify that you wrote the patch or have the right to submit it under the Apache-2.0 license.
3. Commits SHOULD be cryptographically signed (SSH/GPG) when possible.
