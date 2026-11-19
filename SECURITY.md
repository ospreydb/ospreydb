# Security Policy

## Reporting a Vulnerability

Do **not** open a public GitHub issue or pull request for security vulnerabilities.

Send a private report to **security@ospreydb.dev** with:

- A description of the vulnerability and affected crate/command (`osprey-core`, `osprey-simd`, `osprey-storage`, `osprey-cli`, or `ospreydb`).
- Steps or a minimal input file (`.osi`, `.osv`, `.osr`, `.parquet`, `.jsonl`, or `.npy`) that reproduces the issue.
- Output of `osprey --version` and `osprey doctor`.

We acknowledge reports within **48 hours**, triage severity and reproduction within **5 business days**, and coordinate a fix before public disclosure.

## Scope

Security-sensitive areas in `ospreydb/ospreydb` include:

- Out-of-bounds memory reads/writes or unaligned `unsafe` SIMD loads (`_mm512_load_si512`, `memmap2` slices, `bytemuck` casts) triggered by malformed `.osi`, `.osv`, or `.osr` files.
- `xxHash64` checksum validation bypasses (`header_xxh64`, `file_xxh64`, `page_xxh64`) or Zstd decompression bombs on untrusted `.osv` pages.
- Integer overflows during offset/buffer size calculations in `osprey-core` or `osprey-storage`.
- Credential handling in `object_store` S3/GCS/Azure/HTTP URIs.

## Supported Versions

Security patches are released on the latest `0.1.x` tag and `main` branch.
