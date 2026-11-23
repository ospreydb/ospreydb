# AGENTS.md — Diretrizes de Engenharia do `ospreydb`

Este repositório contém o projeto unificado **OspreyDB (`ospreydb/ospreydb`)**, **100% Open Source sob a licença Apache-2.0**, estruturado como um workspace Cargo de 5 crates (`osprey-core`, `osprey-simd`, `osprey-storage`, `osprey-cli`, `ospreydb`).

---

## 1. Topologia do Workspace (Apache-2.0)
- `crates/osprey-core`: formatos binários alinhados a 64 bytes (`.osi`, `.osv`, `.osr`), FWHT, `1-Bit Binary K-Means`, `RaBitQ`, `LUT-Rank`, `RangePlanner` e traits de extensão (`DeltaBufferExt` / `StorageEngineExt`). Sem I/O de rede ou sistema de arquivos.
- `crates/osprey-simd`: kernels SIMD com despacho em runtime (`AVX-512 VPOPCNTDQ`/`BW`, `AVX2`, `ARM64 NEON`, fallback escalar). Zero alocações de heap nos hot loops.
- `crates/osprey-storage`: leitores `memmap2`, empacotamento TSP, codec Zstd, `RoaringBitmap` tombstones, cliente `object_store` (S3/GCS/Azure/HTTP/local) e `RangePlanner` DP.
- `crates/osprey-cli`: biblioteca e CLI de engine (`import`, `search`, `rerank`, `delete`, `inspect`, `bench`, `doctor`, `completions`).
- `crates/ospreydb`: binário unificado `osprey` integrando daemon (`serve` HTTP `:6333`, Arrow Flight `:8815`, WAL, `LsmDeltaBuffer`, `CowSlabCompactor`, `TombstoneGarbageCollector`, `manifest.json` CAS) e todos os subcomandos CLI.
- Nunca adicione caminhos locais de máquina (ex.: `/home/...`) no [`Cargo.toml`](./Cargo.toml); mantenha dependências reproduzíveis e travadas no [`Cargo.lock`](./Cargo.lock).

## 2. Invariantes de Código e Portões de Qualidade
- **Zero `unwrap()` / `expect()` em Produção:** Estritamente proibido fora de `#[cfg(test)]` (imposto pelos workspace lints no [`Cargo.toml`](./Cargo.toml)).
- **Alinhamento de 64 bytes:** Headers (`.osi`, `.osv`, `.osr`) são `#[repr(C, align(64))]` com `bytemuck::Pod`/`Zeroable` sem padding implícito e seções tabulares em múltiplos de 64 bytes.
- **Toolchain & Formatação:** Respeite [`rust-toolchain.toml`](./rust-toolchain.toml), [`rustfmt.toml`](./rustfmt.toml) e [`.editorconfig`](./.editorconfig).
- **Portões obrigatórios antes de commit:**
  - `cargo fmt --all -- --check`
  - `cargo check --workspace --all-targets --locked`
  - `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings`
  - `cargo test --workspace --release --locked`
  - `cargo check --manifest-path fuzz/Cargo.toml`
  - `python3 reference/poc_reference.py`
- **Higiene de Git e CI:**
  - Nunca versione `target/`, `infra/ci/.env`, dumps de memória, `*.parquet`, `*.bin` (exceto `reference/golden/**`), `*.osi`, `*.osv`, `*.osr`, `*.pem` ou `*.key`.
