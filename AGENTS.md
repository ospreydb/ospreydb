# AGENTS.md — Diretrizes de Engenharia do `ospreydb`

Este repositório contém o servidor de banco vetorial e camada comercial (**`ospreydb/ospreydb`**), consumindo `osprey-core`, `osprey-simd` e `osprey-storage` de `ospreydb/osprey-engine` exclusivamente via dependências externas no [`Cargo.toml`](./Cargo.toml).

---

## 1. Fronteira Estrita Open-Core vs. Pro
- Nunca adicione código comercial (`WAL`, `LsmDeltaBuffer`, `CowSlabCompactor`, `manifest.json` CAS, servidores HTTP/Arrow Flight ou chaves/verificadores `Ed25519`) no repositório aberto `ospreydb/osprey-engine`.
- Nunca adicione caminhos locais de máquina (ex.: `/home/...`) no [`Cargo.toml`](./Cargo.toml); mantenha dependências remotas reproduzíveis e travadas no [`Cargo.lock`](./Cargo.lock).

## 2. Regras de Licenciamento Offline (`Ed25519`)
- **Zero Phone-Home:** Toda verificação de licença é 100% offline sobre a chave pública embutida (`crates/ospreydb/src/osprey_pro_ed25519.pub`). Nunca versione chaves privadas (`*.pem`, `*.key`, `*.sk`) nem arquivos `.env`.
- **Modelo de Assinatura Anual + Grace Period de 30 dias:**
  - Não existe licença perpétua (`perpetual_fallback`).
  - Se `now > expires_at` e `now <= expires_at + 30 dias`, o daemon deve iniciar emitindo `tracing::warn!` claro indicando o período de tolerância.
  - Se `now > expires_at + 30 dias`, a verificação falha com `ProLicenseError::LicenseExpired` bloqueando o modo Pro.
- **Plano Free (`Community Tier`, 5M vetores):**
  - Ao atingir 5.000.000 de vetores, o daemon **não pode parar ou recusar leituras**: buscas HTTP, reranking, `/metrics`, `/healthz` e Arrow Flight `DoExchange`/`DoGet` permanecem ativos; apenas novas inserções (`PUT /collections/{name}/points` e Arrow Flight `DoPut`) são bloqueadas.

## 3. Invariantes de Código e Portões de Qualidade
- **Zero `unwrap()` / `expect()` em Produção:** Estritamente proibido fora de `#[cfg(test)]` (imposto pelos workspace lints no [`Cargo.toml`](./Cargo.toml)).
- **Toolchain & Formatação:** Respeite [`rust-toolchain.toml`](./rust-toolchain.toml), [`rustfmt.toml`](./rustfmt.toml) e [`.editorconfig`](./.editorconfig).
- **Portões obrigatórios antes de commit:**
  - `cargo fmt --all -- --check`
  - `cargo check --workspace --all-targets --locked`
  - `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings`
  - `cargo test --workspace --release --locked`
- **Higiene de Git e CI:**
  - Nunca versione `target/`, `infra/ci/.env`, dumps de memória, `*.parquet`, `*.bin`, `*.osi`, `*.osv`, `*.osr`, `*.pem` ou `*.key`.
  - O workflow [`.github/workflows/release.yml`](./.github/workflows/release.yml) roda exclusivamente no runner self-hosted Linux (`[self-hosted, osprey-pro-builder]`) para PRs em `main` e tags `v*` / `pro-v*`.
