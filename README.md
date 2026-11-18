# OspreyDB Pro (`ospreydb/osprey-pro`)

Commercial production daemon and offline Ed25519 licensing engine for **OspreyDB**, consuming `osprey-core`, `osprey-simd`, and `osprey-storage` from `ospreydb/ospreydb` without modifying the Apache-2.0 open-core repository.

## Capabilities

- **Real-Time Ingestion & Durability:** Crash-safe `Write-Ahead Log (WAL)` (`fsync` + torn-tail recovery) feeding an in-memory `LSM Delta Buffer` with immediate read-your-own-writes visibility.
- **Autonomous Maintenance:** Background Copy-on-Write (`CoW`) slab compactor and tombstone garbage collector writing `.osi`/`.osv` segments to local storage or S3-compatible object stores via atomic `manifest.json` CAS (`If-Match` / `ETag`).
- **Drop-In Production Servers:**
  - HTTP API (`:6333`): Qdrant-compatible search/upsert (`/collections/{name}/points/search`, `/collections/{name}/points`), Cohere-compatible reranking (`/v1/rerank` via `LUT-Rank`), OpenAI-compatible `/v1/embeddings`, `/v1/license/activate`, `/metrics`, and `/healthz`.
  - Apache Arrow Flight gRPC (`:8815`): Zero-copy `DoPut`, `DoExchange` (search & `LUT-Rank`), and `DoGet` / `ListFlights`.

## Licensing Model

- **Community / Personal Free Tier (No Key Required):** Full single-node WAL, Delta Buffer, HTTP API, and `/v1/embeddings` up to **5,000,000 vectors**. Reaching 5M vectors puts the node into read/search-only mode for vectors (existing data stays online and searchable; only new inserts are rejected with HTTP `402` / gRPC `RESOURCE_EXHAUSTED`).
- **Pro Annual Subscription:** Pure annual subscription (`OSPREY-PRO-v1.<payload>.<sig>`, 100% offline Ed25519 verification, zero phone-home) with all updates included and a **30-day renewal grace period** past `expires_at` (warnings logged during grace; Pro startup blocked after `expires_at + 30d`).

### Quickstart

```bash
# Optional: persist a Pro license key to ~/.osprey/license.key (mode 0600)
osprey-pro activate --license "OSPREY-PRO-v1...."

# Start the daemon (reads ~/.osprey/license.key, --license, or OSPREY_PRO_LICENSE)
osprey-pro serve \
  --dim 128 \
  --wal-path ./osprey-pro.wal \
  --s3-manifest s3://your-bucket/osprey/manifest.json
```

## Self-Hosted CI/CD & Multi-Arch Releases

- Runner stack: [`infra/ci/docker-compose.runner.yml`](./infra/ci/docker-compose.runner.yml) + [`infra/ci/setup-runner.sh`](./infra/ci/setup-runner.sh)
- Workflow: [`.github/workflows/pro-release.yml`](./.github/workflows/pro-release.yml) (runs PR checks on `main` and builds Linux `x86_64`/`aarch64` `musl` binaries + Distroless `ghcr.io/ospreydb/osprey-pro` image on `pro-v*` tags).

See [`EULA.md`](./EULA.md) and [`SUPPORT.md`](./SUPPORT.md) for commercial terms and support SLAs.
