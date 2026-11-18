# OspreyDB (`ospreydb/ospreydb`)

Production vector database server and commercial offline Ed25519 licensing engine for **OspreyDB**, consuming `osprey-core`, `osprey-simd`, and `osprey-storage` from `ospreydb/osprey-engine` without modifying the Apache-2.0 open-core repository.

## Capabilities

- **Real-Time Ingestion & Durability:** Crash-safe `Write-Ahead Log (WAL)` (`fsync` + torn-tail recovery) feeding an in-memory `LSM Delta Buffer` with immediate read-your-own-writes visibility and zero-invisibility-window CoW compaction.
- **Autonomous Maintenance:** Background Copy-on-Write (`CoW`) slab compactor and 64-bit `RoaringTreemap` tombstone garbage collector writing `.osi`/`.osv` segments to local storage or S3-compatible object stores via atomic `manifest.json` CAS (`If-Match` / `ETag`).
- **Drop-In Production Servers:**
  - **HTTP API (`:6333`):** Qdrant-compatible collection & point endpoints, Cohere/Jina-compatible `/v1/rerank` (supporting both Unicode text projection and explicit ColBERT `query_tokens`/`document_tokens`), OpenAI-compatible `/v1/embeddings`, `/v1/license/activate`, `/metrics`, and `/healthz` (64 MiB max request body + optional `--api-key` auth).
  - **Apache Arrow Flight gRPC (`:8815`):** Zero-copy `DoPut` (accepting `FixedSizeList<Float32/Float64>` and `List<Float32/Float64>`), `DoExchange` (vector search & `LUT-Rank` rerank), `DoGet`, and `ListFlights`.

## Zero-Config Quickstart (Local S3 Filesystem)

`--data-dir` and `--s3-uri file://...` automatically create missing directories on startup:

```bash
# Start single-node Community Tier server with local WAL + local S3 segment store
osprey serve \
  --dim 128 \
  --data-dir ./data \
  --s3-uri file:///tmp/osprey-s3
```

Optional Pro license activation and distributed multi-node mode:

```bash
# Persist a Pro/Enterprise license key to ~/.osprey/license.key (mode 0600)
osprey activate --license "OSPREY-PRO-v1...."

# Dedicated writer + compactor node (Pro/Enterprise)
osprey serve --role writer --dim 128 --data-dir ./writer-data --s3-uri s3://your-bucket/osprey

# Stateless reader node polling manifest.json every 5s (Pro/Enterprise)
osprey serve --role reader --dim 128 --s3-uri s3://your-bucket/osprey --manifest-poll-secs 5
```

## Licensing & Community Tier Rules

- **Community / Personal Free Tier (No Key Required, `--role unified`):**
  - All single-node features unlocked up to **5,000,000 vectors**: WAL, Delta Buffer, CoW Compactor, Tombstone GC, HTTP API, `/v1/rerank`, `/v1/embeddings`, and Arrow Flight `:8815`.
  - Reaching 5,000,000 vectors keeps the node online in **read/search-only mode** (`POST /collections/{name}/points/search`, `/v1/rerank`, `/metrics`, `/healthz`, Flight `DoExchange`/`DoGet` stay active; new inserts are rejected with HTTP `402` / gRPC `RESOURCE_EXHAUSTED`).
  - Distributed multi-node roles (`--role writer` / `--role reader`) require a Pro or Enterprise license.
- **Pro / Enterprise Annual Subscription:**
  - Pure annual subscription (`OSPREY-PRO-v1.<payload>.<sig>`, 100% offline Ed25519 verification, zero phone-home) with a **30-day renewal grace period** past `expires_at` (warnings logged during grace; Pro startup blocked after `expires_at + 30d`).

## CLI Flags & Environment Variables (`osprey serve`)

| Flag | Env Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `--license` | `OSPREY_PRO_LICENSE` | `~/.osprey/license.key` | Offline `OSPREY-PRO-v1` license token |
| `--role` | `OSPREY_ROLE` | `unified` | Node topology role: `unified`, `writer`, or `reader` |
| `--collection` | — | `default` | Default collection name |
| `--dim` | — | `128` | Embedding vector dimensionality |
| `--fwht-seed` | — | `0x4F5350520001` | Deterministic FWHT Rademacher seed |
| `--data-dir` | `OSPREY_DATA_DIR` | — | Data directory (auto-created; default parent for `osprey.wal`) |
| `--wal-path` | `OSPREY_WAL_PATH` | `./osprey.wal` (or `<data-dir>/osprey.wal`) | Write-Ahead Log file path |
| `--object-store-uri` (`--s3-uri`, `--s3-manifest`) | `OSPREY_OBJECT_STORE_URI` | In-memory | Segment & `manifest.json` URI (`file://...`, `s3://...`, `gs://...`, `az://...`) |
| `--manifest-poll-secs` | `OSPREY_MANIFEST_POLL_SECS` | `5` | Background `manifest.json` ETag poll interval in seconds (`0` disables) |
| `--http-addr` | — | `0.0.0.0:6333` | HTTP API listen address |
| `--flight-addr` | — | `0.0.0.0:8815` | Apache Arrow Flight gRPC listen address |
| `--api-key` | `OSPREY_API_KEY` | — | Optional HTTP auth key (`api-key` or `Authorization: Bearer`; `/healthz` exempt) |
| `--delta-flush-threshold` | — | `50000` | Live delta vector count before triggering CoW compaction |
| `--embed-url` (`--embeddings-url`) | `OSPREY_EMBED_URL` | — | OpenAI-compatible base URL for `/v1/embeddings` proxy |
| `--embed-model` | `OSPREY_EMBED_MODEL` | `text-embedding-3-small` | Model name sent to `/v1/embeddings` |
| `--embed-api-key` | `OSPREY_EMBED_API_KEY` | — | Bearer token for upstream `/v1/embeddings` provider |
| `--embed-batch-size` | — | `32` | Max texts per upstream embedding request |

## Supported Endpoints

### HTTP (`:6333`)

- `GET /healthz` — Liveness probe (always open, exempt from `--api-key`)
- `GET /metrics` — Prometheus exposition (`0.0.4`) with P50/P95/P99 latency histograms & S3 call savings
- `GET /collections` — List collections (Qdrant compatible)
- `PUT /collections/{name}` — Ensure/validate collection (Qdrant compatible)
- `GET /collections/{name}` — Collection status, vector counts, license & embedder metadata
- `PUT /collections/{name}/points` — Batch point upsert (`vector` or `text`, up to 64 MiB payload)
- `POST /collections/{name}/points/search` — Hybrid vector or `query_text` search
- `POST /collections/{name}/points/delete` — Logical point deletion (`points: [id, ...]`)
- `POST /v1/rerank` — Cohere/Jina-compatible `LUT-Rank` reranking (`query`/`documents` or explicit `query_tokens`/`document_tokens`)
- `POST /v1/embeddings` — OpenAI-compatible embeddings proxy
- `POST /v1/license/activate` — Hot-activate an `OSPREY-PRO-v1` license key at runtime

### Apache Arrow Flight gRPC (`:8815`)

- `DoPut` — Zero-copy batch ingestion (`doc_id: UInt64`, `vector: FixedSizeList<Float32/Float64> | List<Float32/Float64>`, optional `payload_json`)
- `DoExchange` — Bidirectional hybrid vector search (`vector`) or `LUT-Rank` rerank (`query`, `document`)
- `ListFlights` / `GetFlightInfo` / `DoGet` — Cluster segment metadata streaming

## Self-Hosted CI/CD & Multi-Arch Releases

- Runner stack: [`infra/ci/docker-compose.runner.yml`](./infra/ci/docker-compose.runner.yml) + [`infra/ci/setup-runner.sh`](./infra/ci/setup-runner.sh)
- Workflow: [`.github/workflows/release.yml`](./.github/workflows/release.yml) (runs PR checks on `main`, builds Linux `x86_64`/`aarch64` `musl` binaries + Distroless `ghcr.io/ospreydb/osprey` image on `v*` and `pro-v*` tags, and uploads public `osprey` tarballs to `ospreydb/osprey-engine` releases).

See [`EULA.md`](./EULA.md) and [`SUPPORT.md`](./SUPPORT.md) for commercial terms and support SLAs.
