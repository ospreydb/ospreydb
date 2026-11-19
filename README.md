# OspreyDB

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](./LICENSE)

**OspreyDB** (`ospreydb/ospreydb`) is a 100% Open Source (Apache-2.0) vector database and CPU late-interaction reranking engine built in Rust for local disk and S3-compatible object storage.

It separates vector search into three 64-byte aligned binary formats plus a crash-safe WAL/LSM production daemon:

- **`.osi`**: 1-bit FWHT-rotated `RaBitQ` resident index scanned via `mmap` and SIMD (`AVX-512 VPOPCNTDQ`, `AVX2`, `ARM64 NEON`).
- **`.osv`**: Zstd-compressed `f16` vector slabs ordered by TSP locality and fetched via `O(k)` Dynamic Programming coalesced S3 `Range-GET` requests (`RangePlanner`).
- **`.osr`**: Nibble-packed 2-bit nested-lattice ColBERT/ColPali token store reranked on CPU using `vpshufb`/`vtbl` lookup tables (`LUT-Rank`).
- **`osprey serve` Production Daemon**: Crash-safe `Write-Ahead Log (WAL)` + real-time `LSM Delta Buffer`, background Copy-on-Write (`CoW`) slab compactor, 64-bit `RoaringTreemap` tombstone GC, lock-free multi-node S3 `manifest.json` CAS (`If-Match` / `ETag`), Qdrant Drop-In REST API (`:6333`), OpenAI-compatible `/v1/embeddings`, Cohere-compatible `/v1/rerank`, Prometheus `/metrics`, and Apache Arrow Flight gRPC (`:8815`).

---

## Installation

```bash
curl -fsSL https://ospreydb.dev/install.sh | sh
```

Or via Homebrew, Docker (`ghcr.io/ospreydb/osprey`), or Cargo:

```bash
brew install ospreydb/tap/osprey
docker run --rm ghcr.io/ospreydb/osprey:latest --help
cargo install --path crates/ospreydb
```

---

## Quickstart

### 1. Server Mode (`osprey serve`)

`--data-dir` and `--s3-uri file://...` automatically create missing directories on startup:

```bash
# Start single-node unified server (HTTP :6333 + Arrow Flight :8815 + WAL + CoW Compactor)
osprey serve \
  --dim 128 \
  --data-dir ./data \
  --s3-uri file:///tmp/osprey-s3
```

Multi-node stateless S3 topology (zero ZooKeeper/etcd/Raft):

```bash
# Dedicated writer + CoW compactor node
osprey serve --role writer --dim 128 --data-dir ./writer-data --s3-uri s3://your-bucket/osprey

# Stateless reader node polling manifest.json every 5s
osprey serve --role reader --dim 128 --s3-uri s3://your-bucket/osprey --manifest-poll-secs 5
```

### 2. CLI / Embedded Mode (30 Seconds, Zero External Downloads)

Verify your active SIMD backend, build `.osi`/`.osv` segments from `.jsonl`/`.parquet`/`.npy`, run a search, and verify `xxHash64` checksums:

```bash
osprey --version && osprey doctor
python3 -c 'import json,random; r=random.Random(42); open("docs.jsonl","w").writelines(json.dumps({"doc_id":i,"embedding":[r.gauss(0,1) for _ in range(128)],"text":f"doc {i}"})+"\n" for i in range(512))'
osprey import --input ./docs.jsonl --output ./segments --clusters 16
osprey search --index ./segments/segment.osi --slabs ./segments/segment.osv --query "$(python3 -c 'import json,random; r=random.Random(42); print(json.dumps([r.gauss(0,1) for _ in range(128)]))')" --top-k 5
osprey inspect -f ./segments/segment.osi ./segments/segment.osv --verify-checksums
```

Generate shell completions (`bash`, `zsh`, `fish`, `elvish`, `powershell`):

```bash
osprey completions zsh > _osprey
```

Examples for S3 Parquet data lakes and video/image archives are in [`examples/`](./examples/README.md).

---

## S3 & Object Storage Configuration

`osprey-storage` reads and writes `.osi`/`.osv`/`.osr` slabs and `manifest.json` over S3, GCS, Azure, MinIO, Cloudflare R2, or Ceph RGW using `object_store` HTTP/2 pooled `Range-GET`s with exponential backoff retries (`HTTP 429`/`500`/`502`/`503`/`504`) and speculative hedged requests:

| Environment Variable | Purpose | Example |
| :--- | :--- | :--- |
| `AWS_ACCESS_KEY_ID` | AWS / S3-compatible access key ID | `AKIA...` / `minioadmin` |
| `AWS_SECRET_ACCESS_KEY` | AWS / S3-compatible secret key | `...` / `minioadmin` |
| `AWS_REGION` | Bucket AWS region | `us-east-1` |
| `AWS_ENDPOINT_URL` | Custom S3-compatible endpoint (MinIO, R2, Ceph) | `http://127.0.0.1:9000` |
| `AWS_SESSION_TOKEN` | Optional temporary STS session token | `IQoJb3JpZ2lu...` |
| `AWS_ALLOW_HTTP` | Allow plaintext `http://` endpoints in local/VPC setups | `true` or `1` |

---

## Benchmarks

Run a self-contained latency, `Recall@10`, and AWS TCO projection without external datasets:

```bash
osprey bench --with-lutrank \
  --project-vectors 50000000 --project-dim 1536 --project-qpm 10000000
```

In recall-matched runs (`0.95` and `0.97` targets) on a 12-vCPU / 29 GiB Linux host (OspreyDB bounded to a 256 MiB cache; Qdrant and pgvector in Docker with full host page cache — full protocol in [`ospreydb/benchmarks`](https://github.com/ospreydb/benchmarks/blob/main/results/compare_engines_controlled.md)):

### SIFT1M (`1M × 128d`)

| Engine | Target | R@10 | P50 (ms) | RAM (MiB) | Disk (MiB) | Build (s) |
| :--- | :---: | :---: | ---: | ---: | ---: | ---: |
| OspreyDB (`1-bit .osi + f16 .osv`) | 0.95 | 0.9641 | 2.67 | **462** | **213** | **17.5** |
| OspreyDB (`1-bit .osi + f16 .osv`) | 0.97 | 0.9738 | 3.13 | **462** | **213** | **17.5** |
| Qdrant FP32 (`ef=32`) | 0.95 | 0.9672 | 0.46 | 4,195 | 868 | 34.9 |
| Qdrant FP32 (`ef=64`) | 0.97 | 0.9905 | 0.50 | 4,195 | 868 | 34.9 |
| Qdrant 1-bit BQ (`ef=256, os=3`) | 0.95 | 0.0624* | 0.34 | 4,781 | 911 | 28.3 |
| Qdrant 1-bit BQ (`ef=256, os=3`) | 0.97 | 0.0624* | 0.36 | 4,781 | 911 | 28.3 |
| pgvector FP32 (`ef=64`) | 0.95 | 0.9726 | 2.09 | 2,594 | 2,144 | 306.5 |
| pgvector FP32 (`ef=96`) | 0.97 | 0.9863 | 2.79 | 2,594 | 2,144 | 306.5 |

### Cohere (`100k × 768d`)

| Engine | Target | R@10 | P50 (ms) | RAM (MiB) | Disk (MiB) | Build (s) |
| :--- | :---: | :---: | ---: | ---: | ---: | ---: |
| OspreyDB (`1-bit .osi + f16 .osv`) | 0.95 | 0.9669 | 0.54 | **235** | **152** | 19.6 |
| OspreyDB (`1-bit .osi + f16 .osv`) | 0.97 | **0.9889** | 0.71 | **235** | **152** | 19.6 |
| Qdrant FP32 (`ef=32`) | 0.95 | 0.9647 | 0.28 | 2,731 | 1,368 | 16.8 |
| Qdrant FP32 (`ef=64`) | 0.97 | 0.9855 | 0.44 | 2,731 | 1,368 | 16.8 |
| Qdrant 1-bit BQ (`ef=256, os=3`) | 0.95 | 0.9342* | 0.27 | 2,799 | 661 | 14.9 |
| Qdrant 1-bit BQ (`ef=256, os=3`) | 0.97 | 0.9342* | 0.28 | 2,799 | 661 | 14.9 |
| pgvector FP32 (`ef=64`) | 0.95 | 0.9771 | 1.73 | 1,332 | 1,185 | 64.0 |
| pgvector FP32 (`ef=64`) | 0.97 | 0.9771 | 1.21 | 1,332 | 1,185 | 64.0 |

\* Below recall target.

> **Why does standard 1-bit BQ collapse to `Recall@10 = 0.0624` on SIFT1M?**
> SIFT1M descriptors are strictly non-negative ($\mathbf{x} \in \mathbb{R}_{\ge 0}^{128}$). Uncentered binary quantization ($\text{sign}(x_i) \ge 0$) maps nearly every vector to `111...1`, destroying angular discrimination. OspreyDB avoids this collapse because `RaBitQ` quantizes **centroid-centered residuals** ($\mathbf{x} - \mathbf{c}_k$) after an orthogonal **FWHT** rotation, producing zero-mean subgaussian coordinates regardless of whether the raw embeddings are positive-orthant or anisotropic.

### ColBERT Reranking (`MS MARCO`, `LUT-Rank`, 200 candidates)

| Engine | P50 (ms) | QPS | NDCG@10 | Index (MiB) |
| :--- | ---: | ---: | :---: | ---: |
| OspreyDB `LUT-Rank` (AVX-512 CPU) | **0.33**† | **2,849**† | 0.7768 | **97.2** |
| ColBERTv2 (PyTorch CPU) | 0.42 | 2,301 | 0.7876 | 768.4 |
| ColBERTv2 (PyTorch CUDA GPU) | 0.23 | 4,134 | 0.7876 | 768.4 |
| ColBERTv2 (ONNX Runtime CPU) | 0.48 | 2,005 | 0.7876 | 768.4 |
| ColBERTv2 (NumPy/BLAS CPU) | 0.80 | 1,217 | 0.7876 | 768.4 |

† Fastest CPU engine.

### S3 `RangePlanner` Coalescing

`RangePlanner` merges nearby `.osv` minipage intervals via `O(k)` Dynamic Programming whenever `gap <= gap_coalesce_threshold_bytes`, slashing S3 `Range-GET` round-trips per query while bounding wasted transfer bytes under `max_coalesced_range_bytes`.

---

## Server CLI Flags & Endpoints (`osprey serve`)

| Flag | Env Variable | Default | Description |
| :--- | :--- | :--- | :--- |
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

### HTTP (`:6333`)

- `GET /healthz` — Liveness probe (exempt from `--api-key`)
- `GET /metrics` — Prometheus exposition (`0.0.4`) with P50/P95/P99 latency histograms & S3 call savings
- `GET /collections` — List collections (Qdrant compatible)
- `PUT /collections/{name}` — Ensure/validate collection (Qdrant compatible)
- `GET /collections/{name}` — Collection status, vector counts & embedder metadata
- `PUT /collections/{name}/points` — Batch point upsert (`vector` or `text`, up to 64 MiB payload)
- `POST /collections/{name}/points/search` — Hybrid vector or `query_text` search
- `POST /collections/{name}/points/delete` — Logical point deletion (`points: [id, ...]`)
- `POST /v1/rerank` — Cohere/Jina-compatible `LUT-Rank` reranking (`query`/`documents` or explicit `query_tokens`/`document_tokens`)
- `POST /v1/embeddings` — OpenAI-compatible embeddings proxy

### Apache Arrow Flight gRPC (`:8815`)

- `DoPut` — Zero-copy batch ingestion (`doc_id: UInt64`, `vector: FixedSizeList<Float32/Float64> | List<Float32/Float64>`, optional `payload_json`)
- `DoExchange` — Bidirectional hybrid vector search (`vector`) or `LUT-Rank` rerank (`query`, `document`)
- `ListFlights` / `GetFlightInfo` / `DoGet` — Cluster segment metadata streaming

---

## Development

Build and test from source (requires Rust `1.99.0` per [`rust-toolchain.toml`](./rust-toolchain.toml)):

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --release
cargo check --manifest-path fuzz/Cargo.toml
python3 reference/poc_reference.py
```

Design documents and binary specifications:

- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — byte-level `.osi`, `.osv`, and `.osr` layouts and crate topology
- [`rfcs/`](./rfcs/) — mathematical formulation for `RaBitQ`, DP range coalescing, and `LUT-Rank`
- [`adr/`](./adr/) — architectural decisions
- [`CONTRIBUTING.md`](./CONTRIBUTING.md) · [`SECURITY.md`](./SECURITY.md)

## License

OspreyDB is **100% Open Source under the [Apache-2.0 License](./LICENSE)** (see [`NOTICE`](./NOTICE) and [`CITATION.cff`](./CITATION.cff)).
