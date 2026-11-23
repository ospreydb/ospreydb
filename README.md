# OspreyDB

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](./LICENSE)

**OspreyDB** (`ospreydb/ospreydb`) is a 100% Open Source (Apache-2.0) vector database and CPU late-interaction reranking engine built in Rust for local disk and S3-compatible object storage.

It separates vector search into three 64-byte aligned binary formats plus a crash-safe WAL/LSM production daemon:

- **`.osi`**: 4-bit / 1-bit FWHT-rotated `RaBitQ` resident index scanned via `mmap` and SIMD (`AVX-512 VPOPCNTDQ`, `AVX2`, `ARM64 NEON`).
- **`.osv`**: Zstd-compressed (or uncompressed local zero-copy `mmap`) `f16` vector slabs ordered by TSP locality and fetched via `O(k)` Dynamic Programming coalesced S3 `Range-GET` requests (`RangePlanner`).
- **`.osr`**: Nibble-packed 2-bit nested-lattice ColBERT/ColPali token store reranked on CPU using `vpshufb`/`vtbl` lookup tables (`LUT-Rank`).
- **`osprey serve` Production Daemon**: Crash-safe `Write-Ahead Log (WAL)` + real-time `LSM Delta Buffer`, background Copy-on-Write (`CoW`) slab compactor, 64-bit `RoaringTreemap` tombstone GC, lock-free multi-node S3 `manifest.json` CAS (`If-Match` / `ETag`), Qdrant Drop-In REST API (`:6333`), OpenAI-compatible `/v1/embeddings`, Cohere-compatible `/v1/rerank`, Prometheus `/metrics`, and Apache Arrow Flight gRPC (`:8815`).

---

## Installation

```bash
curl -fsSL https://ospreydb.ramoslucas-lr.workers.dev/install.sh | sh
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
osprey import --input ./docs.jsonl --output ./segments --clusters 16 --bits 4
osprey search --index ./segments/segment.osi --slabs ./segments/segment.osv --query "$(python3 -c 'import json,random; r=random.Random(42); print(json.dumps([r.gauss(0,1) for _ in range(128)]))')" --top-k 5 --cache-mb 256
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

In recall-matched VLDB evaluations (`Recall@10 >= 0.95` and `>= 0.97` targets, calibrated on a 25% validation split and evaluated on unique test queries with 16 concurrent workers and symmetric query-only `cgroup v2 memory.peak` after post-build page-cache eviction) on a 12-vCPU Linux `x86_64` (`avx512_vpopcntdq`) host (OspreyDB running as its containerized production `osprey serve --role unified` daemon with WAL+LSM+CoW active, 4-bit RaBitQ `--bits 4 --clusters 1024`, subsampled K-Means, and `--cache-mb 256`; Qdrant v1.20.2, `pgvector` v0.8.7 on PostgreSQL 16.16, and in-process FAISS v1.16.0 `IndexIVFFlat` `nlist=1024` — full protocol and raw CSVs in [`ospreydb/benchmarks`](https://github.com/ospreydb/benchmarks/blob/main/results/compare_engines_controlled.md), commit `4634c8c`, `osprey_sha=6157dc2`):

### Controlled SIFT1M (`1,000,000 × 128d`)

925 unique test queries (375 validation, 200 warmup, 925 test × 3 rounds, 16 concurrent workers):

| Engine | Target | Config | R@10 | P50 Client / Server (ms) | P95 (ms) | QPS (16w) | Peak RAM / VmHWM (MiB) | Disk (MiB) | Build (s) |
| :--- | :---: | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **OspreyDB Server (`4-bit .osi + f16 .osv`)** | 0.95 | `40/20` | 0.9639 | 0.443 / 0.370 | 0.546 | 16,125.3 | **382.2 / 372.3** | **355.72** | 5.74 |
| **OspreyDB Server (`4-bit .osi + f16 .osv`)** | 0.97 | `96/20` | 0.9721 | 0.906 / 0.828 | 1.075 | 8,503.3 | **382.2 / 372.3** | **355.72** | 5.74 |
| Qdrant HNSW FP32 | 0.95 | `ef=256` | 0.9524 | 0.532 / 0.413 | 0.646 | 7,548.4 | 1,483.2 / 1,477.6 | 1,122.95 | 48.36 |
| Qdrant HNSW FP32 | 0.97 | `ef=512` | 0.9625* | 0.834 / 0.711 | 0.998 | 4,521.8 | 1,483.2 / 1,477.6 | 1,122.95 | 48.36 |
| Qdrant HNSW int8 SQ8 + rescore | 0.95 | `ef=64, os=2.0` | 0.9637 | **0.221 / 0.114** | **0.254** | **20,835.7** | 1,916.0 / 1,911.7 | 1,422.52 | 73.97 |
| Qdrant HNSW int8 SQ8 + rescore | 0.97 | `ef=16, os=10.0` | 0.9852 | **0.267 / 0.152** | **0.310** | **17,974.7** | 1,916.0 / 1,911.7 | 1,422.52 | 73.97 |
| Qdrant HNSW 1-bit BQ + rescore | 0.95 | `ef=384, os=30.0` | 0.0962* | 0.495 / 0.382 | 0.598 | 9,378.1 | 1,654.9 / 1,626.8 | 1,204.86 | 127.03 |
| Qdrant HNSW 1-bit BQ + rescore | 0.97 | `ef=384, os=30.0` | 0.0962* | 0.504 / 0.383 | 0.603 | 9,653.4 | 1,654.9 / 1,626.8 | 1,204.86 | 127.03 |
| pgvector HNSW FP32 | 0.95 | `ef_search=64` | 0.9720 | 0.817 / 0.808 | 1.044 | 6,513.4 | 1,927.9 / 912.6 | 1,361.80 | 197.39 |
| pgvector HNSW FP32 | 0.97 | `ef_search=80` | 0.9811 | 0.965 / 0.925 | 1.242 | 5,809.9 | 1,927.9 / 912.6 | 1,361.80 | 197.39 |
| pgvector HNSW halfvec FP16 | 0.95 | `ef_search=64` | 0.9663 | 0.724 / 0.616 | 0.923 | 8,295.5 | 1,406.9 / 661.7 | 847.09 | 87.93 |
| pgvector HNSW halfvec FP16 | 0.97 | `ef_search=80` | 0.9759 | 0.838 / 0.717 | 1.124 | 7,373.1 | 1,406.9 / 661.7 | 847.09 | 87.93 |
| FAISS IVF-Flat (`nlist=1024`) | 0.95 | `nprobe=32` | **0.9813** | 0.349 / 0.349 | 0.474 | 3,709.8 | 1,342.3 / 1,342.3 | 0.00 | **3.04** |
| FAISS IVF-Flat (`nlist=1024`) | 0.97 | `nprobe=64` | **0.9959** | 0.696 / 0.696 | 0.883 | 1,850.0 | 1,342.3 / 1,342.3 | 0.00 | **3.04** |

- **Footprint & build speed:** OspreyDB leads all engines in query peak memory (`382.2 MiB` cgroup peak; `372.3 MiB` process `VmHWM`) and disk (`355.72 MiB`, **2.4×–4.0× smaller**), using **3.5×–5.0× less RAM** than `pgvector` FP32 (`1,927.9 MiB`), Qdrant SQ8 (`1,916.0 MiB`), Qdrant FP32 (`1,483.2 MiB`), `pgvector halfvec` (`1,406.9 MiB`), and FAISS IVF (`1,342.3 MiB`), while building in **`5.74 s`** (**8.4×–34.4× faster** than containerized Qdrant and `pgvector`).
- **Throughput & latency:** At `Recall@10 >= 0.95` (`R@10 = 0.9639`), OspreyDB achieves `0.443 ms` P50 (`0.370 ms` server) and **`16,125.3 QPS`** (**2×** `pgvector halfvec` and Qdrant FP32, and 77% of Qdrant SQ8's `20,835.7 QPS` at **1/5th the RAM**); at `>= 0.97` (`R@10 = 0.9721`), OspreyDB achieves `0.906 ms` P50 (`0.828 ms` server) and **`8,503.3 QPS`**, outperforming every engine except Qdrant SQ8.

### Controlled Cohere Wikipedia 1M (`1,000,000 × 768d`)

550 unique test queries (250 validation, 200 warmup, 550 test × 3 rounds, 16 concurrent workers):

| Engine | Target | Config | R@10 | P50 Client / Server (ms) | P95 (ms) | QPS (16w) | Peak RAM / VmHWM (MiB) | Disk (MiB) | Build (s) |
| :--- | :---: | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **OspreyDB Server (`4-bit .osi + f16 .osv`)** | 0.95 | `40/20` | 0.9567 | **0.693 / 0.578** | **0.912** | **6,742.9** | **2,018.8 / 1,922.7** | **2,003.51** | **30.16** |
| **OspreyDB Server (`4-bit .osi + f16 .osv`)** | 0.97 | `64/20` | 0.9727 | 0.941 / 0.826 | 1.276 | **5,122.6** | **2,018.8 / 1,922.7** | **2,003.51** | **30.16** |
| Qdrant HNSW FP32 | 0.95 | `ef=80` | 0.9429* | 2.828 / 2.577 | 3.216 | 545.7 | 6,496.2 / 5,826.9 | 6,002.80 | 1,133.57 |
| Qdrant HNSW FP32 | 0.97 | `ef=192` | 0.9651* | 3.246 / 2.959 | 3.666 | 457.4 | 6,496.2 / 5,826.9 | 6,002.80 | 1,133.57 |
| Qdrant HNSW int8 SQ8 + rescore | 0.95 | `ef=96, os=1.0` | 0.9449* | 4.699 / 4.299 | 5.074 | 417.0 | 6,450.6 / 6,001.0 | 5,841.70 | 1,080.50 |
| Qdrant HNSW int8 SQ8 + rescore | 0.97 | `ef=256, os=1.0` | 0.9678* | 4.797 / 4.391 | 5.296 | 406.4 | 6,450.6 / 6,001.0 | 5,841.70 | 1,080.50 |
| Qdrant HNSW 1-bit BQ + rescore | 0.95 | `ef=16, os=20.0` | **0.9658** | 0.805 / 0.641 | 0.963 | 4,870.3 | 4,851.4 / 4,190.2 | 4,455.00 | 773.94 |
| Qdrant HNSW 1-bit BQ + rescore | 0.97 | `ef=16, os=30.0` | **0.9762** | **0.849 / 0.678** | **1.089** | 4,212.1 | 4,851.4 / 4,190.2 | 4,455.00 | 773.94 |
| pgvector HNSW FP32 | 0.95 | `ef_search=48` | 0.9418* | 1.750 / 2.063 | 2.880 | 2,351.1 | 6,306.9 / 2,098.7 | 7,935.14 | 1,656.01 |
| pgvector HNSW FP32 | 0.97 | `ef_search=80` | 0.9593* | 2.485 / 2.684 | 3.608 | 1,757.2 | 6,306.9 / 2,098.7 | 7,935.14 | 1,656.01 |
| pgvector HNSW halfvec FP16 | 0.95 | `ef_search=64` | 0.9496* | 1.330 / 1.369 | 2.113 | 3,038.9 | 4,248.8 / 2,064.5 | 3,537.39 | 580.42 |
| pgvector HNSW halfvec FP16 | 0.97 | `ef_search=96` | 0.9615* | 1.741 / 1.695 | 2.475 | 2,505.4 | 4,248.8 / 2,064.5 | 3,537.39 | 580.42 |
| FAISS IVF-Flat (`nlist=1024`) | 0.95 | `nprobe=32` | 0.9424* | 3.115 / 3.115 | 4.019 | 515.4 | 7,275.1 / 7,275.1 | 0.00 | 42.57 |
| FAISS IVF-Flat (`nlist=1024`) | 0.97 | `nprobe=64` | 0.9718 | 6.110 / 6.110 | 7.471 | 256.8 | 7,275.1 / 7,275.1 | 0.00 | 42.57 |

\* Below recall target on test split.

- **#1 in QPS, RAM, Disk, and Build Time at `1M × 768d`:** OspreyDB is the only engine besides Qdrant BQ meeting both `0.95` (`0.9567`) and `0.97` (`0.9727`) test targets, while delivering the **highest concurrent throughput at both targets (`6,742.9 / 5,122.6 QPS`)** and the fastest `0.95` sequential P50 (`0.693 ms`, server `0.578 ms`), using **`2,018.8 MiB` query peak RSS** (`1,922.7 MiB` `VmHWM`) and **`2,003.51 MiB` disk**, and building in **`30.16 s`** (faster even than in-process FAISS IVF at `42.57 s`).
- **Comparison against quantized & FP32 baselines:** OspreyDB delivers **+38% / +22% higher QPS** than Qdrant 1-bit BQ (`4,870.3 / 4,212.1 QPS`) at **2.4× less RAM**, **2.2× less disk**, and **25.7× faster build** (`30.16 s` vs `773.94 s`); **2.0×–2.2× higher QPS** than `pgvector halfvec` (`3,038.9 / 2,505.4 QPS`) at **2.1× less RAM** and **19.2× faster build**; and **11×–16× higher QPS** than Qdrant FP32/SQ8 at **3.2× less RAM** and **36×–38× faster build**.

> **Why does standard 1-bit BQ collapse to `Recall@10 = 0.0962` on SIFT1M (`ef=384, oversampling=30.0`)?**
> SIFT1M descriptors are strictly non-negative ($\mathbf{x} \in \mathbb{R}_{\ge 0}^{128}$). Uncentered binary quantization ($\text{sign}(x_i) \ge 0$) maps nearly every vector to `111...1`, destroying angular discrimination. OspreyDB avoids this collapse because `RaBitQ` quantizes **centroid-centered residuals** ($\mathbf{x} - \mathbf{c}_k$) after an orthogonal **FWHT** rotation into 4-bit progressive bitplanes (`--bits 4`) or 1-bit codes (`--bits 1`), producing zero-mean subgaussian coordinates regardless of whether the raw embeddings are positive-orthant or anisotropic, followed by exact `f16` Stage-2 rescoring on just `20` candidates.

### ColBERT Reranking (`MS MARCO`, `SciFact`, `FiQA`, `LUT-Rank`, 200 candidates)

Reranking 200 candidate passages per query across 100 queries ([`results/lutrank_vs_colbert.csv`](https://github.com/ospreydb/benchmarks/blob/main/results/lutrank_vs_colbert.csv)):

| Dataset | Engine | Device | P50 (ms) | P99 (ms) | QPS | NDCG@10 | MRR@10 | Index (MiB) |
| :--- | :--- | :--- | ---: | ---: | ---: | :---: | :---: | ---: |
| **MS MARCO** | **OspreyDB `LUT-Rank`** | **AVX-512 CPU** | **0.329†** | 0.552 | **2,849†** | 0.7768 | 0.7389 | **97.2** |
| MS MARCO | ColBERTv2 (PyTorch) | CPU | 0.416 | **0.534†** | 2,301 | 0.7876 | 0.7530 | 768.4 |
| MS MARCO | ColBERTv2 (ONNX Runtime) | CPU | 0.476 | 0.733 | 2,005 | 0.7876 | 0.7530 | 768.4 |
| MS MARCO | ColBERTv2 Exact (NumPy) | BLAS CPU | 0.801 | 0.899 | 1,217 | 0.7876 | 0.7530 | 768.4 |
| MS MARCO | ColBERTv2 (PyTorch) | CUDA GPU | 0.226 | 0.308 | 4,134 | 0.7876 | 0.7530 | 768.4 |
| **SciFact** | **OspreyDB `LUT-Rank`** | **AVX-512 CPU** | 0.505 | **0.551†** | 2,122 | 0.5854 | 0.5437 | **10.2** |
| SciFact | ColBERTv2 (PyTorch) | CPU | 0.345 | 1.584 | **2,398†** | 0.5999 | 0.5629 | 81.0 |
| SciFact | ColBERTv2 (ONNX Runtime) | CPU | **0.338†** | 4.574 | 1,365 | 0.5999 | 0.5629 | 81.0 |
| SciFact | ColBERTv2 Exact (NumPy) | BLAS CPU | 0.679 | 0.849 | 1,453 | 0.5999 | 0.5629 | 81.0 |
| SciFact | ColBERTv2 (PyTorch) | CUDA GPU | 0.207 | 0.703 | 2,581 | 0.6006 | 0.5638 | 81.0 |
| **FiQA** | **OspreyDB `LUT-Rank`** | **AVX-512 CPU** | 0.519 | 0.722 | 2,006 | 0.4257 | 0.4965 | **112.8** |
| FiQA | ColBERTv2 (PyTorch) | CPU | **0.428†** | **0.532†** | **2,220†** | 0.4372 | 0.5122 | 892.0 |
| FiQA | ColBERTv2 (ONNX Runtime) | CPU | 0.469 | 4.090 | 1,050 | 0.4372 | 0.5122 | 892.0 |
| FiQA | ColBERTv2 Exact (NumPy) | BLAS CPU | 0.864 | 0.955 | 1,143 | 0.4372 | 0.5122 | 892.0 |
| FiQA | ColBERTv2 (PyTorch) | CUDA GPU | 0.229 | 0.338 | 4,126 | 0.4389 | 0.5137 | 892.0 |

† Fastest CPU engine (`7.9×` smaller `.osr` index across all datasets).

### Cold-Storage S3 Pareto (`Cohere 1M × 768d`, MinIO S3 + `tc netem 15ms`)

25-point `nprobe × refine_m` sweep on Cohere Wikipedia 1M (`1,000,000 × 768d`: `513.2 MiB` 4-bit `.osi` in RAM + `1,378.7 MiB` Zstd-compressed `f16` `.osv` on MinIO S3 with `17.46 ms` measured RTT, `--clusters 1024 --bits 4`, `--cache-mb 0`, 200 queries — full sweep in [`results/pareto_s3.csv`](https://github.com/ospreydb/benchmarks/blob/main/results/pareto_s3.csv)):

| `nprobe / refine_m` | R@10 | P50 (ms) | P95 (ms) | Mean (ms) | Avg Raw Pages | Avg S3 GETs | Bytes Fetched / Query | VmHWM (MiB) |
| :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `40 / 20` | 0.9555 | **20.26** | 33.27 | 22.03 | 19.74 | **19.50** | 207.9 KB | **1,066.5** |
| `48 / 20` | 0.9685 | **20.37** | 31.63 | 22.00 | 19.74 | **19.50** | 206.3 KB | **1,066.5** |
| `64 / 20` | 0.9780 | **20.54** | 32.19 | 22.15 | 19.74 | **19.51** | 206.2 KB | **1,066.5** |
| `64 / 30` | 0.9795 | 21.32 | 34.78 | 23.29 | 29.52 | 29.03 | 311.8 KB | 1,066.5 |
| `64 / 40` | **0.9815** | 47.21 | 56.37 | 47.63 | 39.17 | 38.37 | 426.0 KB | 1,066.5 |

- With 4-bit progressive RaBitQ Stage-1 pruning, Stage-2 only needs **`refine_m = 20` candidates** instead of `200–300`: OspreyDB reaches **`Recall@10 = 0.9555` in `20.26 ms` P50** (`19.50` S3 Range-GETs, `207.9 KB/query`) and **`Recall@10 = 0.9780` in `20.54 ms` P50** (`19.51` S3 Range-GETs, `206.2 KB/query` — virtually a single parallel HTTP/2 RTT over `17.46 ms` WAN latency) and **`0.9815` at `47.21 ms` P50**, while keeping local `VmHWM` at **`~1,067 MiB`** (`1,066.5–1,067.2 MiB`) with zero local `.osv` slab cache (`--cache-mb 0`).
- `RangePlanner` merges nearby `.osv` minipage intervals via `O(k)` Dynamic Programming whenever `gap <= gap_coalesce_threshold_bytes`, slashing S3 `Range-GET` round-trips per query while bounding wasted transfer bytes under `max_coalesced_range_bytes`.

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
| `--cache-mb` | `OSPREY_CACHE_MB` | `256` | Maximum in-memory `.osv` minipage LRU cache capacity in MiB (`0` disables caching) |
| `--osv-compression` | `OSPREY_OSV_COMPRESSION` | `auto` | `.osv` slab compression mode for compacted segments (`auto` = `none` on `file://`, `zstd` on S3/cloud; `none`; `zstd`) |
| `--embed-url` (`--embeddings-url`) | `OSPREY_EMBED_URL` | — | OpenAI-compatible base URL for `/v1/embeddings` proxy |
| `--embed-model` | `OSPREY_EMBED_MODEL` | `text-embedding-3-small` | Model name sent to `/v1/embeddings` |
| `--embed-api-key` | `OSPREY_EMBED_API_KEY` | — | Bearer token for upstream `/v1/embeddings` provider |
| `--embed-batch-size` | — | `32` | Max texts per upstream embedding request |

Key indexing & search CLI flags (`osprey import` / `osprey search` / `osprey rerank`):

| Subcommand | Flag | Default | Description |
| :--- | :--- | :--- | :--- |
| `osprey import` | `--bits` | `1` (`1` or `4`) | `RaBitQ` residual quantization bit-width per coordinate (`1` = 1-bit bitplane, `4` = 4-bit progressive / FastScan `.osi`) |
| `osprey import` | `--clusters` | `256` (`1024` for 1M+) | Number of `1-Bit Binary K-Means` clusters $K$ (trained via Knuth Algorithm-S reservoir subsampling) |
| `osprey import` | `--sample-size` | `0` (auto) | Pass-1 reservoir sample size for Binary K-Means (`0` = `max(65536, 256 * K)`) |
| `osprey import` | `--osv-compression` | `auto` | `.osv` slab compression mode (`auto` = uncompressed zero-copy `mmap` for local paths, `zstd` for S3/remote; `none`; `zstd`) |
| `osprey search` / `rerank` | `--cache-mb` | `256` | Maximum local `.osv` minipage LRU cache capacity in MiB (`0` disables local slab caching for cold S3 evaluation) |
| `osprey search` / `rerank` | `--nprobe` | `16` | Number of nearest TSP clusters to probe in `.osi` |
| `osprey search` / `rerank` | `--candidates` | `200` | Number of Stage-1 `RaBitQ` candidates passed to DP range coalescing + exact `f16` refinement |
| `osprey search` / `rerank` | `--refine-multiplier` | `0` (adaptive) | Stage-1 oversampling multiplier (`0` = adaptive `2x` when `dim <= 256` for 1-bit `.osi`; `1` disables) |

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
