# ADR-003: Unificação 100% Open Source (Apache-2.0) no Repositório `ospreydb/ospreydb`

| Metadado | Valor |
| :--- | :--- |
| **ADR** | `003` |
| **Status** | **Accepted (Supersedes historical open-core/commercial split)** |
| **Data** | `2026-11-18` |
| **Autores** | Engenharia Principal do OspreyDB (`ospreydb/ospreydb`) |
| **Referências Canônicas** | [`ARCHITECTURE.md`](../ARCHITECTURE.md) (Seção 1.2–1.3), [`ADR-001`](./ADR-001-decoupled-1bit-ram-index-and-s3-slabs.md), [`ADR-002`](./ADR-002-simd-register-lut-nested-lattice-reranking.md) |

---

## 1. Contexto e Decisão

Originalmente, o OspreyDB foi concebido com uma divisão entre núcleo estático CLI (`osprey-core`, `osprey-simd`, `osprey-storage`, `osprey-cli`) e daemon transacional/multi-nó (`ospreydb`), incluindo verificação de licença e cotas de vetores.

Em **2026-11-18**, consolidamos todo o projeto em um **único repositório 100% Open Source sob a licença `Apache-2.0`** (`ospreydb/ospreydb`), removendo integralmente qualquer paywall, verificação de licença, limite de vetores ou restrição de papéis multi-nó.

---

## 2. Escopo do Workspace Unificado (`ospreydb/ospreydb` — Licença `Apache-2.0`)

Todas as capacidades são entregues sem restrições no workspace Cargo de 5 crates:

1. **Formatos Físicos e Núcleo Matemático (`crates/osprey-core`):**
   - `.osi` (1-Bit Resident Index + `RoaringBitmap` de tombstones + checksums `xxHash64`).
   - `.osv` (Slabs ordenados por TSP com minipáginas Zstd de 16 KB + dicionário opcional + *zone maps*).
   - `.osr` (Token store nibble-packed para `LUT-Rank`).
   - Rotação ortogonal `FWHT`, `1-Bit Binary K-Means`, heurística TSP, `RaBitQ` assimétrico, `RangePlanner` DP e `LUT-Rank`.
2. **Kernels SIMD (`crates/osprey-simd`):**
   - Despacho dinâmico para `AVX-512 VPOPCNTDQ`/`BW`, `AVX2`, `ARM64 NEON` e fallback escalar portátil.
3. **Camada de Armazenamento e Rede (`crates/osprey-storage`):**
   - Leitores `memmap2` zero-copy e transporte assíncrono `object_store` (S3, GCS, Azure, HTTP, local) com HTTP/2 pooling, retry exponencial e *hedged requests*.
4. **Biblioteca e Comandos CLI (`crates/osprey-cli`):**
   - `import`, `search`, `rerank`, `delete`, `inspect`, `doctor`, `bench`, `completions`.
5. **Daemon de Produção e Binário Unificado `osprey` (`crates/ospreydb`):**
   - Binário único `osprey` integrando `osprey serve` e todos os subcomandos CLI + `wal-stress`.
   - `Write-Ahead Log (WAL)` com `fsync` + `LsmDeltaBuffer` para escrita contínua e leitura imediata (*read-your-own-writes*).
   - `CowSlabCompactor` + `TombstoneGarbageCollector` em background.
   - Coordenação multi-nó *stateless* via `manifest.json` CAS (`If-Match` / `ETag`) sem ZooKeeper, etcd ou Raft, com papéis `--role unified`, `--role writer` e `--role reader` irrestritos.
   - Servidores de produção: Qdrant Drop-In REST (`:6333`), Cohere/Jina `/v1/rerank`, OpenAI `/v1/embeddings`, Prometheus `/metrics` e Apache Arrow Flight gRPC (`:8815`).
