# OspreyDB — Visão Geral de Arquitetura (`ARCHITECTURE_OVERVIEW.md`)

> **Escopo:** Visão visual de ponta a ponta da arquitetura unificada 100% Apache-2.0 (`ospreydb/ospreydb`), em conformidade estrita com [`ARCHITECTURE.md`](../ARCHITECTURE.md) e [`ADR-001`–`ADR-003`](../adr/).

---

## 1. Por Que Desacoplar Índice 1-Bit em RAM de Slabs no S3?

Bancos vetoriais baseados em grafos HNSW residentes em RAM sofrem de duas assimetrias em escala:
1. **Custo de DRAM:** Manter vetores `FP32` ($6\ \text{KB}$ para $d = 1536$) mais listas de adjacência HNSW na memória principal custa ~300× a 450× mais por GB/mês do que Object Storage (S3 Standard).
2. **Colapso de HNSW no Object Storage:** Tentar ler um grafo HNSW diretamente do S3 via `Range-GET` dispara 12–30 saltos **estritamente sequenciais** dependentes de ponteiros ($v_{t+1}$ depende dos vizinhos de $v_t$). Com TTFB típico de $15\text{–}40\ \text{ms}$ no S3, uma única query acumula $300\text{–}1.200\ \text{ms}$ de rede e explode a fatura de chamadas `GET`.

O **OspreyDB** resolve ambos os problemas particionando cada segmento em três arquivos imutáveis alinhados a 64 bytes:

- **`.osi` (Residente em RAM / `mmap`):** Índice binário `RaBitQ` de **1 bit por dimensão** + centróides `f16` ordenados por circuito TSP + `RoaringBitmap` de tombstones. Poda > 99% do corpus localmente via SIMD (`AVX-512 VPOPCNTDQ`, `AVX2`, `ARM64 NEON`).
- **`.osv` (Frio em S3 / MinIO / Disco):** Minipáginas de ~16 KiB comprimidas com **Zstd** (com dicionário treinado opcional) ordenadas por localidade TSP, contendo `doc_id`, vetores exatos em `f16`, *zone maps* (`min_doc_id`/`max_doc_id`) e payloads JSON. Lido em uma única rodada paralela via planejador DP de ranges (`RangePlanner`).
- **`.osr` (Rerank Multi-Vetor em `mmap`/NVMe):** Matriz de tokens ColBERT/ColPali ($D = 128$) comprimida em reticulado aninhado 2D de 2 camadas (64 bytes/token = 1 linha de cache) e avaliada via `LUT-Rank` (`vpshufb`/`vtbl`) 100% na CPU.

---

## 2. Topologia Acíclica de Crates (Workspace Apache-2.0)

```text
                +-----------------------------------------------+
                |          crates/ospreydb (`osprey`)           |
                |  WAL CRC32/fsync, LSM Delta Buffer, CoW GC,   |
                |  S3 CAS manifest.json, HTTP :6333, Flight :8815|
                +-----------------------+-----------------------+
                                        |
                                        v
+-------------------+     +-----------------------------------------------+
|    osprey-cli     |---->|                osprey-storage                 |
|  Subcomandos CLI: |     |  memmap2 (.osi/.osr), object_store/S3 async,  |
|  doctor, import,  |     |  Zstd + DecoderDictionary, DP range planner,  |
|  search, rerank,  |     |  RoaringBitmap tombstones, LRU de minipáginas |
|  delete, inspect, |     +-----------------------+-----------------------+
|  bench            |                             |
+---------+---------+                             v
          |               +-----------------------------------------------+
          |               |                  osprey-simd                  |
          +-------------->|  Dispatch runtime: AVX-512 (VPOPCNTDQ/BW),    |
          |               |  AVX2+FMA+F16C, ARM64 NEON e fallback escalar |
          |               +-----------------------+-----------------------+
          |                                       |
          |                                       v
          |               +-----------------------------------------------+
          +-------------->|                  osprey-core                  |
                          |  Structs #[repr(C, align(64))] Pod+Zeroable,  |
                          |  FWHT, 1-Bit Binary K-Means, RaBitQ assimét., |
                          |  DP range coalescing, LUT-Rank, xxHash64,     |
                          |  traits DeltaBufferExt / StorageEngineExt     |
                          +-----------------------------------------------+
```

---

## 3. Pipeline de Build Offline (`osprey import` em 2 Passos)

O CLI `osprey` lê tabelas Apache Parquet/JSONL/NPY locais ou no S3 com *column projection pushdown* (lendo apenas a coluna vetorial no Passo 1), mantendo o pico de RAM limitado:

```text
  [S3 / Local: *.parquet | *.jsonl | *.npy]
                     |
                     v  (Passo 1: Arrow column projection na coluna de vetores)
  +-------------------------------------------------------------------------+
  | 1. Rotação FWHT Determinística (O(d log d), sem matriz densa O(d^2))    |
  |    x_rot = (1 / sqrt(d_pad)) * H_d * D(fwht_seed) * x                   |
  |    d_pad = next_power_of_two(max(d_orig, 64))                           |
  +-------------------------------------------------------------------------+
                     |
                     v
  +-------------------------------------------------------------------------+
  | 2. 1-Bit Binary K-Means + Otimizador TSP (Nearest-Neighbor + 2-opt)     |
  |    - Treina K centróides sobre assinaturas binárias pós-FWHT            |
  |    - Reordena clusters [0..K-1] para co-localizar clusters vizinhos     |
  +-------------------------------------------------------------------------+
                     |
         +-----------+----------------------------+
         |                                        |
         v  (Passo 2: streaming de vetores +      v  (Opcional: --with-lutrank)
         |   metadados JSON em lotes)             |
  +-------------------------------+      +----------------------------------+
  | 3a. Codificação RaBitQ (.osi) |      | 3c. Nested Lattice (.osr)        |
  |  - Sinais 1-bit [u64; d/64]   |      |  - FWHT nos tokens (sigma=1/sqrtD)|
  |  - res_norm + l1_norm_factor  |      |  - 64 subespaços 2D x 2 camadas  |
  |  - Locator (doc_id, page,slot)|      |  - 4 bits/camada -> 64 B/token   |
  +-------------------------------+      +----------------------------------+
         |                                        |
         v                                        v
  +-------------------------------+      [segment.osr] (mmap / NVMe)
  | 3b. Minipáginas 16 KiB (.osv) |
  |  - Agrupadas por cluster TSP  |
  |  - Vetores f16 + JSON payload |
  |  - Dicionário Zstd + xxHash64 |
  +-------------------------------+
         |
         v
  [segment.osi] (RAM/mmap) + [segment.osv] (S3 / Object Storage)
```

---

## 4. Pipeline de Consulta Online em 4 Estágios

```text
  [Query q (d_orig) + Tokens de Query Q_tok (m x 128, opcional)]
         |
         v
  +=========================================================================+
  | ESTÁGIO 1 — Rotação FWHT + Scan 1-Bit RaBitQ (RAM / mmap .osi)         |
  |   1. FWHT(q) (< 10 us) -> transposição int4 em 4 planos de bits [u64]   |
  |   2. Seleção dos nprobe centróides f16 mais próximos (em ordem TSP)     |
  |   3. Loop quente AVX-512 VPOPCNTDQ sobre Stream contíguo de bits 1-bit  |
  |   4. Reconstrução assimétrica (res_norm, l1_norm_factor)                |
  |   5. Filtro de tombstones via RoaringBitmap antes do heap Top-C         |
  +=========================================================================+
         |  Top-C candidatos: (doc_id, minipage_id, slot_idx, score_est)
         |  (Para d_orig <= 256, SearchOptions::resolve aplica oversampling 2x)
         v
  +=========================================================================+
  | ESTÁGIO 2 — Planejador DP de Ranges + S3 Range-GET Paralelo (.osv)       |
  |   1. Consulta MinipageDirectoryEntry (48 B) em RAM + zone maps (min/max)|
  |   2. Verifica LRU cache local de minipáginas descomprimidas             |
  |   3. Solver DP 1D (< 0.5 ms) ordena offsets e funde lacunas (gaps)      |
  |      quando custo de latência de novo GET > transferência do gap        |
  |   4. Dispara HTTP/2 get_ranges paralelo (com hedged requests p/ P99)    |
  |   5. Valida page_xxh64 ANTES da descompressão Zstd (DecoderDictionary)  |
  +=========================================================================+
         |  Minipáginas ~16 KiB descomprimidas em RAM
         v
  +=========================================================================+
  | ESTÁGIO 3 — Refinamento Exato f16 + Payloads (.osv)                     |
  |   1. Acesso direto ao slice contíguo f16 (sem parsear JSON no hot path) |
  |   2. Produto interno exato f16 -> acumulador f32 (AVX-512/AVX2/NEON)    |
  |   3. Extrai payloads JSON apenas para os Top-R / Top-K sobreviventes    |
  +=========================================================================+
         |  Top-R documentos candidatos (ex.: 200) -- ou Top-K direto se sem .osr
         v
  +=========================================================================+
  | ESTÁGIO 4 — Reranking Multi-Vetor LUT-Rank na CPU (mmap .osr)           |
  |   1. Rotação FWHT dos tokens da query com OsrHeader::fwht_seed          |
  |   2. Síntese de LUTs int8 de 16 entradas (< 8 us em registradores)      |
  |   3. Busca binária O(log N_docs) em DocTokenRange -> tokens de 64 B     |
  |   4. Kernel MaxSim via vpshufb (x86) / vtbl (ARM64) sem float multiply  |
  |      (< 3.8 ms para 200 documentos; 0.329 ms P50 no MS MARCO AVX-512)   |
  +=========================================================================+
         |
         v
  [Top-K Resultados Verificados + Payloads JSON]
```

### Orçamento de Latência por Estágio ($p_{99}$ Alvo — `ARCHITECTURE.md`)

| Estágio | Operação | Domínio | Orçamento Alvo ($p_{99}$) |
| :--- | :--- | :--- | :---: |
| **1a / 1b** | Rotação FWHT da query + transposição 4 planos `int4` | CPU | $< 15\ \mu\text{s}$ |
| **1c** | Scan `VPOPCNTDQ` + filtro `RoaringBitmap` (`.osi`) | CPU / `mmap` | $0.8 - 2.5\ \text{ms}$ |
| **2a** | Solver de Programação Dinâmica de Ranges | CPU | $< 0.5\ \text{ms}$ |
| **2b** | S3 `get_ranges` paralelo (`.osv`) | S3 / NVMe | $12 - 35\ \text{ms}$ (S3 frio) / $< 0.2\ \text{ms}$ (NVMe/cache) |
| **2c / 3** | Validação `page_xxh64` + Zstd + refinamento exato `f16` | CPU | $< 0.8\ \text{ms}$ |
| **4a** | Síntese das LUTs `int8` de 16 entradas | CPU | $< 8\ \mu\text{s}$ |
| **4b** | Kernel `MaxSim` via `vpshufb` / `vtbl` (`.osr`) | CPU / `mmap` | $< 3.8\ \text{ms}$ |
| **Total CPU** | **Overhead total de computação (excluindo RTT S3)** | **CPU** | **$< 7.6\ \text{ms}$** |

---

## 5. Fluxo do Daemon `osprey serve` (Escrita Contínua + S3 CAS `manifest.json`)

Enquanto os subcomandos `osprey import`/`search`/`delete` operam em modo CLI direto sobre segmentos imutáveis, o modo daemon **`osprey serve`** implementa `StorageEngineExt` para cargas 24/7 com escrita concorrente e múltiplos leitores *stateless* sem ZooKeeper, etcd ou Raft:

```text
  [Clientes: HTTP :6333 (Qdrant / /v1/rerank / /v1/embeddings) | Arrow Flight :8815]
         |                                                         |
         | (Escrita / Upsert / Delete)                             | (Leitura / Search)
         v                                                         v
  +-------------------------------+                     +-------------------------------+
  | 1. Write-Ahead Log (WAL)      |                     | Leitura Unificada (RYOW):     |
  |  - Frames CRC32 + fsync       |                     |  - Query em segmentos .osi    |
  |  - Recuperação de torn-tail   |                     |    ativos via mmap            |
  +---------------+---------------+                     |  + query_live_deltas() sobre  |
                  |                                     |    o LSM Delta Buffer         |
                  v                                     +---------------+---------------+
  +-------------------------------+                                     ^
  | 2. LSM Delta Buffer (RAM)     |                                     |
  |  - FWHT + quantização 1-bit   |-------------------------------------+
  |    instantânea contra os      |
  |    centróides do .osi ativo   |
  +---------------+---------------+
                  | (Threshold de tamanho/tempo atingido)
                  v
  +-------------------------------------------------------------------------+
  | 3. Background Workers (Sem bloquear leitores)                           |
  |  - Copy-on-Write Slab Compactor: materializa novos .osi/.osv/.osr       |
  |  - Tombstone GC: reescreve slabs fragmentados quando taxa > limiar      |
  +-------------------------------------------------------------------------+
                  |
                  v  S3 PutObject Condicional (If-Match: <ETag> / If-None-Match: *)
  +-------------------------------------------------------------------------+
  | 4. Object Storage (S3 / MinIO): manifest.json (CAS) + novos segmentos   |
  +-------------------------------------------------------------------------+
                  |
                  v  Polling leve de HeadObject / GetObject (lock-free)
  +-------------------------------------------------------------------------+
  | 5. Nós Leitores Stateless (100% Apache-2.0): atualizam mmap sem restart |
  +-------------------------------------------------------------------------+
```
