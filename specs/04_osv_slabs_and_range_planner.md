# `04_osv_slabs_and_range_planner.md` — Especificação de Armazenamento `.osv`, Reordenação TSP, DP Range Planner e Ingestão Parquet (Fases 8, 9 e 10)

> **Status:** Especificação Fechada para Implementação (Fases 8, 9 e 10)
> **Nota de Contexto:** Este documento integra as especificações originais de engenharia de engenharia do OspreyDB; a implementação de produção reside em `crates/osprey-storage` e [`ARCHITECTURE.md`](../ARCHITECTURE.md).
> **Escopo:** `crates/osprey-storage` (`tsp_locality.rs`, `osv_writer.rs`, `osv_reader.rs`, `range_planner.rs`, `object_store_client.rs`, `lake_ingest.rs`).
> **Referências:** [`ARCHITECTURE.md`](../ARCHITECTURE.md) (Seção 4), [`ADR-001`](../adr/ADR-001-decoupled-1bit-ram-index-and-s3-slabs.md), [`specs/00_math_invariants.md`](./00_math_invariants.md) (Seção 4).

---

## 1. Reordenação Física de Localidade por TSP (`tsp_locality.rs`)

Para maximizar o coalescimento de leituras no S3, clusters semanticamente próximos devem ocupar posições físicas adjacentes tanto no `.osi` quanto nas minipáginas do `.osv`.

### 1.1 Construção da Matriz de Distâncias e Heurística *Nearest-Neighbor* + 2-opt

Dado o conjunto de $K$ centróides materializados em `f16` (`centroids_f16: &[u16]`, shape `[K, padded_dim]`) e o vetor de cardinalidades `cluster_counts: &[u64]`:

1. **Matriz Simétrica $K \times K$:** Promove cada centróide para `f32` (`.to_f32()`) e preenche `dist_matrix[i * K + j] = ||c_i - c_j||_2^2`.
2. **Semente Determinística (*Nearest-Neighbor*):** Inicia o circuito no cluster $s_0$ de **maior cardinalidade** (desempate pelo menor `cluster_id`):
   - Em cada passo, escolhe o próximo cluster não visitado $u = \arg\min_{v \notin \text{visited}} \text{dist}[s_t, v]$ (desempate por menor índice `v`).
3. **Otimização Local 2-opt até Convergência:**
   ```rust
   /// Permutação física produzida pelo solucionador TSP.
   #[derive(Clone, Debug, PartialEq, Eq)]
   pub struct TspPermutation {
       /// `new_to_old[new_cluster_id] == old_cluster_id`
       pub new_to_old: Vec<u32>,
       /// `old_to_new[old_cluster_id] == new_cluster_id`
       pub old_to_new: Vec<u32>,
   }

   /// Executa Nearest-Neighbor a partir do maior cluster + 2-opt até convergência (ou `max_passes`).
   pub fn solve_centroid_tsp(
       centroids_f16: &[u16],
       num_clusters: usize,
       padded_dim: usize,
       cluster_counts: &[u64],
       max_2opt_passes: usize,
   ) -> OspreyResult<TspPermutation>;
   ```
   - Critério estrito de troca no 2-opt para janela `(i, j)` com `1 <= i < j < K - 1` (usando tolerância `1e-6` para evitar ciclos de ponto flutuante):
     ```rust
     let before = d(tour[i - 1], tour[i]) + d(tour[j], tour[j + 1]);
     let after  = d(tour[i - 1], tour[j]) + d(tour[i], tour[j + 1]);
     if after + 1e-6 < before {
         tour[i..=j].reverse();
         improved = true;
     }
     ```
4. **Remapeamento Determinístico Dual (`.osi` + `.osv`):**
   - Os centróides e os blocos Cluster-SoA do `.osi` são reordenados conforme `new_to_old`.
   - O escritor `.osv` emite todas as minipáginas estritamente na ordem `new_cluster_id = 0 .. K - 1`, gravando o novo `cluster_id` tanto no `MinipageHeader` quanto no `MinipageDirectoryEntry`.

---

## 2. Empacotamento de Minipáginas Zstd de 16 KB (`osv_writer.rs` / `osv_reader.rs`)

### 2.1 Layout da Minipágina Descomprimida (`MINIPAGE_TARGET_BYTES = 16_384`)

Cada minipágina agrupa registros consecutivos de um único `cluster_id` até atingir o teto alvo de **16.384 bytes** (ou marca a flag `IS_OVERSIZED_PAGE = 0x0002` se um único vetor + JSON exceder 16 KB):

```text
[0x00..0x0F] MinipageHeader (16 bytes LE):
    cluster_id: u32, vector_count: u16 (M_p), orig_dim: u16 (d),
    zone_map_offset: u32, payloads_offset: u32
[0x10..]     Array Contíguo de Vetores Exatos f16: M_p * d * 2 bytes (+ pad 8B)
[zone_map]   ZoneMapBlock: doc_id_min (u64), doc_id_max (u64), num_entries (u16),
             flags (u16), reserved (u32), entradas numéricas min/max + Bloom tags
[payloads]   Tabela de Offsets [u32; M_p] + sequências (payload_len: u32, json_utf8)
```

### 2.2 Dicionário Zstd e Compressão Paralela via `rayon`

1. **Treinamento Opcional de Dicionário (`zstd::dict::from_samples`):**
   - Quando habilitado (`HAS_ZSTD_DICT = 0x02`), coleta uma amostra determinística de payloads JSON do segmento (alvo de dicionário: `64 KiB – 112 KiB`) e treina via `zstd::dict::from_samples(&samples, dict_capacity)`.
   - O dicionário é gravado no `.osv` em `zstd_dict_offset` (`% 64 == 0`) com `zstd_dict_len` e `zstd_dict_id` registrados no `OsvHeader`.
2. **Compressão Paralela (`rayon`) e Checksum Pre-Decompress:**
   - As minipáginas não comprimidas são distribuídas via `rayon::par_iter()` com `map_init` mantendo um `zstd::bulk::Compressor` reutilizável por worker thread (carregado com `EncoderDictionary` se presente).
   - Para cada frame Zstd comprimido `frame_bytes`, calcula-se imediatamente `page_xxh64 = xxhash_rust::xxh64::xxh64(&frame_bytes, 0)`.
   - Um escritor sequencial grava os frames contiguamente, anota `byte_offset` e `compressed_size`, e grava ao final o diretório `MinipageDirectoryEntry` (32 bytes cada) em `directory_offset` (`% 64 == 0`), preenchendo `OsvHeader::header_xxh64`.
3. **Leitura Segura (`osv_reader.rs`):**
   - **Invariante de Segurança:** O leitor valida `xxh64(compressed_slice, 0) == entry.page_xxh64` **antes** de invocar `zstd::bulk::Decompressor::decompress_to_buffer`, rejeitando imediatamente dados corrompidos em trânsito com `OspreyError::ChecksumMismatch`.
   - O buffer descomprimido tem tamanho exato `entry.uncompressed_size`, evitando realocações durante a descompressão.

---

## 3. Implementação Zero-Allocation do Planejador DP (`range_planner.rs`)

O planejador converte os `minipage_id`s retornados pelos Top-$M$ candidatos sobreviventes em intervalos HTTP `Range-GET` coalescidos, operando em **tempo linear $O(k)$**, **zero alocações de heap** no hot path e latência **$< 0.2\ \text{ms}$**.

```rust
/// Parâmetros de custo calibrados para o Object Storage alvo.
#[derive(Clone, Copy, Debug)]
pub struct RangePlannerConfig {
    /// Custo fixo por requisição HTTP Range-GET (ex.: latência normalizada / tarifa).
    pub tau_req: f64,
    /// Custo marginal por byte transferido (ex.: largura de banda / egresso).
    pub tau_byte: f64,
    /// Teto máximo de uma única leitura coalescida (ex.: 8 * 1024 * 1024 bytes).
    pub max_coalesce_bytes: u64,
}

/// Intervalo coalescido pronto para `object_store::get_ranges`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoalescedRange {
    pub byte_start: u64,
    pub byte_end: u64,
    /// Faixa de índices em `sorted_unique_pages` atendida por esta requisição.
    pub first_page_idx: usize,
    pub last_page_idx_exclusive: usize,
}

/// Buffer pré-alocado de trabalho reutilizado entre consultas (zero heap alloc por query).
pub struct RangeScratch {
    pub unique_page_ids: Vec<u32>,
    pub dp_costs: Vec<f64>,
    pub dp_prev: Vec<usize>,
    /// Fila monótona de índices `j` armazenando `(j, f(j) = C(j-1) - tau_byte * a_j)`.
    pub deque: std::collections::VecDeque<(usize, f64)>,
}

/// Executa o solver DP O(k) usando apenas buffers em stack/scratch pré-alocados,
/// escrevendo os intervalos coalescidos em `out_ranges` e retornando o slice preenchido.
pub fn plan_coalesced_ranges<'a>(
    candidate_pages: &[u32],
    directory: &[MinipageDirectoryEntry],
    cost: RangePlannerConfig,
    scratch: &mut RangeScratch,
    out_ranges: &'a mut [CoalescedRange],
) -> OspreyResult<&'a [CoalescedRange]>;
```

### Passos do Algoritmo $O(k)$

1. Copia `candidate_pages` para `scratch.unique_page_ids`, ordena (`sort_unstable()`) e deduplica (`dedup()`), resultando em $k$ páginas únicas ordenadas por `byte_offset` crescente.
2. Verifica o limite de lacuna $\tau_{\text{gap}} = \tau_{\text{req}} / \tau_{\text{byte}}$ (Teorema 4 de [`specs/00_math_invariants.md`](./00_math_invariants.md)): sempre que `a[i] - b[i - 1] >= tau_gap`, limpa `scratch.deque` (corte exato garantido).
3. Remove da frente da `deque` índices `j` cujo span `b[i] - a[j] > cost.max_coalesce_bytes`; toma o ótimo $C(i)$ diretamente da frente da fila em $O(1)$ amortizado; mantém a monotonicidade inserindo `(i + 1, C(i) - tau_byte * a[i + 1])` na traseira.
4. Reconstrói os `CoalescedRange` reversamente (ou direto em `out_ranges`) e valida `byte_end - byte_start <= max_coalesce_bytes`.

---

## 4. Camada `object_store`, Cache LRU e Resiliência de Rede

### 4.1 Abstração Multi-Cloud e Fatiamento Zero-Copy

O módulo `object_store_client.rs` constrói uma instância unificada `Arc<dyn object_store::ObjectStore>` a partir de URIs canônicas:
- `s3://bucket/prefix` (`AmazonS3Builder` — AWS S3, compatível com IAM/IRSA/STS)
- `gs://bucket/prefix` (`GoogleCloudStorageBuilder`)
- `az://container/prefix` (`MicrosoftAzureBuilder`)
- `http://...` / `https://...` (`AmazonS3Builder` com `with_endpoint` + `with_allow_http` para MinIO / Cloudflare R2 / Ceph RGW)
- `file:///path/...` (`LocalFileSystem`)

**Pipeline Assíncrono de Leitura:**
1. Recebe `&[CoalescedRange]` do planejador DP (`range_planner`) e dispara **uma única chamada concorrente** `store.get_ranges(&path, &byte_ranges).await` sobre pool HTTP/2 persistente.
2. Recebe `Vec<bytes::Bytes>` (um buffer por janela coalescida).
3. Para cada `MinipageDirectoryEntry` coberta pela janela `w`, fatia em **zero-copy** (`let slice = window_bytes.slice(rel_start..rel_end)`) sem copiar bytes intermediários de *gap*, valida `page_xxh64` e descomprime em paralelo.

### 4.2 Cache LRU em RAM para Minipáginas Quentes

- Mantém um cache LRU particionado (*sharded* em 16–64 shards com `parking_lot::Mutex` / `RwLock` para baixa contenção) indexado por `(segment_id: u64, minipage_id: u32)` com teto estrito de bytes em RAM (`max_cache_bytes`).
- Armazena minipáginas descomprimidas `Arc<[u8]>` (ou blocos `f16` + payloads).
- **Fluxo pré-planejador:** Antes de chamar `plan_coalesced_ranges`, o motor consulta o cache LRU para todas as `minipage_id`s candidatas; apenas os *cache misses* seguem para o planejador DP e `get_ranges`, e as páginas recém-descomprimidas populam o cache imediatamente.

### 4.3 Política de *Retry* Exponencial e *Hedged Requests*

1. **Retry com Backoff Exponencial e Full Jitter:**
   - Falhas transitórias (`429 SlowDown`, `500`, `502`, `503`, reset TCP) disparam até `max_retries = 3` com atraso aleatorizado `rand(0 .. base_ms * 2^attempt)`.
2. ***Hedged Requests* para Contenção de $p_{99}$:**
   - Quando `enable_hedging = true`, o cliente agenda um timer adaptativo `hedge_delay` (ex.: `25 ms`, ou `max(15 ms, 1.5 * p95_ewma)`).
   - Se a requisição primária `get_ranges` não completar dentro de `hedge_delay`, `tokio::select!` dispara uma segunda requisição idêntica em outra conexão do pool HTTP/2, adota a primeira resposta válida que chegar e descarta a remanescente.

---

## 5. Ingestão Out-of-Core de Arquivos Parquet (`lake_ingest.rs`)

Para indexar Data Lakes de 50M–500M de vetores em máquinas com RAM limitada (ex.: 16–32 GB), a ingestão opera em **streaming em duas passagens** via `parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder`:

1. **Detecção Automática de Schema e *Projection Pushdown*:**
   - Inspeciona apenas o footer Parquet (`ArrowReaderMetadata`) antes de ler qualquer row-group.
   - Identifica automaticamente:
     - **Coluna ID:** `id` / `doc_id` (`Int64`, `UInt64` ou `Utf8` mapeado via hash/sequência para `u64`).
     - **Coluna Vetor Denso:** `FixedSizeList<Float32 | Float16>` (ou `List<Float32 | Float16>` com verificação uniforme de comprimento `orig_dim`).
     - **Coluna Tokens ColBERT (opcional -> `.osr`):** `List<FixedSizeList<Float32 | Float16>>` de dimensão `128`.
     - **Colunas Escalares/Metadados:** selecionadas via `ProjectionMask::roots` para geração de *Zone Maps* e payloads JSON sem materializar colunas não requisitadas.
2. **Passagem 1 — Amostragem Limitada e Treinamento:**
   - Faz streaming de um reservatório limitado ($N_{\text{sample}} \le 262.144$ vetores, $\sim 500\ \text{MB}$ máx.) com batch size de `4096` linhas:
     - Valida finitude (`OspreyError::NonFiniteEmbedding`).
     - Executa `FWHT` $\to$ `1-Bit Binary K-Means` ($K$ centróides) $\to$ `solve_centroid_tsp` $\to$ treina dicionário Zstd sobre amostra de payloads JSON.
3. **Passagem 2 — Streaming Out-of-Core para `.osi`, `.osv` e `.osr`:**
   - Relê o(s) arquivo(s) Parquet em *record batches* de `4096` linhas, atribuindo cada vetor ao centróide TSP correspondente, quantizando o resíduo `RaBitQ` em memória mínima e gravando partições temporárias por cluster (ou blocos ordenados em disco NVMe) antes de selar os arquivos finais `.osi`, `.osv` e `.osr` com checksums `xxHash64`.
   - Consumo de pico de RAM permanece constante $O(\text{batch\_size} + K \cdot d_{\text{pad}})$ independentemente do número total de linhas do Data Lake.
