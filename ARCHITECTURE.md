# OspreyDB — Especificação Canônica de Arquitetura de Sistemas

> **Status:** Referência Canônica (v1.0)
> **Escopo:** Workspace Unificado Apache-2.0 (`osprey-core`, `osprey-simd`, `osprey-storage`, `osprey-cli`, `ospreydb`).
> **Público-alvo:** Engenheiros de sistemas, implementadores de kernels SIMD e mantenedores de storage engines do ecossistema `ospreydb`.

---

## 1. Visão Geral do Sistema e Topologia do Workspace

O **OspreyDB** é um motor de busca vetorial híbrida e *late-interaction* orientado a **zero-copy**, **localidade espacial em Object Storage** e **execução SIMD determinística**. A arquitetura separa radicalmente o plano de índice residente em RAM/mmap (`.osi`), o plano de armazenamento de vetores exatos e payloads em object storage (`.osv`) e o plano de reranking multi-vetor por tokens (`.osr`).

### 1.1 Fluxo de Dados de Ponta a Ponta (Ingestão -> Artefatos -> Query em 4 Estágios)

```text
================================================================================
                        FASE OFFLINE: PIPELINE DE BUILD
================================================================================

  [Parquet / JSONL]
         |  (doc_id, vetor denso f32/f16, tokens ColBERT f16, payload JSON)
         v
  +--------------------------------------------------------------------------+
  | 1. Rotação FWHT Determinística: x_rot = (1/sqrt(d_pad)) * H * D(seed) * x|
  |    Pad para d_pad = next_pow2(max(d_orig, 64))                           |
  +--------------------------------------------------------------------------+
         |
         +------------------------------+------------------------------------+
         |                              |                                    |
         v                              v                                    v
  +-----------------------+    +---------------------+    +--------------------+
  | 2a. Binary K-Means    |    | 2b. Otimizador TSP  |    | 2c. Nested Lattice |
  |   Treina K centróides |--->|  Ordena centróides  |    |  2 Camadas x 4-bit |
  |   binarizados + f16   |    |  e agrupa slabs     |    |  subspace_dim = 2  |
  +-----------------------+    +---------------------+    |  gamma = 0.28      |
         |                              |                 +--------------------+
         v                              v                            |
  +-----------------------+    +---------------------+               |
  |  Codificação RaBitQ   |    | Minipáginas 16 KB   |               |
  |  Assimétrica (1-bit   |    | f16 + Zone Maps +   |               |
  |  + res_norm/l1_factor)|    | JSON -> Zstd + Dict |               |
  +-----------------------+    +---------------------+               |
         |                              |                            |
         v                              v                            v
   [ segmento.osi ]              [ segmento.osv ]             [ segmento.osr ]
  (Índice Residente 1-Bit)      (Slabs S3 + Payloads)      (Tokens Nibble-Packed)


================================================================================
                     FASE ONLINE: PIPELINE DE QUERY EM 4 ESTÁGIOS
================================================================================

  [Query q (d_orig) + Tokens Query Q_tok (m x 128)]
         |
         v
  +--------------------------------------------------------------------------+
  | ESTÁGIO 1: Rotação FWHT + Transposição int4 + Scan RaBitQ (RAM/mmap .osi)|
  |  * FWHT(q) -> O(d log d) -> Quantização 4-bit -> 4 Planos de Bits [u64]  |
  |  * Seleção de nprobe clusters via centróides f16 em ordem TSP            |
  |  * Varredura sequencial AVX-512 VPOPCNTDQ + reconstrução assimétrica     |
  |  * Filtragem imediata de tombstones via RoaringBitmap                    |
  +--------------------------------------------------------------------------+
         |  (Top-C candidatos: doc_id, minipage_id, slot_idx, score_est)
         v
  +--------------------------------------------------------------------------+
  | ESTÁGIO 2: Planejador DP + Coalescing S3 get_ranges + Zstd (.osv)       |
  |  * Programação Dinâmica (< 0.5 ms) funde intervalos contíguos            |
  |  * Disparo paralelo HTTP/2 get_ranges no S3/Object Storage               |
  |  * Validação page_xxh64 -> Descompressão Zstd (com DecoderDictionary)    |
  +--------------------------------------------------------------------------+
         |  (Minipáginas 16 KB descomprimidas em RAM)
         v
  +--------------------------------------------------------------------------+
  | ESTÁGIO 3: Refinamento Exato f16 + Zone Map Pruning (.osv)               |
  |  * Leitura direta do slice contíguo f16 na minipágina                    |
  |  * Produto interno exato f16/FMA contra q original                       |
  |  * Extração de payloads JSON apenas para sobreviventes                   |
  +--------------------------------------------------------------------------+
         |  (Top-R documentos candidatos para Late Interaction, ou Top-K final)
         v
  +--------------------------------------------------------------------------+
  | ESTÁGIO 4: Síntese LUT int8 + Kernel MaxSim vpshufb/vtbl (mmap .osr)     |
  |  * Síntese de LUT int8 de 16 entradas por subespaço/camada (< 8 us)      |
  |  * Lookup O(log N) em OsrDocEntry -> stream de tokens de 64 bytes        |
  |  * Decodificação nibble-packed via vpshufb/vtbl + MaxSim (< 3.8 ms)      |
  +--------------------------------------------------------------------------+
         |
         v
  [Top-K Resultados Finais Verificados]
```

### 1.2 Topologia de Crates e Grafo Estrito de Dependências

O workspace é particionado em cinco crates Apache-2.0 com dependências estritamente acíclicas e unidirecionais:

```text
                      +-----------------------------------+
                      |             ospreydb              |
                      |  Binário unificado `osprey`:      |
                      |  WAL contínua, LSM Delta Buffer,  |
                      |  CoW Compactor, S3 manifest CAS,  |
                      |  HTTP (:6333) & Arrow Flight      |
                      +-----------------+-----------------+
                                        |
                      +-----------------v-----------------+
                      |            osprey-cli             |
                      |  Subcomandos CLI (import, search, |
                      |  rerank, delete, inspect, bench,  |
                      |  doctor, completions)             |
                      +-----------------+-----------------+
                                        |
                      +-----------------v-----------------+
                      |          osprey-storage           |
                      |  memmap2, S3 get_ranges, Zstd,    |
                      |  RoaringBitmap, Codec .osi/.osv/  |
                      |  .osr, Manifesto de Segmentos     |
                      +-----------------+-----------------+
                                        |
                      +-----------------v-----------------+
                      |           osprey-simd             |
                      |  Dispatch AVX-512/AVX2/NEON/Scalar|
                      |  VPOPCNTDQ, vpshufb/vtbl, FMA f16 |
                      +-----------------+-----------------+
                                        |
                      +-----------------v-----------------+
                      |           osprey-core             |
                      |  Layouts #[repr(C, align(64))],   |
                      |  FWHT, Binary K-Means, RaBitQ,    |
                      |  xxHash64, Traits de Extensão     |
                      +-----------------------------------+
```

#### Responsabilidades por Crate

1. **`crates/osprey-core`** (Núcleo puro, sem I/O de rede ou sistema de arquivos):
   - Definições binárias canônicas (`OsiHeader`, `OsvHeader`, `OsrHeader`, `MinipageDirectoryEntry`, `OsrDocEntry`) com derivação estrita `bytemuck::Pod` e `bytemuck::Zeroable`.
   - Transformada rápida de Walsh-Hadamard (`FwhtTransform`), quantização `RaBitQ` assimétrica, treinamento `1-Bit Binary K-Means`, heurística TSP para reordenação de centróides/slabs e quantizador em reticulado aninhado (*Nested Lattice*, $\gamma = 0.28$).
   - Primitivas de coalescimento DP de ranges para intervalos de bytes.
   - Primitivas de integridade (`xxHash64`) e definição dos traits de extensão (`DeltaBufferExt` / `StorageEngineExt`).

2. **`crates/osprey-simd`** (Depende apenas de `osprey-core`):
   - Despacho dinâmico em tempo de execução via `std::arch::is_x86_feature_detected!` (`avx512vpopcntdq`, `avx512bw`, `avx512f`, `avx2`, `fma`, `f16c`) e `std::arch::is_aarch64_feature_detected!` (`neon`, `fp16`), com fallback escalar portátil bit-a-bit equivalente.
   - Kernels quentes: `vpopcntdq_rabitq_scan`, `f16_exact_dot`, `lut16_maxsim_vpshufb` / `lut16_maxsim_vtbl` e FWHT vetorizado.

3. **`crates/osprey-storage`** (Depende de `osprey-core` e `osprey-simd`):
   - Leitores zero-copy baseados em `memmap2` para `.osi` e `.osr` com validação de alinhamento e checksum na abertura.
   - Cliente assíncrono de object storage (`object_store` / S3) executando `get_ranges` em paralelo a partir do plano emitido pelo planejador DP (`range_planner`).
   - Pipeline de escrita/leitura Zstd com suporte a `DecoderDictionary` / `EncoderDictionary` e gerenciamento de `RoaringBitmap` de tombstones.

4. **`crates/osprey-cli`** (Depende de `osprey-core`, `osprey-simd` e `osprey-storage`):
   - Biblioteca de comandos CLI para ingestão Parquet/JSONL/NPY (`import`), busca (`search`), reranking (`rerank`), deleção (`delete`), auditoria física (`inspect`), diagnóstico (`doctor`) e benchmarks (`bench`).

5. **`crates/ospreydb`** (Depende de `osprey-core`, `osprey-simd`, `osprey-storage` e `osprey-cli`):
   - Binário unificado **`osprey`** (100% Apache-2.0) que expõe todos os subcomandos CLI e o servidor de produção (`osprey serve`) com WAL durável, `LsmDeltaBuffer`, `CowSlabCompactor`, `TombstoneGarbageCollector`, coordenação S3 `manifest.json` CAS, Qdrant/Cohere/OpenAI HTTP API (`:6333`), Prometheus `/metrics` e Apache Arrow Flight (`:8815`).

### 1.3 Traits de Extensão para Escrita Contínua (`osprey_core::traits`)

A separação modular entre leitura imutável em segmentos (`.osi`/`.osv`/`.osr`) e camada de escrita contínua/compactação (`ospreydb`) ocorre via trait público definido em `osprey-core`:

```rust
/// Ponto de extensão para escrita contínua, WAL, delta-segments e compactação
/// em background fornecidos pelo servidor `osprey` (`crates/ospreydb`).
pub trait DeltaBufferExt: Send + Sync {
    /// Tipo de erro específico da camada estendida, convertível para diagnóstico do motor.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Anexa um lote mutável ao buffer ativo / WAL antes da materialização em segmentos
    /// imutáveis `.osi`/`.osv`/`.osr`.
    fn append_batch(&self, records: &[RawVectorRecord<'_>]) -> Result<u64, Self::Error>;

    /// Executa busca combinada nos segmentos imutáveis e nos delta-buffers em memória.
    fn search_deltas(
        &self,
        q_rot: &[f32],
        top_k: usize,
    ) -> Result<Vec<Stage1Candidate>, Self::Error>;

    /// Dispara compactação incremental mesclando tombstones e delta-segments em um
    /// novo conjunto canônico `.osi`/`.osv`/`.osr`.
    fn compact(&self, segment_ids: &[u64]) -> Result<CompactionReport, Self::Error>;
}
```

---

## 2. Invariantes de Hardware, Memória e Alinhamento

Todas as estruturas binárias persistidas em `.osi`, `.osv` e `.osr` obedecem a invariantes rígidas para garantir mapeamento direto de memória (`memmap2`) sem deserialização ou alocação de heap no caminho crítico.

### 2.1 Alinhamento de Linha de Cache de 64 Bytes (`#[repr(C, align(64))]`)

1. **Estruturas de Cabeçalho e Seções:**
   - Todos os cabeçalhos globais (`OsiHeader`, `OsvHeader`, `OsrHeader`) possuem exatamente **64 bytes** e são anotados com `#[repr(C, align(64))]`.
   - Toda seção tabular principal (`centroids_offset`, `entries_offset`, `tombstones_offset`, `zstd_dict_offset`, `directory_offset`, `doc_table_offset`, `tensors_offset`) **deve** iniciar em um deslocamento múltiplo exato de **64 bytes** (`offset % 64 == 0`), preenchido com bytes `0x00` entre seções quando necessário.
2. **Compatibilidade `memmap2` e `bytemuck::Pod`:**
   - Como páginas do kernel Linux/macOS são alinhadas em $\ge 4096$ bytes, o endereço base retornado por `Mmap::as_ptr()` é garantidamente alinhado a 64 bytes. Ainda assim, `osprey-storage` executa `assert_eq!(ptr as usize % 64, 0)` na abertura de todo segmento.
   - Todas as structs mapeadas implementam `bytemuck::Pod` e `bytemuck::Zeroable`: **é estritamente proibido** qualquer padding implícito do compilador. Todo preenchimento estrutural é declarado explicitamente via campos `_reserved` / `_pad` inicializados com `0x00`, garantindo invariantes determinísticas de checksum e *zero UB* (Undefined Behavior).
   - Verificações estáticas `const _: () = assert!(std::mem::size_of::<T>() == ...)` e `assert!(std::mem::align_of::<T>() == 64)` são obrigatórias em `osprey-core`.
3. **Cargas Alinhadas AVX-512 (`vmovdqa64`):**
   - Em x86_64 com AVX-512, instruções de carga alinhada (`_mm512_load_si512` -> `vmovdqa64`) exigem ponteiro múltiplo de 64 bytes (caso contrário disparam `#GP` fault).
   - Para blocos de 1-bit `RaBitQ` (`[u64; padded_dim / 64]`), quando `padded_dim >= 512`, cada vetor de código possui $\ge 64$ bytes; combinando `entries_offset % 64 == 0` e alinhamento de sub-blocos de cluster em 64 bytes, as varreduras sequenciais consomem `vmovdqa64` diretamente sobre o `mmap` sem cópias intermediárias.
   - Em `.osr`, cada registro de token possui exatamente **64 bytes** e `tensors_offset % 64 == 0`, garantindo que **100% dos tokens** caiam exatamente em uma única linha de cache e sejam lidos com `vmovdqa64` sem *cache-line splits*.

### 2.2 Convenção Little-Endian, Magic Bytes e Protocolo `xxHash64`

| Invariante | Especificação Canônica | Regra de Validação |
| :--- | :--- | :--- |
| **Endianness** | **Little-Endian (LE)** estrito para todos os tipos (`u16`, `u32`, `u64`, `f16` bits, `f32`). | Guard de compilação `#[cfg(target_endian = "little")]` em `osprey-core`. |
| **Magic Bytes Globais** | `"OSPR"` = `[0x4F, 0x53, 0x50, 0x52]` (`0x5250534F` LE) nos bytes `0x00..0x03`. | Rejeição imediata com `InvalidMagic` se diferir. |
| **Discriminante de Formato** | Byte `0x04` (`format_kind: u8`): `0x01` (`.osi`), `0x02` (`.osv`), `0x03` (`.osr`). | Rejeição imediata se o parser receber artefato de tipo divergente. |
| **Integridade de Cabeçalho** | `header_xxh64: u64` nos bytes `0x38..0x3F`, calculado via `xxh64(&header[0..56], 0)` (seed `0`). | Validado **antes** de ler qualquer offset ou alocar qualquer buffer derivado do cabeçalho. |
| **Integridade de Dados** | `.osi`: `file_xxh64` nos últimos 8 bytes do arquivo (seed `0`).<br>`.osv`: `page_xxh64` por minipágina sobre o frame comprimido em `MinipageDirectoryEntry`.<br>`.osr`: `header_xxh64` + `file_xxh64` opcional no rodapé de 8 bytes após os tensores. | `.osv` valida `page_xxh64` **antes** da descompressão Zstd para bloquear corrupção de rede/S3 antes do parser. |

---

## 3. Especificação Física Byte-a-Byte do Formato `.osi` (Osprey Resident 1-Bit Index)

O arquivo `.osi` é o índice residente em RAM/mmap. Ele concentra exclusivamente as estruturas necessárias ao Estágio 1 (seleção de clusters, scan assimétrico `RaBitQ` de 1-bit e filtragem de tombstones).

### 3.1 Layout Global do Arquivo `.osi`

```text
+-----------------------------------------------------------------------+ 0x00
| OsiHeader (64 bytes, #[repr(C, align(64))])                           |
+-----------------------------------------------------------------------+ centroids_offset (align 64)
| Tabela de Centróides (K * padded_dim * 2 bytes, f16 LE, ordem TSP)    |
| + zero-padding até múltiplo de 64 bytes                               |
+-----------------------------------------------------------------------+ entries_offset (align 64)
| Cluster Span Index (K * 16 bytes -> offset relativo + count)          |
| + zero-padding até múltiplo de 64 bytes                               |
+-----------------------------------------------------------------------+
| Clusters 0 .. K-1 (Agrupados por Cluster em ordem TSP):               |
|   Para cada cluster c (alinhado a 64 bytes):                          |
|     1. Blocos de Sinais 1-Bit: M_c * [u64; padded_dim / 64]           |
|        (+ padding até 64 bytes)                                       |
|     2. Fatores RaBitQ: M_c * (res_norm: f16, l1_norm_factor: f16)     |
|        (+ padding até 64 bytes)                                       |
|     3. Locators: M_c * (doc_id: u64, minipage_id: u32, slot_idx: u16, |
|                         flags: u16)                                   |
+-----------------------------------------------------------------------+ tombstones_offset (align 64)
| Tombstones: tombstone_len (u64) + RoaringBitmap serializado (portable)|
| + zero-padding até alinhamento de 8 bytes                             |
+-----------------------------------------------------------------------+ EOF - 8
| file_xxh64: u64 (checksum global dos bytes [0 .. EOF - 8], seed = 0)  |
+-----------------------------------------------------------------------+ EOF
```

### 3.2 `OsiHeader` (64 bytes, `0x00..0x3F`)

| Offset | Tamanho | Campo | Tipo Rust | Descrição e Invariantes |
| :--- | :---: | :--- | :--- | :--- |
| `0x00..0x03` | 4 B | `magic` | `[u8; 4]` | Magic global `"OSPR"` (`[0x4F, 0x53, 0x50, 0x52]`). |
| `0x04` | 1 B | `format_kind` | `u8` | Identificador fixo `0x01` (`.osi`). |
| `0x05` | 1 B | `flags` | `u8` | Bitflags do índice (`0x00` na v1; bit 0 reservado para `COSINE_NORMALIZED`). |
| `0x06..0x07` | 2 B | `version` | `u16` | Versão do layout físico (`1`). |
| `0x08..0x09` | 2 B | `orig_dim` | `u16` | Dimensionalidade original $d_{\text{orig}}$ dos vetores de entrada. |
| `0x0A..0x0B` | 2 B | `padded_dim` | `u16` | Dimensão acolchoada para FWHT: `next_power_of_two(max(orig_dim, 64))` (`padded_dim % 64 == 0`). |
| `0x0C..0x0F` | 4 B | `num_clusters` | `u32` | Número total de clusters $K$ (`1-Bit Binary K-Means`). |
| `0x10..0x17` | 8 B | `num_vectors` | `u64` | Quantidade total de vetores $N$ indexados no segmento. |
| `0x18..0x1F` | 8 B | `fwht_seed` | `u64` | Seed determinística para os sinais aleatórios de Rademacher $D \in \{\pm 1\}^{d_{\text{pad}}}$ da FWHT. |
| `0x20..0x27` | 8 B | `centroids_offset` | `u64` | Offset absoluto do início da Tabela de Centróides (`offset % 64 == 0`). |
| `0x28..0x2F` | 8 B | `entries_offset` | `u64` | Offset absoluto do diretório `RaBitQ` agrupado por cluster (`offset % 64 == 0`). |
| `0x30..0x37` | 8 B | `tombstones_offset` | `u64` | Offset absoluto do bloco `RoaringBitmap` de tombstones (`offset % 64 == 0`). |
| `0x38..0x3F` | 8 B | `header_xxh64` | `u64` | `xxHash64` sobre os 56 bytes `[0x00..0x37]` com seed `0`. |

### 3.3 Tabela de Centróides (Ordenada por Circuito TSP)

- Localizada em `centroids_offset` (`% 64 == 0`).
- Dimensão lógica: matriz densa row-major `[K][padded_dim]` de escalares IEEE-754 `f16` (2 bytes cada, armazenados no espaço rotacionado pela FWHT).
- Tamanho bruto: `K * padded_dim * 2` bytes, seguido de padding de bytes `0x00` até o próximo múltiplo de 64 bytes.
- **Ordenação Física TSP:** O índice `cluster_id = 0..K-1` segue estritamente a permutação calculada pelo solucionador TSP sobre os centróides no espaço rotacionado. Isso garante que clusters frequentemente co-sondados (*co-probed*) por `nprobe` residam adjacentes tanto na Tabela de Centróides quanto nas minipáginas do `.osv`.

### 3.4 Diretório `RaBitQ` Assimétrico e Prevenção de *Cache Thrashing*

Para cada vetor indexado no cluster $c$, o formato define o conjunto completo de metadados de 16 bytes + fator de escala e seu código binário de 1-bit:

- **Campos lógicos por vetor:**
  - `doc_id: u64` (8 B): identificador lógico global do documento.
  - `minipage_id: u32` (4 B): índice da minipágina alvo no `.osv`.
  - `slot_idx: u16` (2 B): índice do vetor dentro da minipágina (0-based).
  - `res_norm: f16` (2 B): norma euclidiana do vetor residual $\|x - \mu_c\|_2$ no espaço rotacionado.
  - `l1_norm_factor: f16` (2 B): fator de correção assimétrico $\langle \bar{o}, o \rangle$ do código `RaBitQ` ($\|o_u\|_1 / \sqrt{d_{\text{pad}}}$ normalizado para reconstrução não-enviesada do produto interno).
  - **Código de 1-Bit:** `[u64; padded_dim / 64]` ($d_{\text{pad}} / 8$ bytes por vetor), contendo os sinais binarizados $\text{sign}(\tilde{x} - \tilde{\mu}_c) \in \{0, 1\}^{d_{\text{pad}}}$.

> **Nota Crítica de ABI e Layout Físico (SoA Agrupado por Cluster):**
> A soma bruta dos 5 campos escalares (`doc_id: u64` + `minipage_id: u32` + `slot_idx: u16` + `res_norm: f16` + `l1_norm_factor: f16`) totaliza 18 bytes (ou 24 bytes sob alinhamento `#[repr(C)]` de `u64`). Ademais, intercalar metadados de identificador (`doc_id`, `minipage_id`) entre blocos `[u64; padded_dim / 64]` poluiria as linhas de cache durante o loop SIMD `VPOPCNTDQ` (*cache thrashing*), já que `doc_id` e `minipage_id` só são necessários para os $< 1\%$ de vetores que sobrevivem ao limiar de distância.
>
> Portanto, cada cluster $c$ dentro de `entries_offset` utiliza **agrupamento colunar por cluster (Cluster-SoA)** com sub-blocos alinhados a 64 bytes:
> 1. **Cluster Span Index (`entries_offset`):** Array de $K$ descritores `OsiClusterSpan` (16 bytes cada: `cluster_base_offset: u64`, `vector_count: u32`, `flags: u32`) permitindo salto $O(1)$ para qualquer cluster sondado.
> 2. **Stream 1 — Blocos de Sinais 1-Bit (Hot Loop):** Vetor contíguo `M_c * [u64; padded_dim / 64]` alinhado a 64 bytes. O kernel `VPOPCNTDQ` varre sequencialmente com prefetcher de hardware a 100% de eficiência de linha de cache.
> 3. **Stream 2 — Fatores `RaBitQ` (Hot Loop Pós-POPCNT):** Vetor contíguo `M_c * OsiRabitqFactors` (`res_norm: f16`, `l1_norm_factor: f16`, **4 bytes** exatos por vetor, `#[repr(C)]`).
> 4. **Stream 3 — Locator de 16 Bytes (Cold / Top-C Only):** Vetor contíguo `M_c * OsiVectorLocator` (**16 bytes exatos**, `#[repr(C)]`, zero padding implícito):
>    - `0x00..0x07` (8 B): `doc_id: u64`
>    - `0x08..0x0B` (4 B): `minipage_id: u32`
>    - `0x0C..0x0D` (2 B): `slot_idx: u16`
>    - `0x0E..0x0F` (2 B): `entry_flags: u16` (`0x0000` padrão; reservado para flags de filtragem inline)

### 3.5 Rodapé `.osi` (Tombstones + Checksum Global)

1. **`tombstones_offset` (`% 64 == 0`):**
   - `tombstone_len: u64` (`0x00..0x07` relativo): tamanho em bytes do payload `RoaringBitmap`. Se `0`, não há tombstones ativos.
   - `bitmap_bytes: [u8; tombstone_len]`: formato portátil padrão `RoaringBitmap::serialize_into` contendo os `doc_id` (ou índices locais de segmento) logicamente deletados.
   - Padding de bytes `0x00` até alinhamento de 8 bytes.
2. **Checksum Global (`EOF - 8 .. EOF`):**
   - `file_xxh64: u64` calculado sobre todo o intervalo `[0 .. file_len - 8]` com seed `0`.

---

## 4. Especificação Física Byte-a-Byte do Formato `.osv` (Osprey Locality-Reordered Vector Slabs)

O arquivo `.osv` armazena os vetores exatos em `f16`, metadados de *zone maps* e payloads JSON completos, organizados em minipáginas de ~16 KB ordenadas por localidade TSP e otimizadas para leituras paralelas `get_ranges` em Object Storage (S3/GCS/MinIO).

### 4.1 Layout Global do Arquivo `.osv`

```text
+-----------------------------------------------------------------------+ 0x00
| OsvHeader (64 bytes, #[repr(C, align(64))])                           |
+-----------------------------------------------------------------------+ zstd_dict_offset (align 64)
| [Opcional] Dicionário Zstd Treinado (zstd_dict_len bytes + pad 64B)   |
+-----------------------------------------------------------------------+ slabs_offset (align 64)
| Minipágina Comprimida #0 (Zstd Frame, ~16 KB descomprimida)           |
| Minipágina Comprimida #1 (Zstd Frame, ~16 KB descomprimida)           |
| ...                                                                   |
| Minipágina Comprimida #(num_minipages - 1) + pad 64B                  |
+-----------------------------------------------------------------------+ directory_offset (align 64)
| Diretório de Minipáginas: num_minipages * MinipageDirectoryEntry (48B)|
+-----------------------------------------------------------------------+ EOF
```

### 4.2 `OsvHeader` (64 bytes, `0x00..0x3F`)

| Offset | Tamanho | Campo | Tipo Rust | Descrição e Invariantes |
| :--- | :---: | :--- | :--- | :--- |
| `0x00..0x03` | 4 B | `magic` | `[u8; 4]` | Magic global `"OSPR"` (`[0x4F, 0x53, 0x50, 0x52]`). |
| `0x04` | 1 B | `format_kind` | `u8` | Identificador fixo `0x02` (`.osv`). |
| `0x05` | 1 B | `compression_flags` | `u8` | Bit 0: `ZSTD_ENABLED` (`0x01`); Bit 1: `HAS_ZSTD_DICT` (`0x02`); Bit 2: `ZSTD_CONTENT_CHECKSUM` (`0x04`); Bits 3..7: `0`. |
| `0x06..0x07` | 2 B | `version` | `u16` | Versão do layout físico (`1`). |
| `0x08..0x0B` | 4 B | `num_minipages` | `u32` | Número total de minipáginas indexadas no diretório. |
| `0x0C..0x0D` | 2 B | `orig_dim` | `u16` | Dimensionalidade exata $d_{\text{orig}}$ dos vetores `f16` nas minipáginas. |
| `0x0E..0x0F` | 2 B | `reserved_flags` | `u16` | Reservado (`0x0000`). |
| `0x10..0x17` | 8 B | `zstd_dict_offset` | `u64` | Offset absoluto do dicionário Zstd (`0` se `HAS_ZSTD_DICT == 0`; senão `% 64 == 0`). |
| `0x18..0x1B` | 4 B | `zstd_dict_len` | `u32` | Tamanho em bytes do dicionário Zstd (`0` se ausente). |
| `0x1C..0x1F` | 4 B | `zstd_dict_id` | `u32` | ID numérico do dicionário embutido no header Zstd (`0` se ausente). |
| `0x20..0x27` | 8 B | `directory_offset` | `u64` | Offset absoluto do array de `MinipageDirectoryEntry` (`% 64 == 0`). |
| `0x28..0x2F` | 8 B | `num_vectors` | `u64` | Quantidade total de vetores armazenados em todas as minipáginas. |
| `0x30..0x37` | 8 B | `reserved` | `[u8; 8]` | Zeros explícitos (`[0; 8]`). |
| `0x38..0x3F` | 8 B | `header_xxh64` | `u64` | `xxHash64` sobre os 56 bytes `[0x00..0x37]` com seed `0`. |

### 4.3 Dicionário Zstd Opcional

- Quando `HAS_ZSTD_DICT` está ativo, `osprey-storage` treina um dicionário Zstd (`zstd::dict::from_samples`, tamanho alvo 64 KiB – 112 KiB) sobre amostras dos payloads JSON e estruturas das minipáginas do segmento.
- No início da consulta ao segmento, o cliente S3 lê apenas os primeiros bytes (ou o range `[zstd_dict_offset .. zstd_dict_offset + zstd_dict_len]`) e o diretório `[directory_offset .. EOF]`, mantendo `Arc<DecoderDictionary>` residente em RAM para descomprimir minipáginas de 16 KB em ~15–40 µs com alta taxa de compressão.

### 4.4 Layout Interno das Minipáginas de 16 KB (Pós-Descompressão Zstd)

Cada minipágina agrupa vetores pertencentes ao mesmo `cluster_id` (respeitando o limite alvo descomprimido de **16.384 bytes**, salvo quando um único payload JSON exceder esse teto). Após a descompressão Zstd, o buffer linear obedece ao layout abaixo:

```text
+-----------------------------------------------------------------------+ 0x00
| MinipageHeader (16 bytes, Little-Endian)                              |
|   0x00..0x03 (4B): cluster_id: u32                                    |
|   0x04..0x05 (2B): vector_count: u16 (M_p)                            |
|   0x06..0x07 (2B): orig_dim: u16 (d)                                  |
|   0x08..0x0B (4B): zone_map_offset: u32 (offset relativo na página)   |
|   0x0C..0x0F (4B): payloads_offset: u32 (offset relativo na página)   |
+-----------------------------------------------------------------------+ 0x10
| Array Contíguo de Vetores Exatos em f16                               |
|   M_p * d * 2 bytes (row-major, slot 0 .. M_p - 1, f16 LE)            |
|   + padding de 0..6 bytes para alinhamento de 8 bytes                 |
+-----------------------------------------------------------------------+ zone_map_offset
| Metadados Escalares para Zone Maps                                    |
|   doc_id_min: u64, doc_id_max: u64                                    |
|   num_zone_entries: u16, flags: u16, reserved: u32                    |
|   Entradas tipadas (min/max numéricos i64/f64 e Bloom de tags)        |
+-----------------------------------------------------------------------+ payloads_offset
| Tabela de Offsets de Payloads: [u32; M_p] (relativos ao início JSON)  |
| Sequência de Payloads JSON com Prefixo de Tamanho:                    |
|   Para slot i = 0 .. M_p - 1:                                         |
|     payload_len: u32                                                  |
|     json_bytes: [u8; payload_len] (UTF-8 válido)                      |
+-----------------------------------------------------------------------+ uncompressed_size
```

- **Isolamento do Hot-Path f16:** O array contíguo de vetores `f16` inicia imediatamente em `0x10`, permitindo que o Estágio 3 calcule produtos internos exatos diretamente via ponteiro de slot `0x10 + slot_idx * orig_dim * 2` sem percorrer ou alocar strings JSON.

### 4.5 Diretório de Minipáginas (`MinipageDirectoryEntry`, 48 bytes)

O diretório reside em `directory_offset` como um array denso de `num_minipages` entradas `#[repr(C)]` de **48 bytes exatos** (alinhamento 8 bytes):

| Offset | Tamanho | Campo | Tipo Rust | Descrição e Invariantes |
| :--- | :---: | :--- | :--- | :--- |
| `0x00..0x07` | 8 B | `byte_offset` | `u64` | Offset absoluto do frame Zstd comprimido no `.osv`. |
| `0x08..0x0F` | 8 B | `min_doc_id` | `u64` | Menor `doc_id` armazenado na minipágina (*zone map*). |
| `0x10..0x17` | 8 B | `max_doc_id` | `u64` | Maior `doc_id` armazenado na minipágina (*zone map*). |
| `0x18..0x1F` | 8 B | `page_xxh64` | `u64` | Checksum `xxHash64` (seed `0`) sobre os `compressed_size` bytes em `[byte_offset .. byte_offset + compressed_size]`. |
| `0x20..0x23` | 4 B | `compressed_size` | `u32` | Tamanho em bytes do payload comprimido a ser requisitado via S3 `Range`. |
| `0x24..0x27` | 4 B | `uncompressed_size` | `u32` | Tamanho exato pós-descompressão para pré-alocação de buffer exato sem realocações. |
| `0x28..0x2B` | 4 B | `cluster_id` | `u32` | Cluster TSP ao qual a minipágina pertence. |
| `0x2C..0x2D` | 2 B | `vector_count` | `u16` | Quantidade de slots ($M_p$) contidos na minipágina. |
| `0x2E..0x2F` | 2 B | `flags` | `u16` | Bit 0: `HAS_ZONE_MAP`; Bit 1: `IS_OVERSIZED_PAGE`; Bits 2..15: `0`. |

---

## 5. Especificação Física Byte-a-Byte do Formato `.osr` (Osprey SIMD-Interleaved Token Rerank Store)

O arquivo `.osr` armazena as representações comprimidas de tokens multi-vetor (ColBERT / *late interaction*) quantizadas por **Reticulado Aninhado de 2 Camadas (Nested Lattice Quantization)** e empacotadas em nibbles prontos para decodificação SIMD sem desempacotamento em memória.

### 5.1 Layout Global do Arquivo `.osr`

```text
+-----------------------------------------------------------------------+ 0x00
| OsrHeader (64 bytes, #[repr(C, align(64))])                           |
+-----------------------------------------------------------------------+ doc_table_offset (align 64)
| Tabela de Offsets de Documentos: num_docs * OsrDocEntry (24 bytes)    |
| (Ordenada estritamente por doc_id crescente para busca binária)       |
| + zero-padding até múltiplo de 64 bytes                               |
+-----------------------------------------------------------------------+ tensors_offset (align 64)
| Tensores de Tokens Intercalados (total_tokens * 64 bytes):            |
|   Token #0 (64 bytes, align 64): Camada 1 (32B) | Camada 2 (32B)      |
|   Token #1 (64 bytes, align 64): Camada 1 (32B) | Camada 2 (32B)      |
|   ...                                                                 |
+-----------------------------------------------------------------------+ EOF - 8
| file_xxh64: u64 (checksum global dos bytes [0 .. EOF - 8], seed = 0)  |
+-----------------------------------------------------------------------+ EOF
```

### 5.2 `OsrHeader` (64 bytes, `0x00..0x3F`)

| Offset | Tamanho | Campo | Tipo Rust | Descrição e Invariantes |
| :--- | :---: | :--- | :--- | :--- |
| `0x00..0x03` | 4 B | `magic` | `[u8; 4]` | Magic global `"OSPR"` (`[0x4F, 0x53, 0x50, 0x52]`). |
| `0x04` | 1 B | `format_kind` | `u8` | Identificador fixo `0x03` (`.osr`). |
| `0x05` | 1 B | `subspace_dim` | `u8` | Dimensão de cada subespaço do reticulado (`2` por padrão). |
| `0x06` | 1 B | `num_layers` | `u8` | Número de camadas aninhadas do reticulado (`2` por padrão). |
| `0x07` | 1 B | `flags` | `u8` | Bitflags de formato (`0x00` na v1). |
| `0x08..0x09` | 2 B | `token_dim` | `u16` | Dimensionalidade dos embeddings de token (`128` por padrão). |
| `0x0A..0x0B` | 2 B | `version` | `u16` | Versão do layout físico (`1`). |
| `0x0C..0x0F` | 4 B | `gamma` | `f32` | Fator de contração de escala da Camada 2 do reticulado (`0.28` por padrão). |
| `0x10..0x17` | 8 B | `fwht_seed` | `u64` | Seed determinística para rotação FWHT dos tokens (deve corresponder à query). |
| `0x18..0x1F` | 8 B | `num_docs` | `u64` | Quantidade de documentos com tokens no `.osr`. |
| `0x20..0x27` | 8 B | `total_tokens` | `u64` | Soma total de tokens armazenados em `tensors_offset`. |
| `0x28..0x2F` | 8 B | `doc_table_offset` | `u64` | Offset absoluto da Tabela de Offsets de Documentos (`% 64 == 0`). |
| `0x30..0x37` | 8 B | `tensors_offset` | `u64` | Offset absoluto dos tensores intercalados de 64 bytes (`% 64 == 0`). |
| `0x38..0x3F` | 8 B | `header_xxh64` | `u64` | `xxHash64` sobre os 56 bytes `[0x00..0x37]` com seed `0`. |

### 5.3 Tabela de Offsets de Documentos (`OsrDocEntry`, 24 bytes)

Localizada em `doc_table_offset`, ordenada estritamente por `doc_id` ascendente para lookup $O(\log N_{\text{docs}})$ via `binary_search_by_key`:

| Offset | Tamanho | Campo | Tipo Rust | Descrição |
| :--- | :---: | :--- | :--- | :--- |
| `0x00..0x07` | 8 B | `doc_id` | `u64` | Identificador lógico do documento (correlacionado com `.osi`). |
| `0x08..0x0F` | 8 B | `token_start_idx` | `u64` | Índice inicial do primeiro token do documento em `tensors_offset` (byte offset = `tensors_offset + token_start_idx * 64`). |
| `0x10..0x11` | 2 B | `token_count` | `u16` | Quantidade de tokens contíguos pertencentes ao documento. |
| `0x12..0x17` | 6 B | `reserved` | `[u8; 6]` | Zeros explícitos (`[0; 6]`) garantindo 24 bytes e alinhamento `u64`. |

### 5.4 Layout Intercalado de Nibbles por Token (64 Bytes por Token de 128 Dims)

Com `token_dim = 128` e `subspace_dim = 2`, cada token rotacionado pela FWHT possui $S = 128 / 2 = 64$ subespaços 2D. Cada subespaço $s \in [0, 63]$ é quantizado em duas camadas aninhadas com 16 centróides de Voronoi ($4$ bits cada, $c \in [0, 15]$):
- **Camada 1 (Grossa):** índice $c^{(1)}_s \in [0, 15]$ do centróide base no reticulado 2D.
- **Camada 2 (Residual Fino):** o erro residual $r_s = \tilde{t}_s - \Lambda(c^{(1)}_s)$ é reescalado por $1 / \gamma$ ($\gamma = 0.28$) e quantizado novamente para $c^{(2)}_s \in [0, 15]$.

Dois subespaços consecutivos (`even = 2k`, `odd = 2k + 1`, para $k = 0 \dots 31$) são empacotados em um único byte:

```text
  Bit:    7   6   5   4   3   2   1   0
        +---------------+---------------+
        |  c_odd (4b)   |  c_even (4b)  |   byte[k] = (c_even & 0x0F) | ((c_odd & 0x0F) << 4)
        +---------------+---------------+
```

Dessa forma, um token completo ocupa exatamente **64 bytes = 1 linha de cache** (`#[repr(C, align(64))]`):

```text
Byte Offset no Token:
0x00 .. 0x1F (32 bytes): Camada 1 — 32 bytes para os 64 subespaços (0..63)
                         byte[k] empacota subespaço 2k (low nibble) e 2k+1 (high nibble)
0x20 .. 0x3F (32 bytes): Camada 2 — 32 bytes para os 64 subespaços residuais (0..63, escala gamma = 0.28)
                         byte[32 + k] empacota subespaço 2k (low nibble) e 2k+1 (high nibble)
```

Esse arranjo permite carregar um token inteiro em um registrador `ZMM` (`vmovdqa64`) ou dois `YMM` (`vmovdqa`), isolar nibbles pares (`vpandd & 0x0F`) e ímpares (`vpsrld >> 4`), e executar lookups simultâneos em LUTs de 16 entradas via `vpshufb` (x86_64) ou `vtbl` (`vqtbl1q_s8` em aarch64 NEON) sem qualquer branch ou descompressão escalar.

---

## 6. Anatomia de Execução da Query e Orçamento de Latência

A execução de ponta a ponta é particionada em estágios determinísticos que minimizam latência de cauda ($p_{99}$) em CPU e o número de *round-trips* em Object Storage.

### 6.1 Passo a Passo Cronometrado

1. **Rotação FWHT da Query ($O(d \log d)$):**
   - A query $q \in \mathbb{R}^{d_{\text{orig}}}$ recebe zero-padding até $d_{\text{pad}}$, multiplicação sinal-a-sinal pelo vetor pseudoaleatório de Rademacher $D(\text{fwht\_seed})$ e $\log_2(d_{\text{pad}})$ estágios borboleta (*butterfly*) Walsh-Hadamard com normalização $1 / \sqrt{d_{\text{pad}}}$.
   - **Custo:** $< 10\ \mu\text{s}$ para $d_{\text{pad}} \le 2048$.

2. **Transposição da Query em 4 Planos de Bits (`int4`):**
   - Cada coordenada rotacionada $\tilde{q}_i$ é normalizada para o intervalo de quantização uniforme de 4 bits sem sinal $u_i \in [0, 15]$ (armazenando escala $\Delta_q$ e offset $z_q$ globais da query).
   - Os $d_{\text{pad}}$ valores `int4` são transpostos para **4 planos de bits** contíguos $P_0, P_1, P_2, P_3$, cada um representado por `[u64; padded_dim / 64]`:
     $$P_b[w] = \bigvee_{j=0}^{63} \left(\left(\frac{u_{64w + j}}{2^b} \bmod 2\right) \ll j\right), \quad b \in \{0, 1, 2, 3\}$$
   - **Custo:** $< 5\ \mu\text{s}$.

3. **Scan SIMD `VPOPCNTDQ` nos Clusters Sondados (`.osi`):**
   - Selecionam-se os `nprobe` centróides mais próximos em ordem TSP via produto interno `f16` contra $\tilde{q}$.
   - Para cada cluster sondado, varre-se sequencialmente o Stream 1 (`[u64; padded_dim / 64]`) usando `_mm512_popcnt_epi64` (`VPOPCNTDQ`) sobre os 4 planos de bits:
     $$\langle u, b_v \rangle = \sum_{b=0}^{3} 2^b \cdot \text{popcnt}(P_b \mathbin{\&} b_v)$$
   - O produto interno aproximado assimétrico é reconstruído com `res_norm` e `l1_norm_factor` do Stream 2; `doc_id` presentes no `RoaringBitmap` de tombstones são descartados antes do heap Top-$C$.
   - **Oversampling adaptativo para baixa dimensionalidade:** `SearchOptions::resolve` (`crates/osprey-core/src/search.rs`) é a fonte única da política de dimensionamento. Quando `orig_dim <= 256` (ex.: SIFT1M, `d = 128`), `nprobe` e o número de candidatos do Estágio 1 são multiplicados por 2 e o pool de refinamento `f16` recebe um piso de 128 candidatos, recuperando `Recall@10 >= 0.95` sem onerar workloads de alta dimensionalidade. `--refine-multiplier 1` desativa integralmente a adaptação (inclusive o piso); `N > 1` força o multiplicador. `osprey search`, `osprey bench` e runners externos consomem o mesmo plano resolvido.
   - **Custo:** ~0.8 – 2.5 ms para 1M vetores (`nprobe` típico), limitado por largura de banda L3/DRAM.

4. **Solver de Programação Dinâmica de Ranges (< 0.5 ms):**
   - Os Top-$C$ candidatos sobreviventes mapeiam para um conjunto de `minipage_id` únicos no diretório `.osv`.
   - O planejador DP (`RangePlanner`) ordena os intervalos `(byte_offset, byte_offset + compressed_size)` e executa programação dinâmica linear sobre as lacunas (*gaps*) entre minipáginas consecutivas:
     - Se o custo de latência de um `GET` adicional exceder o custo de transferência dos bytes intermediários desperdiçados (`gap <= gap_coalesce_threshold_bytes`, ex.: 64 KB – 256 KB no S3), a DP funde os intervalos em uma única requisição `Range` coalescida, respeitando um limite máximo por span (`max_coalesced_range_bytes`, ex.: 4 MB).
   - Como os clusters foram reordenados por TSP, candidatos de clusters vizinhos naturalmente co-residem em janelas contíguas do `.osv`, reduzindo dezenas de leituras aleatórias a poucas requisições coalescidas.
   - **Orçamento estrito:** **$< 0.5\ \text{ms}$** em CPU.

5. **Disparo Paralelo `get_ranges` no S3 e Descompressão Zstd (`.osv`):**
   - O plano emitido pelo planejador DP é despachado concorrentemente via `object_store::ObjectStore::get_ranges` sobre pool HTTP/2 persistente.
   - Ao receber cada janela coalescida, cada fatia de minipágina tem seu `page_xxh64` validado imediatamente sobre os bytes comprimidos; em seguida, os frames Zstd (~16 KB descomprimidos) são descomprimidos em paralelo nas worker threads usando o `DecoderDictionary` em cache.
   - **Custo:** I/O de rede S3 (~12 – 35 ms cold S3; $< 0.2\ \text{ms}$ se em cache NVMe local) + ~0.3 ms de descompressão Zstd paralela.

6. **Refinamento Exato `f16` (`.osv`):**
   - Para cada candidato `(minipage_id, slot_idx)` e após eventual poda por *zone maps* escalares, o motor acessa o slice contíguo `f16` no offset `0x10 + slot_idx * orig_dim * 2` da minipágina descomprimida e calcula o produto interno exato em alta precisão (`f16` -> acumulador `f32` via AVX-512 / AVX2+F16C+FMA).
   - **Custo:** $< 0.4\ \text{ms}$ para centenas de candidatos exatos.

7. **Síntese da LUT `int8` de 16 Entradas (< 8 µs) e Kernel `MaxSim` via `vpshufb`/`vtbl` (< 3.8 ms) (`.osr`):**
   - Para os Top-$R$ documentos sobreviventes ao Estágio 3 (ex.: $R \in [64, 256]$):
     1. **Rotação FWHT dos tokens da query:** Cada um dos $m$ tokens da query ($d = 128$) é rotacionado com `OsrHeader::fwht_seed`.
     2. **Síntese das LUTs `int8` (< 8 µs):** Para cada token de query, subespaço 2D $s \in [0, 63]$ e camada $l \in \{1, 2\}$ (com peso $\gamma = 0.28$ na camada 2), pré-computam-se os 16 produtos internos contra os 16 centróides do reticulado, quantizados em `i8` com fator de escala por token/camada. Cada LUT de 16 bytes cabe exatamente na metade de um `XMM` (ou replicada nas duas metades de um `YMM`/`ZMM` para `vpshufb`).
     3. **Kernel `MaxSim` via `vpshufb` / `vtbl` (< 3.8 ms):** Para cada `doc_id` buscado em `OsrDocEntry`, o kernel lê os `token_count` registros contíguos de 64 bytes no `mmap` do `.osr`, decodifica simultaneamente os 64 subespaços das Camadas 1 e 2 via `_mm256_shuffle_epi8` / `_mm512_shuffle_epi8` (`vpshufb`) ou `vqtbl1q_s8` (`vtbl`), soma as contribuições em acumuladores `i16`/`i32` (evitando *saturation overflow*), reduz ao máximo por token de documento para cada token de query e soma os máximos:
        $$\text{MaxSim}(Q, D) = \sum_{i=1}^{m} \max_{j=1}^{|D|} \langle q_i, d_j \rangle$$

### 6.2 Resumo do Orçamento de Latência

| Estágio | Componente / Operação | Domínio | Orçamento Alvo ($p_{99}$) |
| :--- | :--- | :--- | :---: |
| **1a** | Rotação FWHT da Query ($O(d \log d)$) | CPU | $< 0.01\ \text{ms}$ ($< 10\ \mu\text{s}$) |
| **1b** | Transposição da Query para 4 Planos `int4` | CPU | $< 0.01\ \text{ms}$ ($< 5\ \mu\text{s}$) |
| **1c** | Scan SIMD `VPOPCNTDQ` + Filtragem Tombstones (`.osi`) | CPU / RAM mmap | $0.8 - 2.5\ \text{ms}$ |
| **2a** | Solver de Programação Dinâmica de Ranges | CPU | **$< 0.5\ \text{ms}$** |
| **2b** | S3 `get_ranges` paralelo (`.osv`) | Object Storage I/O | $12 - 35\ \text{ms}$ (S3) / $< 0.2\ \text{ms}$ (NVMe) |
| **2c** | Verificação `page_xxh64` + Descompressão Zstd 16 KB | CPU | $< 0.4\ \text{ms}$ |
| **3** | Refinamento Exato `f16` + Extração Payload | CPU | $< 0.4\ \text{ms}$ |
| **4a** | Síntese das LUTs `int8` de 16 entradas | CPU | **$< 0.008\ \text{ms}$ ($< 8\ \mu\text{s}$)** |
| **4b** | Kernel `MaxSim` via `vpshufb` / `vtbl` (`.osr`) | CPU / mmap | **$< 3.8\ \text{ms}$** |
| **Total CPU** | **Overhead Total de Computação (Excluindo RTT S3)** | **CPU** | **$< 7.6\ \text{ms}$** |

---

## 7. Comparison with Other Object-Storage & Disk ANN Architectures

| Architecture | Routing Plane | Object Storage I/O Pattern | Key Trade-off on S3 / Object Storage |
| :--- | :--- | :--- | :--- |
| **Graph-on-Disk (DiskANN / Vamana)** | Navigable graph adjacency lists on SSD / S3 | **$O(\log N)$ dependent sequential round-trips** | Each hop discovers the next vertex ID only after reading the previous neighbor list; when each S3 `Range-GET` incurs `15–50 ms` RTT, serial hops accumulate hundreds of milliseconds per query. |
| **IVF-PQ on Object Storage (LanceDB / FAISS PQ)** | Coarse centroids + Product Quantization codes | Inverted lists fetched from object storage prior to rerank | PQ suffers from **codebook collapse** on out-of-distribution embeddings and requires fetching inverted lists over the network before exact scoring. |
| **OspreyDB (`.osi` + `.osv` + `.osr`)** | **100% in-RAM `mmap` 1-bit `RaBitQ` + FWHT (`0` network hops)** | **Single parallel wave** of DP-coalesced HTTP `Range-GET`s (`RangePlanner`) | Resolves Stage-1 pruning entirely in RAM (`> 99%` pruned locally without S3 calls); `RaBitQ + FWHT` provides an unbiased inner-product estimator with theoretical angular error bounds and no sub-codebook training, followed by one coalesced parallel wave of `.osv` fetches. |
