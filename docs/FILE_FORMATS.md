# OspreyDB — Guia Visual dos Formatos Binários (`FILE_FORMATS.md`)

> **Escopo:** Guia visual byte-a-byte dos três formatos imutáveis (`.osi`, `.osv`, `.osr`) implementado em [`crates/osprey-core/src/format/`](../crates/osprey-core/src/format/) e [`crates/osprey-storage/src/osv_slab.rs`](../crates/osprey-storage/src/osv_slab.rs), alinhado à especificação canônica [`ARCHITECTURE.md`](../ARCHITECTURE.md).

---

## 1. Invariantes Globais de Hardware e Zero-Copy

1. **Endianness & Magic:** Little-Endian estrito (`#[cfg(target_endian = "little")]`). Todo arquivo começa com magic `"OSPR"` (`[0x4F, 0x53, 0x50, 0x52]`) seguido do discriminante `format_kind` (`0x01` = `.osi`, `0x02` = `.osv`, `0x03` = `.osr`).
2. **Alinhamento de Linha de Cache (64 bytes):**
   - `OsiHeader`, `OsvHeader` e `OsrHeader` possuem exatamente **64 bytes** (`#[repr(C, align(64))]`) e derivam `bytemuck::Pod` + `bytemuck::Zeroable` sem padding implícito do compilador.
   - Todas as seções tabulares principais iniciam em offsets múltiplos de 64 bytes (`offset % 64 == 0`), permitindo cargas AVX-512 alinhadas (`vmovdqa64`) diretamente sobre `mmap` sem *cache-line splits* ou cópias.
3. **Integridade `xxHash64` (Seed `0`):**
   - `header_xxh64` (`0x38..0x3F`) cobre os primeiros 56 bytes do cabeçalho e é validado **antes** de confiar em qualquer offset ou alocar memória.
   - Em `.osv`, `page_xxh64` é validado sobre o frame comprimido **antes** de acionar o descompressor Zstd.

---

## 2. `.osi` — Osprey Resident 1-Bit Index (`format_kind = 0x01`)

Arquivo residente em RAM/`mmap` usado exclusivamente no **Estágio 1** (seleção de clusters, varredura `RaBitQ` 1-bit e filtragem de tombstones).

```text
Offset
0x00                  +-------------------------------------------------------------------+
                      | OsiHeader (64 bytes, #[repr(C, align(64))])                       |
                      |  magic="OSPR" | kind=0x01 | flags | ver=1 | orig_dim | padded_dim |
                      |  num_clusters (K) | num_vectors (N) | fwht_seed                   |
                      |  centroids_offset | entries_offset | tombstones_offset            |
                      |  header_xxh64 (bytes 0x00..0x37, seed=0)                          |
centroids_offset      +-------------------------------------------------------------------+ (align 64B)
                      | Tabela de Centróides f16 LE em ordem TSP                          |
                      |  K * padded_dim * 2 bytes (+ zero-pad para múltiplo de 64B)       |
entries_offset        +-------------------------------------------------------------------+ (align 64B)
                      | Diretório de Clusters: K * OsiClusterSpan (16 bytes cada)         |
                      |  [cluster_offset: u64 | vector_count: u32 | _reserved: u32]       |
                      |  (+ zero-pad para múltiplo de 64B)                                |
                      +-------------------------------------------------------------------+ (align 64B)
                      | Blocos Contíguos por Cluster c = 0 .. K-1 (alinhados a 64B):      |
                      |   1. Hot Stream (1-bit codes): M_c * [u64; padded_dim / 64]       |
                      |      (+ zero-pad até 64B)  <-- varredura sequencial VPOPCNTDQ     |
                      |   2. Metadata & Fatores (RabitqEntryHeader, 24 bytes/vetor):      |
                      |      [doc_id: u64 | minipage_id: u32 | slab_id: u16 | slot_idx:u16|
                      |       res_norm: f16 | l1_norm_factor: f16 | _reserved: u32]       |
                      |      (+ zero-pad até 64B)  <-- lido após limiar / Top-C           |
tombstones_offset     +-------------------------------------------------------------------+ (align 64B)
                      | Rodapé RoaringBitmap de Tombstones:                               |
                      |  [bitmap_byte_len: u64 LE]                                        |
                      |  [portable RoaringBitmap bytes]                                   |
                      |  [tombstone_xxh64: u64 LE (cobre len || bitmap bytes, seed=0)]    |
EOF - 8               +-------------------------------------------------------------------+
                      | file_xxh64: u64 (checksum global dos bytes [0 .. EOF - 8])        |
EOF                   +-------------------------------------------------------------------+
```

> **Por que separar códigos 1-bit de `RabitqEntryHeader`?**
> Durante o hot loop `VPOPCNTDQ` (ou `AVX2`/`NEON`), a CPU lê sequencialmente apenas os vetores de bits `[u64; padded_dim / 64]` sem poluir as linhas de cache L1/L2 com `doc_id`, `minipage_id` ou `slot_idx`. Os identificadores de localização só são acessados para candidatos que sobrevivem ao limiar de distância.

### Campos de `OsiHeader` (64 bytes)

| Offset | Tam. | Campo | Tipo Rust | Descrição |
| :--- | :---: | :--- | :--- | :--- |
| `0x00..0x03` | 4 B | `magic` | `[u8; 4]` | `"OSPR"` (`[0x4F, 0x53, 0x50, 0x52]`). |
| `0x04` | 1 B | `format_kind` | `u8` | `0x01` (`.osi`). |
| `0x05` | 1 B | `flags` | `u8` | Bitflags do índice (`0x00` na v1). |
| `0x06..0x07` | 2 B | `version` | `u16` | Versão do layout (`1`). |
| `0x08..0x09` | 2 B | `orig_dim` | `u16` | Dimensão original $d_{\text{orig}}$. |
| `0x0A..0x0B` | 2 B | `padded_dim` | `u16` | Potência de 2 acolchoada para FWHT ($\ge 64$, `% 64 == 0`). |
| `0x0C..0x0F` | 4 B | `num_clusters` | `u32` | Total de clusters $K$ (`1-Bit Binary K-Means`). |
| `0x10..0x17` | 8 B | `num_vectors` | `u64` | Total de vetores indexados $N$. |
| `0x18..0x1F` | 8 B | `fwht_seed` | `u64` | Seed SplitMix64 dos sinais Rademacher da FWHT. |
| `0x20..0x27` | 8 B | `centroids_offset` | `u64` | Offset da tabela `f16` de centróides (`% 64 == 0`). |
| `0x28..0x2F` | 8 B | `entries_offset` | `u64` | Offset do diretório de clusters `RaBitQ` (`% 64 == 0`). |
| `0x30..0x37` | 8 B | `tombstones_offset` | `u64` | Offset do rodapé `RoaringBitmap` (`% 64 == 0`). |
| `0x38..0x3F` | 8 B | `header_xxh64` | `u64` | `xxHash64` sobre `[0x00..0x37]` (seed `0`). |

---

## 3. `.osv` — Osprey Locality-Reordered Vector Slabs (`format_kind = 0x02`)

Armazena os vetores exatos em `f16` e payloads JSON no Object Storage (S3/MinIO) ou disco, em minipáginas de ~16 KiB ordenadas por localidade TSP e requisitadas em paralelo pelo planejador DP de ranges (`RangePlanner`).

```text
Offset
0x00                  +-------------------------------------------------------------------+
                      | OsvHeader (64 bytes, #[repr(C, align(64))])                       |
                      |  magic="OSPR" | kind=0x02 | zstd_level | ver=1 | orig_dim | flags |
                      |  num_minipages | total_vectors | zstd_dict_offset | zstd_dict_len |
                      |  directory_offset | _reserved | header_xxh64                      |
zstd_dict_offset      +-------------------------------------------------------------------+ (align 64B)
                      | [Opcional] Dicionário Zstd Treinado (se OSV_FLAG_HAS_ZSTD_DICT)   |
                      |  (zstd_dict_len bytes, até 32 KiB padrão + zero-pad até 64B)      |
                      +-------------------------------------------------------------------+ (align 64B)
                      | Frames de Minipáginas Comprimidas Zstd (Ordem TSP de Clusters):   |
                      |   Minipágina #0 (~16 KiB descomprimida)                           |
                      |   Minipágina #1 (~16 KiB descomprimida)                           |
                      |   ...                                                             |
                      |   Minipágina #(num_minipages - 1) (+ zero-pad até 64B)            |
directory_offset      +-------------------------------------------------------------------+ (align 64B)
                      | Diretório de Minipáginas: num_minipages * MinipageDirectoryEntry  |
                      |  48 bytes/entrada:                                                |
                      |   [byte_offset: u64 | min_doc_id: u64 | max_doc_id: u64 |         |
                      |    page_xxh64: u64  | compressed_size: u32 | uncompressed_size:u32|
                      |    cluster_id: u32  | vector_count: u16    | flags: u16]          |
EOF                   +-------------------------------------------------------------------+
```

### Anatomia Interna de Uma Minipágina `.osv` Descomprimida (~16 KiB)

```text
+-------------------------------------------------------------------------+ 0x00
| Minipage Internal Header (8 bytes LE)                                   |
|  [0x00..0x01] vector_count: u16 (M_p)                                   |
|  [0x02..0x03] orig_dim: u16 (d)                                         |
|  [0x04..0x07] cluster_id: u32 (ID físico em ordem TSP)                  |
+-------------------------------------------------------------------------+ 0x08
| doc_ids_le: [u64; M_p] (M_p * 8 bytes LE)                               |
+-------------------------------------------------------------------------+ 0x08 + M_p * 8
| vectors_le: [f16; M_p * d] (M_p * d * 2 bytes LE, row-major por slot)   |
+-------------------------------------------------------------------------+
| payload_offsets_le: [u32; M_p + 1] (prefix-sum relativo a payloads)     |
+-------------------------------------------------------------------------+
| payloads: bytes UTF-8 JSON concatenados (sem separadores extras)        |
+-------------------------------------------------------------------------+ uncompressed_size
```

> **Por que `MinipageDirectoryEntry` inclui `min_doc_id`, `max_doc_id` e `page_xxh64`?**
> - **Zone-map fast-path:** Consultas filtradas por faixa de `doc_id` descartam minipáginas inteiras no diretório em RAM antes de emitir `Range-GET` ao S3.
> - **Proteção contra corrupção em rede:** O `osprey-storage` valida `page_xxh64` sobre o frame comprimido **antes** de chamar `zstd::DecoderDictionary`.

---

## 4. `.osr` — Osprey SIMD-Interleaved Token Rerank Store (`format_kind = 0x03`)

Armazena tokens multi-vetor (padrão ColBERT/ColPali com `token_dim = 128`) quantizados pelo **Reticulado Aninhado 2D de 2 Camadas** ($\gamma = 0.28$), garantindo exatamente **64 bytes = 1 linha de cache por token**.

```text
Offset
0x00                  +-------------------------------------------------------------------+
                      | OsrHeader (64 bytes, #[repr(C, align(64))])                       |
                      |  magic="OSPR" | kind=0x03 | subspace_dim=2 | num_layers=2 | pad=0 |
                      |  ver=1 | token_dim=128 | gamma=0.28f32 | fwht_seed                |
                      |  num_docs | total_tokens | doc_table_offset | tensors_offset      |
                      |  header_xxh64                                                     |
doc_table_offset      +-------------------------------------------------------------------+ (align 64B)
                      | Tabela de Documentos: num_docs * DocTokenRange (24 bytes cada)    |
                      |  Ordenada por doc_id crescente -> binary search O(log N_docs)     |
                      |  [doc_id: u64 | token_start_idx: u64 | token_count: u16 | _pad:6B]|
                      |  (+ zero-pad até 64B)                                             |
tensors_offset        +-------------------------------------------------------------------+ (align 64B)
                      | Tensores de Tokens Intercalados (total_tokens * 64 bytes):        |
                      |   Token #0  [0x00..0x1F: Camada 1 (32B) | 0x20..0x3F: Camada 2]   |
                      |   Token #1  [0x00..0x1F: Camada 1 (32B) | 0x20..0x3F: Camada 2]   |
                      |   ...                                                             |
EOF (- 8 opcional)    +-------------------------------------------------------------------+
```

### Empacotamento Nibble-Packed de 1 Token (64 Bytes = 1 Linha de Cache)

Cada token de $D = 128$ dimensões rotacionado pela FWHT (equalizado globalmente com $\sigma = 1/\sqrt{128}$, **sem normalização individual por subespaço**) possui $S = 64$ subespaços 2D. Cada subespaço recebe dois índices de 4 bits ($c^{(1)}_s, c^{(2)}_s \in [0, 15]$):

```text
  Cada byte k (0..31) empacota dois subespaços consecutivos: (even = 2k, odd = 2k + 1)

    Bits:    7   6   5   4   3   2   1   0
           +---------------+---------------+
           |  c_odd (4b)   |  c_even (4b)  |   byte[k] = (c_even & 0x0F) | ((c_odd & 0x0F) << 4)
           +---------------+---------------+

  Layout linear do registro de 64 bytes (lido em 1 vmovdqa64 no AVX-512):
  +-----------------------------------+-----------------------------------+
  | 0x00 .. 0x1F (32 bytes): Camada 1 | 0x20 .. 0x3F (32 bytes): Camada 2 |
  | 64 subespaços (base do reticulado)| 64 subespaços residuais (gamma=.28)|
  +-----------------------------------+-----------------------------------+
```

Essa disposição permite decodificar todos os 64 subespaços em paralelo usando `_mm512_shuffle_epi8` / `_mm256_shuffle_epi8` (`vpshufb`) ou `vqtbl1q_s8` (`vtbl`) contra Lookup Tables `int8` de 16 entradas sintetizadas em `< 8 µs`, reduzindo o índice em **7.9×** frente ao ColBERTv2 em `FP32` (`97.2 MiB` vs. `768.4 MiB` no MS MARCO) sem nenhuma multiplicação em ponto flutuante no loop de documentos.
