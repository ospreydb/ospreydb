# `05_cli_and_pov_benchmark.md` — Especificação da Interface CLI, Harness `osprey bench` e Simulador FinOps/TCO

> **Status:** Especificação Fechada para Implementação
> **Nota de Contexto:** Este documento integra as especificações de engenharia da CLI unificada do OspreyDB (`osprey`, delegando os subcomandos de engine para `crates/osprey-cli`).
> **Escopo:** `crates/osprey-cli` e `crates/ospreydb` (binário unificado `osprey`).
> **Referências:** [`AGENTS.md`](../AGENTS.md), [`ARCHITECTURE.md`](../ARCHITECTURE.md), [`ADR-001`](../adr/ADR-001-decoupled-1bit-ram-index-and-s3-slabs.md), [`ADR-003`](../adr/ADR-003-open-core-boundary-and-offline-ed25519-licensing.md).

---

## 1. Árvore Completa de Comandos `clap` (`osprey`)

O binário `osprey` utiliza `clap` (`derive`) com saída dual legível por humanos (tabelas ASCII formatadas) e legível por máquinas (`--json`), obedecendo estritamente a códigos de saída POSIX determinísticos.

### 1.1 Códigos de Saída POSIX

| Código | Significado | Exemplos |
| :---: | :--- | :--- |
| **`0`** | Sucesso integral | Consulta executada, segmento importado, benchmark aprovado. |
| **`2`** | Erro de sintaxe ou argumentos CLI | Flag inválida, combinação incompatível (`clap`). |
| **`3`** | Violação estrutural, integridade ou numérica | `InvalidMagic`, `ChecksumMismatch`, `UnalignedPointer`, `NonFiniteEmbedding`. |
| **`4`** | Falha de I/O local ou Object Storage | Arquivo inexistente, falha `memmap2`, erro S3/HTTP `object_store`. |
| **`5`** | Falha em portão de diagnóstico ou benchmark | `Recall@10` abaixo do limiar, falta de requisito em `doctor`. |

### 1.2 Assinaturas Exatas em Rust (`crates/osprey-cli/src/cli.rs`)

```rust
use std::path::PathBuf;
use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Debug, Parser)]
#[command(
    name = "osprey",
    version,
    about = "OspreyDB: Zero-Copy Vector Search + LUT-Rank Engine CLI"
)]
pub struct OspreyCli {
    /// Emite saída estruturada em JSON válido em stdout.
    #[arg(long, global = true)]
    pub json: bool,

    /// Número de threads do pool Rayon (0 = automático por núcleos físicos).
    #[arg(long, global = true, default_value_t = 0)]
    pub threads: usize,

    /// Nível de verbosidade de logs diagnósticos em stderr.
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,

    #[command(subcommand)]
    pub command: OspreyCommand,
}

#[derive(Debug, Subcommand)]
pub enum OspreyCommand {
    /// Verifica recursos de CPU SIMD, alinhamento mmap e conectividade S3.
    Doctor(DoctorArgs),
    /// Ingere Parquet/JSONL out-of-core gerando segmentos .osi, .osv e .osr.
    Import(ImportArgs),
    /// Executa busca vetorial ANN (Estágios 1, 2 e 3).
    Search(SearchArgs),
    /// Executa busca completa + reranking multi-vetor LUT-Rank (.osr, Estágio 4).
    Rerank(RerankArgs),
    /// Aplica deleções lógicas locais no RoaringBitmap de tombstones do .osi.
    Delete(DeleteArgs),
    /// Inspeciona cabeçalhos, offsets, checksums xxHash64 e diretórios de segmentos.
    Inspect(InspectArgs),
    /// Executa benchmark científico reproduzível e projeção FinOps/TCO na nuvem.
    Bench(BenchArgs),
}

#[derive(Debug, Args)]
pub struct DoctorArgs {
    /// URI opcional de Object Storage para testar permissões Head/Range-GET.
    #[arg(long)]
    pub s3_uri: Option<String>,
}

#[derive(Debug, Args)]
pub struct ImportArgs {
    /// Caminho de entrada (.parquet ou .jsonl).
    #[arg(short, long)]
    pub input: PathBuf,
    /// Diretório ou URI base de saída para os artefatos segment.{osi,osv,osr}.
    #[arg(short, long)]
    pub output: String,
    /// Nome da coluna de identificador (u64).
    #[arg(long, default_value = "doc_id")]
    pub id_col: String,
    /// Nome da coluna de vetor denso.
    #[arg(long, default_value = "embedding")]
    pub vec_col: String,
    /// Nome opcional da coluna de tokens multi-vetor (dim=128) para gerar .osr.
    #[arg(long)]
    pub token_col: Option<String>,
    /// Número de clusters K do 1-Bit Binary K-Means.
    #[arg(long, default_value_t = 256)]
    pub clusters: u32,
    /// Tamanho do lote de streaming Arrow/Parquet.
    #[arg(long, default_value_t = 4096)]
    pub batch_size: usize,
    /// Desabilita treinamento de dicionário Zstd sobre payloads.
    #[arg(long)]
    pub no_zstd_dict: bool,
    /// Seed determinística SplitMix64 para FWHT e Binary K-Means.
    #[arg(long, default_value_t = 0x4F53_5052_0001)]
    pub seed: u64,
}

#[derive(Debug, Args)]
pub struct SearchArgs {
    #[arg(long)]
    pub index: PathBuf,
    #[arg(long)]
    pub slabs: String,
    /// Vetor de consulta em JSON (`[0.1, -0.2, ...]`) ou caminho de arquivo JSON.
    #[arg(long)]
    pub query: String,
    #[arg(short = 'k', long, default_value_t = 10)]
    pub top_k: usize,
    #[arg(long, default_value_t = 16)]
    pub nprobe: usize,
    #[arg(long, default_value_t = 150)]
    pub candidates: usize,
    /// Multiplicador de oversampling do Estágio 1 (0 = adaptativo: 2x quando `dim <= 256`;
    /// 1 = desativado; N > 1 = forçado). Também aplica um piso de 128 candidatos ao
    /// refinamento `f16` quando ativo em baixa dimensionalidade.
    #[arg(long, default_value_t = 0)]
    pub refine_multiplier: usize,
    /// Capacidade máxima do cache LRU local de minipáginas em MiB.
    #[arg(long, default_value_t = 256)]
    pub cache_mb: usize,
}

/// Política compartilhada: `SearchOptions::resolve` (osprey-core) multiplica `nprobe` e
/// `candidates` por 2 e aplica piso de 128 candidatos ao refinamento `f16` quando
/// `orig_dim <= 256`; `--refine-multiplier 1` desativa. `osprey search`, `osprey bench`
/// e `osprey-bench-runner` devem usar o plano resolvido, nunca duplicar a regra.
#[derive(Debug, Args)]
pub struct RerankArgs {
    #[command(flatten)]
    pub search: SearchArgs,
    /// Caminho do store de tokens .osr.
    #[arg(long)]
    pub rerank_store: PathBuf,
    /// Matriz de tokens da query (`[[...128...], ...]`) em JSON ou arquivo.
    #[arg(long)]
    pub query_tokens: String,
    /// Número de sobreviventes do Estágio 3 enviados ao LUT-Rank.
    #[arg(long, default_value_t = 200)]
    pub rerank_top_m: usize,
}

#[derive(Debug, Args)]
pub struct DeleteArgs {
    #[arg(long)]
    pub index: PathBuf,
    /// Lista de doc_ids separados por vírgula.
    #[arg(long, value_delimiter = ',')]
    pub ids: Vec<u64>,
    /// Arquivo texto com um doc_id por linha.
    #[arg(long)]
    pub ids_file: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub struct InspectArgs {
    #[arg(short, long)]
    pub file: PathBuf,
    #[arg(long)]
    pub verify_checksums: bool,
    #[arg(long)]
    pub dump_directory: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum RerankCostMode {
    Gpu,
    Api,
    None,
}

#[derive(Debug, Args)]
pub struct BenchArgs {
    #[arg(long, default_value_t = 20_000)]
    pub vectors: usize,
    #[arg(long, default_value_t = 384)]
    pub dim: usize,
    #[arg(long, default_value_t = 100)]
    pub queries: usize,
    #[arg(long, default_value_t = 128)]
    pub clusters: u32,
    #[arg(long, default_value_t = 24)]
    pub nprobe: usize,
    #[arg(long)]
    pub with_lutrank: bool,
    /// Escala alvo para projeção financeira mensal AWS (ex.: 50000000).
    #[arg(long, default_value_t = 50_000_000)]
    pub project_vectors: u64,
    /// Dimensionalidade alvo na projeção financeira (ex.: 1536).
    #[arg(long, default_value_t = 1536)]
    pub project_dim: u64,
    /// Consultas mensais projetadas (para cálculo de API /v1/rerank e S3 GETs).
    #[arg(long, default_value_t = 10_000_000)]
    pub project_qpm: u64,
    #[arg(long, value_enum, default_value_t = RerankCostMode::Gpu)]
    pub rerank_mode: RerankCostMode,
    /// Exporta laudo técnico e financeiro em Markdown (ex.: report.md).
    #[arg(long)]
    pub export_report: Option<PathBuf>,
}
```

---

## 2. Harness Científico do `osprey bench`

O subcomando `osprey bench` executa uma auditoria end-to-end reprodutível sem exigir datasets externos:

1. **Geração Determinística de Corpus Anisotrópico:**
   - Gera `vectors` embeddings anisotrópicos normalizados de dimensão `dim` via `SplitMix64` + decaimento espectral em lei de potência, além de `queries` consultas e ground-truth exato em `FP32`.
2. **Instrumentação Física e Temporal:**
   - **Memória Residente (`RSS`):** Lida do sistema operacional (`/proc/self/statm` no Linux, `task_info` no macOS, `GetProcessMemoryInfo` no Windows) antes do build, após o build e durante a carga `mmap` do `.osi`.
   - **Tempo de Indexação (`1-Bit Binary K-Means` + TSP):** Cronometrado com `std::time::Instant`.
   - **Taxa de Poda em RAM:** Calculada como `1.0 - (osi_bytes as f64 / legacy_fp32_hnsw_bytes as f64)`.
   - **Fator de Coalescência do Planejador DP:** Razão `total_unique_minipages / coalesced_http_ranges` registrada sobre todas as consultas.
   - **Latência $P_{50} / P_{95} / P_{99}$:** Histogramas separados para `ANN` (Estágio 1 + planejador DP + refinamento `f16`) e `LUT-Rank` (síntese LUT + kernel `MaxSim` sobre 200 docs quando `--with-lutrank` ativo).
   - **Qualidade Científica:** `Recall@10` do `ANN` vs ground-truth `FP32` e, sob `--with-lutrank`, correlação de Pearson e `Recall@10` do `MaxSim`.

---

## 3. Fórmulas do Simulador Financeiro de Fatura Cloud (TCO Calculator)

Para qualquer par (`--project-vectors`, `--project-dim`, `--project-qpm`), `osprey bench` calcula de forma determinística a fatura mensal estimada em **AWS `us-east-1`**:

### 3.1 Equações Exatas de Precificação

```rust
pub const HNSW_GRAPH_OVERHEAD: f64 = 1.65;
pub const R6I_8XL_RAM_GB: f64 = 256.0;
pub const R6I_8XL_MONTHLY_USD: f64 = 1_472.0;   // r6i.8xlarge (~$2.016/h * 730h)
pub const R6I_16XL_RAM_GB: f64 = 512.0;
pub const R6I_16XL_MONTHLY_USD: f64 = 2_944.0;  // r6i.16xlarge (~$4.032/h * 730h)
pub const G5_XLARGE_MONTHLY_USD: f64 = 734.0;   // g5.xlarge dedicada para reranking
pub const RERANK_API_PER_1K_USD: f64 = 2.00;    // US$ 2,00 / 1.000 chamadas /v1/rerank
pub const S3_STANDARD_PER_GB_USD: f64 = 0.023;  // US$ 0,023 / GB / mês
pub const S3_GET_PER_1K_USD: f64 = 0.0004;      // US$ 0,0004 / 1.000 Range-GETs
pub const C7I_2XL_MONTHLY_USD: f64 = 245.0;     // Instância CPU leve (16 GB RAM, AVX-512)
```

1. **Custo Arquitetura Legada (HNSW em RAM + Rerank GPU/API):**
   - **Memória RAM HNSW Necessária (GB):**
     $$\text{RAM}_{\text{legacy}}(\text{GB}) = \frac{N_{\text{proj}} \times D_{\text{proj}} \times 4 \times 1.65}{10^9}$$
     *(Ex.: $50\text{M} \times 1536 \times 4 \times 1.65 = 506,88\ \text{GB}$ $\implies$ exige no mínimo 1x `r6i.16xlarge` de 512 GB ou 2x `r6i.8xlarge` = **US\$ 2.944,00/mês** base On-Demand apenas de RAM, chegando a US\$ 4.200–9.500/mês com HA/EBS.)*
   - **Fração/Contagem de Nós `r6i`:** Escolhe o menor múltiplo inteiro de `r6i.8xlarge` (256 GB) / `r6i.16xlarge` (512 GB) capaz de acomodar $\text{RAM}_{\text{legacy}}$.
   - **Adicional de Reranking:**
     - `Gpu`: $+ \$734,00 \times \text{nós\_gpu}$ (`g5.xlarge`).
     - `Api`: $+ \frac{\text{project\_qpm}}{1000} \times \$2,00$.
     - `None`: $\$0,00$.

2. **Custo OspreyDB (`.osi` em CPU Leve + `.osv` no S3 + `LUT-Rank` na Mesma CPU):**
   - **RAM Residente do `.osi` (GB):**
     $$\text{RAM}_{\text{osi}}(\text{GB}) = \frac{N_{\text{proj}} \times (\lceil D_{\text{proj}} / 8 \rceil + 4)}{10^9} \quad (\sim 196\ \text{bytes/vetor para } 1536\ \text{dims} \approx 9,8\ \text{GB})$$
   - **Compute CPU Leve (Vector Search + LUT-Rank no mesmo nó):** Aloca $\lceil \text{RAM}_{\text{osi}} / 14.0 \rceil$ instâncias `c7i.2xlarge` (16 GB RAM, AVX-512, US\$ 245,00/mês; para 50M vetores = **1 instância**, US\$ 245,00/mês) — **custo zero adicional** para `LUT-Rank`.
   - **Armazenamento S3 Standard (`.osv` slabs `f16` + payloads Zstd):**
     $$\text{GB}_{\text{osv}} \approx \frac{N_{\text{proj}} \times (D_{\text{proj}} \times 2 + 2928_{\text{est}})}{10^9} \quad (\approx 300\ \text{GB para } 50\text{M} \times 1536)$$
     $$\text{Custo}_{\text{S3}} = \text{GB}_{\text{osv}} \times \$0,023 + \left(\frac{\text{project\_qpm} \times \text{avg\_coalesced\_gets}}{1000}\right) \times \$0,0004$$

### 3.2 Layout do Relatório no Terminal e Exportação `--export-report report.md`

```text
+------------------------------------------------------------------------------+
| OSPREYDB SCIENTIFIC BENCHMARK & CLOUD TCO REPORT                             |
+------------------------------------------------------------------------------+
| Local Measurement (N=20,000, D=384 -> pad=512, K=128, nprobe=24)             |
|   Build Time (1-Bit Binary K-Means + TSP) :  142.3 ms                        |
|   Index RSS (.osi mmap)                :    2.6 MB  (99.25% RAM pruned)      |
|   DP Coalesce Factor                   :   18.4x    (planner: 0.14 ms)       |
|   ANN Recall@10 vs FP32                :   97.40%   (P50: 1.8 ms, P99: 4.1ms)|
|   LUT-Rank Pearson / Recall@10         :  0.9954 / 100.0% (200 docs: 2.9 ms) |
+------------------------------------------------------------------------------+
| Cloud TCO Projection (50,000,000 vectors x 1536 dims, 10,000,000 QPM)        |
|   Legacy HNSW RAM (r6i.16xlarge, 506.9 GB)   : $ 2,944.00 / mo               |
|   Legacy Rerank (1x g5.xlarge GPU)           : $   734.00 / mo               |
|   TOTAL LEGACY INFRASTRUCTURE                : $ 3,678.00 / mo ($44,136 / yr)|
|   -------------------------------------------------------------------------- |
|   OspreyDB .osi Compute (1x c7i.2xlarge)     : $   245.00 / mo (9.8 GB .osi) |
|   OspreyDB .osv S3 Standard (300 GB)         : $     6.90 / mo               |
|   OspreyDB S3 Range-GETs (DP coalesced)      : $    16.00 / mo               |
|   OspreyDB LUT-Rank Rerank (same CPU)        : $     0.00 / mo               |
|   TOTAL OSPREYDB INFRASTRUCTURE              : $   267.90 / mo ($ 3,215 / yr)|
|   ESTIMATED MONTHLY SAVINGS                  : $ 3,410.10 / mo (92.7% saved) |
+------------------------------------------------------------------------------+
| 100% Apache-2.0 Open Source — Continuous WAL, S3 CAS & Arrow Flight:          |
| `osprey serve --data-dir ./data --s3-uri file:///tmp/osprey-s3`                |
+------------------------------------------------------------------------------+
```

Quando `--export-report report.md` é informado, o mesmo conjunto de métricas auditáveis, parâmetros de execução e tabela comparativa FinOps é gravado atomicamente em Markdown pronto para compartilhamento com times de Engenharia e FinOps.
