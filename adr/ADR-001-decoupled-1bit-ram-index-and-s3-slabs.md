# ADR-001: Rejeição de Grafos HNSW em RAM/S3 em Favor da Arquitetura Desacoplada `.osi` (1-Bit em RAM) + `.osv` (Slabs Reordenados no S3)

| Metadado | Valor |
| :--- | :--- |
| **ADR** | `001` |
| **Status** | **Accepted** |
| **Data** | `2026-11-15` |
| **Autores** | Engenharia Principal do OspreyDB (`ospreydb/ospreydb`) |
| **Referências Canônicas** | [`ARCHITECTURE.md`](../ARCHITECTURE.md), [`RFC-0001`](../rfcs/RFC-0001-osprey-rabitq-range-planner.md) |

---

## 1. Contexto

Nos Data Lakes corporativos modernos, coleções entre **50 milhões e 500 milhões de registros** (documentos corporativos, logs de telemetria semântica, catálogos de e-commerce, bases RAG e transcrições multimodais) já residem de forma consolidada em arquivos **Apache Parquet** armazenados em Object Storage barato (Amazon S3, Google Cloud Storage, Cloudflare R2 ou MinIO). O custo de retenção desses datasets em repouso gira em torno de poucos dólares por terabyte ao mês.

Contudo, para atender consultas vetoriais de baixa latência ($p_{99} < 50\ \text{ms}$), a arquitetura predominante da indústria obriga as equipes de dados a **duplicar integralmente** esses corpora para clusters dedicados baseados em grafos HNSW residentes em RAM (Qdrant, Pinecone, Milvus em modo HNSW, OpenSearch k-NN). Essa duplicação impõe três falhas estruturais graves:

1. **Assimetria Econômica Extrema:** A memória DRAM em nuvem custa aproximadamente **300x a 450x mais** por gigabyte/mês do que o S3 Standard. Manter vetores de 1536 dimensões em `FP32` ($6144\ \text{B/vetor}$) somados às listas de adjacência de grafos navegáveis e réplicas de alta disponibilidade transforma a busca vetorial no maior item da fatura de infraestrutura de IA.
2. **Pipelines de ETL *Split-Brain*:** A sincronização contínua entre a fonte canônica em Parquet no Data Lake e o índice mutável residente em RAM exige orquestradores complexos, reindexações custosas quando há evolução de schema e reconciliação constante de divergências (*consistency drift*).
3. **Subutilização de Hardware:** Instâncias otimizadas para memória (`r6i`, `x2iedn`) são alugadas pela capacidade de RAM, deixando dezenas de vCPUs subutilizadas enquanto o grafo HNSW permanece ocioso entre picos de tráfego.

Diante disso, investigamos duas alternativas principais antes de congelar a arquitetura do **OspreyDB**:
- **(A)** Manter grafos HNSW residentes em RAM (ou híbridos com *disk-backed mmap* sobre EBS `gp3`/`io2`).
- **(B)** Mapeamento direto de grafos HNSW sobre arquivos armazenados no Object Storage (S3).
- **(C)** **Arquitetura Desacoplada de Duas Camadas (`.osi` + `.osv`)** com índice binário residente de 1-bit e slabs ordenados por localidade física no S3.

---

## 2. Análise Matemática da Falha do HNSW sobre Object Storage (Alternativa B)

A tentativa ingênua de "colocar um grafo HNSW no S3" e buscá-lo sob demanda via `Range-GET` falha de maneira fundamental pelas propriedades físicas de latência de armazenamento em nuvem e pela topologia dependente de dados da busca em grafos.

### 2.1 Amplificação Serial por Dependência de Ponteiros

Em um grafo HNSW com fator de ramificação médio $M$ (tipicamente $M = 16\text{–}32$) e $N$ vértices, o caminho da camada superior ($L_{\max} \approx \lfloor \ln N / \ln M \rfloor$) até o mínimo local na camada base $0$, seguido pela busca gulosa com feixe de largura `efSearch` (tipicamente $64\text{–}128$), exige uma cadeia de decisões **estritamente sequenciais**:

$$\text{Passo } t+1 \text{ depende dos vizinhos } \mathcal{N}(v_t) \text{ descobertos no Passo } t$$

Mesmo considerando que as camadas superiores ($l \ge 2$) caibam em cache local, a descida na camada base $0$ para $N = 5 \times 10^7$ vetores executa em média **$H \in [12, 25]$ rodadas sequenciais** de expansão de vértices que não podem ser paralelizadas entre si (porque o identificador do vértice $v_{t+1}$ só é conhecido após avaliar as distâncias de $\mathcal{N}(v_t)$).

Em Amazon S3 Standard, uma requisição HTTP `Range-GET` isolada apresenta um *Time-to-First-Byte* (TTFB) típico de:
$$\tau_{\text{TTFB}} \in [15\ \text{ms}, \, 40\ \text{ms}] \quad (\text{podendo exceder } 70\ \text{ms em } p_{99})$$

A latência total mínima imposta apenas pela barreira de rede sequencial obedece a:

$$L_{\text{HNSW-S3}} \ge \sum_{t=1}^{H} \tau_{\text{TTFB}}(t) \approx H \cdot \bar{\tau}_{\text{TTFB}}$$

Substituindo valores operacionais reais:
- **Cenário otimista** ($H = 15$ rodadas, $\bar{\tau}_{\text{TTFB}} = 20\ \text{ms}$): **$300\ \text{ms}$** por query.
- **Cenário realista com `efSearch = 128` e jitter de cauda** ($H = 30$ rodadas ou $25 \times 40\ \text{ms}$ + penalidade de *tail-at-scale* onde a rodada inteira espera o nó mais lento do lote paralelo): **$600\ \text{ms}$ a $1.200\ \text{ms}$** por query.

Isso viola em mais de uma ordem de grandeza o SLA interativo de sistemas de busca ($< 50\ \text{ms}$).

### 2.2 Explosão de Custos de API `GET` no S3

Além da latência proibitiva, cada vértice expandido no HNSW aponta para $M \approx 32$ vizinhos cujos vetores (`6 KB` cada em `FP32` para 1536 dims) estão dispersos aleatoriamente pelo arquivo de $300+\ \text{GB}$ devido à ordem de inserção ou construção estocástica do grafo. Mesmo agrupando listas de adjacência:
- Uma única query com `efSearch = 64` toca entre **80 e 250 regiões não contíguas** do arquivo.
- A $100\ \text{QPS}$ sustentados ($\approx 2.59 \times 10^8$ queries/mês), com apenas 100 `Range-GET`s por query, o sistema emitiria **$2.59 \times 10^{10}$ chamadas `GET`/mês**.
- À tarifa padrão S3 Standard de $\$0.0004$ por $1.000$ requisições `GET`, apenas a taxa de chamadas de API atingiria **$\$10.368/\text{mês}$** — antes mesmo de contabilizar computação ou transferência.

**Conclusão:** Grafos HNSW são estruturalmente incompatíveis com a geometria de custos e latência de Object Storage.

---

## 3. A Decisão Arquitetural: Duas Camadas Desacopladas (`.osi` + `.osv`)

Rejeitamos tanto os clusters HNSW residentes em RAM (Alternativa A) quanto o HNSW sobre S3 (Alternativa B). O **OspreyDB** adota uma arquitetura desacoplada determinística de dois estágios principais (detalhada byte a byte em [`ARCHITECTURE.md`](../ARCHITECTURE.md) e formalizada na [`RFC-0001`](../rfcs/RFC-0001-osprey-rabitq-range-planner.md)):

### 3.1 Camada Residente em RAM/mmap (`.osi` — Osprey Resident 1-Bit Index)

Em vez de um grafo de ponteiros aleatórios, mantemos residente em memória apenas o índice `.osi`:
- **Particionamento e Rotação:** Os vetores sofrem rotação ortogonal rápida via Transformada de Walsh-Hadamard (**FWHT**, $O(d \log d)$) e são particionados pelo algoritmo **`1-Bit Binary K-Means`** em $K$ clusters ordenados por um circuito TSP.
- **Pegada de Memória Ultra-Compacta:** Cada vetor no diretório `RaBitQ` assimétrico combina um código binário de **1 bit por dimensão** ($D/8$ bytes no stream quente varrido pelo `VPOPCNTDQ`) e uma struct `RabitqEntryHeader` de exatamente **24 bytes** (`doc_id: u64`, `minipage_id: u32`, `slab_id: u16`, `slot_idx: u16`, `res_norm: f16`, `l1_norm_factor: f16`, `_reserved: u32`). Somando $24 + D/8$ bytes por vetor, o consumo é de **`40 B/vetor`** para $D = 128$, **`120 B/vetor`** para $D = 768$ ($152\ \text{B/vetor}$ com $d_{\text{pad}} = 1024$ pós-FWHT) e **`216 B/vetor`** para $D = 1536$ ($280\ \text{B/vetor}$ quando acolchoado para $d_{\text{pad}} = 2048$ na FWHT, mantendo o *working set* quente de varredura 1-bit em `192–256 B/vetor`).
- **Varredura Linear Determinística a $> 50\ \text{GB/s}$:** Na execução da consulta, o motor seleciona os `nprobe` centróides mais próximos e varre os blocos contíguos de 1-bit apenas daqueles clusters usando instruções AVX-512 **`VPOPCNTDQ`** (com a query transposta em 4 planos `int4`). Como o acesso é estritamente sequencial por cluster (layout Cluster-SoA), o prefetcher de hardware atinge $> 50\ \text{GB/s}$ de largura de banda efetiva de memória, **podando $99,25\%$ dos vetores inteiramente em RAM sem tocar na rede**.

### 3.2 Camada Fria Reordenada no S3 (`.osv` + Planejador DP de Ranges)

Os dados completos — vetores exatos em `f16`, *zone maps* escalares e payloads JSON — residem exclusivamente em arquivos `.osv` imutáveis no S3 Standard:
- **Minipáginas de 16 KB Ordenadas por TSP:** Os vetores pertencentes a cada cluster são empacotados em minipáginas descomprimidas de ~16 KB comprimidas com **Zstd** (opcionalmente com dicionário treinado sobre os payloads) e gravadas fisicamente na ordem do circuito TSP entre centróides. Isso garante forte correlação espacial: candidatos co-selecionados pelo Estágio 1 quase sempre residem em minipáginas próximas no `.osv`.
- **Coalescimento via Programação Dinâmica (`RangePlanner`):** O planejador DP recebe os intervalos `(byte_offset, compressed_size)` das minipáginas dos Top-$M$ candidatos sobreviventes e resolve uma recorrência de Programação Dinâmica ($< 0.5\ \text{ms}$) que funde intervalos próximos sempre que o custo dos bytes de preenchimento (*gap*) for inferior ao custo/latência de uma chamada S3 adicional.
- **Rodada Única Paralela de `Range-GET`:** Em vez de $15\text{–}30$ saltos seriais, o `osprey-storage` dispara **uma única rodada paralela** de poucas chamadas HTTP/2 `get_ranges` coalescidas (tipicamente $1\text{–}6$ requisições por query), descomprime as minipáginas de 16 KB em paralelo e executa o refinamento exato `f16` em CPU.

---

## 4. Análise FinOps Auditável (Cenário: 50 Milhões de Vetores, 1536 Dimensões)

Considere um workload corporativo padrão: **$N = 50.000.000$ vetores**, dimensionalidade $d = 1536$, acompanhados de ~2 KB de payload JSON/metadados por registro (dataset bruto $\approx 400\ \text{GB}$ em Parquet/JSONL), servindo **10 milhões de queries/mês** ($\sim 4\text{ QPS}$ médios, picos de $40\text{ QPS}$) em `us-east-1` na AWS.

### 4.1 Cenário Baseline: HNSW Residente em RAM (Qdrant / Milvus / OpenSearch)

- **Memória Necessária:**
  - Vetores em `FP32`: $50\text{M} \times 1536 \times 4\ \text{B} = 307,2\ \text{GB}$.
  - Grafo HNSW ($M = 24$, ponteiros + metadados de camada + estruturas de alocador): $\sim 80\text{–}120\ \text{GB}$.
  - Payloads indexáveis + margem operacional de 20% contra OOM durante compactions/merges: totaliza **$\sim 512\ \text{GB}$ de RAM dedicada** (para 1 réplica; $1.024\ \text{GB}$ em configuração Multi-AZ HA).
- **Fatura Mensal Estimada (AWS `us-east-1` On-Demand / 1-yr Savings):**
  - 1x `r6i.16xlarge` (64 vCPU, 512 GB RAM): $\sim \$4.029/\text{mês}$ (On-Demand `$4.029` / com EBS `gp3` de 1 TB + snapshots + tráfego cross-AZ $\approx \mathbf{\$4.200/\text{mês}}$).
  - Em instância `x2iedn.8xlarge` / cluster gerenciado multi-nó com HA (2x `r6i.16xlarge` ou Pinecone/Cloud gerenciado equivalente para 512 GB RAM): **$\$8.200\text{ a }\$9.500/\text{mês}$**.

### 4.2 Cenário OspreyDB: `.osi` em Container CPU Leve + `.osv` no S3 Standard

- **Armazenamento Frio (`.osv` no S3 Standard):**
  - Vetores exatos `f16` ($153,6\ \text{GB}$) + payloads JSON comprimidos com Zstd + Dicionário ($\sim 135\ \text{GB}$) + diretório de minipáginas ($\sim 1\ \text{GB}$) + margem $\approx \mathbf{300\ \text{GB}}$.
  - Custo S3 Standard: $300\ \text{GB} \times \$0.023/\text{GB-mês} = \mathbf{\$6,90/\text{mês}}$.
- **Computação Leve com `.osi` Residente:**
  - Índice `.osi` com `RabitqEntryHeader` (24 B) + código 1-bit ($D/8$ B): para $D = 1536$, $216\ \text{B/vetor}$ ($50\text{M} \times 216\ \text{B} = \mathbf{10,8\ \text{GB}}$, ou $280\ \text{B/vetor} = \mathbf{14,0\ \text{GB}}$ com acolchoamento FWHT de $d_{\text{pad}} = 2048$ em `mmap`), cabendo confortavelmente em **1 único container de CPU leve com 16 GB–32 GB de RAM** (ex.: `c7i.2xlarge` — 8 vCPU AVX-512, 16 GB RAM On-Demand $\approx \$245/\text{mês}$, ou ECS/Fargate spot/1-yr Savings $\approx \$130\text{–}\$180/\text{mês}$).
- **Chamadas `Range-GET` ao S3 (Coalescidas pelo Planejador DP):**
  - Com média de $4$ chamadas coalescidas por query (antes do cache local): $10\text{M} \times 4 = 40\text{M GETs/mês} \times \$0.0004 / 1.000 = \mathbf{\$16,00/\text{mês}}$ (reduzido para $< \$5/\text{mês}$ com hit rate de $70\%$ no cache LRU local).
  - Transferência intra-região S3 $\rightarrow$ EC2/ECS na mesma região AWS: **$\$0,00/\text{mês}$** (gratuita).

### 4.3 Tabela Comparativa FinOps

| Item de Custo (50M vetores, 1536 dims) | HNSW Tradicional (~512 GB RAM) | OspreyDB (`.osi` + `.osv` S3) | Economia |
| :--- | :---: | :---: | :---: |
| **Armazenamento / RAM Residente** | $512\text{ GB}$ RAM dedicada (`r6i`/`x2iedn`) | **$\sim 10,8\text{–}14,0\ \text{GB}$ `.osi`** em container CPU leve + **$300\text{ GB}$ S3** | **97,3%–97,9% menos RAM** |
| **Custo S3 Storage (300 GB)** | N/A (embutido em RAM/EBS) | **`$6,90/mês`** | — |
| **Custo Compute / Memória** | `$4.200 – $9.500/mês` | **`$180 – $260/mês`** (1x container CPU AVX-512) | **~95% – 97% menor** |
| **Custo S3 `GET` Requests** | `$0` | **`$5 – $16/mês`** (após coalescing DP) | Negligível |
| **Fatura Total Estimada / Mês** | **`$4.200 – $9.500 / mês`** | **`~$195 – $285 / mês`** | **Redução > 15x a 33x** |

---

## 5. Consequências, Limitações e Mitigações

### 5.1 Trade-offs Aceitos

1. **Latência $p_{50}$ em Cache Frio vs. RAM Pura:**
   - Enquanto um índice HNSW 100% em RAM responde em **$\sim 2\text{–}4\ \text{ms}$**, uma consulta do OspreyDB que precise buscar minipáginas frias no S3 sofre uma rodada de RTT de rede, resultando em **$p_{50} \approx 18\text{–}35\ \text{ms}$**. Consideramos esse acréscimo de ~15–30 ms altamente favorável para RAG, busca semântica corporativa e pipelines analíticos diante da redução de $> 95\%$ no TCO.
2. **Imutabilidade dos Segmentos Base e Escrita Contínua:**
   - A ordenação TSP e a compressão em minipáginas do `.osv` pressupõem segmentos imutáveis (com deleções lógicas registradas no `RoaringBitmap` de tombstones do `.osi`). Cargas com escrita contínua em tempo real são atendidas pela camada de WAL, `LsmDeltaBuffer` e `CowSlabCompactor` integrados em `crates/ospreydb` via trait `StorageEngineExt`.
3. **Sensibilidade ao `nprobe` em Distribuições Extremamente Multimodais:**
   - Como o Estágio 1 não percorre arestas transversais entre clusters, recuperar `Recall@10 >= 96.6%` em embeddings fortemente anisotrópicos exige calibração adequada de `nprobe` e da largura do feixe de candidatos `Top-C` enviados ao planejador DP de ranges.

### 5.2 Mitigações de Engenharia Obrigatórias

1. **Cache LRU Local de Minipáginas Quentes (NVMe / RAM):**
   - Workloads de busca reais seguem fortemente distribuições Zipfianas (20% dos clusters/minipáginas concentram 70–85% dos acessos). O `osprey-storage` mantém um cache LRU local indexado por `(segment_id, minipage_id)` armazenando frames Zstd descomprimidos (ou comprimidos em NVMe local). Hits no cache eliminam a ida ao S3, derrubando a latência $p_{50}$ para **$< 5\ \text{ms}$** e reduzindo ainda mais a fatura de `GET`s.
2. ***Hedged Requests* para Controle de Latência de Cauda ($p_{99}$):**
   - O S3 apresenta ocasionalmente picos transitórios de *tail latency* ($> 80\text{ ms}$) em conexões individuais. Para proteger o orçamento $p_{99}$, o cliente `osprey-storage` dispara *hedged requests* configuráveis: se uma chamada `get_ranges` não retornar o primeiro byte após um limiar adaptativo (ex.: `max(25 ms, 1.5 * p95_ewma)`), uma requisição especulativa redundante é enviada em outra conexão HTTP/2 do pool, consumindo a primeira resposta válida e cancelando a remanescente.
3. **Validação Contínua de Recall e Coalescing:**
   - A suíte de benchmarks audita periodicamente `Recall@10`, fator de coalescimento do planejador DP e orçamento de latência por estágio contra o baseline exato em `FP32`.
