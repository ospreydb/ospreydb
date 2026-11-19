# RFC-0001: Motor OspreyDB — Índice Residente `RaBitQ` de 1-Bit, `1-Bit Binary K-Means` e Planejamento DP de Ranges sobre Object Storage

| Metadado | Valor |
| :--- | :--- |
| **RFC** | `0001` |
| **Título** | OspreyDB: Motor de Busca Vetorial Nativo para Object Storage via FWHT, `1-Bit Binary K-Means`, `RaBitQ` Assimétrico e Planejador DP de Ranges |
| **Status** | **Accepted / Frozen** |
| **Data** | `2026-11-15` |
| **Autores** | Lucas Ramos, Engenharia Principal do OspreyDB (`ospreydb/ospreydb`) |
| **Artefatos Relacionados** | [`ARCHITECTURE.md`](../ARCHITECTURE.md) (Seções 3, 4 e 6), `crates/osprey-core`, `crates/osprey-simd`, `crates/osprey-storage` |
| **Referências Científicas** | • `arXiv:2405.12497` — *RaBitQ*<br>• `arXiv:2409.09913` — *Extended RaBitQ*<br>• `arXiv:2608.14648v1` — *Stop Indexing at Full Precision: Revisiting Clustering for Vector Embeddings* |

---

## 1. Resumo Executivo

Esta especificação define a fundação matemática e algorítmica do plano de recuperação de primeiro e segundo estágios do **OspreyDB**. O objetivo é eliminar a dependência de clusters de memória RAM massiva para índices de grafos (HNSW), mantendo apenas um índice binário ultracompacto residente (`.osi`) em RAM/`mmap` e delegando os vetores exatos em `f16` e os payloads JSON para slabs ordenados por localidade em Object Storage (`.osv`), acessados por um planejador de requisições baseado em Programação Dinâmica (`RangePlanner`).

---

## 2. Definição do Problema

### 2.1 O Colapso de Grafos HNSW sobre Object Storage e Busca Desacoplada

Em arquiteturas nativas de busca desacoplada em Object Storage (cf. `arXiv:2610.25172v1`, *RemoteANN*), estruturas baseadas em grafos navegáveis hierárquicos (HNSW) dependem de travessias iterativas guiadas por ponteiros: a seleção do próximo vizinho no passo $t+1$ depende estritamente das distâncias calculadas aos vizinhos expandidos no passo $t$. Essa dependência sequencial impõe $O(\log N)$ saltos de busca dependentes de dados (*data-dependent pointer chasing*):

1. **Amplificação de Latência por TTFB:** Em armazenamento em nuvem (Amazon S3, GCS, R2, MinIO), cada chamada HTTP `Range-GET` isolada sofre um *Time-to-First-Byte* (TTFB) típico de $15\text{–}40\ \text{ms}$ ($p_{50}\text{–}p_{99}$). Uma travessia de $20\text{–}35$ saltos sem localidade espacial acumula $300\text{–}1400\ \text{ms}$ apenas em latência serial de rede por consulta.
2. **Explosão de IOPS e Custo por Requisição:** A leitura de listas de adjacência esparsas espalhadas por milhões de vértices impede o coalescimento eficiente em disco ou rede, multiplicando o custo de API (`GET` requests) em ordens de grandeza.

### 2.2 Custo Proibitivo de Clusters Residentes em RAM

Para evitar o colapso em Object Storage, bancos vetoriais convencionais (Qdrant, Pinecone, Milvus em modo HNSW) mantêm vetores completos e o grafo residentes em RAM:
- Para $N = 10^8$ vetores de dimensão $d = 1536$ em `FP32` ($6144\ \text{B/vetor}$) mais metadados de grafo HNSW ($M = 16\text{–}32$, $\sim 256\text{–}512\ \text{B/vetor}$), o consumo excede **$650\text{ GB}$ de RAM** por réplica sem redundância.
- Em $N = 10^9$ vetores, o custo de instâncias `r6i`/`r7i` de alta memória ultrapassa dezenas de milhares de dólares mensais, enquanto o mesmo volume armazenado em S3 custa uma fração ínfima ($\sim \$0.023/\text{GB-mês}$).

---

## 3. Formulação Matemática Completa

### 3.1 Rotação Ortogonal Rápida via Transformada de Walsh-Hadamard (FWHT)

Embeddings modernos (especialmente modelos treinados com perda contrastiva) exibem forte **anisotropia** e *outliers* de variância concentrados em poucas coordenadas. Quantizadores de 1-bit aplicados diretamente no espaço original sofrem degradação severa de erro.

O OspreyDB aplica uma rotação ortogonal pseudoaleatória $P \in \mathbb{R}^{d \times d}$ antes de qualquer indexação ou quantização, onde $d = d_{\text{pad}} = 2^{\lceil \log_2(\max(d_{\text{orig}}, 64)) \rceil}$ e o vetor de entrada recebe zero-padding até $d$:

$$P x = \frac{1}{\sqrt{d}} \, H_d \, D_{\text{signs}} \, x$$

Onde:
- $D_{\text{signs}} = \text{diag}(s_0, s_1, \dots, s_{d-1})$ é uma matriz diagonal de sinais de Rademacher $s_i \in \{-1, +1\}$ gerados deterministicamente a partir de `fwht_seed` (via PRNG criptográfico/determinístico fixado no cabeçalho do segmento).
- $H_d \in \{-1, +1\}^{d \times d}$ é a matriz recursiva de Walsh-Hadamard definida por:
  $$H_1 = [1], \qquad H_{2m} = \begin{bmatrix} H_m & H_m \\ H_m & -H_m \end{bmatrix}$$

**Propriedades Invariantes:**
1. **Ortogonalidade Exata ($P^\top P = I_d$):** Como $D_{\text{signs}}^\top D_{\text{signs}} = I_d$ e $H_d^\top H_d = d \cdot I_d$, segue que:
   $$\langle P q, P x \rangle = \langle q, x \rangle \quad \text{e} \quad \|P x\|_2 = \|x\|_2$$
2. **Complexidade $O(d \log d)$ sem Multiplicações Matriciais Densas:** Diferente de uma matriz ortogonal Gaussiana densa $Q \in \mathbb{R}^{d \times d}$ ($O(d^2)$ de tempo e $O(d^2)$ de memória), a FWHT executa $\log_2 d$ estágios *butterfly* apenas com somas, subtrações e uma única multiplicação escalar por $1/\sqrt{d}$, reduzindo o custo para $< 10\ \mu\text{s}$ por query e zero armazenamento de matriz de rotação.
3. **Isotropização de Energia:** Pela propriedade de espalhamento espectral de $H_d D_{\text{signs}}$, cada coordenada $(Px)_i$ torna-se uma combinação de Rademacher de todas as coordenadas originais, equalizando a variância por dimensão para $\approx \|x\|_2^2 / d$ com caudas subgaussianas uniformes.

### 3.2 Algoritmo `1-Bit Binary K-Means` (`arXiv:2608.14648v1`)

Treinar K-Means em `FP32` sobre milhões de vetores exige manter a matriz completa $X \in \mathbb{R}^{N \times d}$ em RAM durante o build. Inspirado por `arXiv:2608.14648v1`, o **OspreyDB** executa o particionamento primário diretamente sobre as assinaturas binarizadas pós-FWHT:

Para cada vetor normalizado ou centralizado após rotação, define-se sua assinatura binária de 1-bit:
$$b(x) = \text{sign}(P x) \in \{-1, +1\}^d \quad \Longleftrightarrow \quad \tilde{b}_i(x) = \mathbb{I}[(P x)_i \ge 0] \in \{0, 1\}$$

1. **Passo de Atribuição (HAMMING / XNOR-POPCNT):**
   Cada centróide $c_k \in \mathbb{R}^d$ também é binarizado para $\tilde{b}(c_k) \in \{0, 1\}^d$ durante o loop de atribuição (ou avaliado por produto interno quantizado contra $\tilde{b}(x)$):
   $$k^*(x) = \arg\max_{k \in \{1..K\}} \langle b(x), b(c_k) \rangle = \arg\min_{k \in \{1..K\}} \text{popcnt}\big(\tilde{b}(x) \oplus \tilde{b}(c_k)\big)$$
   Executado via instruções SIMD `VPOPCNTDQ` sobre blocos `[u64; d / 64]`.
2. **Passo de Atualização e Refinamento:**
   Em cada iteração, os novos centróides são acumulados a partir das assinaturas binárias dos membros do cluster $C_k$ e projetados na esfera unitária (ou escalados pela norma média do cluster), com polimento final opcional em micro-amostras para ajustar exatamente as normas em `f16`:
   $$c_k^{(t+1)} = \frac{\sum_{x \in C_k} b(x)}{\left\|\sum_{x \in C_k} b(x)\right\|_2}$$
3. **Ganhos Estruturais:**
   - **60x menos memória RAM no build** (representação de $d$ bits = $d/8$ bytes por vetor vs. $4d$ bytes em `FP32` mais buffers de trabalho de alta precisão).
   - **20x maior velocidade de convergência** devido ao throughput de `XNOR + POPCNT` frente a `FMA` limitado por largura de banda DRAM.
   - **$< 1\%$ de perda relativa** na qualidade geométrica da partição Voronoi comparado ao K-Means exato em `FP32`, preservando integralmente o recall após refinamento.

### 3.3 Quantização Assimétrica `RaBitQ` dos Resíduos Intra-Cluster (`arXiv:2405.12497`, `arXiv:2409.09913`)

Dado um vetor $x \in \mathbb{R}^d$ atribuído ao centróide $c_k$, decompomos o produto interno exato com a query $q$ pela identidade algébrica do resíduo:

$$\langle q, x \rangle = \langle q, c_k \rangle + \|x - c_k\|_2 \cdot \langle q, u \rangle, \qquad u = \frac{x - c_k}{\|x - c_k\|_2}$$

Aplicando a rotação ortogonal $P$ ($P^\top P = I_d$), temos $\langle q, u \rangle = \langle P q, P u \rangle$ com $\|P u\|_2 = 1$.

#### Codificação do Banco (1-Bit por Dimensão)

Para o vetor residual rotacionado $\tilde{u} = P u$, armazenamos no arquivo `.osi`:
- O código binário de sinais $\bar{s} = \text{sign}(P u) \in \{-1, +1\}^d$ (empacotado em `[u64; d / 64]`, $1$ bit/dim).
- A norma do resíduo `res_norm` $= \|x - c_k\|_2$ (em `f16`).
- O fator de projeção L1 médio por coordenada:
  $$\text{l1\_norm\_factor} = \frac{\|P u\|_1}{d} = \frac{1}{d} \sum_{i=1}^{d} |(P u)_i|$$

#### Derivação do Estimador Não-Enviesado

Defina o vetor unitário associado aos sinais quantizados sobre os vértices do hipercubo:
$$\bar{o} = \frac{1}{\sqrt{d}} \, \text{sign}(P u) \in \left\{-\frac{1}{\sqrt{d}}, +\frac{1}{\sqrt{d}}\right\}^d, \qquad \|\bar{o}\|_2 = 1$$

O produto interno exato entre o código unitário $\bar{o}$ e o resíduo rotacionado $P u$ vale:
$$\langle \bar{o}, P u \rangle = \frac{1}{\sqrt{d}} \sum_{i=1}^d \text{sign}((P u)_i) \cdot (P u)_i = \frac{\|P u\|_1}{\sqrt{d}} = \sqrt{d} \cdot \text{l1\_norm\_factor}$$

Pelo teorema fundamental do `RaBitQ`, sob a rotação ortogonal aleatória $P$, a projeção de $\bar{o}$ na direção de $P u$ decompõe $\bar{o}$ como:
$$\bar{o} = \langle \bar{o}, P u \rangle \, P u + e_\perp, \qquad \text{com } \langle P u, e_\perp \rangle = 0 \text{ e } \mathbb{E}[e_\perp \mid P u] = \mathbf{0}$$

Tomando o produto interno com $P q$:
$$\langle \bar{o}, P q \rangle = \langle \bar{o}, P u \rangle \, \langle P u, P q \rangle + \langle e_\perp, P q \rangle$$

Dividindo ambos os lados por $\langle \bar{o}, P u \rangle > 0$ e substituindo $\bar{o} = \frac{1}{\sqrt{d}} \text{sign}(P u)$ e $\langle \bar{o}, P u \rangle = \frac{\|P u\|_1}{\sqrt{d}}$:
$$\frac{\langle \bar{o}, P q \rangle}{\langle \bar{o}, P u \rangle} = \frac{\frac{1}{\sqrt{d}} \langle P q, \text{sign}(P u) \rangle}{\frac{\|P u\|_1}{\sqrt{d}}} = \frac{\langle P q, \text{sign}(P u) \rangle}{\|P u\|_1} = \frac{\langle P q, \text{sign}(P u) \rangle}{d \cdot \text{l1\_norm\_factor}}$$

Como $\mathbb{E}[\langle e_\perp, P q \rangle] = 0$, o estimador possui **viés zero** ($\mathbb{E}[\hat{s}(q, x)] = \langle q, x \rangle$) e variância limitada por $O(1/d)$ independente da anisotropia original dos dados.

#### Quantização Assimétrica da Query em 4 Planos de Bits (`int4`)

Na fase online, quantizamos as coordenadas da query rotacionada $P q$ em 4 bits sem sinal $\text{quant}_4(P q) \in \{0, \dots, 15\}^d$ (com escala $\Delta_q$ e ponto-zero $z_q$ restaurados na acumulação afim) e transpomos esses dígitos para 4 planos de bits $B_0, B_1, B_2, B_3 \in \{0, 1\}^d$. O estimador operacional completo implementado no Estágio 1 é:

$$\hat{s}(q, x) = \langle q, c_k \rangle + \|x - c_k\|_2 \cdot \frac{\langle \text{quant}_4(P q), \text{sign}(P u) \rangle}{d \cdot \text{l1\_norm\_factor}}$$

Onde o produto interno bitwise $\langle \text{quant}_4(P q), \text{sign}(P u) \rangle$ (mapeado para bits $\{0, 1\}$ via $\text{sign} = 2 b - 1$) é avaliado sem nenhuma multiplicação por coordenada usando apenas `AND` + `VPOPCNTDQ`:
$$\langle \text{quant}_4(P q), b \rangle = \sum_{r=0}^{3} 2^r \cdot \text{popcnt}\big(B_r \mathbin{\&} b\big)$$
combinado com o termo escalar constante por query $- \frac{1}{2} \sum_i \text{quant}_4(P q)_i$ e $\text{popcnt}(b)$ conforme a expansão afim de $\Delta_q \cdot \text{quant}_4(P q) + z_q \mathbf{1}$.

### 3.4 Reordenação Física de Localidade via TSP (Nearest-Neighbor + 2-opt)

Para transformar buscas aleatórias em Object Storage em varreduras contíguas, o OspreyDB aplica uma otimização em dois níveis antes de materializar os slabs `.osv` e os clusters `.osi`:

1. **TSP Inter-Cluster:** Constrói-se o grafo completo $G = (V, E)$ sobre os $K$ centróides com pesos de aresta dados pela distância euclidiana $w(c_i, c_j) = \|c_i - c_j\|_2$. Resolve-se um circuito hamiltoniano aproximado via **Nearest-Neighbor** seguido de busca local **2-opt** até convergência:
   $$\pi^* = \arg\min_{\pi \in S_K} \sum_{t=1}^{K-1} \|c_{\pi(t)} - c_{\pi(t+1)}\|_2$$
   Os clusters são gravados fisicamente no `.osi` e no `.osv` estritamente na ordem $\pi^*$. Dessa forma, quando uma query sonda `nprobe` clusters vizinhos no espaço vetorial, suas minipáginas tendem a ocupar offsets contíguos no `.osv`.
2. **Ordenação Intra-Cluster em Minipáginas de 16 KB:** Dentro de cada cluster $\pi^*(t)$, os vetores são agrupados sequencialmente (ou ordenados por projeção principal / sub-clusters TSP locais) em minipáginas descomprimidas de $\sim 16\ \text{KB}$, maximizando a densidade de candidatos úteis por frame Zstd buscado.

### 3.5 Equação de Recorrência de Programação Dinâmica do Planejador `RangePlanner`

Após o Estágio 1, os Top-$C$ candidatos apontam para um conjunto ordenado e deduplicado de $k$ intervalos de bytes no `.osv`:
$$I_1 = [a_1, b_1], \; I_2 = [a_2, b_2], \; \dots, \; I_k = [a_k, b_k], \qquad a_1 \le b_1 < a_2 \le b_2 < \dots < b_k$$

Cada requisição HTTP `Range-GET` no Object Storage incorre em um modelo de custo aditivo calibrado pelo perfil do provedor:
- $\alpha > 0$: custo fixo por requisição (latência TTFB + tarifa por chamada `GET`).
- $\beta > 0$: custo marginal por byte transferido (largura de banda + egresso/leitura).
- $W_{\max}$: tamanho máximo permitido para um único intervalo coalescido (ex.: $4\ \text{MiB}$), evitando bloqueio de *head-of-line* ou leitura excessiva de lixo.

O planejador **DP de ranges** (`RangePlanner`) encontra a partição ótima dos $k$ intervalos em $m \le k$ requisições coalescidas minimizando o custo total esperado. Seja $DP(i)$ o custo mínimo para cobrir os primeiros $i$ intervalos $I_1, \dots, I_i$, com $DP(0) = 0$:

$$DP(i) = \min_{\substack{1 \le j \le i \\ (b_i - a_j) \le W_{\max}}} \Big\{ \, DP(j - 1) + \alpha + \beta \cdot (b_i - a_j) \, \Big\}$$

**Propriedades e Complexidade:**
- **Critério Local de Coalescimento:** Entre dois intervalos adjacentes $[a, b]$ e $[c, d]$ com lacuna $g = c - b$ e extensão total $d - a \le W_{\max}$, fundi-los economiza $\alpha$ e adiciona $\beta \cdot g$; logo, a fusão é estritamente benéfica sempre que $g \le \tau_{\text{gap}} = \alpha / \beta$.
- **Execução em $< 0.5\ \text{ms}$:** Como a função de custo de intervalo é linear em $(b_i - a_j)$ sujeita à janela deslizante $b_i - a_j \le W_{\max}$, a recorrência reduz-se a uma avaliação linear $O(k)$ mantendo uma fila monótona (*monotone deque*) ou ponto de corte único sobre `j`, garantindo execução determinística abaixo de $0.5\ \text{ms}$ para centenas de candidatos.

---

## 4. Metas Empíricas Validadas

Os parâmetros da arquitetura **OspreyDB** (`osprey-core` + `osprey-storage`) são dimensionados e auditados contra as seguintes metas quantitativas:

| Métrica | Meta Canônica | Mecanismo Responsável |
| :--- | :---: | :--- |
| **Redução de Memória RAM** | **Poda de 99,25% em RAM** | Índice `.osi` apenas em 1-bit + fatores `f16`; remoção completa de grafos HNSW e vetores `f32`/`f16` da RAM residente. |
| **Redução de Chamadas ao S3** | **$> 16\times$ menos requisições `GET`** | Reordenação TSP intra/inter-cluster + fusão ótima de intervalos pela recorrência DP do `RangePlanner`. |
| **Qualidade de Recuperação** | **`Recall@10 >= 96.6%`** | Rotação FWHT isotropizante + estimador `RaBitQ` não-enviesado (`int4` x `1-bit`) + refinamento exato `f16` no Estágio 3. |
| **Overhead do Planejador DP** | **$< 0.5\ \text{ms}$ em CPU** | Programação Dinâmica $O(k)$ sobre entradas de 48 bytes do diretório `.osv`. |

> **Nota sobre Orçamento de Latência S3 (Same-Region vs. WAN/TLS):** O orçamento de latência ponta a ponta `< 35 ms` pressupõe execução em instância EC2/ECS na **mesma região** do bucket S3 (`RTT ~2–5 ms`, uma única onda paralela de `get_ranges`). Em benchmarks ou clientes remotos operando através de enlace WAN/TLS de longa distância (`RTT ~40–70 ms`), a latência P50 observada situa-se tipicamente na faixa de **`75–110 ms`**, permanecendo dominada por uma única rodada de RTT de rede em vez de múltiplos saltos sequenciais.
