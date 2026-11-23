# ADR-002: Reranking Multi-Vetor (`MaxSim`) em CPU Commodity via Reticulados Aninhados e Lookups em Registradores SIMD (`vpshufb` / `vtbl`)

| Metadado | Valor |
| :--- | :--- |
| **ADR** | `002` |
| **Status** | **Accepted** |
| **Data** | `2026-11-15` |
| **Autores** | Engenharia Principal do OspreyDB (`ospreydb/ospreydb`) |
| **Referências Canônicas** | [`ARCHITECTURE.md`](../ARCHITECTURE.md), [`RFC-0002`](../rfcs/RFC-0002-lutrank-nested-lattice-simd-reranker.md), [`ADR-001`](./ADR-001-decoupled-1bit-ram-index-and-s3-slabs.md) |

---

## 1. Contexto e Gargalo Computacional

### 1.1 Limite Semântico da Recuperação de Vetor Único

Embora a recuperação vetorial em dois estágios do OspreyDB (`.osi` + `.osv`) recupere os Top-$C$ candidatos com alta eficiência, comprimir um documento técnico completo em um **único vetor denso** (1536 dimensões) impõe uma perda inevitável de resolução lexical e estrutural (*semantic bottleneck*). Em consultas envolvendo termos técnicos específicos — códigos de erro de compiladores, identificadores de peças industriais (*SKUs*), cláusulas contratuais, nomes próprios raros ou combinações booleanas implícitas entre múltiplos conceitos — a média vetorial do documento dilui sinais locais críticos.

Para atingir precisão de nível de produção em sistemas RAG e busca corporativa, é indispensável um estágio final de **reranking por interação tardia** (*late interaction*, arquitetura ColBERT) sobre os **Top-200 candidatos** sobreviventes ao refinamento `f16`. Nesse paradigma, cada consulta $Q$ ($|Q| \approx 32$ tokens) e cada documento $D$ ($|D| \approx 200$ tokens, dimensão $D_{\text{tok}} = 128$) são avaliados pelo operador `MaxSim`:

$$\text{MaxSim}(Q, D) = \sum_{i=1}^{|Q|} \max_{j=1}^{|D|} \langle q_i, d_j \rangle$$

### 1.2 O *Memory Bandwidth Wall* do `MaxSim` em `FP32` / `FP16`

Executar `MaxSim` ingênuo em `FP32` ou `FP16` diretamente sobre CPUs convencionais colide rapidamente contra o **muro de largura de banda de memória** (*Memory Bandwidth Wall*):

- **Volume de Streaming por Consulta:** Para 200 documentos candidatos com média de 200 tokens de 128 dimensões:
  - Em `FP32`: $200 \times 200 \times 128 \times 4\ \text{bytes} = \mathbf{20,48\ \text{MB}}$ por consulta.
  - Em `FP16`: $200 \times 200 \times 128 \times 2\ \text{bytes} = \mathbf{10,24\ \text{MB}}$ por consulta.
- **Saturação de Cache e DRAM:** Esse conjunto de trabalho estoura as caches L1/L2 privadas do núcleo e pressiona a cache L3 compartilhada sob concorrência multi-thread, forçando leituras repetidas da DRAM. Combinado às $32 \times 200 \times 200 \times 128 \approx 1,64 \times 10^8$ operações de multiplicação-e-soma (`FMA`) por consulta, um único núcleo de CPU leva dezenas de milissegundos e satura o barramento de memória do socket sob QPS moderado.

---

## 2. A Falha Econômica e Operacional das GPUs e APIs Externas de Rerank

Diante desse gargalo em CPU, a reação convencional do mercado divide-se em duas abordagens, ambas rejeitadas por esta decisão arquitetural:

### 2.1 Instâncias GPU Dedicadas (`g5.xlarge` / `g6.xlarge`)

Provisionar um microserviço separado em GPU para rodar Cross-Encoders ou ColBERT `FP16`:
- **Desperdício FinOps:** Uma única instância AWS `g5.xlarge` (NVIDIA A10G) ou `g6.xlarge` (NVIDIA L4) custa entre **$\$730$ e $\$1.200/\text{mês}$ por nó** (multiplicado por 2–3 nós para alta disponibilidade Multi-AZ), superando em várias vezes o custo inteiro do nó CPU do OspreyDB calculado no [`ADR-001`](./ADR-001-decoupled-1bit-ram-index-and-s3-slabs.md). Em workloads corporativos com perfil diurno, essas GPUs operam com **taxa média de ociosidade de ~85%** fora dos horários de pico.
- **Fragilidade Operacional:** Exige gerenciamento de drivers NVIDIA/CUDA no host, imagens de container de **~10 GB** (PyTorch + CUDA + cuBLAS) com *cold starts* lentos em autoscaling, filas de micro-batching para amortizar transferência PCIe Host-to-Device e risco recorrente de falhas de **OOM em VRAM** quando picos de concorrência ou documentos com sequências longas de tokens colidem simultaneamente.

### 2.2 APIs Externas de `/v1/rerank` (ex.: Cohere Rerank)

Terceirizar os Top-200 documentos para APIs HTTP gerenciadas:
- **Explosão de Custo Variável:** À tarifa de mercado típica de **$\$2,00$ por $1.000$ chamadas de busca**, um volume modesto de 10 milhões de queries/mês gera uma fatura de **$\$20.000/\text{mês}$** — mais de 70x o custo total de infraestrutura do motor de busca inteiro no OspreyDB.
- **Risco Crítico de InfoSec e Conformidade:** Enviar os 200 trechos textuais confidenciais mais relevantes do Data Lake corporativo (contratos, prontuários, código-fonte, dados financeiros) para endpoints de terceiros fora da VPC cria uma superfície inaceitável de exfiltração de dados e viola políticas estritas de residência/LGPD/GDPR/HIPAA.
- **Penalidade de Latência:** Adiciona $40\text{–}150\ \text{ms}$ de RTT externo na cauda de cada consulta.

---

## 3. A Decisão Técnica: Motor `LUT-Rank` em Registradores SIMD (`.osr`)

Decidimos implementar todo o estágio de reranking multi-vetor **localmente na mesma CPU commodity** que já executa a busca vetorial de dois estágios, usando o formato binário `.osr` e o motor **`LUT-Rank`** ([`RFC-0002`](../rfcs/RFC-0002-lutrank-nested-lattice-simd-reranker.md)). A arquitetura elimina 100% das multiplicações de ponto flutuante sobre os tokens dos documentos mediante quatro decisões acopladas:

### 3.1 Estabilização Espectral via FWHT ($\sigma^2 = 1/D$)

Todos os embeddings de tokens ($D = 128$, normalizados unitariamente) recebem a rotação ortogonal rápida FWHT ($P t = \frac{1}{\sqrt{D}} H_D D_{\text{signs}} t$). Além de preservar produtos internos exatos, a FWHT espalha outliers anisotrópicos uniformemente por todas as coordenadas, estabilizando a variância coordenada a coordenada em:

$$\sigma^2 = \frac{1}{D} = \frac{1}{128} \implies \sigma = \frac{1}{\sqrt{128}} \approx 0.088388$$

### 3.2 Proibição Estrita de Normalização Individual por Subespaço

Cada token rotacionado de 128 dimensões é dividido em $S = 64$ subespaços 2D ($d_{\text{sub}} = 2$). Diferente de abordagens de *Product Quantization* que normalizam vetores ou sub-vetores localmente, **é proibido normalizar cada subespaço 2D individualmente**.

Matematicamente, no produto interno $\langle \tilde{q}, \tilde{d} \rangle = \sum_{s=0}^{63} \|\tilde{q}_s\|_2 \|\tilde{d}_s\|_2 \cos \theta_s$, a norma local $\|\tilde{d}_s\|_2$ varia naturalmente entre subespaços e carrega a distribuição real de energia do token. Normalizar cada par 2D apagaria essa ponderação relativa (ou exigiria armazenar 64 fatores de escala por token, triplicando o espaço em memória). Como a FWHT já equalizou globalmente a variância para $\sigma = 1/\sqrt{D}$, o mesmo reticulado global captura tanto a direção quanto a magnitude relativa de cada par de dimensões com **zero escalas por subespaço**.

### 3.3 Quantização em Reticulado Aninhado 2D ($M = 2$ Camadas de 4 Bits)

Cada subespaço 2D é quantizado em $M = 2$ camadas hierárquicas sobre o produto cartesiano 2D do alfabeto escalar $[-1.75, -0.55, 0.55, 1.75] \cdot \sigma$:
- **Camada 1 (Grossa):** Índice $c^{(1)}_s \in \{0..15\}$ (4 bits = 1 nibble) escolhido por minimização euclidiana direta (`argmin`).
- **Camada 2 (Residual):** O resíduo $r_s = \tilde{d}_s - p_{c^{(1)}_s}$ é quantizado sobre o reticulado contraído pelo fator $\gamma = 0.28$, gerando um segundo nibble $c^{(2)}_s \in \{0..15\}$.

Empacotando dois subespaços consecutivos por byte (`c_even | (c_odd << 4)`), os 64 subespaços de cada camada ocupam exatamente **32 bytes**, totalizando **64 bytes por token** (32 B da Camada 1 + 32 B da Camada 2). Isso reduz um token de 128 dimensões de **512 bytes (`f32`) para 64 bytes** (**compressão de 8x**, ou 4x frente a `f16`) e faz com que **cada token coincida com exatamente 1 linha de cache de 64 bytes** no `.osr`.

### 3.4 Lookup em Registradores SIMD Sem Multiplicações Float (`vpshufb` / `vtbl`)

No início da query, o motor sintetiza online Lookup Tables `int8` de 16 bytes por subespaço/camada para os tokens da query em **$< 8\ \mu\text{s}$**. Durante a varredura dos Top-200 documentos no `mmap` do `.osr`:

1. O registro de token (64 bytes alinhados) é carregado com uma única instrução alinhada (`vmovdqa64` em AVX-512 ou dois `vmovdqa` em AVX2).
2. Nibbles pares e ímpares são isolados via máscara `& 0x0F` e shift `>> 4`.
3. A instrução de embaralhamento de bytes em registrador resolve as buscas nas tabelas de 16 entradas inteiramente dentro dos registradores da CPU:
   - **`_mm512_shuffle_epi8`** (AVX-512BW, `vpshufb`): resolve **64 buscas de tabela simultaneamente** em **1 único ciclo de clock** (latência de 1 ciclo em CPUs servidor modernas).
   - **`_mm256_shuffle_epi8`** (AVX2, `vpshufb`): resolve 32 buscas por ciclo em CPUs x86_64 mais antigas.
   - **`vqtbl1q_s8`** (ARM64 NEON, `vtbl`): resolve 16 buscas por instrução em arquiteturas Graviton/Apple Silicon.
4. Os valores `int8` retornados são acumulados em inteiros `i16` livres de overflow, convertidos para `f32` apenas uma vez na redução final do token para ponderar a Camada 2 por $\gamma = 0.28$ e alimentar o `max` do operador `MaxSim`.

Com $8\times$ menos bytes trafegando pela memória e zero multiplicações `float` no loop interno sobre documentos, o gargalo de *Memory Bandwidth* desaparece completamente.

---

## 4. Consequências e Garantias de Qualidade

### 4.1 Garantias Numéricas e de Latência

- **Fidelidade Estatística (`0.9950` de Pearson):** A combinação da equalização FWHT com a quantização residual em duas camadas ($\gamma = 0.28$) e escalas `int8` calibradas preserva correlação linear de Pearson de **`0.9950`** contra o score `MaxSim` exato calculado em `FP32`.
- **`100%` de `Recall@10` no Reranking:** Na reordenação dos Top-200 candidatos para seleção dos Top-10 finais, o erro residual de quantização situa-se muito abaixo das margens de separação semântica, garantindo **`100,0%` de `Recall@10`** relativo ao baseline `FP32` exato.
- **Execução `< 3,8 ms` na Mesma CPU:** Para 200 documentos ($\sim 40.000$ tokens $= 2,56\ \text{MB}$ contíguos no `.osr`, cabendo na cache L3 de qualquer CPU moderna), o kernel `LUT-Rank` completa todo o `MaxSim` em **$< 3,8\ \text{ms}$** na mesma instância CPU leve que hospeda o índice `.osi` — eliminando 100% do custo de GPUs (`$730–$1.200/mês`) e de APIs `/v1/rerank` (`$2,00 / 1.000` chamadas), mantendo 100% dos dados dentro da VPC.

### 4.2 Limitações Aceitas e Mitigações

1. **Armazenamento do `.osr`:** Embora $8\times$ menor que `FP32`, `64 bytes/token` para 50M de documentos ($\sim 10^{10}$ tokens) soma $\sim 640\ \text{GB}$. Como os tokens de documento em `.osr` só são lidos para os Top-200 sobreviventes do Estágio 3, o arquivo `.osr` pode residir em volume NVMe local barato (ex.: `gp3` ou `instance store` de instâncias `c7id`) ou ser buscado sob demanda via `pread`/mmap cacheado, sem jamais exigir alocação integral em RAM.
2. **Especialização para `token_dim = 128`:** O empacotamento canônico de 64 bytes/token pressupõe embeddings multi-vetor de 128 dimensões (padrão ColBERTv2 / Jina-ColBERT / BGE-M3 multi-vec projetado). Modelos com dimensionalidade distinta exigem projeção linear para 128 dims ou ajuste proporcional de múltiplos da linha de cache.
3. **Portabilidade de ISA:** Sistemas sem `AVX-512BW` executam automaticamente o caminho `AVX2` (`_mm256_shuffle_epi8`), sistemas ARM64 executam `NEON` (`vqtbl1q_s8`), e máquinas legadas dispõem de fallback escalar determinístico em `osprey-simd`, garantindo portabilidade binária sem perda de corretude.
