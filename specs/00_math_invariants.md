# `00_math_invariants.md` — Invariantes Matemáticos, Numéricos e Algorítmicos do OspreyDB

> **Status:** Especificação Normativa (Fase 2)
> **Nota de Contexto:** Este documento integra as especificações originais de engenharia de engenharia do OspreyDB; a referência canônica de implementação reside em `crates/*` e [`ARCHITECTURE.md`](../ARCHITECTURE.md).
> **Escopo:** Todos os crates do workspace (`osprey-core`, `osprey-simd`, `osprey-storage`, `osprey-cli`) e validação contra o oráculo [`reference/poc_reference.py`](../reference/poc_reference.py) e vetores ouro em [`reference/golden/`](../reference/golden/golden_manifest.json).

---

## 1. Preservação de Norma $L_2$ e Produto Interno sob Rotação Ortogonal FWHT

### 1.1 Definição Algébrica do Operador Ortogonal $P$

Seja $d = 2^k$ ($k \in \mathbb{N}_{\ge 1}$) a dimensão acolchoada de trabalho. A matriz de **Walsh-Hadamard** $H_d \in \{-1, +1\}^{d \times d}$ é definida recursivamente por:

$$H_1 = [1], \qquad H_{2^k} = \begin{pmatrix} H_{2^{k-1}} & H_{2^{k-1}} \\ H_{2^{k-1}} & -H_{2^{k-1}} \end{pmatrix}$$

Seja $D_{\text{signs}} = \text{diag}(s_0, s_1, \dots, s_{d-1}) \in \{-1, +1\}^{d \times d}$ a matriz diagonal de sinais de Rademacher gerada deterministicamente pelo PRNG **SplitMix64** sobre `fwht_seed: u64`, onde a palavra de 64 bits $w_m$ define os sinais dos índices $64m + b$ ($0 \le b < 64$) segundo a regra bit-exata:

$$s_{64m+b} = \begin{cases} -1 & \text{se } ((w_m \gg b) \mathbin{\&} 1) = 0 \\ +1 & \text{se } ((w_m \gg b) \mathbin{\&} 1) = 1 \end{cases}$$

Definimos o operador linear de rotação rápida do **OspreyDB** por:

$$P = \frac{1}{\sqrt{d}} \, H_d \, D_{\text{signs}} \in \mathbb{R}^{d \times d}$$

Computacionalmente, $P x$ é avaliado via algoritmo *butterfly* *in-place* em tempo $O(d \log_2 d)$ sem materialização da matriz densa $H_d$.

---

### 1.2 Prova Formal de Ortogonalidade e Preservação Isométrica

**Teorema 1 (Ortogonalidade de $P$).** *Para todo $d = 2^k$, vale $P^T P = I_d$. Consequentemente, para quaisquer $x, y \in \mathbb{R}^d$, $\|P x\|_2 = \|x\|_2$ e $\langle P x, P y \rangle = \langle x, y \rangle$.*

**Demonstração.**
1. *Simetria e ortogonalidade escalada de $H_d$:* Por indução em $k$. Para $k = 0$, $H_1^T H_1 = [1] = 1 \cdot I_1$ e $H_1^T = H_1$. Supondo $H_{2^{k-1}}^T = H_{2^{k-1}}$ e $H_{2^{k-1}}^T H_{2^{k-1}} = 2^{k-1} I_{2^{k-1}}$:
   $$H_{2^k}^T H_{2^k} = \begin{pmatrix} H_{2^{k-1}} & H_{2^{k-1}} \\ H_{2^{k-1}} & -H_{2^{k-1}} \end{pmatrix} \begin{pmatrix} H_{2^{k-1}} & H_{2^{k-1}} \\ H_{2^{k-1}} & -H_{2^{k-1}} \end{pmatrix} = \begin{pmatrix} 2 H_{2^{k-1}}^2 & 0 \\ 0 & 2 H_{2^{k-1}}^2 \end{pmatrix} = 2^k I_{2^k} = d \cdot I_d$$
2. *Involução de $D_{\text{signs}}$:* Como $s_i \in \{-1, +1\}$, temos $s_i^2 = 1$, logo $D_{\text{signs}}^T = D_{\text{signs}}$ e $D_{\text{signs}}^2 = I_d$.
3. *Produto $P^T P$:*
   $$P^T P = \left( \frac{1}{\sqrt{d}} H_d D_{\text{signs}} \right)^T \left( \frac{1}{\sqrt{d}} H_d D_{\text{signs}} \right) = D_{\text{signs}} \left( \frac{1}{d} H_d^T H_d \right) D_{\text{signs}} = D_{\text{signs}} \, I_d \, D_{\text{signs}} = I_d$$
4. *Preservação de Produto Interno e Norma $L_2$:*
   $$\langle P x, P y \rangle = (P x)^T (P y) = x^T (P^T P) y = x^T I_d y = \langle x, y \rangle$$
   Tomando $y = x$, $\|P x\|_2^2 = \langle P x, P x \rangle = \langle x, x \rangle = \|x\|_2^2 \implies \|P x\|_2 = \|x\|_2$. $\blacksquare$

---

### 1.3 Teorema de Delocalização (Estabilização Sub-Gaussiana de Outliers Anisotrópicos)

Embeddings produzidos por redes neurais profundas frequentemente concentram energia em poucas dimensões dominantes (*outliers* anisotrópicos onde $\max_j |x_j| = \Omega(1)$ para $\|x\|_2 = 1$). Sem rotação, tanto a binarização de 1-bit (`RaBitQ`) quanto o reticulado cartesiano uniforme (`LUT-Rank`) colapsam devido ao erro de truncamento nessas coordenadas extremas.

**Teorema 2 (Delocalização Espectral sob FWHT Aleatorizada).** *Seja $x \in \mathbb{R}^d$ um vetor unitário arbitrário ($\|x\|_2 = 1$) e $s_0, \dots, s_{d-1}$ variáveis de Rademacher independentes e uniformes em $\{-1, +1\}$. Então cada coordenada rotacionada $(P x)_i$ satisfaz:*
1. *Média nula e variância exata equalizada:*
   $$\mathbb{E}\big[(P x)_i\big] = 0, \qquad \mathbb{E}\big[(P x)_i^2\big] = \frac{1}{d}$$
2. *Concentração sub-gaussiana de cauda (Desigualdade de Hoeffding / Hanson-Wright):*
   $$\mathbb{P}\big(|(P x)_i| \ge t\big) \le 2 \exp\!\left( -\frac{d \, t^2}{2} \right) \quad \forall t > 0$$
3. *Cota uniforme no máximo coordenado ($\ell_\infty$):* Para qualquer $\delta \in (0, 1)$, com probabilidade pelo menos $1 - \delta$:
   $$\|P x\|_\infty \le \sqrt{\frac{2 \ln(2d / \delta)}{d}}$$

**Demonstração.**
Pela definição de $P$, a $i$-ésima coordenada vale $(P x)_i = \sum_{j=0}^{d-1} s_j \cdot \frac{(H_d)_{ij} \, x_j}{\sqrt{d}}$. Como $(H_d)_{ij} \in \{-1, +1\}$ e $\mathbb{E}[s_j] = 0$, temos $\mathbb{E}[(P x)_i] = 0$. Pela independência $\mathbb{E}[s_j s_{j'}] = \delta_{jj'}$:

$$\mathbb{E}\big[(P x)_i^2\big] = \frac{1}{d} \sum_{j=0}^{d-1} (H_d)_{ij}^2 \, x_j^2 \, \mathbb{E}[s_j^2] = \frac{1}{d} \sum_{j=0}^{d-1} x_j^2 = \frac{\|x\|_2^2}{d} = \frac{1}{d}$$

Cada parcela $Z_j = s_j \frac{(H_d)_{ij} x_j}{\sqrt{d}}$ é simétrica e limitada em $[-c_j, +c_j]$ com $c_j = |x_j| / \sqrt{d}$. Pela desigualdade de Hoeffding para somas independentes limitadas:

$$\mathbb{P}\big(|(P x)_i| \ge t\big) \le 2 \exp\!\left( -\frac{2 t^2}{\sum_{j=0}^{d-1} (2 c_j)^2} \right) = 2 \exp\!\left( -\frac{t^2}{2 \sum_{j=0}^{d-1} x_j^2 / d} \right) = 2 \exp\!\left( -\frac{d \, t^2}{2} \right)$$

Aplicando a cota da união (*union bound*) sobre todas as $d$ coordenadas $i \in \{0, \dots, d-1\}$ e igualando $2d \exp(-d t^2 / 2) = \delta$, obtemos $t = \sqrt{\frac{2 \ln(2d / \delta)}{d}}$. $\blacksquare$

**Implicação Prática nos Crates:**
- No `RaBitQ`, a delocalização assegura que $\|P u\|_1 / d \approx \sqrt{2 / (\pi d)} \cdot \sqrt{d} = \sqrt{2 / \pi} \approx 0.7979$ em expectativa, impedindo que uma única dimensão concentre toda a norma do resíduo.
- No `LUT-Rank` ($D = 128$), o desvio-padrão universal por coordenada estabiliza-se em $\sigma = 1/\sqrt{128} \approx 0.088388$, fundamentando o reticulado 2D global sem necessidade de fatores de escala por subespaço (**proibida qualquer normalização individual por subespaço**).

---

## 2. Equação Exata de Zero-Padding para Dimensões Não-Potência de 2

Modelos de embedding comerciais frequentemente operam em dimensões que não são potências de 2 (ex.: $d = 384$, $d = 768$, $d = 1536$). Para preservar a aplicabilidade exata da FWHT sem descarte de informação nem distorção espectral, definimos:

$$d_{\text{padded}} = 2^{\lceil \log_2 d \rceil}$$

*(Para o índice residente `.osi`, adota-se $d_{\text{padded}} = 2^{\lceil \log_2 \max(d, 64) \rceil}$ de modo a garantir ao menos 1 palavra `u64` por vetor.)*

Os mapeamentos canônicos de produção são:
- $384 \to 512$
- $768 \to 1024$
- $1536 \to 2048$

### 2.1 Operador de Imersão Isométrica $\iota$

Definimos $\iota: \mathbb{R}^{d} \to \mathbb{R}^{d_{\text{padded}}}$ por:

$$\tilde{x} = \iota(x), \qquad \tilde{x}_i = \begin{cases} x_i & \text{se } 0 \le i < d \\ 0 & \text{se } d \le i < d_{\text{padded}} \end{cases}$$

**Proposição 1 (Preservação Exata após Imersão e Rotação).** *Para quaisquer $x, y \in \mathbb{R}^d$, o produto interno e as normas no espaço acolchoado e rotacionado são idênticos aos do espaço original:*

$$\langle P_{d_{\text{padded}}} \iota(x), \, P_{d_{\text{padded}}} \iota(y) \rangle_{\mathbb{R}^{d_{\text{padded}}}} = \langle \iota(x), \iota(y) \rangle_{\mathbb{R}^{d_{\text{padded}}}} = \sum_{i=0}^{d-1} x_i y_i + \sum_{i=d}^{d_{\text{padded}}-1} 0 \cdot 0 = \langle x, y \rangle_{\mathbb{R}^d}$$

**Proposição 2 (Redistribuição Uniforme de Energia).** *Aplicar $P_{d_{\text{padded}}}$ sobre $\iota(x)$ distribui a energia das $d$ coordenadas ativas sobre **todas** as $d_{\text{padded}}$ coordenadas rotacionadas com variância idêntica:*

$$\mathbb{E}\Big[\big(P_{d_{\text{padded}}} \iota(x)\big)_i^2\Big] = \frac{1}{d_{\text{padded}}} \sum_{j=0}^{d-1} x_j^2 = \frac{\|x\|_2^2}{d_{\text{padded}}} \qquad \forall i \in \{0, \dots, d_{\text{padded}} - 1\}$$

**Regra Normativa:** É estritamente proibido truncar coordenadas ($d \to 2^{\lfloor \log_2 d \rfloor}$) ou substituir coordenadas de padding por constantes não nulas.

---

## 3. Tratamento Numérico Estrito na Conversão `f32` $\to$ IEEE-754 `f16` (`half::f16`)

Os formatos `.osi` (tabela de centróides, `res_norm`, `l1_norm_factor`) e `.osv` (vetores exatos em minipáginas) persistem escalares em formato **IEEE-754 binary16** (`half::f16`, 1 bit de sinal, 5 bits de expoente com viés $15$, 10 bits explícitos de mantissa).

### 3.1 Limites Físicos do Formato `binary16`

| Grandeza IEEE-754 `f16` | Expressão Exata | Valor Decimal Aproximado |
| :--- | :---: | :---: |
| **Máximo Finito Normal** (`f16::MAX`) | $(2 - 2^{-10}) \times 2^{15}$ | $65504.0$ |
| **Mínimo Positivo Normal** (`f16::MIN_POSITIVE`) | $2^{1 - 15} = 2^{-14}$ | $6.103515625 \times 10^{-5}$ |
| **Mínimo Positivo Subnormal** | $2^{-10} \times 2^{-14} = 2^{-24}$ | $5.9604644775 \times 10^{-8}$ |
| **Épsilon de Máquina** (`f16::EPSILON`) | $2^{-10}$ | $9.765625 \times 10^{-4}$ |

### 3.2 Regras Determinísticas de Sanitização e Conversão

1. **Rejeição Imediata de Não-Finitos na Ingestão:**
   - Antes de qualquer rotação FWHT, clusterização ou escrita em `.osi`/`.osv`/`.osr`, todo elemento `v: f32` de entrada deve satisfazer `v.is_finite()`.
   - A presença de qualquer `NaN`, $+\infty$ ou $-\infty$ deve abortar imediatamente a ingestão retornando `Err(OspreyError::NonFiniteEmbedding)` (sem clamping silencioso de `NaN` ou `Inf`).

2. **Clamping de Segurança em `res_norm` e `l1_norm_factor`:**
   - No estimador `RaBitQ`, o fator `l1_norm_factor` figura no denominador ($\frac{\langle \text{quant}_4(Pq), \text{sign}(Pu) \rangle}{d \cdot \text{l1\_norm\_factor}}$). Embora para qualquer vetor unitário $u \in \mathbb{R}^d$ valha teoricamente $\|P u\|_1 / d \ge \|P u\|_2 / d = 1/d$ (e tipicamente $\approx \sqrt{2/\pi} \approx 0.7979$), vetores residuais degenerados ($x \approx c_k \implies \|x - c_k\|_2 \approx 0$) poderiam sofrer *underflow* para zero ou subnormal em `f16`.
   - Portanto, antes da conversão para `half::f16`, tanto `res_norm` quanto `l1_norm_factor` devem passar pelo clamping seguro para a faixa normal positiva garantida:
     $$\epsilon_{\min} = 2^{-14} \approx 6.1035 \times 10^{-5}, \qquad v_{\text{clamped}} = \text{clamp}\big(v, \, \epsilon_{\min}, \, 65500.0\big)$$
   - *(Para `res_norm = 0.0` exato quando $x = c_k$, pode-se armazenar $\epsilon_{\min}$ caso `res_norm > 0` ou manter proteção explícita contra divisão por zero em `l1_norm_factor >= epsilon_min` no decodificador.)*

3. **Arredondamento e Promoção na Acumulação:**
   - Toda conversão `f32` $\to$ `f16` usa estritamente arredondamento IEEE-754 **Round-to-Nearest-Even** via `half::f16::from_f32(val)`.
   - **Proibição de Acumulação em `f16`:** Devido aos apenas 10 bits de mantissa ($10^{-3}$ de precisão relativa), somar $d \in [384, 2048]$ termos diretamente em `f16` produz estagnação e erro catastrófico de cancelamento. Todo produto interno envolvendo dados `f16` (centróides do `.osi` e refinamento exato do `.osv`) **deve** promover cada operando via `.to_f32()` (ou instrução `vcvtph2ps` / `fcvt`) antes da multiplicação e acumular integralmente em registradores `f32` (ou `f64` nos somatórios de redução crítica).

---

## 4. Recorrência de Programação Dinâmica do Planejador DP de Ranges (`RangePlanner`) e Prova de Otimalidade

### 4.1 Formulação do Problema de Coalescimento de Intervalos

Sejam $I_1 = [a_1, b_1], I_2 = [a_2, b_2], \dots, I_k = [a_k, b_k]$ os intervalos de bytes ordenados e disjuntos no arquivo `.osv` correspondentes às minipáginas únicas solicitadas pelos candidatos sobreviventes do Estágio 1, satisfazendo:

$$0 \le a_1 < b_1 \le a_2 < b_2 \le \dots \le a_k < b_k$$

Sejam os parâmetros do modelo de custo do Object Storage (S3/GCS/MinIO):
- $\tau_{\text{req}} > 0$: custo fixo de latência/tarifa por requisição HTTP `Range-GET` isolada.
- $\tau_{\text{byte}} > 0$: custo marginal por byte transferido na requisição.
- $B_{\max} > 0$: tamanho máximo permitido para uma única requisição coalescida (ex.: $B_{\max} = 8\ \text{MiB} = 8.388.608$ bytes), garantindo $\max_i (b_i - a_i) \le B_{\max}$.

Uma partição válida consiste em agrupar intervalos contíguos $I_j, \dots, I_i$ em uma única requisição `[a_j, b_i]` desde que $b_i - a_j \le B_{\max}$, pagando o custo $\tau_{\text{req}} + \tau_{\text{byte}} \cdot (b_i - a_j)$.

### 4.2 Equação de Bellman

Seja $C(i)$ o custo total mínimo para buscar todos os intervalos $I_1, \dots, I_i$. A recorrência de Programação Dinâmica de coalescimento é:

$$C(0) = 0, \qquad C(i) = \min_{\substack{1 \le j \le i \\ b_i - a_j \le B_{\max}}} \Big\{ \, C(j - 1) + \tau_{\text{req}} + \tau_{\text{byte}} \cdot (b_i - a_j) \, \Big\}$$

### 4.3 Prova de Subestrutura Ótima e Condição Exata de Poda

**Teorema 3 (Subestrutura Ótima).** *Se uma partição ótima de $\{I_1, \dots, I_i\}$ tem como último bloco coalescido $[a_j, b_i]$ (cobrindo $I_j, \dots, I_i$), então os blocos anteriores formam necessariamente uma partição ótima de $\{I_1, \dots, I_{j-1}\}$ com custo $C(j - 1)$.*

**Demonstração.** Como a partição é contígua e a função objetivo é aditiva com custos não negativos ($\tau_{\text{req}}, \tau_{\text{byte}} > 0$), substituir o prefixo por uma partição de $\{I_1, \dots, I_{j-1}\}$ de custo estritamente menor que $C(j - 1)$ produziria uma partição válida de $\{I_1, \dots, I_i\}$ de custo estritamente menor, contradizendo a otimalidade global. Logo, $C(i)$ satisfaz a equação de Bellman acima. $\blacksquare$

**Teorema 4 (Condição Exata de Fusão e Poda em $O(k)$).** *Considere dois intervalos adjacentes $I_m = [a_m, b_m]$ e $I_{m+1} = [a_{m+1}, b_{m+1}]$ com lacuna $g_m = a_{m+1} - b_m \ge 0$, contidos em blocos admissíveis cuja união respeita o teto de leitura ($b_r - a_\ell \le B_{\max}$). Então:*
1. *Fundir os blocos através da lacuna $g_m$ reduz estritamente o custo total se e somente se:*
   $$\tau_{\text{byte}} \cdot (a_{m+1} - b_m) < \tau_{\text{req}} \qquad \Longleftrightarrow \qquad g_m < g^* = \frac{\tau_{\text{req}}}{\tau_{\text{byte}}}$$
2. *Inversamente, se $\tau_{\text{byte}} \cdot (a_{m+1} - b_m) \ge \tau_{\text{req}}$, **nenhuma** solução ótima estrita precisa fundir através da lacuna $g_m$ — qualquer corte em $m$ tem custo menor ou igual ao da fusão.*

**Demonstração.**
Compare o custo da requisição unificada cobrindo $[a_\ell, b_r]$ contra o custo de duas requisições $[a_\ell, b_m]$ e $[a_{m+1}, b_r]$ separadas pela lacuna $g_m = a_{m+1} - b_m$:

$$\begin{aligned}
\Delta(m) &= \text{Custo}_{\text{unificado}} - \text{Custo}_{\text{separado}} \\
&= \Big(\tau_{\text{req}} + \tau_{\text{byte}}(b_r - a_\ell)\Big) - \Big(2\tau_{\text{req}} + \tau_{\text{byte}}(b_m - a_\ell) + \tau_{\text{byte}}(b_r - a_{m+1})\Big) \\
&= \tau_{\text{byte}} \big( (b_r - a_\ell) - (b_m - a_\ell) - (b_r - a_{m+1}) \big) - \tau_{\text{req}} \\
&= \tau_{\text{byte}} \cdot (a_{m+1} - b_m) - \tau_{\text{req}}
\end{aligned}$$

Logo, $\Delta(m) < 0 \iff \tau_{\text{byte}} \cdot (a_{m+1} - b_m) < \tau_{\text{req}}$. Além disso, como $b_r - a_\ell \le B_{\max}$ é monótono decrescente em $j$ (quanto maior o bloco, maior a extensão $b_i - a_j$), qualquer fronteira $m$ com $\tau_{\text{byte}} \cdot (a_{m+1} - b_m) \ge \tau_{\text{req}}$ particiona o problema em subproblemas independentes, pois nenhum candidato $j \le m$ pode melhorar $C(i)$ para $i \ge m + 1$.

Por fim, mesmo dentro de uma corrida contígua de lacunas pequenas ($g_m < \tau_{\text{req}}/\tau_{\text{byte}}$), a janela admissível $j \in [j_{\min}(i), i]$ satisfazendo $b_i - a_j \le B_{\max}$ avança monotonicamente com $i$. Escrevendo a função objetivo da DP como:

$$C(i) = \tau_{\text{req}} + \tau_{\text{byte}} \cdot b_i + \min_{\substack{j_{\min}(i) \le j \le i}} \underbrace{\big( C(j - 1) - \tau_{\text{byte}} \cdot a_j \big)}_{f(j)}$$

o termo $f(j)$ depende **apenas** de $j$ (e não de $i$). Mantendo uma fila monótona (*monotone deque*) de índices $j$ com valores crescentes de $f(j)$ sobre a janela deslizante $b_i - a_j \le B_{\max}$ e reiniciando a fila sempre que $g_{i-1} \ge \tau_{\text{req}} / \tau_{\text{byte}}$, a recorrência exata de Bellman é resolvida em **complexidade garantida $O(k)$ de pior caso** em uma única passagem linear. $\blacksquare$
