# `03_lattice_lutrank_simd.md` — Especificação do Motor Multi-Vetorial `LUT-Rank` (Fase 7)

> **Status:** Especificação Fechada para Implementação (Fase 7)
> **Nota de Contexto:** Este documento integra as especificações originais de engenharia de engenharia do OspreyDB; a implementação de produção reside em `crates/osprey-core/src/lutrank.rs` e `crates/osprey-simd`.
> **Escopo:** `crates/osprey-core` (`lutrank.rs`) e kernels vetorizados em `crates/osprey-simd` (`AVX-512BW`, `AVX2`, `ARM64 NEON` e fallback escalar).
> **Invariantes Obrigatórias:** Zero `unwrap()`/`expect()`/`panic!()` em produção; **zero alocação de heap no hot loop**; proibição estrita de normalização por subespaço pós-FWHT; paridade exata com [`reference/golden/lutrank_golden.bin`](../reference/golden/golden_manifest.json).

---

## 1. Construção do Codebook 2D de Reticulado Aninhado ($M = 2$ Camadas)

Para embeddings de tokens de dimensão canônica $D = 128$ particionados em $S = D / 2 = 64$ subespaços bidimensionais ($d_{\text{sub}} = 2$), o desvio-padrão estabilizado pela FWHT vale:

$$\sigma = \frac{1}{\sqrt{D}} = \frac{1}{\sqrt{128}} \approx 0.0883883476$$

### 1.1 Geração das Grades Cartesianas $4 \times 4$

O alfabeto escalar 1D de 4 níveis é multiplicado por $\sigma$ na **Camada 1** e por $\sigma \cdot \gamma$ ($\gamma = 0.28$) na **Camada 2**:

```rust
pub const LUTRANK_TOKEN_DIM: usize = 128;
pub const LUTRANK_SUBSPACES: usize = 64;
pub const LUTRANK_GAMMA: f32 = 0.28;
pub const LATTICE_LEVELS_1D: [f32; 4] = [-1.75, -0.55, 0.55, 1.75];

/// Codebook 2D de 16 centróides para uma camada do reticulado aninhado.
pub type LatticeCodebook2D = [[f32; 2]; 16];

/// Gera os codebooks canônicos da Camada 1 (`scale = sigma`) e
/// Camada 2 (`scale = sigma * gamma`) para `token_dim = 128`.
pub fn build_nested_lattice_codebooks(
    token_dim: usize,
    gamma: f32,
) -> (LatticeCodebook2D, LatticeCodebook2D) {
    let sigma = 1.0_f32 / (token_dim as f32).sqrt();
    let mut cb1 = [[0.0_f32; 2]; 16];
    let mut cb2 = [[0.0_f32; 2]; 16];

    for c in 0..16 {
        let x = LATTICE_LEVELS_1D[c % 4] * sigma;
        let y = LATTICE_LEVELS_1D[c / 4] * sigma;
        cb1[c] = [x, y];
        cb2[c] = [x * gamma, y * gamma];
    }
    (cb1, cb2)
}
```

### 1.2 Proibição de Normalização Individual por Subespaço e Codificação `argmin`

- **Regra Inviolável:** Após a rotação ortogonal FWHT do token ($D = 128$), cada par de coordenadas `t_s = [t[2*s], t[2*s + 1]]` **nunca** deve ser dividido por sua norma local `||t_s||_2`. Preservar a magnitude relativa entre os 64 subespaços é essencial para a acurácia do `MaxSim` (cf. [`RFC-0002`](../rfcs/RFC-0002-lutrank-nested-lattice-simd-reranker.md) §3.2).
- **Codificação em 2 Camadas (`argmin` Euclidiano):**
  Para cada subespaço $s \in [0, 63]$:
  1. **Camada 1:**
     $$c_1[s] = \arg\min_{c \in \{0..15\}} \big((t_{s,0} - \text{cb1}[c][0])^2 + (t_{s,1} - \text{cb1}[c][1])^2\big)$$
  2. **Resíduo e Camada 2:**
     $$r_s = [t_{s,0} - \text{cb1}[c_1[s]][0], \; t_{s,1} - \text{cb1}[c_1[s]][1]]$$
     $$c_2[s] = \arg\min_{c \in \{0..15\}} \big((r_{s,0} - \text{cb2}[c][0])^2 + (r_{s,1} - \text{cb2}[c][1])^2\big)$$

---

## 2. Layout de Memória SIMD-Interleaved no `.osr`

Cada token de 128 dimensões é persistido como exatamente **64 bytes = 1 linha de cache** (`#[repr(C, align(64))]`):

```rust
/// Representação física de 1 token (128 dims) no arquivo `.osr` (64 bytes alinhados).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OsrTokenCode {
    /// Bytes 0..32: Camada 1 — 64 subespaços empacotados em nibbles.
    /// `layer1[k] = (c1[2*k] & 0x0F) | ((c1[2*k + 1] & 0x0F) << 4)`
    pub layer1: [u8; 32],
    /// Bytes 32..64: Camada 2 (escala gamma=0.28) — 64 subespaços residuais em nibbles.
    /// `layer2[k] = (c2[2*k] & 0x0F) | ((c2[2*k + 1] & 0x0F) << 4)`
    pub layer2: [u8; 32],
}

const _: () = {
    assert!(std::mem::size_of::<OsrTokenCode>() == 64);
    assert!(std::mem::align_of::<OsrTokenCode>() == 64);
};
```

- **Alinhamento no `.osr`:** Como `tensors_offset % 64 == 0` e cada `OsrTokenCode` possui 64 bytes, todo token $j$ do documento inicia em um endereço múltiplo de 64 bytes, eliminando *cache-line splits* e habilitando cargas alinhadas (`vmovdqa64`).

---

## 3. Síntese Online da Query LUT (`< 8 µs`)

Na chegada de uma consulta multi-vetor rotacionada pela FWHT ($Q_{\text{rot}} \in \mathbb{R}^{L_q \times 128}$), pré-computam-se as Lookup Tables `i8` de 16 entradas em um buffer pré-alocado na stack/workspace da consulta:

```rust
/// Tabelas de busca quantizadas de 1 token de query (64 subespaços x 16 centróides).
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct QueryTokenLuts {
    pub lut1: [[i8; 16]; LUTRANK_SUBSPACES], // 1024 bytes
    pub lut2: [[i8; 16]; LUTRANK_SUBSPACES], // 1024 bytes
    pub scale1: f32,
    pub scale2: f32,
}
```

### Algoritmo de Síntese (`< 8 µs` para $L_q \le 32$)

1. Para cada token de query $q_i$ e subespaço $s \in [0, 63]$ (`qx = q_rot[2*s]`, `qy = q_rot[2*s+1]`):
   - Calcula os 4 produtos 1D em X (`px[a] = qx * A1[a]`) e os 4 produtos 1D em Y (`py[b] = qy * A1[b]`) — apenas 8 multiplicações por subespaço!
   - Forma os 16 produtos internos da Camada 1 por adição: `dot1[b * 4 + a] = px[a] + py[b]`, e os da Camada 2 por `dot2[c] = gamma * dot1[c]`.
   - Rastreia os máximos absolutos globais do token: `max1 = max(|dot1|)` e `max2 = max(|dot2|)`.
2. Extrai as escalas simétricas por token:
   - `scale1 = if max1 > 1e-12 { max1 / 127.0 } else { 1.0 }`
   - `scale2 = if max2 > 1e-12 { max2 / 127.0 } else { 1.0 }`
3. Quantiza simetricamente para `i8` em `[-127, +127]` (usando `f32::round_ties_even()` para paridade IEEE-754 bit-exata com `np.round` do oráculo Python):
   - `lut1[s][c] = (dot1[s][c] / scale1).round_ties_even().clamp(-127.0, 127.0) as i8`
   - `lut2[s][c] = (dot2[s][c] / scale2).round_ties_even().clamp(-127.0, 127.0) as i8`

---

## 4. Kernels Assembly / Intrinsics SIMD (`vpshufb` / `vtbl`)

### 4.1 Estratégia de Transposição em Registradores + Prefetching

Como cada subespaço $s \in [0, 63]$ possui sua própria tabela de 16 bytes `lut[s][0..16]`, enquanto o `.osr` armazena tokens contíguos (`[L_d, 64B]`), o kernel processa os tokens do documento em **blocos (tiles) de $T$ tokens** ($T = 16$ em AVX2/NEON; $T = 32$ ou $64$ em AVX-512BW) mantidos inteiramente em registradores/L1:

1. **Software Prefetching:** Ao processar o tile `t .. t + T`, emite `_mm_prefetch::<_MM_HINT_T0>(ptr.add((t + 2 * T) * 64))` para trazer as próximas linhas de cache de 64 bytes antes do uso.
2. **Transposição de Tile em L1/Registradores:** Para cada par de subespaços `(2k, 2k+1)` ($k = 0 \dots 31$), agrupa o byte `k` (Camada 1) e o byte `32 + k` (Camada 2) dos $T$ tokens em vetores SIMD contíguos (`packed_l1_k` e `packed_l2_k`).
   *(Alternativamente, ao avaliar um único par de bytes de $T = 16$ tokens com stride 64B em buffer L1 de $16 \times 64 = 1024\ \text{B}$, um laço sobre `k = 0..32` mantém os 16 acumuladores de token quentes em registradores `ymm`/`zmm`/`q` durante todos os 64 subespaços sem nenhum spill para a memória.)*

### 4.2 Passo a Passo por ISA (Separação de Nibbles, Lookup 1-Ciclo e Acumulação Sem Overflow)

Para um bloco de $T$ tokens no par de subespaços `(even = 2k, odd = 2k + 1)` e uma camada (`layer` $\in \{1, 2\}$):

#### A) x86_64 AVX2 ($T = 16$ tokens por vetor de 128 bits ou $T = 32$ tokens em `ymm`)

```rust
// `bytes`: __m256i contendo o byte `k` de 32 tokens consecutivos do documento
// `mask_0f = _mm256_set1_epi8(0x0F)`
// `ones_i16 = _mm256_set1_epi16(1)`

// 1. Separar nibbles baixos (subespaço par 2k) e altos (subespaço ímpar 2k+1)
let idx_even = _mm256_and_si256(bytes, mask_0f);
// Nota crítica x86: não existe `srli_epi8`; usa-se `srli_epi16(..., 4)` + `& 0x0F`
let idx_odd  = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), mask_0f);

// 2. Carregar tabelas de 16 bytes replicadas nas duas lanes de 128 bits do `ymm`
let lut_even_256 = _mm256_broadcastsi128_si256(load_128(&lut[2 * k]));
let lut_odd_256  = _mm256_broadcastsi128_si256(load_128(&lut[2 * k + 1]));

// 3. Lookup paralelo de 32 entradas em 1 ciclo de clock via `vpshufb`
let val_even_i8 = _mm256_shuffle_epi8(lut_even_256, idx_even); // 32 x i8
let val_odd_i8  = _mm256_shuffle_epi8(lut_odd_256,  idx_odd);  // 32 x i8

// 4. Alargamento `i8 -> i16` e acumulação em `i16`/`i32` livre de overflow:
// Para a metade baixa (tokens 0..15) e alta (tokens 16..31):
let even_lo_16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(val_even_i8));
let odd_lo_16  = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(val_odd_i8));
// Soma dos 2 subespaços para os 16 tokens em `i16` (ou combina pares [even, odd] para `_mm256_madd_epi16`):
let sum_lo_16  = _mm256_add_epi16(even_lo_16, odd_lo_16);
acc16_lo = _mm256_add_epi16(acc16_lo, sum_lo_16);
// Alternativa direta para acumulador `i32` via `_mm256_madd_epi16`:
//   let pairs_lo = _mm256_unpacklo_epi16(even_lo_16, odd_lo_16);
//   let pairs_hi = _mm256_unpackhi_epi16(even_lo_16, odd_lo_16);
//   acc32_0 = _mm256_add_epi32(acc32_0, _mm256_madd_epi16(pairs_lo, ones_i16));
//   acc32_1 = _mm256_add_epi32(acc32_1, _mm256_madd_epi16(pairs_hi, ones_i16));
```
*(Em `i16`, o somatório máximo possível de 64 subespaços é $64 \times 127 = 8128 \ll 32767$, garantindo **zero overflow** antes de promover para `i32`/`f32` ao final dos 64 subespaços!)*

#### B) x86_64 AVX-512BW ($T = 64$ tokens em `zmm`)

- Idêntico ao fluxo AVX2, porém dobrando a largura para 512 bits:
  - `_mm512_and_si512` / `_mm512_srli_epi16::<4>` para separar 64 nibbles pares e 64 ímpares.
  - `_mm512_broadcast_i32x4` replica a LUT de 16 bytes em todas as **4 lanes** de 128 bits do `ZMM`.
  - `_mm512_shuffle_epi8` (`vpshufb`) executa **64 buscas de tabela simultaneamente** em 1 ciclo.
  - `_mm512_cvtepi8_epi16` + `_mm512_add_epi16` (ou `_mm512_madd_epi16` contra `ones_i16`) acumula os 64 tokens sem overflow.

#### C) ARM64 NEON ($T = 16$ tokens em registrador de 128 bits `int8x16_t` / `uint8x16_t`)

```rust
// `bytes`: uint8x16_t contendo o byte `k` de 16 tokens consecutivos
let idx_even = vandq_u8(bytes, vdupq_n_u8(0x0F));
let idx_odd  = vshrq_n_u8(bytes, 4); // ARM NEON possui shift de byte nativo!

let lut_even = vld1q_s8(lut[2 * k].as_ptr());
let lut_odd  = vld1q_s8(lut[2 * k + 1].as_ptr());

// Lookup de 16 entradas via `vtbl` (`vqtbl1q_s8` em AArch64)
let val_even_i8 = vqtbl1q_s8(lut_even, idx_even);
let val_odd_i8  = vqtbl1q_s8(lut_odd,  idx_odd);

// Alargamento `i8 -> i16` com `vmovl_s8` e acumulação segura em `i16`
let sum_lo = vaddq_s16(vmovl_s8(vget_low_s8(val_even_i8)),  vmovl_s8(vget_low_s8(val_odd_i8)));
let sum_hi = vaddq_s16(vmovl_s8(vget_high_s8(val_even_i8)), vmovl_s8(vget_high_s8(val_odd_i8)));
acc16_lo = vaddq_s16(acc16_lo, sum_lo);
acc16_hi = vaddq_s16(acc16_hi, sum_hi);
```

### 4.3 Redução `MaxSim` por Documento e Soma por Query

Ao concluir os $k = 0 \dots 31$ pares de subespaços para os $T$ tokens do tile (para ambas as camadas `acc1` e `acc2`):
1. Promove `acc1[t]` e `acc2[t]` para `f32` uma única vez por token:
   ```rust
   let score_t = scale1 * (acc1[t] as f32) + scale2 * (acc2[t] as f32);
   max_for_query_token = max_for_query_token.max(score_t);
   ```
2. Ao varrer todos os `token_count` tokens do documento no `.osr`, soma `total_maxsim += max_for_query_token` sobre todos os tokens $i = 0 \dots L_q - 1$ da consulta.

---

## 5. Critérios de Validação Cruzada (Portão Obrigatório da Fase 7)

1. **Paridade Bit-a-Bit com `reference/golden/lutrank_golden.bin` (`golden_lutrank_parity`):**
   - Verifica igualdade exata (`<= 1e-6`) entre `build_nested_lattice_codebooks` e `cb_layer1` / `cb_layer2`.
   - Verifica igualdade **bit-a-bit (`==`)** dos bytes `doc_nibbles` gerados pelo codificador Rust em `doc_tokens_rot` contra o arquivo ouro.
   - Verifica igualdade exata (`==` ou `<= 1` LSB por arredondamento tie) das tabelas `q_lut1_i8`, `q_lut2_i8` e escalas `scale1`, `scale2`.
   - Verifica que todos os backends SIMD (`AVX-512BW`, `AVX2`, `ARM64 NEON`, `Scalar`) produzem valores de `maxsim_lut_f32` idênticos entre si (erro `<= 1e-4` vs `maxsim_lut_f32` ouro).
2. **Qualidade Estatística e Recall:**
   - Correlação de Pearson entre `maxsim_lut_f32` e `maxsim_exact_f32` **$> 0.99$** (alvo empírico `0.9950+`).
   - **`Recall@10 == 100.0%`** frente ao baseline `FP32` exato.
3. **Orçamento de Latência e Alocação:**
   - Síntese das LUTs da query em **$< 8\ \mu\text{s}$**.
   - Kernel `MaxSim` sobre **200 documentos** (`~96-200` tokens/doc) em **$< 3.8\ \text{ms}$** em CPU `--release`, com **zero alocações de heap** dentro do loop quente de avaliação de tokens.
