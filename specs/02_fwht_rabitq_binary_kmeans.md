# `02_fwht_rabitq_binary_kmeans.md` — Especificação Algorítmica e SIMD das Fases 4, 5 e 6 (`FWHT`, `1-Bit Binary K-Means` e `RaBitQ`)

> **Status:** Especificação Fechada para Implementação (Fases 4, 5 e 6)
> **Nota de Contexto:** Este documento integra as especificações originais de engenharia de engenharia do OspreyDB; a implementação de produção reside em `crates/osprey-core` e `crates/osprey-simd`.
> **Escopo:** `crates/osprey-core` (`fwht.rs`, `binary_kmeans.rs`, `rabitq.rs`) e kernels vetorizados em `crates/osprey-simd` (`AVX-512VPOPCNTDQ`, `AVX-512BW/F`, `AVX2`, `ARM64 NEON` e fallback escalar).
> **Invariantes Obrigatórias:** Zero `unwrap()`/`expect()`/`panic!()` em produção; **zero alocação de heap dentro de loops SIMD**; paridade bit-exata com [`reference/poc_reference.py`](../reference/poc_reference.py) e [`reference/generate_golden_vectors.py`](../reference/generate_golden_vectors.py).

---

## 1. Gerador `SplitMix64` e Rotação Ortogonal FWHT (`fwht.rs`)

### 1.1 Gerador Determinístico `SplitMix64` Bit-Exato

O gerador de sinais de Rademacher $\{-1.0, +1.0\}^{d_{\text{padded}}}$ em Rust **deve** reproduzir bit a bit a função `splitmix64_step` do oráculo Python, eliminando qualquer divergência entre PRNGs de linguagens distintas:

```rust
/// PRNG SplitMix64 determinístico para geração de sinais de Rademacher pós-seed.
#[derive(Clone, Copy, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    #[inline]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Preenche `out_words` (`[u64; padded_dim / 64]`) com máscaras de sinais.
    /// Convenção: bit `b` == 0 => `-1.0`; bit `b` == 1 => `+1.0`.
    #[inline]
    pub fn fill_sign_words(&mut self, out_words: &mut [u64]) {
        for word in out_words.iter_mut() {
            *word = self.next_u64();
        }
    }
}
```

**Aplicação da Máscara de Sinais em Zero Multiplicações:**
Em IEEE-754 `f32`, inverter o sinal equivale a aplicar `XOR` com `0x8000_0000` (quando o bit de sinal correspondente em `sign_words` for `0`). Pré-expandindo `sign_words` para uma máscara `sign_xor_masks: [u32; padded_dim]` fora do hot loop (ou em registradores via broadcast/test bit), a aplicação de $D_{\text{signs}}$ reduz-se a uma única instrução vetorial `vxorps` / `veorq_u32`.

### 1.2 Assinaturas e Implementação Vetorizada da FWHT

O kernel opera *in-place* sobre um buffer alinhado a 64 bytes de comprimento `padded_dim = orig_dim.next_power_of_two()` (previamente preenchido com zeros em `[orig_dim..padded_dim]`):

```rust
/// Rotaciona `buf` (comprimento `d_padded = 2^k`) in-place aplicando
/// `P x = (1 / sqrt(d_padded)) * H * D_signs * x` sem alocar heap.
pub trait FwhtKernel {
    fn fwht_rotate_inplace(buf: &mut [f32], sign_words: &[u64]) -> OspreyResult<()>;
}
```

#### Estratégia de Estágios *Butterfly* por ISA (Sem Alocação)

1. **Estágios Intra-Registrador ($h \in \{1, 2, 4, \dots, W/2\}$ onde $W$ é a largura SIMD em `f32`):**
   - **Scalar Fallback:** Loop padrão `for base in (0..d).step_by(2*h) { for j in 0..h { let a = buf[base+j]; let b = buf[base+j+h]; buf[base+j] = a+b; buf[base+j+h] = a-b; } }`.
   - **AVX2 ($W = 8$ floats em `ymm`):**
     - Carrega bloco de 8 floats (`_mm256_load_ps` alinhado).
     - $h = 1$: permuta pares adjacentes via `_mm256_permute_ps::<0b10_11_00_01>(v)` e aplica sinal alternado `[+1, -1, +1, -1, ...]` via `vxorps` + `_mm256_addsub_ps` ou `_mm256_add_ps`/`_mm256_sub_ps` com blend `0b1010_1010`.
     - $h = 2$: permuta metades de 64 bits via `_mm256_permute_ps::<0b01_00_11_10>(v)` + `_mm256_add_ps`/`_mm256_sub_ps` + `_mm256_blend_ps::<0b1100_1100>`.
     - $h = 4$: permuta lanes de 128 bits via `_mm256_permute2f128_ps::<0x01>(v, v)` + `_mm256_add_ps`/`_mm256_sub_ps` + `_mm256_blend_ps::<0b1111_0000>`.
   - **AVX-512F ($W = 16$ floats em `zmm`):**
     - Estágios $h = 1, 2, 4$ análogos em 512 bits via `_mm512_permute_ps` + `_mm512_mask_add_ps` / `_mm512_mask_sub_ps` (usando máscaras `0x5555`, `0x3333`, `0x0F0F`).
     - Estágio $h = 8$: permuta metades de 256 bits via `_mm512_shuffle_f32x4::<0b01_00_11_10>(v, v)` + `_mm512_mask_add_ps` / `_mm512_mask_sub_ps` com máscara `0x00FF` / `0xFF00`.
   - **ARM64 NEON ($W = 4$ floats em `float32x4_t`):**
     - $h = 1$: `vtrn1q_f32` / `vtrn2q_f32` + `vaddq_f32` / `vsubq_f32` + re-intercalação `vzip1q_f32` / `vzip2q_f32`.
     - $h = 2$: troca de metades de 64 bits via `vextq_f32::<2>(v, v)` + `vaddq_f32` / `vsubq_f32` + `vbslq_f32`.

2. **Estágios Inter-Registrador ($h \ge W$) e *Loop Tiling* em Cache L1 para $d \ge 1024$:**
   - Para $d \ge 1024$ (ex.: $d = 1024 \to 4\ \text{KB}$, $d = 2048 \to 8\ \text{KB}$, ou lotes de múltiplos vetores), a FWHT aplica **loop tiling em L1**:
     ```text
     const L1_TILE_FLOATS: usize = 1024; // 4 KB — reside 100% em L1D (32-48 KB)
     // Passo A: Para cada tile contíguo T de tamanho min(d_padded, L1_TILE_FLOATS):
     //   Executar TODOS os estágios h = 1, 2, 4, ..., T/2 mantendo o tile quente em L1/registradores!
     // Passo B: Para estágios globais h = L1_TILE_FLOATS, 2*L1_TILE_FLOATS, ..., d_padded/2:
     //   Varrer pares de blocos distantes `h` em passos SIMD lineares de W floats:
     //     va = load(ptr + i); vb = load(ptr + i + h);
     //     store(ptr + i, add(va, vb)); store(ptr + i + h, sub(va, vb));
     // Passo C: Normalizar todo o vetor por `inv_sqrt_d = 1.0 / (d_padded as f32).sqrt()` via vmulps.
     ```
   - Esse particionamento garante que os primeiros $\log_2(1024) = 10$ estágios tenham **taxa de acerto de 100% em L1D**, reduzindo tráfego L2/L3 em até $3\times$ para vetores de 1536/2048 dimensões.

---

## 2. Motor `1-Bit Binary K-Means` (`binary_kmeans.rs`)

### 2.1 Binarização e Inicialização Determinística `1-Bit K-Means++`

Dado o conjunto rotacionado $X_{\text{rot}} \in \mathbb{R}^{N \times d_{\text{pad}}}$:
1. **Binarização:** Cada vetor é empacotado em `[u64; W]` ($W = d_{\text{pad}} / 64$) com bit $b = 1 \iff \tilde{x}_i \ge 0.0$.
2. **Inicialização `1-Bit K-Means++` por Distância de Hamming ao Quadrado:**
   - Seleciona o centróide inicial $c_0$ via `SplitMix64::next_u64() % N`.
   - Mantém o vetor `min_dist_sq: Vec<u64>` (pré-alocado uma única vez fora do loop) com $D_H(x_i, c_{\text{nearest}})^2$, onde $D_H(x, c) = \sum_{w=0}^{W-1} \text{popcnt}(x_w \oplus c_w)$.
   - Para $k = 1 \dots K - 1$: sorteia um limiar $u \in [0, \sum_i \text{min\_dist\_sq}[i])$ usando `SplitMix64`, seleciona $c_k$ por soma de prefixos e atualiza `min_dist_sq` contra $c_k$ em paralelo.

### 2.2 Kernel SIMD `XNOR + POPCNT` Desenrolado $4\times$

Durante a atribuição iterativa, minimizar a distância de Hamming $D_H(x, c_k)$ equivale a maximizar a concordância $\text{XNOR}$. O kernel interno desenrola **4 centróides simultaneamente** (ou 4 vetores de banco contra 1 centróide) para esconder a latência de 3 ciclos de `popcnt` / `VPOPCNTDQ`:

```rust
/// Calcula as distâncias de Hamming entre 1 vetor `x_words` (`[u64; W]`) e 4 centróides
/// `[c0, c1, c2, c3]` simultaneamente com zero alocação de heap.
#[inline]
pub fn hamming_4x_avx512(
    x: &[u64],
    c0: &[u64],
    c1: &[u64],
    c2: &[u64],
    c3: &[u64],
) -> [u32; 4] {
    // Pseudocódigo AVX-512VPOPCNTDQ (8 palavras u64 = 512 bits por iteração):
    // acc0 = _mm512_setzero_si512(); ... acc3 = _mm512_setzero_si512();
    // for chunk in (0..W).step_by(8):
    //   vx = _mm512_load_si512(x[chunk..])
    //   acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64(_mm512_xor_si512(vx, load(c0))))
    //   acc1 = _mm512_add_epi64(acc1, _mm512_popcnt_epi64(_mm512_xor_si512(vx, load(c1))))
    //   acc2 = _mm512_add_epi64(acc2, _mm512_popcnt_epi64(_mm512_xor_si512(vx, load(c2))))
    //   acc3 = _mm512_add_epi64(acc3, _mm512_popcnt_epi64(_mm512_xor_si512(vx, load(c3))))
    // Reduz horizontalmente _mm512_reduce_add_epi64(acc0..3) -> [u32; 4]
    // Implementação completa em `crates/osprey-simd/src/avx512.rs`
    [0; 4]
}
```
- **Fallback AVX2 (sem `VPOPCNTDQ`):** Utiliza a técnica clássica *Harley-Seal / `vpshufb` nibble-popcount* (`low = x & 0x0F`, `high = (x >> 4) & 0x0F`, lookup em LUT de 16 bytes `[0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4]`, soma de bytes via `_mm256_sad_epu8` contra zero).
- **Fallback ARM64 NEON:** `veorq_u8` $\to$ `vcntq_u8` $\to$ `vpaddlq_u8` $\to$ `vpaddlq_u16` $\to$ `vpaddlq_u32` (acumulado por bloco para evitar saturação de `u8`).

### 2.3 Acumuladores Thread-Local (`rayon`), *Empty Cluster Splitting* e `f16`

1. **Acumuladores Thread-Local sem Contenção:**
   - Usamos `rayon::slice::ParallelSlice::par_chunks` com `fold_with` / `map_init` pré-alocando **por worker thread** um único par `(votes: Vec<i32>, counts: Vec<u32>)` de tamanho `K * d_pad` e `K`.
   - Para cada vetor atribuído ao cluster $k$, soma `+1` ou `-1` nos `d_pad` bits da linha $k$ de `votes` e incrementa `counts[k]`. A redução final (`reduce`) soma os buffers de thread determinística e sequencialmente por índice de chunk.
2. ***Empty Cluster Splitting*:**
   - Ao final da redução, se existir algum cluster vazio (`counts[k] == 0`):
     - Seleciona-se o maior cluster $k_{\max} = \arg\max_j \text{counts}[j]$ (desempate por menor índice).
     - Entre os membros de $k_{\max}$, localiza-se o vetor $x^*$ com **maior distância de Hamming** até o centróide atual $c_{k_{\max}}$.
     - Reatribui-se $x^*$ para fundar o novo centróide $c_k \leftarrow \text{sign}(x^*)$ e perturba-se $c_{k_{\max}}$ subtraindo a contribuição de $x^*$.
3. **Materialização Final dos Centróides em `f16`:**
   - Após a convergência das iterações 1-bit, calcula-se a média exata dos vetores rotacionados $X_{\text{rot}}$ atribuídos a cada cluster, sanitiza-se contra não-finitos e converte-se cada coordenada com `half::f16::from_f32(val)` para a Tabela de Centróides `.osi` (`AlignedBuffer<u16>` de tamanho `K * d_pad`).
   - Executa-se uma passagem final de reatribuição exata contra os centróides materializados em `f16` (promovidos via `.to_f32()`) garantindo consistência estrita com a extração de resíduos do `RaBitQ`.

---

## 3. Quantizador e Scanner Assimétrico `RaBitQ` (`rabitq.rs`)

### 3.1 Codificação dos Resíduos e Empacotamento Little-Endian

Para cada vetor rotacionado $\tilde{x}$ atribuído ao centróide $c_k$ (promovido de `f16` via `.to_f32()`):
1. Resíduo: $r_i = \tilde{x}_i - c_{k,i}$ (em `f32`, ou acumulação `f64`).
2. Norma euclidiana: `norm = sqrt(sum(r_i^2))`.
   - Se `norm > 1e-12`: $u_i = r_i / \text{norm}$. Caso contrário, $u_i = 1/\sqrt{d_{\text{pad}}}$.
3. Empacotamento `rabitq_code`: `[u64; d_pad / 64]`, onde para `word = i / 64`, `bit = i % 64`, seta-se `1u64 << bit` se $u_i \ge 0.0$.
4. Sanitização e clamping anti-subnormal ([`specs/00_math_invariants.md`](./00_math_invariants.md) §3.2):
   - `l1_avg = sum(|u_i|) / (d_pad as f32)`
   - `res_norm_f16 = half::f16::from_f32(norm.clamp(6.103515625e-5, 65500.0))` (ou `0.0` se `norm == 0.0` exato, mantendo `l1_norm_factor >= 6.103515625e-5`)
   - `l1_factor_f16 = half::f16::from_f32(l1_avg.clamp(6.103515625e-5, 65500.0))`

### 3.2 Quantização Simétrica `int4` ($[-7, +7]$) e Transposição Rápida em 4 Planos de Bits

Na fase online, dada a query rotacionada `q_rot: &[f32]` (comprimento `d_pad`):
1. **Quantização Simétrica `int4`:**
   - `q_max = max(|q_rot|)`
   - `q_scale = if q_max > 1e-12 { q_max / 7.0 } else { 1.0 }`
   - `q_int4[i] = (q_rot[i] / q_scale).round_ties_even().clamp(-7.0, 7.0) as i8` *(usa `f32::round_ties_even()` para paridade IEEE-754 bit-exata com `np.round` do oráculo Python)*
2. **Transposição Rápida em 4 Planos de Bits (`[u64; W]` por plano) + Máscara de Sinal:**
   - Deslocamento positivo sem sinal: `q_u4[i] = (q_int4[i] + 7) as u8` $\in [0, 14]$, acumulando `sum_u4 = sum(q_u4)`.
   - Opcionalmente extrai-se também a máscara de sinal `sign_mask[w]` (`q_int4[i] < 0`) para diagnósticos.
   - **Algoritmo de Transposição Rápida (Zero Loops por Bit em Escalares):**
     Processando blocos de 32 elementos `q_u4` (AVX2) ou 64 elementos (AVX-512BW):
     ```text
     // AVX2: cada bloco de 32 bytes gera a metade baixa ou alta de um u64 para cada um dos 4 planos
     // Desloca `bit b` para o bit mais significativo (bit 7) do byte e extrai 32 bits em 1 ciclo:
     //   plane_0_half = _mm256_movemask_epi8(_mm256_slli_epi16(v, 7)) as u32
     //   plane_1_half = _mm256_movemask_epi8(_mm256_slli_epi16(v, 6)) as u32
     //   plane_2_half = _mm256_movemask_epi8(_mm256_slli_epi16(v, 5)) as u32
     //   plane_3_half = _mm256_movemask_epi8(_mm256_slli_epi16(v, 4)) as u32
     // Em AVX-512BW: `_mm512_movepi8_mask(_mm512_slli_epi16(v, 7 - b))` produz o `u64` completo em 1 instrução!
     ```
   - Complexidade total da transposição para $d_{\text{pad}} = 2048$: apenas $32$ blocos de 64 bytes $\approx 128$ instruções (~$< 0.5\ \mu\text{s}$).

### 3.3 Kernel SIMD de Produto Interno `4-bit × 1-bit`, Tombstones e `BoundedTopKHeap`

Para cada vetor `code: &[u64; W]` no Stream 1 de um cluster sondado $k$ (com score de centróide `c_dot_k`):

1. **Produto Interno Bitwise Ponderado por $1, 2, 4, 8$:**
   ```rust
   // Em AVX-512VPOPCNTDQ (ou fallback AVX2 vpshufb / NEON vcntq_u8):
   // Acumula popcounts sobre todos os W words de 64 bits:
   //   pop_b  = sum_w popcnt(code[w])
   //   dot_p0 = sum_w popcnt(code[w] & plane0[w])
   //   dot_p1 = sum_w popcnt(code[w] & plane1[w])
   //   dot_p2 = sum_w popcnt(code[w] & plane2[w])
   //   dot_p3 = sum_w popcnt(code[w] & plane3[w])
   // Soma ponderada:
   //   dot_u4_b = dot_p0 + (dot_p1 << 1) + (dot_p2 << 2) + (dot_p3 << 3)
   // Reconstrução inteira exata <q_int4, sign(u)>:
   //   dot_int4_sign = 2 * dot_u4_b - sum_u4 - 14 * pop_b + 7 * (padded_dim as i32)
   ```
   - **Nota sobre AVX2 `vpshufb`:** Quando `AVX-512VPOPCNTDQ` não está disponível, o kernel calcula `popcnt` de `code & plane_b` via `_mm256_shuffle_epi8(popcnt_lut_16, nibble)` + `_mm256_sad_epu8` dentro do loop de `W`, mantendo 4 acumuladores vetoriais (1 por plano) + 1 para `pop_b` com zero ramificações.

2. **Reconstrução do Score Assimétrico e Poda via `RoaringBitmap`:**
   - Lê `(res_norm, l1_norm_factor)` do Stream 2 (`RabitqFactors`) e calcula:
     ```rust
     let scale_factor = (q_scale * (dot_int4_sign as f32))
         / ((padded_dim as f32) * l1_norm_factor.to_f32());
     let approx_score = c_dot_k + res_norm.to_f32() * scale_factor;
     ```
   - **Filtragem Instantânea:** Antes de inserir no heap (ou comparando contra o limiar mínimo atual `heap.min_score()` para pular a busca de `doc_id` quando `approx_score <= heap.min_score()`!), verifica `!tombstones.contains(doc_id)`.

3. **Fila de Prioridade Fixa na Stack / Pré-Alocada (`BoundedTopKHeap`):**
   ```rust
   /// Min-heap binário de capacidade fixa `K` com buffer pré-alocado fora do hot loop,
   /// garantindo zero alocações durante toda a varredura dos clusters.
   pub struct BoundedTopKHeap {
       entries: Vec<Stage1Candidate>, // `Vec::with_capacity(top_c)` alocado antes do scan
       capacity: usize,
   }
   ```
   - Se `entries.len() < capacity`: insere e executa `sift_up` $O(\log C)$.
   - Caso contrário, se `cand.approx_score > entries[0].approx_score`: substitui `entries[0]` e executa `sift_down` $O(\log C)$. Caso contrário, descarta em $O(1)$ com **zero escrita de memória**.

---

## 4. Critérios de Validação Cruzada (Portões Obrigatórios das Fases 4, 5 e 6)

Nenhum módulo avança de fase sem aprovação integral dos testes determinísticos abaixo em todas as arquiteturas do CI (`AVX-512`, `AVX2`, `NEON` e `Scalar`):

### 4.1 Testes Contra Vetores Ouro (`reference/golden/`)

1. **Fase 4 — FWHT (`golden_fwht_parity`):**
   - Carrega todos os 7 arquivos `reference/golden/fwht_d{64,128,256,512,768,1024,1536}.bin`.
   - Verifica igualdade **bit-exata (`==`)** entre `SplitMix64::fill_sign_words` e `sign_words` do arquivo ouro.
   - Verifica que todos os backends (`AVX-512`, `AVX2`, `NEON`, `Scalar`) produzem saídas rotacionadas com erro máximo absoluto `<= 1e-5` contra `outputs` em `f32`.
2. **Fase 5 — `1-Bit Binary K-Means` (`golden_binary_kmeans_parity`):**
   - Carrega `reference/golden/binary_kmeans_golden.bin` ($N=512, d=128, K=16$).
   - Verifica equivalência bit-a-bit da distância de Hamming entre `hamming_4x_*` (todos os backends SIMD) e o fallback escalar.
   - Verifica que o algoritmo `1-Bit Binary K-Means` converge com perda de distorção intra-cluster `< 1%` e materializa centróides `f16` válidos e livres de clusters vazios.
3. **Fase 6 — `RaBitQ` Assimétrico (`golden_rabitq_parity`):**
   - Carrega `reference/golden/rabitq_golden.bin` ($N=512, d=128, K=16, N_q=16$).
   - Verifica igualdade **bit-exata (`==`)** para `rabitq_code`, `res_norm` (`f16`), `l1_norm_factor` (`f16`), `q_int4` e `q_bit_planes` (`[4, 2]`).
   - Verifica erro máximo `<= 1e-4` entre os scores estimados pelo kernel SIMD `VPOPCNTDQ`/`vpshufb`/`NEON`/`Scalar` e `asym_scores`, além de correspondência exata dos IDs em `top10_ids_f16`.

### 4.2 Teste de Integração End-to-End (20.000 Vetores Anisotrópicos)

- **`ann_stage1_plus_f16_recall_20k`:**
  - Gera corpus anisotrópico determinístico de **$N = 20.000$ vetores** ($d = 384 \to d_{\text{pad}} = 512$, $K = 128$ clusters, `nprobe = 24`, Top-150 candidatos refinados em `f16`) e $100$ consultas.
  - Exige **`Recall@10 >= 96.5%`** frente ao ground-truth exato em `FP32`, validando simultaneamente exclusão lógica via `RoaringBitmap` de tombstones e execução com `0` alocações dentro do loop SIMD de varredura.
