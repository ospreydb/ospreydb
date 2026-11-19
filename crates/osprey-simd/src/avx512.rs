//! AVX-512F vectorized FWHT kernel with L1 cache loop tiling (`L1_TILE_FLOATS = 512`).

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use osprey_core::{OspreyError, OspreyResult};

use crate::avx2::L1_TILE_FLOATS;

/// Applies Rademacher XOR sign masks and orthogonal FWHT in-place using AVX-512F.
///
/// # Safety
/// Caller must ensure the host CPU supports `avx512f`.
#[target_feature(enable = "avx512f")]
pub unsafe fn fwht_rotate_in_place_avx512(
    data: &mut [f32],
    signs_xor_mask: &[u32],
) -> OspreyResult<()> {
    let d = data.len();
    if d < 16 || !d.is_power_of_two() || d != signs_xor_mask.len() {
        return Err(OspreyError::InvalidDimension { dim: d });
    }

    let d_u32 = u32::try_from(d).map_err(|_| OspreyError::InvalidDimension { dim: d })?;
    #[allow(clippy::cast_possible_truncation)]
    let inv_sqrt_d = (1.0_f64 / f64::from(d_u32).sqrt()) as f32;

    if d <= L1_TILE_FLOATS {
        // SAFETY: `d >= 16` is a power of 2 and equals `signs_xor_mask.len()`.
        unsafe {
            fwht_tile_prefix_avx512(data, signs_xor_mask);
        }
        let mut h = 16usize;
        while h < d {
            let is_final = h * 2 == d;
            // SAFETY: `h >= 16` divides `d`.
            unsafe {
                fwht_stage_avx512(data, h, is_final.then_some(inv_sqrt_d));
            }
            h *= 2;
        }
        if d == 16 {
            // SAFETY: `data.len() == 16`.
            unsafe {
                scale_avx512(data, inv_sqrt_d);
            }
        }
    } else {
        // Phase 1 (Local L1-tiled): keep each 512-float (2 KiB) tile hot in L1d.
        for (tile, mask_tile) in data
            .as_chunks_mut::<L1_TILE_FLOATS>()
            .0
            .iter_mut()
            .zip(signs_xor_mask.as_chunks::<L1_TILE_FLOATS>().0.iter())
        {
            // SAFETY: `tile.len() == L1_TILE_FLOATS == 512`.
            unsafe {
                fwht_tile_prefix_avx512(tile, mask_tile);
            }
            let mut h = 16usize;
            while h < L1_TILE_FLOATS {
                // SAFETY: `h >= 16` divides `L1_TILE_FLOATS`.
                unsafe {
                    fwht_stage_avx512(tile, h, None);
                }
                h *= 2;
            }
        }

        // Phase 2 (Global cross-tile): fuse `1/sqrt(d)` into final stage `h == d / 2`.
        let mut h = L1_TILE_FLOATS;
        while h < d {
            let is_final = h * 2 == d;
            // SAFETY: `h >= 512` divides `d`.
            unsafe {
                fwht_stage_avx512(data, h, is_final.then_some(inv_sqrt_d));
            }
            h *= 2;
        }
    }

    Ok(())
}

/// Applies XOR sign masks and intra-register stages `h = 1, 2, 4, 8` over 16-float ZMM blocks.
///
/// # Safety
/// `data.len() == signs_xor_mask.len()` must be a multiple of 16, and `avx512f` must be enabled.
#[target_feature(enable = "avx512f")]
unsafe fn fwht_tile_prefix_avx512(data: &mut [f32], signs_xor_mask: &[u32]) {
    let n = data.len();
    let d_ptr = data.as_mut_ptr();
    let m_ptr = signs_xor_mask.as_ptr();
    let mut i = 0usize;

    while i < n {
        // SAFETY: `i + 16 <= n` because `n` is a multiple of 16.
        unsafe {
            let v = _mm512_loadu_ps(d_ptr.add(i));
            let mask = _mm512_loadu_si512(m_ptr.add(i).cast::<__m512i>());
            let mut x = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(v), mask));

            // Stage h = 1
            let sw1 = _mm512_permute_ps::<0b10_11_00_01>(x);
            let s1 = _mm512_add_ps(x, sw1);
            x = _mm512_mask_sub_ps(s1, 0xAAAA, sw1, x);

            // Stage h = 2
            let sw2 = _mm512_permute_ps::<0b01_00_11_10>(x);
            let s2 = _mm512_add_ps(x, sw2);
            x = _mm512_mask_sub_ps(s2, 0xCCCC, sw2, x);

            // Stage h = 4: swap adjacent 128-bit lanes (0<->1, 2<->3)
            let sw4 = _mm512_shuffle_f32x4::<0b10_11_00_01>(x, x);
            let s4 = _mm512_add_ps(x, sw4);
            x = _mm512_mask_sub_ps(s4, 0xF0F0, sw4, x);

            // Stage h = 8: swap 256-bit halves ([0,1]<->[2,3])
            let sw8 = _mm512_shuffle_f32x4::<0b01_00_11_10>(x, x);
            let s8 = _mm512_add_ps(x, sw8);
            x = _mm512_mask_sub_ps(s8, 0xFF00, sw8, x);

            _mm512_storeu_ps(d_ptr.add(i), x);
        }
        i += 16;
    }
}

/// Executes a single wide butterfly stage (`h >= 16`) with optional fused final scaling.
///
/// # Safety
/// `data.len()` must be a multiple of `2 * h`, with `h >= 16` and `h % 16 == 0`.
#[target_feature(enable = "avx512f")]
unsafe fn fwht_stage_avx512(data: &mut [f32], h: usize, scale: Option<f32>) {
    let d = data.len();
    let ptr = data.as_mut_ptr();
    let step = h * 2;

    if let Some(inv_sqrt_d) = scale {
        let vs = _mm512_set1_ps(inv_sqrt_d);
        let mut base = 0usize;
        while base < d {
            let mut j = base;
            let limit = base + h;
            while j < limit {
                // SAFETY: `j + h + 16 <= base + step <= d`.
                unsafe {
                    let va = _mm512_loadu_ps(ptr.add(j));
                    let vb = _mm512_loadu_ps(ptr.add(j + h));
                    _mm512_storeu_ps(ptr.add(j), _mm512_mul_ps(_mm512_add_ps(va, vb), vs));
                    _mm512_storeu_ps(ptr.add(j + h), _mm512_mul_ps(_mm512_sub_ps(va, vb), vs));
                }
                j += 16;
            }
            base += step;
        }
    } else {
        let mut base = 0usize;
        while base < d {
            let mut j = base;
            let limit = base + h;
            while j < limit {
                // SAFETY: `j + h + 16 <= base + step <= d`.
                unsafe {
                    let va = _mm512_loadu_ps(ptr.add(j));
                    let vb = _mm512_loadu_ps(ptr.add(j + h));
                    _mm512_storeu_ps(ptr.add(j), _mm512_add_ps(va, vb));
                    _mm512_storeu_ps(ptr.add(j + h), _mm512_sub_ps(va, vb));
                }
                j += 16;
            }
            base += step;
        }
    }
}

/// # Safety
/// `data.len()` must be a multiple of 16 and `avx512f` must be enabled.
#[target_feature(enable = "avx512f")]
unsafe fn scale_avx512(data: &mut [f32], inv_sqrt_d: f32) {
    let n = data.len();
    let ptr = data.as_mut_ptr();
    let vs = _mm512_set1_ps(inv_sqrt_d);
    let mut i = 0usize;
    while i < n {
        // SAFETY: `i + 16 <= n`.
        unsafe {
            let v = _mm512_loadu_ps(ptr.add(i));
            _mm512_storeu_ps(ptr.add(i), _mm512_mul_ps(v, vs));
        }
        i += 16;
    }
}

/// Computes Hamming distance `sum(popcnt(a[w] ^ b[w]))` using `AVX-512VPOPCNTDQ`.
///
/// Processes 8 `u64` words (512 bits) per ZMM instruction with 4x unrolling for long vectors,
/// and handles any remaining `< 8` tail words cleanly.
///
/// # Safety
/// Caller must ensure the host CPU supports `avx512f` and `avx512vpopcntdq`.
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
pub unsafe fn hamming_distance_u64_avx512(a: &[u64], b: &[u64]) -> u32 {
    let n = a.len().min(b.len());
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let mut i = 0usize;
    let mut total = 0_u32;

    if n >= 8 {
        // SAFETY: `avx512f,avx512vpopcntdq` are enabled and `i + 8 <= n` on every load.
        unsafe {
            let mut acc0 = _mm512_setzero_si512();
            let mut acc1 = _mm512_setzero_si512();
            let mut acc2 = _mm512_setzero_si512();
            let mut acc3 = _mm512_setzero_si512();

            while i + 32 <= n {
                let va0 = _mm512_loadu_si512(a_ptr.add(i).cast::<__m512i>());
                let vb0 = _mm512_loadu_si512(b_ptr.add(i).cast::<__m512i>());
                acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64(_mm512_xor_si512(va0, vb0)));

                let va1 = _mm512_loadu_si512(a_ptr.add(i + 8).cast::<__m512i>());
                let vb1 = _mm512_loadu_si512(b_ptr.add(i + 8).cast::<__m512i>());
                acc1 = _mm512_add_epi64(acc1, _mm512_popcnt_epi64(_mm512_xor_si512(va1, vb1)));

                let va2 = _mm512_loadu_si512(a_ptr.add(i + 16).cast::<__m512i>());
                let vb2 = _mm512_loadu_si512(b_ptr.add(i + 16).cast::<__m512i>());
                acc2 = _mm512_add_epi64(acc2, _mm512_popcnt_epi64(_mm512_xor_si512(va2, vb2)));

                let va3 = _mm512_loadu_si512(a_ptr.add(i + 24).cast::<__m512i>());
                let vb3 = _mm512_loadu_si512(b_ptr.add(i + 24).cast::<__m512i>());
                acc3 = _mm512_add_epi64(acc3, _mm512_popcnt_epi64(_mm512_xor_si512(va3, vb3)));

                i += 32;
            }

            while i + 8 <= n {
                let va = _mm512_loadu_si512(a_ptr.add(i).cast::<__m512i>());
                let vb = _mm512_loadu_si512(b_ptr.add(i).cast::<__m512i>());
                acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64(_mm512_xor_si512(va, vb)));
                i += 8;
            }

            let sum01 = _mm512_add_epi64(acc0, acc1);
            let sum23 = _mm512_add_epi64(acc2, acc3);
            let sum = _mm512_reduce_add_epi64(_mm512_add_epi64(sum01, sum23));
            total = u32::try_from(sum).unwrap_or(u32::MAX);
        }
    }

    while i < n {
        // SAFETY: `i < n <= a.len().min(b.len())`.
        let diff = unsafe { *a_ptr.add(i) ^ *b_ptr.add(i) };
        total = total.saturating_add(diff.count_ones());
        i += 1;
    }

    total
}

/// Finds `(best_idx, min_hamming_dist)` among `num_centroids` binary centroids using
/// 4x centroid unrolling with `AVX-512VPOPCNTDQ`.
///
/// Ties are broken deterministically in favor of the smallest centroid index.
///
/// # Safety
/// Caller must ensure the host CPU supports `avx512f` and `avx512vpopcntdq`.
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
pub unsafe fn batch_nearest_binary_centroid_avx512(
    vector_bits: &[u64],
    centroids_bits: &[u64],
    words_per_vec: usize,
    num_centroids: usize,
) -> (usize, u32) {
    if words_per_vec == 0 || num_centroids == 0 || vector_bits.len() < words_per_vec {
        return (0, u32::MAX);
    }
    let valid_centroids = num_centroids.min(centroids_bits.len() / words_per_vec);
    if valid_centroids == 0 {
        return (0, u32::MAX);
    }

    let v_ptr = vector_bits.as_ptr();
    let c_base_ptr = centroids_bits.as_ptr();
    let mut best_idx = 0usize;
    let mut best_dist = u32::MAX;
    let mut c = 0usize;

    while c + 4 <= valid_centroids {
        // SAFETY: `(c + 3) < valid_centroids`, so all 4 centroid slices of length `words_per_vec`
        // lie strictly inside `centroids_bits`.
        let (c0_ptr, c1_ptr, c2_ptr, c3_ptr) = unsafe {
            (
                c_base_ptr.add(c * words_per_vec),
                c_base_ptr.add((c + 1) * words_per_vec),
                c_base_ptr.add((c + 2) * words_per_vec),
                c_base_ptr.add((c + 3) * words_per_vec),
            )
        };

        let mut d0 = 0_u32;
        let mut d1 = 0_u32;
        let mut d2 = 0_u32;
        let mut d3 = 0_u32;
        let mut w = 0usize;

        if words_per_vec >= 8 {
            // SAFETY: `avx512f,avx512vpopcntdq` are enabled and `w + 8 <= words_per_vec` on every load.
            unsafe {
                let mut acc0 = _mm512_setzero_si512();
                let mut acc1 = _mm512_setzero_si512();
                let mut acc2 = _mm512_setzero_si512();
                let mut acc3 = _mm512_setzero_si512();

                while w + 8 <= words_per_vec {
                    let vv = _mm512_loadu_si512(v_ptr.add(w).cast::<__m512i>());
                    let vc0 = _mm512_loadu_si512(c0_ptr.add(w).cast::<__m512i>());
                    let vc1 = _mm512_loadu_si512(c1_ptr.add(w).cast::<__m512i>());
                    let vc2 = _mm512_loadu_si512(c2_ptr.add(w).cast::<__m512i>());
                    let vc3 = _mm512_loadu_si512(c3_ptr.add(w).cast::<__m512i>());

                    acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64(_mm512_xor_si512(vv, vc0)));
                    acc1 = _mm512_add_epi64(acc1, _mm512_popcnt_epi64(_mm512_xor_si512(vv, vc1)));
                    acc2 = _mm512_add_epi64(acc2, _mm512_popcnt_epi64(_mm512_xor_si512(vv, vc2)));
                    acc3 = _mm512_add_epi64(acc3, _mm512_popcnt_epi64(_mm512_xor_si512(vv, vc3)));
                    w += 8;
                }

                d0 = u32::try_from(_mm512_reduce_add_epi64(acc0)).unwrap_or(u32::MAX);
                d1 = u32::try_from(_mm512_reduce_add_epi64(acc1)).unwrap_or(u32::MAX);
                d2 = u32::try_from(_mm512_reduce_add_epi64(acc2)).unwrap_or(u32::MAX);
                d3 = u32::try_from(_mm512_reduce_add_epi64(acc3)).unwrap_or(u32::MAX);
            }
        }

        while w < words_per_vec {
            // SAFETY: `w < words_per_vec`.
            unsafe {
                let vw = *v_ptr.add(w);
                d0 = d0.saturating_add((vw ^ *c0_ptr.add(w)).count_ones());
                d1 = d1.saturating_add((vw ^ *c1_ptr.add(w)).count_ones());
                d2 = d2.saturating_add((vw ^ *c2_ptr.add(w)).count_ones());
                d3 = d3.saturating_add((vw ^ *c3_ptr.add(w)).count_ones());
            }
            w += 1;
        }

        if d0 < best_dist {
            best_dist = d0;
            best_idx = c;
        }
        if d1 < best_dist {
            best_dist = d1;
            best_idx = c + 1;
        }
        if d2 < best_dist {
            best_dist = d2;
            best_idx = c + 2;
        }
        if d3 < best_dist {
            best_dist = d3;
            best_idx = c + 3;
        }

        c += 4;
    }

    let v_slice = &vector_bits[..words_per_vec];
    while c < valid_centroids {
        let start = c * words_per_vec;
        let c_slice = &centroids_bits[start..start + words_per_vec];
        // SAFETY: `avx512f,avx512vpopcntdq` are enabled.
        let dist = unsafe { hamming_distance_u64_avx512(v_slice, c_slice) };
        if dist < best_dist {
            best_dist = dist;
            best_idx = c;
        }
        c += 1;
    }

    (best_idx, best_dist)
}

/// Computes the exact `int4 x 1-bit` inner product using AVX-512F + AVX-512VPOPCNTDQ
/// with `vpternlogq` / `xor`+`andnot` and zero heap allocations in the hot loop.
///
/// # Safety
/// Caller must ensure the host CPU supports `avx512f` and `avx512vpopcntdq`.
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
pub unsafe fn rabitq_4bit_dot_u64_avx512(
    query_planes: &[[u64; 4]],
    query_signs: &[u64],
    query_abs_sum: i32,
    db_bits: &[u64],
) -> i32 {
    let n_words = query_planes.len().min(query_signs.len()).min(db_bits.len());
    let qp_ptr = query_planes.as_ptr().cast::<u64>();
    let sign_ptr = query_signs.as_ptr();
    let db_ptr = db_bits.as_ptr();

    let idx_lo = _mm512_set_epi64(13, 9, 12, 8, 5, 1, 4, 0);
    let idx_hi = _mm512_set_epi64(15, 11, 14, 10, 7, 3, 6, 2);
    let mut acc0 = _mm512_setzero_si512();
    let mut acc1 = _mm512_setzero_si512();
    let mut acc2 = _mm512_setzero_si512();
    let mut acc3 = _mm512_setzero_si512();

    let mut w = 0usize;
    while w + 8 <= n_words {
        // SAFETY: `w + 8 <= n_words` guarantees all four `[u64; 4]` blocks (32 `u64`s)
        // and the 8-word `sign_ptr` / `db_ptr` slices are within bounds.
        unsafe {
            if w + 16 <= n_words {
                _mm_prefetch::<_MM_HINT_T0>(db_ptr.add(w + 8).cast::<i8>());
                _mm_prefetch::<_MM_HINT_T0>(qp_ptr.add((w + 8) * 4).cast::<i8>());
            }
            let v_db = _mm512_loadu_si512(db_ptr.add(w).cast::<__m512i>());
            let v_sign = _mm512_loadu_si512(sign_ptr.add(w).cast::<__m512i>());

            // Load 8 x [p0, p1, p2, p3] words and transpose 4x8 -> 4 planar ZMM registers.
            let base = qp_ptr.add(w * 4);
            let r0 = _mm512_loadu_si512(base.cast::<__m512i>());
            let r1 = _mm512_loadu_si512(base.add(8).cast::<__m512i>());
            let r2 = _mm512_loadu_si512(base.add(16).cast::<__m512i>());
            let r3 = _mm512_loadu_si512(base.add(24).cast::<__m512i>());

            let t0 = _mm512_unpacklo_epi64(r0, r1);
            let t1 = _mm512_unpackhi_epi64(r0, r1);
            let t2 = _mm512_unpacklo_epi64(r2, r3);
            let t3 = _mm512_unpackhi_epi64(r2, r3);

            let vp0 = _mm512_permutex2var_epi64(t0, idx_lo, t2);
            let vp2 = _mm512_permutex2var_epi64(t0, idx_hi, t2);
            let vp1 = _mm512_permutex2var_epi64(t1, idx_lo, t3);
            let vp3 = _mm512_permutex2var_epi64(t1, idx_hi, t3);

            // `vpternlogq` imm8=0x90 computes `A & !(B ^ C)` = `plane & !(q_sign ^ db_bits)` in 1 op.
            let m0 = _mm512_ternarylogic_epi64::<0x90>(vp0, v_sign, v_db);
            let m1 = _mm512_ternarylogic_epi64::<0x90>(vp1, v_sign, v_db);
            let m2 = _mm512_ternarylogic_epi64::<0x90>(vp2, v_sign, v_db);
            let m3 = _mm512_ternarylogic_epi64::<0x90>(vp3, v_sign, v_db);

            acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64(m0));
            acc1 = _mm512_add_epi64(acc1, _mm512_popcnt_epi64(m1));
            acc2 = _mm512_add_epi64(acc2, _mm512_popcnt_epi64(m2));
            acc3 = _mm512_add_epi64(acc3, _mm512_popcnt_epi64(m3));
        }
        w += 8;
    }

    let mut s0 = _mm512_reduce_add_epi64(acc0);
    let mut s1 = _mm512_reduce_add_epi64(acc1);
    let mut s2 = _mm512_reduce_add_epi64(acc2);
    let mut s3 = _mm512_reduce_add_epi64(acc3);

    while w < n_words {
        let match_word = !(query_signs[w] ^ db_bits[w]);
        let planes = query_planes[w];
        s0 += i64::from((planes[0] & match_word).count_ones());
        s1 += i64::from((planes[1] & match_word).count_ones());
        s2 += i64::from((planes[2] & match_word).count_ones());
        s3 += i64::from((planes[3] & match_word).count_ones());
        w += 1;
    }

    let weighted = s0 + (s1 << 1) + (s2 << 2) + (s3 << 3);
    let res = weighted.wrapping_mul(2) - i64::from(query_abs_sum);
    i32::try_from(res).unwrap_or(i32::MAX)
}

/// Evaluates one nibble-packed `.osr` document token (32B Layer 1 + 32B Layer 2) against
/// `64 * 16`-entry `i8` query LUTs using AVX-512BW `_mm512_shuffle_epi8` across 4 ZMM lanes,
/// `_mm512_cvtepi8_epi16`, `_mm512_madd_epi16`, and `_mm512_reduce_add_epi32`.
///
/// # Safety
/// Caller must ensure the host CPU supports `avx512f` and `avx512bw`.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn lutrank_token_sim_i32_avx512(
    doc_l1_packed: &[u8],
    doc_l2_packed: &[u8],
    q_lut1_i8: &[i8],
    q_lut2_i8: &[i8],
    scale1: f32,
    scale2: f32,
) -> f32 {
    if doc_l1_packed.len() != 32
        || doc_l2_packed.len() != 32
        || q_lut1_i8.len() < 1024
        || q_lut2_i8.len() < 1024
    {
        return crate::backend::SimdBackend::lutrank_token_sim_i32(
            &crate::backend::ScalarBackend,
            doc_l1_packed,
            doc_l2_packed,
            q_lut1_i8,
            q_lut2_i8,
            scale1,
            scale2,
        );
    }

    let l1_ptr = doc_l1_packed.as_ptr();
    let l2_ptr = doc_l2_packed.as_ptr();
    let ql1_ptr = q_lut1_i8.as_ptr();
    let ql2_ptr = q_lut2_i8.as_ptr();

    let ones_i16 = _mm512_set1_epi16(1);
    let mask_dword_hi = 0x8080_8000_u32.cast_signed();
    let mask_dword_all = 0x8080_8080_u32.cast_signed();

    let mut acc1 = _mm512_setzero_si512();
    let mut acc2 = _mm512_setzero_si512();

    // 16 groups of 4 consecutive subspaces (64 bytes of LUT per ZMM load).
    for g in 0..16 {
        // SAFETY: `g < 16`, so `2 * g + 1 < 32` and `g * 64 + 64 <= 1024`.
        unsafe {
            if g + 2 < 16 {
                _mm_prefetch::<_MM_HINT_T0>(ql1_ptr.add((g + 2) * 64).cast::<i8>());
                _mm_prefetch::<_MM_HINT_T0>(ql2_ptr.add((g + 2) * 64).cast::<i8>());
            }
            let b1_a = *l1_ptr.add(2 * g);
            let b1_b = *l1_ptr.add(2 * g + 1);
            let b2_a = *l2_ptr.add(2 * g);
            let b2_b = *l2_ptr.add(2 * g + 1);

            let lut1 = _mm512_loadu_si512(ql1_ptr.add(g * 64).cast::<__m512i>());
            let lut2 = _mm512_loadu_si512(ql2_ptr.add(g * 64).cast::<__m512i>());

            let idx1 = _mm512_set_epi32(
                mask_dword_all,
                mask_dword_all,
                mask_dword_all,
                i32::from(b1_b >> 4) | mask_dword_hi,
                mask_dword_all,
                mask_dword_all,
                mask_dword_all,
                i32::from(b1_b & 0x0F) | mask_dword_hi,
                mask_dword_all,
                mask_dword_all,
                mask_dword_all,
                i32::from(b1_a >> 4) | mask_dword_hi,
                mask_dword_all,
                mask_dword_all,
                mask_dword_all,
                i32::from(b1_a & 0x0F) | mask_dword_hi,
            );
            let idx2 = _mm512_set_epi32(
                mask_dword_all,
                mask_dword_all,
                mask_dword_all,
                i32::from(b2_b >> 4) | mask_dword_hi,
                mask_dword_all,
                mask_dword_all,
                mask_dword_all,
                i32::from(b2_b & 0x0F) | mask_dword_hi,
                mask_dword_all,
                mask_dword_all,
                mask_dword_all,
                i32::from(b2_a >> 4) | mask_dword_hi,
                mask_dword_all,
                mask_dword_all,
                mask_dword_all,
                i32::from(b2_a & 0x0F) | mask_dword_hi,
            );

            let res1 = _mm512_shuffle_epi8(lut1, idx1);
            let res2 = _mm512_shuffle_epi8(lut2, idx2);

            let w1_lo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(res1));
            let w1_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64::<1>(res1));
            let w2_lo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(res2));
            let w2_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64::<1>(res2));

            acc1 = _mm512_add_epi32(
                acc1,
                _mm512_madd_epi16(_mm512_add_epi16(w1_lo, w1_hi), ones_i16),
            );
            acc2 = _mm512_add_epi32(
                acc2,
                _mm512_madd_epi16(_mm512_add_epi16(w2_lo, w2_hi), ones_i16),
            );
        }
    }

    let sum1 = _mm512_reduce_add_epi32(acc1);
    let sum2 = _mm512_reduce_add_epi32(acc2);
    #[allow(clippy::cast_precision_loss)]
    {
        (sum1 as f32) * scale1 + (sum2 as f32) * scale2
    }
}

/// Computes `MaxSim(Q, D)` over a document's contiguous 64-byte-aligned `.osr` token buffer
/// using 64-token ZMM tiling, software prefetching, `_mm512_shuffle_epi8` across 4 lanes
/// simultaneously, and overflow-safe `i16 -> i32` widening.
///
/// # Safety
/// Caller must ensure the host CPU supports `avx512f` and `avx512bw`.
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn lutrank_doc_maxsim_avx512(
    doc_tokens_bytes: &[u8],
    num_doc_tokens: usize,
    num_subspaces: usize,
    q_lut1_i8: &[i8],
    q_lut2_i8: &[i8],
    num_query_tokens: usize,
    scale1: f32,
    scale2: f32,
) -> f32 {
    if num_doc_tokens == 0 || num_query_tokens == 0 {
        return 0.0;
    }
    if num_subspaces != 64
        || doc_tokens_bytes.len() < num_doc_tokens * 64
        || q_lut1_i8.len() < num_query_tokens * 1024
        || q_lut2_i8.len() < num_query_tokens * 1024
    {
        return crate::backend::SimdBackend::lutrank_doc_maxsim(
            &crate::backend::ScalarBackend,
            doc_tokens_bytes,
            num_doc_tokens,
            num_subspaces,
            q_lut1_i8,
            q_lut2_i8,
            num_query_tokens,
            scale1,
            scale2,
        );
    }

    let doc_ptr = doc_tokens_bytes.as_ptr();
    let ql1_base = q_lut1_i8.as_ptr();
    let ql2_base = q_lut2_i8.as_ptr();

    let mask_0f = _mm512_set1_epi8(0x0F);
    let zero = _mm512_setzero_si512();
    let vs1 = _mm512_set1_ps(scale1);
    let vs2 = _mm512_set1_ps(scale2);

    let mut total_maxsim = 0.0_f32;
    let mut q_base = 0usize;

    while q_base < num_query_tokens {
        let q_len = (num_query_tokens - q_base).min(64);
        let mut best_per_q = [f32::NEG_INFINITY; 64];

        let mut t_base = 0usize;
        while t_base < num_doc_tokens {
            let tile_len = (num_doc_tokens - t_base).min(64);

            if t_base + 64 < num_doc_tokens {
                // SAFETY: `t_base + 64 < num_doc_tokens`, so `(t_base + 64) * 64` is in-bounds.
                unsafe {
                    let next_tile_tokens = (num_doc_tokens - (t_base + 64)).min(8);
                    let pf = doc_ptr.add((t_base + 64) * 64).cast::<i8>();
                    for k in 0..next_tile_tokens {
                        _mm_prefetch::<_MM_HINT_T0>(pf.add(k * 64));
                    }
                }
            }

            let mut l1_t = [[0_u8; 64]; 32];
            let mut l2_t = [[0_u8; 64]; 32];
            for tok in 0..tile_len {
                // SAFETY: `(t_base + tok) * 64 + 64 <= doc_tokens_bytes.len()`.
                unsafe {
                    let tok_ptr = doc_ptr.add((t_base + tok) * 64);
                    if tok + 4 < tile_len {
                        _mm_prefetch::<_MM_HINT_T0>(tok_ptr.add(256).cast::<i8>());
                    }
                    for s in 0..32 {
                        l1_t[s][tok] = *tok_ptr.add(s);
                        l2_t[s][tok] = *tok_ptr.add(32 + s);
                    }
                }
            }

            for (qi, best_slot) in best_per_q[..q_len].iter_mut().enumerate() {
                // SAFETY: `q_base + qi < num_query_tokens`, `s < 32`, and `l1_t`/`l2_t` are 32x64B.
                unsafe {
                    if qi + 1 < q_len {
                        let next_ql1 = ql1_base.add((q_base + qi + 1) * 1024).cast::<i8>();
                        let next_ql2 = ql2_base.add((q_base + qi + 1) * 1024).cast::<i8>();
                        _mm_prefetch::<_MM_HINT_T0>(next_ql1);
                        _mm_prefetch::<_MM_HINT_T0>(next_ql1.add(64));
                        _mm_prefetch::<_MM_HINT_T0>(next_ql2);
                        _mm_prefetch::<_MM_HINT_T0>(next_ql2.add(64));
                    }
                    let ql1 = ql1_base.add((q_base + qi) * 1024);
                    let ql2 = ql2_base.add((q_base + qi) * 1024);

                    let mut acc1_lo = zero;
                    let mut acc1_hi = zero;
                    let mut acc2_lo = zero;
                    let mut acc2_hi = zero;

                    for s in 0..32 {
                        let b1 = _mm512_loadu_si512(l1_t[s].as_ptr().cast::<__m512i>());
                        let b2 = _mm512_loadu_si512(l2_t[s].as_ptr().cast::<__m512i>());

                        let idx1_e = _mm512_and_si512(b1, mask_0f);
                        let idx1_o = _mm512_and_si512(_mm512_srli_epi16::<4>(b1), mask_0f);
                        let idx2_e = _mm512_and_si512(b2, mask_0f);
                        let idx2_o = _mm512_and_si512(_mm512_srli_epi16::<4>(b2), mask_0f);

                        let lut1_e = _mm512_broadcast_i32x4(_mm_loadu_si128(
                            ql1.add((2 * s) * 16).cast::<__m128i>(),
                        ));
                        let lut1_o = _mm512_broadcast_i32x4(_mm_loadu_si128(
                            ql1.add((2 * s + 1) * 16).cast::<__m128i>(),
                        ));
                        let lut2_e = _mm512_broadcast_i32x4(_mm_loadu_si128(
                            ql2.add((2 * s) * 16).cast::<__m128i>(),
                        ));
                        let lut2_o = _mm512_broadcast_i32x4(_mm_loadu_si128(
                            ql2.add((2 * s + 1) * 16).cast::<__m128i>(),
                        ));

                        let v1e = _mm512_shuffle_epi8(lut1_e, idx1_e);
                        let v1o = _mm512_shuffle_epi8(lut1_o, idx1_o);
                        let v2e = _mm512_shuffle_epi8(lut2_e, idx2_e);
                        let v2o = _mm512_shuffle_epi8(lut2_o, idx2_o);

                        acc1_lo = _mm512_add_epi16(
                            acc1_lo,
                            _mm512_add_epi16(
                                _mm512_cvtepi8_epi16(_mm512_castsi512_si256(v1e)),
                                _mm512_cvtepi8_epi16(_mm512_castsi512_si256(v1o)),
                            ),
                        );
                        acc1_hi = _mm512_add_epi16(
                            acc1_hi,
                            _mm512_add_epi16(
                                _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64::<1>(v1e)),
                                _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64::<1>(v1o)),
                            ),
                        );

                        acc2_lo = _mm512_add_epi16(
                            acc2_lo,
                            _mm512_add_epi16(
                                _mm512_cvtepi8_epi16(_mm512_castsi512_si256(v2e)),
                                _mm512_cvtepi8_epi16(_mm512_castsi512_si256(v2o)),
                            ),
                        );
                        acc2_hi = _mm512_add_epi16(
                            acc2_hi,
                            _mm512_add_epi16(
                                _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64::<1>(v2e)),
                                _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64::<1>(v2o)),
                            ),
                        );
                    }

                    let f1_0 =
                        _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(acc1_lo)));
                    let f1_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                        _mm512_extracti64x4_epi64::<1>(acc1_lo),
                    ));
                    let f1_2 =
                        _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(acc1_hi)));
                    let f1_3 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                        _mm512_extracti64x4_epi64::<1>(acc1_hi),
                    ));

                    let f2_0 =
                        _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(acc2_lo)));
                    let f2_1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                        _mm512_extracti64x4_epi64::<1>(acc2_lo),
                    ));
                    let f2_2 =
                        _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(acc2_hi)));
                    let f2_3 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                        _mm512_extracti64x4_epi64::<1>(acc2_hi),
                    ));

                    let s_0 = _mm512_add_ps(_mm512_mul_ps(f1_0, vs1), _mm512_mul_ps(f2_0, vs2));
                    let s_1 = _mm512_add_ps(_mm512_mul_ps(f1_1, vs1), _mm512_mul_ps(f2_1, vs2));
                    let s_2 = _mm512_add_ps(_mm512_mul_ps(f1_2, vs1), _mm512_mul_ps(f2_2, vs2));
                    let s_3 = _mm512_add_ps(_mm512_mul_ps(f1_3, vs1), _mm512_mul_ps(f2_3, vs2));

                    if tile_len == 64 {
                        let m01 = _mm512_max_ps(s_0, s_1);
                        let m23 = _mm512_max_ps(s_2, s_3);
                        let tile_best = _mm512_reduce_max_ps(_mm512_max_ps(m01, m23));
                        if tile_best > *best_slot {
                            *best_slot = tile_best;
                        }
                    } else {
                        let mut scores = [0.0_f32; 64];
                        _mm512_storeu_ps(scores.as_mut_ptr(), s_0);
                        _mm512_storeu_ps(scores.as_mut_ptr().add(16), s_1);
                        _mm512_storeu_ps(scores.as_mut_ptr().add(32), s_2);
                        _mm512_storeu_ps(scores.as_mut_ptr().add(48), s_3);
                        for &sc in &scores[..tile_len] {
                            if sc > *best_slot {
                                *best_slot = sc;
                            }
                        }
                    }
                }
            }

            t_base += 64;
        }

        for &b in &best_per_q[..q_len] {
            total_maxsim += b;
        }
        q_base += 64;
    }

    total_maxsim
}
