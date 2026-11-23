//! ARM64 NEON vectorized FWHT kernel with L1 cache loop tiling (`L1_TILE_FLOATS = 512`).

use std::arch::aarch64::*;

use osprey_core::{OspreyError, OspreyResult};

/// L1 data cache tile size in `f32` elements (`512 * 4 = 2048` bytes = 2 KiB).
pub const L1_TILE_FLOATS: usize = 512;

/// Applies Rademacher XOR sign masks and orthogonal FWHT in-place using ARM64 NEON.
///
/// # Safety
/// Caller must ensure the host CPU supports `neon`.
#[target_feature(enable = "neon")]
pub unsafe fn fwht_rotate_in_place_neon(
    data: &mut [f32],
    signs_xor_mask: &[u32],
) -> OspreyResult<()> {
    let d = data.len();
    if d < 4 || !d.is_power_of_two() || d != signs_xor_mask.len() {
        return Err(OspreyError::InvalidDimension { dim: d });
    }

    let d_u32 = u32::try_from(d).map_err(|_| OspreyError::InvalidDimension { dim: d })?;
    #[allow(clippy::cast_possible_truncation)]
    let inv_sqrt_d = (1.0_f64 / f64::from(d_u32).sqrt()) as f32;

    if d <= L1_TILE_FLOATS {
        // SAFETY: `d >= 4` is a power of 2 and equals `signs_xor_mask.len()`.
        unsafe {
            fwht_tile_prefix_neon(data, signs_xor_mask);
        }
        let mut h = 4usize;
        while h < d {
            let is_final = h * 2 == d;
            // SAFETY: `h >= 4` divides `d`.
            unsafe {
                fwht_stage_neon(data, h, is_final.then_some(inv_sqrt_d));
            }
            h *= 2;
        }
        if d == 4 {
            // SAFETY: `data.len() == 4`.
            unsafe {
                scale_neon(data, inv_sqrt_d);
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
                fwht_tile_prefix_neon(tile, mask_tile);
            }
            let mut h = 4usize;
            while h < L1_TILE_FLOATS {
                // SAFETY: `h >= 4` divides `L1_TILE_FLOATS`.
                unsafe {
                    fwht_stage_neon(tile, h, None);
                }
                h *= 2;
            }
        }

        // Phase 2 (Global cross-tile): fuse `1/sqrt(d)` into the final pass `h == d / 2`.
        let mut h = L1_TILE_FLOATS;
        while h < d {
            let is_final = h * 2 == d;
            // SAFETY: `h >= 512` divides `d`.
            unsafe {
                fwht_stage_neon(data, h, is_final.then_some(inv_sqrt_d));
            }
            h *= 2;
        }
    }

    Ok(())
}

/// Applies XOR sign masks and intra-register stages `h = 1, 2` over 4-float NEON Q-registers.
///
/// # Safety
/// `data.len() == signs_xor_mask.len()` must be a multiple of 4, and `neon` must be enabled.
#[target_feature(enable = "neon")]
unsafe fn fwht_tile_prefix_neon(data: &mut [f32], signs_xor_mask: &[u32]) {
    let n = data.len();
    let d_ptr = data.as_mut_ptr();
    let m_ptr = signs_xor_mask.as_ptr();
    let mut i = 0usize;

    while i < n {
        // SAFETY: `i + 4 <= n` because `n` is a multiple of 4.
        unsafe {
            let v = vld1q_f32(d_ptr.add(i));
            let mask = vld1q_u32(m_ptr.add(i));
            let x = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v), mask));

            // Stage h = 1: transpose pairs [x0, x2, x0, x2] & [x1, x3, x1, x3]
            let t1 = vtrn1q_f32(x, x); // [x0, x0, x2, x2]
            let t2 = vtrn2q_f32(x, x); // [x1, x1, x3, x3]
            let s1 = vaddq_f32(t1, t2); // [x0+x1, x0+x1, x2+x3, x2+x3]
            let d1 = vsubq_f32(t1, t2); // [x0-x1, x0-x1, x2-x3, x2-x3]
            let x1 = vtrn1q_f32(s1, d1); // [x0+x1, x0-x1, x2+x3, x2-x3]

            // Stage h = 2: 64-bit halves [lo, hi]
            let lo = vget_low_f32(x1);
            let hi = vget_high_f32(x1);
            let s2 = vadd_f32(lo, hi);
            let d2 = vsub_f32(lo, hi);
            let x2 = vcombine_f32(s2, d2);

            vst1q_f32(d_ptr.add(i), x2);
        }
        i += 4;
    }
}

/// Executes a single wide butterfly stage (`h >= 4`) with optional fused final scaling.
///
/// # Safety
/// `data.len()` must be a multiple of `2 * h`, with `h >= 4` and `h % 4 == 0`.
#[target_feature(enable = "neon")]
unsafe fn fwht_stage_neon(data: &mut [f32], h: usize, scale: Option<f32>) {
    let d = data.len();
    let ptr = data.as_mut_ptr();
    let step = h * 2;

    if let Some(inv_sqrt_d) = scale {
        let mut base = 0usize;
        while base < d {
            let mut j = base;
            let limit = base + h;
            while j < limit {
                // SAFETY: `j + h + 4 <= base + step <= d`.
                unsafe {
                    let va = vld1q_f32(ptr.add(j));
                    let vb = vld1q_f32(ptr.add(j + h));
                    vst1q_f32(ptr.add(j), vmulq_n_f32(vaddq_f32(va, vb), inv_sqrt_d));
                    vst1q_f32(ptr.add(j + h), vmulq_n_f32(vsubq_f32(va, vb), inv_sqrt_d));
                }
                j += 4;
            }
            base += step;
        }
    } else {
        let mut base = 0usize;
        while base < d {
            let mut j = base;
            let limit = base + h;
            while j < limit {
                // SAFETY: `j + h + 4 <= base + step <= d`.
                unsafe {
                    let va = vld1q_f32(ptr.add(j));
                    let vb = vld1q_f32(ptr.add(j + h));
                    vst1q_f32(ptr.add(j), vaddq_f32(va, vb));
                    vst1q_f32(ptr.add(j + h), vsubq_f32(va, vb));
                }
                j += 4;
            }
            base += step;
        }
    }
}

/// # Safety
/// `data.len()` must be a multiple of 4 and `neon` must be enabled.
#[target_feature(enable = "neon")]
unsafe fn scale_neon(data: &mut [f32], inv_sqrt_d: f32) {
    let n = data.len();
    let ptr = data.as_mut_ptr();
    let mut i = 0usize;
    while i < n {
        // SAFETY: `i + 4 <= n`.
        unsafe {
            let v = vld1q_f32(ptr.add(i));
            vst1q_f32(ptr.add(i), vmulq_n_f32(v, inv_sqrt_d));
        }
        i += 4;
    }
}

/// Computes Hamming distance `sum(popcnt(a[w] ^ b[w]))` using ARM64 NEON (`vcntq_u8`).
///
/// Loads 128 bits (2 `u64` words) per iteration via `vld1q_u8`, XORs via `veorq_u8`, counts
/// byte popcounts via `vcntq_u8`, and flushes every 16 iterations via
/// `vpaddlq_u8` -> `vpaddlq_u16` -> `vaddq_u32` -> `vaddvq_u32` to avoid `u8` saturation.
///
/// # Safety
/// Caller must ensure the host CPU supports `neon`.
#[target_feature(enable = "neon")]
pub unsafe fn hamming_distance_u64_neon(a: &[u64], b: &[u64]) -> u32 {
    const FLUSH_WORDS: usize = 32; // 16 NEON iterations * 8 max bits/byte = 128 <= 255
    let n = a.len().min(b.len());
    let pairs = n & !1;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    // SAFETY: `neon` is enabled and `i + 2 <= pairs <= n` on every 16-byte load.
    let mut total = unsafe {
        let mut acc_u32 = vdupq_n_u32(0);
        let mut i = 0usize;
        while i < pairs {
            let block_end = (i + FLUSH_WORDS).min(pairs);
            let mut acc_u8 = vdupq_n_u8(0);
            while i < block_end {
                let va = vld1q_u8(a_ptr.add(i).cast::<u8>());
                let vb = vld1q_u8(b_ptr.add(i).cast::<u8>());
                let diff = veorq_u8(va, vb);
                acc_u8 = vaddq_u8(acc_u8, vcntq_u8(diff));
                i += 2;
            }
            let sum16 = vpaddlq_u8(acc_u8);
            let sum32 = vpaddlq_u16(sum16);
            acc_u32 = vaddq_u32(acc_u32, sum32);
        }
        vaddvq_u32(acc_u32)
    };

    if pairs < n {
        // SAFETY: `pairs < n <= a.len().min(b.len())`.
        let diff = unsafe { *a_ptr.add(pairs) ^ *b_ptr.add(pairs) };
        total = total.saturating_add(diff.count_ones());
    }

    total
}

/// Finds `(best_idx, min_hamming_dist)` among `num_centroids` binary centroids using
/// 4x centroid unrolling with ARM64 NEON (`vcntq_u8`).
///
/// Ties are broken deterministically in favor of the smallest centroid index.
///
/// # Safety
/// Caller must ensure the host CPU supports `neon`.
#[target_feature(enable = "neon")]
pub unsafe fn batch_nearest_binary_centroid_neon(
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

    const FLUSH_WORDS: usize = 32; // 16 NEON iterations * 8 max bits/byte = 128 <= 255
    let pairs = words_per_vec & !1;
    let v_ptr = vector_bits.as_ptr();
    let c_base_ptr = centroids_bits.as_ptr();
    let mut best_idx = 0usize;
    let mut best_dist = u32::MAX;
    let mut c = 0usize;

    while c + 4 <= valid_centroids {
        // SAFETY: `(c + 3) < valid_centroids`, so all 4 centroid slices lie inside `centroids_bits`.
        let (c0_ptr, c1_ptr, c2_ptr, c3_ptr) = unsafe {
            (
                c_base_ptr.add(c * words_per_vec),
                c_base_ptr.add((c + 1) * words_per_vec),
                c_base_ptr.add((c + 2) * words_per_vec),
                c_base_ptr.add((c + 3) * words_per_vec),
            )
        };

        // SAFETY: `neon` is enabled and `w + 2 <= pairs <= words_per_vec` on every 16-byte load.
        let (mut d0, mut d1, mut d2, mut d3) = unsafe {
            let mut acc_u32_0 = vdupq_n_u32(0);
            let mut acc_u32_1 = vdupq_n_u32(0);
            let mut acc_u32_2 = vdupq_n_u32(0);
            let mut acc_u32_3 = vdupq_n_u32(0);
            let mut w = 0usize;

            while w < pairs {
                let block_end = (w + FLUSH_WORDS).min(pairs);
                let mut acc_u8_0 = vdupq_n_u8(0);
                let mut acc_u8_1 = vdupq_n_u8(0);
                let mut acc_u8_2 = vdupq_n_u8(0);
                let mut acc_u8_3 = vdupq_n_u8(0);

                while w < block_end {
                    let vv = vld1q_u8(v_ptr.add(w).cast::<u8>());
                    let vc0 = vld1q_u8(c0_ptr.add(w).cast::<u8>());
                    let vc1 = vld1q_u8(c1_ptr.add(w).cast::<u8>());
                    let vc2 = vld1q_u8(c2_ptr.add(w).cast::<u8>());
                    let vc3 = vld1q_u8(c3_ptr.add(w).cast::<u8>());

                    acc_u8_0 = vaddq_u8(acc_u8_0, vcntq_u8(veorq_u8(vv, vc0)));
                    acc_u8_1 = vaddq_u8(acc_u8_1, vcntq_u8(veorq_u8(vv, vc1)));
                    acc_u8_2 = vaddq_u8(acc_u8_2, vcntq_u8(veorq_u8(vv, vc2)));
                    acc_u8_3 = vaddq_u8(acc_u8_3, vcntq_u8(veorq_u8(vv, vc3)));
                    w += 2;
                }

                acc_u32_0 = vaddq_u32(acc_u32_0, vpaddlq_u16(vpaddlq_u8(acc_u8_0)));
                acc_u32_1 = vaddq_u32(acc_u32_1, vpaddlq_u16(vpaddlq_u8(acc_u8_1)));
                acc_u32_2 = vaddq_u32(acc_u32_2, vpaddlq_u16(vpaddlq_u8(acc_u8_2)));
                acc_u32_3 = vaddq_u32(acc_u32_3, vpaddlq_u16(vpaddlq_u8(acc_u8_3)));
            }

            (
                vaddvq_u32(acc_u32_0),
                vaddvq_u32(acc_u32_1),
                vaddvq_u32(acc_u32_2),
                vaddvq_u32(acc_u32_3),
            )
        };

        if pairs < words_per_vec {
            // SAFETY: `pairs < words_per_vec`.
            unsafe {
                let vw = *v_ptr.add(pairs);
                d0 = d0.saturating_add((vw ^ *c0_ptr.add(pairs)).count_ones());
                d1 = d1.saturating_add((vw ^ *c1_ptr.add(pairs)).count_ones());
                d2 = d2.saturating_add((vw ^ *c2_ptr.add(pairs)).count_ones());
                d3 = d3.saturating_add((vw ^ *c3_ptr.add(pairs)).count_ones());
            }
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
        // SAFETY: `neon` is enabled.
        let dist = unsafe { hamming_distance_u64_neon(v_slice, c_slice) };
        if dist < best_dist {
            best_dist = dist;
            best_idx = c;
        }
        c += 1;
    }

    (best_idx, best_dist)
}

/// Computes the exact `int4 x 1-bit` inner product using ARM64 NEON
/// (`veorq_u8`, `vmvnq_u8`, `vandq_u8`, `vcntq_u8`) with zero heap allocations.
///
/// # Safety
/// Caller must ensure the host CPU supports `neon`.
#[target_feature(enable = "neon")]
pub unsafe fn rabitq_4bit_dot_u64_neon(
    query_planes: &[[u64; 4]],
    query_signs: &[u64],
    query_abs_sum: i32,
    db_bits: &[u64],
) -> i32 {
    let n_words = query_planes.len().min(query_signs.len()).min(db_bits.len());
    let qp_ptr = query_planes.as_ptr().cast::<u64>();
    let sign_ptr = query_signs.as_ptr();
    let db_ptr = db_bits.as_ptr();

    let mut acc0 = vdupq_n_u64(0);
    let mut acc1 = vdupq_n_u64(0);
    let mut acc2 = vdupq_n_u64(0);
    let mut acc3 = vdupq_n_u64(0);

    /// # Safety
    /// `neon` target feature must be enabled on the active CPU.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn popcnt_128_u64(v: uint8x16_t) -> uint64x2_t {
        // SAFETY: `neon` target feature is enabled by the caller.
        let bits = vcntq_u8(v);
        let p16 = vpaddlq_u8(bits);
        let p32 = vpaddlq_u16(p16);
        vpaddlq_u32(p32)
    }

    let mut w = 0usize;
    while w + 2 <= n_words {
        // SAFETY: `w + 2 <= n_words` guarantees 2 `[u64; 4]` entries (8 `u64`s)
        // and 2 `u64`s in `sign_ptr` / `db_ptr` are within bounds.
        unsafe {
            let v_db = vreinterpretq_u8_u64(vld1q_u64(db_ptr.add(w)));
            let v_sign = vreinterpretq_u8_u64(vld1q_u64(sign_ptr.add(w)));
            let v_match = vmvnq_u8(veorq_u8(v_sign, v_db));

            let base = qp_ptr.add(w * 4);
            let r0 = vld1q_u64(base);
            let r1 = vld1q_u64(base.add(2));
            let r2 = vld1q_u64(base.add(4));
            let r3 = vld1q_u64(base.add(6));

            let vp0 = vreinterpretq_u8_u64(vtrn1q_u64(r0, r2));
            let vp1 = vreinterpretq_u8_u64(vtrn2q_u64(r0, r2));
            let vp2 = vreinterpretq_u8_u64(vtrn1q_u64(r1, r3));
            let vp3 = vreinterpretq_u8_u64(vtrn2q_u64(r1, r3));

            acc0 = vaddq_u64(acc0, popcnt_128_u64(vandq_u8(vp0, v_match)));
            acc1 = vaddq_u64(acc1, popcnt_128_u64(vandq_u8(vp1, v_match)));
            acc2 = vaddq_u64(acc2, popcnt_128_u64(vandq_u8(vp2, v_match)));
            acc3 = vaddq_u64(acc3, popcnt_128_u64(vandq_u8(vp3, v_match)));
        }
        w += 2;
    }

    let mut s0 = i64::try_from(vaddvq_u64(acc0)).unwrap_or(i64::MAX);
    let mut s1 = i64::try_from(vaddvq_u64(acc1)).unwrap_or(i64::MAX);
    let mut s2 = i64::try_from(vaddvq_u64(acc2)).unwrap_or(i64::MAX);
    let mut s3 = i64::try_from(vaddvq_u64(acc3)).unwrap_or(i64::MAX);

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
/// `64 * 16`-entry `i8` query LUTs using ARM64 NEON `vqtbl1q_s8` (`vtbl`), `vaddl_s8`,
/// `vpadalq_s16`, and `vaddvq_s32`.
///
/// # Safety
/// Caller must ensure the host CPU supports `neon`.
#[target_feature(enable = "neon")]
pub unsafe fn lutrank_token_sim_i32_neon(
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

    let mask_80 = vdupq_n_u8(0x80);
    let mut acc1 = vdupq_n_s32(0);
    let mut acc2 = vdupq_n_s32(0);

    for s in 0..32 {
        // SAFETY: `s < 32` and `q_lut1_i8.len() >= 1024`, `q_lut2_i8.len() >= 1024`.
        unsafe {
            let b1 = *l1_ptr.add(s);
            let b2 = *l2_ptr.add(s);

            let idx1_e = vsetq_lane_u8::<0>(b1 & 0x0F, mask_80);
            let idx1_o = vsetq_lane_u8::<0>(b1 >> 4, mask_80);
            let idx2_e = vsetq_lane_u8::<0>(b2 & 0x0F, mask_80);
            let idx2_o = vsetq_lane_u8::<0>(b2 >> 4, mask_80);

            let lut1_e = vld1q_s8(ql1_ptr.add((2 * s) * 16));
            let lut1_o = vld1q_s8(ql1_ptr.add((2 * s + 1) * 16));
            let lut2_e = vld1q_s8(ql2_ptr.add((2 * s) * 16));
            let lut2_o = vld1q_s8(ql2_ptr.add((2 * s + 1) * 16));

            let v1e = vqtbl1q_s8(lut1_e, idx1_e);
            let v1o = vqtbl1q_s8(lut1_o, idx1_o);
            let v2e = vqtbl1q_s8(lut2_e, idx2_e);
            let v2o = vqtbl1q_s8(lut2_o, idx2_o);

            acc1 = vpadalq_s16(acc1, vaddl_s8(vget_low_s8(v1e), vget_low_s8(v1o)));
            acc2 = vpadalq_s16(acc2, vaddl_s8(vget_low_s8(v2e), vget_low_s8(v2o)));
        }
    }

    let sum1 = vaddvq_s32(acc1);
    let sum2 = vaddvq_s32(acc2);
    #[allow(clippy::cast_precision_loss)]
    {
        (sum1 as f32) * scale1 + (sum2 as f32) * scale2
    }
}

/// Computes `MaxSim(Q, D)` over a document's contiguous 64-byte-aligned `.osr` token buffer
/// using 16-token NEON tiling, `vqtbl1q_s8` (`vtbl`), and overflow-safe `i8 -> i16 -> i32`
/// widening.
///
/// # Safety
/// Caller must ensure the host CPU supports `neon`.
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn lutrank_doc_maxsim_neon(
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

    let mask_0f = vdupq_n_u8(0x0F);
    let zero_s16 = vdupq_n_s16(0);
    let vs1 = vdupq_n_f32(scale1);
    let vs2 = vdupq_n_f32(scale2);

    let mut total_maxsim = 0.0_f32;
    let mut q_base = 0usize;

    while q_base < num_query_tokens {
        let q_len = (num_query_tokens - q_base).min(64);
        let mut best_per_q = [f32::NEG_INFINITY; 64];

        let mut t_base = 0usize;
        while t_base < num_doc_tokens {
            let tile_len = (num_doc_tokens - t_base).min(16);

            let mut l1_t = [[0_u8; 16]; 32];
            let mut l2_t = [[0_u8; 16]; 32];
            for tok in 0..tile_len {
                // SAFETY: `(t_base + tok) * 64 + 64 <= doc_tokens_bytes.len()`.
                unsafe {
                    let tok_ptr = doc_ptr.add((t_base + tok) * 64);
                    for s in 0..32 {
                        l1_t[s][tok] = *tok_ptr.add(s);
                        l2_t[s][tok] = *tok_ptr.add(32 + s);
                    }
                }
            }

            for (qi, best_slot) in best_per_q[..q_len].iter_mut().enumerate() {
                // SAFETY: `q_base + qi < num_query_tokens`, `s < 32`, and `l1_t`/`l2_t` are 32x16B.
                unsafe {
                    let ql1 = ql1_base.add((q_base + qi) * 1024);
                    let ql2 = ql2_base.add((q_base + qi) * 1024);

                    let mut acc1_lo = zero_s16;
                    let mut acc1_hi = zero_s16;
                    let mut acc2_lo = zero_s16;
                    let mut acc2_hi = zero_s16;

                    for s in 0..32 {
                        let b1 = vld1q_u8(l1_t[s].as_ptr());
                        let b2 = vld1q_u8(l2_t[s].as_ptr());

                        let idx1_e = vandq_u8(b1, mask_0f);
                        let idx1_o = vshrq_n_u8::<4>(b1);
                        let idx2_e = vandq_u8(b2, mask_0f);
                        let idx2_o = vshrq_n_u8::<4>(b2);

                        let lut1_e = vld1q_s8(ql1.add((2 * s) * 16));
                        let lut1_o = vld1q_s8(ql1.add((2 * s + 1) * 16));
                        let lut2_e = vld1q_s8(ql2.add((2 * s) * 16));
                        let lut2_o = vld1q_s8(ql2.add((2 * s + 1) * 16));

                        let v1e = vqtbl1q_s8(lut1_e, idx1_e);
                        let v1o = vqtbl1q_s8(lut1_o, idx1_o);
                        let v2e = vqtbl1q_s8(lut2_e, idx2_e);
                        let v2o = vqtbl1q_s8(lut2_o, idx2_o);

                        acc1_lo = vaddq_s16(acc1_lo, vaddl_s8(vget_low_s8(v1e), vget_low_s8(v1o)));
                        acc1_hi =
                            vaddq_s16(acc1_hi, vaddl_s8(vget_high_s8(v1e), vget_high_s8(v1o)));
                        acc2_lo = vaddq_s16(acc2_lo, vaddl_s8(vget_low_s8(v2e), vget_low_s8(v2o)));
                        acc2_hi =
                            vaddq_s16(acc2_hi, vaddl_s8(vget_high_s8(v2e), vget_high_s8(v2o)));
                    }

                    let f1_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(acc1_lo)));
                    let f1_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(acc1_lo)));
                    let f1_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(acc1_hi)));
                    let f1_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(acc1_hi)));

                    let f2_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(acc2_lo)));
                    let f2_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(acc2_lo)));
                    let f2_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(acc2_hi)));
                    let f2_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(acc2_hi)));

                    let s_0 = vaddq_f32(vmulq_f32(f1_0, vs1), vmulq_f32(f2_0, vs2));
                    let s_1 = vaddq_f32(vmulq_f32(f1_1, vs1), vmulq_f32(f2_1, vs2));
                    let s_2 = vaddq_f32(vmulq_f32(f1_2, vs1), vmulq_f32(f2_2, vs2));
                    let s_3 = vaddq_f32(vmulq_f32(f1_3, vs1), vmulq_f32(f2_3, vs2));

                    if tile_len == 16 {
                        let m01 = vmaxq_f32(s_0, s_1);
                        let m23 = vmaxq_f32(s_2, s_3);
                        let tile_best = vmaxvq_f32(vmaxq_f32(m01, m23));
                        if tile_best > *best_slot {
                            *best_slot = tile_best;
                        }
                    } else {
                        let mut scores = [0.0_f32; 16];
                        vst1q_f32(scores.as_mut_ptr(), s_0);
                        vst1q_f32(scores.as_mut_ptr().add(4), s_1);
                        vst1q_f32(scores.as_mut_ptr().add(8), s_2);
                        vst1q_f32(scores.as_mut_ptr().add(12), s_3);
                        for &sc in &scores[..tile_len] {
                            if sc > *best_slot {
                                *best_slot = sc;
                            }
                        }
                    }
                }
            }

            t_base += 16;
        }

        for &b in &best_per_q[..q_len] {
            total_maxsim += b;
        }
        q_base += 64;
    }

    total_maxsim
}
