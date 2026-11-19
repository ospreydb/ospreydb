//! AVX2 + FMA vectorized FWHT kernel with L1 cache loop tiling (`L1_TILE_FLOATS = 512`).

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use osprey_core::{OspreyError, OspreyResult};

/// L1 data cache tile size in `f32` elements (`512 * 4 = 2048` bytes = 2 KiB).
pub const L1_TILE_FLOATS: usize = 512;

/// Applies Rademacher XOR sign masks and orthogonal FWHT in-place using AVX2.
///
/// # Safety
/// Caller must ensure the host CPU supports `avx2` and `fma` target features.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn fwht_rotate_in_place_avx2(
    data: &mut [f32],
    signs_xor_mask: &[u32],
) -> OspreyResult<()> {
    let d = data.len();
    if d < 8 || !d.is_power_of_two() || d != signs_xor_mask.len() {
        return Err(OspreyError::InvalidDimension { dim: d });
    }

    let d_u32 = u32::try_from(d).map_err(|_| OspreyError::InvalidDimension { dim: d })?;
    #[allow(clippy::cast_possible_truncation)]
    let inv_sqrt_d = (1.0_f64 / f64::from(d_u32).sqrt()) as f32;

    if d <= L1_TILE_FLOATS {
        // SAFETY: `d >= 8`, `d == signs_xor_mask.len()`, and `d` is a power of 2.
        unsafe {
            fwht_tile_prefix_avx2(data, signs_xor_mask);
        }
        let mut h = 8usize;
        while h < d {
            let is_final = h * 2 == d;
            // SAFETY: `h >= 8` and `d` is a power of 2 divisible by `2 * h`.
            unsafe {
                fwht_stage_avx2(data, h, is_final.then_some(inv_sqrt_d));
            }
            h *= 2;
        }
        if d == 8 {
            // Stages 1..4 completed inside `fwht_tile_prefix_avx2`; apply scaling here.
            // SAFETY: `data.len() == 8`.
            unsafe {
                scale_avx2(data, inv_sqrt_d);
            }
        }
    } else {
        // Phase 1 (Local L1-tiled): keep each 512-float (2 KiB) slice hot in L1d.
        for (tile, mask_tile) in data
            .as_chunks_mut::<L1_TILE_FLOATS>()
            .0
            .iter_mut()
            .zip(signs_xor_mask.as_chunks::<L1_TILE_FLOATS>().0.iter())
        {
            // SAFETY: each tile has exactly `L1_TILE_FLOATS == 512` elements.
            unsafe {
                fwht_tile_prefix_avx2(tile, mask_tile);
            }
            let mut h = 8usize;
            while h < L1_TILE_FLOATS {
                // SAFETY: `h >= 8` divides `L1_TILE_FLOATS == 512`.
                unsafe {
                    fwht_stage_avx2(tile, h, None);
                }
                h *= 2;
            }
        }

        // Phase 2 (Global cross-tile): fuse `1/sqrt(d)` into the final butterfly pass.
        let mut h = L1_TILE_FLOATS;
        while h < d {
            let is_final = h * 2 == d;
            // SAFETY: `h >= 512` divides `d`.
            unsafe {
                fwht_stage_avx2(data, h, is_final.then_some(inv_sqrt_d));
            }
            h *= 2;
        }
    }

    Ok(())
}

/// Applies XOR sign masks and intra-register stages `h = 1, 2, 4` over 8-float YMM blocks.
///
/// # Safety
/// `data.len() == signs_xor_mask.len()` must be a multiple of 8, and `avx2` must be enabled.
#[target_feature(enable = "avx2")]
unsafe fn fwht_tile_prefix_avx2(data: &mut [f32], signs_xor_mask: &[u32]) {
    let n = data.len();
    let d_ptr = data.as_mut_ptr();
    let m_ptr = signs_xor_mask.as_ptr();
    let mut i = 0usize;

    while i < n {
        // SAFETY: `i + 8 <= n` because `n` is a multiple of 8.
        unsafe {
            let v = _mm256_loadu_ps(d_ptr.add(i));
            let mask = _mm256_loadu_si256(m_ptr.add(i).cast::<__m256i>());
            let mut x = _mm256_xor_ps(v, _mm256_castsi256_ps(mask));

            // Stage h = 1: adjacent pairs within 128-bit lanes
            let sw1 = _mm256_permute_ps::<0b10_11_00_01>(x);
            let s1 = _mm256_add_ps(x, sw1);
            let d1 = _mm256_sub_ps(sw1, x);
            x = _mm256_blend_ps::<0b1010_1010>(s1, d1);

            // Stage h = 2: 64-bit halves within 128-bit lanes
            let sw2 = _mm256_permute_ps::<0b01_00_11_10>(x);
            let s2 = _mm256_add_ps(x, sw2);
            let d2 = _mm256_sub_ps(sw2, x);
            x = _mm256_blend_ps::<0b1100_1100>(s2, d2);

            // Stage h = 4: swap 128-bit lanes
            let sw4 = _mm256_permute2f128_ps::<0x01>(x, x);
            let s4 = _mm256_add_ps(x, sw4);
            let d4 = _mm256_sub_ps(sw4, x);
            x = _mm256_blend_ps::<0b1111_0000>(s4, d4);

            _mm256_storeu_ps(d_ptr.add(i), x);
        }
        i += 8;
    }
}

/// Executes a single wide butterfly stage (`h >= 8`) with optional fused final scaling.
///
/// # Safety
/// `data.len()` must be a multiple of `2 * h`, with `h >= 8` and `h % 8 == 0`.
#[target_feature(enable = "avx2")]
unsafe fn fwht_stage_avx2(data: &mut [f32], h: usize, scale: Option<f32>) {
    let d = data.len();
    let ptr = data.as_mut_ptr();
    let step = h * 2;

    if let Some(inv_sqrt_d) = scale {
        let vs = _mm256_set1_ps(inv_sqrt_d);
        let mut base = 0usize;
        while base < d {
            let mut j = base;
            let limit = base + h;
            while j < limit {
                // SAFETY: `j + h + 8 <= base + step <= d`.
                unsafe {
                    let va = _mm256_loadu_ps(ptr.add(j));
                    let vb = _mm256_loadu_ps(ptr.add(j + h));
                    _mm256_storeu_ps(ptr.add(j), _mm256_mul_ps(_mm256_add_ps(va, vb), vs));
                    _mm256_storeu_ps(ptr.add(j + h), _mm256_mul_ps(_mm256_sub_ps(va, vb), vs));
                }
                j += 8;
            }
            base += step;
        }
    } else {
        let mut base = 0usize;
        while base < d {
            let mut j = base;
            let limit = base + h;
            while j < limit {
                // SAFETY: `j + h + 8 <= base + step <= d`.
                unsafe {
                    let va = _mm256_loadu_ps(ptr.add(j));
                    let vb = _mm256_loadu_ps(ptr.add(j + h));
                    _mm256_storeu_ps(ptr.add(j), _mm256_add_ps(va, vb));
                    _mm256_storeu_ps(ptr.add(j + h), _mm256_sub_ps(va, vb));
                }
                j += 8;
            }
            base += step;
        }
    }
}

/// # Safety
/// `data.len()` must be a multiple of 8 and `avx2` must be enabled.
#[target_feature(enable = "avx2")]
unsafe fn scale_avx2(data: &mut [f32], inv_sqrt_d: f32) {
    let n = data.len();
    let ptr = data.as_mut_ptr();
    let vs = _mm256_set1_ps(inv_sqrt_d);
    let mut i = 0usize;
    while i < n {
        // SAFETY: `i + 8 <= n`.
        unsafe {
            let v = _mm256_loadu_ps(ptr.add(i));
            _mm256_storeu_ps(ptr.add(i), _mm256_mul_ps(v, vs));
        }
        i += 8;
    }
}

/// 16-entry nibble popcount lookup table for `_mm256_shuffle_epi8`.
const LOW_NIBBLE_POPCNT: [u8; 16] = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];

#[target_feature(enable = "avx2")]
#[inline]
fn nibble_lut_avx2() -> __m256i {
    let t = LOW_NIBBLE_POPCNT;
    _mm256_setr_epi8(
        t[0].cast_signed(),
        t[1].cast_signed(),
        t[2].cast_signed(),
        t[3].cast_signed(),
        t[4].cast_signed(),
        t[5].cast_signed(),
        t[6].cast_signed(),
        t[7].cast_signed(),
        t[8].cast_signed(),
        t[9].cast_signed(),
        t[10].cast_signed(),
        t[11].cast_signed(),
        t[12].cast_signed(),
        t[13].cast_signed(),
        t[14].cast_signed(),
        t[15].cast_signed(),
        t[0].cast_signed(),
        t[1].cast_signed(),
        t[2].cast_signed(),
        t[3].cast_signed(),
        t[4].cast_signed(),
        t[5].cast_signed(),
        t[6].cast_signed(),
        t[7].cast_signed(),
        t[8].cast_signed(),
        t[9].cast_signed(),
        t[10].cast_signed(),
        t[11].cast_signed(),
        t[12].cast_signed(),
        t[13].cast_signed(),
        t[14].cast_signed(),
        t[15].cast_signed(),
    )
}

#[target_feature(enable = "avx2")]
#[inline]
fn popcnt_bytes_avx2(x: __m256i, lut: __m256i, low_mask: __m256i) -> __m256i {
    let lo = _mm256_and_si256(x, low_mask);
    let hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), low_mask);
    let cnt_lo = _mm256_shuffle_epi8(lut, lo);
    let cnt_hi = _mm256_shuffle_epi8(lut, hi);
    _mm256_add_epi8(cnt_lo, cnt_hi)
}

#[target_feature(enable = "avx2")]
#[inline]
fn hsum_epi64_avx2(v: __m256i) -> u32 {
    let hi128 = _mm256_extracti128_si256::<1>(v);
    let lo128 = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi64(lo128, hi128);
    let hi64 = _mm_unpackhi_epi64(sum128, sum128);
    let total64 = _mm_add_epi64(sum128, hi64);
    #[cfg(target_arch = "x86_64")]
    {
        u32::try_from(_mm_cvtsi128_si64(total64)).unwrap_or(u32::MAX)
    }
    #[cfg(target_arch = "x86")]
    {
        _mm_cvtsi128_si32(total64).cast_unsigned()
    }
}

#[target_feature(enable = "popcnt")]
#[inline]
fn popcnt_u64_native(val: u64) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        _popcnt64(val.cast_signed()).cast_unsigned()
    }
    #[cfg(target_arch = "x86")]
    {
        val.count_ones()
    }
}

/// Computes Hamming distance `sum(popcnt(a[w] ^ b[w]))` using AVX2 + `POPCNT`.
///
/// For long vectors (`n >= 8` words, i.e. `d >= 512`), uses YMM `vpshufb` nibble lookup
/// summed via `_mm256_sad_epu8` with periodic flushing to prevent `u8` lane overflow.
/// For short vectors (`n < 8`, i.e. `d in {64, 128, 256}`), uses unrolled scalar `_popcnt64`
/// to avoid horizontal setup overhead.
///
/// # Safety
/// Caller must ensure the host CPU supports `avx2` and `popcnt`.
#[target_feature(enable = "avx2,popcnt")]
pub unsafe fn hamming_distance_u64_avx2(a: &[u64], b: &[u64]) -> u32 {
    let n = a.len().min(b.len());
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let mut i = 0usize;

    let mut total = if n >= 8 {
        // Flush every 16 YMM steps (64 u64 words): max per-byte sum = 16 * 8 = 128 <= 255.
        const FLUSH_WORDS: usize = 64;
        // SAFETY: `avx2` is enabled; `y_end <= n` guarantees all 32-byte loads stay in-bounds.
        unsafe {
            let lut = nibble_lut_avx2();
            let low_mask = _mm256_set1_epi8(0x0F);
            let zero = _mm256_setzero_si256();
            let mut tot0 = zero;
            let mut tot1 = zero;
            let y_end = n & !3;

            while i < y_end {
                let block_end = (i + FLUSH_WORDS).min(y_end);
                let mut acc0 = zero;
                let mut acc1 = zero;

                while i + 8 <= block_end {
                    let va0 = _mm256_loadu_si256(a_ptr.add(i).cast::<__m256i>());
                    let vb0 = _mm256_loadu_si256(b_ptr.add(i).cast::<__m256i>());
                    acc0 = _mm256_add_epi8(
                        acc0,
                        popcnt_bytes_avx2(_mm256_xor_si256(va0, vb0), lut, low_mask),
                    );

                    let va1 = _mm256_loadu_si256(a_ptr.add(i + 4).cast::<__m256i>());
                    let vb1 = _mm256_loadu_si256(b_ptr.add(i + 4).cast::<__m256i>());
                    acc1 = _mm256_add_epi8(
                        acc1,
                        popcnt_bytes_avx2(_mm256_xor_si256(va1, vb1), lut, low_mask),
                    );

                    i += 8;
                }

                if i < block_end {
                    let va = _mm256_loadu_si256(a_ptr.add(i).cast::<__m256i>());
                    let vb = _mm256_loadu_si256(b_ptr.add(i).cast::<__m256i>());
                    acc0 = _mm256_add_epi8(
                        acc0,
                        popcnt_bytes_avx2(_mm256_xor_si256(va, vb), lut, low_mask),
                    );
                    i += 4;
                }

                tot0 = _mm256_add_epi64(tot0, _mm256_sad_epu8(acc0, zero));
                tot1 = _mm256_add_epi64(tot1, _mm256_sad_epu8(acc1, zero));
            }

            hsum_epi64_avx2(_mm256_add_epi64(tot0, tot1))
        }
    } else {
        // Short vectors (d in {64, 128, 256}): 4x unrolled native `_popcnt64`.
        let mut acc0 = 0_u32;
        let mut acc1 = 0_u32;
        let mut acc2 = 0_u32;
        let mut acc3 = 0_u32;
        while i + 4 <= n {
            // SAFETY: `i + 4 <= n <= a.len().min(b.len())` and `popcnt` is enabled.
            unsafe {
                acc0 += popcnt_u64_native(*a_ptr.add(i) ^ *b_ptr.add(i));
                acc1 += popcnt_u64_native(*a_ptr.add(i + 1) ^ *b_ptr.add(i + 1));
                acc2 += popcnt_u64_native(*a_ptr.add(i + 2) ^ *b_ptr.add(i + 2));
                acc3 += popcnt_u64_native(*a_ptr.add(i + 3) ^ *b_ptr.add(i + 3));
            }
            i += 4;
        }
        acc0 + acc1 + acc2 + acc3
    };

    while i < n {
        // SAFETY: `i < n <= a.len().min(b.len())` and `popcnt` is enabled.
        let cnt = unsafe { popcnt_u64_native(*a_ptr.add(i) ^ *b_ptr.add(i)) };
        total = total.saturating_add(cnt);
        i += 1;
    }

    total
}

/// Finds `(best_idx, min_hamming_dist)` among `num_centroids` binary centroids using
/// 4x centroid unrolling with AVX2 `vpshufb` (for `d >= 512`) and `_popcnt64` (for `d < 512`).
///
/// Ties are broken deterministically in favor of the smallest centroid index.
///
/// # Safety
/// Caller must ensure the host CPU supports `avx2` and `popcnt`.
#[target_feature(enable = "avx2,popcnt")]
pub unsafe fn batch_nearest_binary_centroid_avx2(
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

    if words_per_vec >= 8 {
        const FLUSH_WORDS: usize = 64; // 16 YMM iterations * 8 max bits/byte = 128 <= 255
        let (lut, low_mask, zero) = (
            nibble_lut_avx2(),
            _mm256_set1_epi8(0x0F),
            _mm256_setzero_si256(),
        );
        let y_end = words_per_vec & !3;

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

            let mut w = 0usize;
            // SAFETY: `avx2` is enabled and `w + 4 <= y_end <= words_per_vec` on every load.
            let (mut d0, mut d1, mut d2, mut d3) = unsafe {
                let mut tot0 = zero;
                let mut tot1 = zero;
                let mut tot2 = zero;
                let mut tot3 = zero;

                while w < y_end {
                    let block_end = (w + FLUSH_WORDS).min(y_end);
                    let mut acc0 = zero;
                    let mut acc1 = zero;
                    let mut acc2 = zero;
                    let mut acc3 = zero;

                    while w < block_end {
                        let vv = _mm256_loadu_si256(v_ptr.add(w).cast::<__m256i>());
                        let vc0 = _mm256_loadu_si256(c0_ptr.add(w).cast::<__m256i>());
                        let vc1 = _mm256_loadu_si256(c1_ptr.add(w).cast::<__m256i>());
                        let vc2 = _mm256_loadu_si256(c2_ptr.add(w).cast::<__m256i>());
                        let vc3 = _mm256_loadu_si256(c3_ptr.add(w).cast::<__m256i>());

                        acc0 = _mm256_add_epi8(
                            acc0,
                            popcnt_bytes_avx2(_mm256_xor_si256(vv, vc0), lut, low_mask),
                        );
                        acc1 = _mm256_add_epi8(
                            acc1,
                            popcnt_bytes_avx2(_mm256_xor_si256(vv, vc1), lut, low_mask),
                        );
                        acc2 = _mm256_add_epi8(
                            acc2,
                            popcnt_bytes_avx2(_mm256_xor_si256(vv, vc2), lut, low_mask),
                        );
                        acc3 = _mm256_add_epi8(
                            acc3,
                            popcnt_bytes_avx2(_mm256_xor_si256(vv, vc3), lut, low_mask),
                        );
                        w += 4;
                    }

                    tot0 = _mm256_add_epi64(tot0, _mm256_sad_epu8(acc0, zero));
                    tot1 = _mm256_add_epi64(tot1, _mm256_sad_epu8(acc1, zero));
                    tot2 = _mm256_add_epi64(tot2, _mm256_sad_epu8(acc2, zero));
                    tot3 = _mm256_add_epi64(tot3, _mm256_sad_epu8(acc3, zero));
                }

                (
                    hsum_epi64_avx2(tot0),
                    hsum_epi64_avx2(tot1),
                    hsum_epi64_avx2(tot2),
                    hsum_epi64_avx2(tot3),
                )
            };

            while w < words_per_vec {
                // SAFETY: `w < words_per_vec` and `popcnt` is enabled.
                unsafe {
                    let vw = *v_ptr.add(w);
                    d0 = d0.saturating_add(popcnt_u64_native(vw ^ *c0_ptr.add(w)));
                    d1 = d1.saturating_add(popcnt_u64_native(vw ^ *c1_ptr.add(w)));
                    d2 = d2.saturating_add(popcnt_u64_native(vw ^ *c2_ptr.add(w)));
                    d3 = d3.saturating_add(popcnt_u64_native(vw ^ *c3_ptr.add(w)));
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
    } else {
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

            let mut d0 = 0_u32;
            let mut d1 = 0_u32;
            let mut d2 = 0_u32;
            let mut d3 = 0_u32;
            for w in 0..words_per_vec {
                // SAFETY: `w < words_per_vec` and `popcnt` is enabled.
                unsafe {
                    let vw = *v_ptr.add(w);
                    d0 += popcnt_u64_native(vw ^ *c0_ptr.add(w));
                    d1 += popcnt_u64_native(vw ^ *c1_ptr.add(w));
                    d2 += popcnt_u64_native(vw ^ *c2_ptr.add(w));
                    d3 += popcnt_u64_native(vw ^ *c3_ptr.add(w));
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
    }

    let v_slice = &vector_bits[..words_per_vec];
    while c < valid_centroids {
        let start = c * words_per_vec;
        let c_slice = &centroids_bits[start..start + words_per_vec];
        // SAFETY: `avx2,popcnt` are enabled.
        let dist = unsafe { hamming_distance_u64_avx2(v_slice, c_slice) };
        if dist < best_dist {
            best_dist = dist;
            best_idx = c;
        }
        c += 1;
    }

    (best_idx, best_dist)
}

/// Computes the exact `int4 x 1-bit` inner product using AVX2 `vpshufb` nibble-popcount
/// and `_mm256_sad_epu8` with zero heap allocations in the hot loop.
///
/// # Safety
/// Caller must ensure the host CPU supports `avx2`.
#[target_feature(enable = "avx2")]
pub unsafe fn rabitq_4bit_dot_u64_avx2(
    query_planes: &[[u64; 4]],
    query_signs: &[u64],
    query_abs_sum: i32,
    db_bits: &[u64],
) -> i32 {
    let n_words = query_planes.len().min(query_signs.len()).min(db_bits.len());
    let qp_ptr = query_planes.as_ptr().cast::<u64>();
    let sign_ptr = query_signs.as_ptr();
    let db_ptr = db_bits.as_ptr();

    let popcnt_lut = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3,
        3, 4,
    );
    let low_mask = _mm256_set1_epi8(0x0F);
    let zero = _mm256_setzero_si256();

    let mut acc0 = zero;
    let mut acc1 = zero;
    let mut acc2 = zero;
    let mut acc3 = zero;

    #[target_feature(enable = "avx2")]
    #[inline]
    fn popcnt_256(v: __m256i, lut: __m256i, low_mask: __m256i, zero: __m256i) -> __m256i {
        let lo = _mm256_and_si256(v, low_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
        let c_lo = _mm256_shuffle_epi8(lut, lo);
        let c_hi = _mm256_shuffle_epi8(lut, hi);
        _mm256_sad_epu8(_mm256_add_epi8(c_lo, c_hi), zero)
    }

    let mut w = 0usize;
    while w + 4 <= n_words {
        // SAFETY: `w + 4 <= n_words` guarantees all four `[u64; 4]` entries (16 `u64`s)
        // and the 4-word `sign_ptr` / `db_ptr` slices are within bounds.
        unsafe {
            if w + 12 <= n_words && (w & 7) == 0 {
                _mm_prefetch::<_MM_HINT_T0>(db_ptr.add(w + 8).cast::<i8>());
                _mm_prefetch::<_MM_HINT_T0>(qp_ptr.add((w + 8) * 4).cast::<i8>());
            }
            let v_db = _mm256_loadu_si256(db_ptr.add(w).cast::<__m256i>());
            let v_sign = _mm256_loadu_si256(sign_ptr.add(w).cast::<__m256i>());
            let v_xor = _mm256_xor_si256(v_sign, v_db);

            let base = qp_ptr.add(w * 4);
            let r0 = _mm256_loadu_si256(base.cast::<__m256i>());
            let r1 = _mm256_loadu_si256(base.add(4).cast::<__m256i>());
            let r2 = _mm256_loadu_si256(base.add(8).cast::<__m256i>());
            let r3 = _mm256_loadu_si256(base.add(12).cast::<__m256i>());

            let t0 = _mm256_unpacklo_epi64(r0, r1);
            let t1 = _mm256_unpackhi_epi64(r0, r1);
            let t2 = _mm256_unpacklo_epi64(r2, r3);
            let t3 = _mm256_unpackhi_epi64(r2, r3);

            let vp0 = _mm256_permute2x128_si256::<0x20>(t0, t2);
            let vp1 = _mm256_permute2x128_si256::<0x20>(t1, t3);
            let vp2 = _mm256_permute2x128_si256::<0x31>(t0, t2);
            let vp3 = _mm256_permute2x128_si256::<0x31>(t1, t3);

            // `andnot(v_xor, vp)` == `!(q_sign ^ db_bits) & plane`.
            let m0 = _mm256_andnot_si256(v_xor, vp0);
            let m1 = _mm256_andnot_si256(v_xor, vp1);
            let m2 = _mm256_andnot_si256(v_xor, vp2);
            let m3 = _mm256_andnot_si256(v_xor, vp3);

            acc0 = _mm256_add_epi64(acc0, popcnt_256(m0, popcnt_lut, low_mask, zero));
            acc1 = _mm256_add_epi64(acc1, popcnt_256(m1, popcnt_lut, low_mask, zero));
            acc2 = _mm256_add_epi64(acc2, popcnt_256(m2, popcnt_lut, low_mask, zero));
            acc3 = _mm256_add_epi64(acc3, popcnt_256(m3, popcnt_lut, low_mask, zero));
        }
        w += 4;
    }

    /// # Safety
    /// `avx2` target feature must be enabled on the active CPU.
    #[inline(always)]
    unsafe fn hsum_epi64(v: __m256i) -> i64 {
        let mut lanes = [0_i64; 4];
        // SAFETY: `lanes` is a 32-byte stack buffer.
        unsafe {
            _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), v);
        }
        lanes[0] + lanes[1] + lanes[2] + lanes[3]
    }

    // SAFETY: AVX2 is active on this function.
    let (mut s0, mut s1, mut s2, mut s3) = unsafe {
        (
            hsum_epi64(acc0),
            hsum_epi64(acc1),
            hsum_epi64(acc2),
            hsum_epi64(acc3),
        )
    };

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
/// `64 * 16`-entry `i8` query LUTs using AVX2 `vpshufb` (`_mm256_shuffle_epi8`),
/// `_mm256_cvtepi8_epi16`, and `_mm256_madd_epi16`.
///
/// # Safety
/// Caller must ensure the host CPU supports `avx2`.
#[target_feature(enable = "avx2")]
pub unsafe fn lutrank_token_sim_i32_avx2(
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

    let ones_i16 = _mm256_set1_epi16(1);
    let mask_dword_hi = 0x8080_8000_u32.cast_signed();
    let mask_dword_all = 0x8080_8080_u32.cast_signed();

    let mut acc1 = _mm256_setzero_si256();
    let mut acc2 = _mm256_setzero_si256();

    for s in 0..32 {
        // SAFETY: `s < 32`, `doc_l1_packed.len() == doc_l2_packed.len() == 32`, and
        // `q_lut1_i8.len() >= 1024`, `q_lut2_i8.len() >= 1024` (`s * 32 + 32 <= 1024`).
        unsafe {
            if s + 4 < 32 && (s & 1) == 0 {
                _mm_prefetch::<_MM_HINT_T0>(ql1_ptr.add((s + 4) * 32).cast::<i8>());
                _mm_prefetch::<_MM_HINT_T0>(ql2_ptr.add((s + 4) * 32).cast::<i8>());
            }
            let b1 = *l1_ptr.add(s);
            let b2 = *l2_ptr.add(s);
            let c1_e = i32::from(b1 & 0x0F);
            let c1_o = i32::from(b1 >> 4);
            let c2_e = i32::from(b2 & 0x0F);
            let c2_o = i32::from(b2 >> 4);

            // Low 128-bit lane holds subspace `2*s`, high 128-bit lane holds subspace `2*s + 1`.
            let lut1 = _mm256_loadu_si256(ql1_ptr.add(s * 32).cast::<__m256i>());
            let lut2 = _mm256_loadu_si256(ql2_ptr.add(s * 32).cast::<__m256i>());

            let idx1 = _mm256_set_epi32(
                mask_dword_all,
                mask_dword_all,
                mask_dword_all,
                c1_o | mask_dword_hi,
                mask_dword_all,
                mask_dword_all,
                mask_dword_all,
                c1_e | mask_dword_hi,
            );
            let idx2 = _mm256_set_epi32(
                mask_dword_all,
                mask_dword_all,
                mask_dword_all,
                c2_o | mask_dword_hi,
                mask_dword_all,
                mask_dword_all,
                mask_dword_all,
                c2_e | mask_dword_hi,
            );

            let res1 = _mm256_shuffle_epi8(lut1, idx1);
            let res2 = _mm256_shuffle_epi8(lut2, idx2);

            let w1_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(res1));
            let w1_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(res1));
            let w2_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(res2));
            let w2_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(res2));

            acc1 = _mm256_add_epi32(
                acc1,
                _mm256_madd_epi16(_mm256_add_epi16(w1_lo, w1_hi), ones_i16),
            );
            acc2 = _mm256_add_epi32(
                acc2,
                _mm256_madd_epi16(_mm256_add_epi16(w2_lo, w2_hi), ones_i16),
            );
        }
    }

    /// # Safety
    /// `avx2` target feature must be enabled on the active CPU.
    #[inline(always)]
    unsafe fn hsum_epi32(v: __m256i) -> i32 {
        let mut lanes = [0_i32; 8];
        // SAFETY: `lanes` is a 32-byte stack buffer.
        unsafe {
            _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), v);
        }
        lanes.iter().sum()
    }

    // SAFETY: AVX2 is active on this function.
    let (sum1, sum2) = unsafe { (hsum_epi32(acc1), hsum_epi32(acc2)) };
    #[allow(clippy::cast_precision_loss)]
    {
        (sum1 as f32) * scale1 + (sum2 as f32) * scale2
    }
}

/// Computes `MaxSim(Q, D)` over a document's contiguous 64-byte-aligned `.osr` token buffer
/// using 32-token YMM tiling, software prefetching, `_mm256_shuffle_epi8` (`vpshufb`), and
/// overflow-safe `i16 -> i32` widening.
///
/// # Safety
/// Caller must ensure the host CPU supports `avx2`.
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn lutrank_doc_maxsim_avx2(
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

    let mask_0f = _mm256_set1_epi8(0x0F);
    let zero = _mm256_setzero_si256();
    let vs1 = _mm256_set1_ps(scale1);
    let vs2 = _mm256_set1_ps(scale2);

    let mut total_maxsim = 0.0_f32;
    let mut q_base = 0usize;

    while q_base < num_query_tokens {
        let q_len = (num_query_tokens - q_base).min(64);
        let mut best_per_q = [f32::NEG_INFINITY; 64];

        let mut t_base = 0usize;
        while t_base < num_doc_tokens {
            let tile_len = (num_doc_tokens - t_base).min(32);

            if t_base + 32 < num_doc_tokens {
                // SAFETY: `t_base + 32 < num_doc_tokens`, so `(t_base + 32) * 64` is in-bounds.
                unsafe {
                    let next_tile_tokens = (num_doc_tokens - (t_base + 32)).min(8);
                    let pf = doc_ptr.add((t_base + 32) * 64).cast::<i8>();
                    for k in 0..next_tile_tokens {
                        _mm_prefetch::<_MM_HINT_T0>(pf.add(k * 64));
                    }
                }
            }

            let mut l1_t = [[0_u8; 32]; 32];
            let mut l2_t = [[0_u8; 32]; 32];
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
                // SAFETY: `q_base + qi < num_query_tokens`, `s < 32`, and `l1_t`/`l2_t` are 32x32B.
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
                        let b1 = _mm256_loadu_si256(l1_t[s].as_ptr().cast::<__m256i>());
                        let b2 = _mm256_loadu_si256(l2_t[s].as_ptr().cast::<__m256i>());

                        let idx1_e = _mm256_and_si256(b1, mask_0f);
                        let idx1_o = _mm256_and_si256(_mm256_srli_epi16::<4>(b1), mask_0f);
                        let idx2_e = _mm256_and_si256(b2, mask_0f);
                        let idx2_o = _mm256_and_si256(_mm256_srli_epi16::<4>(b2), mask_0f);

                        let lut1_e = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                            ql1.add((2 * s) * 16).cast::<__m128i>(),
                        ));
                        let lut1_o = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                            ql1.add((2 * s + 1) * 16).cast::<__m128i>(),
                        ));
                        let lut2_e = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                            ql2.add((2 * s) * 16).cast::<__m128i>(),
                        ));
                        let lut2_o = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                            ql2.add((2 * s + 1) * 16).cast::<__m128i>(),
                        ));

                        let v1e = _mm256_shuffle_epi8(lut1_e, idx1_e);
                        let v1o = _mm256_shuffle_epi8(lut1_o, idx1_o);
                        let v2e = _mm256_shuffle_epi8(lut2_e, idx2_e);
                        let v2o = _mm256_shuffle_epi8(lut2_o, idx2_o);

                        // Safe i8->i16 accumulation: max |sum| = 64 * 127 = 8128 < 32767.
                        acc1_lo = _mm256_add_epi16(
                            acc1_lo,
                            _mm256_add_epi16(
                                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v1e)),
                                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v1o)),
                            ),
                        );
                        acc1_hi = _mm256_add_epi16(
                            acc1_hi,
                            _mm256_add_epi16(
                                _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(v1e)),
                                _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(v1o)),
                            ),
                        );

                        acc2_lo = _mm256_add_epi16(
                            acc2_lo,
                            _mm256_add_epi16(
                                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v2e)),
                                _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v2o)),
                            ),
                        );
                        acc2_hi = _mm256_add_epi16(
                            acc2_hi,
                            _mm256_add_epi16(
                                _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(v2e)),
                                _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(v2o)),
                            ),
                        );
                    }

                    let f1_0 =
                        _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc1_lo)));
                    let f1_1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
                        _mm256_extracti128_si256::<1>(acc1_lo),
                    ));
                    let f1_2 =
                        _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc1_hi)));
                    let f1_3 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
                        _mm256_extracti128_si256::<1>(acc1_hi),
                    ));

                    let f2_0 =
                        _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc2_lo)));
                    let f2_1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
                        _mm256_extracti128_si256::<1>(acc2_lo),
                    ));
                    let f2_2 =
                        _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc2_hi)));
                    let f2_3 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
                        _mm256_extracti128_si256::<1>(acc2_hi),
                    ));

                    let s_0 = _mm256_add_ps(_mm256_mul_ps(f1_0, vs1), _mm256_mul_ps(f2_0, vs2));
                    let s_1 = _mm256_add_ps(_mm256_mul_ps(f1_1, vs1), _mm256_mul_ps(f2_1, vs2));
                    let s_2 = _mm256_add_ps(_mm256_mul_ps(f1_2, vs1), _mm256_mul_ps(f2_2, vs2));
                    let s_3 = _mm256_add_ps(_mm256_mul_ps(f1_3, vs1), _mm256_mul_ps(f2_3, vs2));

                    if tile_len == 32 {
                        let m01 = _mm256_max_ps(s_0, s_1);
                        let m23 = _mm256_max_ps(s_2, s_3);
                        let m = _mm256_max_ps(m01, m23);
                        let hi = _mm256_extractf128_ps::<1>(m);
                        let lo = _mm256_castps256_ps128(m);
                        let m128 = _mm_max_ps(lo, hi);
                        let shuf = _mm_movehl_ps(m128, m128);
                        let m64 = _mm_max_ps(m128, shuf);
                        let m32 = _mm_max_ss(m64, _mm_shuffle_ps::<0x55>(m64, m64));
                        let tile_best = _mm_cvtss_f32(m32);
                        if tile_best > *best_slot {
                            *best_slot = tile_best;
                        }
                    } else {
                        let mut scores = [0.0_f32; 32];
                        _mm256_storeu_ps(scores.as_mut_ptr(), s_0);
                        _mm256_storeu_ps(scores.as_mut_ptr().add(8), s_1);
                        _mm256_storeu_ps(scores.as_mut_ptr().add(16), s_2);
                        _mm256_storeu_ps(scores.as_mut_ptr().add(24), s_3);
                        for &sc in &scores[..tile_len] {
                            if sc > *best_slot {
                                *best_slot = sc;
                            }
                        }
                    }
                }
            }

            t_base += 32;
        }

        for &b in &best_per_q[..q_len] {
            total_maxsim += b;
        }
        q_base += 64;
    }

    total_maxsim
}

/// Computes the exact `f32 x f16` inner product accumulated in `f32` using AVX2 + F16C + FMA.
///
/// # Safety
/// Caller must ensure the host CPU supports `avx2`, `f16c`, and `fma`.
#[target_feature(enable = "avx2,f16c,fma")]
pub unsafe fn dot_f32_f16_avx2(a_f32: &[f32], b_f16: &[half::f16]) -> f32 {
    let n = a_f32.len().min(b_f16.len());
    let a_ptr = a_f32.as_ptr();
    let b_ptr = b_f16.as_ptr().cast::<u16>();

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let mut i = 0usize;
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n <= a_f32.len().min(b_f16.len())`.
        unsafe {
            let va0 = _mm256_loadu_ps(a_ptr.add(i));
            let va1 = _mm256_loadu_ps(a_ptr.add(i + 8));
            let va2 = _mm256_loadu_ps(a_ptr.add(i + 16));
            let va3 = _mm256_loadu_ps(a_ptr.add(i + 24));

            let vb0 = _mm256_cvtph_ps(_mm_loadu_si128(b_ptr.add(i).cast::<__m128i>()));
            let vb1 = _mm256_cvtph_ps(_mm_loadu_si128(b_ptr.add(i + 8).cast::<__m128i>()));
            let vb2 = _mm256_cvtph_ps(_mm_loadu_si128(b_ptr.add(i + 16).cast::<__m128i>()));
            let vb3 = _mm256_cvtph_ps(_mm_loadu_si128(b_ptr.add(i + 24).cast::<__m128i>()));

            acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
            acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
            acc2 = _mm256_fmadd_ps(va2, vb2, acc2);
            acc3 = _mm256_fmadd_ps(va3, vb3, acc3);
        }
        i += 32;
    }

    while i + 8 <= n {
        // SAFETY: `i + 8 <= n <= a_f32.len().min(b_f16.len())`.
        unsafe {
            let va = _mm256_loadu_ps(a_ptr.add(i));
            let vb = _mm256_cvtph_ps(_mm_loadu_si128(b_ptr.add(i).cast::<__m128i>()));
            acc0 = _mm256_fmadd_ps(va, vb, acc0);
        }
        i += 8;
    }

    let sum_256 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let lo = _mm256_castps256_ps128(sum_256);
    let hi = _mm256_extractf128_ps::<1>(sum_256);
    let sum_128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum_128);
    let sums = _mm_add_ps(sum_128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let mut sum = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));

    while i < n {
        sum += a_f32[i] * b_f16[i].to_f32();
        i += 1;
    }
    sum
}
