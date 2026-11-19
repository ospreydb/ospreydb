//! Safe SIMD backend trait, portable `ScalarBackend`, and runtime `SimdEngine` dispatcher.

use osprey_core::{OspreyError, OspreyResult};

use crate::detect::{detect_best_simd_level, is_level_supported, SimdLevel};

/// Unified safe contract for hardware-accelerated and scalar kernels across Phases 4–7.
///
/// Implementations must never allocate heap memory inside any hot-loop method.
pub trait SimdBackend: Send + Sync + 'static {
    /// Returns the active SIMD tier of this backend.
    fn level(&self) -> SimdLevel;

    /// Phase 4: Applies XOR sign masks and orthogonal FWHT `O(d log d)` in-place.
    fn fwht_rotate_in_place(&self, data: &mut [f32], signs_xor_mask: &[u32]) -> OspreyResult<()>;

    /// Phase 5: Computes Hamming distance `sum(popcnt(a[w] ^ b[w]))` over 64-bit word slices.
    fn hamming_distance_u64(&self, a: &[u64], b: &[u64]) -> u32;

    /// Phase 5: Finds the nearest binary centroid `(best_idx, best_hamming_dist)` for `vector_bits`.
    fn batch_nearest_binary_centroid(
        &self,
        vector_bits: &[u64],
        centroids_bits: &[u64],
        words_per_vec: usize,
        num_centroids: usize,
    ) -> (usize, u32);

    /// Phase 6: Computes the exact `int4 x 1-bit` inner product `sum_i q_int[i] * (2 * b_x[i] - 1)`
    /// with zero floating-point operations in the inner loop:
    /// `2 * sum_w sum_{p=0..4} (1 << p) * popcnt(query_planes[w][p] & !(query_signs[w] ^ db_bits[w])) - query_abs_sum`.
    fn rabitq_4bit_dot_u64(
        &self,
        query_planes: &[[u64; 4]],
        query_signs: &[u64],
        query_abs_sum: i32,
        db_bits: &[u64],
    ) -> i32;

    /// Phase 7: Evaluates one nibble-packed `.osr` document token (32B Layer 1 + 32B Layer 2)
    /// against `64 * 16`-entry `int8` query LUTs with zero float muls in the subspace loop.
    fn lutrank_token_sim_i32(
        &self,
        doc_l1_packed: &[u8],
        doc_l2_packed: &[u8],
        q_lut1_i8: &[i8],
        q_lut2_i8: &[i8],
        scale1: f32,
        scale2: f32,
    ) -> f32;

    /// Evaluates `MaxSim(Q, D)` over a document's contiguous 64-byte-aligned `.osr`
    /// token buffer against `[Q, N_sub, 16]` `int8` query LUTs.
    #[allow(clippy::too_many_arguments)]
    fn lutrank_doc_maxsim(
        &self,
        doc_tokens_bytes: &[u8],
        num_doc_tokens: usize,
        num_subspaces: usize,
        q_lut1_i8: &[i8],
        q_lut2_i8: &[i8],
        num_query_tokens: usize,
        scale1: f32,
        scale2: f32,
    ) -> f32 {
        if num_doc_tokens == 0 || num_query_tokens == 0 || num_subspaces == 0 {
            return 0.0;
        }
        let layer_bytes = num_subspaces / 2;
        let token_bytes = layer_bytes * 2;
        let lut_stride = num_subspaces * 16;
        if doc_tokens_bytes.len() < num_doc_tokens * token_bytes
            || q_lut1_i8.len() < num_query_tokens * lut_stride
            || q_lut2_i8.len() < num_query_tokens * lut_stride
        {
            return 0.0;
        }

        let mut total = 0.0_f32;
        for q_idx in 0..num_query_tokens {
            let ql1 = &q_lut1_i8[q_idx * lut_stride..(q_idx + 1) * lut_stride];
            let ql2 = &q_lut2_i8[q_idx * lut_stride..(q_idx + 1) * lut_stride];
            let mut best = f32::NEG_INFINITY;
            for t_idx in 0..num_doc_tokens {
                let tok_off = t_idx * token_bytes;
                let doc_l1 = &doc_tokens_bytes[tok_off..tok_off + layer_bytes];
                let doc_l2 = &doc_tokens_bytes[tok_off + layer_bytes..tok_off + token_bytes];
                let sim = self.lutrank_token_sim_i32(doc_l1, doc_l2, ql1, ql2, scale1, scale2);
                if sim > best {
                    best = sim;
                }
            }
            total += best;
        }
        total
    }
}

impl osprey_core::cluster::BinaryKMeansKernel for SimdEngine {
    #[inline]
    fn fwht_rotate_in_place(&self, data: &mut [f32], signs_xor_mask: &[u32]) -> OspreyResult<()> {
        SimdBackend::fwht_rotate_in_place(self, data, signs_xor_mask)
    }

    #[inline]
    fn hamming_distance_u64(&self, a: &[u64], b: &[u64]) -> u32 {
        SimdBackend::hamming_distance_u64(self, a, b)
    }

    #[inline]
    fn batch_nearest_binary_centroid(
        &self,
        vector_bits: &[u64],
        centroids_bits: &[u64],
        words_per_vec: usize,
        num_centroids: usize,
    ) -> (usize, u32) {
        SimdBackend::batch_nearest_binary_centroid(
            self,
            vector_bits,
            centroids_bits,
            words_per_vec,
            num_centroids,
        )
    }
}

impl osprey_core::cluster::BinaryKMeansKernel for ScalarBackend {
    #[inline]
    fn fwht_rotate_in_place(&self, data: &mut [f32], signs_xor_mask: &[u32]) -> OspreyResult<()> {
        SimdBackend::fwht_rotate_in_place(self, data, signs_xor_mask)
    }

    #[inline]
    fn hamming_distance_u64(&self, a: &[u64], b: &[u64]) -> u32 {
        SimdBackend::hamming_distance_u64(self, a, b)
    }

    #[inline]
    fn batch_nearest_binary_centroid(
        &self,
        vector_bits: &[u64],
        centroids_bits: &[u64],
        words_per_vec: usize,
        num_centroids: usize,
    ) -> (usize, u32) {
        SimdBackend::batch_nearest_binary_centroid(
            self,
            vector_bits,
            centroids_bits,
            words_per_vec,
            num_centroids,
        )
    }
}

/// Portable, deterministic scalar reference implementation of `SimdBackend`.
#[derive(Copy, Clone, Debug, Default)]
pub struct ScalarBackend;

impl osprey_core::rabitq::RabitqFwhtKernel for ScalarBackend {
    #[inline]
    fn fwht_rotate_in_place(&self, data: &mut [f32], signs_xor_mask: &[u32]) -> OspreyResult<()> {
        SimdBackend::fwht_rotate_in_place(self, data, signs_xor_mask)
    }
}

impl osprey_core::rabitq::RabitqScanKernel for ScalarBackend {
    #[inline]
    fn rabitq_4bit_dot_u64(
        &self,
        query_planes: &[[u64; 4]],
        query_signs: &[u64],
        query_abs_sum: i32,
        db_bits: &[u64],
    ) -> i32 {
        SimdBackend::rabitq_4bit_dot_u64(self, query_planes, query_signs, query_abs_sum, db_bits)
    }
}

impl osprey_core::rabitq::RabitqFwhtKernel for SimdEngine {
    #[inline]
    fn fwht_rotate_in_place(&self, data: &mut [f32], signs_xor_mask: &[u32]) -> OspreyResult<()> {
        SimdBackend::fwht_rotate_in_place(self, data, signs_xor_mask)
    }
}

impl osprey_core::rabitq::RabitqScanKernel for SimdEngine {
    #[inline(always)]
    fn prefetch_l1(&self, bytes: &[u8]) {
        prefetch_l1_bytes(bytes);
    }

    #[inline]
    fn rabitq_4bit_dot_u64(
        &self,
        query_planes: &[[u64; 4]],
        query_signs: &[u64],
        query_abs_sum: i32,
        db_bits: &[u64],
    ) -> i32 {
        SimdBackend::rabitq_4bit_dot_u64(self, query_planes, query_signs, query_abs_sum, db_bits)
    }
}

#[inline(always)]
fn prefetch_l1_bytes(bytes: &[u8]) {
    #[cfg(all(not(miri), target_arch = "x86_64"))]
    if !bytes.is_empty() {
        // SAFETY: `_mm_prefetch` is a non-faulting architectural cache-hint instruction;
        // `bytes.as_ptr()` points to at least 1 valid byte.
        unsafe {
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(
                bytes.as_ptr().cast::<i8>(),
            );
        }
    }
    #[cfg(all(not(miri), target_arch = "x86"))]
    if !bytes.is_empty() {
        // SAFETY: `_mm_prefetch` is a non-faulting cache-hint instruction.
        unsafe {
            std::arch::x86::_mm_prefetch::<{ std::arch::x86::_MM_HINT_T0 }>(
                bytes.as_ptr().cast::<i8>(),
            );
        }
    }
    let _ = bytes;
}

impl osprey_core::LutRankKernel for ScalarBackend {
    #[inline]
    fn lutrank_token_sim_i32(
        &self,
        doc_l1_packed: &[u8],
        doc_l2_packed: &[u8],
        q_lut1_i8: &[i8],
        q_lut2_i8: &[i8],
        scale1: f32,
        scale2: f32,
    ) -> f32 {
        SimdBackend::lutrank_token_sim_i32(
            self,
            doc_l1_packed,
            doc_l2_packed,
            q_lut1_i8,
            q_lut2_i8,
            scale1,
            scale2,
        )
    }

    #[inline]
    fn lutrank_doc_maxsim(
        &self,
        doc_tokens_bytes: &[u8],
        num_doc_tokens: usize,
        num_subspaces: usize,
        q_lut1_i8: &[i8],
        q_lut2_i8: &[i8],
        num_query_tokens: usize,
        scale1: f32,
        scale2: f32,
    ) -> f32 {
        SimdBackend::lutrank_doc_maxsim(
            self,
            doc_tokens_bytes,
            num_doc_tokens,
            num_subspaces,
            q_lut1_i8,
            q_lut2_i8,
            num_query_tokens,
            scale1,
            scale2,
        )
    }
}

impl osprey_core::LutRankKernel for SimdEngine {
    #[inline(always)]
    fn prefetch_l1(&self, bytes: &[u8]) {
        prefetch_l1_bytes(bytes);
    }

    #[inline]
    fn lutrank_token_sim_i32(
        &self,
        doc_l1_packed: &[u8],
        doc_l2_packed: &[u8],
        q_lut1_i8: &[i8],
        q_lut2_i8: &[i8],
        scale1: f32,
        scale2: f32,
    ) -> f32 {
        SimdBackend::lutrank_token_sim_i32(
            self,
            doc_l1_packed,
            doc_l2_packed,
            q_lut1_i8,
            q_lut2_i8,
            scale1,
            scale2,
        )
    }

    #[inline]
    fn lutrank_doc_maxsim(
        &self,
        doc_tokens_bytes: &[u8],
        num_doc_tokens: usize,
        num_subspaces: usize,
        q_lut1_i8: &[i8],
        q_lut2_i8: &[i8],
        num_query_tokens: usize,
        scale1: f32,
        scale2: f32,
    ) -> f32 {
        SimdBackend::lutrank_doc_maxsim(
            self,
            doc_tokens_bytes,
            num_doc_tokens,
            num_subspaces,
            q_lut1_i8,
            q_lut2_i8,
            num_query_tokens,
            scale1,
            scale2,
        )
    }
}

impl SimdBackend for ScalarBackend {
    #[inline]
    fn level(&self) -> SimdLevel {
        SimdLevel::Scalar
    }

    fn fwht_rotate_in_place(&self, data: &mut [f32], signs_xor_mask: &[u32]) -> OspreyResult<()> {
        let d = data.len();
        if d == 0 || !d.is_power_of_two() || d != signs_xor_mask.len() {
            return Err(OspreyError::InvalidDimension { dim: d });
        }

        for (val, &mask) in data.iter_mut().zip(signs_xor_mask.iter()) {
            *val = f32::from_bits(val.to_bits() ^ mask);
        }

        let mut h = 1usize;
        while h < d {
            let step = h * 2;
            for i in (0..d).step_by(step) {
                for j in i..(i + h) {
                    let a = data[j];
                    let b = data[j + h];
                    data[j] = a + b;
                    data[j + h] = a - b;
                }
            }
            h = step;
        }

        let d_u32 = u32::try_from(d).map_err(|_| OspreyError::InvalidDimension { dim: d })?;
        #[allow(clippy::cast_possible_truncation)]
        let inv_sqrt_d = (1.0_f64 / f64::from(d_u32).sqrt()) as f32;
        for val in data.iter_mut() {
            *val *= inv_sqrt_d;
        }

        Ok(())
    }

    #[inline]
    fn hamming_distance_u64(&self, a: &[u64], b: &[u64]) -> u32 {
        let n = a.len().min(b.len());
        let mut acc0 = 0_u32;
        let mut acc1 = 0_u32;
        let mut acc2 = 0_u32;
        let mut acc3 = 0_u32;
        let mut i = 0usize;

        while i + 4 <= n {
            acc0 += (a[i] ^ b[i]).count_ones();
            acc1 += (a[i + 1] ^ b[i + 1]).count_ones();
            acc2 += (a[i + 2] ^ b[i + 2]).count_ones();
            acc3 += (a[i + 3] ^ b[i + 3]).count_ones();
            i += 4;
        }

        let mut total = acc0 + acc1 + acc2 + acc3;
        while i < n {
            total += (a[i] ^ b[i]).count_ones();
            i += 1;
        }
        total
    }

    fn batch_nearest_binary_centroid(
        &self,
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

        let v = &vector_bits[..words_per_vec];
        let mut best_idx = 0usize;
        let mut best_dist = u32::MAX;
        let mut c = 0usize;

        while c + 4 <= valid_centroids {
            let base = c * words_per_vec;
            let c0 = &centroids_bits[base..base + words_per_vec];
            let c1 = &centroids_bits[base + words_per_vec..base + 2 * words_per_vec];
            let c2 = &centroids_bits[base + 2 * words_per_vec..base + 3 * words_per_vec];
            let c3 = &centroids_bits[base + 3 * words_per_vec..base + 4 * words_per_vec];

            let mut acc0 = 0_u32;
            let mut acc1 = 0_u32;
            let mut acc2 = 0_u32;
            let mut acc3 = 0_u32;

            for w in 0..words_per_vec {
                let vw = v[w];
                acc0 += (vw ^ c0[w]).count_ones();
                acc1 += (vw ^ c1[w]).count_ones();
                acc2 += (vw ^ c2[w]).count_ones();
                acc3 += (vw ^ c3[w]).count_ones();
            }

            if acc0 < best_dist {
                best_dist = acc0;
                best_idx = c;
            }
            if acc1 < best_dist {
                best_dist = acc1;
                best_idx = c + 1;
            }
            if acc2 < best_dist {
                best_dist = acc2;
                best_idx = c + 2;
            }
            if acc3 < best_dist {
                best_dist = acc3;
                best_idx = c + 3;
            }

            c += 4;
        }

        while c < valid_centroids {
            let start = c * words_per_vec;
            let c_slice = &centroids_bits[start..start + words_per_vec];
            let dist = self.hamming_distance_u64(v, c_slice);
            if dist < best_dist {
                best_dist = dist;
                best_idx = c;
            }
            c += 1;
        }

        (best_idx, best_dist)
    }

    #[inline]
    fn rabitq_4bit_dot_u64(
        &self,
        query_planes: &[[u64; 4]],
        query_signs: &[u64],
        query_abs_sum: i32,
        db_bits: &[u64],
    ) -> i32 {
        let mut weighted_matches = 0_i64;
        for ((planes, &q_sign), &db_word) in query_planes
            .iter()
            .zip(query_signs.iter())
            .zip(db_bits.iter())
        {
            let match_word = !(q_sign ^ db_word);
            let p0 = i64::from((planes[0] & match_word).count_ones());
            let p1 = i64::from((planes[1] & match_word).count_ones());
            let p2 = i64::from((planes[2] & match_word).count_ones());
            let p3 = i64::from((planes[3] & match_word).count_ones());
            weighted_matches += p0 + (p1 << 1) + (p2 << 2) + (p3 << 3);
        }
        let res = weighted_matches.wrapping_mul(2) - i64::from(query_abs_sum);
        i32::try_from(res).unwrap_or(i32::MAX)
    }

    #[inline]
    fn lutrank_token_sim_i32(
        &self,
        doc_l1_packed: &[u8],
        doc_l2_packed: &[u8],
        q_lut1_i8: &[i8],
        q_lut2_i8: &[i8],
        scale1: f32,
        scale2: f32,
    ) -> f32 {
        let pairs = doc_l1_packed.len().min(doc_l2_packed.len());
        let mut acc_l1 = 0_i32;
        let mut acc_l2 = 0_i32;

        if pairs == 32 && q_lut1_i8.len() >= 1024 && q_lut2_i8.len() >= 1024 {
            for s in 0..32 {
                let b1 = doc_l1_packed[s];
                let b2 = doc_l2_packed[s];
                let c1_even = usize::from(b1 & 0x0F);
                let c1_odd = usize::from(b1 >> 4);
                let c2_even = usize::from(b2 & 0x0F);
                let c2_odd = usize::from(b2 >> 4);

                let base_even = (2 * s) * 16;
                let base_odd = (2 * s + 1) * 16;
                acc_l1 += i32::from(q_lut1_i8[base_even + c1_even])
                    + i32::from(q_lut1_i8[base_odd + c1_odd]);
                acc_l2 += i32::from(q_lut2_i8[base_even + c2_even])
                    + i32::from(q_lut2_i8[base_odd + c2_odd]);
            }
        } else {
            for s in 0..pairs {
                let b1 = doc_l1_packed[s];
                let b2 = doc_l2_packed[s];
                let c1_even = usize::from(b1 & 0x0F);
                let c1_odd = usize::from(b1 >> 4);
                let c2_even = usize::from(b2 & 0x0F);
                let c2_odd = usize::from(b2 >> 4);

                let base_even = (2 * s) * 16;
                let base_odd = (2 * s + 1) * 16;

                if let (Some(&v1e), Some(&v1o), Some(&v2e), Some(&v2o)) = (
                    q_lut1_i8.get(base_even + c1_even),
                    q_lut1_i8.get(base_odd + c1_odd),
                    q_lut2_i8.get(base_even + c2_even),
                    q_lut2_i8.get(base_odd + c2_odd),
                ) {
                    acc_l1 += i32::from(v1e) + i32::from(v1o);
                    acc_l2 += i32::from(v2e) + i32::from(v2o);
                }
            }
        }

        #[allow(clippy::cast_precision_loss)]
        {
            (acc_l1 as f32) * scale1 + (acc_l2 as f32) * scale2
        }
    }
}

/// Runtime SIMD dispatcher supporting automatic hardware selection or forced level testing.
#[derive(Copy, Clone, Debug)]
pub struct SimdEngine {
    level: SimdLevel,
}

impl SimdEngine {
    /// Constructs a `SimdEngine` bound to the highest SIMD tier detected on this CPU.
    #[inline]
    pub fn new_auto() -> Self {
        Self {
            level: detect_best_simd_level(),
        }
    }

    /// Constructs a `SimdEngine` targeting `level`, falling back to `Scalar` if unsupported on host.
    #[inline]
    pub fn with_level(level: SimdLevel) -> Self {
        let active = if is_level_supported(level) {
            level
        } else {
            SimdLevel::Scalar
        };
        Self { level: active }
    }

    /// Attempts to construct a `SimdEngine` strictly at `level`, returning `None` if unsupported.
    #[inline]
    pub fn try_with_level(level: SimdLevel) -> Option<Self> {
        is_level_supported(level).then_some(Self { level })
    }

    /// Returns the active `SimdLevel` selected by this engine.
    #[inline]
    pub fn active_level(&self) -> SimdLevel {
        self.level
    }
}

impl Default for SimdEngine {
    #[inline]
    fn default() -> Self {
        Self::new_auto()
    }
}

impl SimdBackend for SimdEngine {
    #[inline]
    fn level(&self) -> SimdLevel {
        self.level
    }

    #[inline]
    fn fwht_rotate_in_place(&self, data: &mut [f32], signs_xor_mask: &[u32]) -> OspreyResult<()> {
        match self.level {
            #[cfg(all(not(miri), any(target_arch = "x86", target_arch = "x86_64")))]
            SimdLevel::Avx512 if data.len() >= 16 && is_level_supported(SimdLevel::Avx512) => {
                // SAFETY: `is_level_supported(Avx512)` confirmed `avx512f` support on host.
                unsafe { crate::avx512::fwht_rotate_in_place_avx512(data, signs_xor_mask) }
            }
            #[cfg(all(not(miri), any(target_arch = "x86", target_arch = "x86_64")))]
            SimdLevel::Avx2 | SimdLevel::Avx512
                if data.len() >= 8 && is_level_supported(SimdLevel::Avx2) =>
            {
                // SAFETY: `is_level_supported(Avx2)` confirmed `avx2,fma` support on host.
                unsafe { crate::avx2::fwht_rotate_in_place_avx2(data, signs_xor_mask) }
            }
            #[cfg(all(not(miri), target_arch = "aarch64"))]
            SimdLevel::Neon if data.len() >= 4 && is_level_supported(SimdLevel::Neon) => {
                // SAFETY: `is_level_supported(Neon)` confirmed `neon` support on host.
                unsafe { crate::neon::fwht_rotate_in_place_neon(data, signs_xor_mask) }
            }
            _ => ScalarBackend.fwht_rotate_in_place(data, signs_xor_mask),
        }
    }

    #[inline]
    fn hamming_distance_u64(&self, a: &[u64], b: &[u64]) -> u32 {
        match self.level {
            #[cfg(all(not(miri), any(target_arch = "x86", target_arch = "x86_64")))]
            SimdLevel::Avx512 if is_level_supported(SimdLevel::Avx512) => {
                // SAFETY: `is_level_supported(Avx512)` confirmed `avx512f` and `avx512vpopcntdq`.
                unsafe { crate::avx512::hamming_distance_u64_avx512(a, b) }
            }
            #[cfg(all(not(miri), any(target_arch = "x86", target_arch = "x86_64")))]
            SimdLevel::Avx2 | SimdLevel::Avx512 if is_level_supported(SimdLevel::Avx2) => {
                // SAFETY: `is_level_supported(Avx2)` confirmed `avx2` and `popcnt`.
                unsafe { crate::avx2::hamming_distance_u64_avx2(a, b) }
            }
            #[cfg(all(not(miri), target_arch = "aarch64"))]
            SimdLevel::Neon if is_level_supported(SimdLevel::Neon) => {
                // SAFETY: `is_level_supported(Neon)` confirmed `neon`.
                unsafe { crate::neon::hamming_distance_u64_neon(a, b) }
            }
            _ => ScalarBackend.hamming_distance_u64(a, b),
        }
    }

    #[inline]
    fn batch_nearest_binary_centroid(
        &self,
        vector_bits: &[u64],
        centroids_bits: &[u64],
        words_per_vec: usize,
        num_centroids: usize,
    ) -> (usize, u32) {
        match self.level {
            #[cfg(all(not(miri), any(target_arch = "x86", target_arch = "x86_64")))]
            SimdLevel::Avx512 if is_level_supported(SimdLevel::Avx512) => {
                // SAFETY: `is_level_supported(Avx512)` confirmed `avx512f` and `avx512vpopcntdq`.
                unsafe {
                    crate::avx512::batch_nearest_binary_centroid_avx512(
                        vector_bits,
                        centroids_bits,
                        words_per_vec,
                        num_centroids,
                    )
                }
            }
            #[cfg(all(not(miri), any(target_arch = "x86", target_arch = "x86_64")))]
            SimdLevel::Avx2 | SimdLevel::Avx512 if is_level_supported(SimdLevel::Avx2) => {
                // SAFETY: `is_level_supported(Avx2)` confirmed `avx2` and `popcnt`.
                unsafe {
                    crate::avx2::batch_nearest_binary_centroid_avx2(
                        vector_bits,
                        centroids_bits,
                        words_per_vec,
                        num_centroids,
                    )
                }
            }
            #[cfg(all(not(miri), target_arch = "aarch64"))]
            SimdLevel::Neon if is_level_supported(SimdLevel::Neon) => {
                // SAFETY: `is_level_supported(Neon)` confirmed `neon`.
                unsafe {
                    crate::neon::batch_nearest_binary_centroid_neon(
                        vector_bits,
                        centroids_bits,
                        words_per_vec,
                        num_centroids,
                    )
                }
            }
            _ => ScalarBackend.batch_nearest_binary_centroid(
                vector_bits,
                centroids_bits,
                words_per_vec,
                num_centroids,
            ),
        }
    }

    #[inline]
    fn rabitq_4bit_dot_u64(
        &self,
        query_planes: &[[u64; 4]],
        query_signs: &[u64],
        query_abs_sum: i32,
        db_bits: &[u64],
    ) -> i32 {
        let _n_words = query_planes.len().min(query_signs.len()).min(db_bits.len());
        match self.level {
            #[cfg(all(not(miri), any(target_arch = "x86", target_arch = "x86_64")))]
            SimdLevel::Avx512 if _n_words >= 8 && is_level_supported(SimdLevel::Avx512) => {
                // SAFETY: `is_level_supported(Avx512)` confirmed `avx512f` and `avx512vpopcntdq`.
                unsafe {
                    crate::avx512::rabitq_4bit_dot_u64_avx512(
                        query_planes,
                        query_signs,
                        query_abs_sum,
                        db_bits,
                    )
                }
            }
            #[cfg(all(not(miri), any(target_arch = "x86", target_arch = "x86_64")))]
            SimdLevel::Avx2 | SimdLevel::Avx512
                if _n_words >= 4 && is_level_supported(SimdLevel::Avx2) =>
            {
                // SAFETY: `is_level_supported(Avx2)` confirmed `avx2` support on host.
                unsafe {
                    crate::avx2::rabitq_4bit_dot_u64_avx2(
                        query_planes,
                        query_signs,
                        query_abs_sum,
                        db_bits,
                    )
                }
            }
            #[cfg(all(not(miri), target_arch = "aarch64"))]
            SimdLevel::Neon if _n_words >= 2 && is_level_supported(SimdLevel::Neon) => {
                // SAFETY: `is_level_supported(Neon)` confirmed `neon` support on host.
                unsafe {
                    crate::neon::rabitq_4bit_dot_u64_neon(
                        query_planes,
                        query_signs,
                        query_abs_sum,
                        db_bits,
                    )
                }
            }
            _ => {
                ScalarBackend.rabitq_4bit_dot_u64(query_planes, query_signs, query_abs_sum, db_bits)
            }
        }
    }

    #[inline]
    fn lutrank_token_sim_i32(
        &self,
        doc_l1_packed: &[u8],
        doc_l2_packed: &[u8],
        q_lut1_i8: &[i8],
        q_lut2_i8: &[i8],
        scale1: f32,
        scale2: f32,
    ) -> f32 {
        match self.level {
            #[cfg(all(not(miri), any(target_arch = "x86", target_arch = "x86_64")))]
            SimdLevel::Avx512 if is_level_supported(SimdLevel::Avx512) => {
                // SAFETY: `is_level_supported(Avx512)` confirmed `avx512f` and `avx512bw`.
                unsafe {
                    crate::avx512::lutrank_token_sim_i32_avx512(
                        doc_l1_packed,
                        doc_l2_packed,
                        q_lut1_i8,
                        q_lut2_i8,
                        scale1,
                        scale2,
                    )
                }
            }
            #[cfg(all(not(miri), any(target_arch = "x86", target_arch = "x86_64")))]
            SimdLevel::Avx2 | SimdLevel::Avx512 if is_level_supported(SimdLevel::Avx2) => {
                // SAFETY: `is_level_supported(Avx2)` confirmed `avx2` support on host.
                unsafe {
                    crate::avx2::lutrank_token_sim_i32_avx2(
                        doc_l1_packed,
                        doc_l2_packed,
                        q_lut1_i8,
                        q_lut2_i8,
                        scale1,
                        scale2,
                    )
                }
            }
            #[cfg(all(not(miri), target_arch = "aarch64"))]
            SimdLevel::Neon if is_level_supported(SimdLevel::Neon) => {
                // SAFETY: `is_level_supported(Neon)` confirmed `neon` support on host.
                unsafe {
                    crate::neon::lutrank_token_sim_i32_neon(
                        doc_l1_packed,
                        doc_l2_packed,
                        q_lut1_i8,
                        q_lut2_i8,
                        scale1,
                        scale2,
                    )
                }
            }
            _ => ScalarBackend.lutrank_token_sim_i32(
                doc_l1_packed,
                doc_l2_packed,
                q_lut1_i8,
                q_lut2_i8,
                scale1,
                scale2,
            ),
        }
    }

    #[inline]
    fn lutrank_doc_maxsim(
        &self,
        doc_tokens_bytes: &[u8],
        num_doc_tokens: usize,
        num_subspaces: usize,
        q_lut1_i8: &[i8],
        q_lut2_i8: &[i8],
        num_query_tokens: usize,
        scale1: f32,
        scale2: f32,
    ) -> f32 {
        match self.level {
            #[cfg(all(not(miri), any(target_arch = "x86", target_arch = "x86_64")))]
            SimdLevel::Avx512 if is_level_supported(SimdLevel::Avx512) => {
                // SAFETY: `is_level_supported(Avx512)` confirmed `avx512f` and `avx512bw`.
                unsafe {
                    crate::avx512::lutrank_doc_maxsim_avx512(
                        doc_tokens_bytes,
                        num_doc_tokens,
                        num_subspaces,
                        q_lut1_i8,
                        q_lut2_i8,
                        num_query_tokens,
                        scale1,
                        scale2,
                    )
                }
            }
            #[cfg(all(not(miri), any(target_arch = "x86", target_arch = "x86_64")))]
            SimdLevel::Avx2 | SimdLevel::Avx512 if is_level_supported(SimdLevel::Avx2) => {
                // SAFETY: `is_level_supported(Avx2)` confirmed `avx2` support on host.
                unsafe {
                    crate::avx2::lutrank_doc_maxsim_avx2(
                        doc_tokens_bytes,
                        num_doc_tokens,
                        num_subspaces,
                        q_lut1_i8,
                        q_lut2_i8,
                        num_query_tokens,
                        scale1,
                        scale2,
                    )
                }
            }
            #[cfg(all(not(miri), target_arch = "aarch64"))]
            SimdLevel::Neon if is_level_supported(SimdLevel::Neon) => {
                // SAFETY: `is_level_supported(Neon)` confirmed `neon` support on host.
                unsafe {
                    crate::neon::lutrank_doc_maxsim_neon(
                        doc_tokens_bytes,
                        num_doc_tokens,
                        num_subspaces,
                        q_lut1_i8,
                        q_lut2_i8,
                        num_query_tokens,
                        scale1,
                        scale2,
                    )
                }
            }
            _ => ScalarBackend.lutrank_doc_maxsim(
                doc_tokens_bytes,
                num_doc_tokens,
                num_subspaces,
                q_lut1_i8,
                q_lut2_i8,
                num_query_tokens,
                scale1,
                scale2,
            ),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use osprey_core::math::{fwht_scalar_in_place, RademacherSigns, SplitMix64};

    use super::*;

    #[test]
    fn scalar_backend_fwht_matches_core_oracle() {
        let engine = SimdEngine::with_level(SimdLevel::Scalar);
        assert_eq!(engine.active_level(), SimdLevel::Scalar);

        let signs = RademacherSigns::from_seed(128, 0x4F53_5052_0001).unwrap();
        let mut a = vec![0.25_f32; 128];
        let mut b = a.clone();

        engine
            .fwht_rotate_in_place(&mut a, signs.xor_masks())
            .unwrap();
        fwht_scalar_in_place(&mut b, &signs).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn active_simd_backend_matches_scalar_across_small_and_tiled_dims() {
        let auto = SimdEngine::new_auto();
        let scalar = SimdEngine::with_level(SimdLevel::Scalar);

        for dim in [64usize, 128, 256, 512, 1024, 2048] {
            let signs = RademacherSigns::from_seed(dim, 0x4F53_5052_0001).unwrap();
            let mut input = vec![0.0_f32; dim];
            for (i, val) in input.iter_mut().enumerate() {
                let x = f32::from(u16::try_from(i).unwrap());
                *val = ((x * 0.17).sin() + (x * 0.03).cos()) / (1.0 + x * 0.01);
            }
            let mut got = input.clone();
            let mut expected = input;

            auto.fwht_rotate_in_place(&mut got, signs.xor_masks())
                .unwrap();
            scalar
                .fwht_rotate_in_place(&mut expected, signs.xor_masks())
                .unwrap();

            let max_diff = got
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_diff <= 1e-5,
                "dim={dim} level={:?} max_diff={max_diff}",
                auto.active_level()
            );
        }
    }

    #[test]
    fn scalar_backend_hamming_rabitq_and_lutrank_sanity() {
        let engine = SimdEngine::new_auto();
        let a = [0xFFFF_0000_FFFF_0000_u64, 0xAAAA_AAAA_AAAA_AAAA_u64];
        let b = [0x0000_0000_FFFF_0000_u64, 0xAAAA_AAAA_AAAA_AAAA_u64];
        assert_eq!(engine.hamming_distance_u64(&a, &b), 16);

        let centroids = [
            0x0000_0000_0000_0000_u64,
            0x0000_0000_0000_0000_u64,
            0xFFFF_0000_FFFF_0000_u64,
            0xAAAA_AAAA_AAAA_AAAA_u64,
        ];
        assert_eq!(
            engine.batch_nearest_binary_centroid(&a, &centroids, 2, 2),
            (1, 0)
        );

        let planes = [[0b0011_u64, 0b0101_u64, 0b0001_u64, 0b1000_u64]];
        let signs = [u64::MAX];
        let query_abs_sum = 5 + 3 + 2 + 8;
        let db = [0b1011_u64];
        // Naive: +5 +3 -2 +8 = 14; bitwise: 2 * (2 + 2 + 4 + 8) - 18 = 14
        assert_eq!(
            engine.rabitq_4bit_dot_u64(&planes, &signs, query_abs_sum, &db),
            14
        );

        let doc_l1 = [0x10_u8; 32]; // even=0, odd=1 across 32 pairs
        let doc_l2 = [0x00_u8; 32]; // even=0, odd=0
        let mut lut1 = [0_i8; 64 * 16];
        let mut lut2 = [0_i8; 64 * 16];
        for s in 0..64 {
            lut1[s * 16] = 2;
            lut1[s * 16 + 1] = 3;
            lut2[s * 16] = 1;
        }
        // Per pair: L1 gets 2 + 3 = 5 -> 32 * 5 = 160; L2 gets 64 * 1 = 64
        let sim = engine.lutrank_token_sim_i32(&doc_l1, &doc_l2, &lut1, &lut2, 0.5, 0.25);
        assert!((sim - (160.0 * 0.5 + 64.0 * 0.25)).abs() < 1e-5);
    }

    fn naive_hamming_u64(a: &[u64], b: &[u64]) -> u32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x ^ y).count_ones())
            .sum()
    }

    fn supported_backends() -> Vec<SimdEngine> {
        [
            SimdLevel::Scalar,
            SimdLevel::Avx2,
            SimdLevel::Avx512,
            SimdLevel::Neon,
        ]
        .into_iter()
        .filter_map(SimdEngine::try_with_level)
        .collect()
    }

    #[test]
    fn simd_backends_hamming_distance_equivalence_and_xnor_complement() {
        let backends = supported_backends();
        assert!(!backends.is_empty());

        let mut rng = SplitMix64::new(0x5EED_0069_0001);
        // Test canonical dimensions (64..2048), non-power-of-two tail lengths, and > 64-word flush boundary.
        let word_counts = [0usize, 1, 2, 3, 4, 5, 8, 9, 16, 32, 65, 128];

        for &words in &word_counts {
            let mut a = vec![0_u64; words];
            let mut b = vec![0_u64; words];
            rng.fill_words(&mut a);
            rng.fill_words(&mut b);

            let expected = naive_hamming_u64(&a, &b);
            let xnor_sim: u32 = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| (!(x ^ y)).count_ones())
                .sum();
            assert_eq!(expected + xnor_sim, u32::try_from(words * 64).unwrap());

            for engine in &backends {
                let got = engine.hamming_distance_u64(&a, &b);
                assert_eq!(
                    got,
                    expected,
                    "hamming mismatch for words={words} level={:?}",
                    engine.active_level()
                );
            }
        }
    }

    #[test]
    fn simd_backends_batch_nearest_centroid_equivalence_all_dims_and_k() {
        let backends = supported_backends();
        let mut rng = SplitMix64::new(0x5EED_0072_0002);

        for dim in [64usize, 128, 256, 512, 1024, 2048] {
            let words_per_vec = dim / 64;
            let mut query = vec![0_u64; words_per_vec];
            rng.fill_words(&mut query);

            for k in [1usize, 7, 16, 64, 67] {
                let mut centroids = vec![0_u64; k * words_per_vec];
                rng.fill_words(&mut centroids);

                // Inject a closer match near the end (inside the % 4 tail when k % 4 != 0).
                if k >= 3 {
                    let target_k = k - 1;
                    let start = target_k * words_per_vec;
                    centroids[start..start + words_per_vec].copy_from_slice(&query);
                    centroids[start] ^= 0b101; // Hamming distance = 2
                }

                let expected = ScalarBackend.batch_nearest_binary_centroid(
                    &query,
                    &centroids,
                    words_per_vec,
                    k,
                );

                // Cross-check ScalarBackend against naive centroid-by-centroid scan.
                let mut naive_best = (0usize, u32::MAX);
                for idx in 0..k {
                    let slice = &centroids[idx * words_per_vec..(idx + 1) * words_per_vec];
                    let dist = naive_hamming_u64(&query, slice);
                    if dist < naive_best.1 {
                        naive_best = (idx, dist);
                    }
                }
                assert_eq!(expected, naive_best, "dim={dim} k={k}");

                for engine in &backends {
                    let got =
                        engine.batch_nearest_binary_centroid(&query, &centroids, words_per_vec, k);
                    assert_eq!(
                        got,
                        expected,
                        "nearest centroid mismatch for dim={dim} k={k} level={:?}",
                        engine.active_level()
                    );
                }
            }
        }
    }

    #[test]
    fn rabitq_4bit_dot_zero_error_across_1000_pairs_all_dims_and_backends() {
        use osprey_core::math::SplitMix64;
        use osprey_core::rabitq::TransposedQuery4Bit;

        let mut backends = vec![SimdEngine::with_level(SimdLevel::Scalar)];
        for level in [SimdLevel::Avx2, SimdLevel::Avx512, SimdLevel::Neon] {
            if let Some(eng) = SimdEngine::try_with_level(level) {
                backends.push(eng);
            }
        }

        let pairs = if cfg!(miri) { 4usize } else { 1000usize };
        let mut rng = SplitMix64::new(0x4F53_5052_0088);
        for dim in [64usize, 128, 256, 512, 1024, 2048] {
            let n_words = dim / 64;
            let mut q_int = vec![0_i8; dim];
            let mut db_bits = vec![0_u64; n_words];

            for pair_idx in 0..pairs {
                for slot in &mut q_int {
                    // Uniformly sample int4 in [-7, +7] (and exercise full [-15, +15] periodically)
                    let r = rng.next_u64();
                    let val = if pair_idx % 10 == 0 {
                        i8::try_from((r % 31) as i64 - 15).unwrap_or(0)
                    } else {
                        i8::try_from((r % 15) as i64 - 7).unwrap_or(0)
                    };
                    *slot = val;
                }
                for w in &mut db_bits {
                    *w = rng.next_u64();
                }

                let tq = TransposedQuery4Bit::from_quantized_i8(&q_int, 1.0, Vec::new()).unwrap();

                let mut naive_dot = 0_i32;
                for (i, &qi) in q_int.iter().enumerate() {
                    let bit = (db_bits[i / 64] >> (i % 64)) & 1;
                    let sigma = if bit == 1 { 1_i32 } else { -1_i32 };
                    naive_dot += i32::from(qi) * sigma;
                }

                for engine in &backends {
                    let got = engine.rabitq_4bit_dot_u64(
                        &tq.planes,
                        &tq.sign_mask,
                        tq.query_abs_sum,
                        &db_bits,
                    );
                    assert_eq!(
                        got,
                        naive_dot,
                        "mismatch at dim={dim}, pair={pair_idx}, level={:?}",
                        engine.active_level()
                    );
                }
            }
        }
    }

    #[test]
    fn simd_backends_tie_breaking_prefers_lowest_index() {
        let backends = supported_backends();

        for dim in [64usize, 128, 256, 512, 1024, 2048] {
            let words_per_vec = dim / 64;
            let query = vec![0_u64; words_per_vec];

            // Test ties both inside the 4x unrolled block (indices 2 & 3) and in the % 4 tail (indices 5 & 6).
            let k = 7usize;
            let mut centroids = vec![u64::MAX; k * words_per_vec];
            for c in [2usize, 3, 5, 6] {
                let start = c * words_per_vec;
                centroids[start..start + words_per_vec].fill(0_u64);
                centroids[start] = 0b1111; // exact Hamming distance = 4 for all tied centroids
            }

            for engine in &backends {
                let got =
                    engine.batch_nearest_binary_centroid(&query, &centroids, words_per_vec, k);
                assert_eq!(
                    got,
                    (2, 4),
                    "tie-break failed for dim={dim} level={:?}",
                    engine.active_level()
                );
            }

            // Also test edge cases (zero centroids / zero words).
            for engine in &backends {
                assert_eq!(
                    engine.batch_nearest_binary_centroid(&query, &centroids, 0, k),
                    (0, u32::MAX)
                );
                assert_eq!(
                    engine.batch_nearest_binary_centroid(&query, &centroids, words_per_vec, 0),
                    (0, u32::MAX)
                );
            }
        }
    }

    #[test]
    fn osi_index_search_candidates_end_to_end_with_simd_engine() {
        use osprey_core::math::{prepare_and_pad_vector, SplitMix64};
        use osprey_core::{AlignedBuffer, BinaryKMeansResult, OsiIndex, VectorLocation};
        use roaring::RoaringBitmap;

        let engine = SimdEngine::new_auto();
        let n = 128usize;
        let orig_dim = 128usize;
        let k = 8usize;
        let fwht_seed = 0x4F53_5052_1234;

        let mut rng = SplitMix64::new(0x9999);
        let mut vectors = Vec::with_capacity(n * orig_dim);
        for _ in 0..n {
            let mut sq = 0.0_f64;
            let start = vectors.len();
            for _ in 0..orig_dim {
                #[allow(clippy::cast_precision_loss)]
                let v = ((rng.next_u64() >> 11) as f64) / ((1_u64 << 53) as f64) * 2.0 - 1.0;
                sq += v * v;
                #[allow(clippy::cast_possible_truncation)]
                vectors.push(v as f32);
            }
            let inv = 1.0_f64 / sq.sqrt();
            for v in &mut vectors[start..] {
                #[allow(clippy::cast_possible_truncation)]
                {
                    *v = (f64::from(*v) * inv) as f32;
                }
            }
        }

        let signs = osprey_core::math::RademacherSigns::from_seed(orig_dim, fwht_seed).unwrap();
        let mut centroids_f16 = AlignedBuffer::<half::f16>::new(k * orig_dim).unwrap();
        let mut scratch = vec![0.0_f32; orig_dim];
        for c in 0..k {
            prepare_and_pad_vector(&vectors[c * orig_dim..(c + 1) * orig_dim], &mut scratch)
                .unwrap();
            SimdBackend::fwht_rotate_in_place(&engine, &mut scratch, signs.xor_masks()).unwrap();
            for (d, &val) in scratch.iter().enumerate() {
                centroids_f16[c * orig_dim + d] = half::f16::from_f32(val);
            }
        }

        let kmeans = BinaryKMeansResult {
            centroids_f16,
            binary_centroids: vec![0_u64; k * (orig_dim / 64)],
            assignments: (0..n).map(|i| u32::try_from(i % k).unwrap()).collect(),
            final_hamming_inertia: 0,
            iterations_run: 1,
        };
        let doc_ids: Vec<u64> = (0..n).map(|i| u64::try_from(i).unwrap()).collect();
        let locations = vec![VectorLocation::default(); n];

        let index = OsiIndex::build(
            &vectors,
            &doc_ids,
            &locations,
            &kmeans,
            orig_dim,
            fwht_seed,
            RoaringBitmap::new(),
            &engine,
        )
        .unwrap();

        let q = &vectors[37 * orig_dim..38 * orig_dim];
        let results = index.search_candidates(q, k, 10, &engine).unwrap();
        assert_eq!(results.len(), 10);
        assert_eq!(results[0].doc_id, 37);
    }

    #[test]
    fn lutrank_all_simd_backends_match_scalar_and_prevent_saturation() {
        use osprey_core::{OsrDocumentTokens, OsrStore, QuantizedQueryLut, DEFAULT_LUTRANK_GAMMA};

        let backends = supported_backends();
        let scalar = SimdEngine::with_level(SimdLevel::Scalar);
        let seed = 0x4F53_5052_0107_u64;
        let dim = 128usize;
        let mut rng = SplitMix64::new(seed);

        let mut make_unit_tokens = |count: usize| -> Vec<f32> {
            let mut out = Vec::with_capacity(count * dim);
            for _ in 0..count {
                let mut sq = 0.0_f64;
                let start = out.len();
                for _ in 0..dim {
                    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
                    let v = (((rng.next_u64() >> 11) as f64) / ((1_u64 << 53) as f64) * 2.0 - 1.0)
                        as f32;
                    sq += f64::from(v) * f64::from(v);
                    out.push(v);
                }
                let inv = 1.0_f64 / sq.sqrt();
                for v in &mut out[start..] {
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        *v = (f64::from(*v) * inv) as f32;
                    }
                }
            }
            out
        };

        // Cover tile-edge token counts: 1, 15, 16, 17, 31, 32, 33, 64, 79 (non-multiple of 16/32/64)
        let doc_counts: &[usize] = if cfg!(miri) {
            &[1, 3]
        } else {
            &[1, 15, 16, 17, 31, 32, 33, 64, 79]
        };
        let mut doc_bufs = Vec::with_capacity(doc_counts.len());
        for &cnt in doc_counts {
            doc_bufs.push(make_unit_tokens(cnt));
        }
        let docs: Vec<OsrDocumentTokens<'_>> = doc_counts
            .iter()
            .enumerate()
            .map(|(i, &cnt)| OsrDocumentTokens {
                doc_id: u64::try_from(i + 1).unwrap(),
                tokens: &doc_bufs[i],
                num_tokens: cnt,
            })
            .collect();

        let store = OsrStore::build(&docs, dim, DEFAULT_LUTRANK_GAMMA, seed, &scalar).unwrap();
        let q_count = if cfg!(miri) { 2 } else { 16 };
        let q_tokens = make_unit_tokens(q_count);
        let q_lut = QuantizedQueryLut::synthesize(
            &q_tokens,
            q_count,
            dim,
            DEFAULT_LUTRANK_GAMMA,
            seed,
            &scalar,
        )
        .unwrap();

        for (doc_idx, &doc_tok_count) in doc_counts.iter().enumerate() {
            let expected = store.score_document_maxsim(doc_idx, &q_lut, &scalar);
            let doc_range = store.doc_table()[doc_idx];
            let doc_bytes = store.token_bytes(&doc_range).unwrap();

            // Check single-token sim equivalence on first token
            let (ql1, ql2) = q_lut.query_luts(0).unwrap();
            let expected_tok = scalar.lutrank_token_sim_i32(
                &doc_bytes[..32],
                &doc_bytes[32..64],
                ql1,
                ql2,
                q_lut.scale1,
                q_lut.scale2,
            );

            for engine in &backends {
                let got_tok = engine.lutrank_token_sim_i32(
                    &doc_bytes[..32],
                    &doc_bytes[32..64],
                    ql1,
                    ql2,
                    q_lut.scale1,
                    q_lut.scale2,
                );
                assert!(
                    (got_tok - expected_tok).abs() < 1e-5,
                    "token_sim mismatch doc={doc_idx} level={:?}: got={got_tok} expected={expected_tok}",
                    engine.active_level()
                );

                let got_doc = store.score_document_maxsim(doc_idx, &q_lut, engine);
                assert!(
                    (got_doc - expected).abs() < 1e-5,
                    "doc_maxsim mismatch doc={doc_idx} (tokens={doc_tok_count}) level={:?}: got={got_doc} expected={expected}",
                    engine.active_level()
                );
            }
        }

        // Saturation stress test: all LUT entries = +127 or -127 across 64 subspaces (sum = ±8128)
        let doc_l1 = [0x00_u8; 32];
        let doc_l2 = [0x11_u8; 32];
        let mut max_lut1 = [0_i8; 64 * 16];
        let mut max_lut2 = [0_i8; 64 * 16];
        for s in 0..64 {
            max_lut1[s * 16] = 127;
            max_lut2[s * 16 + 1] = -127;
        }
        let mut doc_64_bytes = Vec::with_capacity(65 * 64);
        for _ in 0..65 {
            doc_64_bytes.extend_from_slice(&doc_l1);
            doc_64_bytes.extend_from_slice(&doc_l2);
        }
        let expected_sat = 8128.0_f32 * 0.5 + (-8128.0_f32) * 0.25;
        for engine in &backends {
            let tok_sat =
                engine.lutrank_token_sim_i32(&doc_l1, &doc_l2, &max_lut1, &max_lut2, 0.5, 0.25);
            assert!(
                (tok_sat - expected_sat).abs() < 1e-5,
                "saturation failed on token_sim level={:?}: got={tok_sat}",
                engine.active_level()
            );
            let doc_sat = engine.lutrank_doc_maxsim(
                &doc_64_bytes,
                65,
                64,
                &max_lut1,
                &max_lut2,
                1,
                0.5,
                0.25,
            );
            assert!(
                (doc_sat - expected_sat).abs() < 1e-5,
                "saturation failed on doc_maxsim level={:?}: got={doc_sat}",
                engine.active_level()
            );
        }
    }
}
