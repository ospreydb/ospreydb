//! `LUT-Rank` multi-vector late-interaction engine (`arXiv:2505.13164v1`).
//!
//! Implements deterministic 2-layer 2D Nested Lattice codebooks ($\gamma = 0.28$),
//! post-FWHT token quantization **without** per-subspace normalization, SIMD-interleaved
//! 4-bit nibble packing (64 bytes = 1 cache line per 128-dim token), online `int8` query
//! LUT synthesis, and zero-copy `.osr` serialization/reading via `memmap2::Mmap` and
//! 64-byte-aligned slices.

use std::fs::File;
use std::mem::{align_of, size_of};
use std::path::Path;

use bytemuck::{Pod, Zeroable};
use memmap2::Mmap;
#[cfg(not(miri))]
use rayon::prelude::*;
use xxhash_rust::xxh64::xxh64;

use crate::aligned::{ensure_aligned_64, AlignedBuffer, CACHE_LINE_ALIGN};
use crate::error::{OspreyError, OspreyResult};
use crate::format::checked_align_up_64;
use crate::format::osr::{
    cast_doc_token_ranges, DocTokenRange, OsrHeader, OSR_CANONICAL_GAMMA, OSR_CANONICAL_NUM_LAYERS,
    OSR_CANONICAL_SUBSPACE_DIM, OSR_CANONICAL_TOKEN_DIM,
};
use crate::math::{prepare_and_pad_vector, RademacherSigns, SplitMix64};
use crate::rabitq::{RabitqFwhtKernel, ScalarRabitqKernel};
use crate::traits::RerankStoreReader;

/// Canonical 1D 4-level Lloyd-Max-style scalar alphabet for Gaussianized FWHT coordinates.
pub const LATTICE_1D_LEVELS: [f32; 4] = [-1.75, -0.55, 0.55, 1.75];

/// Canonical Layer-2 residual contraction factor ($\gamma = 0.28$).
pub const DEFAULT_LUTRANK_GAMMA: f32 = OSR_CANONICAL_GAMMA;

/// Canonical token dimensionality for `.osr` nibble-packed late-interaction tensors.
pub const LUTRANK_TOKEN_DIM: usize = OSR_CANONICAL_TOKEN_DIM as usize;

/// Canonical number of 2D subspaces per 128-dim token ($128 / 2 = 64$).
pub const LUTRANK_SUBSPACES: usize = LUTRANK_TOKEN_DIM / 2;

/// Canonical byte size of a single nibble-packed 2-layer `.osr` token (`32B L1 + 32B L2 = 64B`).
pub const OSR_TOKEN_BYTES: usize = 64;

/// 16-entry 2D lattice codebook for a single nested layer.
pub type LatticeCodebook2D = [[f32; 2]; 16];

/// Deterministic 2-layer 2D Nested Lattice codebooks scaled by $\sigma = 1/\sqrt{D}$
/// (Layer 1) and $\sigma \cdot \gamma$ (Layer 2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NestedLatticeCodebooks {
    /// Token dimensionality $D$ used to derive $\sigma = 1/\sqrt{D}$.
    pub token_dim: usize,
    /// Residual contraction factor $\gamma$ (default `0.28`).
    pub gamma: f32,
    /// Global FWHT-equalized per-coordinate standard deviation $\sigma = 1/\sqrt{D}$.
    pub sigma: f32,
    /// Layer 1 coarse 2D codebook (`16` centroids scaled by $\sigma$).
    pub layer1: LatticeCodebook2D,
    /// Layer 2 residual 2D codebook (`16` centroids scaled by $\sigma \cdot \gamma$).
    pub layer2: LatticeCodebook2D,
}

impl NestedLatticeCodebooks {
    /// Builds deterministic Layer 1 and Layer 2 codebooks for `token_dim` and `gamma`.
    ///
    /// Uses the canonical Cartesian product index $c = 4 \cdot i + j \in \{0..15\}$
    /// ($i \in 0..4, j \in 0..4$) matching `reference/poc_reference.py` and
    /// `reference/golden/lutrank_golden.bin`:
    /// `x = LATTICE_1D_LEVELS[j] * sigma`, `y = LATTICE_1D_LEVELS[i] * sigma`.
    pub fn new(token_dim: usize, gamma: f32) -> OspreyResult<Self> {
        if token_dim == 0 || !token_dim.is_multiple_of(2) {
            return Err(OspreyError::InvalidDimension { dim: token_dim });
        }
        if !gamma.is_finite() || gamma <= 0.0 {
            return Err(OspreyError::NonFiniteEmbedding { index: 0 });
        }

        let dim_u32 = u32::try_from(token_dim)
            .map_err(|_| OspreyError::InvalidDimension { dim: token_dim })?;
        let sigma_f64 = 1.0_f64 / f64::from(dim_u32).sqrt();
        let gamma_f64 = f64::from(gamma);

        let mut layer1 = [[0.0_f32; 2]; 16];
        let mut layer2 = [[0.0_f32; 2]; 16];

        for (i, &lvl_y) in LATTICE_1D_LEVELS.iter().enumerate() {
            let y1 = f64::from(lvl_y) * sigma_f64;
            for (j, &lvl_x) in LATTICE_1D_LEVELS.iter().enumerate() {
                let x1 = f64::from(lvl_x) * sigma_f64;
                let idx = 4 * i + j;
                #[allow(clippy::cast_possible_truncation)]
                {
                    layer1[idx] = [x1 as f32, y1 as f32];
                    layer2[idx] = [(x1 * gamma_f64) as f32, (y1 * gamma_f64) as f32];
                }
            }
        }

        #[allow(clippy::cast_possible_truncation)]
        let sigma = sigma_f64 as f32;

        Ok(Self {
            token_dim,
            gamma,
            sigma,
            layer1,
            layer2,
        })
    }

    /// Builds canonical `D = 128`, $\gamma = 0.28$ codebooks.
    pub fn canonical() -> OspreyResult<Self> {
        Self::new(LUTRANK_TOKEN_DIM, DEFAULT_LUTRANK_GAMMA)
    }
}

/// Builds the canonical Layer 1 (`scale = sigma`) and Layer 2 (`scale = sigma * gamma`)
/// 2D nested lattice codebooks.
pub fn build_nested_lattice_codebooks(
    token_dim: usize,
    gamma: f32,
) -> OspreyResult<(LatticeCodebook2D, LatticeCodebook2D)> {
    let cb = NestedLatticeCodebooks::new(token_dim, gamma)?;
    Ok((cb.layer1, cb.layer2))
}

/// Physical representation of one 128-dim document token in `.osr` (`64 bytes = 1 cache line`).
///
/// Layout:
/// - `layer1[k]` (`0..32`): packs subspaces `2*k` (low nibble) and `2*k + 1` (high nibble).
/// - `layer2[k]` (`32..64`): packs residual subspaces `2*k` (low nibble) and `2*k + 1` (high nibble).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct OsrTokenCode {
    /// Bytes `0..32`: Layer 1 — 64 subspaces packed as 4-bit nibbles.
    pub layer1: [u8; 32],
    /// Bytes `32..64`: Layer 2 ($\gamma = 0.28$) — 64 residual subspaces packed as 4-bit nibbles.
    pub layer2: [u8; 32],
}

const _: () = {
    assert!(size_of::<OsrTokenCode>() == OSR_TOKEN_BYTES);
    assert!(align_of::<OsrTokenCode>() == CACHE_LINE_ALIGN);
};

impl OsrTokenCode {
    /// Packs 64 Layer-1 and 64 Layer-2 4-bit centroid indices into a 64-byte token record.
    pub fn from_codes(c1: &[u8; LUTRANK_SUBSPACES], c2: &[u8; LUTRANK_SUBSPACES]) -> Self {
        let mut layer1 = [0_u8; 32];
        let mut layer2 = [0_u8; 32];
        for s in 0..32 {
            layer1[s] = (c1[2 * s] & 0x0F) | ((c1[2 * s + 1] & 0x0F) << 4);
            layer2[s] = (c2[2 * s] & 0x0F) | ((c2[2 * s + 1] & 0x0F) << 4);
        }
        Self { layer1, layer2 }
    }

    /// Unpacks the 64 Layer-1 centroid indices into `out` (zero heap allocation).
    #[inline]
    pub fn unpack_layer1(&self, out: &mut [u8; LUTRANK_SUBSPACES]) {
        for (s, &byte) in self.layer1.iter().enumerate() {
            out[2 * s] = byte & 0x0F;
            out[2 * s + 1] = (byte >> 4) & 0x0F;
        }
    }

    /// Unpacks the 64 Layer-2 centroid indices into `out` (zero heap allocation).
    #[inline]
    pub fn unpack_layer2(&self, out: &mut [u8; LUTRANK_SUBSPACES]) {
        for (s, &byte) in self.layer2.iter().enumerate() {
            out[2 * s] = byte & 0x0F;
            out[2 * s + 1] = (byte >> 4) & 0x0F;
        }
    }
}

/// Quantized 16-entry `i8` Lookup Tables for one 128-dim query token (`64 subspaces x 16 entries`).
#[repr(C, align(64))]
#[derive(Clone, Debug, PartialEq)]
pub struct QueryTokenLuts {
    /// Layer 1 quantized inner-product LUT (`64 * 16 = 1024 bytes`).
    pub lut1: [[i8; 16]; LUTRANK_SUBSPACES],
    /// Layer 2 quantized inner-product LUT (`64 * 16 = 1024 bytes`).
    pub lut2: [[i8; 16]; LUTRANK_SUBSPACES],
    /// Symmetric dequantization scale for `lut1` (`max1 / 127.0`).
    pub scale1: f32,
    /// Symmetric dequantization scale for `lut2` (`max2 / 127.0`).
    pub scale2: f32,
}

impl Default for QueryTokenLuts {
    fn default() -> Self {
        Self {
            lut1: [[0; 16]; LUTRANK_SUBSPACES],
            lut2: [[0; 16]; LUTRANK_SUBSPACES],
            scale1: 1.0,
            scale2: 1.0,
        }
    }
}

/// Packs an even-length slice of 4-bit centroid codes (`c[m] in 0..16`) into SIMD-interleaved
/// nibble bytes: `out[s] = (codes[2*s] & 0x0F) | ((codes[2*s + 1] & 0x0F) << 4)`.
pub fn pack_nibbles(codes: &[u8], out_packed: &mut [u8]) -> OspreyResult<()> {
    if !codes.len().is_multiple_of(2) {
        return Err(OspreyError::InvalidDimension { dim: codes.len() });
    }
    let expected_bytes = codes.len() / 2;
    if out_packed.len() < expected_bytes {
        return Err(OspreyError::BufferTooSmall {
            needed: expected_bytes,
            actual: out_packed.len(),
        });
    }
    if out_packed.len() != expected_bytes {
        return Err(OspreyError::InvalidDimension {
            dim: out_packed.len(),
        });
    }

    for (s, dst) in out_packed.iter_mut().enumerate() {
        let even = codes[2 * s] & 0x0F;
        let odd = codes[2 * s + 1] & 0x0F;
        *dst = even | (odd << 4);
    }
    Ok(())
}

/// Unpacks SIMD-interleaved nibble bytes back into 4-bit centroid indices (`0..15`).
pub fn unpack_nibbles(packed: &[u8], out_codes: &mut [u8]) -> OspreyResult<()> {
    let expected_codes = packed
        .len()
        .checked_mul(2)
        .ok_or(OspreyError::InvalidDimension { dim: packed.len() })?;
    if out_codes.len() < expected_codes {
        return Err(OspreyError::BufferTooSmall {
            needed: expected_codes,
            actual: out_codes.len(),
        });
    }
    if out_codes.len() != expected_codes {
        return Err(OspreyError::InvalidDimension {
            dim: out_codes.len(),
        });
    }

    for (s, &byte) in packed.iter().enumerate() {
        out_codes[2 * s] = byte & 0x0F;
        out_codes[2 * s + 1] = (byte >> 4) & 0x0F;
    }
    Ok(())
}

/// Encodes an already FWHT-rotated token `t_rot` into Layer 1 (`out_c1`) and Layer 2 (`out_c2`)
/// 4-bit centroid indices **without** per-subspace normalization.
///
/// Zero heap allocation: all computations stay in registers/stack.
pub fn encode_prerotated_token_codes(
    t_rot: &[f32],
    codebooks: &NestedLatticeCodebooks,
    out_c1: &mut [u8],
    out_c2: &mut [u8],
) -> OspreyResult<()> {
    if t_rot.len() != codebooks.token_dim {
        return Err(OspreyError::InvalidDimension { dim: t_rot.len() });
    }
    let n_subspaces = codebooks.token_dim / 2;
    if out_c1.len() != n_subspaces {
        return Err(OspreyError::BufferTooSmall {
            needed: n_subspaces,
            actual: out_c1.len(),
        });
    }
    if out_c2.len() != n_subspaces {
        return Err(OspreyError::BufferTooSmall {
            needed: n_subspaces,
            actual: out_c2.len(),
        });
    }

    for (index, &val) in t_rot.iter().enumerate() {
        if !val.is_finite() {
            return Err(OspreyError::NonFiniteEmbedding { index });
        }
    }

    // INVARIANT: Never normalize 2D subspaces individually; FWHT already equalized
    // per-coordinate variance to sigma^2 = 1/D.
    for m in 0..n_subspaces {
        let x0 = t_rot[2 * m];
        let x1 = t_rot[2 * m + 1];

        // Layer 1 Euclidean argmin over 16 centroids.
        let mut best_c1 = 0_usize;
        let mut best_dist1 = f32::INFINITY;
        for (k, &[cx, cy]) in codebooks.layer1.iter().enumerate() {
            let dx = x0 - cx;
            let dy = x1 - cy;
            let dist = dx * dx + dy * dy;
            if dist < best_dist1 {
                best_dist1 = dist;
                best_c1 = k;
            }
        }

        // Residual r = x_sub - cb1[c1] -> Layer 2 Euclidean argmin over 16 centroids.
        let r0 = x0 - codebooks.layer1[best_c1][0];
        let r1 = x1 - codebooks.layer1[best_c1][1];
        let mut best_c2 = 0_usize;
        let mut best_dist2 = f32::INFINITY;
        for (k, &[cx, cy]) in codebooks.layer2.iter().enumerate() {
            let dx = r0 - cx;
            let dy = r1 - cy;
            let dist = dx * dx + dy * dy;
            if dist < best_dist2 {
                best_dist2 = dist;
                best_c2 = k;
            }
        }

        #[allow(clippy::cast_possible_truncation)]
        {
            out_c1[m] = best_c1 as u8;
            out_c2[m] = best_c2 as u8;
        }
    }

    Ok(())
}

/// Encodes an already FWHT-rotated canonical 128-dim token directly into a 64-byte [`OsrTokenCode`].
pub fn encode_prerotated_token(
    t_rot: &[f32],
    codebooks: &NestedLatticeCodebooks,
) -> OspreyResult<OsrTokenCode> {
    if codebooks.token_dim != LUTRANK_TOKEN_DIM {
        return Err(OspreyError::InvalidDimension {
            dim: codebooks.token_dim,
        });
    }
    let mut c1 = [0_u8; LUTRANK_SUBSPACES];
    let mut c2 = [0_u8; LUTRANK_SUBSPACES];
    encode_prerotated_token_codes(t_rot, codebooks, &mut c1, &mut c2)?;
    Ok(OsrTokenCode::from_codes(&c1, &c2))
}

/// Sanitizes, L2-normalizes, rotates via FWHT (`fwht_seed`), and encodes a raw 128-dim token
/// into a 64-byte nibble-packed [`OsrTokenCode`] with zero heap allocations.
pub fn encode_token<K: RabitqFwhtKernel + ?Sized>(
    token_raw: &[f32],
    signs: &RademacherSigns,
    codebooks: &NestedLatticeCodebooks,
    simd: &K,
    scratch_rot: &mut [f32],
) -> OspreyResult<OsrTokenCode> {
    if codebooks.token_dim != LUTRANK_TOKEN_DIM || token_raw.len() != LUTRANK_TOKEN_DIM {
        return Err(OspreyError::InvalidDimension {
            dim: token_raw.len(),
        });
    }
    if signs.padded_dim() != LUTRANK_TOKEN_DIM {
        return Err(OspreyError::InvalidDimension {
            dim: signs.padded_dim(),
        });
    }
    if scratch_rot.len() != LUTRANK_TOKEN_DIM {
        return Err(OspreyError::BufferTooSmall {
            needed: LUTRANK_TOKEN_DIM,
            actual: scratch_rot.len(),
        });
    }

    prepare_and_pad_vector(token_raw, scratch_rot)?;
    simd.fwht_rotate_in_place(scratch_rot, signs.xor_masks())?;
    encode_prerotated_token(scratch_rot, codebooks)
}

/// Synthesizes 16-entry `int8` Lookup Tables for an already FWHT-rotated 128-dim query token
/// using only 8 1D multiplications per subspace (`< 8 us` budget, zero heap allocations).
pub fn synthesize_query_token_luts(
    q_rot: &[f32],
    codebooks: &NestedLatticeCodebooks,
    out: &mut QueryTokenLuts,
) -> OspreyResult<()> {
    if codebooks.token_dim != LUTRANK_TOKEN_DIM || q_rot.len() != LUTRANK_TOKEN_DIM {
        return Err(OspreyError::InvalidDimension { dim: q_rot.len() });
    }
    for (index, &val) in q_rot.iter().enumerate() {
        if !val.is_finite() {
            return Err(OspreyError::NonFiniteEmbedding { index });
        }
    }

    let mut dots1 = [[0.0_f32; 16]; LUTRANK_SUBSPACES];
    let mut dots2 = [[0.0_f32; 16]; LUTRANK_SUBSPACES];
    let mut max1 = 0.0_f32;
    let mut max2 = 0.0_f32;

    let l1_levels = [
        LATTICE_1D_LEVELS[0] * codebooks.sigma,
        LATTICE_1D_LEVELS[1] * codebooks.sigma,
        LATTICE_1D_LEVELS[2] * codebooks.sigma,
        LATTICE_1D_LEVELS[3] * codebooks.sigma,
    ];
    let gamma = codebooks.gamma;

    for s in 0..LUTRANK_SUBSPACES {
        let qx = q_rot[2 * s];
        let qy = q_rot[2 * s + 1];

        let px = [
            qx * l1_levels[0],
            qx * l1_levels[1],
            qx * l1_levels[2],
            qx * l1_levels[3],
        ];
        let py = [
            qy * l1_levels[0],
            qy * l1_levels[1],
            qy * l1_levels[2],
            qy * l1_levels[3],
        ];

        for (i, &py_val) in py.iter().enumerate() {
            for (j, &px_val) in px.iter().enumerate() {
                let c = 4 * i + j;
                let d1 = px_val + py_val;
                let d2 = d1 * gamma;
                dots1[s][c] = d1;
                dots2[s][c] = d2;
                let a1 = d1.abs();
                let a2 = d2.abs();
                if a1 > max1 {
                    max1 = a1;
                }
                if a2 > max2 {
                    max2 = a2;
                }
            }
        }
    }

    let scale1 = if max1 > 1e-12 { max1 / 127.0 } else { 1.0 };
    let scale2 = if max2 > 1e-12 { max2 / 127.0 } else { 1.0 };
    out.scale1 = scale1;
    out.scale2 = scale2;

    let inv1 = 1.0_f32 / scale1;
    let inv2 = 1.0_f32 / scale2;
    for s in 0..LUTRANK_SUBSPACES {
        for c in 0..16 {
            #[allow(clippy::cast_possible_truncation)]
            {
                out.lut1[s][c] = (dots1[s][c] * inv1).round_ties_even().clamp(-127.0, 127.0) as i8;
                out.lut2[s][c] = (dots2[s][c] * inv2).round_ties_even().clamp(-127.0, 127.0) as i8;
            }
        }
    }

    Ok(())
}

/// Portable scalar reference scoring of one nibble-packed [`OsrTokenCode`] against [`QueryTokenLuts`].
#[inline]
pub fn score_token_with_luts_scalar(token: &OsrTokenCode, q_luts: &QueryTokenLuts) -> f32 {
    let mut acc1 = 0_i32;
    let mut acc2 = 0_i32;

    for s in 0..32 {
        let b1 = token.layer1[s];
        let b2 = token.layer2[s];
        let even_sub = 2 * s;
        let odd_sub = even_sub + 1;

        let c1_even = usize::from(b1 & 0x0F);
        let c1_odd = usize::from((b1 >> 4) & 0x0F);
        let c2_even = usize::from(b2 & 0x0F);
        let c2_odd = usize::from((b2 >> 4) & 0x0F);

        acc1 += i32::from(q_luts.lut1[even_sub][c1_even]) + i32::from(q_luts.lut1[odd_sub][c1_odd]);
        acc2 += i32::from(q_luts.lut2[even_sub][c2_even]) + i32::from(q_luts.lut2[odd_sub][c2_odd]);
    }

    #[allow(clippy::cast_precision_loss)]
    {
        q_luts.scale1 * (acc1 as f32) + q_luts.scale2 * (acc2 as f32)
    }
}

/// Hardware-accelerated or scalar `LUT-Rank` token & `MaxSim` kernel hook.
pub trait LutRankKernel: RabitqFwhtKernel {
    /// Hints to the CPU cache hierarchy to prefetch the 64-byte cache line starting at `bytes`
    /// into L1 data cache (`_MM_HINT_T0`) ahead of `.osr` lattice evaluation.
    #[inline]
    fn prefetch_l1(&self, _bytes: &[u8]) {}

    /// Evaluates one nibble-packed `.osr` token (`L1` + `L2`) against one query token's
    /// `[N_sub, 16]` `int8` lookup tables with zero floating-point multiplications inside
    /// the subspace loop.
    fn lutrank_token_sim_i32(
        &self,
        doc_l1_packed: &[u8],
        doc_l2_packed: &[u8],
        q_lut1_i8: &[i8],
        q_lut2_i8: &[i8],
        scale1: f32,
        scale2: f32,
    ) -> f32;

    /// Evaluates `MaxSim(Q, D) = sum_{q=0}^{Q-1} max_{t=0}^{L_d-1} sim(q, t)` over a
    /// document's contiguous 64-byte-aligned `.osr` token byte slice.
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

impl LutRankKernel for ScalarRabitqKernel {
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
        let mut acc1 = 0_i32;
        let mut acc2 = 0_i32;

        for s in 0..pairs {
            let b1 = doc_l1_packed[s];
            let b2 = doc_l2_packed[s];
            let c1_even = usize::from(b1 & 0x0F);
            let c1_odd = usize::from((b1 >> 4) & 0x0F);
            let c2_even = usize::from(b2 & 0x0F);
            let c2_odd = usize::from((b2 >> 4) & 0x0F);

            let base_even = (2 * s) * 16;
            let base_odd = (2 * s + 1) * 16;
            if let (Some(&v1e), Some(&v1o), Some(&v2e), Some(&v2o)) = (
                q_lut1_i8.get(base_even + c1_even),
                q_lut1_i8.get(base_odd + c1_odd),
                q_lut2_i8.get(base_even + c2_even),
                q_lut2_i8.get(base_odd + c2_odd),
            ) {
                acc1 += i32::from(v1e) + i32::from(v1o);
                acc2 += i32::from(v2e) + i32::from(v2o);
            }
        }

        #[allow(clippy::cast_precision_loss)]
        {
            (acc1 as f32) * scale1 + (acc2 as f32) * scale2
        }
    }
}

/// Online-synthesized, 64-byte-aligned `i8` Lookup Tables for an entire multi-token
/// ColBERT query (`[Q, N_sub, 16]` layout, `< 8 us` synthesis budget).
pub struct QuantizedQueryLut {
    /// Number of query tokens ($Q$).
    pub num_query_tokens: usize,
    /// Number of 2D subspaces per token ($N_{\text{sub}} = D / 2$, `64` for $D = 128$).
    pub num_subspaces: usize,
    /// Contiguous Layer-1 LUT `[Q, N_sub, 16]` aligned to 64 bytes.
    pub lut1_i8: AlignedBuffer<i8>,
    /// Contiguous Layer-2 LUT `[Q, N_sub, 16]` aligned to 64 bytes.
    pub lut2_i8: AlignedBuffer<i8>,
    /// Symmetric `i8` quantization scale for Layer 1 (`max_abs1 / 127.0`).
    pub scale1: f32,
    /// Symmetric `i8` quantization scale for Layer 2 (`scale1 * gamma`).
    pub scale2: f32,
}

impl std::fmt::Debug for QuantizedQueryLut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantizedQueryLut")
            .field("num_query_tokens", &self.num_query_tokens)
            .field("num_subspaces", &self.num_subspaces)
            .field("lut1_len", &self.lut1_i8.len())
            .field("lut2_len", &self.lut2_i8.len())
            .field("scale1", &self.scale1)
            .field("scale2", &self.scale2)
            .finish()
    }
}

impl QuantizedQueryLut {
    /// Rotates `query_tokens` (`[num_query_tokens, token_dim]`) via FWHT and synthesizes
    /// contiguous `[Q, N_sub, 16]` `i8` lookup tables.
    ///
    /// Uses a stack-allocated rotation buffer for standard query sizes (`Q * D <= 8192`,
    /// e.g. $Q \le 64, D = 128$) to avoid extra intermediate heap allocations.
    pub fn synthesize<K: RabitqFwhtKernel + ?Sized>(
        query_tokens: &[f32],
        num_query_tokens: usize,
        token_dim: usize,
        gamma: f32,
        fwht_seed: u64,
        simd: &K,
    ) -> OspreyResult<Self> {
        if token_dim < 64 || !token_dim.is_power_of_two() || !token_dim.is_multiple_of(2) {
            return Err(OspreyError::InvalidDimension { dim: token_dim });
        }
        let codebooks = NestedLatticeCodebooks::new(token_dim, gamma)?;
        let expected_floats =
            num_query_tokens
                .checked_mul(token_dim)
                .ok_or(OspreyError::InvalidDimension {
                    dim: num_query_tokens,
                })?;
        if query_tokens.len() < expected_floats {
            return Err(OspreyError::BufferTooSmall {
                needed: expected_floats * size_of::<f32>(),
                actual: std::mem::size_of_val(query_tokens),
            });
        }
        if query_tokens.len() != expected_floats {
            return Err(OspreyError::InvalidDimension {
                dim: query_tokens.len(),
            });
        }

        let num_subspaces = token_dim / 2;
        if num_query_tokens == 0 {
            return Ok(Self {
                num_query_tokens: 0,
                num_subspaces,
                lut1_i8: AlignedBuffer::new(0)?,
                lut2_i8: AlignedBuffer::new(0)?,
                scale1: 1.0,
                scale2: gamma,
            });
        }

        let mut stack_masks = [0_u32; 256];
        let owned_signs;
        let xor_masks: &[u32] = if token_dim <= 256 {
            let mut rng = SplitMix64::new(fwht_seed);
            let n_words = token_dim / 64;
            for w in 0..n_words {
                let word = rng.next_u64();
                let base = w * 64;
                for bit in 0..64 {
                    stack_masks[base + bit] = u32::from(((word >> bit) & 1) == 0) << 31;
                }
            }
            &stack_masks[..token_dim]
        } else {
            owned_signs = RademacherSigns::from_seed(token_dim, fwht_seed)?;
            owned_signs.xor_masks()
        };

        const SMALL_STACK_CAP: usize = 16 * 128;
        const LARGE_STACK_CAP: usize = 64 * 128;
        if expected_floats <= SMALL_STACK_CAP {
            let mut stack_rot = [0.0_f32; SMALL_STACK_CAP];
            let rot_slice = &mut stack_rot[..expected_floats];
            for q in 0..num_query_tokens {
                let raw = &query_tokens[q * token_dim..(q + 1) * token_dim];
                let dst = &mut rot_slice[q * token_dim..(q + 1) * token_dim];
                prepare_and_pad_vector(raw, dst)?;
                simd.fwht_rotate_in_place(dst, xor_masks)?;
            }
            Self::quantize_from_rotated(rot_slice, num_query_tokens, &codebooks)
        } else if expected_floats <= LARGE_STACK_CAP {
            let mut stack_rot = [0.0_f32; LARGE_STACK_CAP];
            let rot_slice = &mut stack_rot[..expected_floats];
            for q in 0..num_query_tokens {
                let raw = &query_tokens[q * token_dim..(q + 1) * token_dim];
                let dst = &mut rot_slice[q * token_dim..(q + 1) * token_dim];
                prepare_and_pad_vector(raw, dst)?;
                simd.fwht_rotate_in_place(dst, xor_masks)?;
            }
            Self::quantize_from_rotated(rot_slice, num_query_tokens, &codebooks)
        } else {
            let mut heap_rot = vec![0.0_f32; expected_floats];
            for q in 0..num_query_tokens {
                let raw = &query_tokens[q * token_dim..(q + 1) * token_dim];
                let dst = &mut heap_rot[q * token_dim..(q + 1) * token_dim];
                prepare_and_pad_vector(raw, dst)?;
                simd.fwht_rotate_in_place(dst, xor_masks)?;
            }
            Self::quantize_from_rotated(&heap_rot, num_query_tokens, &codebooks)
        }
    }

    /// Synthesizes `[Q, N_sub, 16]` `i8` LUTs directly from already FWHT-rotated query tokens.
    pub fn synthesize_from_prerotated(
        query_tokens_rot: &[f32],
        num_query_tokens: usize,
        token_dim: usize,
        gamma: f32,
    ) -> OspreyResult<Self> {
        if token_dim < 2 || !token_dim.is_multiple_of(2) {
            return Err(OspreyError::InvalidDimension { dim: token_dim });
        }
        let codebooks = NestedLatticeCodebooks::new(token_dim, gamma)?;
        let expected_floats =
            num_query_tokens
                .checked_mul(token_dim)
                .ok_or(OspreyError::InvalidDimension {
                    dim: num_query_tokens,
                })?;
        if query_tokens_rot.len() < expected_floats {
            return Err(OspreyError::BufferTooSmall {
                needed: expected_floats * size_of::<f32>(),
                actual: std::mem::size_of_val(query_tokens_rot),
            });
        }
        if query_tokens_rot.len() != expected_floats {
            return Err(OspreyError::InvalidDimension {
                dim: query_tokens_rot.len(),
            });
        }
        for (index, &val) in query_tokens_rot.iter().enumerate() {
            if !val.is_finite() {
                return Err(OspreyError::NonFiniteEmbedding { index });
            }
        }
        Self::quantize_from_rotated(query_tokens_rot, num_query_tokens, &codebooks)
    }

    fn quantize_from_rotated(
        rot_tokens: &[f32],
        num_query_tokens: usize,
        codebooks: &NestedLatticeCodebooks,
    ) -> OspreyResult<Self> {
        let token_dim = codebooks.token_dim;
        let num_subspaces = token_dim / 2;
        if num_query_tokens == 0 {
            return Ok(Self {
                num_query_tokens: 0,
                num_subspaces,
                lut1_i8: AlignedBuffer::new(0)?,
                lut2_i8: AlignedBuffer::new(0)?,
                scale1: 1.0,
                scale2: codebooks.gamma,
            });
        }

        let l1_levels = [
            LATTICE_1D_LEVELS[0] * codebooks.sigma,
            LATTICE_1D_LEVELS[1] * codebooks.sigma,
            LATTICE_1D_LEVELS[2] * codebooks.sigma,
            LATTICE_1D_LEVELS[3] * codebooks.sigma,
        ];

        // Pass 1: find global max_abs1 = max |q_rot[2m] * cb1[k][0] + q_rot[2m+1] * cb1[k][1]|
        let mut max_abs1 = 0.0_f32;
        for q in 0..num_query_tokens {
            let q_rot = &rot_tokens[q * token_dim..(q + 1) * token_dim];
            for m in 0..num_subspaces {
                let qx = q_rot[2 * m];
                let qy = q_rot[2 * m + 1];
                let max_x = (qx * l1_levels[0]).abs().max((qx * l1_levels[3]).abs());
                let max_y = (qy * l1_levels[0]).abs().max((qy * l1_levels[3]).abs());
                let sub_max = max_x + max_y;
                if sub_max > max_abs1 {
                    max_abs1 = sub_max;
                }
            }
        }

        let scale1 = if max_abs1 < 1e-12 {
            1.0_f32
        } else {
            max_abs1 / 127.0_f32
        };
        let scale2 = scale1 * codebooks.gamma;
        let inv_scale1 = 1.0_f32 / scale1;

        let total_entries = num_query_tokens
            .checked_mul(num_subspaces)
            .and_then(|v| v.checked_mul(16))
            .ok_or(OspreyError::AllocationFailed {
                size: usize::MAX,
                align: CACHE_LINE_ALIGN,
            })?;
        let mut lut1_i8 = AlignedBuffer::<i8>::new(total_entries)?;
        let mut lut2_i8 = AlignedBuffer::<i8>::new(total_entries)?;

        let l1_slice = lut1_i8.as_mut_slice();
        let l2_slice = lut2_i8.as_mut_slice();

        // Pass 2: compute 8 1D products per subspace, add 4x4 Cartesian sums, and quantize to i8.
        for q in 0..num_query_tokens {
            let q_rot = &rot_tokens[q * token_dim..(q + 1) * token_dim];
            let q_base = q * num_subspaces * 16;
            for m in 0..num_subspaces {
                let qx = q_rot[2 * m];
                let qy = q_rot[2 * m + 1];

                let px = [
                    qx * l1_levels[0],
                    qx * l1_levels[1],
                    qx * l1_levels[2],
                    qx * l1_levels[3],
                ];
                let py = [
                    qy * l1_levels[0],
                    qy * l1_levels[1],
                    qy * l1_levels[2],
                    qy * l1_levels[3],
                ];

                let off = q_base + m * 16;
                for (i, &py_val) in py.iter().enumerate() {
                    for (j, &px_val) in px.iter().enumerate() {
                        let k = 4 * i + j;
                        let scaled = ((px_val + py_val) * inv_scale1).clamp(-127.0_f32, 127.0_f32);
                        #[allow(clippy::cast_possible_truncation)]
                        let q_i8 = ((scaled + 127.5_f32) as i32 - 127) as i8;
                        l1_slice[off + k] = q_i8;
                        l2_slice[off + k] = q_i8;
                    }
                }
            }
        }

        Ok(Self {
            num_query_tokens,
            num_subspaces,
            lut1_i8,
            lut2_i8,
            scale1,
            scale2,
        })
    }

    /// Returns `(lut1_q, lut2_q)` (`[N_sub * 16]` bytes each) for query token `q_idx`.
    #[inline]
    pub fn query_luts(&self, q_idx: usize) -> Option<(&[i8], &[i8])> {
        if q_idx >= self.num_query_tokens {
            return None;
        }
        let stride = self.num_subspaces * 16;
        let start = q_idx * stride;
        let end = start + stride;
        Some((&self.lut1_i8[start..end], &self.lut2_i8[start..end]))
    }
}

/// Portable scalar `MaxSim(Q, D)` over a document's contiguous [`OsrTokenCode`] slice.
pub fn score_doc_maxsim_scalar(doc_tokens: &[OsrTokenCode], query_luts: &[QueryTokenLuts]) -> f32 {
    if doc_tokens.is_empty() || query_luts.is_empty() {
        return 0.0;
    }

    let mut total_maxsim = 0.0_f32;
    for q_lut in query_luts {
        let mut best_sim = f32::NEG_INFINITY;
        for d_tok in doc_tokens {
            let sim = score_token_with_luts_scalar(d_tok, q_lut);
            if sim > best_sim {
                best_sim = sim;
            }
        }
        total_maxsim += best_sim;
    }
    total_maxsim
}

/// Input batch descriptor for building an `.osr` store from flattened document token embeddings.
#[derive(Clone, Copy, Debug)]
pub struct OsrDocumentTokens<'a> {
    /// Logical document identifier (does not need to be pre-sorted).
    pub doc_id: u64,
    /// Flattened row-major token embeddings (`[num_tokens * token_dim]`).
    pub tokens: &'a [f32],
    /// Number of tokens in this document (`<= u16::MAX`).
    pub num_tokens: usize,
}

enum OsrBacking<'a> {
    Owned(AlignedBuffer<u8>),
    Borrowed(&'a [u8]),
    Mmap(Mmap),
}

impl std::fmt::Debug for OsrBacking<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Owned(buf) => f.debug_tuple("Owned").field(&buf.len()).finish(),
            Self::Borrowed(slice) => f.debug_tuple("Borrowed").field(&slice.len()).finish(),
            Self::Mmap(mmap) => f.debug_tuple("Mmap").field(&mmap.len()).finish(),
        }
    }
}

/// Zero-copy `.osr` (Osprey SIMD-Interleaved Token Rerank Store) reader and builder.
///
/// Enforces 64-byte alignment on `OsrHeader`, `doc_table_offset`, `tensors_offset`, and every
/// 64-byte `OsrTokenCode` record, plus `header_xxh64` and global `file_xxh64` integrity checks.
/// Supports zero-copy reading over both `memmap2::Mmap` and 64-byte-aligned `&[u8]` slices.
#[derive(Debug)]
pub struct OsrStore<'a> {
    header: OsrHeader,
    doc_table_offset: usize,
    num_docs: usize,
    tensors_offset: usize,
    total_tokens: usize,
    backing: OsrBacking<'a>,
}

impl<'a> OsrStore<'a> {
    /// Opens a zero-copy `.osr` view borrowing directly from a 64-byte-aligned `&[u8]` slice.
    pub fn from_bytes(bytes: &'a [u8]) -> OspreyResult<Self> {
        let (header, doc_table_offset, num_docs, tensors_offset, total_tokens) =
            validate_osr_image(bytes)?;
        Ok(Self {
            header,
            doc_table_offset,
            num_docs,
            tensors_offset,
            total_tokens,
            backing: OsrBacking::Borrowed(bytes),
        })
    }

    /// Opens a zero-copy `.osr` view backed by a `memmap2::Mmap`.
    pub fn from_mmap(mmap: Mmap) -> OspreyResult<OsrStore<'static>> {
        let (header, doc_table_offset, num_docs, tensors_offset, total_tokens) =
            validate_osr_image(&mmap)?;
        Ok(OsrStore {
            header,
            doc_table_offset,
            num_docs,
            tensors_offset,
            total_tokens,
            backing: OsrBacking::Mmap(mmap),
        })
    }

    /// Memory-maps an `.osr` file from `path` in read-only zero-copy mode and validates all checksums.
    pub fn open_mmap(path: impl AsRef<Path>) -> OspreyResult<OsrStore<'static>> {
        let file = File::open(path)?;
        // SAFETY: The `.osr` file format is immutable once sealed, and `validate_osr_image`
        // verifies alignment, bounds, and `xxHash64` checksums before exposing any slice.
        let mmap = unsafe { Mmap::map(&file)? };
        OsrStore::from_mmap(mmap)
    }

    /// Wraps an owned 64-byte-aligned `.osr` byte buffer after full validation.
    pub fn from_aligned_buffer(buf: AlignedBuffer<u8>) -> OspreyResult<OsrStore<'static>> {
        let (header, doc_table_offset, num_docs, tensors_offset, total_tokens) =
            validate_osr_image(&buf)?;
        Ok(OsrStore {
            header,
            doc_table_offset,
            num_docs,
            tensors_offset,
            total_tokens,
            backing: OsrBacking::Owned(buf),
        })
    }

    /// Builds a sealed 64-byte-aligned `.osr` store in parallel from raw unrotated `f32`
    /// document token embeddings (`token_dim = 128`).
    ///
    /// Each token is L2-normalized, FWHT-rotated with `fwht_seed` via `simd`, and quantized
    /// across 2 nested 2D lattice layers without per-subspace normalization.
    pub fn build<K: RabitqFwhtKernel + ?Sized>(
        docs: &[OsrDocumentTokens<'_>],
        token_dim: usize,
        gamma: f32,
        fwht_seed: u64,
        simd: &K,
    ) -> OspreyResult<OsrStore<'static>> {
        if token_dim != LUTRANK_TOKEN_DIM {
            return Err(OspreyError::InvalidDimension { dim: token_dim });
        }
        let codebooks = NestedLatticeCodebooks::new(token_dim, gamma)?;
        let signs = RademacherSigns::from_seed(token_dim, fwht_seed)?;
        Self::build_internal(docs, &codebooks, fwht_seed, |scratch, raw_tok| {
            encode_token(raw_tok, &signs, &codebooks, simd, scratch)
        })
    }

    /// Builds a sealed 64-byte-aligned `.osr` store in parallel from **already FWHT-rotated**
    /// document token embeddings (`token_dim = 128`).
    pub fn build_from_prerotated(
        docs: &[OsrDocumentTokens<'_>],
        token_dim: usize,
        gamma: f32,
        fwht_seed: u64,
    ) -> OspreyResult<OsrStore<'static>> {
        if token_dim != LUTRANK_TOKEN_DIM {
            return Err(OspreyError::InvalidDimension { dim: token_dim });
        }
        let codebooks = NestedLatticeCodebooks::new(token_dim, gamma)?;
        Self::build_internal(docs, &codebooks, fwht_seed, |_scratch, rot_tok| {
            encode_prerotated_token(rot_tok, &codebooks)
        })
    }

    /// Builds a sealed 64-byte-aligned `.osr` store directly from pre-encoded [`OsrTokenCode`] slices.
    pub fn build_from_packed(
        docs: &[(u64, &[OsrTokenCode])],
        gamma: f32,
        fwht_seed: u64,
    ) -> OspreyResult<OsrStore<'static>> {
        let _ = NestedLatticeCodebooks::new(LUTRANK_TOKEN_DIM, gamma)?;

        let mut order: Vec<usize> = (0..docs.len()).collect();
        order.sort_unstable_by_key(|&idx| docs[idx].0);

        for win in order.windows(2) {
            if docs[win[0]].0 == docs[win[1]].0 {
                return Err(OspreyError::InvalidDimension {
                    dim: usize::try_from(docs[win[0]].0).unwrap_or(usize::MAX),
                });
            }
        }

        let mut total_tokens = 0_usize;
        let mut ranges = Vec::with_capacity(docs.len());
        for &idx in &order {
            let (doc_id, toks) = docs[idx];
            let count_u16 = u16::try_from(toks.len())
                .map_err(|_| OspreyError::InvalidDimension { dim: toks.len() })?;
            let start_u64 =
                u64::try_from(total_tokens).map_err(|_| OspreyError::AllocationFailed {
                    size: usize::MAX,
                    align: CACHE_LINE_ALIGN,
                })?;
            total_tokens =
                total_tokens
                    .checked_add(toks.len())
                    .ok_or(OspreyError::AllocationFailed {
                        size: usize::MAX,
                        align: CACHE_LINE_ALIGN,
                    })?;
            ranges.push(DocTokenRange {
                doc_id,
                token_start_idx: start_u64,
                token_count: count_u16,
                _reserved: [0; 6],
            });
        }

        let (mut out, header, doc_table_offset, tensors_offset, payload_end) =
            allocate_osr_layout(ranges.len(), total_tokens, gamma, fwht_seed)?;

        {
            let buf = out.as_mut_slice();
            buf[..size_of::<OsrHeader>()].copy_from_slice(bytemuck::bytes_of(&header));

            let table_bytes = ranges.len() * size_of::<DocTokenRange>();
            if table_bytes > 0 {
                buf[doc_table_offset..doc_table_offset + table_bytes]
                    .copy_from_slice(bytemuck::cast_slice(&ranges));
            }

            let tensors_bytes = total_tokens * OSR_TOKEN_BYTES;
            if tensors_bytes > 0 {
                let dst_tokens: &mut [OsrTokenCode] = bytemuck::cast_slice_mut(
                    &mut buf[tensors_offset..tensors_offset + tensors_bytes],
                );
                let mut cursor = 0_usize;
                for &idx in &order {
                    let src = docs[idx].1;
                    dst_tokens[cursor..cursor + src.len()].copy_from_slice(src);
                    cursor += src.len();
                }
            }

            let file_hash = xxh64(&buf[..payload_end], 0);
            buf[payload_end..].copy_from_slice(&file_hash.to_le_bytes());
        }

        Ok(OsrStore {
            header,
            doc_table_offset,
            num_docs: ranges.len(),
            tensors_offset,
            total_tokens,
            backing: OsrBacking::Owned(out),
        })
    }

    fn build_internal<F>(
        docs: &[OsrDocumentTokens<'_>],
        codebooks: &NestedLatticeCodebooks,
        fwht_seed: u64,
        encode_fn: F,
    ) -> OspreyResult<OsrStore<'static>>
    where
        F: Fn(&mut [f32], &[f32]) -> OspreyResult<OsrTokenCode> + Send + Sync,
    {
        let token_dim = codebooks.token_dim;
        let mut order: Vec<usize> = (0..docs.len()).collect();
        order.sort_unstable_by_key(|&idx| docs[idx].doc_id);

        for win in order.windows(2) {
            if docs[win[0]].doc_id == docs[win[1]].doc_id {
                return Err(OspreyError::InvalidDimension {
                    dim: usize::try_from(docs[win[0]].doc_id).unwrap_or(usize::MAX),
                });
            }
        }

        let mut total_tokens = 0_usize;
        let mut ranges = Vec::with_capacity(docs.len());
        for &idx in &order {
            let doc = &docs[idx];
            let expected_floats =
                doc.num_tokens
                    .checked_mul(token_dim)
                    .ok_or(OspreyError::InvalidDimension {
                        dim: doc.num_tokens,
                    })?;
            if doc.tokens.len() < expected_floats {
                return Err(OspreyError::BufferTooSmall {
                    needed: expected_floats * size_of::<f32>(),
                    actual: std::mem::size_of_val(doc.tokens),
                });
            }
            if doc.tokens.len() != expected_floats {
                return Err(OspreyError::InvalidDimension {
                    dim: doc.tokens.len(),
                });
            }
            let count_u16 =
                u16::try_from(doc.num_tokens).map_err(|_| OspreyError::InvalidDimension {
                    dim: doc.num_tokens,
                })?;
            let start_u64 =
                u64::try_from(total_tokens).map_err(|_| OspreyError::AllocationFailed {
                    size: usize::MAX,
                    align: CACHE_LINE_ALIGN,
                })?;
            total_tokens =
                total_tokens
                    .checked_add(doc.num_tokens)
                    .ok_or(OspreyError::AllocationFailed {
                        size: usize::MAX,
                        align: CACHE_LINE_ALIGN,
                    })?;

            ranges.push(DocTokenRange {
                doc_id: doc.doc_id,
                token_start_idx: start_u64,
                token_count: count_u16,
                _reserved: [0; 6],
            });
        }

        let (mut out, header, doc_table_offset, tensors_offset, payload_end) =
            allocate_osr_layout(ranges.len(), total_tokens, codebooks.gamma, fwht_seed)?;

        {
            let buf = out.as_mut_slice();
            buf[..size_of::<OsrHeader>()].copy_from_slice(bytemuck::bytes_of(&header));

            let table_bytes = ranges.len() * size_of::<DocTokenRange>();
            if table_bytes > 0 {
                buf[doc_table_offset..doc_table_offset + table_bytes]
                    .copy_from_slice(bytemuck::cast_slice(&ranges));
            }

            let tensors_bytes = total_tokens * OSR_TOKEN_BYTES;
            if tensors_bytes > 0 {
                let dst_tokens: &mut [OsrTokenCode] = bytemuck::cast_slice_mut(
                    &mut buf[tensors_offset..tensors_offset + tensors_bytes],
                );
                #[cfg(not(miri))]
                let chunk_size = 128_usize;
                #[cfg(not(miri))]
                dst_tokens
                    .par_chunks_mut(chunk_size)
                    .enumerate()
                    .try_for_each_init(
                        || vec![0.0_f32; token_dim],
                        |scratch, (chunk_idx, out_chunk)| -> OspreyResult<()> {
                            let start_tok = chunk_idx * chunk_size;
                            let start_tok_u64 = u64::try_from(start_tok)
                                .map_err(|_| OspreyError::CorruptedFooter)?;
                            let mut doc_pos = ranges.partition_point(|r| {
                                r.token_start_idx + u64::from(r.token_count) <= start_tok_u64
                            });

                            for (local_i, dst_code) in out_chunk.iter_mut().enumerate() {
                                let global_tok = start_tok + local_i;
                                let global_u64 = u64::try_from(global_tok)
                                    .map_err(|_| OspreyError::CorruptedFooter)?;
                                while doc_pos < ranges.len()
                                    && global_u64
                                        >= ranges[doc_pos].token_start_idx
                                            + u64::from(ranges[doc_pos].token_count)
                                {
                                    doc_pos += 1;
                                }
                                let range =
                                    ranges.get(doc_pos).ok_or(OspreyError::CorruptedFooter)?;
                                let local_tok = usize::try_from(global_u64 - range.token_start_idx)
                                    .map_err(|_| OspreyError::CorruptedFooter)?;
                                let src_doc = &docs[order[doc_pos]];
                                let tok_slice = &src_doc.tokens
                                    [local_tok * token_dim..(local_tok + 1) * token_dim];
                                *dst_code = encode_fn(scratch, tok_slice)?;
                            }
                            Ok(())
                        },
                    )?;
                #[cfg(miri)]
                {
                    let mut scratch = vec![0.0_f32; token_dim];
                    let mut doc_pos = 0_usize;
                    for (global_tok, dst_code) in dst_tokens.iter_mut().enumerate() {
                        let global_u64 =
                            u64::try_from(global_tok).map_err(|_| OspreyError::CorruptedFooter)?;
                        while doc_pos < ranges.len()
                            && global_u64
                                >= ranges[doc_pos].token_start_idx
                                    + u64::from(ranges[doc_pos].token_count)
                        {
                            doc_pos += 1;
                        }
                        let range = ranges.get(doc_pos).ok_or(OspreyError::CorruptedFooter)?;
                        let local_tok = usize::try_from(global_u64 - range.token_start_idx)
                            .map_err(|_| OspreyError::CorruptedFooter)?;
                        let src_doc = &docs[order[doc_pos]];
                        let tok_slice =
                            &src_doc.tokens[local_tok * token_dim..(local_tok + 1) * token_dim];
                        *dst_code = encode_fn(&mut scratch, tok_slice)?;
                    }
                }
            }

            let file_hash = xxh64(&buf[..payload_end], 0);
            buf[payload_end..].copy_from_slice(&file_hash.to_le_bytes());
        }

        Ok(OsrStore {
            header,
            doc_table_offset,
            num_docs: ranges.len(),
            tensors_offset,
            total_tokens,
            backing: OsrBacking::Owned(out),
        })
    }

    /// Returns the raw sealed `.osr` byte slice (`OsrHeader` .. `file_xxh64`).
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        match &self.backing {
            OsrBacking::Owned(buf) => buf.as_slice(),
            OsrBacking::Borrowed(slice) => slice,
            OsrBacking::Mmap(mmap) => mmap,
        }
    }

    /// Copies this `.osr` image into a fresh 64-byte-aligned owned buffer.
    pub fn to_aligned_bytes(&self) -> OspreyResult<AlignedBuffer<u8>> {
        AlignedBuffer::from_slice(self.as_bytes())
    }

    /// Returns the validated `.osr` header.
    #[inline]
    pub fn header(&self) -> &OsrHeader {
        &self.header
    }

    /// Returns the sorted `DocTokenRange` table zero-copy.
    #[inline]
    pub fn doc_table(&self) -> &[DocTokenRange] {
        if self.num_docs == 0 {
            return &[];
        }
        let table_bytes = self.num_docs * size_of::<DocTokenRange>();
        bytemuck::cast_slice(
            &self.as_bytes()[self.doc_table_offset..self.doc_table_offset + table_bytes],
        )
    }

    /// Returns all packed 64-byte [`OsrTokenCode`] records zero-copy.
    #[inline]
    pub fn all_tokens(&self) -> &[OsrTokenCode] {
        if self.total_tokens == 0 {
            return &[];
        }
        let tensors_bytes = self.total_tokens * OSR_TOKEN_BYTES;
        bytemuck::cast_slice(
            &self.as_bytes()[self.tensors_offset..self.tensors_offset + tensors_bytes],
        )
    }

    /// Resolves `doc_id` to its [`DocTokenRange`] in $O(\log N_{\text{docs}})$ via binary search.
    #[inline]
    pub fn lookup_doc(&self, doc_id: u64) -> OspreyResult<Option<DocTokenRange>> {
        let table = self.doc_table();
        Ok(table
            .binary_search_by_key(&doc_id, |entry| entry.doc_id)
            .ok()
            .map(|idx| table[idx]))
    }

    /// Returns a 64-byte-aligned byte slice (`token_count * 64` bytes) for `range`.
    #[inline]
    pub fn token_bytes(&self, range: &DocTokenRange) -> OspreyResult<&[u8]> {
        let start =
            usize::try_from(range.token_start_idx).map_err(|_| OspreyError::CorruptedFooter)?;
        let count = usize::from(range.token_count);
        let end = start
            .checked_add(count)
            .ok_or(OspreyError::CorruptedFooter)?;
        if end > self.total_tokens {
            return Err(OspreyError::BufferTooSmall {
                needed: end.saturating_mul(OSR_TOKEN_BYTES),
                actual: self.total_tokens.saturating_mul(OSR_TOKEN_BYTES),
            });
        }
        let byte_off_start = start
            .checked_mul(OSR_TOKEN_BYTES)
            .ok_or(OspreyError::CorruptedFooter)?;
        let byte_off_end = end
            .checked_mul(OSR_TOKEN_BYTES)
            .ok_or(OspreyError::CorruptedFooter)?;
        let byte_start = self
            .tensors_offset
            .checked_add(byte_off_start)
            .ok_or(OspreyError::CorruptedFooter)?;
        let byte_end = self
            .tensors_offset
            .checked_add(byte_off_end)
            .ok_or(OspreyError::CorruptedFooter)?;
        self.as_bytes()
            .get(byte_start..byte_end)
            .ok_or(OspreyError::BufferTooSmall {
                needed: byte_end,
                actual: self.as_bytes().len(),
            })
    }

    /// Returns the zero-copy [`OsrTokenCode`] slice for `range`.
    #[inline]
    pub fn doc_tokens(&self, range: &DocTokenRange) -> OspreyResult<&[OsrTokenCode]> {
        let bytes = self.token_bytes(range)?;
        if bytes.is_empty() {
            return Ok(&[]);
        }
        bytemuck::try_cast_slice(bytes).map_err(|_| OspreyError::UnalignedMemory {
            address: bytes.as_ptr() as usize,
            required_align: CACHE_LINE_ALIGN,
        })
    }

    /// Convenience lookup returning the zero-copy [`OsrTokenCode`] slice for `doc_id`.
    #[inline]
    pub fn lookup_doc_tokens(&self, doc_id: u64) -> OspreyResult<Option<&[OsrTokenCode]>> {
        match self.lookup_doc(doc_id)? {
            Some(range) => self.doc_tokens(&range).map(Some),
            None => Ok(None),
        }
    }

    /// Computes late-interaction `MaxSim(Q, D)` for document index `doc_idx` in `doc_table()`
    /// against `query_lut` using `simd`.
    ///
    /// Returns `0.0` if `doc_idx` is out of bounds, the document has `0` tokens, or
    /// `query_lut.num_query_tokens == 0`.
    #[inline]
    pub fn score_document_maxsim<K: LutRankKernel + ?Sized>(
        &self,
        doc_idx: usize,
        query_lut: &QuantizedQueryLut,
        simd: &K,
    ) -> f32 {
        let Some(range) = self.doc_table().get(doc_idx) else {
            return 0.0;
        };
        if range.token_count == 0
            || query_lut.num_query_tokens == 0
            || query_lut.num_subspaces != LUTRANK_SUBSPACES
        {
            return 0.0;
        }
        let Ok(doc_bytes) = self.token_bytes(range) else {
            return 0.0;
        };
        simd.lutrank_doc_maxsim(
            doc_bytes,
            usize::from(range.token_count),
            query_lut.num_subspaces,
            query_lut.lut1_i8.as_slice(),
            query_lut.lut2_i8.as_slice(),
            query_lut.num_query_tokens,
            query_lut.scale1,
            query_lut.scale2,
        )
    }

    /// Resolves `doc_id` via $O(\log N_{\text{docs}})$ binary search and computes `MaxSim(Q, D)`.
    #[inline]
    pub fn score_doc_id_maxsim<K: LutRankKernel + ?Sized>(
        &self,
        doc_id: u64,
        query_lut: &QuantizedQueryLut,
        simd: &K,
    ) -> OspreyResult<Option<f32>> {
        let table = self.doc_table();
        let Some(doc_idx) = table
            .binary_search_by_key(&doc_id, |entry| entry.doc_id)
            .ok()
        else {
            return Ok(None);
        };
        Ok(Some(self.score_document_maxsim(doc_idx, query_lut, simd)))
    }

    /// Prefetches the leading `.osr` token cache lines for `doc_idx` into L1 (`_MM_HINT_T0`)
    /// while the caller evaluates the preceding document.
    #[inline]
    pub fn prefetch_doc_idx<K: LutRankKernel + ?Sized>(&self, doc_idx: usize, simd: &K) {
        if let Some(range) = self.doc_table().get(doc_idx) {
            if let Ok(bytes) = self.token_bytes(range) {
                if !bytes.is_empty() {
                    simd.prefetch_l1(bytes);
                    if bytes.len() > 64 {
                        simd.prefetch_l1(&bytes[64..]);
                    }
                }
            }
        }
    }

    /// Resolves `doc_id` via binary search and prefetches its leading `.osr` token cache lines
    /// into L1 (`_MM_HINT_T0`).
    #[inline]
    pub fn prefetch_doc_id<K: LutRankKernel + ?Sized>(&self, doc_id: u64, simd: &K) {
        if let Ok(doc_idx) = self
            .doc_table()
            .binary_search_by_key(&doc_id, |entry| entry.doc_id)
        {
            self.prefetch_doc_idx(doc_idx, simd);
        }
    }
}

impl RerankStoreReader for OsrStore<'_> {
    #[inline]
    fn header(&self) -> &OsrHeader {
        OsrStore::header(self)
    }

    #[inline]
    fn lookup_doc(&self, doc_id: u64) -> OspreyResult<Option<DocTokenRange>> {
        OsrStore::lookup_doc(self, doc_id)
    }

    #[inline]
    fn token_bytes(&self, range: &DocTokenRange) -> OspreyResult<&[u8]> {
        OsrStore::token_bytes(self, range)
    }
}

fn allocate_osr_layout(
    num_docs: usize,
    total_tokens: usize,
    gamma: f32,
    fwht_seed: u64,
) -> OspreyResult<(AlignedBuffer<u8>, OsrHeader, usize, usize, usize)> {
    let doc_table_offset = size_of::<OsrHeader>();
    let doc_table_bytes =
        num_docs
            .checked_mul(size_of::<DocTokenRange>())
            .ok_or(OspreyError::AllocationFailed {
                size: usize::MAX,
                align: CACHE_LINE_ALIGN,
            })?;
    let tensors_offset = doc_table_offset
        .checked_add(doc_table_bytes)
        .and_then(checked_align_up_64)
        .ok_or(OspreyError::AllocationFailed {
            size: usize::MAX,
            align: CACHE_LINE_ALIGN,
        })?;
    let tensors_bytes =
        total_tokens
            .checked_mul(OSR_TOKEN_BYTES)
            .ok_or(OspreyError::AllocationFailed {
                size: usize::MAX,
                align: CACHE_LINE_ALIGN,
            })?;
    let payload_end =
        tensors_offset
            .checked_add(tensors_bytes)
            .ok_or(OspreyError::AllocationFailed {
                size: usize::MAX,
                align: CACHE_LINE_ALIGN,
            })?;
    let total_len =
        payload_end
            .checked_add(size_of::<u64>())
            .ok_or(OspreyError::AllocationFailed {
                size: usize::MAX,
                align: CACHE_LINE_ALIGN,
            })?;

    let num_docs_u64 = u64::try_from(num_docs).map_err(|_| OspreyError::AllocationFailed {
        size: usize::MAX,
        align: CACHE_LINE_ALIGN,
    })?;
    let total_tokens_u64 =
        u64::try_from(total_tokens).map_err(|_| OspreyError::AllocationFailed {
            size: usize::MAX,
            align: CACHE_LINE_ALIGN,
        })?;
    let doc_table_offset_u64 =
        u64::try_from(doc_table_offset).map_err(|_| OspreyError::AllocationFailed {
            size: usize::MAX,
            align: CACHE_LINE_ALIGN,
        })?;
    let tensors_offset_u64 =
        u64::try_from(tensors_offset).map_err(|_| OspreyError::AllocationFailed {
            size: usize::MAX,
            align: CACHE_LINE_ALIGN,
        })?;

    let header = OsrHeader::new_sealed(
        OSR_CANONICAL_SUBSPACE_DIM,
        OSR_CANONICAL_NUM_LAYERS,
        OSR_CANONICAL_TOKEN_DIM,
        gamma,
        fwht_seed,
        num_docs_u64,
        total_tokens_u64,
        doc_table_offset_u64,
        tensors_offset_u64,
    )?;

    let out = AlignedBuffer::<u8>::new(total_len)?;
    Ok((out, header, doc_table_offset, tensors_offset, payload_end))
}

fn validate_osr_image(bytes: &[u8]) -> OspreyResult<(OsrHeader, usize, usize, usize, usize)> {
    ensure_aligned_64(bytes)?;
    let header = *OsrHeader::from_bytes(bytes)?;

    if header.subspace_dim != OSR_CANONICAL_SUBSPACE_DIM
        || header.num_layers != OSR_CANONICAL_NUM_LAYERS
        || header.token_dim != OSR_CANONICAL_TOKEN_DIM
    {
        return Err(OspreyError::InvalidDimension {
            dim: usize::from(header.token_dim),
        });
    }

    let doc_table_offset =
        usize::try_from(header.doc_table_offset).map_err(|_| OspreyError::CorruptedFooter)?;
    let tensors_offset =
        usize::try_from(header.tensors_offset).map_err(|_| OspreyError::CorruptedFooter)?;
    let num_docs = usize::try_from(header.num_docs).map_err(|_| OspreyError::CorruptedFooter)?;
    let total_tokens =
        usize::try_from(header.total_tokens).map_err(|_| OspreyError::CorruptedFooter)?;

    let doc_table_bytes = num_docs
        .checked_mul(size_of::<DocTokenRange>())
        .ok_or(OspreyError::CorruptedFooter)?;
    let min_tensors_offset = doc_table_offset
        .checked_add(doc_table_bytes)
        .and_then(checked_align_up_64)
        .ok_or(OspreyError::CorruptedFooter)?;

    if doc_table_offset < size_of::<OsrHeader>() || tensors_offset < min_tensors_offset {
        return Err(OspreyError::CorruptedFooter);
    }

    let tensors_bytes = total_tokens
        .checked_mul(OSR_TOKEN_BYTES)
        .ok_or(OspreyError::CorruptedFooter)?;
    let payload_end = tensors_offset
        .checked_add(tensors_bytes)
        .ok_or(OspreyError::CorruptedFooter)?;
    let expected_total_len = payload_end
        .checked_add(size_of::<u64>())
        .ok_or(OspreyError::CorruptedFooter)?;

    if bytes.len() < expected_total_len {
        return Err(OspreyError::BufferTooSmall {
            needed: expected_total_len,
            actual: bytes.len(),
        });
    }
    if bytes.len() != expected_total_len {
        return Err(OspreyError::CorruptedFooter);
    }

    let hash_bytes: [u8; 8] = bytes[payload_end..expected_total_len]
        .try_into()
        .map_err(|_| OspreyError::CorruptedFooter)?;
    let expected_hash = u64::from_le_bytes(hash_bytes);
    let computed_hash = xxh64(&bytes[..payload_end], 0);
    if expected_hash != computed_hash {
        return Err(OspreyError::ChecksumMismatch {
            expected: expected_hash,
            computed: computed_hash,
        });
    }

    if num_docs > 0 {
        let ranges = cast_doc_token_ranges(&bytes[doc_table_offset..], header.num_docs)?;
        let mut prev_doc_id: Option<u64> = None;
        for range in ranges {
            if let Some(prev) = prev_doc_id {
                if range.doc_id <= prev {
                    return Err(OspreyError::CorruptedFooter);
                }
            }
            prev_doc_id = Some(range.doc_id);

            let end_tok = range
                .token_start_idx
                .checked_add(u64::from(range.token_count))
                .ok_or(OspreyError::CorruptedFooter)?;
            if end_tok > header.total_tokens {
                return Err(OspreyError::CorruptedFooter);
            }
        }
    }

    if total_tokens > 0 {
        let tensor_slice = &bytes[tensors_offset..payload_end];
        let _: &[OsrTokenCode] =
            bytemuck::try_cast_slice(tensor_slice).map_err(|_| OspreyError::UnalignedMemory {
                address: tensor_slice.as_ptr() as usize,
                required_align: CACHE_LINE_ALIGN,
            })?;
    }

    Ok((
        header,
        doc_table_offset,
        num_docs,
        tensors_offset,
        total_tokens,
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::math::SplitMix64;
    use crate::rabitq::ScalarRabitqKernel;

    const LUTRANK_GOLDEN_BYTES: &[u8] =
        include_bytes!("../../../reference/golden/lutrank_golden.bin");

    fn read_f32_le(bytes: &[u8], offset: usize) -> f32 {
        f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    }

    #[test]
    fn codebooks_match_poc_reference_and_golden_fixture() {
        let cb = NestedLatticeCodebooks::canonical().unwrap();
        let (cb1, cb2) = build_nested_lattice_codebooks(128, 0.28).unwrap();
        assert_eq!(cb.layer1, cb1);
        assert_eq!(cb.layer2, cb2);

        // Verify exact Cartesian order (i in 0..4, j in 0..4 -> idx = 4*i + j)
        // matches poc_reference.py: cb[c] = [levels[c % 4] * sigma, levels[c / 4] * sigma].
        let expected_sigma = 1.0_f32 / (128.0_f32).sqrt();
        assert!((cb.sigma - expected_sigma).abs() <= 1e-7);

        for c in 0..16 {
            let off = c * 8;
            let golden_x1 = read_f32_le(LUTRANK_GOLDEN_BYTES, off);
            let golden_y1 = read_f32_le(LUTRANK_GOLDEN_BYTES, off + 4);
            let golden_x2 = read_f32_le(LUTRANK_GOLDEN_BYTES, 128 + off);
            let golden_y2 = read_f32_le(LUTRANK_GOLDEN_BYTES, 128 + off + 4);

            assert!(
                (cb.layer1[c][0] - golden_x1).abs() <= 1e-6,
                "cb1[{c}][0]: got {}, golden {}",
                cb.layer1[c][0],
                golden_x1
            );
            assert!(
                (cb.layer1[c][1] - golden_y1).abs() <= 1e-6,
                "cb1[{c}][1]: got {}, golden {}",
                cb.layer1[c][1],
                golden_y1
            );
            assert!(
                (cb.layer2[c][0] - golden_x2).abs() <= 1e-6,
                "cb2[{c}][0]: got {}, golden {}",
                cb.layer2[c][0],
                golden_x2
            );
            assert!(
                (cb.layer2[c][1] - golden_y2).abs() <= 1e-6,
                "cb2[{c}][1]: got {}, golden {}",
                cb.layer2[c][1],
                golden_y2
            );
        }

        // Validation guards
        assert!(matches!(
            build_nested_lattice_codebooks(0, 0.28).unwrap_err(),
            OspreyError::InvalidDimension { dim: 0 }
        ));
        assert!(matches!(
            build_nested_lattice_codebooks(127, 0.28).unwrap_err(),
            OspreyError::InvalidDimension { dim: 127 }
        ));
        assert!(matches!(
            build_nested_lattice_codebooks(128, 0.0).unwrap_err(),
            OspreyError::NonFiniteEmbedding { index: 0 }
        ));
        assert!(matches!(
            build_nested_lattice_codebooks(128, f32::NAN).unwrap_err(),
            OspreyError::NonFiniteEmbedding { index: 0 }
        ));
    }

    #[test]
    fn nibble_packing_roundtrip_and_bounds() {
        let mut c1 = [0_u8; LUTRANK_SUBSPACES];
        let mut c2 = [0_u8; LUTRANK_SUBSPACES];
        for s in 0..LUTRANK_SUBSPACES {
            c1[s] = u8::try_from(s % 16).unwrap();
            c2[s] = u8::try_from((15 - s % 16) % 16).unwrap();
        }

        let packed_tok = OsrTokenCode::from_codes(&c1, &c2);
        let mut out_c1 = [0_u8; LUTRANK_SUBSPACES];
        let mut out_c2 = [0_u8; LUTRANK_SUBSPACES];
        packed_tok.unpack_layer1(&mut out_c1);
        packed_tok.unpack_layer2(&mut out_c2);
        assert_eq!(out_c1, c1);
        assert_eq!(out_c2, c2);

        let mut p1 = [0_u8; 32];
        let mut p2 = [0_u8; 32];
        pack_nibbles(&c1, &mut p1).unwrap();
        pack_nibbles(&c2, &mut p2).unwrap();
        assert_eq!(p1, packed_tok.layer1);
        assert_eq!(p2, packed_tok.layer2);

        let mut u1 = [0_u8; 64];
        unpack_nibbles(&p1, &mut u1).unwrap();
        assert_eq!(u1, c1);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn golden_lutrank_parity_bit_exact_nibbles_luts_and_maxsim() {
        assert_eq!(LUTRANK_GOLDEN_BYTES.len(), 582_080);

        let n_docs = 16usize;
        let l_d = 32usize;
        let l_q = 8usize;
        let dim = 128usize;
        let seed = 0x4F53_5052_0004_u64;

        let codebooks = NestedLatticeCodebooks::canonical().unwrap();
        let signs = RademacherSigns::from_seed(dim, seed).unwrap();
        let simd = ScalarRabitqKernel;

        // 1. Verify doc_tokens_raw -> FWHT -> encode_token matches doc_nibbles bit-exact
        //    and doc_tokens_rot -> encode_prerotated_token matches doc_nibbles bit-exact.
        let mut doc_tokens_raw = vec![0.0_f32; n_docs * l_d * dim];
        for (i, dst) in doc_tokens_raw.iter_mut().enumerate() {
            *dst = read_f32_le(LUTRANK_GOLDEN_BYTES, 256 + i * 4);
        }
        let mut doc_tokens_rot = vec![0.0_f32; n_docs * l_d * dim];
        for (i, dst) in doc_tokens_rot.iter_mut().enumerate() {
            *dst = read_f32_le(LUTRANK_GOLDEN_BYTES, 262_400 + i * 4);
        }

        let golden_nibbles = &LUTRANK_GOLDEN_BYTES[524_544..524_544 + n_docs * l_d * 64];
        let mut scratch = vec![0.0_f32; dim];

        for d_idx in 0..n_docs {
            for t_idx in 0..l_d {
                let tok_off = (d_idx * l_d + t_idx) * dim;
                let nib_off = (d_idx * l_d + t_idx) * 64;
                let expected_tok_bytes = &golden_nibbles[nib_off..nib_off + 64];

                let code_from_rot =
                    encode_prerotated_token(&doc_tokens_rot[tok_off..tok_off + dim], &codebooks)
                        .unwrap();
                assert_eq!(
                    bytemuck::bytes_of(&code_from_rot),
                    expected_tok_bytes,
                    "prerotated nibble mismatch at doc={d_idx} tok={t_idx}"
                );

                let code_from_raw = encode_token(
                    &doc_tokens_raw[tok_off..tok_off + dim],
                    &signs,
                    &codebooks,
                    &simd,
                    &mut scratch,
                )
                .unwrap();
                assert_eq!(
                    bytemuck::bytes_of(&code_from_raw),
                    expected_tok_bytes,
                    "raw->FWHT nibble mismatch at doc={d_idx} tok={t_idx}"
                );
            }
        }

        // 2. Verify Query LUT synthesis against q_lut1_i8, q_lut2_i8, scale1, scale2
        let mut q_luts = vec![QueryTokenLuts::default(); l_q];
        for qi in 0..l_q {
            let mut q_rot = [0.0_f32; LUTRANK_TOKEN_DIM];
            for (d, dst) in q_rot.iter_mut().enumerate() {
                *dst = read_f32_le(LUTRANK_GOLDEN_BYTES, 561_408 + (qi * dim + d) * 4);
            }
            synthesize_query_token_luts(&q_rot, &codebooks, &mut q_luts[qi]).unwrap();

            let golden_s1 = read_f32_le(LUTRANK_GOLDEN_BYTES, 581_888 + qi * 4);
            let golden_s2 = read_f32_le(LUTRANK_GOLDEN_BYTES, 581_920 + qi * 4);
            assert!(
                (q_luts[qi].scale1 - golden_s1).abs() <= 1e-6,
                "scale1 mismatch q={qi}"
            );
            assert!(
                (q_luts[qi].scale2 - golden_s2).abs() <= 1e-6,
                "scale2 mismatch q={qi}"
            );

            for s in 0..LUTRANK_SUBSPACES {
                for c in 0..16 {
                    let g1 =
                        i8::from_le_bytes([LUTRANK_GOLDEN_BYTES[565_504 + (qi * 64 + s) * 16 + c]]);
                    let g2 =
                        i8::from_le_bytes([LUTRANK_GOLDEN_BYTES[573_696 + (qi * 64 + s) * 16 + c]]);
                    assert!(
                        (i16::from(q_luts[qi].lut1[s][c]) - i16::from(g1)).abs() <= 1,
                        "lut1 mismatch q={qi} s={s} c={c}: got {}, golden {g1}",
                        q_luts[qi].lut1[s][c]
                    );
                    assert!(
                        (i16::from(q_luts[qi].lut2[s][c]) - i16::from(g2)).abs() <= 1,
                        "lut2 mismatch q={qi} s={s} c={c}: got {}, golden {g2}",
                        q_luts[qi].lut2[s][c]
                    );
                }
            }
        }

        // 3. Build OsrStore in reverse doc_id order, serialize, reopen via &[u8] & Mmap,
        //    and verify lookup_doc + scalar MaxSim matches maxsim_lut_f32 (<= 1e-4).
        let doc_inputs: Vec<OsrDocumentTokens<'_>> = (0..n_docs)
            .rev()
            .map(|d_idx| OsrDocumentTokens {
                doc_id: u64::try_from(1000 + d_idx * 10).unwrap(),
                tokens: &doc_tokens_raw[d_idx * l_d * dim..(d_idx + 1) * l_d * dim],
                num_tokens: l_d,
            })
            .collect();

        let store = OsrStore::build(&doc_inputs, dim, DEFAULT_LUTRANK_GAMMA, seed, &simd).unwrap();
        let bytes = store.to_aligned_bytes().unwrap();
        assert_eq!((bytes.as_ptr() as usize) % 64, 0);

        let borrowed_store = OsrStore::from_bytes(&bytes).unwrap();
        let reader: &dyn RerankStoreReader = &borrowed_store;
        assert_eq!(reader.header(), store.header());

        // Also test memmap2::Mmap zero-copy reading via temp file
        let temp_path =
            std::env::temp_dir().join(format!("osprey_test_lutrank_{}.osr", std::process::id()));
        std::fs::write(&temp_path, &bytes).unwrap();
        let mmap_store = OsrStore::open_mmap(&temp_path).unwrap();
        let _ = std::fs::remove_file(&temp_path);

        for d_idx in 0..n_docs {
            let doc_id = u64::try_from(1000 + d_idx * 10).unwrap();
            let range = borrowed_store.lookup_doc(doc_id).unwrap().unwrap();
            assert_eq!(range.token_count, 32);

            let toks_borrowed = borrowed_store.doc_tokens(&range).unwrap();
            let toks_mmap = mmap_store.lookup_doc_tokens(doc_id).unwrap().unwrap();
            assert_eq!(toks_borrowed, toks_mmap);

            let maxsim = score_doc_maxsim_scalar(toks_mmap, &q_luts);
            let golden_maxsim = read_f32_le(LUTRANK_GOLDEN_BYTES, 582_016 + d_idx * 4);
            assert!(
                (maxsim - golden_maxsim).abs() <= 1e-3,
                "maxsim mismatch doc={d_idx}: got {maxsim}, golden {golden_maxsim}"
            );
        }
    }

    #[test]
    fn osr_store_empty_zero_token_doc_and_corruption_guards() {
        let simd = ScalarRabitqKernel;
        let seed = 0x4F53_5052_9999_u64;

        // Empty store (0 docs)
        let empty = OsrStore::build(&[], 128, 0.28, seed, &simd).unwrap();
        assert_eq!(empty.doc_table().len(), 0);
        assert_eq!(empty.all_tokens().len(), 0);
        assert_eq!(empty.lookup_doc(42).unwrap(), None);
        let empty_bytes = empty.to_aligned_bytes().unwrap();
        let empty_rt = OsrStore::from_bytes(&empty_bytes).unwrap();
        assert_eq!(empty_rt.lookup_doc(42).unwrap(), None);

        // Store with a 0-token doc + normal doc + binary search miss
        let mut rng = SplitMix64::new(0x1234_5678);
        let tok: Vec<f32> = (0..128)
            .map(|_| {
                #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
                {
                    ((rng.next_u64() >> 11) as f64 / ((1_u64 << 53) as f64) * 2.0 - 1.0) as f32
                }
            })
            .collect();
        let docs = [
            OsrDocumentTokens {
                doc_id: 50,
                tokens: &tok,
                num_tokens: 1,
            },
            OsrDocumentTokens {
                doc_id: 10,
                tokens: &[],
                num_tokens: 0,
            },
        ];

        let store = OsrStore::build(&docs, 128, 0.28, seed, &simd).unwrap();
        let bytes = store.to_aligned_bytes().unwrap();
        let rt = OsrStore::from_bytes(&bytes).unwrap();
        assert_eq!(rt.doc_table().len(), 2);
        assert_eq!(rt.doc_table()[0].doc_id, 10);
        assert_eq!(rt.doc_table()[1].doc_id, 50);
        assert_eq!(rt.lookup_doc_tokens(10).unwrap().unwrap().len(), 0);
        assert_eq!(rt.lookup_doc_tokens(50).unwrap().unwrap().len(), 1);
        assert_eq!(rt.lookup_doc_tokens(99).unwrap(), None);

        // Reject unaligned slice
        assert!(matches!(
            OsrStore::from_bytes(&bytes[1..]).unwrap_err(),
            OspreyError::UnalignedMemory {
                required_align: 64,
                ..
            }
        ));

        // Reject single-bit corruption in tensor payload (file_xxh64)
        let mut corrupted = AlignedBuffer::from_slice(&bytes).unwrap();
        let tens_off = usize::try_from(rt.header().tensors_offset).unwrap();
        corrupted[tens_off + 7] ^= 0x04;
        assert!(matches!(
            OsrStore::from_bytes(&corrupted).unwrap_err(),
            OspreyError::ChecksumMismatch { .. }
        ));

        // Reject duplicate doc_id on build
        let dup_docs = [
            OsrDocumentTokens {
                doc_id: 7,
                tokens: &tok,
                num_tokens: 1,
            },
            OsrDocumentTokens {
                doc_id: 7,
                tokens: &tok,
                num_tokens: 1,
            },
        ];
        assert!(OsrStore::build(&dup_docs, 128, 0.28, seed, &simd).is_err());

        // Reject non-finite token
        let mut bad_tok = tok;
        bad_tok[13] = f32::NAN;
        let bad_docs = [OsrDocumentTokens {
            doc_id: 1,
            tokens: &bad_tok,
            num_tokens: 1,
        }];
        assert!(matches!(
            OsrStore::build(&bad_docs, 128, 0.28, seed, &simd).unwrap_err(),
            OspreyError::NonFiniteEmbedding { index: 13 }
        ));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn quantized_query_lut_and_score_document_maxsim_match_golden() {
        let n_docs = 16usize;
        let l_d = 32usize;
        let l_q = 8usize;
        let dim = 128usize;
        let seed = 0x4F53_5052_0004_u64;
        let simd = ScalarRabitqKernel;

        let mut doc_tokens_raw = vec![0.0_f32; n_docs * l_d * dim];
        for (i, dst) in doc_tokens_raw.iter_mut().enumerate() {
            *dst = read_f32_le(LUTRANK_GOLDEN_BYTES, 256 + i * 4);
        }
        let mut q_tokens_raw = vec![0.0_f32; l_q * dim];
        for (i, dst) in q_tokens_raw.iter_mut().enumerate() {
            *dst = read_f32_le(LUTRANK_GOLDEN_BYTES, 557_312 + i * 4);
        }

        let doc_inputs: Vec<OsrDocumentTokens<'_>> = (0..n_docs)
            .map(|d_idx| OsrDocumentTokens {
                doc_id: u64::try_from(d_idx).unwrap(),
                tokens: &doc_tokens_raw[d_idx * l_d * dim..(d_idx + 1) * l_d * dim],
                num_tokens: l_d,
            })
            .collect();
        let store = OsrStore::build(&doc_inputs, dim, DEFAULT_LUTRANK_GAMMA, seed, &simd).unwrap();

        let q_lut = QuantizedQueryLut::synthesize(
            &q_tokens_raw,
            l_q,
            dim,
            DEFAULT_LUTRANK_GAMMA,
            seed,
            &simd,
        )
        .unwrap();
        assert_eq!(q_lut.num_query_tokens, l_q);
        assert_eq!(q_lut.num_subspaces, 64);
        assert_eq!((q_lut.lut1_i8.as_ptr() as usize) % 64, 0);
        assert_eq!((q_lut.lut2_i8.as_ptr() as usize) % 64, 0);
        assert_eq!(q_lut.lut1_i8.len(), l_q * 64 * 16);
        assert_eq!(q_lut.lut2_i8.len(), l_q * 64 * 16);
        assert!((q_lut.scale2 - q_lut.scale1 * DEFAULT_LUTRANK_GAMMA).abs() <= 1e-7);

        for d_idx in 0..n_docs {
            let score = store.score_document_maxsim(d_idx, &q_lut, &simd);
            let golden_maxsim = read_f32_le(LUTRANK_GOLDEN_BYTES, 582_016 + d_idx * 4);
            assert!(
                (score - golden_maxsim).abs() <= 0.05,
                "doc={d_idx} QuantizedQueryLut score={score}, golden={golden_maxsim}"
            );
        }

        // Out-of-bounds doc_idx and empty query return 0.0 safely
        assert_eq!(store.score_document_maxsim(999, &q_lut, &simd), 0.0);
        let empty_q =
            QuantizedQueryLut::synthesize(&[], 0, dim, DEFAULT_LUTRANK_GAMMA, seed, &simd).unwrap();
        assert_eq!(store.score_document_maxsim(0, &empty_q, &simd), 0.0);
    }
}
