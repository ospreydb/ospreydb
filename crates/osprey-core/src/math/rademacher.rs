//! Deterministic Rademacher sign generation (`SplitMix64` / optional `ChaCha8`)
//! and branchless IEEE-754 XOR sign-flip masks for FWHT.

use crate::aligned::AlignedBuffer;
use crate::error::{OspreyError, OspreyResult};

/// IEEE-754 `f32` sign-bit mask (`-0.0_f32.to_bits()`).
pub const F32_SIGN_BIT_MASK: u32 = 0x8000_0000;

/// Bit-exact `SplitMix64` PRNG matching `reference/poc_reference.py` and
/// `reference/generate_golden_vectors.py`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Initializes `SplitMix64` with the segment's `fwht_seed`.
    #[inline]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advances state by one 64-bit step and returns the mixed output word.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Fills `out_words` with packed 64-bit Rademacher sign words.
    ///
    /// Bit `b` in `[0, 64)` maps to `-1.0` when `0` and `+1.0` when `1`.
    #[inline]
    pub fn fill_words(&mut self, out_words: &mut [u64]) {
        for word in out_words.iter_mut() {
            *word = self.next_u64();
        }
    }
}

/// Self-contained RFC 8439 ChaCha8 stream generator for optional future format flags.
#[derive(Clone, Debug)]
struct ChaCha8Stream {
    key_words: [u32; 8],
    counter: u32,
}

impl ChaCha8Stream {
    fn new(seed: u64) -> Self {
        let lo = u32::try_from(seed & 0xFFFF_FFFF).unwrap_or(0);
        let hi = u32::try_from(seed >> 32).unwrap_or(0);
        Self {
            key_words: [
                lo,
                hi,
                lo ^ 0x4F53_5052,
                hi ^ 0x4657_4854,
                lo.rotate_left(13),
                hi.rotate_right(7),
                0x9E37_79B9,
                0x7F4A_7C15,
            ],
            counter: 0,
        }
    }

    #[inline]
    fn quarter_round(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
        state[a] = state[a].wrapping_add(state[b]);
        state[d] = (state[d] ^ state[a]).rotate_left(16);
        state[c] = state[c].wrapping_add(state[d]);
        state[b] = (state[b] ^ state[c]).rotate_left(12);
        state[a] = state[a].wrapping_add(state[b]);
        state[d] = (state[d] ^ state[a]).rotate_left(8);
        state[c] = state[c].wrapping_add(state[d]);
        state[b] = (state[b] ^ state[c]).rotate_left(7);
    }

    fn next_block_words(&mut self) -> [u64; 8] {
        let initial: [u32; 16] = [
            0x6170_7865,
            0x3320_646E,
            0x7962_2D32,
            0x6B20_6574,
            self.key_words[0],
            self.key_words[1],
            self.key_words[2],
            self.key_words[3],
            self.key_words[4],
            self.key_words[5],
            self.key_words[6],
            self.key_words[7],
            self.counter,
            0,
            0,
            0,
        ];
        self.counter = self.counter.wrapping_add(1);

        let mut work = initial;
        for _ in 0..4 {
            Self::quarter_round(&mut work, 0, 4, 8, 12);
            Self::quarter_round(&mut work, 1, 5, 9, 13);
            Self::quarter_round(&mut work, 2, 6, 10, 14);
            Self::quarter_round(&mut work, 3, 7, 11, 15);
            Self::quarter_round(&mut work, 0, 5, 10, 15);
            Self::quarter_round(&mut work, 1, 6, 11, 12);
            Self::quarter_round(&mut work, 2, 7, 8, 13);
            Self::quarter_round(&mut work, 3, 4, 9, 14);
        }

        let mut out = [0_u64; 8];
        for (i, slot) in out.iter_mut().enumerate() {
            let lo = work[2 * i].wrapping_add(initial[2 * i]);
            let hi = work[2 * i + 1].wrapping_add(initial[2 * i + 1]);
            *slot = u64::from(lo) | (u64::from(hi) << 32);
        }
        out
    }
}

/// Pre-computed Rademacher signs in packed bit-words and 64-byte-aligned IEEE-754 XOR masks.
pub struct RademacherSigns {
    padded_dim: usize,
    words: Vec<u64>,
    xor_masks: AlignedBuffer<u32>,
}

impl RademacherSigns {
    /// Generates canonical v1 signs using deterministic `SplitMix64` over `fwht_seed`.
    pub fn from_seed(padded_dim: usize, fwht_seed: u64) -> OspreyResult<Self> {
        validate_padded_dim(padded_dim)?;
        let n_words = padded_dim.div_ceil(64);
        let mut words = vec![0_u64; n_words];
        let mut rng = SplitMix64::new(fwht_seed);
        rng.fill_words(&mut words);
        Self::from_words(padded_dim, words)
    }

    /// Generates signs using an optional `ChaCha8` stream for future format flags.
    pub fn from_chacha8(padded_dim: usize, fwht_seed: u64) -> OspreyResult<Self> {
        validate_padded_dim(padded_dim)?;
        let n_words = padded_dim.div_ceil(64);
        let mut words = Vec::with_capacity(n_words);
        let mut chacha = ChaCha8Stream::new(fwht_seed);
        while words.len() < n_words {
            let block = chacha.next_block_words();
            let take = (n_words - words.len()).min(block.len());
            words.extend_from_slice(&block[..take]);
        }
        Self::from_words(padded_dim, words)
    }

    /// Constructs `RademacherSigns` directly from pre-packed `u64` words.
    pub fn from_words(padded_dim: usize, words: Vec<u64>) -> OspreyResult<Self> {
        validate_padded_dim(padded_dim)?;
        let expected_words = padded_dim.div_ceil(64);
        if words.len() != expected_words {
            return Err(OspreyError::InvalidDimension { dim: padded_dim });
        }

        let mut xor_masks = AlignedBuffer::<u32>::new(padded_dim)?;
        let masks_slice = xor_masks.as_mut_slice();
        for (word_idx, &word) in words.iter().enumerate() {
            let base = word_idx * 64;
            let take = (padded_dim - base).min(64);
            for bit in 0..take {
                // Bit == 0 -> -1.0 (0x8000_0000), bit == 1 -> +1.0 (0x0000_0000), branchless.
                let is_zero = u32::from(((word >> bit) & 1) == 0);
                masks_slice[base + bit] = is_zero << 31;
            }
        }

        Ok(Self {
            padded_dim,
            words,
            xor_masks,
        })
    }

    /// Returns the power-of-two padded dimension covered by these signs.
    #[inline]
    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    /// Returns the packed 64-bit Rademacher sign words (`bit 0 -> -1.0`, `bit 1 -> +1.0`).
    #[inline]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Returns the 64-byte-aligned IEEE-754 XOR masks (`0x8000_0000` or `0x0000_0000`).
    #[inline]
    pub fn xor_masks(&self) -> &[u32] {
        self.xor_masks.as_slice()
    }

    /// Applies Rademacher sign flips in-place via branchless IEEE-754 bit XOR.
    #[inline]
    pub fn apply_in_place(&self, data: &mut [f32]) -> OspreyResult<()> {
        if data.len() != self.padded_dim {
            return Err(OspreyError::InvalidDimension { dim: data.len() });
        }
        let masks = self.xor_masks.as_slice();
        for (val, &mask) in data.iter_mut().zip(masks.iter()) {
            *val = f32::from_bits(val.to_bits() ^ mask);
        }
        Ok(())
    }
}

#[inline]
fn validate_padded_dim(padded_dim: usize) -> OspreyResult<()> {
    if padded_dim == 0 || !padded_dim.is_power_of_two() || padded_dim > 65_536 {
        return Err(OspreyError::InvalidDimension { dim: padded_dim });
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_matches_golden_seed_deterministically() {
        let signs = RademacherSigns::from_seed(128, 0x4F53_5052_0001).unwrap();
        assert_eq!(signs.words().len(), 2);
        assert_eq!(signs.xor_masks().len(), 128);
        assert_eq!((signs.xor_masks().as_ptr() as usize) % 64, 0);

        // Verify bit-to-mask correspondence and branchless sign flip
        let mut data = vec![1.25_f32; 128];
        signs.apply_in_place(&mut data).unwrap();
        for (i, &val) in data.iter().enumerate() {
            let bit = (signs.words()[i / 64] >> (i % 64)) & 1;
            let expected = if bit == 0 { -1.25_f32 } else { 1.25_f32 };
            assert_eq!(val.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn chacha8_signs_are_deterministic_and_valid() {
        let a = RademacherSigns::from_chacha8(256, 42).unwrap();
        let b = RademacherSigns::from_chacha8(256, 42).unwrap();
        assert_eq!(a.words(), b.words());
        assert_eq!(a.xor_masks(), b.xor_masks());
    }
}
