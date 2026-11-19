//! Vector sanitization, exact L2 norm computation, and power-of-two zero-padding.

use half::f16;

use crate::error::{OspreyError, OspreyResult};

/// Maximum supported input dimensionality before power-of-two padding.
pub const MAX_INPUT_DIM: usize = 65_536;

/// Minimum L2 norm threshold below which division is skipped to preserve stability.
pub const NORM_EPSILON: f64 = 1e-12;

/// Returns the power-of-two padded dimensionality `max(64, orig_dim.next_power_of_two())`.
///
/// Validates `0 < orig_dim <= 65536`.
#[inline]
pub const fn padded_dimension(orig_dim: usize) -> OspreyResult<usize> {
    if orig_dim == 0 || orig_dim > MAX_INPUT_DIM {
        return Err(OspreyError::InvalidDimension { dim: orig_dim });
    }
    let pow2 = orig_dim.next_power_of_two();
    Ok(if pow2 < 64 { 64 } else { pow2 })
}

/// Validates finiteness, computes exact `f64` L2 norm, normalizes `input` into
/// `out_padded[..input.len()]` (when `norm > 1e-12`), and zero-pads `[input.len()..]`.
///
/// Returns the computed L2 norm in `f32`.
pub fn prepare_and_pad_vector(input: &[f32], out_padded: &mut [f32]) -> OspreyResult<f32> {
    prepare_pad_internal(input, out_padded, true)
}

/// Validates finiteness, computes exact `f64` L2 norm, copies `input` preserving its
/// original magnitude (for residuals or already-normalized inputs), and zero-pads tail.
pub fn pad_vector_preserve_scale(input: &[f32], out_padded: &mut [f32]) -> OspreyResult<f32> {
    prepare_pad_internal(input, out_padded, false)
}

fn prepare_pad_internal(
    input: &[f32],
    out_padded: &mut [f32],
    normalize: bool,
) -> OspreyResult<f32> {
    let expected_padded = padded_dimension(input.len())?;
    if out_padded.len() < expected_padded {
        return Err(OspreyError::BufferTooSmall {
            needed: expected_padded,
            actual: out_padded.len(),
        });
    }
    if out_padded.len() != expected_padded {
        return Err(OspreyError::InvalidDimension {
            dim: out_padded.len(),
        });
    }

    let mut sq_sum = 0.0_f64;
    for (index, &val) in input.iter().enumerate() {
        if !val.is_finite() {
            return Err(OspreyError::NonFiniteEmbedding { index });
        }
        let v = f64::from(val);
        sq_sum += v * v;
    }

    let norm_f64 = sq_sum.sqrt();
    if normalize && norm_f64 > NORM_EPSILON {
        let inv_norm = 1.0_f64 / norm_f64;
        for (dst, &src) in out_padded[..input.len()].iter_mut().zip(input.iter()) {
            #[allow(clippy::cast_possible_truncation)]
            {
                *dst = (f64::from(src) * inv_norm) as f32;
            }
        }
    } else {
        out_padded[..input.len()].copy_from_slice(input);
    }

    out_padded[input.len()..expected_padded].fill(0.0_f32);
    #[allow(clippy::cast_possible_truncation)]
    let norm_f32 = norm_f64 as f32;
    Ok(norm_f32)
}

/// `f16` variant of [`prepare_and_pad_vector`]: validates finiteness, computes the exact `f64`
/// L2 norm over promoted `f16` values, normalizes into `out_padded[..input.len()]`, and
/// zero-pads the tail. Avoids materializing an intermediate `f32` vector during streaming
/// `f16`-backed `.osi` builds.
pub fn prepare_and_pad_vector_f16(input: &[f16], out_padded: &mut [f32]) -> OspreyResult<f32> {
    let expected_padded = padded_dimension(input.len())?;
    if out_padded.len() != expected_padded {
        return Err(OspreyError::InvalidDimension {
            dim: out_padded.len(),
        });
    }

    let mut sq_sum = 0.0_f64;
    for (index, &val) in input.iter().enumerate() {
        let f32_val = val.to_f32();
        if !f32_val.is_finite() {
            return Err(OspreyError::NonFiniteEmbedding { index });
        }
        let v = f64::from(f32_val);
        sq_sum += v * v;
    }

    let norm_f64 = sq_sum.sqrt();
    if norm_f64 > NORM_EPSILON {
        let inv_norm = 1.0_f64 / norm_f64;
        for (dst, &src) in out_padded[..input.len()].iter_mut().zip(input.iter()) {
            #[allow(clippy::cast_possible_truncation)]
            {
                *dst = (f64::from(src.to_f32()) * inv_norm) as f32;
            }
        }
    } else {
        for (dst, &src) in out_padded[..input.len()].iter_mut().zip(input.iter()) {
            *dst = src.to_f32();
        }
    }

    out_padded[input.len()..expected_padded].fill(0.0_f32);
    #[allow(clippy::cast_possible_truncation)]
    let norm_f32 = norm_f64 as f32;
    Ok(norm_f32)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn padded_dimension_mappings_and_bounds() {
        assert_eq!(padded_dimension(1).unwrap(), 64);
        assert_eq!(padded_dimension(64).unwrap(), 64);
        assert_eq!(padded_dimension(384).unwrap(), 512);
        assert_eq!(padded_dimension(768).unwrap(), 1024);
        assert_eq!(padded_dimension(1536).unwrap(), 2048);
        assert_eq!(padded_dimension(65_536).unwrap(), 65_536);

        assert!(matches!(
            padded_dimension(0).unwrap_err(),
            OspreyError::InvalidDimension { dim: 0 }
        ));
        assert!(matches!(
            padded_dimension(65_537).unwrap_err(),
            OspreyError::InvalidDimension { dim: 65_537 }
        ));
    }

    #[test]
    fn prepare_and_pad_rejects_non_finite_and_normalizes() {
        let mut out = vec![99.0_f32; 64];
        let bad = [1.0_f32, f32::NAN, 2.0];
        let err = prepare_and_pad_vector(&bad, &mut out).unwrap_err();
        assert!(matches!(err, OspreyError::NonFiniteEmbedding { index: 1 }));

        let bad_inf = [1.0_f32, 2.0, f32::INFINITY];
        let err_inf = prepare_and_pad_vector(&bad_inf, &mut out).unwrap_err();
        assert!(matches!(
            err_inf,
            OspreyError::NonFiniteEmbedding { index: 2 }
        ));

        let good = [3.0_f32, 4.0];
        let norm = prepare_and_pad_vector(&good, &mut out).unwrap();
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((out[0] - 0.6).abs() < 1e-6);
        assert!((out[1] - 0.8).abs() < 1e-6);
        assert!(out[2..].iter().all(|&x| x == 0.0));
    }
}
