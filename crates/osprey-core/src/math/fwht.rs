//! Reference scalar Fast Walsh-Hadamard Transform (FWHT) orthogonal rotation.

use crate::error::{OspreyError, OspreyResult};
use crate::math::rademacher::RademacherSigns;

/// Applies orthogonal FWHT `P x = (1 / sqrt(d)) * H_d * D_signs * x` in-place.
///
/// Validates that `data.len()` is a power of two matching `signs.padded_dim()`.
pub fn fwht_scalar_in_place(data: &mut [f32], signs: &RademacherSigns) -> OspreyResult<()> {
    let d = data.len();
    if d == 0 || !d.is_power_of_two() || d != signs.padded_dim() {
        return Err(OspreyError::InvalidDimension { dim: d });
    }

    // Step 1: branchless IEEE-754 XOR Rademacher sign flips
    signs.apply_in_place(data)?;

    // Step 2: O(d log d) in-place Walsh-Hadamard butterfly stages
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

    // Step 3: orthonormal scaling factor 1 / sqrt(d)
    let d_u32 = u32::try_from(d).map_err(|_| OspreyError::InvalidDimension { dim: d })?;
    #[allow(clippy::cast_possible_truncation)]
    let inv_sqrt_d = (1.0_f64 / f64::from(d_u32).sqrt()) as f32;
    for val in data.iter_mut() {
        *val *= inv_sqrt_d;
    }

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::math::padding::{pad_vector_preserve_scale, padded_dimension};

    #[test]
    fn fwht_scalar_preserves_l2_norm_and_inner_products() {
        let orig_dim = 96;
        let p_dim = padded_dimension(orig_dim).unwrap();
        assert_eq!(p_dim, 128);

        let signs = RademacherSigns::from_seed(p_dim, 0x4F53_5052_0001).unwrap();

        let mut x = vec![0.0_f32; orig_dim];
        let mut y = vec![0.0_f32; orig_dim];
        for i in 0..orig_dim {
            let idx = f32::from(u16::try_from(i).unwrap());
            x[i] = (idx * 0.13).sin();
            y[i] = (idx * 0.29).cos();
        }

        let exact_dot: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum();

        let mut px = vec![0.0_f32; p_dim];
        let mut py = vec![0.0_f32; p_dim];
        let nx = pad_vector_preserve_scale(&x, &mut px).unwrap();
        let ny = pad_vector_preserve_scale(&y, &mut py).unwrap();

        fwht_scalar_in_place(&mut px, &signs).unwrap();
        fwht_scalar_in_place(&mut py, &signs).unwrap();

        let rot_nx = px
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            .sqrt();
        let rot_ny = py
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            .sqrt();
        let rot_dot: f64 = px
            .iter()
            .zip(py.iter())
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum();

        assert!((rot_nx - f64::from(nx)).abs() < 1e-5);
        assert!((rot_ny - f64::from(ny)).abs() < 1e-5);
        assert!((rot_dot - exact_dot).abs() < 1e-4);
    }

    #[test]
    fn fwht_scalar_matches_golden_d64_fixture() {
        let raw: &[u8] = include_bytes!("../../../../reference/golden/fwht_d64.bin");
        // fwht_d64.bin sections: sign_words (1 x u64 = 8B), inputs (16 x 64 x f32 = 4096B), outputs (16 x 64 x f32 = 4096B)
        assert_eq!(raw.len(), 8 + 4096 + 4096);
        let golden_word = u64::from_le_bytes(raw[..8].try_into().unwrap());
        let signs = RademacherSigns::from_seed(64, 0x4F53_5052_0001).unwrap();
        assert_eq!(signs.words()[0], golden_word);

        for vec_idx in 0..16 {
            let in_off = 8 + vec_idx * 64 * 4;
            let out_off = 8 + 4096 + vec_idx * 64 * 4;
            let mut work = [0.0_f32; 64];
            let mut expected = [0.0_f32; 64];
            for j in 0..64 {
                work[j] = f32::from_le_bytes(
                    raw[in_off + j * 4..in_off + (j + 1) * 4]
                        .try_into()
                        .unwrap(),
                );
                expected[j] = f32::from_le_bytes(
                    raw[out_off + j * 4..out_off + (j + 1) * 4]
                        .try_into()
                        .unwrap(),
                );
            }
            fwht_scalar_in_place(&mut work, &signs).unwrap();
            let max_err = work
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(max_err <= 1e-5, "vec {vec_idx} max_err={max_err}");
        }
    }
}
