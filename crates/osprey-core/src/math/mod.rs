//! Pure algebraic primitives, deterministic Rademacher signs, padding, and scalar FWHT.

pub mod fwht;
pub mod padding;
pub mod rademacher;

pub use fwht::fwht_scalar_in_place;
pub use padding::{
    pad_vector_preserve_scale, padded_dimension, prepare_and_pad_vector, prepare_and_pad_vector_f16,
};
pub use rademacher::{RademacherSigns, SplitMix64, F32_SIGN_BIT_MASK};
