//! Canonical zero-copy binary format definitions for `.osi`, `.osv`, and `.osr`.

pub mod osi;
pub mod osr;
pub mod osv;

/// Global magic bytes `"OSPR"` (`[0x4F, 0x53, 0x50, 0x52]`).
pub const OSPREY_MAGIC: [u8; 4] = *b"OSPR";

/// Current canonical binary format version (`v1`).
pub const FORMAT_VERSION_V1: u16 = 1;

/// Discriminant for `.osi` (Osprey Resident 1-Bit Index).
pub const FORMAT_KIND_OSI: u8 = 0x01;

/// Discriminant for `.osv` (Osprey Locality-Reordered Vector Slabs).
pub const FORMAT_KIND_OSV: u8 = 0x02;

/// Discriminant for `.osr` (Osprey SIMD-Interleaved Token Rerank Store).
pub const FORMAT_KIND_OSR: u8 = 0x03;

/// Rounds `val` up to the next 64-byte cache-line boundary (`(val + 63) & !63`).
#[inline]
pub const fn align_up_64(val: usize) -> usize {
    (val + 63) & !63
}

/// Checked 64-byte alignment helper returning `None` on `usize` overflow.
#[inline]
pub const fn checked_align_up_64(val: usize) -> Option<usize> {
    match val.checked_add(63) {
        Some(v) => Some(v & !63),
        None => None,
    }
}
