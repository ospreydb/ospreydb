//! 64-byte cache-line aligned buffer allocator and alignment guards.
//!
//! Guarantees compatibility with AVX-512 aligned loads (`vmovdqa64`), `memmap2`
//! page alignments, and `bytemuck::Pod` zero-copy casts without undefined behavior.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

use bytemuck::Pod;

use crate::error::{OspreyError, OspreyResult};

/// Required cache-line alignment in bytes (64 bytes = AVX-512 ZMM / CPU cache line).
pub const CACHE_LINE_ALIGN: usize = 64;

/// Contiguous heap buffer guaranteed to be aligned to at least 64 bytes.
pub struct AlignedBuffer<T: Pod> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
}

// SAFETY: `AlignedBuffer<T>` exclusively owns its backing allocation of `T: Pod`.
unsafe impl<T: Pod + Send> Send for AlignedBuffer<T> {}
// SAFETY: Shared references expose `&[T]`, which is thread-safe whenever `T: Sync`.
unsafe impl<T: Pod + Sync> Sync for AlignedBuffer<T> {}

impl<T: Pod> AlignedBuffer<T> {
    /// Allocates a zero-initialized buffer for `capacity` elements of `T` aligned to 64 bytes.
    ///
    /// Returns `NonNull::dangling()` without calling the global allocator when
    /// `capacity == 0` or `T` is a zero-sized type.
    pub fn new(capacity: usize) -> OspreyResult<Self> {
        let elem_size = std::mem::size_of::<T>();
        let align = CACHE_LINE_ALIGN.max(std::mem::align_of::<T>());

        if !align.is_power_of_two() {
            return Err(OspreyError::AllocationFailed { size: 0, align });
        }

        if capacity == 0 || elem_size == 0 {
            let ptr = match NonZeroUsize::new(align) {
                Some(nz) => NonNull::without_provenance(nz),
                None => NonNull::dangling(),
            };
            return Ok(Self {
                ptr,
                len: capacity,
                capacity: 0,
            });
        }

        let byte_len = capacity
            .checked_mul(elem_size)
            .ok_or(OspreyError::AllocationFailed {
                size: usize::MAX,
                align,
            })?;

        let layout = Layout::from_size_align(byte_len, align).map_err(|_| {
            OspreyError::AllocationFailed {
                size: byte_len,
                align,
            }
        })?;

        // SAFETY: `byte_len > 0` and `align` is a power of two >= `align_of::<T>()`.
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(raw.cast::<T>()).ok_or(OspreyError::AllocationFailed {
            size: byte_len,
            align,
        })?;

        Ok(Self {
            ptr,
            len: capacity,
            capacity,
        })
    }

    /// Allocates a 64-byte-aligned buffer initialized by copying `src`.
    pub fn from_slice(src: &[T]) -> OspreyResult<Self> {
        let mut buf = Self::new(src.len())?;
        buf.as_mut_slice().copy_from_slice(src);
        Ok(buf)
    }

    /// Returns the number of elements in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the buffer contains zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the allocated capacity in elements (`0` for empty or ZST buffers).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns an immutable slice view of all elements.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `self.ptr` is non-null, 64-byte aligned, and valid for `self.len` elements
        // (including `len == 0` / ZST via `NonNull::without_provenance`).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Returns a mutable slice view of all elements.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: `&mut self` guarantees exclusive access over `self.len` elements.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Pod> Deref for AlignedBuffer<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: Pod> DerefMut for AlignedBuffer<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T: Pod> AsRef<[u8]> for AlignedBuffer<T> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        bytemuck::cast_slice(self.as_slice())
    }
}

impl<T: Pod> Drop for AlignedBuffer<T> {
    fn drop(&mut self) {
        let elem_size = std::mem::size_of::<T>();
        if self.capacity == 0 || elem_size == 0 {
            return;
        }
        let align = CACHE_LINE_ALIGN.max(std::mem::align_of::<T>());
        let byte_len = self.capacity * elem_size;
        if let Ok(layout) = Layout::from_size_align(byte_len, align) {
            // SAFETY: `self.ptr` was allocated via `alloc_zeroed` with this exact `layout`.
            unsafe {
                dealloc(self.ptr.as_ptr().cast::<u8>(), layout);
            }
        }
    }
}

/// Verifies that the starting pointer of `bytes` is 64-byte aligned.
#[inline]
pub fn ensure_aligned_64(bytes: &[u8]) -> OspreyResult<()> {
    let address = bytes.as_ptr() as usize;
    if !address.is_multiple_of(CACHE_LINE_ALIGN) {
        return Err(OspreyError::UnalignedMemory {
            address,
            required_align: CACHE_LINE_ALIGN,
        });
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn aligned_buffer_respects_64_byte_alignment() {
        for len in [0usize, 1, 7, 17, 64, 513, 4096] {
            let mut buf = AlignedBuffer::<u64>::new(len).unwrap();
            assert_eq!(buf.len(), len);
            assert_eq!(buf.is_empty(), len == 0);
            if len > 0 {
                let addr = buf.as_ptr() as usize;
                assert_eq!(addr % CACHE_LINE_ALIGN, 0);
                assert!(ensure_aligned_64(buf.as_ref()).is_ok());
                assert!(buf.iter().all(|&x| x == 0));
                buf[0] = 0xDEAD_BEEF;
                assert_eq!(buf[0], 0xDEAD_BEEF);
            }
        }
    }

    #[test]
    fn aligned_buffer_zst_and_from_slice() {
        let zst = AlignedBuffer::<()>::new(128).unwrap();
        assert_eq!(zst.len(), 128);
        assert_eq!(zst.capacity(), 0);

        let src = [1.0_f32, -2.5, 3.25, 0.0];
        let buf = AlignedBuffer::from_slice(&src).unwrap();
        assert_eq!(buf.as_slice(), &src);
        assert_eq!(buf.as_ref().len(), src.len() * std::mem::size_of::<f32>());
    }

    #[test]
    fn ensure_aligned_64_rejects_unaligned_slice() {
        let buf = AlignedBuffer::<u8>::new(128).unwrap();
        assert!(ensure_aligned_64(&buf[..]).is_ok());
        let err = ensure_aligned_64(&buf[1..]).unwrap_err();
        assert!(matches!(
            err,
            OspreyError::UnalignedMemory {
                required_align: 64,
                ..
            }
        ));
    }

    #[test]
    fn aligned_buffer_overflow_returns_error() {
        let err = AlignedBuffer::<u64>::new(usize::MAX).err().unwrap();
        assert!(matches!(err, OspreyError::AllocationFailed { .. }));
    }
}
