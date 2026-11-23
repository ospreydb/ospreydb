//! `cargo-fuzz` target for `.osv` (`OsvHeader`, `MinipageDirectoryEntry`,
//! `OsvSlabReader::from_bytes`, `OsvSlabView::from_bytes`, `decompress_minipage`,
//! and truncated/malformed Zstd streams).

#![no_main]

use libfuzzer_sys::fuzz_target;
use osprey_core::{AlignedBuffer, OsvHeader};
use osprey_storage::{OsvSlabReader, OsvSlabView};
use xxhash_rust::xxh64::xxh64;

fn exercise_osv_aligned(aligned: AlignedBuffer<u8>) {
    let slice = aligned.as_slice();
    let _ = OsvHeader::from_bytes(slice);
    if let Ok(view) = OsvSlabView::from_bytes(slice) {
        let count = view.directory.len().min(32);
        for id in 0..count {
            let _ = view.read_minipage(id);
        }
    }

    if let Ok(reader) = OsvSlabReader::from_bytes(aligned) {
        let count = u32::try_from(reader.directory().len().min(32)).unwrap_or(0);
        for id in 0..count {
            if let Ok(page) = reader.decompress_minipage(id) {
                let _ = page.doc_ids();
                if page.vector_count() > 0 {
                    let _ = page.vector_f16(0);
                    let _ = page.payload_bytes(0);
                }
            }
        }
    }
}

fuzz_target!(|data: &[u8]| {
    // 1. Raw (potentially unaligned) input slice.
    let _ = OsvHeader::from_bytes(data);
    let _ = OsvSlabView::from_bytes(data);

    // 2. 64-byte aligned copy.
    if let Ok(aligned) = AlignedBuffer::<u8>::from_slice(data) {
        exercise_osv_aligned(aligned);
    }

    // 3. Structure-aware header checksum sealing to probe integer overflow and Zstd paths.
    if data.len() >= 64 {
        if let Ok(mut aligned) = AlignedBuffer::<u8>::from_slice(data) {
            let buf = aligned.as_mut_slice();
            let hdr_hash = xxh64(&buf[..56], 0);
            buf[56..64].copy_from_slice(&hdr_hash.to_le_bytes());
            exercise_osv_aligned(aligned);
        }
    }
});
