//! `cargo-fuzz` target for `.osi` (`OsiHeader`, `f16` centroids, `RabitqEntryHeader`,
//! `[u64]` 1-bit codes, `RoaringBitmap` tombstones, and `OsiIndex::search_candidates`).

#![no_main]

use libfuzzer_sys::fuzz_target;
use osprey_core::{
    deserialize_tombstone_footer, AlignedBuffer, OsiHeader, OsiIndex, OsiMappedView,
    ScalarRabitqKernel,
};
use xxhash_rust::xxh64::xxh64;

fn exercise_osi_slice(slice: &[u8]) {
    let _ = OsiHeader::from_bytes(slice);
    let _ = deserialize_tombstone_footer(slice);

    if let Ok(view) = OsiMappedView::from_bytes(slice) {
        let _ = view.centroids_f16();
        let _ = view.tombstones();
        let k = view.header().num_clusters.min(64);
        for c in 0..k {
            let _ = view.cluster_view(c);
        }
    }

    if let Ok(index) = OsiIndex::from_bytes(slice) {
        let orig_dim = usize::from(index.header().orig_dim);
        if orig_dim > 0 && orig_dim <= 65_536 {
            let query = vec![0.1_f32; orig_dim];
            let _ = index.search_candidates(&query, 4, 16, &ScalarRabitqKernel);
        }
    }
}

fuzz_target!(|data: &[u8]| {
    // 1. Raw (potentially unaligned) input slice.
    exercise_osi_slice(data);

    // 2. 64-byte aligned copy.
    if let Ok(mut aligned) = AlignedBuffer::<u8>::from_slice(data) {
        exercise_osi_slice(aligned.as_slice());

        // 3. Structure-aware checksum sealing so fuzzer penetrates post-checksum layout checks.
        if aligned.len() >= 72 {
            let buf = aligned.as_mut_slice();
            let hdr_hash = xxh64(&buf[..56], 0);
            buf[56..64].copy_from_slice(&hdr_hash.to_le_bytes());

            let payload_end = buf.len() - 8;
            let file_hash = xxh64(&buf[..payload_end], 0);
            buf[payload_end..].copy_from_slice(&file_hash.to_le_bytes());

            exercise_osi_slice(buf);
        }
    }
});
