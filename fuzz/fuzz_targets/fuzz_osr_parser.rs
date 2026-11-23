//! `cargo-fuzz` target for `.osr` (`OsrHeader`, `DocTokenRange`,
//! `OsrStore::from_bytes`, `OsrTokenCode` nibbles, and `score_document_maxsim`).

#![no_main]

use libfuzzer_sys::fuzz_target;
use osprey_core::{
    cast_doc_token_ranges, AlignedBuffer, OsrHeader, OsrStore, QuantizedQueryLut,
    ScalarRabitqKernel, DEFAULT_LUTRANK_GAMMA, LUTRANK_TOKEN_DIM,
};
use xxhash_rust::xxh64::xxh64;

fn exercise_osr_slice(slice: &[u8], q_lut: &QuantizedQueryLut) {
    let _ = OsrHeader::from_bytes(slice);
    let _ = cast_doc_token_ranges(slice, 16);

    if let Ok(store) = OsrStore::from_bytes(slice) {
        let _ = store.all_tokens();
        let _ = store.lookup_doc(0);
        let docs_to_probe = store.doc_table().len().min(32);
        for doc_idx in 0..docs_to_probe {
            let _ = store.score_document_maxsim(doc_idx, q_lut, &ScalarRabitqKernel);
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let Ok(q_lut) = QuantizedQueryLut::synthesize(
        &[0.1_f32; LUTRANK_TOKEN_DIM],
        1,
        LUTRANK_TOKEN_DIM,
        DEFAULT_LUTRANK_GAMMA,
        0x4F53_5052,
        &ScalarRabitqKernel,
    ) else {
        return;
    };

    // 1. Raw (potentially unaligned) input slice.
    exercise_osr_slice(data, &q_lut);

    // 2. 64-byte aligned copy.
    if let Ok(mut aligned) = AlignedBuffer::<u8>::from_slice(data) {
        exercise_osr_slice(aligned.as_slice(), &q_lut);

        // 3. Structure-aware checksum sealing to probe DocTokenRange OOB and truncated nibbles.
        if aligned.len() >= 72 {
            let buf = aligned.as_mut_slice();
            let hdr_hash = xxh64(&buf[..56], 0);
            buf[56..64].copy_from_slice(&hdr_hash.to_le_bytes());

            let payload_end = buf.len() - 8;
            let file_hash = xxh64(&buf[..payload_end], 0);
            buf[payload_end..].copy_from_slice(&file_hash.to_le_bytes());

            exercise_osr_slice(buf, &q_lut);
        }
    }
});
