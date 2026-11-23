//! Deterministic 50,000+ hostile-mutation smoke fuzzer for `.osi`
//! and `.osr` integrated into `cargo test`.
//!
//! Exercises:
//! - Arbitrary random bytes (aligned and unaligned);
//! - Prefix truncations at every single byte offset of valid `.osi` and `.osr` images;
//! - Adversarial `u64::MAX` / `u32::MAX` / `0` size and offset mutations (both unsealed and
//!   with re-sealed `header_xxh64` and `file_xxh64` to test post-checksum arithmetic & bounds);
//! - Bit-flips across headers, centroid tables, `OsiClusterSpan`, `RabitqEntryHeader`,
//!   `RoaringBitmap` tombstones, `DocTokenRange`, and 64-byte nibble tensors.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use osprey_core::math::SplitMix64;
use osprey_core::{
    cast_doc_token_ranges, deserialize_tombstone_footer, fit_binary_kmeans, AlignedBuffer,
    BinaryKMeansConfig, OsiHeader, OsiIndex, OsiMappedView, OsrDocumentTokens, OsrHeader, OsrStore,
    QuantizedQueryLut, ScalarClusterKernel, ScalarRabitqKernel, VectorLocation,
    DEFAULT_LUTRANK_GAMMA, LUTRANK_TOKEN_DIM,
};
use roaring::RoaringBitmap;
use xxhash_rust::xxh64::xxh64;

fn build_seed_osi() -> Vec<u8> {
    let dim = 96usize;
    let n = 24usize;
    let mut rng = SplitMix64::new(0x5EED_0510_0001);
    let vectors: Vec<f32> = (0..n * dim)
        .map(|_| (((rng.next_u64() >> 40) as f32) / ((1_u64 << 24) as f32)) * 2.0 - 1.0)
        .collect();
    let kmeans = fit_binary_kmeans(
        &vectors,
        n,
        dim,
        0x1111_2222,
        &BinaryKMeansConfig::new(4),
        &ScalarClusterKernel,
    )
    .unwrap();
    let doc_ids: Vec<u64> = (0..n).map(|i| 100 + i as u64).collect();
    let locs: Vec<VectorLocation> = (0..n)
        .map(|i| VectorLocation {
            slab_id: 0,
            minipage_id: u32::try_from(i / 8).unwrap(),
            slot_idx: u16::try_from(i % 8).unwrap(),
        })
        .collect();
    let mut tombstones = RoaringBitmap::new();
    tombstones.insert(103);
    tombstones.insert(117);

    let idx = OsiIndex::build(
        &vectors,
        &doc_ids,
        &locs,
        &kmeans,
        dim,
        0x1111_2222,
        tombstones,
        &ScalarRabitqKernel,
    )
    .unwrap();
    idx.to_aligned_bytes().unwrap().as_slice().to_vec()
}

fn build_seed_osr() -> (Vec<u8>, QuantizedQueryLut) {
    let mut rng = SplitMix64::new(0x5EED_0520_0002);
    let mut doc_vecs = Vec::new();
    for doc_idx in 0..6u64 {
        let num_toks = usize::try_from((doc_idx % 3) + 1).unwrap();
        let toks: Vec<f32> = (0..num_toks * LUTRANK_TOKEN_DIM)
            .map(|_| (((rng.next_u64() >> 40) as f32) / ((1_u64 << 24) as f32)) * 2.0 - 1.0)
            .collect();
        doc_vecs.push((100 + doc_idx * 10, num_toks, toks));
    }
    let descs: Vec<OsrDocumentTokens<'_>> = doc_vecs
        .iter()
        .map(|(doc_id, num_tokens, toks)| OsrDocumentTokens {
            doc_id: *doc_id,
            tokens: toks,
            num_tokens: *num_tokens,
        })
        .collect();
    let store = OsrStore::build(
        &descs,
        LUTRANK_TOKEN_DIM,
        DEFAULT_LUTRANK_GAMMA,
        0x3333_4444,
        &ScalarRabitqKernel,
    )
    .unwrap();

    let q_lut = QuantizedQueryLut::synthesize(
        &[0.15_f32; LUTRANK_TOKEN_DIM],
        1,
        LUTRANK_TOKEN_DIM,
        DEFAULT_LUTRANK_GAMMA,
        0x3333_4444,
        &ScalarRabitqKernel,
    )
    .unwrap();

    (store.as_bytes().to_vec(), q_lut)
}

#[inline]
fn reseal_osi_or_osr(buf: &mut [u8]) {
    if buf.len() >= 72 {
        let hdr_hash = xxh64(&buf[..56], 0);
        buf[56..64].copy_from_slice(&hdr_hash.to_le_bytes());
        let payload_end = buf.len() - 8;
        let file_hash = xxh64(&buf[..payload_end], 0);
        buf[payload_end..].copy_from_slice(&file_hash.to_le_bytes());
    }
}

fn run_osi_target_once(data: &[u8]) -> bool {
    let _ = OsiHeader::from_bytes(data);
    let _ = deserialize_tombstone_footer(data);
    let _ = OsiMappedView::from_bytes(data);
    let _ = OsiIndex::from_bytes(data);

    let Ok(aligned) = AlignedBuffer::<u8>::from_slice(data) else {
        return false;
    };
    let slice = aligned.as_slice();
    let _ = OsiHeader::from_bytes(slice);
    if let Ok(view) = OsiMappedView::from_bytes(slice) {
        let _ = view.centroids_f16();
        let k = view.header().num_clusters.min(16);
        for c in 0..k {
            let _ = view.cluster_view(c);
        }
    }
    match OsiIndex::from_bytes(slice) {
        Ok(index) => {
            let orig_dim = usize::from(index.header().orig_dim);
            if orig_dim > 0 && orig_dim <= 65_536 {
                let q = vec![0.1_f32; orig_dim];
                let _ = index.search_candidates(&q, 4, 8, &ScalarRabitqKernel);
            }
            true
        }
        Err(_) => false,
    }
}

fn run_osr_target_once(data: &[u8], q_lut: &QuantizedQueryLut) -> bool {
    let _ = OsrHeader::from_bytes(data);
    let _ = cast_doc_token_ranges(data, 16);
    let _ = OsrStore::from_bytes(data);

    let Ok(aligned) = AlignedBuffer::<u8>::from_slice(data) else {
        return false;
    };
    let slice = aligned.as_slice();
    let _ = OsrHeader::from_bytes(slice);
    match OsrStore::from_bytes(slice) {
        Ok(store) => {
            let _ = store.all_tokens();
            let _ = store.lookup_doc(100);
            for doc_idx in 0..store.doc_table().len().min(16) {
                let _ = store.score_document_maxsim(doc_idx, q_lut, &ScalarRabitqKernel);
            }
            true
        }
        Err(_) => false,
    }
}

#[test]
#[cfg_attr(miri, ignore)] // Runs 50,000+ hostile binary parser mutations
fn fuzz_osi_parser_50000_hostile_mutations_zero_panics() {
    let seed = build_seed_osi();
    let mut rng = SplitMix64::new(0xFA22_0510_5000);
    let mut total_cases = 0usize;

    // 1. Every prefix truncation & unaligned tail slice of the valid `.osi` image.
    for cut in 0..seed.len() {
        assert!(!run_osi_target_once(&seed[..cut]));
        if cut > 0 {
            assert!(!run_osi_target_once(&seed[cut..]));
        }
        total_cases += 2;
    }

    // 2. Explicit adversarial u64::MAX / u32::MAX / u16::MAX field mutations (unsealed & sealed).
    let header_offsets = [0x08usize, 0x0A, 0x0C, 0x10, 0x20, 0x28, 0x30];
    let extreme_vals = [0_u64, 1, 63, 64, u64::from(u32::MAX), u64::MAX];
    for &off in &header_offsets {
        for &val in &extreme_vals {
            let mut mutated = seed.clone();
            if off == 0x08 || off == 0x0A {
                let v16 = u16::try_from(val).unwrap_or(u16::MAX);
                mutated[off..off + 2].copy_from_slice(&v16.to_le_bytes());
            } else if off == 0x0C {
                let v32 = u32::try_from(val).unwrap_or(u32::MAX);
                mutated[off..off + 4].copy_from_slice(&v32.to_le_bytes());
            } else {
                mutated[off..off + 8].copy_from_slice(&val.to_le_bytes());
            }
            if mutated == seed {
                continue;
            }
            assert!(!run_osi_target_once(&mutated));
            reseal_osi_or_osr(&mut mutated);
            if val == 0 || val >= u64::from(u32::MAX) {
                assert!(!run_osi_target_once(&mutated));
            } else {
                let _ = run_osi_target_once(&mutated);
            }
            total_cases += 2;
        }
    }

    // 3. Overlapping / u64::MAX cluster spans & corrupted tombstone length (sealed).
    {
        let aligned_seed = AlignedBuffer::<u8>::from_slice(&seed).unwrap();
        let hdr = *OsiHeader::from_bytes(aligned_seed.as_slice()).unwrap();
        let ent_off = usize::try_from(hdr.entries_offset).unwrap();
        let tomb_off = usize::try_from(hdr.tombstones_offset).unwrap();

        for bad in [0_u64, 64, u64::MAX, u64::MAX - 63] {
            let mut m = seed.clone();
            // Corrupt first OsiClusterSpan.cluster_offset or vector_count.
            m[ent_off..ent_off + 8].copy_from_slice(&bad.to_le_bytes());
            m[ent_off + 8..ent_off + 12].copy_from_slice(&u32::MAX.to_le_bytes());
            reseal_osi_or_osr(&mut m);
            assert!(!run_osi_target_once(&m));

            // Corrupt tombstone_len to u64::MAX.
            let mut m2 = seed.clone();
            m2[tomb_off..tomb_off + 8].copy_from_slice(&bad.to_le_bytes());
            reseal_osi_or_osr(&mut m2);
            let _ = run_osi_target_once(&m2);
            total_cases += 2;
        }
    }

    // 4. Random noise + bit-flip / splice mutations up to 50,000 total cases.
    while total_cases < 50_000 {
        let mode = rng.next_u64() % 4;
        match mode {
            0 => {
                // Pure random bytes (0..=1536 bytes).
                let len = usize::try_from(rng.next_u64() % 1537).unwrap();
                let mut buf = vec![0_u8; len];
                for chunk in buf.chunks_mut(8) {
                    let w = rng.next_u64().to_le_bytes();
                    chunk.copy_from_slice(&w[..chunk.len()]);
                }
                let _ = run_osi_target_once(&buf);
            }
            1 => {
                // Bit-flip(s) in valid `.osi` (both unsealed & sealed).
                let mut m = seed.clone();
                let flips = usize::try_from((rng.next_u64() % 4) + 1).unwrap();
                for _ in 0..flips {
                    let pos = usize::try_from(rng.next_u64() % (m.len() as u64)).unwrap();
                    let bit = u8::try_from(rng.next_u64() % 8).unwrap();
                    m[pos] ^= 1_u8 << bit;
                }
                let _ = run_osi_target_once(&m);
                if (rng.next_u64() & 1) == 0 {
                    reseal_osi_or_osr(&mut m);
                    let _ = run_osi_target_once(&m);
                }
            }
            2 => {
                // Random u64::MAX / u32::MAX injection inside header or cluster spans.
                let mut m = seed.clone();
                let pos = usize::try_from(rng.next_u64() % ((m.len() - 8) as u64)).unwrap();
                let extreme = if (rng.next_u64() & 1) == 0 {
                    u64::MAX
                } else {
                    u64::from(u32::MAX)
                };
                m[pos..pos + 8].copy_from_slice(&extreme.to_le_bytes());
                reseal_osi_or_osr(&mut m);
                let _ = run_osi_target_once(&m);
            }
            _ => {
                // Random truncation + tail corruption.
                let cut = usize::try_from(rng.next_u64() % (seed.len() as u64)).unwrap();
                let mut m = seed[..cut].to_vec();
                if !m.is_empty() {
                    let pos = usize::try_from(rng.next_u64() % (m.len() as u64)).unwrap();
                    m[pos] = u8::try_from(rng.next_u64() & 0xFF).unwrap();
                }
                let _ = run_osi_target_once(&m);
            }
        }
        total_cases += 1;
    }
}

#[test]
#[cfg_attr(miri, ignore)] // Runs 50,000+ hostile binary parser mutations
fn fuzz_osr_parser_50000_hostile_mutations_zero_panics() {
    let (seed, q_lut) = build_seed_osr();
    let mut rng = SplitMix64::new(0xFA22_0520_5000);
    let mut total_cases = 0usize;

    // 1. Every prefix truncation & unaligned tail slice of the valid `.osr` image.
    for cut in 0..seed.len() {
        assert!(!run_osr_target_once(&seed[..cut], &q_lut));
        if cut > 0 {
            assert!(!run_osr_target_once(&seed[cut..], &q_lut));
        }
        total_cases += 2;
    }

    // 2. Adversarial DocTokenRange OOB (token_start_idx / token_count = u64::MAX / u16::MAX)
    //    and OsrHeader overflow values (both unsealed & sealed).
    let header_u64_offsets = [0x18usize, 0x20, 0x28, 0x30];
    for &off in &header_u64_offsets {
        for val in [0_u64, 63, 64, u64::from(u32::MAX), u64::MAX] {
            let mut m = seed.clone();
            m[off..off + 8].copy_from_slice(&val.to_le_bytes());
            if m == seed {
                continue;
            }
            assert!(!run_osr_target_once(&m, &q_lut));
            reseal_osi_or_osr(&mut m);
            let _ = run_osr_target_once(&m, &q_lut);
            total_cases += 2;
        }
    }

    // DocTokenRange[0] starts at byte offset 64: mutate token_start_idx & token_count OOB.
    for bad_start in [u64::MAX, u64::MAX - 1, 999_999] {
        for bad_count in [0_u16, 1, u16::MAX] {
            let mut m = seed.clone();
            m[64 + 8..64 + 16].copy_from_slice(&bad_start.to_le_bytes());
            m[64 + 16..64 + 18].copy_from_slice(&bad_count.to_le_bytes());
            reseal_osi_or_osr(&mut m);
            assert!(!run_osr_target_once(&m, &q_lut));
            total_cases += 1;
        }
    }

    // 3. Random noise, truncated nibbles, and bit-flips up to 50,000 total cases.
    while total_cases < 50_000 {
        let mode = rng.next_u64() % 4;
        match mode {
            0 => {
                let len = usize::try_from(rng.next_u64() % 1537).unwrap();
                let mut buf = vec![0_u8; len];
                for chunk in buf.chunks_mut(8) {
                    let w = rng.next_u64().to_le_bytes();
                    chunk.copy_from_slice(&w[..chunk.len()]);
                }
                let _ = run_osr_target_once(&buf, &q_lut);
            }
            1 => {
                let mut m = seed.clone();
                let flips = usize::try_from((rng.next_u64() % 4) + 1).unwrap();
                for _ in 0..flips {
                    let pos = usize::try_from(rng.next_u64() % (m.len() as u64)).unwrap();
                    let bit = u8::try_from(rng.next_u64() % 8).unwrap();
                    m[pos] ^= 1_u8 << bit;
                }
                let _ = run_osr_target_once(&m, &q_lut);
                if (rng.next_u64() & 1) == 0 {
                    reseal_osi_or_osr(&mut m);
                    let _ = run_osr_target_once(&m, &q_lut);
                }
            }
            2 => {
                // Corrupt DocTokenRange entries with extreme offsets/counts + reseal.
                let mut m = seed.clone();
                let pos = 64 + usize::try_from(rng.next_u64() % 64).unwrap();
                if pos + 8 <= m.len() {
                    m[pos..pos + 8].copy_from_slice(&rng.next_u64().to_le_bytes());
                }
                reseal_osi_or_osr(&mut m);
                let _ = run_osr_target_once(&m, &q_lut);
            }
            _ => {
                // Truncated nibble streams (keep header + doc_table, chop middle of tensors).
                let cut = usize::try_from(rng.next_u64() % (seed.len() as u64)).unwrap();
                let mut m = seed[..cut].to_vec();
                reseal_osi_or_osr(&mut m);
                let _ = run_osr_target_once(&m, &q_lut);
            }
        }
        total_cases += 1;
    }
}
