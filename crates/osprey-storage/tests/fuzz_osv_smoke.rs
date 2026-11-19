//! Deterministic 50,000+ hostile-mutation smoke fuzzer for `.osv`
//! integrated into `cargo test`.
//!
//! Exercises:
//! - Arbitrary random bytes (aligned and unaligned);
//! - Prefix truncations at every single byte offset of a valid `.osv` slab with Zstd dict;
//! - Adversarial `u64::MAX` / `u32::MAX` / `0` mutations on `OsvHeader` (`directory_offset`,
//!   `zstd_dict_offset`, `zstd_dict_len`, `num_minipages`) and `MinipageDirectoryEntry`
//!   (`byte_offset`, `compressed_size`, `uncompressed_size`), both unsealed and re-sealed;
//! - Truncated / corrupted Zstd frames (both before `page_xxh64` and with re-sealed `page_xxh64`).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use half::f16;
use osprey_core::math::SplitMix64;
use osprey_core::{AlignedBuffer, OsvHeader};
use osprey_storage::{OsvSlabReader, OsvSlabView, OsvVectorRecord, OsvWriter, OsvWriterConfig};
use xxhash_rust::xxh64::xxh64;

fn build_seed_osv() -> Vec<u8> {
    let orig_dim = 16usize;
    let mut vecs = Vec::new();
    let mut payloads = Vec::new();
    for i in 0..24usize {
        let v: Vec<f16> = (0..orig_dim)
            .map(|d| f16::from_f32(((i * orig_dim + d) as f32) * 0.01))
            .collect();
        let p = format!(
            "{{\"doc\":{i},\"tenant\":\"fuzz-osv\",\"cluster\":{}}}",
            i / 4
        )
        .into_bytes();
        vecs.push(v);
        payloads.push(p);
    }
    let records: Vec<OsvVectorRecord<'_>> = (0..24usize)
        .map(|i| OsvVectorRecord {
            doc_id: 1_000 + i as u64,
            cluster_id: u32::try_from(i / 4).unwrap(),
            vector_f16: &vecs[i],
            payload: &payloads[i],
        })
        .collect();

    let cfg = OsvWriterConfig {
        minipage_target_bytes: 512,
        zstd_level: 3,
        enable_zstd_dict: true,
        max_dict_bytes: 8 * 1024,
    };
    let mut raw_slab = Vec::new();
    OsvWriter::new(cfg)
        .write_slab(&mut raw_slab, &records, orig_dim, None)
        .unwrap();
    raw_slab
}

#[inline]
fn reseal_osv_header(buf: &mut [u8]) {
    if buf.len() >= 64 {
        let hdr_hash = xxh64(&buf[..56], 0);
        buf[56..64].copy_from_slice(&hdr_hash.to_le_bytes());
    }
}

fn run_osv_target_once(data: &[u8]) -> bool {
    let _ = OsvHeader::from_bytes(data);
    let _ = OsvSlabView::from_bytes(data);

    let Ok(aligned) = AlignedBuffer::<u8>::from_slice(data) else {
        return false;
    };
    if let Ok(view) = OsvSlabView::from_bytes(aligned.as_slice()) {
        for id in 0..view.directory.len().min(16) {
            let _ = view.read_minipage(id);
        }
    }
    match OsvSlabReader::from_bytes(aligned) {
        Ok(reader) => {
            let mut all_pages_ok = !reader.directory().is_empty();
            for id in 0..reader.directory().len().min(16) {
                match reader.decompress_minipage(u32::try_from(id).unwrap()) {
                    Ok(page) => {
                        let _ = page.doc_ids();
                        if page.vector_count() > 0 {
                            let _ = page.vector_f16(0);
                            let _ = page.payload_bytes(0);
                        }
                    }
                    Err(_) => {
                        all_pages_ok = false;
                    }
                }
            }
            all_pages_ok
        }
        Err(_) => false,
    }
}

#[test]
#[cfg_attr(miri, ignore)] // Runs 50,000+ hostile binary parser mutations
fn fuzz_osv_parser_50000_hostile_mutations_zero_panics() {
    let seed = build_seed_osv();
    let mut rng = SplitMix64::new(0xFA22_0530_5000);
    let mut total_cases = 0usize;

    // 1. Every prefix truncation & unaligned tail slice of the valid `.osv` slab.
    for cut in 0..seed.len() {
        let _ = run_osv_target_once(&seed[..cut]);
        if cut > 0 {
            let _ = run_osv_target_once(&seed[cut..]);
        }
        total_cases += 2;
    }

    // 2. Adversarial u64::MAX / u32::MAX / 0 header offset & size mutations (unsealed & sealed).
    let header_offsets = [0x08usize, 0x0C, 0x10, 0x18, 0x20, 0x28];
    let extreme_vals = [0_u64, 1, 63, 64, u64::from(u32::MAX), u64::MAX];
    for &off in &header_offsets {
        for &val in &extreme_vals {
            let mut m = seed.clone();
            if off == 0x08 {
                let v16 = u16::try_from(val).unwrap_or(u16::MAX);
                m[off..off + 2].copy_from_slice(&v16.to_le_bytes());
            } else if off == 0x0C {
                let v32 = u32::try_from(val).unwrap_or(u32::MAX);
                m[off..off + 4].copy_from_slice(&v32.to_le_bytes());
            } else {
                m[off..off + 8].copy_from_slice(&val.to_le_bytes());
            }
            let _ = run_osv_target_once(&m);
            reseal_osv_header(&mut m);
            let _ = run_osv_target_once(&m);
            total_cases += 2;
        }
    }

    // 3. Adversarial MinipageDirectoryEntry mutations (byte_offset = u64::MAX,
    //    compressed_size = u32::MAX, uncompressed_size = u32::MAX, plus truncated Zstd with valid page_xxh64).
    let reader = OsvSlabReader::from_slice(&seed).unwrap();
    let dir_off = usize::try_from(reader.header().directory_offset).unwrap();
    let entry0 = reader.directory()[0];
    let p0_start = usize::try_from(entry0.byte_offset).unwrap();
    let p0_len = usize::try_from(entry0.compressed_size).unwrap();

    for bad_off in [0_u64, 63, u64::MAX, u64::MAX - 8] {
        for bad_size in [0_u32, 1, u32::MAX] {
            let mut m = seed.clone();
            m[dir_off..dir_off + 8].copy_from_slice(&bad_off.to_le_bytes());
            m[dir_off + 0x20..dir_off + 0x24].copy_from_slice(&bad_size.to_le_bytes());
            m[dir_off + 0x24..dir_off + 0x28].copy_from_slice(&bad_size.to_le_bytes());
            assert!(!run_osv_target_once(&m));
            total_cases += 1;
        }
    }

    // Truncated & corrupted Zstd frames where page_xxh64 IS recomputed to match the garbage frame.
    for trunc_len in [0_usize, 1, 2, 4, p0_len / 2, p0_len.saturating_sub(1)] {
        if p0_start + trunc_len <= seed.len() {
            let mut m = seed.clone();
            if trunc_len > 0 {
                m[p0_start + trunc_len - 1] ^= 0xFF;
            }
            let new_hash = xxh64(&m[p0_start..p0_start + trunc_len], 0);
            m[dir_off + 0x18..dir_off + 0x20].copy_from_slice(&new_hash.to_le_bytes());
            m[dir_off + 0x20..dir_off + 0x24]
                .copy_from_slice(&u32::try_from(trunc_len).unwrap().to_le_bytes());
            assert!(!run_osv_target_once(&m));
            total_cases += 1;
        }
    }

    // 4. Random noise + bit-flips up to 50,000 total cases.
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
                let _ = run_osv_target_once(&buf);
            }
            1 => {
                let mut m = seed.clone();
                let flips = usize::try_from((rng.next_u64() % 4) + 1).unwrap();
                for _ in 0..flips {
                    let pos = usize::try_from(rng.next_u64() % (m.len() as u64)).unwrap();
                    let bit = u8::try_from(rng.next_u64() % 8).unwrap();
                    m[pos] ^= 1_u8 << bit;
                }
                let _ = run_osv_target_once(&m);
                if (rng.next_u64() & 1) == 0 {
                    reseal_osv_header(&mut m);
                    let _ = run_osv_target_once(&m);
                }
            }
            2 => {
                let mut m = seed.clone();
                let pos = usize::try_from(rng.next_u64() % ((m.len() - 8) as u64)).unwrap();
                let extreme = if (rng.next_u64() & 1) == 0 {
                    u64::MAX
                } else {
                    u64::from(u32::MAX)
                };
                m[pos..pos + 8].copy_from_slice(&extreme.to_le_bytes());
                reseal_osv_header(&mut m);
                let _ = run_osv_target_once(&m);
            }
            _ => {
                let cut = usize::try_from(rng.next_u64() % (seed.len() as u64)).unwrap();
                let mut m = seed[..cut].to_vec();
                reseal_osv_header(&mut m);
                let _ = run_osv_target_once(&m);
            }
        }
        total_cases += 1;
    }
}
