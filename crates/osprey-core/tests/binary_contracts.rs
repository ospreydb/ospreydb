//! Exhaustive Phase 3 binary contract, alignment, little-endian, and 1-bit corruption tests.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::mem::{align_of, offset_of, size_of};

use half::f16;
use osprey_core::{
    deserialize_tombstone_footer, ensure_aligned_64, serialize_tombstone_footer, AlignedBuffer,
    DocTokenRange, MinipageDirectoryEntry, OsiHeader, OspreyError, OsrHeader, OsvHeader,
    RabitqEntryHeader, CACHE_LINE_ALIGN, FORMAT_KIND_OSI, FORMAT_KIND_OSR, FORMAT_KIND_OSV,
    FORMAT_VERSION_V1, OSPREY_MAGIC,
};
use roaring::RoaringBitmap;
use xxhash_rust::xxh64::xxh64;

const _: () = {
    assert!(size_of::<OsiHeader>() == 64);
    assert!(align_of::<OsiHeader>() == 64);
    assert!(offset_of!(OsiHeader, tombstones_offset) == 0x30);
    assert!(offset_of!(OsiHeader, header_xxh64) == 0x38);

    assert!(size_of::<RabitqEntryHeader>() == 24);
    assert!(align_of::<RabitqEntryHeader>() == 8);
    assert!(offset_of!(RabitqEntryHeader, res_norm) == 0x10);
    assert!(offset_of!(RabitqEntryHeader, l1_norm_factor) == 0x12);
    assert!(offset_of!(RabitqEntryHeader, _reserved) == 0x14);

    assert!(size_of::<OsvHeader>() == 64);
    assert!(align_of::<OsvHeader>() == 64);
    assert!(offset_of!(OsvHeader, zstd_dict_offset) == 0x18);
    assert!(offset_of!(OsvHeader, directory_offset) == 0x28);
    assert!(offset_of!(OsvHeader, header_xxh64) == 0x38);

    assert!(size_of::<MinipageDirectoryEntry>() == 48);
    assert!(align_of::<MinipageDirectoryEntry>() == 8);
    assert!(offset_of!(MinipageDirectoryEntry, page_xxh64) == 0x18);
    assert!(offset_of!(MinipageDirectoryEntry, flags) == 0x2E);

    assert!(size_of::<OsrHeader>() == 64);
    assert!(align_of::<OsrHeader>() == 64);
    assert!(offset_of!(OsrHeader, gamma) == 0x0C);
    assert!(offset_of!(OsrHeader, tensors_offset) == 0x30);
    assert!(offset_of!(OsrHeader, header_xxh64) == 0x38);

    assert!(size_of::<DocTokenRange>() == 24);
    assert!(align_of::<DocTokenRange>() == 8);
    assert!(offset_of!(DocTokenRange, token_count) == 0x10);
    assert!(offset_of!(DocTokenRange, _reserved) == 0x12);
};

#[test]
fn runtime_abi_sizes_alignments_and_offsets() {
    assert_eq!(size_of::<OsiHeader>(), 64);
    assert_eq!(align_of::<OsiHeader>(), 64);
    assert_eq!(offset_of!(OsiHeader, magic), 0x00);
    assert_eq!(offset_of!(OsiHeader, format_kind), 0x04);
    assert_eq!(offset_of!(OsiHeader, version), 0x06);
    assert_eq!(offset_of!(OsiHeader, orig_dim), 0x08);
    assert_eq!(offset_of!(OsiHeader, padded_dim), 0x0A);
    assert_eq!(offset_of!(OsiHeader, num_clusters), 0x0C);
    assert_eq!(offset_of!(OsiHeader, num_vectors), 0x10);
    assert_eq!(offset_of!(OsiHeader, fwht_seed), 0x18);
    assert_eq!(offset_of!(OsiHeader, centroids_offset), 0x20);
    assert_eq!(offset_of!(OsiHeader, entries_offset), 0x28);
    assert_eq!(offset_of!(OsiHeader, tombstones_offset), 0x30);
    assert_eq!(offset_of!(OsiHeader, header_xxh64), 0x38);

    assert_eq!(size_of::<RabitqEntryHeader>(), 24);
    assert_eq!(align_of::<RabitqEntryHeader>(), 8);
    assert_eq!(offset_of!(RabitqEntryHeader, doc_id), 0x00);
    assert_eq!(offset_of!(RabitqEntryHeader, minipage_id), 0x08);
    assert_eq!(offset_of!(RabitqEntryHeader, slab_id), 0x0C);
    assert_eq!(offset_of!(RabitqEntryHeader, slot_idx), 0x0E);
    assert_eq!(offset_of!(RabitqEntryHeader, res_norm), 0x10);
    assert_eq!(offset_of!(RabitqEntryHeader, l1_norm_factor), 0x12);
    assert_eq!(offset_of!(RabitqEntryHeader, _reserved), 0x14);

    assert_eq!(size_of::<OsvHeader>(), 64);
    assert_eq!(align_of::<OsvHeader>(), 64);
    assert_eq!(offset_of!(OsvHeader, zstd_level), 0x05);
    assert_eq!(offset_of!(OsvHeader, num_minipages), 0x0C);
    assert_eq!(offset_of!(OsvHeader, total_vectors), 0x10);
    assert_eq!(offset_of!(OsvHeader, zstd_dict_offset), 0x18);
    assert_eq!(offset_of!(OsvHeader, zstd_dict_len), 0x20);
    assert_eq!(offset_of!(OsvHeader, directory_offset), 0x28);
    assert_eq!(offset_of!(OsvHeader, _reserved), 0x30);
    assert_eq!(offset_of!(OsvHeader, header_xxh64), 0x38);

    assert_eq!(size_of::<MinipageDirectoryEntry>(), 48);
    assert_eq!(align_of::<MinipageDirectoryEntry>(), 8);
    assert_eq!(offset_of!(MinipageDirectoryEntry, byte_offset), 0x00);
    assert_eq!(offset_of!(MinipageDirectoryEntry, min_doc_id), 0x08);
    assert_eq!(offset_of!(MinipageDirectoryEntry, max_doc_id), 0x10);
    assert_eq!(offset_of!(MinipageDirectoryEntry, page_xxh64), 0x18);
    assert_eq!(offset_of!(MinipageDirectoryEntry, compressed_size), 0x20);
    assert_eq!(offset_of!(MinipageDirectoryEntry, uncompressed_size), 0x24);
    assert_eq!(offset_of!(MinipageDirectoryEntry, cluster_id), 0x28);
    assert_eq!(offset_of!(MinipageDirectoryEntry, vector_count), 0x2C);
    assert_eq!(offset_of!(MinipageDirectoryEntry, flags), 0x2E);

    assert_eq!(size_of::<OsrHeader>(), 64);
    assert_eq!(align_of::<OsrHeader>(), 64);
    assert_eq!(offset_of!(OsrHeader, subspace_dim), 0x05);
    assert_eq!(offset_of!(OsrHeader, num_layers), 0x06);
    assert_eq!(offset_of!(OsrHeader, version), 0x08);
    assert_eq!(offset_of!(OsrHeader, token_dim), 0x0A);
    assert_eq!(offset_of!(OsrHeader, gamma), 0x0C);
    assert_eq!(offset_of!(OsrHeader, fwht_seed), 0x10);
    assert_eq!(offset_of!(OsrHeader, num_docs), 0x18);
    assert_eq!(offset_of!(OsrHeader, total_tokens), 0x20);
    assert_eq!(offset_of!(OsrHeader, doc_table_offset), 0x28);
    assert_eq!(offset_of!(OsrHeader, tensors_offset), 0x30);
    assert_eq!(offset_of!(OsrHeader, header_xxh64), 0x38);

    assert_eq!(size_of::<DocTokenRange>(), 24);
    assert_eq!(align_of::<DocTokenRange>(), 8);
    assert_eq!(offset_of!(DocTokenRange, doc_id), 0x00);
    assert_eq!(offset_of!(DocTokenRange, token_start_idx), 0x08);
    assert_eq!(offset_of!(DocTokenRange, token_count), 0x10);
    assert_eq!(offset_of!(DocTokenRange, _reserved), 0x12);
}

#[test]
fn aligned_buffer_all_capacities_and_unaligned_rejection() {
    for cap in [0usize, 1, 63, 64, 1024, 4096] {
        let b_u8 = AlignedBuffer::<u8>::new(cap).unwrap();
        assert_eq!((b_u8.as_ptr() as usize) % CACHE_LINE_ALIGN, 0);
        assert_eq!(b_u8.len(), cap);

        let b_u64 = AlignedBuffer::<u64>::new(cap).unwrap();
        assert_eq!((b_u64.as_ptr() as usize) % CACHE_LINE_ALIGN, 0);
        assert_eq!(b_u64.len(), cap);

        let b_osi = AlignedBuffer::<OsiHeader>::new(cap).unwrap();
        assert_eq!((b_osi.as_ptr() as usize) % CACHE_LINE_ALIGN, 0);
        assert_eq!(b_osi.len(), cap);

        if cap > 1 {
            let err = ensure_aligned_64(&b_u8[1..]).unwrap_err();
            assert!(matches!(
                err,
                OspreyError::UnalignedMemory {
                    required_align: 64,
                    ..
                }
            ));
        }
    }
}

#[test]
fn little_endian_byte_layout_and_bytemuck_roundtrip() {
    let osi = OsiHeader::new_sealed(
        0x03,
        384,
        512,
        0x0102_0304,
        0x1122_3344_5566_7788,
        0xAABB_CCDD_EEFF_0011,
        64,
        65600,
        131_072,
    )
    .unwrap();
    let osi_raw = bytemuck::bytes_of(&osi);
    assert_eq!(&osi_raw[0..4], &[0x4F, 0x53, 0x50, 0x52]);
    assert_eq!(osi_raw[4], FORMAT_KIND_OSI);
    assert_eq!(osi_raw[5], 0x03);
    assert_eq!(&osi_raw[6..8], &FORMAT_VERSION_V1.to_le_bytes());
    assert_eq!(&osi_raw[0x0C..0x10], &[0x04, 0x03, 0x02, 0x01]);
    assert_eq!(
        &osi_raw[0x10..0x18],
        &[0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11]
    );
    let osi_buf = AlignedBuffer::from_slice(osi_raw).unwrap();
    let osi_rt: &OsiHeader = bytemuck::try_from_bytes(&osi_buf).unwrap();
    assert_eq!(osi_rt, &osi);

    let osv = OsvHeader::new_sealed(5, 1536, 0x0003, 42, 999_999, 64, 8192, 65_600).unwrap();
    let osv_raw = bytemuck::bytes_of(&osv);
    assert_eq!(&osv_raw[0..4], &OSPREY_MAGIC);
    assert_eq!(osv_raw[4], FORMAT_KIND_OSV);
    assert_eq!(osv_raw[5], 5);
    let osv_buf = AlignedBuffer::from_slice(osv_raw).unwrap();
    let osv_rt: &OsvHeader = bytemuck::try_from_bytes(&osv_buf).unwrap();
    assert_eq!(osv_rt, &osv);

    let osr = OsrHeader::new_sealed(2, 2, 128, 0.28, 0xCAFE_BABE, 100, 3200, 64, 4096).unwrap();
    let osr_raw = bytemuck::bytes_of(&osr);
    assert_eq!(&osr_raw[0..4], &[0x4F, 0x53, 0x50, 0x52]);
    assert_eq!(osr_raw[4], FORMAT_KIND_OSR);
    assert_eq!(&osr_raw[0x0C..0x10], &0.28_f32.to_le_bytes());
    let osr_buf = AlignedBuffer::from_slice(osr_raw).unwrap();
    let osr_rt: &OsrHeader = bytemuck::try_from_bytes(&osr_buf).unwrap();
    assert_eq!(osr_rt, &osr);

    let rab = RabitqEntryHeader {
        doc_id: 0x0102_0304_0506_0708,
        minipage_id: 0x0A0B_0C0D,
        slab_id: 0x1122,
        slot_idx: 0x3344,
        res_norm: f16::from_f32(1.25),
        l1_norm_factor: f16::from_f32(0.5),
        _reserved: 0,
    };
    let rab_buf = AlignedBuffer::from_slice(bytemuck::bytes_of(&rab)).unwrap();
    assert_eq!(
        &rab_buf[0..8],
        &[0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]
    );
    let rab_rt: &RabitqEntryHeader = bytemuck::try_from_bytes(&rab_buf).unwrap();
    assert_eq!(rab_rt, &rab);

    let mp = MinipageDirectoryEntry {
        byte_offset: 0x1000,
        min_doc_id: 1,
        max_doc_id: 250,
        page_xxh64: 0x9988_7766_5544_3322,
        compressed_size: 4096,
        uncompressed_size: 16384,
        cluster_id: 7,
        vector_count: 250,
        flags: 1,
    };
    let mp_buf = AlignedBuffer::from_slice(bytemuck::bytes_of(&mp)).unwrap();
    let mp_rt: &MinipageDirectoryEntry = bytemuck::try_from_bytes(&mp_buf).unwrap();
    assert_eq!(mp_rt, &mp);

    let tr = DocTokenRange {
        doc_id: 777,
        token_start_idx: 1024,
        token_count: 96,
        _reserved: [0; 6],
    };
    let tr_buf = AlignedBuffer::from_slice(bytemuck::bytes_of(&tr)).unwrap();
    let tr_rt: &DocTokenRange = bytemuck::try_from_bytes(&tr_buf).unwrap();
    assert_eq!(tr_rt, &tr);
}

#[test]
fn exhaustive_single_bit_corruption_detection_all_formats() {
    // 1. .osi header: flip all 512 bits
    let osi = OsiHeader::new_sealed(0, 384, 512, 64, 8192, 42, 64, 65600, 131_072).unwrap();
    let base_osi = AlignedBuffer::from_slice(bytemuck::bytes_of(&osi)).unwrap();
    for byte_idx in 0..64 {
        for bit in 0..8 {
            let mut mutated = AlignedBuffer::from_slice(&base_osi).unwrap();
            mutated[byte_idx] ^= 1u8 << bit;
            let err = OsiHeader::from_bytes(&mutated).unwrap_err();
            match byte_idx {
                0..=3 => assert!(matches!(err, OspreyError::InvalidMagic { .. })),
                4 => assert!(matches!(err, OspreyError::InvalidFormatKind { .. })),
                6..=7 => assert!(matches!(err, OspreyError::UnsupportedVersion { .. })),
                _ => assert!(matches!(err, OspreyError::ChecksumMismatch { .. })),
            }
        }
    }

    // 2. .osv header: flip all 512 bits
    let osv = OsvHeader::new_sealed(3, 1536, 1, 10, 500, 64, 1024, 65600).unwrap();
    let base_osv = AlignedBuffer::from_slice(bytemuck::bytes_of(&osv)).unwrap();
    for byte_idx in 0..64 {
        for bit in 0..8 {
            let mut mutated = AlignedBuffer::from_slice(&base_osv).unwrap();
            mutated[byte_idx] ^= 1u8 << bit;
            let err = OsvHeader::from_bytes(&mutated).unwrap_err();
            match byte_idx {
                0..=3 => assert!(matches!(err, OspreyError::InvalidMagic { .. })),
                4 => assert!(matches!(err, OspreyError::InvalidFormatKind { .. })),
                6..=7 => assert!(matches!(err, OspreyError::UnsupportedVersion { .. })),
                _ => assert!(matches!(err, OspreyError::ChecksumMismatch { .. })),
            }
        }
    }

    // 3. .osr header: flip all 512 bits
    let osr = OsrHeader::new_sealed(2, 2, 128, 0.28, 77, 16, 512, 64, 512).unwrap();
    let base_osr = AlignedBuffer::from_slice(bytemuck::bytes_of(&osr)).unwrap();
    for byte_idx in 0..64 {
        for bit in 0..8 {
            let mut mutated = AlignedBuffer::from_slice(&base_osr).unwrap();
            mutated[byte_idx] ^= 1u8 << bit;
            let err = OsrHeader::from_bytes(&mutated).unwrap_err();
            match byte_idx {
                0..=3 => assert!(matches!(err, OspreyError::InvalidMagic { .. })),
                4 => assert!(matches!(err, OspreyError::InvalidFormatKind { .. })),
                8..=9 => assert!(matches!(err, OspreyError::UnsupportedVersion { .. })),
                _ => assert!(matches!(err, OspreyError::ChecksumMismatch { .. })),
            }
        }
    }

    // 4. .osv page_xxh64: flip every bit of compressed minipage payload
    let page_payload = b"osprey-compressed-minipage-frame-payload-0123456789abcdef";
    let expected_page_hash = xxh64(page_payload, 0);
    for byte_idx in 0..page_payload.len() {
        for bit in 0..8 {
            let mut mutated = page_payload.to_vec();
            mutated[byte_idx] ^= 1u8 << bit;
            assert_ne!(xxh64(&mutated, 0), expected_page_hash);
        }
    }

    // 5. .osi RoaringBitmap tombstone footer: flip every single bit
    let mut bm = RoaringBitmap::new();
    bm.insert_range(100..150);
    bm.insert(987_654);
    let mut footer = Vec::new();
    serialize_tombstone_footer(&bm, &mut footer).unwrap();
    for byte_idx in 0..footer.len() {
        for bit in 0..8 {
            let mut mutated = footer.clone();
            mutated[byte_idx] ^= 1u8 << bit;
            let err = deserialize_tombstone_footer(&mutated).unwrap_err();
            assert!(matches!(
                err,
                OspreyError::ChecksumMismatch { .. }
                    | OspreyError::BufferTooSmall { .. }
                    | OspreyError::CorruptedFooter
            ));
        }
    }
}
