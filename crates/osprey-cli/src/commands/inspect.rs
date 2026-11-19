//! `osprey inspect`: inspects `.osi`, `.osv`, and `.osr` headers, exact
//! compression ratios, centroid/cluster counts, minipage size distributions, tombstones,
//! and `xxHash64` integrity checks.

use std::fs::File;
use std::path::Path;

use comfy_table::{Cell, CellAlignment};
use memmap2::Mmap;
use osprey_core::format::osv::MINIPAGE_FLAG_OVERSIZED;
use osprey_core::{
    OsiMappedView, OspreyError, OsrStore, FORMAT_KIND_OSI, FORMAT_KIND_OSR, FORMAT_KIND_OSV,
    OSPREY_MAGIC, OSR_TOKEN_BYTES,
};
use osprey_simd::active_simd_backend;
use osprey_storage::OsvSlabReader;
use serde::Serialize;
use xxhash_rust::xxh64::xxh64;

use crate::cli::InspectArgs;
use crate::output::{
    format_bytes, new_rounded_table, paint_accent, paint_dim, paint_pass, paint_warn,
    print_indented_table, print_json, print_kv, print_section, CliContext, CliError, CliResult,
};

#[derive(Debug, Serialize)]
#[serde(tag = "format", rename_all = "lowercase")]
enum InspectedArtifact {
    Osi(OsiInspectReport),
    Osv(OsvInspectReport),
    Osr(OsrInspectReport),
}

#[derive(Debug, Serialize)]
struct OsiClusterSummary {
    cluster_id: u32,
    vector_count: u32,
    cluster_offset: u64,
}

#[derive(Debug, Serialize)]
struct OsiInspectReport {
    path: String,
    simd_backend: &'static str,
    file_size_bytes: u64,
    version: u16,
    flags: u8,
    orig_dim: u16,
    padded_dim: u16,
    num_centroids: u32,
    num_vectors: u64,
    fwht_seed: String,
    centroids_offset: u64,
    entries_offset: u64,
    tombstones_offset: u64,
    active_tombstones: u64,
    header_xxh64: String,
    file_xxh64: String,
    xxh64_status: String,
    clusters: Option<Vec<OsiClusterSummary>>,
}

#[derive(Debug, Serialize)]
struct SizeDistribution {
    min_bytes: u32,
    mean_bytes: f64,
    p50_bytes: u32,
    p95_bytes: u32,
    max_bytes: u32,
}

#[derive(Debug, Serialize)]
struct OsvPageSummary {
    minipage_id: usize,
    cluster_id: u32,
    vector_count: u16,
    compressed_size: u32,
    uncompressed_size: u32,
    byte_offset: u64,
    page_xxh64: String,
}

#[derive(Debug, Serialize)]
struct OsvInspectReport {
    path: String,
    simd_backend: &'static str,
    file_size_bytes: u64,
    version: u16,
    zstd_level: i8,
    orig_dim: u16,
    flags: u16,
    has_zstd_dict: bool,
    zstd_dict_len: u64,
    num_minipages: u32,
    oversized_minipages: usize,
    total_vectors: u64,
    total_uncompressed_bytes: u64,
    total_compressed_bytes: u64,
    compression_ratio: f64,
    space_savings_pct: f64,
    compressed_size_dist: SizeDistribution,
    uncompressed_size_dist: SizeDistribution,
    header_xxh64: String,
    pages_verified: usize,
    xxh64_status: String,
    directory: Option<Vec<OsvPageSummary>>,
}

#[derive(Debug, Serialize)]
struct OsrDocSummary {
    doc_id: u64,
    token_start_idx: u64,
    token_count: u16,
}

#[derive(Debug, Serialize)]
struct OsrInspectReport {
    path: String,
    simd_backend: &'static str,
    file_size_bytes: u64,
    version: u16,
    token_dim: u16,
    subspace_dim: u8,
    num_layers: u8,
    gamma: f32,
    fwht_seed: String,
    num_docs: u64,
    total_tokens: u64,
    avg_tokens_per_doc: f64,
    raw_fp32_bytes: u64,
    raw_fp16_bytes: u64,
    quantized_tensor_bytes: u64,
    compression_vs_fp32: f64,
    compression_vs_fp16: f64,
    header_xxh64: String,
    file_xxh64: String,
    xxh64_status: String,
    docs: Option<Vec<OsrDocSummary>>,
}

/// Executes `osprey inspect`.
pub fn run(args: &InspectArgs, ctx: &CliContext) -> CliResult<()> {
    let mut reports = Vec::with_capacity(args.file.len());
    for path in &args.file {
        ctx.log_info(format!(
            "Inspecting artifact '{}' (simd={})...",
            path.display(),
            active_simd_backend()
        ));
        reports.push(inspect_one(
            path,
            args.verify_checksums,
            args.dump_directory,
        )?);
    }

    if ctx.json {
        if reports.len() == 1 {
            print_json(&reports[0])?;
        } else {
            print_json(&reports)?;
        }
    } else {
        for report in &reports {
            match report {
                InspectedArtifact::Osi(r) => print_osi_human(r),
                InspectedArtifact::Osv(r) => print_osv_human(r),
                InspectedArtifact::Osr(r) => print_osr_human(r),
            }
        }
        println!();
    }

    Ok(())
}

fn inspect_one(
    path: &Path,
    verify_checksums: bool,
    dump_directory: bool,
) -> CliResult<InspectedArtifact> {
    let file = File::open(path)?;
    let file_size_bytes = file.metadata()?.len();
    // SAFETY: Artifacts are read-only during inspection and validated with `xxHash64`.
    let mmap = unsafe { Mmap::map(&file)? };
    let bytes: &[u8] = &mmap;

    if bytes.len() < 64 {
        return Err(CliError::Core(OspreyError::BufferTooSmall {
            needed: 64,
            actual: bytes.len(),
        }));
    }
    let magic: [u8; 4] = bytes[0..4]
        .try_into()
        .map_err(|_| CliError::Core(OspreyError::CorruptedFooter))?;
    if magic != OSPREY_MAGIC {
        return Err(CliError::Core(OspreyError::InvalidMagic { found: magic }));
    }

    match bytes[4] {
        FORMAT_KIND_OSI => inspect_osi(path, file_size_bytes, bytes, dump_directory),
        FORMAT_KIND_OSV => inspect_osv(
            path,
            file_size_bytes,
            bytes,
            verify_checksums,
            dump_directory,
        ),
        FORMAT_KIND_OSR => inspect_osr(path, file_size_bytes, bytes, dump_directory),
        other => Err(CliError::Core(OspreyError::InvalidFormatKind {
            found: other,
            expected: FORMAT_KIND_OSI,
        })),
    }
}

fn inspect_osi(
    path: &Path,
    file_size_bytes: u64,
    bytes: &[u8],
    dump_directory: bool,
) -> CliResult<InspectedArtifact> {
    let view = OsiMappedView::from_bytes(bytes)?;
    let hdr = view.header();

    let file_hash_bytes: [u8; 8] = bytes[bytes.len() - 8..]
        .try_into()
        .map_err(|_| CliError::Core(OspreyError::CorruptedFooter))?;
    let file_xxh64 = u64::from_le_bytes(file_hash_bytes);

    let clusters = if dump_directory {
        Some(
            view.cluster_spans()
                .iter()
                .enumerate()
                .map(|(cid, span)| OsiClusterSummary {
                    cluster_id: u32::try_from(cid).unwrap_or(u32::MAX),
                    vector_count: span.vector_count,
                    cluster_offset: span.cluster_offset,
                })
                .collect(),
        )
    } else {
        None
    };

    Ok(InspectedArtifact::Osi(OsiInspectReport {
        path: path.display().to_string(),
        simd_backend: active_simd_backend(),
        file_size_bytes,
        version: hdr.version,
        flags: hdr.flags,
        orig_dim: hdr.orig_dim,
        padded_dim: hdr.padded_dim,
        num_centroids: hdr.num_clusters,
        num_vectors: hdr.num_vectors,
        fwht_seed: format!("0x{:016x}", hdr.fwht_seed),
        centroids_offset: hdr.centroids_offset,
        entries_offset: hdr.entries_offset,
        tombstones_offset: hdr.tombstones_offset,
        active_tombstones: view.tombstones().len(),
        header_xxh64: format!("0x{:016x}", hdr.header_xxh64),
        file_xxh64: format!("0x{file_xxh64:016x}"),
        xxh64_status: "VERIFIED (header + tombstones + file_xxh64)".to_string(),
        clusters,
    }))
}

fn inspect_osv(
    path: &Path,
    file_size_bytes: u64,
    bytes: &[u8],
    verify_checksums: bool,
    dump_directory: bool,
) -> CliResult<InspectedArtifact> {
    let reader = OsvSlabReader::open_mmap(path)?;
    let hdr = *reader.header();
    let dir = reader.directory();

    let mut total_uncompressed = 0_u64;
    let mut total_compressed = 0_u64;
    let mut comp_sizes = Vec::with_capacity(dir.len());
    let mut uncomp_sizes = Vec::with_capacity(dir.len());
    let mut oversized_minipages = 0_usize;
    let mut pages_verified = 0_usize;

    for (idx, entry) in dir.iter().enumerate() {
        total_uncompressed += u64::from(entry.uncompressed_size);
        total_compressed += u64::from(entry.compressed_size);
        comp_sizes.push(entry.compressed_size);
        uncomp_sizes.push(entry.uncompressed_size);
        if (entry.flags & MINIPAGE_FLAG_OVERSIZED) != 0 {
            oversized_minipages += 1;
        }
        if verify_checksums {
            let start = usize::try_from(entry.byte_offset)
                .map_err(|_| CliError::Core(OspreyError::CorruptedFooter))?;
            let len = usize::try_from(entry.compressed_size)
                .map_err(|_| CliError::Core(OspreyError::CorruptedFooter))?;
            let end = start
                .checked_add(len)
                .ok_or(CliError::Core(OspreyError::CorruptedFooter))?;
            let slice =
                bytes
                    .get(start..end)
                    .ok_or(CliError::Core(OspreyError::BufferTooSmall {
                        needed: end,
                        actual: bytes.len(),
                    }))?;
            let computed = xxh64(slice, 0);
            if computed != entry.page_xxh64 {
                return Err(CliError::Core(OspreyError::ChecksumMismatch {
                    expected: entry.page_xxh64,
                    computed,
                }));
            }
            pages_verified = idx + 1;
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let compression_ratio = if total_compressed > 0 {
        (total_uncompressed as f64) / (total_compressed as f64)
    } else {
        1.0
    };
    #[allow(clippy::cast_precision_loss)]
    let space_savings_pct = if total_uncompressed > 0 {
        (1.0 - (total_compressed as f64) / (total_uncompressed as f64)) * 100.0
    } else {
        0.0
    };

    let compressed_size_dist = compute_size_distribution(&mut comp_sizes);
    let uncompressed_size_dist = compute_size_distribution(&mut uncomp_sizes);

    let directory = if dump_directory {
        Some(
            dir.iter()
                .enumerate()
                .map(|(minipage_id, e)| OsvPageSummary {
                    minipage_id,
                    cluster_id: e.cluster_id,
                    vector_count: e.vector_count,
                    compressed_size: e.compressed_size,
                    uncompressed_size: e.uncompressed_size,
                    byte_offset: e.byte_offset,
                    page_xxh64: format!("0x{:016x}", e.page_xxh64),
                })
                .collect(),
        )
    } else {
        None
    };

    let xxh64_status = if verify_checksums {
        format!(
            "VERIFIED (header_xxh64 + {pages_verified}/{} page_xxh64)",
            dir.len()
        )
    } else {
        "HEADER VERIFIED (pass --verify-checksums to scan all minipage frames)".to_string()
    };

    Ok(InspectedArtifact::Osv(OsvInspectReport {
        path: path.display().to_string(),
        simd_backend: active_simd_backend(),
        file_size_bytes,
        version: hdr.version,
        zstd_level: hdr.zstd_level,
        orig_dim: hdr.orig_dim,
        flags: hdr.flags,
        has_zstd_dict: reader.has_zstd_dict(),
        zstd_dict_len: hdr.zstd_dict_len,
        num_minipages: hdr.num_minipages,
        oversized_minipages,
        total_vectors: hdr.total_vectors,
        total_uncompressed_bytes: total_uncompressed,
        total_compressed_bytes: total_compressed,
        compression_ratio,
        space_savings_pct,
        compressed_size_dist,
        uncompressed_size_dist,
        header_xxh64: format!("0x{:016x}", hdr.header_xxh64),
        pages_verified,
        xxh64_status,
        directory,
    }))
}

fn inspect_osr(
    path: &Path,
    file_size_bytes: u64,
    bytes: &[u8],
    dump_directory: bool,
) -> CliResult<InspectedArtifact> {
    let store = OsrStore::from_bytes(bytes)?;
    let hdr = *store.header();

    let file_hash_bytes: [u8; 8] = bytes[bytes.len() - 8..]
        .try_into()
        .map_err(|_| CliError::Core(OspreyError::CorruptedFooter))?;
    let file_xxh64 = u64::from_le_bytes(file_hash_bytes);

    let raw_fp32_bytes = hdr
        .total_tokens
        .saturating_mul(u64::from(hdr.token_dim))
        .saturating_mul(4);
    let raw_fp16_bytes = hdr
        .total_tokens
        .saturating_mul(u64::from(hdr.token_dim))
        .saturating_mul(2);
    let quantized_tensor_bytes = hdr
        .total_tokens
        .saturating_mul(u64::try_from(OSR_TOKEN_BYTES).unwrap_or(64));

    #[allow(clippy::cast_precision_loss)]
    let avg_tokens_per_doc = if hdr.num_docs > 0 {
        (hdr.total_tokens as f64) / (hdr.num_docs as f64)
    } else {
        0.0
    };
    #[allow(clippy::cast_precision_loss)]
    let compression_vs_fp32 = if file_size_bytes > 0 {
        (raw_fp32_bytes as f64) / (file_size_bytes as f64)
    } else {
        1.0
    };
    #[allow(clippy::cast_precision_loss)]
    let compression_vs_fp16 = if file_size_bytes > 0 {
        (raw_fp16_bytes as f64) / (file_size_bytes as f64)
    } else {
        1.0
    };

    let docs = if dump_directory {
        Some(
            store
                .doc_table()
                .iter()
                .map(|d| OsrDocSummary {
                    doc_id: d.doc_id,
                    token_start_idx: d.token_start_idx,
                    token_count: d.token_count,
                })
                .collect(),
        )
    } else {
        None
    };

    Ok(InspectedArtifact::Osr(OsrInspectReport {
        path: path.display().to_string(),
        simd_backend: active_simd_backend(),
        file_size_bytes,
        version: hdr.version,
        token_dim: hdr.token_dim,
        subspace_dim: hdr.subspace_dim,
        num_layers: hdr.num_layers,
        gamma: hdr.gamma,
        fwht_seed: format!("0x{:016x}", hdr.fwht_seed),
        num_docs: hdr.num_docs,
        total_tokens: hdr.total_tokens,
        avg_tokens_per_doc,
        raw_fp32_bytes,
        raw_fp16_bytes,
        quantized_tensor_bytes,
        compression_vs_fp32,
        compression_vs_fp16,
        header_xxh64: format!("0x{:016x}", hdr.header_xxh64),
        file_xxh64: format!("0x{file_xxh64:016x}"),
        xxh64_status: "VERIFIED (header_xxh64 + file_xxh64)".to_string(),
        docs,
    }))
}

fn compute_size_distribution(sizes: &mut [u32]) -> SizeDistribution {
    if sizes.is_empty() {
        return SizeDistribution {
            min_bytes: 0,
            mean_bytes: 0.0,
            p50_bytes: 0,
            p95_bytes: 0,
            max_bytes: 0,
        };
    }
    sizes.sort_unstable();
    let sum: u64 = sizes.iter().map(|&s| u64::from(s)).sum();
    #[allow(clippy::cast_precision_loss)]
    let mean_bytes = (sum as f64) / (sizes.len() as f64);
    let p50_idx = (sizes.len() - 1) / 2;
    let p95_idx = ((sizes.len() - 1) * 95) / 100;
    SizeDistribution {
        min_bytes: sizes[0],
        mean_bytes,
        p50_bytes: sizes[p50_idx],
        p95_bytes: sizes[p95_idx],
        max_bytes: sizes[sizes.len() - 1],
    }
}

fn print_osi_human(r: &OsiInspectReport) {
    print_section(&format!("{} {}", r.path, paint_accent("(.osi)")));
    print_kv("SIMD backend", &paint_pass(r.simd_backend));
    print_kv("File size", &format_bytes(r.file_size_bytes));
    print_kv(
        "Dimensions",
        &format!("orig_dim={} -> padded_dim={}", r.orig_dim, r.padded_dim),
    );
    print_kv("Centroids (K)", &paint_accent(&r.num_centroids.to_string()));
    print_kv("Vectors (N)", &paint_accent(&r.num_vectors.to_string()));
    print_kv(
        "Tombstones",
        &if r.active_tombstones == 0 {
            paint_dim("0 active")
        } else {
            paint_warn(&format!("{} active", r.active_tombstones))
        },
    );
    print_kv("FWHT seed", &paint_dim(&r.fwht_seed));
    print_kv(
        "64B offsets",
        &format!(
            "centroids={}, entries={}, tombstones={}",
            r.centroids_offset, r.entries_offset, r.tombstones_offset
        ),
    );
    print_kv(
        "Checksums",
        &format!("header={}, file={}", r.header_xxh64, r.file_xxh64),
    );
    print_kv("Integrity", &paint_pass(&r.xxh64_status));

    if let Some(clusters) = &r.clusters {
        print_section(".osi Cluster Spans");
        let mut c_table = new_rounded_table(&["Cluster ID", "Vectors", "Relative Offset"]);
        for c in clusters {
            c_table.add_row(vec![
                Cell::new(c.cluster_id).set_alignment(CellAlignment::Right),
                Cell::new(c.vector_count).set_alignment(CellAlignment::Right),
                Cell::new(c.cluster_offset).set_alignment(CellAlignment::Right),
            ]);
        }
        print_indented_table(&c_table);
    }
}

fn print_osv_human(r: &OsvInspectReport) {
    print_section(&format!("{} {}", r.path, paint_accent("(.osv)")));
    print_kv("SIMD backend", &paint_pass(r.simd_backend));
    print_kv("File size", &format_bytes(r.file_size_bytes));
    print_kv("Original dim", &r.orig_dim.to_string());
    print_kv(
        "Zstd codec",
        &format!(
            "level={}, dict={}",
            r.zstd_level,
            if r.has_zstd_dict {
                paint_pass(&format_bytes(r.zstd_dict_len))
            } else {
                paint_dim("none")
            }
        ),
    );
    print_kv(
        "Minipages",
        &format!("{} (oversized={})", r.num_minipages, r.oversized_minipages),
    );
    print_kv("Total vectors", &paint_accent(&r.total_vectors.to_string()));
    print_kv(
        "Payload bytes",
        &format!(
            "{} raw -> {} compressed",
            format_bytes(r.total_uncompressed_bytes),
            format_bytes(r.total_compressed_bytes)
        ),
    );
    print_kv(
        "Compression ratio",
        &paint_pass(&format!(
            "{:.3}x ({:.2}% saved)",
            r.compression_ratio, r.space_savings_pct
        )),
    );
    print_kv(
        "Compressed pages",
        &format!(
            "min={} B · p50={} B · mean={:.0} B · p95={} B · max={} B",
            r.compressed_size_dist.min_bytes,
            r.compressed_size_dist.p50_bytes,
            r.compressed_size_dist.mean_bytes,
            r.compressed_size_dist.p95_bytes,
            r.compressed_size_dist.max_bytes
        ),
    );
    print_kv(
        "Uncompressed pages",
        &format!(
            "min={} B · p50={} B · mean={:.0} B · p95={} B · max={} B",
            r.uncompressed_size_dist.min_bytes,
            r.uncompressed_size_dist.p50_bytes,
            r.uncompressed_size_dist.mean_bytes,
            r.uncompressed_size_dist.p95_bytes,
            r.uncompressed_size_dist.max_bytes
        ),
    );
    print_kv("header_xxh64", &paint_dim(&r.header_xxh64));
    let status_val = if r.xxh64_status.starts_with("VERIFIED") {
        paint_pass(&r.xxh64_status)
    } else {
        paint_warn(&r.xxh64_status)
    };
    print_kv("Integrity", &status_val);

    if let Some(pages) = &r.directory {
        print_section(".osv Minipage Directory");
        let mut p_table = new_rounded_table(&[
            "Page ID",
            "Cluster",
            "Vecs",
            "Comp (B)",
            "Raw (B)",
            "Offset",
            "page_xxh64",
        ]);
        for p in pages {
            p_table.add_row(vec![
                Cell::new(p.minipage_id).set_alignment(CellAlignment::Right),
                Cell::new(p.cluster_id).set_alignment(CellAlignment::Right),
                Cell::new(p.vector_count).set_alignment(CellAlignment::Right),
                Cell::new(p.compressed_size).set_alignment(CellAlignment::Right),
                Cell::new(p.uncompressed_size).set_alignment(CellAlignment::Right),
                Cell::new(p.byte_offset).set_alignment(CellAlignment::Right),
                Cell::new(paint_dim(&p.page_xxh64)),
            ]);
        }
        print_indented_table(&p_table);
    }
}

fn print_osr_human(r: &OsrInspectReport) {
    print_section(&format!("{} {}", r.path, paint_accent("(.osr)")));
    print_kv("SIMD backend", &paint_pass(r.simd_backend));
    print_kv("File size", &format_bytes(r.file_size_bytes));
    print_kv(
        "Lattice",
        &format!(
            "token_dim={}, subspace_dim={}, layers={}, gamma={:.2}",
            r.token_dim, r.subspace_dim, r.num_layers, r.gamma
        ),
    );
    print_kv(
        "Docs / Tokens",
        &format!(
            "{} docs · {} tokens (avg {:.1}/doc)",
            paint_accent(&r.num_docs.to_string()),
            paint_accent(&r.total_tokens.to_string()),
            r.avg_tokens_per_doc
        ),
    );
    print_kv(
        "Compression",
        &paint_pass(&format!(
            "{:.2}x vs FP32 ({}) · {:.2}x vs FP16 ({})",
            r.compression_vs_fp32,
            format_bytes(r.raw_fp32_bytes),
            r.compression_vs_fp16,
            format_bytes(r.raw_fp16_bytes)
        )),
    );
    print_kv("FWHT seed", &paint_dim(&r.fwht_seed));
    print_kv(
        "Checksums",
        &format!("header={}, file={}", r.header_xxh64, r.file_xxh64),
    );
    print_kv("Integrity", &paint_pass(&r.xxh64_status));

    if let Some(docs) = &r.docs {
        print_section(".osr Document Token Table");
        let mut d_table = new_rounded_table(&["Doc ID", "Token Start", "Token Count"]);
        for d in docs {
            d_table.add_row(vec![
                Cell::new(d.doc_id).set_alignment(CellAlignment::Right),
                Cell::new(d.token_start_idx).set_alignment(CellAlignment::Right),
                Cell::new(d.token_count).set_alignment(CellAlignment::Right),
            ]);
        }
        print_indented_table(&d_table);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use half::f16;
    use osprey_core::{
        fit_binary_kmeans, AlignedBuffer, BinaryKMeansConfig, BinaryKMeansResult, OsiIndex,
        OsrDocumentTokens, VectorLocation, DEFAULT_LUTRANK_GAMMA, LUTRANK_TOKEN_DIM,
    };
    use osprey_simd::SimdEngine;
    use osprey_storage::{OsvVectorRecord, OsvWriter, OsvWriterConfig};
    use roaring::RoaringBitmap;
    use std::io::BufWriter;

    #[test]
    fn inspect_osi_osv_osr_and_detect_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let osi_path = dir.path().join("seg.osi");
        let osv_path = dir.path().join("seg.osv");
        let osr_path = dir.path().join("seg.osr");

        let n = 16;
        let dim = 64;
        let mut vecs = Vec::with_capacity(n * dim);
        let mut doc_ids = Vec::with_capacity(n);
        for i in 0..n {
            doc_ids.push(i as u64);
            for d in 0..dim {
                vecs.push(((i * 5 + d) as f32 * 0.03).cos());
            }
        }
        let vecs_f16: Vec<f16> = vecs.iter().map(|&v| f16::from_f32(v)).collect();

        let simd = SimdEngine::new_auto();
        let km = fit_binary_kmeans(&vecs, n, dim, 42, &BinaryKMeansConfig::new(2), &simd).unwrap();
        let mut cents = AlignedBuffer::<f16>::new(km.centroids_f16.len()).unwrap();
        cents
            .as_mut_slice()
            .copy_from_slice(km.centroids_f16.as_slice());

        let records: Vec<OsvVectorRecord<'_>> = (0..n)
            .map(|i| OsvVectorRecord {
                doc_id: doc_ids[i],
                cluster_id: km.assignments[i],
                vector_f16: &vecs_f16[i * dim..(i + 1) * dim],
                payload: b"{\"ok\":true}",
            })
            .collect();
        let osv_report = OsvWriter::new(OsvWriterConfig::default())
            .write_slab(
                &mut BufWriter::new(File::create(&osv_path).unwrap()),
                &records,
                dim,
                None,
            )
            .unwrap();

        let locs: Vec<VectorLocation> = doc_ids
            .iter()
            .map(|&id| {
                let (mp, slot) = osv_report.location_for_doc(id).unwrap();
                VectorLocation {
                    minipage_id: mp,
                    slot_idx: slot,
                    slab_id: 0,
                }
            })
            .collect();
        let km_res = BinaryKMeansResult {
            centroids_f16: cents,
            binary_centroids: km.binary_centroids,
            assignments: km.assignments,
            final_hamming_inertia: km.final_hamming_inertia,
            iterations_run: km.iterations_run,
        };
        let osi = OsiIndex::build(
            &vecs,
            &doc_ids,
            &locs,
            &km_res,
            dim,
            42,
            RoaringBitmap::new(),
            &simd,
        )
        .unwrap();
        std::fs::write(&osi_path, osi.to_aligned_bytes().unwrap().as_slice()).unwrap();

        let tok_buf = vec![0.1_f32; 4 * LUTRANK_TOKEN_DIM];
        let osr_docs = vec![OsrDocumentTokens {
            doc_id: 0,
            tokens: &tok_buf,
            num_tokens: 4,
        }];
        let osr = OsrStore::build(
            &osr_docs,
            LUTRANK_TOKEN_DIM,
            DEFAULT_LUTRANK_GAMMA,
            42,
            &simd,
        )
        .unwrap();
        std::fs::write(&osr_path, osr.to_aligned_bytes().unwrap().as_slice()).unwrap();

        let args = InspectArgs {
            file: vec![osi_path.clone(), osv_path.clone(), osr_path],
            verify_checksums: true,
            dump_directory: true,
        };
        let ctx = CliContext {
            json: false,
            verbose: 0,
            threads: 1,
        };
        run(&args, &ctx).unwrap();

        // Corrupt a byte in .osv compressed frame -> --verify-checksums must detect ChecksumMismatch
        let mut raw_osv = std::fs::read(&osv_path).unwrap();
        raw_osv[72] ^= 0xFF;
        std::fs::write(&osv_path, raw_osv).unwrap();
        let err = inspect_one(&osv_path, true, false).unwrap_err();
        assert_eq!(err.numeric_code(), 3);
    }
}
