//! `osprey delete`: atomically inserts `doc_id` tombstones into an existing
//! `.osi` `RoaringBitmap` footer and re-seals checksums via temp-file + atomic rename.

use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use memmap2::Mmap;
use osprey_core::OsiIndex;
use serde::Serialize;
use tempfile::NamedTempFile;

use crate::cli::DeleteArgs;
use crate::output::{
    format_bytes, paint_accent, paint_dim, paint_pass, parse_doc_ids_file, print_json, print_kv,
    print_section, CliContext, CliError, CliResult,
};

#[derive(Debug, Serialize)]
struct DeleteReport {
    index_path: String,
    requested_ids: usize,
    newly_tombstoned: usize,
    already_tombstoned: usize,
    total_tombstones: u64,
    file_size_bytes: u64,
    new_file_xxh64: String,
}

/// Executes `osprey delete`.
pub fn run(args: &DeleteArgs, ctx: &CliContext) -> CliResult<()> {
    let mut target_ids: Vec<u64> = args.ids.clone();
    if let Some(ids_file) = &args.ids_file {
        let file_ids = parse_doc_ids_file(ids_file)?;
        target_ids.extend(file_ids);
    }

    let mut unique_ids = HashSet::with_capacity(target_ids.len());
    let mut deduped = Vec::with_capacity(target_ids.len());
    for id in target_ids {
        if unique_ids.insert(id) {
            deduped.push(id);
        }
    }

    if deduped.is_empty() {
        return Err(CliError::InvalidArgs(
            "provide at least one doc_id via --ids or --ids-file".to_string(),
        ));
    }

    let mut ids_u32 = Vec::with_capacity(deduped.len());
    for &id in &deduped {
        let id_u32 = u32::try_from(id).map_err(|_| {
            CliError::InvalidArgs(format!(
                "doc_id {id} exceeds u32::MAX (RoaringBitmap tombstones support 0..=4294967295)"
            ))
        })?;
        ids_u32.push(id_u32);
    }

    ctx.log_info(format!(
        "Opening .osi index '{}' to tombstone {} unique doc_id(s)...",
        args.index.display(),
        ids_u32.len()
    ));

    let mut index = {
        let file = File::open(&args.index)?;
        // SAFETY: Read-only mapping while loading and validating `.osi`; dropped before atomic rename.
        let mmap = unsafe { Mmap::map(&file)? };
        OsiIndex::from_bytes(&mmap)?
    };

    let mut newly_tombstoned = 0_usize;
    let mut already_tombstoned = 0_usize;
    {
        let tombstones = index.tombstones_mut();
        for id in ids_u32 {
            if tombstones.insert(id) {
                newly_tombstoned += 1;
            } else {
                already_tombstoned += 1;
            }
        }
    }

    let total_tombstones = index.tombstones().len();
    let sealed_bytes = index.to_aligned_bytes()?;
    let slice = sealed_bytes.as_slice();
    let file_size_bytes = u64::try_from(slice.len()).unwrap_or(u64::MAX);
    let new_hash = if slice.len() >= 8 {
        let tail: [u8; 8] = slice[slice.len() - 8..]
            .try_into()
            .map_err(|_| CliError::Core(osprey_core::OspreyError::CorruptedFooter))?;
        u64::from_le_bytes(tail)
    } else {
        0
    };

    atomic_write_file(&args.index, slice)?;

    let report = DeleteReport {
        index_path: args.index.display().to_string(),
        requested_ids: deduped.len(),
        newly_tombstoned,
        already_tombstoned,
        total_tombstones,
        file_size_bytes,
        new_file_xxh64: format!("0x{new_hash:016x}"),
    };

    if ctx.json {
        print_json(&report)?;
    } else {
        print_section(&format!("Tombstones updated ({})", report.index_path));
        print_kv("Requested IDs", &report.requested_ids.to_string());
        print_kv(
            "Newly tombstoned",
            &paint_pass(&report.newly_tombstoned.to_string()),
        );
        print_kv(
            "Already deleted",
            &paint_dim(&report.already_tombstoned.to_string()),
        );
        print_kv(
            "Active tombstones",
            &paint_accent(&report.total_tombstones.to_string()),
        );
        print_kv("Updated size", &format_bytes(report.file_size_bytes));
        print_kv("New file_xxh64", &paint_dim(&report.new_file_xxh64));
        println!();
    }

    Ok(())
}

/// Writes `bytes` to a temporary sibling file, flushes/fsyncs, and atomically renames over `dest`.
pub fn atomic_write_file(dest: &Path, bytes: &[u8]) -> CliResult<()> {
    let parent = match dest.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let mut tmp = NamedTempFile::new_in(parent)?;
    tmp.write_all(bytes)?;
    tmp.flush()?;
    tmp.as_file().sync_all()?;
    tmp.persist(dest).map_err(|err| err.error)?;
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]
mod tests {
    use super::*;
    use half::f16;
    use osprey_core::{
        fit_binary_kmeans, AlignedBuffer, BinaryKMeansConfig, BinaryKMeansResult, VectorLocation,
    };
    use osprey_simd::SimdEngine;
    use roaring::RoaringBitmap;

    fn write_sample_osi(path: &Path, n: usize, dim: usize) {
        let mut vecs = Vec::with_capacity(n * dim);
        let mut doc_ids = Vec::with_capacity(n);
        let mut locs = Vec::with_capacity(n);
        for i in 0..n {
            doc_ids.push(i as u64);
            locs.push(VectorLocation {
                minipage_id: 0,
                slot_idx: i as u16,
                slab_id: 0,
            });
            for d in 0..dim {
                vecs.push(((i * 7 + d) as f32 * 0.01).sin());
            }
        }
        let simd = SimdEngine::new_auto();
        let km =
            fit_binary_kmeans(&vecs, n, dim, 0x1234, &BinaryKMeansConfig::new(4), &simd).unwrap();
        let mut cents = AlignedBuffer::<f16>::new(km.centroids_f16.len()).unwrap();
        cents
            .as_mut_slice()
            .copy_from_slice(km.centroids_f16.as_slice());
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
            0x1234,
            RoaringBitmap::new(),
            &simd,
        )
        .unwrap();
        let bytes = osi.to_aligned_bytes().unwrap();
        std::fs::write(path, bytes.as_slice()).unwrap();
    }

    #[test]
    fn delete_updates_tombstones_atomically_and_filters_search() {
        let dir = tempfile::tempdir().unwrap();
        let osi_path = dir.path().join("test.osi");
        let ids_file = dir.path().join("ids.txt");
        write_sample_osi(&osi_path, 32, 64);
        std::fs::write(&ids_file, "# comment\n3\n7, 9\n").unwrap();

        let args = DeleteArgs {
            index: osi_path.clone(),
            ids: vec![1, 3],
            ids_file: Some(ids_file),
        };
        let ctx = CliContext {
            json: false,
            verbose: 0,
            threads: 1,
        };
        run(&args, &ctx).unwrap();

        let bytes = AlignedBuffer::from_slice(&std::fs::read(&osi_path).unwrap()).unwrap();
        let reloaded = OsiIndex::from_bytes(bytes.as_slice()).unwrap();
        assert!(reloaded.is_deleted(1));
        assert!(reloaded.is_deleted(3));
        assert!(reloaded.is_deleted(7));
        assert!(reloaded.is_deleted(9));
        assert!(!reloaded.is_deleted(2));
        assert_eq!(reloaded.tombstones().len(), 4);
    }
}
