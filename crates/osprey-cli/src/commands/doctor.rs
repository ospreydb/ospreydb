//! `osprey doctor`: CPU SIMD inspection, alignment check, memory bandwidth
//! micro-benchmark, and optional Object Storage connectivity verification.

use std::hint::black_box;
use std::time::Instant;

use osprey_core::{AlignedBuffer, CACHE_LINE_ALIGN};
use osprey_simd::{
    active_simd_backend, detect_best_simd_level, is_level_supported, SimdBackend, SimdEngine,
    SimdLevel,
};
use osprey_storage::{build_object_store, RemoteStoreConfig};
use serde::Serialize;

use crate::cli::DoctorArgs;
use crate::output::{
    current_dir_short, format_bytes, paint_accent, paint_dim, paint_fail, paint_pass,
    print_cli_header, print_json, print_kv, print_section, CliContext, CliError, CliResult,
};

#[derive(Debug, Serialize)]
struct DoctorSimdReport {
    best_tier: String,
    active_engine_tier: String,
    active_backend: &'static str,
    avx512_supported: bool,
    avx2_supported: bool,
    neon_supported: bool,
    scalar_supported: bool,
    isa_features: Vec<(&'static str, bool)>,
}

#[derive(Debug, Serialize)]
struct MemoryBandwidthReport {
    buffer_bytes: usize,
    alignment_bytes: usize,
    alignment_ok: bool,
    write_gb_per_sec: f64,
    read_gb_per_sec: f64,
    copy_gb_per_sec: f64,
}

#[derive(Debug, Serialize)]
struct S3CheckReport {
    uri: String,
    reachable: bool,
    object_size_bytes: Option<u64>,
    head_ms: Option<f64>,
    range_get_ms: Option<f64>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct DoctorReport {
    os: &'static str,
    arch: &'static str,
    logical_cores: usize,
    simd: DoctorSimdReport,
    memory: MemoryBandwidthReport,
    s3_check: Option<S3CheckReport>,
}

/// Executes `osprey doctor`.
pub fn run(args: &DoctorArgs, ctx: &CliContext) -> CliResult<()> {
    ctx.log_info("Running OspreyDB host CPU, alignment, and memory bandwidth diagnostics...");

    let simd_report = inspect_simd();
    let bench_mb = args.bench_mb.clamp(1, 512);
    let mem_report = benchmark_memory_bandwidth(bench_mb)?;

    let s3_check = if let Some(uri) = &args.s3_uri {
        ctx.log_info(format!(
            "Checking Object Storage connectivity for '{uri}'..."
        ));
        Some(check_s3_connectivity(uri)?)
    } else {
        None
    };

    let logical_cores = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);

    let report = DoctorReport {
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        logical_cores,
        simd: simd_report,
        memory: mem_report,
        s3_check,
    };

    if ctx.json {
        print_json(&report)?;
    } else {
        print_doctor_human(&report);
    }

    if !report.memory.alignment_ok {
        return Err(CliError::GateFailed(
            "64-byte cache-line alignment check failed".to_string(),
        ));
    }
    if let Some(s3) = &report.s3_check {
        if !s3.reachable {
            return Err(CliError::GateFailed(format!(
                "Object Storage check failed for '{}': {}",
                s3.uri,
                s3.error.as_deref().unwrap_or("unknown error")
            )));
        }
    }

    Ok(())
}

fn simd_level_label(level: SimdLevel) -> &'static str {
    match level {
        SimdLevel::Avx512 => "AVX-512 (512-bit ZMM: F + BW + VPOPCNTDQ)",
        SimdLevel::Avx2 => "AVX2 (256-bit YMM: AVX2 + FMA + POPCNT)",
        SimdLevel::Neon => "ARM64 NEON (128-bit Q + vtbl)",
        SimdLevel::Scalar => "Portable Scalar Fallback (64-bit)",
    }
}

fn inspect_simd() -> DoctorSimdReport {
    let best = detect_best_simd_level();
    let engine = SimdEngine::new_auto();

    let mut isa_features = Vec::new();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        isa_features.push(("avx512f", std::arch::is_x86_feature_detected!("avx512f")));
        isa_features.push(("avx512bw", std::arch::is_x86_feature_detected!("avx512bw")));
        isa_features.push((
            "avx512vpopcntdq",
            std::arch::is_x86_feature_detected!("avx512vpopcntdq"),
        ));
        isa_features.push(("avx2", std::arch::is_x86_feature_detected!("avx2")));
        isa_features.push(("fma", std::arch::is_x86_feature_detected!("fma")));
        isa_features.push(("f16c", std::arch::is_x86_feature_detected!("f16c")));
        isa_features.push(("popcnt", std::arch::is_x86_feature_detected!("popcnt")));
    }
    #[cfg(target_arch = "aarch64")]
    {
        isa_features.push(("neon", std::arch::is_aarch64_feature_detected!("neon")));
        isa_features.push(("fp16", std::arch::is_aarch64_feature_detected!("fp16")));
    }

    DoctorSimdReport {
        best_tier: simd_level_label(best).to_string(),
        active_engine_tier: simd_level_label(engine.level()).to_string(),
        active_backend: active_simd_backend(),
        avx512_supported: is_level_supported(SimdLevel::Avx512),
        avx2_supported: is_level_supported(SimdLevel::Avx2),
        neon_supported: is_level_supported(SimdLevel::Neon),
        scalar_supported: is_level_supported(SimdLevel::Scalar),
        isa_features,
    }
}

fn benchmark_memory_bandwidth(bench_mb: usize) -> CliResult<MemoryBandwidthReport> {
    let buffer_bytes = bench_mb * 1024 * 1024;
    let words = buffer_bytes / std::mem::size_of::<u64>();
    let mut src = AlignedBuffer::<u64>::new(words)?;
    let mut dst = AlignedBuffer::<u64>::new(words)?;

    let ptr_src = src.as_slice().as_ptr() as usize;
    let ptr_dst = dst.as_slice().as_ptr() as usize;
    let alignment_ok =
        ptr_src.is_multiple_of(CACHE_LINE_ALIGN) && ptr_dst.is_multiple_of(CACHE_LINE_ALIGN);

    // Warm up pages so OS page-fault latency does not skew bandwidth measurements.
    for (idx, word) in src.as_mut_slice().iter_mut().enumerate() {
        *word = (idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    dst.as_mut_slice().copy_from_slice(src.as_slice());

    let iters = if bench_mb >= 64 { 3_usize } else { 6_usize };
    #[allow(clippy::cast_precision_loss)]
    let total_gb = ((buffer_bytes * iters) as f64) / 1e9;

    // 1. Sequential aligned write benchmark
    let t_write = Instant::now();
    for iter in 0..iters {
        let pattern = 0x55AA_55AA_55AA_55AA_u64.wrapping_add(iter as u64);
        dst.as_mut_slice().fill(pattern);
        black_box(dst.as_slice()[0]);
    }
    let write_secs = t_write.elapsed().as_secs_f64().max(1e-9);
    let write_gb_per_sec = total_gb / write_secs;

    // 2. Sequential aligned read benchmark (8-way unrolled accumulation)
    let t_read = Instant::now();
    let mut sink = 0_u64;
    for _ in 0..iters {
        let slice = src.as_slice();
        let mut a0 = 0_u64;
        let mut a1 = 0_u64;
        let mut a2 = 0_u64;
        let mut a3 = 0_u64;
        let mut a4 = 0_u64;
        let mut a5 = 0_u64;
        let mut a6 = 0_u64;
        let mut a7 = 0_u64;
        for chunk in slice.as_chunks::<8>().0 {
            a0 = a0.wrapping_add(chunk[0]);
            a1 = a1.wrapping_add(chunk[1]);
            a2 = a2.wrapping_add(chunk[2]);
            a3 = a3.wrapping_add(chunk[3]);
            a4 = a4.wrapping_add(chunk[4]);
            a5 = a5.wrapping_add(chunk[5]);
            a6 = a6.wrapping_add(chunk[6]);
            a7 = a7.wrapping_add(chunk[7]);
        }
        sink ^= a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
    }
    black_box(sink);
    let read_secs = t_read.elapsed().as_secs_f64().max(1e-9);
    let read_gb_per_sec = total_gb / read_secs;

    // 3. Sequential aligned copy benchmark (read + write DRAM traffic = 2 * buffer_bytes)
    let t_copy = Instant::now();
    for _ in 0..iters {
        dst.as_mut_slice().copy_from_slice(src.as_slice());
        black_box(dst.as_slice()[words / 2]);
    }
    let copy_secs = t_copy.elapsed().as_secs_f64().max(1e-9);
    let copy_gb_per_sec = (total_gb * 2.0) / copy_secs;

    Ok(MemoryBandwidthReport {
        buffer_bytes,
        alignment_bytes: CACHE_LINE_ALIGN,
        alignment_ok,
        write_gb_per_sec,
        read_gb_per_sec,
        copy_gb_per_sec,
    })
}

fn check_s3_connectivity(uri: &str) -> CliResult<S3CheckReport> {
    let remote_cfg = RemoteStoreConfig::default();
    let (store, obj_path) = match build_object_store(uri, &remote_cfg) {
        Ok(pair) => pair,
        Err(err) => {
            return Ok(S3CheckReport {
                uri: uri.to_string(),
                reachable: false,
                object_size_bytes: None,
                head_ms: None,
                range_get_ms: None,
                error: Some(err.to_string()),
            });
        }
    };

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| CliError::Io(format!("failed to create tokio runtime: {err}")))?;

    rt.block_on(async move {
        let t_head = Instant::now();
        let meta = match store.head(&obj_path).await {
            Ok(meta) => meta,
            Err(err) => {
                return Ok(S3CheckReport {
                    uri: uri.to_string(),
                    reachable: false,
                    object_size_bytes: None,
                    head_ms: None,
                    range_get_ms: None,
                    error: Some(err.to_string()),
                });
            }
        };
        let head_ms = t_head.elapsed().as_secs_f64() * 1_000.0;
        let obj_size = u64::try_from(meta.size).unwrap_or(u64::MAX);

        let range_len = meta.size.min(64);
        let t_get = Instant::now();
        if range_len > 0 {
            if let Err(err) = store.get_range(&obj_path, 0..range_len).await {
                return Ok(S3CheckReport {
                    uri: uri.to_string(),
                    reachable: false,
                    object_size_bytes: Some(obj_size),
                    head_ms: Some(head_ms),
                    range_get_ms: None,
                    error: Some(err.to_string()),
                });
            }
        }
        let range_get_ms = t_get.elapsed().as_secs_f64() * 1_000.0;

        Ok(S3CheckReport {
            uri: uri.to_string(),
            reachable: true,
            object_size_bytes: Some(obj_size),
            head_ms: Some(head_ms),
            range_get_ms: Some(range_get_ms),
            error: None,
        })
    })
}

fn print_doctor_human(report: &DoctorReport) {
    print_cli_header(
        &format!(
            "doctor · {}/{} · {} cores",
            report.os, report.arch, report.logical_cores
        ),
        &paint_pass(&report.simd.active_engine_tier),
        &current_dir_short(),
    );

    print_section("SIMD & Alignment");
    print_kv(
        "Active engine",
        &paint_pass(&report.simd.active_engine_tier),
    );
    print_kv("Best detected", &paint_pass(&report.simd.best_tier));
    print_kv(
        "Tiers",
        &format!(
            "{}  {}  {}  {}",
            format_tier_status("avx512", report.simd.avx512_supported),
            format_tier_status("avx2", report.simd.avx2_supported),
            format_tier_status("neon", report.simd.neon_supported),
            format_tier_status("scalar", report.simd.scalar_supported),
        ),
    );

    if !report.simd.isa_features.is_empty() {
        let flags: Vec<String> = report
            .simd
            .isa_features
            .iter()
            .map(|(name, ok)| {
                if *ok {
                    paint_pass(name)
                } else {
                    paint_dim(name)
                }
            })
            .collect();
        print_kv("ISA flags", &flags.join("  "));
    }

    let align_val = if report.memory.alignment_ok {
        paint_pass(&format!(
            "PASS ({}-byte aligned)",
            report.memory.alignment_bytes
        ))
    } else {
        paint_fail("FAIL (unaligned)")
    };
    print_kv("Cache alignment", &align_val);

    let bench_buf = format_bytes(u64::try_from(report.memory.buffer_bytes).unwrap_or(0));
    print_section(&format!("Memory Bandwidth ({bench_buf} aligned)"));
    print_kv(
        "Seq read",
        &paint_pass(&format!("{:>7.2} GB/s", report.memory.read_gb_per_sec)),
    );
    print_kv(
        "Seq write",
        &paint_pass(&format!("{:>7.2} GB/s", report.memory.write_gb_per_sec)),
    );
    print_kv(
        "Copy (R+W)",
        &paint_accent(&format!("{:>7.2} GB/s", report.memory.copy_gb_per_sec)),
    );

    if let Some(s3) = &report.s3_check {
        print_section("Object Storage");
        let status = if s3.reachable {
            paint_pass(&format!(
                "PASS ({}, head {:.2} ms, range-get {:.2} ms)",
                s3.object_size_bytes
                    .map(format_bytes)
                    .unwrap_or_else(|| "n/a".to_string()),
                s3.head_ms.unwrap_or(0.0),
                s3.range_get_ms.unwrap_or(0.0),
            ))
        } else {
            paint_fail(&format!(
                "FAIL ({})",
                s3.error.as_deref().unwrap_or("unknown")
            ))
        };
        print_kv("Endpoint", &s3.uri);
        print_kv("Status", &status);
    }
    println!();
}

fn format_tier_status(name: &str, supported: bool) -> String {
    if supported {
        paint_pass(name)
    } else {
        paint_dim(&format!("{name}:no"))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn doctor_simd_and_memory_bandwidth_succeed() {
        let args = DoctorArgs {
            s3_uri: None,
            bench_mb: 4,
        };
        let ctx = CliContext {
            json: false,
            verbose: 0,
            threads: 1,
        };
        let res = run(&args, &ctx);
        assert!(res.is_ok());

        let simd = inspect_simd();
        assert!(simd.scalar_supported);

        let bw = benchmark_memory_bandwidth(2).unwrap();
        assert!(bw.alignment_ok);
        assert!(bw.read_gb_per_sec > 0.0);
        assert!(bw.write_gb_per_sec > 0.0);
        assert!(bw.copy_gb_per_sec > 0.0);
    }

    #[test]
    fn doctor_s3_file_uri_connectivity_check() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), vec![0x42_u8; 128]).unwrap();
        let uri = url::Url::from_file_path(tmp.path()).unwrap().to_string();

        let s3 = check_s3_connectivity(&uri).unwrap();
        assert!(s3.reachable);
        assert_eq!(s3.object_size_bytes, Some(128));
    }
}
