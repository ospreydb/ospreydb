//! End-to-end subprocess integration tests for the `osprey-cli` binary.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use arrow::array::{FixedSizeListBuilder, Float32Builder, ListBuilder, UInt64Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;

fn osprey_bin() -> &'static str {
    env!("CARGO_BIN_EXE_osprey-cli")
}

fn write_npy_f32(path: &Path, rows: usize, dim: usize, data: &[f32]) {
    let dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {dim}), }}");
    let prefix_len = 10;
    let total_hdr = (prefix_len + dict.len() + 1).next_multiple_of(64);
    let hdr_len = total_hdr - prefix_len;
    let mut file = File::create(path).unwrap();
    file.write_all(b"\x93NUMPY\x01\x00").unwrap();
    file.write_all(&(hdr_len as u16).to_le_bytes()).unwrap();
    file.write_all(dict.as_bytes()).unwrap();
    let pad = hdr_len - dict.len() - 1;
    file.write_all(&vec![b' '; pad]).unwrap();
    file.write_all(b"\n").unwrap();
    for &v in data {
        file.write_all(&v.to_le_bytes()).unwrap();
    }
}

fn write_parquet_corpus(path: &Path, n: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
    let mut id_builder = UInt64Builder::new();
    let mut vec_builder = FixedSizeListBuilder::new(Float32Builder::new(), dim as i32);
    let mut tok_builder = ListBuilder::new(FixedSizeListBuilder::new(Float32Builder::new(), 128));

    let mut target_vec = vec![0.0_f32; dim];
    let mut target_tok = vec![0.0_f32; 128];

    for i in 0..n {
        id_builder.append_value(100 + i as u64);
        for (d, dst_v) in target_vec.iter_mut().enumerate().take(dim) {
            let v = if i == 5 {
                if d == 0 {
                    1.0
                } else {
                    0.0
                }
            } else {
                ((i * 13 + d) as f32 * 0.09).sin() * 0.1
            };
            if i == 5 {
                *dst_v = v;
            }
            vec_builder.values().append_value(v);
        }
        vec_builder.append(true);

        for t in 0..4 {
            for (d, dst_t) in target_tok.iter_mut().enumerate().take(128) {
                let tv = if i == 5 && t == 0 {
                    if d == 0 {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    ((i + t + d) as f32 * 0.04).cos() * 0.05
                };
                if i == 5 && t == 0 {
                    *dst_t = tv;
                }
                tok_builder.values().values().append_value(tv);
            }
            tok_builder.values().append(true);
        }
        tok_builder.append(true);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::UInt64, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        ),
        Field::new(
            "colbert_tokens",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 128),
                true,
            ))),
            false,
        ),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(id_builder.finish()),
            Arc::new(vec_builder.finish()),
            Arc::new(tok_builder.finish()),
        ],
    )
    .unwrap();
    let mut writer =
        ArrowWriter::try_new(BufWriter::new(File::create(path).unwrap()), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    (target_vec, target_tok)
}

#[test]
fn cli_doctor_json_reports_simd_and_bandwidth() {
    let out = Command::new(osprey_bin())
        .args(["--json", "doctor", "--bench-mb", "2"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let val: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(val["simd"]["scalar_supported"].as_bool().unwrap());
    assert!(val["memory"]["alignment_ok"].as_bool().unwrap());
    assert!(val["memory"]["read_gb_per_sec"].as_f64().unwrap() > 0.0);
}

#[test]
fn cli_full_pipeline_import_inspect_search_rerank_delete() {
    let dir = tempfile::tempdir().unwrap();
    let pq_path = dir.path().join("data.parquet");
    let seg_dir = dir.path().join("segments");
    let (target_vec, target_tok) = write_parquet_corpus(&pq_path, 24, 64);

    // 1. import
    let imp = Command::new(osprey_bin())
        .args([
            "--json",
            "import",
            "--input",
            pq_path.to_str().unwrap(),
            "--output",
            seg_dir.to_str().unwrap(),
            "--clusters",
            "4",
            "--page-size-kb",
            "16",
            "--with-lutrank",
            "--batch-size",
            "12",
            "--seed",
            "42",
        ])
        .output()
        .unwrap();
    assert!(
        imp.status.success(),
        "import stderr: {}",
        String::from_utf8_lossy(&imp.stderr)
    );
    let imp_json: serde_json::Value = serde_json::from_slice(&imp.stdout).unwrap();
    assert_eq!(imp_json["vectors_indexed"].as_u64(), Some(24));
    assert!(imp_json["osr_path"].is_string());

    let osi_path = seg_dir.join("segment.osi");
    let osv_path = seg_dir.join("segment.osv");
    let osr_path = seg_dir.join("segment.osr");

    // 2. inspect all three artifacts with --verify-checksums
    let insp = Command::new(osprey_bin())
        .args([
            "--json",
            "inspect",
            "--verify-checksums",
            "-f",
            osi_path.to_str().unwrap(),
            osv_path.to_str().unwrap(),
            osr_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        insp.status.success(),
        "inspect stderr: {}",
        String::from_utf8_lossy(&insp.stderr)
    );
    let insp_json: serde_json::Value = serde_json::from_slice(&insp.stdout).unwrap();
    assert_eq!(insp_json.as_array().unwrap().len(), 3);
    assert_eq!(insp_json[0]["format"], "osi");
    assert_eq!(insp_json[1]["format"], "osv");
    assert_eq!(insp_json[2]["format"], "osr");

    // 3. search with optional LUT-Rank rerank
    let q_json = serde_json::to_string(&target_vec).unwrap();
    let qt_json = serde_json::to_string(&vec![target_tok]).unwrap();
    let srch = Command::new(osprey_bin())
        .args([
            "--json",
            "search",
            "--index",
            osi_path.to_str().unwrap(),
            "--slabs",
            osv_path.to_str().unwrap(),
            "--query",
            &q_json,
            "--top-k",
            "5",
            "--nprobe",
            "4",
            "--candidates",
            "24",
            "--rerank-store",
            osr_path.to_str().unwrap(),
            "--query-tokens",
            &qt_json,
            "--rerank-top-m",
            "200",
        ])
        .output()
        .unwrap();
    assert!(
        srch.status.success(),
        "search stderr: {}",
        String::from_utf8_lossy(&srch.stderr)
    );
    let srch_json: serde_json::Value = serde_json::from_slice(&srch.stdout).unwrap();
    assert_eq!(srch_json["hits"][0]["doc_id"].as_u64(), Some(105));
    assert!(srch_json["telemetry"]["rerank_applied"].as_bool().unwrap());

    // 4. standalone CPU rerank
    let rrk = Command::new(osprey_bin())
        .args([
            "--json",
            "rerank",
            "--rerank-store",
            osr_path.to_str().unwrap(),
            "--query-tokens",
            &qt_json,
            "--doc-ids",
            "100,105,110",
            "--top-k",
            "2",
        ])
        .output()
        .unwrap();
    assert!(
        rrk.status.success(),
        "rerank stderr: {}",
        String::from_utf8_lossy(&rrk.stderr)
    );
    let rrk_json: serde_json::Value = serde_json::from_slice(&rrk.stdout).unwrap();
    assert_eq!(rrk_json["hits"][0]["doc_id"].as_u64(), Some(105));

    // 5. atomic delete doc 105 -> inspect shows 1 tombstone -> search no longer returns 105
    let del = Command::new(osprey_bin())
        .args([
            "--json",
            "delete",
            "--index",
            osi_path.to_str().unwrap(),
            "--ids",
            "105",
        ])
        .output()
        .unwrap();
    assert!(
        del.status.success(),
        "delete stderr: {}",
        String::from_utf8_lossy(&del.stderr)
    );
    let del_json: serde_json::Value = serde_json::from_slice(&del.stdout).unwrap();
    assert_eq!(del_json["total_tombstones"].as_u64(), Some(1));

    let srch_after = Command::new(osprey_bin())
        .args([
            "--json",
            "search",
            "--index",
            osi_path.to_str().unwrap(),
            "--slabs",
            osv_path.to_str().unwrap(),
            "--query",
            &q_json,
            "--top-k",
            "10",
            "--nprobe",
            "4",
            "--candidates",
            "24",
        ])
        .output()
        .unwrap();
    assert!(srch_after.status.success());
    let after_json: serde_json::Value = serde_json::from_slice(&srch_after.stdout).unwrap();
    let returned_ids: Vec<u64> = after_json["hits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| h["doc_id"].as_u64().unwrap())
        .collect();
    assert!(!returned_ids.contains(&105));

    // 6. import .npy file
    let npy_path = dir.path().join("vecs.npy");
    let npy_out = dir.path().join("npy_out");
    let data: Vec<f32> = (0..16 * 64).map(|i| (i as f32 * 0.01).sin()).collect();
    write_npy_f32(&npy_path, 16, 64, &data);

    let imp_npy = Command::new(osprey_bin())
        .args([
            "--json",
            "import",
            "--input",
            npy_path.to_str().unwrap(),
            "--output",
            npy_out.to_str().unwrap(),
            "--clusters",
            "2",
        ])
        .output()
        .unwrap();
    assert!(
        imp_npy.status.success(),
        "npy import stderr: {}",
        String::from_utf8_lossy(&imp_npy.stderr)
    );
    assert!(npy_out.join("segment.osi").exists());
    assert!(npy_out.join("segment.osv").exists());

    // 7. bench + executive Markdown report export
    let report_md = dir.path().join("exec_report.md");
    let bnc = Command::new(osprey_bin())
        .args([
            "--json",
            "bench",
            "--vectors",
            "500",
            "--dim",
            "128",
            "--queries",
            "8",
            "--clusters",
            "16",
            "--nprobe",
            "8",
            "--candidates",
            "64",
            "--with-lutrank",
            "--export-report",
            report_md.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        bnc.status.success(),
        "bench stderr: {}",
        String::from_utf8_lossy(&bnc.stderr)
    );
    let bnc_json: serde_json::Value = serde_json::from_slice(&bnc.stdout).unwrap();
    assert!(
        bnc_json["tco"]["pinecone"]["total_monthly_usd"]
            .as_f64()
            .unwrap()
            > 0.0
    );
    assert!(
        bnc_json["tco"]["qdrant"]["total_monthly_usd"]
            .as_f64()
            .unwrap()
            > 0.0
    );
    assert!(
        bnc_json["tco"]["ospreydb"]["total_monthly_usd"]
            .as_f64()
            .unwrap()
            > 0.0
    );
    assert!(report_md.exists());

    // 8. bench human output footer: RAM & S3 savings + non-intrusive Pro CTA
    let bnc_human = Command::new(osprey_bin())
        .args([
            "bench",
            "--vectors",
            "200",
            "--dim",
            "128",
            "--queries",
            "4",
            "--clusters",
            "8",
            "--nprobe",
            "4",
            "--candidates",
            "32",
        ])
        .output()
        .unwrap();
    assert!(
        bnc_human.status.success(),
        "bench human stderr: {}",
        String::from_utf8_lossy(&bnc_human.stderr)
    );
    let bnc_stdout = String::from_utf8_lossy(&bnc_human.stdout);
    assert!(bnc_stdout.contains("Estimated RAM & S3 Savings vs FP32:"));
    assert!(bnc_stdout.contains("RAM saved"));
    assert!(bnc_stdout.contains("S3 .osv"));
    assert!(bnc_stdout.contains(
        "100% Apache-2.0 Open Source — unlimited vectors, built-in /v1/embeddings, Qdrant/Cohere HTTP & Arrow Flight: osprey serve --data-dir ./data --s3-uri file:///tmp/osprey-s3"
    ));
}

#[test]
fn cli_version_includes_simd_backend_and_completions_generate_scripts() {
    let ver = Command::new(osprey_bin())
        .arg("--version")
        .output()
        .unwrap();
    assert!(ver.status.success());
    let ver_str = String::from_utf8_lossy(&ver.stdout);
    assert!(ver_str.contains("osprey"));
    assert!(ver_str.contains("simd:"));

    for shell in ["bash", "zsh", "fish"] {
        let comp = Command::new(osprey_bin())
            .args(["completions", shell])
            .output()
            .unwrap();
        assert!(
            comp.status.success(),
            "completions {shell} failed: {}",
            String::from_utf8_lossy(&comp.stderr)
        );
        let script = String::from_utf8_lossy(&comp.stdout);
        assert!(
            script.contains("osprey"),
            "completions {shell} output missing binary name"
        );
    }
}
