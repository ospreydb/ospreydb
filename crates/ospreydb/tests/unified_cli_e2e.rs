#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::process::Command;

fn osprey_bin() -> &'static str {
    env!("CARGO_BIN_EXE_osprey")
}

#[test]
fn osprey_unified_binary_exposes_all_subcommands_and_runs_cli_commands() {
    let help = Command::new(osprey_bin()).arg("--help").output().unwrap();
    assert!(help.status.success());
    let help_str = String::from_utf8_lossy(&help.stdout);
    for sub in [
        "serve",
        "import",
        "search",
        "rerank",
        "delete",
        "inspect",
        "bench",
        "doctor",
        "completions",
        "wal-stress",
    ] {
        assert!(
            help_str.contains(sub),
            "osprey --help missing subcommand '{sub}'"
        );
    }
    assert!(!help_str.contains("activate"));
    assert!(!help_str.contains("OSPREY_PRO_LICENSE"));

    let serve_help = Command::new(osprey_bin())
        .args(["serve", "--help"])
        .output()
        .unwrap();
    assert!(serve_help.status.success());
    let serve_help_str = String::from_utf8_lossy(&serve_help.stdout);
    assert!(!serve_help_str.contains("--license"));

    let doc = Command::new(osprey_bin())
        .args(["--json", "doctor"])
        .output()
        .unwrap();
    assert!(
        doc.status.success(),
        "osprey doctor stderr: {}",
        String::from_utf8_lossy(&doc.stderr)
    );
    let doc_json: serde_json::Value = serde_json::from_slice(&doc.stdout).unwrap();
    assert!(doc_json["simd"]["active_backend"].is_string());
    assert_eq!(doc_json["memory"]["alignment_ok"], true);

    let bench = Command::new(osprey_bin())
        .args([
            "--json",
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
        bench.status.success(),
        "osprey bench stderr: {}",
        String::from_utf8_lossy(&bench.stderr)
    );
    let bench_json: serde_json::Value = serde_json::from_slice(&bench.stdout).unwrap();
    assert_eq!(bench_json["local"]["vectors"], 200);

    let comp = Command::new(osprey_bin())
        .args(["completions", "bash"])
        .output()
        .unwrap();
    assert!(comp.status.success());
    let comp_str = String::from_utf8_lossy(&comp.stdout);
    assert!(comp_str.contains("osprey"));
    assert!(comp_str.contains("serve"));
    assert!(comp_str.contains("import"));
}
