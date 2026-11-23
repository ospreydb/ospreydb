#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use osprey_core::RawVectorRecord;
use ospreydb::{LsmDeltaBuffer, OspreyWal, WalConfig, WalRecoveryMode, WalSyncPolicy};

static CRASH_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_wal_path(tag: &str) -> PathBuf {
    let n = CRASH_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "osprey_pro_crash_{}_{}_{}.wal",
        tag,
        std::process::id(),
        n
    ))
}

#[test]
fn sigkill_during_intense_wal_writes_recovers_100_percent_committed_state_in_ms() {
    let wal_path = temp_wal_path("sigkill");
    let _ = fs::remove_file(&wal_path);

    let bin = env!("CARGO_BIN_EXE_osprey");
    let dim = 64_usize;
    let batch_size = 50_usize;

    let mut child = Command::new(bin)
        .args([
            "wal-stress",
            "--wal-path",
            wal_path.to_str().unwrap(),
            "--dim",
            "64",
            "--batches",
            "0",
            "--batch-size",
            "50",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let stdout = child.stdout.take().unwrap();
    let mut reader = BufReader::new(stdout);
    let mut committed_batches = 0_usize;
    let mut last_committed_seq = 0_u64;
    let mut last_committed_doc = 0_u64;

    let mut line = String::new();
    while committed_batches < 25 {
        line.clear();
        let n = reader.read_line(&mut line).unwrap();
        assert!(n > 0, "wal-stress child exited prematurely");
        if let Some(rest) = line.trim().strip_prefix("COMMITTED ") {
            let parts: Vec<&str> = rest.split_whitespace().collect();
            assert_eq!(parts.len(), 2);
            last_committed_seq = parts[0].parse().unwrap();
            last_committed_doc = parts[1].parse().unwrap();
            committed_batches += 1;
        }
    }

    // Abruptly SIGKILL the osprey process while it continues hammering WAL writes!
    child.kill().unwrap();
    let status = child.wait().unwrap();
    assert!(!status.success());

    // Measure WAL replay + LsmDeltaBuffer reconstruction latency
    let t_recover = Instant::now();
    let (wal, replayed) = OspreyWal::open(
        &wal_path,
        WalConfig {
            recovery_mode: WalRecoveryMode::TruncateTornTail,
            sync_policy: WalSyncPolicy::SyncData,
        },
    )
    .unwrap();
    let delta =
        LsmDeltaBuffer::new(dim, dim, 0x4F53_5052_0001, Vec::new(), Some(Arc::new(wal))).unwrap();
    let applied = delta.recover_from_wal(&replayed).unwrap();
    let recovery_ms = t_recover.elapsed().as_secs_f64() * 1_000.0;

    eprintln!(
        "SIGKILL recovery: committed_batches={committed_batches}, last_committed_doc={last_committed_doc}, \
         replayed_records={}, recovery_ms={recovery_ms:.2} ms",
        replayed.len()
    );

    assert!(
        replayed.len() >= committed_batches * batch_size,
        "lost committed records: expected >= {}, got {}",
        committed_batches * batch_size,
        replayed.len()
    );
    assert_eq!(applied, replayed.len());
    assert!(wal_last_seq(&replayed) >= last_committed_seq);
    assert!(replayed.iter().any(|r| r.doc_id == last_committed_doc));

    // Verify every committed doc_id 1..=last_committed_doc is present and intact
    for expected_id in 1..=last_committed_doc {
        let rec = &replayed[usize::try_from(expected_id - 1).unwrap()];
        assert_eq!(rec.doc_id, expected_id);
        assert_eq!(rec.vector.len(), dim);
    }

    // Verify recovered state is immediately searchable
    let mut query = vec![0.0_f32; dim];
    let hot = usize::try_from(last_committed_doc).unwrap() % dim;
    query[hot] = 1.0;
    query[(hot + 1) % dim] = 0.25;
    let hits = delta.search_exact(&query, 5).unwrap();
    assert!(!hits.is_empty());

    if !cfg!(debug_assertions) {
        assert!(
            recovery_ms < 250.0,
            "expected millisecond-scale WAL recovery, took {recovery_ms:.2} ms"
        );
    }

    let _ = fs::remove_file(&wal_path);
}

#[test]
fn concurrent_high_load_wal_writes_and_torn_tail_recovery_in_milliseconds() {
    let wal_path = temp_wal_path("concurrent_load");
    let _ = fs::remove_file(&wal_path);

    let dim = 64_usize;
    let threads = 4_usize;
    let batches_per_thread = 15_usize;
    let docs_per_batch = 40_usize;
    let expected_total = threads * batches_per_thread * docs_per_batch;

    {
        let (wal, _) = OspreyWal::open(
            &wal_path,
            WalConfig {
                recovery_mode: WalRecoveryMode::TruncateTornTail,
                sync_policy: WalSyncPolicy::SyncData,
            },
        )
        .unwrap();
        let wal = Arc::new(wal);
        let delta = Arc::new(LsmDeltaBuffer::new(dim, dim, 31337, Vec::new(), Some(wal)).unwrap());

        std::thread::scope(|s| {
            for t in 0..threads {
                let d = Arc::clone(&delta);
                s.spawn(move || {
                    for b in 0..batches_per_thread {
                        let base_id = (t * batches_per_thread + b) * docs_per_batch + 1;
                        let vecs: Vec<Vec<f32>> = (0..docs_per_batch)
                            .map(|i| {
                                let mut v = vec![0.0_f32; dim];
                                v[(base_id + i) % dim] = 1.0;
                                v
                            })
                            .collect();
                        let payloads: Vec<Vec<u8>> = (0..docs_per_batch)
                            .map(|i| format!(r#"{{"id":{}}}"#, base_id + i).into_bytes())
                            .collect();
                        let batch: Vec<RawVectorRecord<'_>> = (0..docs_per_batch)
                            .map(|i| RawVectorRecord {
                                doc_id: (base_id + i) as u64,
                                vector: &vecs[i],
                                payload_json: &payloads[i],
                            })
                            .collect();
                        d.ingest_batch(&batch).unwrap();
                    }
                });
            }
        });
        assert_eq!(delta.live_count().unwrap(), expected_total);
    }

    // Simulate power outage mid-write by appending partial garbage bytes at the WAL tail
    {
        use std::io::Write as _;
        let mut f = fs::OpenOptions::new().append(true).open(&wal_path).unwrap();
        f.write_all(&[0x50, 0x00, 0x00, 0x00, 0x01, 0xAA, 0xBB, 0xCC])
            .unwrap();
        f.sync_all().unwrap();
    }

    let t0 = Instant::now();
    let (wal_reopened, replayed) = OspreyWal::open(&wal_path, WalConfig::default()).unwrap();
    let delta_rebuilt =
        LsmDeltaBuffer::new(dim, dim, 31337, Vec::new(), Some(Arc::new(wal_reopened))).unwrap();
    let applied = delta_rebuilt.recover_from_wal(&replayed).unwrap();
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1_000.0;

    eprintln!(
        "Concurrent load torn-tail recovery: {applied} records recovered in {elapsed_ms:.2} ms"
    );
    assert_eq!(applied, expected_total);
    assert_eq!(delta_rebuilt.live_count().unwrap(), expected_total);

    let _ = fs::remove_file(&wal_path);
}

fn wal_last_seq(records: &[ospreydb::WalRecord]) -> u64 {
    records.last().map_or(0, |r| r.seq_no)
}
