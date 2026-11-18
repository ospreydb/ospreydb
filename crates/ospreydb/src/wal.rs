//! Crash-safe Write-Ahead Log (`OspreyWal`) with per-record `xxHash64` checksums
//! and batched `fsync` durability.
//!
//! Binary layout (strict Little-Endian):
//! - File header (8 bytes): magic `"OPWL"` (`[0x4F, 0x50, 0x57, 0x4C]`), `version: u16` (`1`),
//!   `flags: u16` (`0`).
//! - Record frames:
//!   - `body_len: u32` (`28 + dim * 4 + payload_len`)
//!   - `body`:
//!     - `op_kind: u8` (`1 = Upsert`, `2 = Delete`)
//!     - `reserved: [u8; 3]` (`[0; 3]`)
//!     - `seq_no: u64`
//!     - `doc_id: u64`
//!     - `dim: u32`
//!     - `payload_len: u32`
//!     - `vector_f32_le`: `dim * 4` bytes
//!     - `payload_json`: `payload_len` bytes
//!   - `record_xxh64: u64` (`xxh64` over `[body_len_le || body]`, seed `0`)

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use osprey_core::RawVectorRecord;
use xxhash_rust::xxh64::xxh64;

use crate::error::{ProEngineError, ProResult};

/// Magic prefix `"OPWL"` for `OspreyWal` files.
pub const WAL_MAGIC: [u8; 4] = [0x4F, 0x50, 0x57, 0x4C];

/// Current WAL binary format version.
pub const WAL_VERSION_V1: u16 = 1;

/// Fixed 8-byte WAL file header size.
pub const WAL_FILE_HEADER_BYTES: usize = 8;

/// Fixed record body header size (`op_kind(1) + reserved(3) + seq_no(8) + doc_id(8) + dim(4) + payload_len(4)`).
const RECORD_BODY_FIXED_BYTES: usize = 28;

/// Maximum single WAL record size (64 MiB) to guard against corrupted length prefixes.
const MAX_WAL_RECORD_BODY_BYTES: usize = 64 * 1024 * 1024;

/// Operation discriminant stored inside each WAL record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum WalOpKind {
    /// Insert or update a dense vector and its JSON payload.
    Upsert = 1,
    /// Logically delete a document by `doc_id`.
    Delete = 2,
}

impl WalOpKind {
    fn from_u8(val: u8, offset: u64) -> ProResult<Self> {
        match val {
            1 => Ok(Self::Upsert),
            2 => Ok(Self::Delete),
            _ => Err(ProEngineError::WalCorrupted {
                offset,
                reason: "unknown WAL operation discriminant",
            }),
        }
    }
}

/// Owned record replayed from [`OspreyWal::open`].
#[derive(Clone, Debug, PartialEq)]
pub struct WalRecord {
    /// Monotonic log sequence number.
    pub seq_no: u64,
    /// Operation kind (`Upsert` or `Delete`).
    pub op: WalOpKind,
    /// Logical document identifier.
    pub doc_id: u64,
    /// Dense `f32` vector (`empty` for `Delete`).
    pub vector: Vec<f32>,
    /// Raw UTF-8 JSON metadata payload (`empty` for `Delete`).
    pub payload_json: Vec<u8>,
}

/// Recovery behavior when an incomplete or corrupted record is encountered during replay.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WalRecoveryMode {
    /// Truncate incomplete/corrupted tail frames back to the last valid checksummed frame.
    #[default]
    TruncateTornTail,
    /// Fail immediately with [`ProEngineError::WalChecksumMismatch`] or [`ProEngineError::WalCorrupted`].
    Strict,
}

/// Durability sync policy executed at the end of every WAL batch write.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WalSyncPolicy {
    /// Sync file data (`File::sync_data`) per batch.
    #[default]
    SyncData,
    /// Full metadata + data sync (`File::sync_all`) per batch.
    SyncAll,
    /// Flush OS buffer without `fsync` (intended only for non-durable tests/benchmarks).
    None,
}

/// Configuration for [`OspreyWal`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WalConfig {
    /// Recovery mode on WAL open.
    pub recovery_mode: WalRecoveryMode,
    /// Batch durability sync policy.
    pub sync_policy: WalSyncPolicy,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            recovery_mode: WalRecoveryMode::TruncateTornTail,
            sync_policy: WalSyncPolicy::SyncData,
        }
    }
}

#[derive(Debug)]
struct WalInner {
    file: File,
    next_seq_no: u64,
    bytes_written: u64,
    scratch: Vec<u8>,
}

/// Thread-safe, batched-`fsync` Write-Ahead Log protected by per-record `xxHash64` checksums.
#[derive(Debug)]
pub struct OspreyWal {
    path: PathBuf,
    config: WalConfig,
    inner: Mutex<WalInner>,
}

impl OspreyWal {
    /// Opens or creates a WAL file at `path`, replaying all durable records.
    pub fn open<P: AsRef<Path>>(path: P, config: WalConfig) -> ProResult<(Self, Vec<WalRecord>)> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;

        let file_len = file.metadata()?.len();
        let mut records = Vec::new();
        let mut max_seq_no = 0_u64;
        let mut valid_len = WAL_FILE_HEADER_BYTES as u64;

        if file_len == 0 {
            write_wal_header(&mut file)?;
            sync_file(&file, config.sync_policy)?;
        } else {
            let mut raw = Vec::with_capacity(usize::try_from(file_len).unwrap_or(0));
            file.seek(SeekFrom::Start(0))?;
            file.read_to_end(&mut raw)?;
            validate_wal_header(&raw)?;

            let (replayed, last_valid_offset) = replay_frames(&raw, config.recovery_mode)?;
            if last_valid_offset < raw.len() {
                let discarded_bytes = raw.len().saturating_sub(last_valid_offset);
                tracing::warn!(
                    path = %path.display(),
                    discarded_bytes,
                    last_valid_offset,
                    replayed_records = replayed.len(),
                    "truncated incomplete or corrupted trailing WAL bytes during recovery"
                );
                let target_len =
                    u64::try_from(last_valid_offset).map_err(|_| ProEngineError::WalCorrupted {
                        offset: 0,
                        reason: "WAL length overflow",
                    })?;
                file.set_len(target_len)?;
                file.sync_all()?;
            }
            if let Some(last) = replayed.last() {
                max_seq_no = last.seq_no;
            }
            valid_len = u64::try_from(last_valid_offset).unwrap_or(WAL_FILE_HEADER_BYTES as u64);
            file.seek(SeekFrom::Start(valid_len))?;
            records = replayed;
        }

        let wal = Self {
            path,
            config,
            inner: Mutex::new(WalInner {
                file,
                next_seq_no: max_seq_no.saturating_add(1),
                bytes_written: valid_len,
                scratch: Vec::with_capacity(64 * 1024),
            }),
        };
        Ok((wal, records))
    }

    /// Returns the filesystem path of this WAL file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the last assigned monotonic sequence number (`0` if empty).
    pub fn last_seq_no(&self) -> ProResult<u64> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| ProEngineError::Io(std::io::Error::other("WAL mutex poisoned")))?;
        Ok(guard.next_seq_no.saturating_sub(1))
    }

    /// Returns total valid bytes currently persisted in the WAL file.
    pub fn bytes_written(&self) -> ProResult<u64> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| ProEngineError::Io(std::io::Error::other("WAL mutex poisoned")))?;
        Ok(guard.bytes_written)
    }

    /// Appends a batch of vector upserts into the WAL in a single write + `fsync`,
    /// returning the highest assigned sequence number.
    pub fn append_upserts(&self, records: &[RawVectorRecord<'_>]) -> ProResult<u64> {
        if records.is_empty() {
            return self.last_seq_no();
        }
        for rec in records {
            for (idx, &val) in rec.vector.iter().enumerate() {
                if !val.is_finite() {
                    return Err(osprey_core::OspreyError::NonFiniteEmbedding { index: idx }.into());
                }
            }
        }

        let mut guard = self
            .inner
            .lock()
            .map_err(|_| ProEngineError::Io(std::io::Error::other("WAL mutex poisoned")))?;

        let mut next_seq = guard.next_seq_no;
        let mut buf = std::mem::take(&mut guard.scratch);
        buf.clear();

        for rec in records {
            encode_wal_frame(
                &mut buf,
                WalOpKind::Upsert,
                next_seq,
                rec.doc_id,
                rec.vector,
                rec.payload_json,
            )?;
            next_seq = next_seq.saturating_add(1);
        }

        let written_bytes = u64::try_from(buf.len()).unwrap_or(0);
        let write_res = guard
            .file
            .write_all(&buf)
            .and_then(|()| guard.file.flush())
            .and_then(|()| sync_file(&guard.file, self.config.sync_policy).map_err(to_io_err));

        guard.scratch = buf;
        write_res?;

        guard.next_seq_no = next_seq;
        guard.bytes_written = guard.bytes_written.saturating_add(written_bytes);
        Ok(next_seq.saturating_sub(1))
    }

    /// Appends a batch of logical deletions (`doc_id`s) into the WAL with batched `fsync`.
    pub fn append_deletes(&self, doc_ids: &[u64]) -> ProResult<u64> {
        if doc_ids.is_empty() {
            return self.last_seq_no();
        }

        let mut guard = self
            .inner
            .lock()
            .map_err(|_| ProEngineError::Io(std::io::Error::other("WAL mutex poisoned")))?;

        let mut next_seq = guard.next_seq_no;
        let mut buf = std::mem::take(&mut guard.scratch);
        buf.clear();

        for &doc_id in doc_ids {
            encode_wal_frame(&mut buf, WalOpKind::Delete, next_seq, doc_id, &[], &[])?;
            next_seq = next_seq.saturating_add(1);
        }

        let written_bytes = u64::try_from(buf.len()).unwrap_or(0);
        let write_res = guard
            .file
            .write_all(&buf)
            .and_then(|()| guard.file.flush())
            .and_then(|()| sync_file(&guard.file, self.config.sync_policy).map_err(to_io_err));

        guard.scratch = buf;
        write_res?;

        guard.next_seq_no = next_seq;
        guard.bytes_written = guard.bytes_written.saturating_add(written_bytes);
        Ok(next_seq.saturating_sub(1))
    }

    /// Truncates the WAL cleanly back to its 8-byte header after compaction checkpointing.
    pub fn reset(&self) -> ProResult<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| ProEngineError::Io(std::io::Error::other("WAL mutex poisoned")))?;
        guard.file.set_len(0)?;
        guard.file.seek(SeekFrom::Start(0))?;
        write_wal_header(&mut guard.file)?;
        sync_file(&guard.file, self.config.sync_policy)?;
        guard.bytes_written = WAL_FILE_HEADER_BYTES as u64;
        Ok(())
    }

    /// Truncates all WAL records with `seq_no <= watermark_seq`, preserving any concurrent
    /// writes appended after `watermark_seq` during background compaction.
    pub fn retain_after_seq(&self, watermark_seq: u64) -> ProResult<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| ProEngineError::Io(std::io::Error::other("WAL mutex poisoned")))?;

        if guard.next_seq_no.saturating_sub(1) <= watermark_seq {
            guard.file.set_len(0)?;
            guard.file.seek(SeekFrom::Start(0))?;
            write_wal_header(&mut guard.file)?;
            sync_file(&guard.file, self.config.sync_policy)?;
            guard.bytes_written = WAL_FILE_HEADER_BYTES as u64;
            return Ok(());
        }

        let mut raw = Vec::with_capacity(usize::try_from(guard.bytes_written).unwrap_or(0));
        guard.file.seek(SeekFrom::Start(0))?;
        guard.file.read_to_end(&mut raw)?;
        validate_wal_header(&raw)?;
        let (replayed, _) = replay_frames(&raw, WalRecoveryMode::TruncateTornTail)?;

        let mut buf = std::mem::take(&mut guard.scratch);
        buf.clear();
        for rec in replayed.into_iter().filter(|r| r.seq_no > watermark_seq) {
            encode_wal_frame(
                &mut buf,
                rec.op,
                rec.seq_no,
                rec.doc_id,
                &rec.vector,
                &rec.payload_json,
            )?;
        }

        guard.file.set_len(0)?;
        guard.file.seek(SeekFrom::Start(0))?;
        write_wal_header(&mut guard.file)?;
        if !buf.is_empty() {
            guard.file.write_all(&buf)?;
            guard.file.flush()?;
        }
        sync_file(&guard.file, self.config.sync_policy)?;
        guard.bytes_written =
            (WAL_FILE_HEADER_BYTES as u64).saturating_add(u64::try_from(buf.len()).unwrap_or(0));
        guard.scratch = buf;
        Ok(())
    }

    /// Flushes and `fsync`s the WAL file descriptor (used during graceful shutdown).
    pub fn sync(&self) -> ProResult<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| ProEngineError::Io(std::io::Error::other("WAL mutex poisoned")))?;
        guard.file.flush()?;
        guard.file.sync_all()?;
        Ok(())
    }
}

fn to_io_err(err: ProEngineError) -> std::io::Error {
    match err {
        ProEngineError::Io(io) => io,
        other => std::io::Error::other(other),
    }
}

fn write_wal_header(file: &mut File) -> ProResult<()> {
    let mut hdr = [0_u8; WAL_FILE_HEADER_BYTES];
    hdr[0..4].copy_from_slice(&WAL_MAGIC);
    hdr[4..6].copy_from_slice(&WAL_VERSION_V1.to_le_bytes());
    hdr[6..8].copy_from_slice(&0_u16.to_le_bytes());
    file.write_all(&hdr)?;
    file.flush()?;
    Ok(())
}

fn validate_wal_header(bytes: &[u8]) -> ProResult<()> {
    if bytes.len() < WAL_FILE_HEADER_BYTES {
        return Err(ProEngineError::WalCorrupted {
            offset: 0,
            reason: "truncated WAL file header",
        });
    }
    let magic: [u8; 4] = bytes[0..4]
        .try_into()
        .map_err(|_| ProEngineError::WalCorrupted {
            offset: 0,
            reason: "missing WAL magic",
        })?;
    if magic != WAL_MAGIC {
        return Err(ProEngineError::InvalidWalMagic { found: magic });
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != WAL_VERSION_V1 {
        return Err(ProEngineError::UnsupportedWalVersion {
            expected: WAL_VERSION_V1,
            found: version,
        });
    }
    Ok(())
}

fn sync_file(file: &File, policy: WalSyncPolicy) -> ProResult<()> {
    match policy {
        WalSyncPolicy::SyncData => file.sync_data()?,
        WalSyncPolicy::SyncAll => file.sync_all()?,
        WalSyncPolicy::None => {}
    }
    Ok(())
}

fn encode_wal_frame(
    out: &mut Vec<u8>,
    op: WalOpKind,
    seq_no: u64,
    doc_id: u64,
    vector: &[f32],
    payload_json: &[u8],
) -> ProResult<()> {
    let dim_u32 = u32::try_from(vector.len()).map_err(|_| ProEngineError::DimensionMismatch {
        expected: usize::from(u16::MAX),
        actual: vector.len(),
    })?;
    let payload_len_u32 = u32::try_from(payload_json.len())
        .map_err(|_| ProEngineError::InvalidConfig("WAL payload exceeds u32 length"))?;

    let vec_bytes_len = vector
        .len()
        .checked_mul(4)
        .ok_or(ProEngineError::InvalidConfig("WAL vector byte overflow"))?;
    let body_len = RECORD_BODY_FIXED_BYTES
        .checked_add(vec_bytes_len)
        .and_then(|n| n.checked_add(payload_json.len()))
        .ok_or(ProEngineError::InvalidConfig("WAL record length overflow"))?;
    if body_len > MAX_WAL_RECORD_BODY_BYTES {
        return Err(ProEngineError::InvalidConfig(
            "WAL record exceeds MAX_WAL_RECORD_BODY_BYTES",
        ));
    }
    let body_len_u32 = u32::try_from(body_len)
        .map_err(|_| ProEngineError::InvalidConfig("WAL record body length exceeds u32::MAX"))?;

    let frame_start = out.len();
    out.reserve(4 + body_len + 8);
    out.extend_from_slice(&body_len_u32.to_le_bytes());
    out.push(op as u8);
    out.extend_from_slice(&[0_u8; 3]);
    out.extend_from_slice(&seq_no.to_le_bytes());
    out.extend_from_slice(&doc_id.to_le_bytes());
    out.extend_from_slice(&dim_u32.to_le_bytes());
    out.extend_from_slice(&payload_len_u32.to_le_bytes());
    for &val in vector {
        out.extend_from_slice(&val.to_le_bytes());
    }
    out.extend_from_slice(payload_json);

    let checksum = xxh64(&out[frame_start..], 0);
    out.extend_from_slice(&checksum.to_le_bytes());
    Ok(())
}

fn replay_frames(bytes: &[u8], mode: WalRecoveryMode) -> ProResult<(Vec<WalRecord>, usize)> {
    let mut cursor = WAL_FILE_HEADER_BYTES;
    let mut records = Vec::new();

    while cursor < bytes.len() {
        let frame_offset = u64::try_from(cursor).unwrap_or(u64::MAX);
        if bytes.len() - cursor < 4 {
            return handle_corrupt(
                mode,
                records,
                cursor,
                ProEngineError::WalCorrupted {
                    offset: frame_offset,
                    reason: "truncated record length prefix",
                },
            );
        }

        let body_len_u32 = u32::from_le_bytes([
            bytes[cursor],
            bytes[cursor + 1],
            bytes[cursor + 2],
            bytes[cursor + 3],
        ]);
        let body_len = usize::try_from(body_len_u32).unwrap_or(usize::MAX);
        if !(RECORD_BODY_FIXED_BYTES..=MAX_WAL_RECORD_BODY_BYTES).contains(&body_len) {
            return handle_corrupt(
                mode,
                records,
                cursor,
                ProEngineError::WalCorrupted {
                    offset: frame_offset,
                    reason: "invalid WAL record body length",
                },
            );
        }

        let total_frame_len = match 4_usize.checked_add(body_len).and_then(|n| n.checked_add(8)) {
            Some(len) => len,
            None => {
                return handle_corrupt(
                    mode,
                    records,
                    cursor,
                    ProEngineError::WalCorrupted {
                        offset: frame_offset,
                        reason: "WAL frame size overflow",
                    },
                );
            }
        };

        if bytes.len() - cursor < total_frame_len {
            return handle_corrupt(
                mode,
                records,
                cursor,
                ProEngineError::WalCorrupted {
                    offset: frame_offset,
                    reason: "truncated WAL record frame",
                },
            );
        }

        let hash_start = cursor + 4 + body_len;
        let expected_hash = u64::from_le_bytes([
            bytes[hash_start],
            bytes[hash_start + 1],
            bytes[hash_start + 2],
            bytes[hash_start + 3],
            bytes[hash_start + 4],
            bytes[hash_start + 5],
            bytes[hash_start + 6],
            bytes[hash_start + 7],
        ]);
        let computed_hash = xxh64(&bytes[cursor..hash_start], 0);
        if expected_hash != computed_hash {
            return handle_corrupt(
                mode,
                records,
                cursor,
                ProEngineError::WalChecksumMismatch {
                    offset: frame_offset,
                    expected: expected_hash,
                    computed: computed_hash,
                },
            );
        }

        let body = &bytes[cursor + 4..hash_start];
        let op = match WalOpKind::from_u8(body[0], frame_offset) {
            Ok(op) => op,
            Err(err) => return handle_corrupt(mode, records, cursor, err),
        };
        let seq_no = u64::from_le_bytes([
            body[4], body[5], body[6], body[7], body[8], body[9], body[10], body[11],
        ]);
        let doc_id = u64::from_le_bytes([
            body[12], body[13], body[14], body[15], body[16], body[17], body[18], body[19],
        ]);
        let dim = u32::from_le_bytes([body[20], body[21], body[22], body[23]]);
        let payload_len = u32::from_le_bytes([body[24], body[25], body[26], body[27]]);

        let dim_usize = usize::try_from(dim).unwrap_or(usize::MAX);
        let payload_len_usize = usize::try_from(payload_len).unwrap_or(usize::MAX);
        let expected_body_len = dim_usize
            .checked_mul(4)
            .and_then(|n| n.checked_add(RECORD_BODY_FIXED_BYTES))
            .and_then(|n| n.checked_add(payload_len_usize));
        if expected_body_len != Some(body_len) {
            return handle_corrupt(
                mode,
                records,
                cursor,
                ProEngineError::WalCorrupted {
                    offset: frame_offset,
                    reason: "mismatched vector/payload lengths inside WAL frame",
                },
            );
        }

        let vec_end = RECORD_BODY_FIXED_BYTES + dim_usize * 4;
        let vec_bytes = &body[RECORD_BODY_FIXED_BYTES..vec_end];
        let (chunks, _) = vec_bytes.as_chunks::<4>();
        let mut vector = Vec::with_capacity(dim_usize);
        for &chunk in chunks {
            vector.push(f32::from_le_bytes(chunk));
        }
        let payload_json = body[vec_end..].to_vec();

        records.push(WalRecord {
            seq_no,
            op,
            doc_id,
            vector,
            payload_json,
        });

        cursor += total_frame_len;
    }

    Ok((records, cursor))
}

fn handle_corrupt(
    mode: WalRecoveryMode,
    records: Vec<WalRecord>,
    last_valid_offset: usize,
    err: ProEngineError,
) -> ProResult<(Vec<WalRecord>, usize)> {
    match mode {
        WalRecoveryMode::TruncateTornTail => Ok((records, last_valid_offset)),
        WalRecoveryMode::Strict => Err(err),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_wal_path(name: &str) -> PathBuf {
        let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "osprey_pro_wal_test_{}_{}_{}.wal",
            name,
            std::process::id(),
            id
        ))
    }

    #[test]
    fn wal_batch_upsert_delete_and_replay() {
        let path = temp_wal_path("replay");
        let _ = fs::remove_file(&path);

        {
            let (wal, initial) = OspreyWal::open(&path, WalConfig::default()).unwrap();
            assert!(initial.is_empty());

            let v1 = [1.0_f32, 2.0, 3.0, 4.0];
            let v2 = [5.0_f32, 6.0, 7.0, 8.0];
            let seq = wal
                .append_upserts(&[
                    RawVectorRecord {
                        doc_id: 10,
                        vector: &v1,
                        payload_json: br#"{"k":"a"}"#,
                    },
                    RawVectorRecord {
                        doc_id: 20,
                        vector: &v2,
                        payload_json: br#"{"k":"b"}"#,
                    },
                ])
                .unwrap();
            assert_eq!(seq, 2);

            let del_seq = wal.append_deletes(&[10]).unwrap();
            assert_eq!(del_seq, 3);
        }

        let (wal2, replayed) = OspreyWal::open(&path, WalConfig::default()).unwrap();
        assert_eq!(replayed.len(), 3);
        assert_eq!(replayed[0].doc_id, 10);
        assert_eq!(replayed[0].op, WalOpKind::Upsert);
        assert_eq!(replayed[0].vector, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(replayed[1].doc_id, 20);
        assert_eq!(replayed[2].doc_id, 10);
        assert_eq!(replayed[2].op, WalOpKind::Delete);
        assert_eq!(wal2.last_seq_no().unwrap(), 3);

        wal2.reset().unwrap();
        let (_, after_reset) = OspreyWal::open(&path, WalConfig::default()).unwrap();
        assert!(after_reset.is_empty());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn wal_detects_bit_flip_in_strict_mode_and_recovers_torn_tail() {
        let path = temp_wal_path("corrupt");
        let _ = fs::remove_file(&path);

        {
            let (wal, _) = OspreyWal::open(&path, WalConfig::default()).unwrap();
            let v1 = [1.0_f32; 8];
            let v2 = [2.0_f32; 8];
            wal.append_upserts(&[
                RawVectorRecord {
                    doc_id: 1,
                    vector: &v1,
                    payload_json: b"{}",
                },
                RawVectorRecord {
                    doc_id: 2,
                    vector: &v2,
                    payload_json: b"{}",
                },
            ])
            .unwrap();
        }

        // Flip 1 bit near the end of record 2 (torn/corrupted tail)
        let mut bytes = fs::read(&path).unwrap();
        let last_idx = bytes.len() - 10;
        bytes[last_idx] ^= 0x01;
        fs::write(&path, &bytes).unwrap();

        let strict_res = OspreyWal::open(
            &path,
            WalConfig {
                recovery_mode: WalRecoveryMode::Strict,
                sync_policy: WalSyncPolicy::SyncData,
            },
        );
        assert!(matches!(
            strict_res,
            Err(ProEngineError::WalChecksumMismatch { .. })
        ));

        // Torn-tail recovery preserves record 1 and truncates corrupted record 2
        let (_, recovered) = OspreyWal::open(&path, WalConfig::default()).unwrap();
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].doc_id, 1);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn wal_truncates_incomplete_final_frame_and_preserves_prior_records() {
        let path = temp_wal_path("truncated_tail");
        let _ = fs::remove_file(&path);

        let valid_prefix_len: usize;
        {
            let (wal, _) = OspreyWal::open(&path, WalConfig::default()).unwrap();
            let v1 = [1.0_f32; 8];
            let v2 = [2.0_f32; 8];
            wal.append_upserts(&[RawVectorRecord {
                doc_id: 101,
                vector: &v1,
                payload_json: br#"{"k":1}"#,
            }])
            .unwrap();
            valid_prefix_len = usize::try_from(fs::metadata(&path).unwrap().len()).unwrap();
            wal.append_upserts(&[RawVectorRecord {
                doc_id: 102,
                vector: &v2,
                payload_json: br#"{"k":2}"#,
            }])
            .unwrap();
        }

        // Simulate crash mid-write of the last record by chopping off its final 15 bytes (EOF inside frame)
        let full_len = usize::try_from(fs::metadata(&path).unwrap().len()).unwrap();
        assert!(full_len > valid_prefix_len + 15);
        {
            let f = OpenOptions::new().write(true).open(&path).unwrap();
            f.set_len((full_len - 15) as u64).unwrap();
        }

        let (wal2, replayed) = OspreyWal::open(&path, WalConfig::default()).unwrap();
        assert_eq!(replayed.len(), 1);
        assert_eq!(replayed[0].doc_id, 101);
        assert_eq!(
            usize::try_from(fs::metadata(&path).unwrap().len()).unwrap(),
            valid_prefix_len
        );

        // Appending new records after recovery works cleanly
        let v3 = [3.0_f32; 8];
        let seq = wal2
            .append_upserts(&[RawVectorRecord {
                doc_id: 103,
                vector: &v3,
                payload_json: br#"{"k":3}"#,
            }])
            .unwrap();
        assert_eq!(seq, 2);

        let (_, replayed2) = OspreyWal::open(&path, WalConfig::default()).unwrap();
        assert_eq!(replayed2.len(), 2);
        assert_eq!(replayed2[0].doc_id, 101);
        assert_eq!(replayed2[1].doc_id, 103);

        let _ = fs::remove_file(&path);
    }
}
