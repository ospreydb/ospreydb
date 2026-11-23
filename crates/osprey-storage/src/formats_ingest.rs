//! Streaming readers for `.jsonl` (JSON Lines) and native NumPy `.npy` (v1.0 / v2.0) matrices.

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use half::f16;
use osprey_core::{OspreyError, OspreyResult};

use crate::lake_ingest::map_string_id_to_u64;

const NPY_MAGIC: [u8; 6] = [0x93, b'N', b'U', b'M', b'P', b'Y'];
const MAX_NPY_HEADER_BYTES: usize = 16 * 1024 * 1024;

#[inline]
fn invalid_data(msg: impl Into<String>) -> OspreyError {
    OspreyError::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        msg.into(),
    ))
}

/// Parsed record produced by [`JsonlRecordReader`].
#[derive(Clone, Debug, PartialEq)]
pub struct JsonlParsedRecord {
    /// Document identifier (`u64` integer directly, or `xxh64(str_bytes, 0)` for string IDs).
    pub doc_id: u64,
    /// Dense embedding vector (`f32`, finite-validated).
    pub vector: Vec<f32>,
    /// Optional ColBERT multi-vector token embeddings (`Vec<Vec<f32>>`).
    pub colbert_tokens: Option<Vec<Vec<f32>>>,
    /// Remaining JSON object fields serialized as compact UTF-8 JSON bytes.
    pub metadata_payload: Vec<u8>,
}

/// Streaming `.jsonl` (JSON Lines) reader using `std::io::BufRead` + `serde_json`.
///
/// Extracts `id` (or `doc_id`), `vector` (or `embedding`), optional `tokens` (or
/// `colbert_vectors` / `colbert`), and preserves all remaining object fields as a JSON
/// metadata payload.
pub struct JsonlRecordReader<R: BufRead> {
    reader: R,
    line_buf: String,
    line_number: usize,
}

impl JsonlRecordReader<BufReader<File>> {
    /// Opens a `.jsonl` file on disk for buffered streaming.
    pub fn open(path: impl AsRef<Path>) -> OspreyResult<Self> {
        let file = File::open(path)?;
        Ok(Self::new(BufReader::new(file)))
    }
}

impl<R: BufRead> JsonlRecordReader<R> {
    /// Creates a [`JsonlRecordReader`] wrapping any [`BufRead`] source.
    #[inline]
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            line_buf: String::new(),
            line_number: 0,
        }
    }

    /// Returns the 1-based line number of the most recently read line.
    #[inline]
    pub fn lines_read(&self) -> usize {
        self.line_number
    }

    /// Reads up to `max_rows` records into a chunk vector for streaming batch ingestion.
    pub fn read_batch(&mut self, max_rows: usize) -> OspreyResult<Vec<JsonlParsedRecord>> {
        let cap = max_rows.max(1);
        let mut batch = Vec::with_capacity(cap.min(4096));
        while batch.len() < cap {
            match self.next_record()? {
                Some(rec) => batch.push(rec),
                None => break,
            }
        }
        Ok(batch)
    }

    /// Parses and returns the next non-empty JSONL line, or `None` at EOF.
    pub fn next_record(&mut self) -> OspreyResult<Option<JsonlParsedRecord>> {
        loop {
            self.line_buf.clear();
            let bytes_read = self.reader.read_line(&mut self.line_buf)?;
            if bytes_read == 0 {
                return Ok(None);
            }
            self.line_number += 1;
            let trimmed = self.line_buf.trim();
            if trimmed.is_empty() {
                continue;
            }

            let mut value: serde_json::Value = serde_json::from_str(trimmed).map_err(|err| {
                invalid_data(format!("invalid JSON on line {}: {err}", self.line_number))
            })?;

            let obj = value.as_object_mut().ok_or_else(|| {
                invalid_data(format!("expected JSON object on line {}", self.line_number))
            })?;

            let id_val = obj
                .remove("id")
                .or_else(|| obj.remove("doc_id"))
                .ok_or_else(|| {
                    invalid_data(format!(
                        "missing 'id' or 'doc_id' field on line {}",
                        self.line_number
                    ))
                })?;
            let doc_id = parse_json_doc_id(&id_val, self.line_number)?;

            let vec_val = obj
                .remove("vector")
                .or_else(|| obj.remove("embedding"))
                .ok_or_else(|| {
                    invalid_data(format!(
                        "missing 'vector' or 'embedding' field on line {}",
                        self.line_number
                    ))
                })?;
            let vector = parse_json_f32_vec(&vec_val, self.line_number)?;

            let tokens_val = obj
                .remove("tokens")
                .or_else(|| obj.remove("colbert_vectors"))
                .or_else(|| obj.remove("colbert"));
            let colbert_tokens = match tokens_val {
                Some(serde_json::Value::Null) | None => None,
                Some(v) => Some(parse_json_colbert_tokens(&v, self.line_number)?),
            };

            let metadata_payload = serde_json::to_vec(obj).map_err(|err| {
                invalid_data(format!(
                    "failed to serialize JSON metadata on line {}: {err}",
                    self.line_number
                ))
            })?;

            return Ok(Some(JsonlParsedRecord {
                doc_id,
                vector,
                colbert_tokens,
                metadata_payload,
            }));
        }
    }
}

impl<R: BufRead> Iterator for JsonlRecordReader<R> {
    type Item = OspreyResult<JsonlParsedRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_record() {
            Ok(Some(rec)) => Some(Ok(rec)),
            Ok(None) => None,
            Err(err) => Some(Err(err)),
        }
    }
}

fn parse_json_doc_id(val: &serde_json::Value, line: usize) -> OspreyResult<u64> {
    match val {
        serde_json::Value::Number(n) => n.as_u64().ok_or_else(|| {
            invalid_data(format!(
                "numeric ID on line {line} must be an unsigned 64-bit integer"
            ))
        }),
        serde_json::Value::String(s) => Ok(map_string_id_to_u64(s)),
        other => Err(invalid_data(format!(
            "unsupported ID type {other:?} on line {line}"
        ))),
    }
}

fn parse_json_f32_vec(val: &serde_json::Value, line: usize) -> OspreyResult<Vec<f32>> {
    let arr = val.as_array().ok_or_else(|| {
        invalid_data(format!(
            "expected JSON array for embedding vector on line {line}"
        ))
    })?;
    if arr.is_empty() {
        return Err(OspreyError::InvalidDimension { dim: 0 });
    }
    let mut out = Vec::with_capacity(arr.len());
    for (idx, item) in arr.iter().enumerate() {
        let f64_val = item.as_f64().ok_or_else(|| {
            invalid_data(format!(
                "non-numeric embedding coordinate at index {idx} on line {line}"
            ))
        })?;
        #[allow(clippy::cast_possible_truncation)]
        let f32_val = f64_val as f32;
        if !f32_val.is_finite() {
            return Err(OspreyError::NonFiniteEmbedding { index: idx });
        }
        out.push(f32_val);
    }
    Ok(out)
}

fn parse_json_colbert_tokens(val: &serde_json::Value, line: usize) -> OspreyResult<Vec<Vec<f32>>> {
    let outer = val.as_array().ok_or_else(|| {
        invalid_data(format!(
            "expected JSON array of token vectors on line {line}"
        ))
    })?;
    let mut tokens = Vec::with_capacity(outer.len());
    let mut expected_dim: Option<usize> = None;
    for tok_val in outer {
        let tok = parse_json_f32_vec(tok_val, line)?;
        match expected_dim {
            Some(d) if d != tok.len() => {
                return Err(OspreyError::InvalidDimension { dim: tok.len() });
            }
            None => expected_dim = Some(tok.len()),
            Some(_) => {}
        }
        tokens.push(tok);
    }
    Ok(tokens)
}

// -----------------------------------------------------------------------------
// Native Zero-Dependency NumPy `.npy` (v1.0 / v2.0) Streaming Matrix Reader
// -----------------------------------------------------------------------------

/// Supported little-endian floating-point scalar dtypes in `.npy` files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NpyDtype {
    /// IEEE-754 little-endian `float32` (`'<f4'`).
    F32,
    /// IEEE-754 little-endian `float16` (`'<f2'`).
    F16,
}

impl NpyDtype {
    /// Element byte size (`4` for `<f4`, `2` for `<f2`).
    #[inline]
    pub const fn byte_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }
}

/// Parsed `.npy` v1.0 / v2.0 matrix header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NpyHeader {
    /// `.npy` format major version (`1` or `2`).
    pub major_version: u8,
    /// `.npy` format minor version.
    pub minor_version: u8,
    /// Element scalar dtype (`<f4` or `<f2`).
    pub dtype: NpyDtype,
    /// Number of rows `N` in `(N, D)`.
    pub num_rows: usize,
    /// Vector dimensionality `D` in `(N, D)`.
    pub dim: usize,
}

/// Batch of `f32` vectors streamed from [`NpyMatrixReader`].
#[derive(Clone, Debug, PartialEq)]
pub struct NpyVectorBatch {
    /// Row-index document IDs (`doc_id = row_idx as u64`).
    pub doc_ids: Vec<u64>,
    /// Flat row-major `f32` vectors (`doc_ids.len() * dim`).
    pub vectors_f32: Vec<f32>,
}

/// Batch of `half::f16` vectors streamed from [`NpyMatrixReader`].
#[derive(Clone, Debug, PartialEq)]
pub struct NpyF16Batch {
    /// Row-index document IDs (`doc_id = row_idx as u64`).
    pub doc_ids: Vec<u64>,
    /// Flat row-major `half::f16` vectors (`doc_ids.len() * dim`).
    pub vectors_f16: Vec<f16>,
}

/// Native zero-dependency streaming reader for NumPy `.npy` v1.0 / v2.0 2D matrices
/// (`'<f4'` / `'<f2'`, C-contiguous `fortran_order: False`).
pub struct NpyMatrixReader<R: Read> {
    reader: R,
    header: NpyHeader,
    rows_read: usize,
    scratch_bytes: Vec<u8>,
}

impl NpyMatrixReader<BufReader<File>> {
    /// Opens a `.npy` matrix file on disk and immediately validates its header.
    pub fn open(path: impl AsRef<Path>) -> OspreyResult<Self> {
        let file = File::open(path)?;
        Self::new(BufReader::new(file))
    }
}

impl<R: Read> NpyMatrixReader<R> {
    /// Parses and validates the `.npy` v1.0 / v2.0 header from `reader`.
    pub fn new(mut reader: R) -> OspreyResult<Self> {
        let mut prefix = [0_u8; 8];
        reader.read_exact(&mut prefix)?;

        if prefix[0..6] != NPY_MAGIC {
            return Err(invalid_data(format!(
                "invalid .npy magic bytes: expected \\x93NUMPY, found {:?}",
                &prefix[0..6]
            )));
        }

        let major_version = prefix[6];
        let minor_version = prefix[7];
        let header_len: usize = match major_version {
            1 => {
                let mut len_buf = [0_u8; 2];
                reader.read_exact(&mut len_buf)?;
                usize::from(u16::from_le_bytes(len_buf))
            }
            2 => {
                let mut len_buf = [0_u8; 4];
                reader.read_exact(&mut len_buf)?;
                usize::try_from(u32::from_le_bytes(len_buf))
                    .map_err(|_| invalid_data("header length overflows usize"))?
            }
            other => {
                return Err(OspreyError::UnsupportedVersion {
                    found: u16::from(other),
                    expected: 1,
                });
            }
        };

        if header_len == 0 || header_len > MAX_NPY_HEADER_BYTES {
            return Err(invalid_data(format!(
                "invalid .npy header length: {header_len}"
            )));
        }

        let mut hdr_bytes = vec![0_u8; header_len];
        reader.read_exact(&mut hdr_bytes)?;
        let hdr_str = std::str::from_utf8(&hdr_bytes)
            .map_err(|err| invalid_data(format!("non-UTF-8 .npy header: {err}")))?;

        let (dtype, num_rows, dim) = parse_npy_header_dict(hdr_str)?;
        let header = NpyHeader {
            major_version,
            minor_version,
            dtype,
            num_rows,
            dim,
        };

        Ok(Self {
            reader,
            header,
            rows_read: 0,
            scratch_bytes: Vec::new(),
        })
    }

    /// Returns the parsed `.npy` matrix header.
    #[inline]
    pub fn header(&self) -> &NpyHeader {
        &self.header
    }

    /// Total number of rows `N` in the matrix.
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.header.num_rows
    }

    /// Vector dimensionality `D` in the matrix.
    #[inline]
    pub fn dim(&self) -> usize {
        self.header.dim
    }

    /// Scalar element type (`F32` or `F16`).
    #[inline]
    pub fn dtype(&self) -> NpyDtype {
        self.header.dtype
    }

    /// Number of unread rows remaining.
    #[inline]
    pub fn rows_remaining(&self) -> usize {
        self.header.num_rows.saturating_sub(self.rows_read)
    }

    /// Streams up to `max_rows` vectors converted to `f32`, assigning `doc_id = row_idx as u64`.
    /// Returns `Ok(None)` when all `num_rows` have been read.
    pub fn read_batch_f32(&mut self, max_rows: usize) -> OspreyResult<Option<NpyVectorBatch>> {
        let remaining = self.rows_remaining();
        if remaining == 0 {
            return Ok(None);
        }
        let take = max_rows.max(1).min(remaining);
        let dim = self.header.dim;
        let row_bytes = dim
            .checked_mul(self.header.dtype.byte_size())
            .ok_or(OspreyError::InvalidDimension { dim })?;
        let total_bytes = take
            .checked_mul(row_bytes)
            .ok_or(OspreyError::InvalidDimension { dim: take })?;

        self.scratch_bytes.resize(total_bytes, 0);
        self.reader.read_exact(&mut self.scratch_bytes)?;

        let mut doc_ids = Vec::with_capacity(take);
        let mut vectors_f32 = Vec::with_capacity(take * dim);

        for r in 0..take {
            let doc_id =
                u64::try_from(self.rows_read + r).map_err(|_| OspreyError::InvalidDimension {
                    dim: self.rows_read + r,
                })?;
            doc_ids.push(doc_id);
        }

        match self.header.dtype {
            NpyDtype::F32 => {
                for (elem_idx, chunk) in self.scratch_bytes.as_chunks::<4>().0.iter().enumerate() {
                    let val = f32::from_le_bytes(*chunk);
                    if !val.is_finite() {
                        return Err(OspreyError::NonFiniteEmbedding {
                            index: elem_idx % dim,
                        });
                    }
                    vectors_f32.push(val);
                }
            }
            NpyDtype::F16 => {
                for (elem_idx, chunk) in self.scratch_bytes.as_chunks::<2>().0.iter().enumerate() {
                    let bits = u16::from_le_bytes(*chunk);
                    let val = f16::from_bits(bits).to_f32();
                    if !val.is_finite() {
                        return Err(OspreyError::NonFiniteEmbedding {
                            index: elem_idx % dim,
                        });
                    }
                    vectors_f32.push(val);
                }
            }
        }

        self.rows_read += take;
        Ok(Some(NpyVectorBatch {
            doc_ids,
            vectors_f32,
        }))
    }

    /// Streams up to `max_rows` vectors as `half::f16` (lossless for `<f2`), assigning
    /// `doc_id = row_idx as u64`.
    pub fn read_batch_f16(&mut self, max_rows: usize) -> OspreyResult<Option<NpyF16Batch>> {
        let remaining = self.rows_remaining();
        if remaining == 0 {
            return Ok(None);
        }
        let take = max_rows.max(1).min(remaining);
        let dim = self.header.dim;
        let row_bytes = dim
            .checked_mul(self.header.dtype.byte_size())
            .ok_or(OspreyError::InvalidDimension { dim })?;
        let total_bytes = take
            .checked_mul(row_bytes)
            .ok_or(OspreyError::InvalidDimension { dim: take })?;

        self.scratch_bytes.resize(total_bytes, 0);
        self.reader.read_exact(&mut self.scratch_bytes)?;

        let mut doc_ids = Vec::with_capacity(take);
        let mut vectors_f16 = Vec::with_capacity(take * dim);
        for r in 0..take {
            let doc_id =
                u64::try_from(self.rows_read + r).map_err(|_| OspreyError::InvalidDimension {
                    dim: self.rows_read + r,
                })?;
            doc_ids.push(doc_id);
        }

        match self.header.dtype {
            NpyDtype::F16 => {
                for (elem_idx, chunk) in self.scratch_bytes.as_chunks::<2>().0.iter().enumerate() {
                    let val = f16::from_le_bytes(*chunk);
                    if !val.is_finite() {
                        return Err(OspreyError::NonFiniteEmbedding {
                            index: elem_idx % dim,
                        });
                    }
                    vectors_f16.push(val);
                }
            }
            NpyDtype::F32 => {
                for (elem_idx, chunk) in self.scratch_bytes.as_chunks::<4>().0.iter().enumerate() {
                    let f = f32::from_le_bytes(*chunk);
                    if !f.is_finite() {
                        return Err(OspreyError::NonFiniteEmbedding {
                            index: elem_idx % dim,
                        });
                    }
                    vectors_f16.push(f16::from_f32(f));
                }
            }
        }

        self.rows_read += take;
        Ok(Some(NpyF16Batch {
            doc_ids,
            vectors_f16,
        }))
    }

    /// Reads the next single row as `(doc_id, Vec<f32>)`.
    pub fn next_row_f32(&mut self) -> OspreyResult<Option<(u64, Vec<f32>)>> {
        match self.read_batch_f32(1)? {
            Some(batch) => {
                let doc_id = batch.doc_ids[0];
                Ok(Some((doc_id, batch.vectors_f32)))
            }
            None => Ok(None),
        }
    }
}

fn parse_npy_header_dict(hdr: &str) -> OspreyResult<(NpyDtype, usize, usize)> {
    let trimmed = hdr.trim();
    let inner = trimmed
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .ok_or_else(|| invalid_data(format!("malformed .npy header dict: {trimmed}")))?;

    let descr = extract_dict_value(inner, "descr")
        .ok_or_else(|| invalid_data("missing 'descr' in .npy header"))?;
    let descr_clean = descr.trim().trim_matches('\'').trim_matches('"');
    let dtype = match descr_clean {
        "<f4" | "f4" => NpyDtype::F32,
        "<f2" | "f2" => NpyDtype::F16,
        other => {
            return Err(invalid_data(format!(
                "unsupported .npy descr '{other}': expected '<f4' or '<f2'"
            )));
        }
    };

    let fortran = extract_dict_value(inner, "fortran_order")
        .ok_or_else(|| invalid_data("missing 'fortran_order' in .npy header"))?;
    match fortran.trim() {
        "False" => {}
        "True" => {
            return Err(invalid_data(
                "Fortran-order .npy matrices (fortran_order=True) are not supported",
            ));
        }
        other => {
            return Err(invalid_data(format!(
                "invalid 'fortran_order' value '{other}' in .npy header"
            )));
        }
    }

    let shape_raw = extract_dict_value(inner, "shape")
        .ok_or_else(|| invalid_data("missing 'shape' in .npy header"))?;
    let shape_inner = shape_raw
        .trim()
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .ok_or_else(|| invalid_data(format!("malformed .npy shape '{shape_raw}'")))?;

    let parts: Vec<usize> = shape_inner
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|_| invalid_data(format!("invalid dimension '{s}' in .npy shape")))
        })
        .collect::<OspreyResult<Vec<_>>>()?;

    if parts.len() != 2 {
        return Err(OspreyError::InvalidDimension { dim: parts.len() });
    }
    let num_rows = parts[0];
    let dim = parts[1];
    if dim == 0 {
        return Err(OspreyError::InvalidDimension { dim: 0 });
    }

    Ok((dtype, num_rows, dim))
}

fn extract_dict_value<'a>(dict_body: &'a str, key: &str) -> Option<&'a str> {
    let single_pat = format!("'{key}'");
    let double_pat = format!("\"{key}\"");

    let key_pos = dict_body
        .find(&single_pat)
        .map(|p| p + single_pat.len())
        .or_else(|| dict_body.find(&double_pat).map(|p| p + double_pat.len()))?;

    let after_key = dict_body[key_pos..].trim_start();
    let after_colon = after_key.strip_prefix(':')?.trim_start();

    let mut paren_depth = 0_usize;
    let mut in_single = false;
    let mut in_double = false;
    for (idx, ch) in after_colon.char_indices() {
        match ch {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '(' if !in_single && !in_double => paren_depth += 1,
            ')' if !in_single && !in_double => paren_depth = paren_depth.saturating_sub(1),
            ',' if !in_single && !in_double && paren_depth == 0 => {
                return Some(after_colon[..idx].trim());
            }
            _ => {}
        }
    }
    Some(after_colon.trim())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn build_npy_bytes(
        major: u8,
        descr: &str,
        fortran_order: bool,
        shape: &str,
        payload: &[u8],
    ) -> Vec<u8> {
        let fortran_str = if fortran_order { "True" } else { "False" };
        let mut dict =
            format!("{{'descr': '{descr}', 'fortran_order': {fortran_str}, 'shape': {shape}, }}");
        let prefix_len = if major == 1 { 10 } else { 12 };
        // Align total header + 1 newline to 64 bytes per NumPy spec.
        let unaligned = prefix_len + dict.len() + 1;
        let pad = (64 - (unaligned % 64)) % 64;
        for _ in 0..pad {
            dict.push(' ');
        }
        dict.push('\n');

        let mut out = Vec::new();
        out.extend_from_slice(&NPY_MAGIC);
        out.push(major);
        out.push(0);
        if major == 1 {
            let len_u16 = u16::try_from(dict.len()).unwrap_or(u16::MAX);
            out.extend_from_slice(&len_u16.to_le_bytes());
        } else {
            let len_u32 = u32::try_from(dict.len()).unwrap_or(u32::MAX);
            out.extend_from_slice(&len_u32.to_le_bytes());
        }
        out.extend_from_slice(dict.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn test_jsonl_reader_integer_and_string_ids_with_colbert_and_metadata() -> OspreyResult<()> {
        let input = concat!(
            "{\"id\": 42, \"vector\": [0.5, -1.25], \"tokens\": [[1.0, 0.0], [0.0, 1.0]], \"tenant\": \"acme\", \"score\": 9}\n",
            "\n",
            "{\"doc_id\": \"doc-abc\", \"embedding\": [2.0, 3.0], \"category\": \"finance\"}\n"
        );
        let mut reader = JsonlRecordReader::new(Cursor::new(input.as_bytes()));

        let first_batch = reader.read_batch(10)?;
        assert_eq!(first_batch.len(), 2);

        let r0 = &first_batch[0];
        assert_eq!(r0.doc_id, 42);
        assert_eq!(r0.vector, vec![0.5, -1.25]);
        assert_eq!(
            r0.colbert_tokens,
            Some(vec![vec![1.0, 0.0], vec![0.0, 1.0]])
        );
        let meta0: serde_json::Value = serde_json::from_slice(&r0.metadata_payload)
            .map_err(|e| invalid_data(e.to_string()))?;
        assert_eq!(meta0["tenant"], "acme");
        assert_eq!(meta0["score"], 9);
        assert!(meta0.get("id").is_none());
        assert!(meta0.get("vector").is_none());
        assert!(meta0.get("tokens").is_none());

        let r1 = &first_batch[1];
        assert_eq!(r1.doc_id, map_string_id_to_u64("doc-abc"));
        assert_eq!(r1.vector, vec![2.0, 3.0]);
        assert_eq!(r1.colbert_tokens, None);
        let meta1: serde_json::Value = serde_json::from_slice(&r1.metadata_payload)
            .map_err(|e| invalid_data(e.to_string()))?;
        assert_eq!(meta1["category"], "finance");
        Ok(())
    }

    #[test]
    fn test_npy_v1_f32_and_v2_f16_streaming() -> OspreyResult<()> {
        // 1. NPY v1.0 '<f4' shape (3, 2)
        let f32_vals: [f32; 6] = [1.0, -2.5, 3.25, 4.0, -5.5, 6.75];
        let mut f32_bytes = Vec::new();
        for v in f32_vals {
            f32_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let npy_v1 = build_npy_bytes(1, "<f4", false, "(3, 2)", &f32_bytes);

        let mut r1 = NpyMatrixReader::new(Cursor::new(npy_v1))?;
        assert_eq!(r1.header().major_version, 1);
        assert_eq!(r1.dtype(), NpyDtype::F32);
        assert_eq!(r1.num_rows(), 3);
        assert_eq!(r1.dim(), 2);

        let b1 = r1.read_batch_f32(2)?;
        assert!(b1.is_some());
        if let Some(batch1) = b1 {
            assert_eq!(batch1.doc_ids, vec![0, 1]);
            assert_eq!(batch1.vectors_f32, vec![1.0, -2.5, 3.25, 4.0]);
        }
        let b2 = r1.read_batch_f32(2)?;
        assert!(b2.is_some());
        if let Some(batch2) = b2 {
            assert_eq!(batch2.doc_ids, vec![2]);
            assert_eq!(batch2.vectors_f32, vec![-5.5, 6.75]);
        }
        assert!(r1.read_batch_f32(2)?.is_none());

        // 2. NPY v2.0 '<f2' shape (2, 3)
        let f16_vals: [f16; 6] = [
            f16::from_f32(0.5),
            f16::from_f32(-1.0),
            f16::from_f32(2.0),
            f16::from_f32(4.5),
            f16::from_f32(-0.25),
            f16::from_f32(8.0),
        ];
        let mut f16_bytes = Vec::new();
        for v in f16_vals {
            f16_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let npy_v2 = build_npy_bytes(2, "<f2", false, "(2, 3)", &f16_bytes);

        let mut r2 = NpyMatrixReader::new(Cursor::new(npy_v2))?;
        assert_eq!(r2.header().major_version, 2);
        assert_eq!(r2.dtype(), NpyDtype::F16);
        assert_eq!(r2.num_rows(), 2);
        assert_eq!(r2.dim(), 3);

        let f16_batch = r2.read_batch_f16(10)?;
        assert!(f16_batch.is_some());
        if let Some(batch) = f16_batch {
            assert_eq!(batch.doc_ids, vec![0, 1]);
            assert_eq!(batch.vectors_f16, f16_vals.to_vec());
        }
        Ok(())
    }

    #[test]
    fn test_npy_rejects_fortran_order_and_invalid_dtype_or_shape() {
        let payload = [0_u8; 16];

        let fortran_npy = build_npy_bytes(1, "<f4", true, "(2, 2)", &payload);
        assert!(NpyMatrixReader::new(Cursor::new(fortran_npy)).is_err());

        let big_endian_npy = build_npy_bytes(1, ">f4", false, "(2, 2)", &payload);
        assert!(NpyMatrixReader::new(Cursor::new(big_endian_npy)).is_err());

        let bad_shape_npy = build_npy_bytes(1, "<f4", false, "(4,)", &payload);
        assert!(matches!(
            NpyMatrixReader::new(Cursor::new(bad_shape_npy)),
            Err(OspreyError::InvalidDimension { dim: 1 })
        ));

        let bad_magic = b"NOTNPY\x01\x00\x10\x00{}".to_vec();
        assert!(NpyMatrixReader::new(Cursor::new(bad_magic)).is_err());
    }
}
