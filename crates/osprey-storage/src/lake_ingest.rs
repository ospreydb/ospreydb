//! Native Data Lake ingestion for Apache Parquet & Arrow with schema auto-detection,
//! column projection pushdown, scalar inverted bitmaps, and minipage Zone Maps.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::{
    Array, ArrayRef, BooleanArray, FixedSizeListArray, Float16Array, Float32Array, Float64Array,
    Int32Array, Int64Array, LargeListArray, LargeStringArray, ListArray, RecordBatchReader,
    StringArray, TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
    TimestampSecondArray, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, Schema, SchemaRef, TimeUnit};
use arrow::record_batch::RecordBatch;
use futures::Stream;
use half::f16;
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;
use osprey_core::math::{padded_dimension, prepare_and_pad_vector, RademacherSigns, SplitMix64};
use osprey_core::{
    fit_binary_kmeans, BinaryKMeansConfig, BinaryKMeansKernel, OspreyError, OspreyResult,
    ScalarClusterKernel,
};
use parquet::arrow::arrow_reader::{
    ArrowReaderOptions, ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder,
};
use parquet::arrow::async_reader::{
    AsyncFileReader, ParquetObjectReader, ParquetRecordBatchStream, ParquetRecordBatchStreamBuilder,
};
use parquet::arrow::ProjectionMask;
use parquet::file::reader::ChunkReader;
use roaring::RoaringBitmap;
use xxhash_rust::xxh64::xxh64;

use crate::osv_slab::{OsvVectorRecord, OsvWriteReport, OsvWriter, OsvWriterConfig};
use crate::tsp_locality::{solve_tsp_locality_largest_first, ClusterLocalityPermutation};

/// Default record batch row count for out-of-core Parquet streaming.
pub const DEFAULT_INGEST_BATCH_SIZE: usize = 4096;

const PREFERRED_ID_COLUMNS: &[&str] = &["id", "doc_id", "document_id", "_id", "uid", "pk"];
const PREFERRED_VECTOR_COLUMNS: &[&str] = &[
    "vector",
    "embedding",
    "embeddings",
    "vec",
    "dense_vector",
    "values",
];
const PREFERRED_COLBERT_COLUMNS: &[&str] = &[
    "colbert",
    "colbert_tokens",
    "tokens",
    "token_embeddings",
    "multi_vector",
];
const AUTO_METADATA_COLUMNS: &[&str] = &[
    "tenant_id",
    "tenant",
    "timestamp_epoch_secs",
    "timestamp",
    "created_at",
    "category_hash",
    "category",
];

#[inline]
fn invalid_data(msg: impl Into<String>) -> OspreyError {
    OspreyError::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        msg.into(),
    ))
}

#[inline]
fn invalid_input(msg: impl Into<String>) -> OspreyError {
    OspreyError::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        msg.into(),
    ))
}

/// Deterministically maps a string document identifier to a `u64` via `xxHash64` (seed `0`).
#[inline]
pub fn map_string_id_to_u64(id: &str) -> u64 {
    xxh64(id.as_bytes(), 0)
}

/// Deterministically hashes a category label into a `u32` via `xxHash64` (seed `0`).
#[inline]
pub fn hash_category(category: &str) -> u32 {
    let bytes = xxh64(category.as_bytes(), 0).to_le_bytes();
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

/// Optional column overrides and dimension expectations for Data Lake schema inspection.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IngestColumnHints {
    /// Explicit document ID column name; auto-detected when `None`.
    pub id_column: Option<String>,
    /// Explicit dense embedding column name; auto-detected when `None`.
    pub vector_column: Option<String>,
    /// Explicit ColBERT multi-vector token column name; auto-detected when `None`.
    pub colbert_column: Option<String>,
    /// Filterable scalar metadata / payload column names to include in projection pushdown.
    pub metadata_columns: Vec<String>,
    /// Optional expected dense vector dimensionality (used to validate `FixedSizeList` or seed `List`).
    pub expected_vector_dim: Option<usize>,
    /// Optional expected ColBERT token dimensionality (used when inner tokens are `List<Float*>`).
    pub expected_colbert_dim: Option<usize>,
}

/// Resolved Data Lake column layout detected from a Parquet / Arrow schema.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DetectedLakeSchema {
    /// Column index of the document ID in the source Arrow schema.
    pub id_col_idx: usize,
    /// Name of the document ID column.
    pub id_col_name: String,
    /// Whether the document ID column is UTF-8 (`Utf8`/`LargeUtf8`) requiring `xxh64` mapping.
    pub id_is_string: bool,
    /// Column index of the dense embedding vector in the source Arrow schema.
    pub vector_col_idx: usize,
    /// Name of the dense embedding column.
    pub vector_col_name: String,
    /// Original unpadded dimensionality (`0` if variable `List` without `expected_vector_dim` hint).
    pub orig_dim: usize,
    /// Whether dense vector elements are IEEE-754 `Float16` (`false` means `Float32`).
    pub is_f16: bool,
    /// Optional column index of ColBERT multi-vector tokens in the source Arrow schema.
    pub colbert_col_idx: Option<usize>,
    /// Optional token vector dimensionality for ColBERT tokens.
    pub colbert_token_dim: Option<usize>,
    /// Source Arrow schema indices of projected scalar metadata columns.
    pub metadata_col_indices: Vec<usize>,
}

impl DetectedLakeSchema {
    /// Returns the sorted, deduplicated root column indices required for `ProjectionMask::roots`.
    pub fn projected_indices(&self) -> Vec<usize> {
        let mut indices = Vec::with_capacity(3 + self.metadata_col_indices.len());
        indices.push(self.id_col_idx);
        indices.push(self.vector_col_idx);
        if let Some(colbert_idx) = self.colbert_col_idx {
            indices.push(colbert_idx);
        }
        indices.extend_from_slice(&self.metadata_col_indices);
        indices.sort_unstable();
        indices.dedup();
        indices
    }

    /// Computes the column-index remapping inside a projected `RecordBatch`.
    pub fn projected_column_map(&self) -> ProjectedColumnMap {
        let projected = self.projected_indices();
        let batch_pos =
            |orig_idx: usize| -> usize { projected.binary_search(&orig_idx).unwrap_or(0) };
        ProjectedColumnMap {
            id_batch_idx: batch_pos(self.id_col_idx),
            vector_batch_idx: batch_pos(self.vector_col_idx),
            colbert_batch_idx: self.colbert_col_idx.map(batch_pos),
            metadata_batch_indices: self
                .metadata_col_indices
                .iter()
                .map(|&orig_idx| (orig_idx, batch_pos(orig_idx)))
                .collect(),
        }
    }
}

/// Mapping from original Parquet/Arrow root column indices to projected `RecordBatch` indices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectedColumnMap {
    /// Column index of the ID column inside projected `RecordBatch`es.
    pub id_batch_idx: usize,
    /// Column index of the dense vector column inside projected `RecordBatch`es.
    pub vector_batch_idx: usize,
    /// Optional column index of the ColBERT token column inside projected `RecordBatch`es.
    pub colbert_batch_idx: Option<usize>,
    /// Pairs of `(original_schema_idx, projected_batch_idx)` for metadata columns.
    pub metadata_batch_indices: Vec<(usize, usize)>,
}

impl ProjectedColumnMap {
    /// Translates an original schema column index to its projected `RecordBatch` column index.
    pub fn batch_idx_for_orig(&self, orig_idx: usize) -> Option<usize> {
        self.metadata_batch_indices
            .iter()
            .find_map(|&(orig, batch)| (orig == orig_idx).then_some(batch))
    }
}

#[inline]
fn is_supported_id_type(dt: &DataType) -> Option<bool> {
    match dt {
        DataType::UInt64 | DataType::Int64 => Some(false),
        DataType::Utf8 | DataType::LargeUtf8 => Some(true),
        _ => None,
    }
}

#[inline]
fn float_elem_is_f16(dt: &DataType) -> Option<bool> {
    match dt {
        DataType::Float32 => Some(false),
        DataType::Float16 => Some(true),
        _ => None,
    }
}

fn inspect_dense_vector_type(
    dt: &DataType,
    expected_dim: Option<usize>,
) -> OspreyResult<Option<(usize, bool)>> {
    match dt {
        DataType::FixedSizeList(field, size) => {
            let Some(is_f16) = float_elem_is_f16(field.data_type()) else {
                return Ok(None);
            };
            if *size <= 0 {
                return Err(OspreyError::InvalidDimension { dim: 0 });
            }
            let dim =
                usize::try_from(*size).map_err(|_| OspreyError::InvalidDimension { dim: 0 })?;
            if let Some(exp) = expected_dim {
                if exp != dim {
                    return Err(OspreyError::InvalidDimension { dim });
                }
            }
            Ok(Some((dim, is_f16)))
        }
        DataType::List(field) | DataType::LargeList(field) => {
            let Some(is_f16) = float_elem_is_f16(field.data_type()) else {
                return Ok(None);
            };
            Ok(Some((expected_dim.unwrap_or(0), is_f16)))
        }
        _ => Ok(None),
    }
}

fn inspect_colbert_type(
    dt: &DataType,
    expected_dim: Option<usize>,
) -> OspreyResult<Option<Option<usize>>> {
    let inner_dt = match dt {
        DataType::List(field) | DataType::LargeList(field) => field.data_type(),
        _ => return Ok(None),
    };

    match inner_dt {
        DataType::FixedSizeList(elem_field, size) => {
            if float_elem_is_f16(elem_field.data_type()).is_none() {
                return Ok(None);
            }
            if *size <= 0 {
                return Err(OspreyError::InvalidDimension { dim: 0 });
            }
            let dim =
                usize::try_from(*size).map_err(|_| OspreyError::InvalidDimension { dim: 0 })?;
            if let Some(exp) = expected_dim {
                if exp != dim {
                    return Err(OspreyError::InvalidDimension { dim });
                }
            }
            Ok(Some(Some(dim)))
        }
        DataType::List(elem_field) | DataType::LargeList(elem_field) => {
            if float_elem_is_f16(elem_field.data_type()).is_none() {
                return Ok(None);
            }
            Ok(Some(expected_dim))
        }
        _ => Ok(None),
    }
}

#[inline]
fn is_supported_auto_metadata_type(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::UInt32
            | DataType::Int32
            | DataType::UInt64
            | DataType::Int64
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Timestamp(_, _)
    )
}

fn find_field_by_name<'a>(
    schema: &'a Schema,
    name: &str,
) -> OspreyResult<(usize, &'a arrow::datatypes::FieldRef)> {
    schema
        .fields()
        .find(name)
        .ok_or_else(|| invalid_input(format!("column '{name}' not found in Parquet/Arrow schema")))
}

fn resolve_id_column(
    schema: &Schema,
    hints: &IngestColumnHints,
) -> OspreyResult<(usize, String, bool)> {
    if let Some(ref name) = hints.id_column {
        let (idx, field) = find_field_by_name(schema, name)?;
        let id_is_string = is_supported_id_type(field.data_type()).ok_or_else(|| {
            invalid_data(format!(
                "column '{name}' has unsupported ID type {:?}",
                field.data_type()
            ))
        })?;
        return Ok((idx, field.name().clone(), id_is_string));
    }

    for &preferred in PREFERRED_ID_COLUMNS {
        if let Some((idx, field)) = schema.fields().find(preferred) {
            if let Some(id_is_string) = is_supported_id_type(field.data_type()) {
                return Ok((idx, field.name().clone(), id_is_string));
            }
        }
    }

    for (idx, field) in schema.fields().iter().enumerate() {
        let lower = field.name().to_ascii_lowercase();
        let looks_like_non_id = AUTO_METADATA_COLUMNS.contains(&lower.as_str())
            || matches!(
                lower.as_str(),
                "raw_html" | "html" | "body" | "logs" | "log" | "payload" | "blob"
            );
        if looks_like_non_id {
            continue;
        }
        if let Some(id_is_string) = is_supported_id_type(field.data_type()) {
            return Ok((idx, field.name().clone(), id_is_string));
        }
    }

    for (idx, field) in schema.fields().iter().enumerate() {
        if let Some(id_is_string) = is_supported_id_type(field.data_type()) {
            return Ok((idx, field.name().clone(), id_is_string));
        }
    }

    Err(invalid_data(
        "no compatible ID column (UInt64, Int64, Utf8, LargeUtf8) found in schema",
    ))
}

fn resolve_vector_column(
    schema: &Schema,
    hints: &IngestColumnHints,
    id_col_idx: usize,
) -> OspreyResult<(usize, String, usize, bool)> {
    if let Some(ref name) = hints.vector_column {
        let (idx, field) = find_field_by_name(schema, name)?;
        if idx == id_col_idx {
            return Err(invalid_input(format!(
                "vector column '{name}' collides with ID column"
            )));
        }
        let (orig_dim, is_f16) =
            inspect_dense_vector_type(field.data_type(), hints.expected_vector_dim)?.ok_or_else(
                || {
                    invalid_data(format!(
                        "column '{name}' has unsupported dense vector type {:?}",
                        field.data_type()
                    ))
                },
            )?;
        return Ok((idx, field.name().clone(), orig_dim, is_f16));
    }

    for &preferred in PREFERRED_VECTOR_COLUMNS {
        if let Some((idx, field)) = schema.fields().find(preferred) {
            if idx != id_col_idx {
                if let Some((orig_dim, is_f16)) =
                    inspect_dense_vector_type(field.data_type(), hints.expected_vector_dim)?
                {
                    return Ok((idx, field.name().clone(), orig_dim, is_f16));
                }
            }
        }
    }

    for (idx, field) in schema.fields().iter().enumerate() {
        if idx == id_col_idx {
            continue;
        }
        if let Some((orig_dim, is_f16)) =
            inspect_dense_vector_type(field.data_type(), hints.expected_vector_dim)?
        {
            return Ok((idx, field.name().clone(), orig_dim, is_f16));
        }
    }

    Err(invalid_data(
        "no compatible dense vector column (FixedSizeList<Float32/Float16> or List<Float32/Float16>) found in schema",
    ))
}

fn resolve_colbert_column(
    schema: &Schema,
    hints: &IngestColumnHints,
    id_col_idx: usize,
    vector_col_idx: usize,
) -> OspreyResult<(Option<usize>, Option<usize>)> {
    if let Some(ref name) = hints.colbert_column {
        let (idx, field) = find_field_by_name(schema, name)?;
        if idx == id_col_idx || idx == vector_col_idx {
            return Err(invalid_input(format!(
                "ColBERT column '{name}' collides with ID or vector column"
            )));
        }
        let token_dim = inspect_colbert_type(field.data_type(), hints.expected_colbert_dim)?
            .ok_or_else(|| {
                invalid_data(format!(
                    "column '{name}' has unsupported ColBERT type {:?}",
                    field.data_type()
                ))
            })?;
        return Ok((Some(idx), token_dim));
    }

    for &preferred in PREFERRED_COLBERT_COLUMNS {
        if let Some((idx, field)) = schema.fields().find(preferred) {
            if idx != id_col_idx && idx != vector_col_idx {
                if let Some(token_dim) =
                    inspect_colbert_type(field.data_type(), hints.expected_colbert_dim)?
                {
                    return Ok((Some(idx), token_dim));
                }
            }
        }
    }

    for (idx, field) in schema.fields().iter().enumerate() {
        if idx == id_col_idx || idx == vector_col_idx {
            continue;
        }
        if let Some(token_dim) =
            inspect_colbert_type(field.data_type(), hints.expected_colbert_dim)?
        {
            return Ok((Some(idx), token_dim));
        }
    }

    Ok((None, None))
}

fn resolve_metadata_columns(
    schema: &Schema,
    hints: &IngestColumnHints,
    id_col_idx: usize,
    vector_col_idx: usize,
    colbert_col_idx: Option<usize>,
) -> OspreyResult<Vec<usize>> {
    let mut indices = Vec::new();
    let mut seen = HashSet::new();

    if !hints.metadata_columns.is_empty() {
        for col_name in &hints.metadata_columns {
            let (idx, _) = find_field_by_name(schema, col_name)?;
            if idx == vector_col_idx || Some(idx) == colbert_col_idx {
                return Err(invalid_input(format!(
                    "metadata column '{col_name}' collides with vector/ColBERT column"
                )));
            }
            if idx != id_col_idx && seen.insert(idx) {
                indices.push(idx);
            }
        }
        return Ok(indices);
    }

    for (idx, field) in schema.fields().iter().enumerate() {
        if idx == id_col_idx || idx == vector_col_idx || Some(idx) == colbert_col_idx {
            continue;
        }
        if AUTO_METADATA_COLUMNS.contains(&field.name().as_str())
            && is_supported_auto_metadata_type(field.data_type())
            && seen.insert(idx)
        {
            indices.push(idx);
        }
    }

    Ok(indices)
}

/// Inspects an Arrow schema derived from a Parquet footer and detects ID, dense vector,
/// optional ColBERT multi-vector, and filterable scalar metadata columns.
pub fn inspect_parquet_schema(
    arrow_schema: &Schema,
    hints: &IngestColumnHints,
) -> OspreyResult<DetectedLakeSchema> {
    let (id_col_idx, id_col_name, id_is_string) = resolve_id_column(arrow_schema, hints)?;
    let (vector_col_idx, vector_col_name, orig_dim, is_f16) =
        resolve_vector_column(arrow_schema, hints, id_col_idx)?;
    let (colbert_col_idx, colbert_token_dim) =
        resolve_colbert_column(arrow_schema, hints, id_col_idx, vector_col_idx)?;
    let metadata_col_indices = resolve_metadata_columns(
        arrow_schema,
        hints,
        id_col_idx,
        vector_col_idx,
        colbert_col_idx,
    )?;

    Ok(DetectedLakeSchema {
        id_col_idx,
        id_col_name,
        id_is_string,
        vector_col_idx,
        vector_col_name,
        orig_dim,
        is_f16,
        colbert_col_idx,
        colbert_token_dim,
        metadata_col_indices,
    })
}

/// Synchronous Parquet record-batch reader configured with schema inspection and
/// `ProjectionMask::roots` column pushdown.
pub struct ParquetBatchReader {
    detected: DetectedLakeSchema,
    projected_map: ProjectedColumnMap,
    projection_mask: ProjectionMask,
    projected_schema: SchemaRef,
    num_row_groups: usize,
    inner: ParquetRecordBatchReader,
}

impl ParquetBatchReader {
    /// Opens a local Parquet file, inspects only its footer schema, and configures
    /// `ProjectionMask::roots` pushdown with `batch_size`.
    pub fn open(
        path: impl AsRef<Path>,
        hints: &IngestColumnHints,
        batch_size: usize,
    ) -> OspreyResult<Self> {
        let file = File::open(path)?;
        Self::from_chunk_reader(file, hints, batch_size)
    }

    /// Creates a projected `ParquetBatchReader` from any [`ChunkReader`] (e.g. `File` or `Bytes`).
    pub fn from_chunk_reader<R: ChunkReader + 'static>(
        reader: R,
        hints: &IngestColumnHints,
        batch_size: usize,
    ) -> OspreyResult<Self> {
        let effective_batch_size = batch_size.max(1);
        let builder = ParquetRecordBatchReaderBuilder::try_new_with_options(
            reader,
            ArrowReaderOptions::new(),
        )
        .map_err(|err| invalid_data(format!("failed to read Parquet metadata: {err}")))?;

        let num_row_groups = builder.metadata().num_row_groups();
        let detected = inspect_parquet_schema(builder.schema().as_ref(), hints)?;
        let projected_indices = detected.projected_indices();
        let projected_map = detected.projected_column_map();
        let projection_mask = ProjectionMask::roots(
            builder.metadata().file_metadata().schema_descr(),
            projected_indices.iter().copied(),
        );

        let inner = builder
            .with_projection(projection_mask.clone())
            .with_batch_size(effective_batch_size)
            .build()
            .map_err(|err| invalid_data(format!("failed to build Parquet batch reader: {err}")))?;
        let projected_schema = inner.schema();

        Ok(Self {
            detected,
            projected_map,
            projection_mask,
            projected_schema,
            num_row_groups,
            inner,
        })
    }

    /// Returns the detected source Data Lake schema.
    #[inline]
    pub fn detected_schema(&self) -> &DetectedLakeSchema {
        &self.detected
    }

    /// Returns the remapped column indices within each projected `RecordBatch`.
    #[inline]
    pub fn projected_map(&self) -> &ProjectedColumnMap {
        &self.projected_map
    }

    /// Returns the effective `ProjectionMask` applied to the Parquet reader.
    #[inline]
    pub fn projection_mask(&self) -> &ProjectionMask {
        &self.projection_mask
    }

    /// Returns the projected Arrow schema emitted by batches from this reader.
    #[inline]
    pub fn projected_schema(&self) -> &SchemaRef {
        &self.projected_schema
    }

    /// Returns the number of Parquet row groups in the underlying file.
    #[inline]
    pub fn num_row_groups(&self) -> usize {
        self.num_row_groups
    }
}

impl Iterator for ParquetBatchReader {
    type Item = OspreyResult<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|res| res.map_err(|err| invalid_data(format!("Parquet batch read failed: {err}"))))
    }
}

/// Asynchronous Parquet record-batch stream (S3 / GCS / Azure / local `ObjectStore`)
/// configured with `ProjectionMask::roots` column pushdown.
pub struct ParquetBatchStream<T = ParquetObjectReader> {
    detected: DetectedLakeSchema,
    projected_map: ProjectedColumnMap,
    projection_mask: ProjectionMask,
    projected_schema: SchemaRef,
    num_row_groups: usize,
    inner: ParquetRecordBatchStream<T>,
}

impl ParquetBatchStream<ParquetObjectReader> {
    /// Opens an object in `ObjectStore` (S3/GCS/Azure/MinIO/local), inspects its footer
    /// metadata via range requests, and configures `ProjectionMask::roots` pushdown.
    pub async fn from_object_store(
        store: Arc<dyn ObjectStore>,
        location: &ObjectPath,
        hints: &IngestColumnHints,
        batch_size: usize,
    ) -> OspreyResult<Self> {
        let meta = store.head(location).await.map_err(|err| {
            OspreyError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("object_store head failed for '{location}': {err}"),
            ))
        })?;
        let reader = ParquetObjectReader::new(store, meta);
        Self::from_async_reader(reader, hints, batch_size).await
    }
}

impl<T: AsyncFileReader + Unpin + Send + 'static> ParquetBatchStream<T> {
    /// Creates a projected `ParquetBatchStream` from any [`AsyncFileReader`].
    pub async fn from_async_reader(
        reader: T,
        hints: &IngestColumnHints,
        batch_size: usize,
    ) -> OspreyResult<Self> {
        let effective_batch_size = batch_size.max(1);
        let builder =
            ParquetRecordBatchStreamBuilder::new_with_options(reader, ArrowReaderOptions::new())
                .await
                .map_err(|err| {
                    invalid_data(format!("failed to read async Parquet metadata: {err}"))
                })?;

        let num_row_groups = builder.metadata().num_row_groups();
        let detected = inspect_parquet_schema(builder.schema().as_ref(), hints)?;
        let projected_indices = detected.projected_indices();
        let projected_map = detected.projected_column_map();
        let projection_mask = ProjectionMask::roots(
            builder.metadata().file_metadata().schema_descr(),
            projected_indices,
        );

        let inner = builder
            .with_projection(projection_mask.clone())
            .with_batch_size(effective_batch_size)
            .build()
            .map_err(|err| invalid_data(format!("failed to build Parquet batch stream: {err}")))?;
        let projected_schema = inner.schema().clone();

        Ok(Self {
            detected,
            projected_map,
            projection_mask,
            projected_schema,
            num_row_groups,
            inner,
        })
    }

    /// Returns the detected source Data Lake schema.
    #[inline]
    pub fn detected_schema(&self) -> &DetectedLakeSchema {
        &self.detected
    }

    /// Returns the remapped column indices within each projected `RecordBatch`.
    #[inline]
    pub fn projected_map(&self) -> &ProjectedColumnMap {
        &self.projected_map
    }

    /// Returns the effective `ProjectionMask` applied to the Parquet stream.
    #[inline]
    pub fn projection_mask(&self) -> &ProjectionMask {
        &self.projection_mask
    }

    /// Returns the projected Arrow schema emitted by batches from this stream.
    #[inline]
    pub fn projected_schema(&self) -> &SchemaRef {
        &self.projected_schema
    }

    /// Returns the number of Parquet row groups in the underlying object.
    #[inline]
    pub fn num_row_groups(&self) -> usize {
        self.num_row_groups
    }
}

impl<T: AsyncFileReader + Unpin + Send + 'static> Stream for ParquetBatchStream<T> {
    type Item = OspreyResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner)
            .poll_next(cx)
            .map_err(|err| invalid_data(format!("async Parquet batch read failed: {err}")))
    }
}

/// Extracts `u64` document IDs from an Arrow `RecordBatch` column (`UInt64`, `Int64`,
/// `Utf8`, or `LargeUtf8`). String IDs are hashed deterministically with `xxh64(..., 0)`.
pub fn extract_batch_doc_ids(batch: &RecordBatch, col_idx: usize) -> OspreyResult<Vec<u64>> {
    let col = batch
        .columns()
        .get(col_idx)
        .ok_or_else(|| invalid_input(format!("ID column index {col_idx} out of bounds")))?;

    let mut out = Vec::with_capacity(batch.num_rows());
    match col.data_type() {
        DataType::UInt64 => {
            let arr = col
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| invalid_data("expected UInt64Array for ID column"))?;
            for row in 0..arr.len() {
                if arr.is_null(row) {
                    return Err(invalid_data(format!("null document ID at row {row}")));
                }
                out.push(arr.value(row));
            }
        }
        DataType::Int64 => {
            let arr = col
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| invalid_data("expected Int64Array for ID column"))?;
            for row in 0..arr.len() {
                if arr.is_null(row) {
                    return Err(invalid_data(format!("null document ID at row {row}")));
                }
                let val = arr.value(row);
                let uval = u64::try_from(val)
                    .map_err(|_| invalid_data(format!("negative Int64 ID {val} at row {row}")))?;
                out.push(uval);
            }
        }
        DataType::Utf8 => {
            let arr = col
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| invalid_data("expected StringArray for ID column"))?;
            for row in 0..arr.len() {
                if arr.is_null(row) {
                    return Err(invalid_data(format!("null string ID at row {row}")));
                }
                out.push(map_string_id_to_u64(arr.value(row)));
            }
        }
        DataType::LargeUtf8 => {
            let arr = col
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(|| invalid_data("expected LargeStringArray for ID column"))?;
            for row in 0..arr.len() {
                if arr.is_null(row) {
                    return Err(invalid_data(format!("null string ID at row {row}")));
                }
                out.push(map_string_id_to_u64(arr.value(row)));
            }
        }
        other => {
            return Err(invalid_data(format!(
                "unsupported ID column data type {other:?}"
            )));
        }
    }
    Ok(out)
}

fn append_float_slice_f32(
    values: &ArrayRef,
    start: usize,
    len: usize,
    out: &mut Vec<f32>,
) -> OspreyResult<()> {
    match values.data_type() {
        DataType::Float32 => {
            let arr = values
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| invalid_data("expected Float32Array inside vector list"))?;
            for idx in start..start + len {
                if arr.is_null(idx) {
                    return Err(invalid_data(format!(
                        "null vector coordinate at element index {idx}"
                    )));
                }
                let v = arr.value(idx);
                if !v.is_finite() {
                    return Err(OspreyError::NonFiniteEmbedding { index: idx - start });
                }
                out.push(v);
            }
        }
        DataType::Float16 => {
            let arr = values
                .as_any()
                .downcast_ref::<Float16Array>()
                .ok_or_else(|| invalid_data("expected Float16Array inside vector list"))?;
            for idx in start..start + len {
                if arr.is_null(idx) {
                    return Err(invalid_data(format!(
                        "null vector coordinate at element index {idx}"
                    )));
                }
                let v = arr.value(idx).to_f32();
                if !v.is_finite() {
                    return Err(OspreyError::NonFiniteEmbedding { index: idx - start });
                }
                out.push(v);
            }
        }
        other => {
            return Err(invalid_data(format!(
                "unsupported vector element type {other:?}"
            )));
        }
    }
    Ok(())
}

/// Extracts dense embedding vectors into a flat row-major `Vec<f32>`, validating uniform
/// dimensionality and finiteness (`OspreyError::NonFiniteEmbedding`).
pub fn extract_batch_vectors_f32(
    batch: &RecordBatch,
    col_idx: usize,
    expected_dim: Option<usize>,
) -> OspreyResult<(usize, Vec<f32>)> {
    let col = batch
        .columns()
        .get(col_idx)
        .ok_or_else(|| invalid_input(format!("vector column index {col_idx} out of bounds")))?;

    if batch.num_rows() == 0 {
        return Ok((expected_dim.unwrap_or(0), Vec::new()));
    }

    match col.data_type() {
        DataType::FixedSizeList(_, size) => {
            let dim =
                usize::try_from(*size).map_err(|_| OspreyError::InvalidDimension { dim: 0 })?;
            if dim == 0 {
                return Err(OspreyError::InvalidDimension { dim: 0 });
            }
            if let Some(exp) = expected_dim {
                if exp > 0 && exp != dim {
                    return Err(OspreyError::InvalidDimension { dim });
                }
            }
            let list = col
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| invalid_data("expected FixedSizeListArray for vector column"))?;
            let mut out = Vec::with_capacity(batch.num_rows() * dim);
            let values = list.values();
            for row in 0..list.len() {
                if list.is_null(row) {
                    return Err(invalid_data(format!("null vector at row {row}")));
                }
                let start = row * dim;
                append_float_slice_f32(values, start, dim, &mut out)?;
            }
            Ok((dim, out))
        }
        DataType::List(_) => {
            let list = col
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| invalid_data("expected ListArray for vector column"))?;
            extract_var_list_vectors(
                list.len(),
                |r| list.is_null(r),
                list.value_offsets(),
                list.values(),
                expected_dim,
            )
        }
        DataType::LargeList(_) => {
            let list = col
                .as_any()
                .downcast_ref::<LargeListArray>()
                .ok_or_else(|| invalid_data("expected LargeListArray for vector column"))?;
            extract_var_list_vectors(
                list.len(),
                |r| list.is_null(r),
                list.value_offsets(),
                list.values(),
                expected_dim,
            )
        }
        other => Err(invalid_data(format!(
            "unsupported dense vector column type {other:?}"
        ))),
    }
}

fn extract_var_list_vectors<O, F>(
    len: usize,
    is_null: F,
    offsets: &[O],
    values: &ArrayRef,
    expected_dim: Option<usize>,
) -> OspreyResult<(usize, Vec<f32>)>
where
    O: Copy,
    usize: TryFrom<O>,
    F: Fn(usize) -> bool,
{
    let mut dim = expected_dim.filter(|&d| d > 0);
    let mut out = Vec::new();
    for row in 0..len {
        if is_null(row) {
            return Err(invalid_data(format!("null vector at row {row}")));
        }
        let start = usize::try_from(offsets[row])
            .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
        let end = usize::try_from(offsets[row + 1])
            .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
        let row_dim = end.saturating_sub(start);
        if row_dim == 0 {
            return Err(OspreyError::InvalidDimension { dim: 0 });
        }
        match dim {
            Some(expected) if expected != row_dim => {
                return Err(OspreyError::InvalidDimension { dim: row_dim });
            }
            None => {
                dim = Some(row_dim);
                out.reserve(len * row_dim);
            }
            Some(_) => {}
        }
        append_float_slice_f32(values, start, row_dim, &mut out)?;
    }
    Ok((dim.unwrap_or(0), out))
}

/// Extracts ColBERT multi-vector token matrices per document (`Vec<Vec<f32>>`, where each
/// document contains `num_tokens * token_dim` flattened finite `f32` values).
pub fn extract_batch_colbert_tokens_f32(
    batch: &RecordBatch,
    col_idx: usize,
    expected_token_dim: Option<usize>,
) -> OspreyResult<(usize, Vec<Vec<f32>>)> {
    let col = batch
        .columns()
        .get(col_idx)
        .ok_or_else(|| invalid_input(format!("ColBERT column index {col_idx} out of bounds")))?;

    if batch.num_rows() == 0 {
        return Ok((expected_token_dim.unwrap_or(0), Vec::new()));
    }

    match col.data_type() {
        DataType::List(_) => {
            let list = col
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| invalid_data("expected ListArray for ColBERT column"))?;
            extract_colbert_outer(
                list.len(),
                |r| list.is_null(r),
                list.value_offsets(),
                list.values(),
                expected_token_dim,
            )
        }
        DataType::LargeList(_) => {
            let list = col
                .as_any()
                .downcast_ref::<LargeListArray>()
                .ok_or_else(|| invalid_data("expected LargeListArray for ColBERT column"))?;
            extract_colbert_outer(
                list.len(),
                |r| list.is_null(r),
                list.value_offsets(),
                list.values(),
                expected_token_dim,
            )
        }
        other => Err(invalid_data(format!(
            "unsupported ColBERT column type {other:?}"
        ))),
    }
}

fn extract_colbert_outer<O, F>(
    len: usize,
    is_null: F,
    offsets: &[O],
    inner_array: &ArrayRef,
    expected_token_dim: Option<usize>,
) -> OspreyResult<(usize, Vec<Vec<f32>>)>
where
    O: Copy + TryInto<usize>,
    F: Fn(usize) -> bool,
{
    let mut token_dim = expected_token_dim.filter(|&d| d > 0);
    let mut docs = Vec::with_capacity(len);

    for doc_row in 0..len {
        if is_null(doc_row) {
            docs.push(Vec::new());
            continue;
        }
        let tok_start: usize = offsets[doc_row]
            .try_into()
            .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
        let tok_end: usize = offsets[doc_row + 1]
            .try_into()
            .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
        let num_tokens = tok_end.saturating_sub(tok_start);
        let mut doc_tokens = Vec::new();

        match inner_array.data_type() {
            DataType::FixedSizeList(_, size) => {
                let d =
                    i32::try_into(*size).map_err(|_| OspreyError::InvalidDimension { dim: 0 })?;
                if d == 0 {
                    return Err(OspreyError::InvalidDimension { dim: 0 });
                }
                match token_dim {
                    Some(exp) if exp != d => return Err(OspreyError::InvalidDimension { dim: d }),
                    None => token_dim = Some(d),
                    Some(_) => {}
                }
                let inner = inner_array
                    .as_any()
                    .downcast_ref::<FixedSizeListArray>()
                    .ok_or_else(|| {
                        invalid_data("expected FixedSizeListArray for ColBERT tokens")
                    })?;
                doc_tokens.reserve(num_tokens * d);
                for tok_idx in tok_start..tok_end {
                    if inner.is_null(tok_idx) {
                        return Err(invalid_data(format!("null ColBERT token at {tok_idx}")));
                    }
                    append_float_slice_f32(inner.values(), tok_idx * d, d, &mut doc_tokens)?;
                }
            }
            DataType::List(_) => {
                let inner = inner_array
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .ok_or_else(|| invalid_data("expected ListArray for ColBERT tokens"))?;
                let inner_offsets = inner.value_offsets();
                for tok_idx in tok_start..tok_end {
                    if inner.is_null(tok_idx) {
                        return Err(invalid_data(format!("null ColBERT token at {tok_idx}")));
                    }
                    let s: usize = i32::try_into(inner_offsets[tok_idx])
                        .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
                    let e: usize = i32::try_into(inner_offsets[tok_idx + 1])
                        .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
                    let d = e.saturating_sub(s);
                    if d == 0 {
                        return Err(OspreyError::InvalidDimension { dim: 0 });
                    }
                    match token_dim {
                        Some(exp) if exp != d => {
                            return Err(OspreyError::InvalidDimension { dim: d })
                        }
                        None => token_dim = Some(d),
                        Some(_) => {}
                    }
                    append_float_slice_f32(inner.values(), s, d, &mut doc_tokens)?;
                }
            }
            DataType::LargeList(_) => {
                let inner = inner_array
                    .as_any()
                    .downcast_ref::<LargeListArray>()
                    .ok_or_else(|| invalid_data("expected LargeListArray for ColBERT tokens"))?;
                let inner_offsets = inner.value_offsets();
                for tok_idx in tok_start..tok_end {
                    if inner.is_null(tok_idx) {
                        return Err(invalid_data(format!("null ColBERT token at {tok_idx}")));
                    }
                    let s: usize = i64::try_into(inner_offsets[tok_idx])
                        .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
                    let e: usize = i64::try_into(inner_offsets[tok_idx + 1])
                        .map_err(|_| OspreyError::InvalidDimension { dim: usize::MAX })?;
                    let d = e.saturating_sub(s);
                    if d == 0 {
                        return Err(OspreyError::InvalidDimension { dim: 0 });
                    }
                    match token_dim {
                        Some(exp) if exp != d => {
                            return Err(OspreyError::InvalidDimension { dim: d })
                        }
                        None => token_dim = Some(d),
                        Some(_) => {}
                    }
                    append_float_slice_f32(inner.values(), s, d, &mut doc_tokens)?;
                }
            }
            other => {
                return Err(invalid_data(format!(
                    "unsupported inner ColBERT token type {other:?}"
                )));
            }
        }
        docs.push(doc_tokens);
    }

    Ok((token_dim.unwrap_or(0), docs))
}

// -----------------------------------------------------------------------------
// Scalar Filter Attributes, Inverted RoaringBitmaps, and Zone Maps
// -----------------------------------------------------------------------------

#[inline]
fn bloom_mask_u32(val: u32) -> u64 {
    let h = xxh64(&val.to_le_bytes(), 0);
    let b1 = (h & 63) as u32;
    let b2 = ((h >> 6) & 63) as u32;
    (1_u64 << b1) | (1_u64 << b2)
}

/// Compact scalar filtering attributes stored per indexed document row.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct RowFilterAttributes {
    /// Tenant / namespace identifier for multi-tenant isolation.
    pub tenant_id: u32,
    /// Document timestamp in Unix epoch seconds.
    pub timestamp_epoch_secs: u64,
    /// 32-bit `xxHash64`-derived category hash.
    pub category_hash: u32,
}

impl RowFilterAttributes {
    /// Constructs filter attributes from pre-hashed `category_hash`.
    #[inline]
    pub const fn new(tenant_id: u32, timestamp_epoch_secs: u64, category_hash: u32) -> Self {
        Self {
            tenant_id,
            timestamp_epoch_secs,
            category_hash,
        }
    }

    /// Constructs filter attributes, hashing `category` via `xxHash64`.
    #[inline]
    pub fn with_category(tenant_id: u32, timestamp_epoch_secs: u64, category: &str) -> Self {
        Self {
            tenant_id,
            timestamp_epoch_secs,
            category_hash: hash_category(category),
        }
    }

    /// Evaluates `predicate` against this single row in strictly $O(1)$ time (modulo `And` arity).
    #[inline]
    pub fn matches(&self, predicate: &FilterPredicate) -> bool {
        match predicate {
            FilterPredicate::TenantEq(tid) => self.tenant_id == *tid,
            FilterPredicate::CategoryEq(cat) => self.category_hash == *cat,
            FilterPredicate::TimestampRange { min, max } => {
                *min <= *max
                    && self.timestamp_epoch_secs >= *min
                    && self.timestamp_epoch_secs <= *max
            }
            FilterPredicate::And(clauses) => clauses.iter().all(|c| self.matches(c)),
        }
    }
}

/// Hybrid scalar filter predicate evaluated during `.osi` scan and DP range planning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FilterPredicate {
    /// Exact match on `tenant_id`.
    TenantEq(u32),
    /// Exact match on `category_hash` (see [`hash_category`]).
    CategoryEq(u32),
    /// Inclusive Unix epoch seconds range `[min, max]`.
    TimestampRange {
        /// Inclusive lower bound in epoch seconds.
        min: u64,
        /// Inclusive upper bound in epoch seconds.
        max: u64,
    },
    /// Conjunction (`AND`) of sub-predicates.
    And(Vec<FilterPredicate>),
}

impl FilterPredicate {
    /// Convenience constructor for `CategoryEq(hash_category(category))`.
    #[inline]
    pub fn category_eq_str(category: &str) -> Self {
        Self::CategoryEq(hash_category(category))
    }
}

/// Per-minipage Zone Map summary enabling $O(1)$ pruning before DP range coalescing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MinipageZoneMap {
    /// Target `.osv` minipage identifier.
    pub minipage_id: u32,
    /// Minimum timestamp observed among all rows in this minipage.
    pub min_timestamp: u64,
    /// Maximum timestamp observed among all rows in this minipage.
    pub max_timestamp: u64,
    /// 64-bit 2-probe Bloom mask over `tenant_id` values present in this minipage.
    pub tenant_bloom_mask: u64,
    /// 64-bit 2-probe Bloom mask over `category_hash` values present in this minipage.
    pub category_bloom_mask: u64,
    /// Number of rows indexed in this minipage.
    pub row_count: u32,
}

impl MinipageZoneMap {
    /// Creates an empty Zone Map for `minipage_id`.
    #[inline]
    pub const fn new(minipage_id: u32) -> Self {
        Self {
            minipage_id,
            min_timestamp: u64::MAX,
            max_timestamp: 0,
            tenant_bloom_mask: 0,
            category_bloom_mask: 0,
            row_count: 0,
        }
    }

    /// Updates the Zone Map bounds and Bloom masks with a new row's attributes.
    #[inline]
    pub fn record(&mut self, attrs: RowFilterAttributes) {
        self.min_timestamp = self.min_timestamp.min(attrs.timestamp_epoch_secs);
        self.max_timestamp = self.max_timestamp.max(attrs.timestamp_epoch_secs);
        self.tenant_bloom_mask |= bloom_mask_u32(attrs.tenant_id);
        self.category_bloom_mask |= bloom_mask_u32(attrs.category_hash);
        self.row_count = self.row_count.saturating_add(1);
    }

    /// Returns `false` if this minipage is guaranteed to contain zero matching rows ($O(1)$).
    /// Zero false negatives; may return `true` on Bloom collision or overlapping ranges.
    #[inline]
    pub fn might_match(&self, predicate: &FilterPredicate) -> bool {
        if self.row_count == 0 {
            return false;
        }
        match predicate {
            FilterPredicate::TenantEq(tid) => {
                let req = bloom_mask_u32(*tid);
                (self.tenant_bloom_mask & req) == req
            }
            FilterPredicate::CategoryEq(cat) => {
                let req = bloom_mask_u32(*cat);
                (self.category_bloom_mask & req) == req
            }
            FilterPredicate::TimestampRange { min, max } => {
                *min <= *max && self.max_timestamp >= *min && self.min_timestamp <= *max
            }
            FilterPredicate::And(clauses) => clauses.iter().all(|c| self.might_match(c)),
        }
    }
}

/// Maps Arrow column indices in a projected `RecordBatch` to [`RowFilterAttributes`] fields.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ScalarFilterColumnMapping {
    /// Optional column index containing `tenant_id`.
    pub tenant_col_idx: Option<usize>,
    /// Optional column index containing timestamp (`UInt64`, `Int64`, or `Timestamp`).
    pub timestamp_col_idx: Option<usize>,
    /// Optional column index containing `category` (`Utf8`/`LargeUtf8`) or `category_hash` (`UInt32`..).
    pub category_col_idx: Option<usize>,
}

impl ScalarFilterColumnMapping {
    /// Automatically detects standard scalar filter columns from a projected Arrow schema.
    pub fn from_schema(schema: &Schema) -> Self {
        let mut mapping = Self::default();
        for (idx, field) in schema.fields().iter().enumerate() {
            match field.name().as_str() {
                "tenant_id" | "tenant" if mapping.tenant_col_idx.is_none() => {
                    mapping.tenant_col_idx = Some(idx);
                }
                "timestamp_epoch_secs" | "timestamp" | "created_at"
                    if mapping.timestamp_col_idx.is_none() =>
                {
                    mapping.timestamp_col_idx = Some(idx);
                }
                "category_hash" | "category" if mapping.category_col_idx.is_none() => {
                    mapping.category_col_idx = Some(idx);
                }
                _ => {}
            }
        }
        mapping
    }
}

/// Compact inverted `RoaringBitmap` index + per-minipage Zone Maps for $O(1)$ hybrid pruning.
#[derive(Clone, Debug, Default)]
pub struct ScalarFilterIndex {
    rows: Vec<RowFilterAttributes>,
    row_minipages: Vec<Option<u32>>,
    doc_id_to_row: HashMap<u64, u32>,
    by_tenant_category: HashMap<(u32, u32), RoaringBitmap>,
    by_tenant: HashMap<u32, RoaringBitmap>,
    by_category: HashMap<u32, RoaringBitmap>,
    zone_maps: HashMap<u32, MinipageZoneMap>,
    minipage_rows: HashMap<u32, RoaringBitmap>,
}

impl ScalarFilterIndex {
    /// Creates an empty `ScalarFilterIndex`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty `ScalarFilterIndex` with pre-allocated row capacity.
    pub fn with_capacity(num_rows: usize) -> Self {
        Self {
            rows: Vec::with_capacity(num_rows),
            row_minipages: Vec::with_capacity(num_rows),
            doc_id_to_row: HashMap::with_capacity(num_rows),
            by_tenant_category: HashMap::new(),
            by_tenant: HashMap::new(),
            by_category: HashMap::new(),
            zone_maps: HashMap::new(),
            minipage_rows: HashMap::new(),
        }
    }

    /// Returns the total number of indexed rows.
    #[inline]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Returns `true` if no rows have been indexed.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Inserts a single row's filter attributes (and optional `.osv` minipage assignment),
    /// returning its assigned dense `row_id: u32`.
    pub fn insert_row(
        &mut self,
        minipage_id: Option<u32>,
        attrs: RowFilterAttributes,
    ) -> OspreyResult<u32> {
        self.insert_row_internal(None, minipage_id, attrs)
    }

    /// Inserts a single row's filter attributes bound to an explicit 64-bit `doc_id`,
    /// enabling $O(1)$ eligibility lookups for 64-bit hashed IDs during `.osi` scanning.
    pub fn insert_row_with_doc_id(
        &mut self,
        doc_id: u64,
        minipage_id: Option<u32>,
        attrs: RowFilterAttributes,
    ) -> OspreyResult<u32> {
        self.insert_row_internal(Some(doc_id), minipage_id, attrs)
    }

    fn insert_row_internal(
        &mut self,
        doc_id: Option<u64>,
        minipage_id: Option<u32>,
        attrs: RowFilterAttributes,
    ) -> OspreyResult<u32> {
        let row_id = u32::try_from(self.rows.len())
            .map_err(|_| invalid_data("ScalarFilterIndex row count exceeded u32::MAX"))?;

        self.rows.push(attrs);
        self.row_minipages.push(minipage_id);
        if let Some(did) = doc_id {
            self.doc_id_to_row.insert(did, row_id);
        }

        self.by_tenant_category
            .entry((attrs.tenant_id, attrs.category_hash))
            .or_default()
            .insert(row_id);
        self.by_tenant
            .entry(attrs.tenant_id)
            .or_default()
            .insert(row_id);
        self.by_category
            .entry(attrs.category_hash)
            .or_default()
            .insert(row_id);

        if let Some(mp_id) = minipage_id {
            self.zone_maps
                .entry(mp_id)
                .or_insert_with(|| MinipageZoneMap::new(mp_id))
                .record(attrs);
            self.minipage_rows.entry(mp_id).or_default().insert(row_id);
        }

        Ok(row_id)
    }

    /// Extracts filter attributes from an Arrow `RecordBatch` and indexes all rows,
    /// returning the assigned `row_id`s.
    pub fn insert_arrow_batch(
        &mut self,
        batch: &RecordBatch,
        mapping: &ScalarFilterColumnMapping,
        minipage_ids: Option<&[u32]>,
    ) -> OspreyResult<Vec<u32>> {
        self.insert_arrow_batch_with_doc_ids(batch, mapping, None, minipage_ids)
    }

    /// Extracts filter attributes from an Arrow `RecordBatch` and indexes all rows with
    /// optional explicit 64-bit `doc_ids` and `.osv` `minipage_ids`.
    pub fn insert_arrow_batch_with_doc_ids(
        &mut self,
        batch: &RecordBatch,
        mapping: &ScalarFilterColumnMapping,
        doc_ids: Option<&[u64]>,
        minipage_ids: Option<&[u32]>,
    ) -> OspreyResult<Vec<u32>> {
        let num_rows = batch.num_rows();
        if let Some(dids) = doc_ids {
            if dids.len() != num_rows {
                return Err(OspreyError::InvalidDimension { dim: dids.len() });
            }
        }
        if let Some(mps) = minipage_ids {
            if mps.len() != num_rows {
                return Err(OspreyError::InvalidDimension { dim: mps.len() });
            }
        }

        let tenants = match mapping.tenant_col_idx {
            Some(idx) => Some(extract_tenant_col(batch, idx)?),
            None => None,
        };
        let timestamps = match mapping.timestamp_col_idx {
            Some(idx) => Some(extract_timestamp_col(batch, idx)?),
            None => None,
        };
        let categories = match mapping.category_col_idx {
            Some(idx) => Some(extract_category_col(batch, idx)?),
            None => None,
        };

        let mut assigned = Vec::with_capacity(num_rows);
        for row in 0..num_rows {
            let attrs = RowFilterAttributes {
                tenant_id: tenants.as_ref().map_or(0, |v| v[row]),
                timestamp_epoch_secs: timestamps.as_ref().map_or(0, |v| v[row]),
                category_hash: categories.as_ref().map_or(0, |v| v[row]),
            };
            let mp_id = minipage_ids.map(|mps| mps[row]);
            let did = doc_ids.map(|dids| dids[row]);
            assigned.push(self.insert_row_internal(did, mp_id, attrs)?);
        }
        Ok(assigned)
    }

    /// Returns the stored filter attributes for `row_id` in $O(1)$.
    #[inline]
    pub fn attributes(&self, row_id: u32) -> Option<&RowFilterAttributes> {
        usize::try_from(row_id)
            .ok()
            .and_then(|idx| self.rows.get(idx))
    }

    /// Returns the Zone Map summary for `minipage_id` in $O(1)$.
    #[inline]
    pub fn zone_map(&self, minipage_id: u32) -> Option<&MinipageZoneMap> {
        self.zone_maps.get(&minipage_id)
    }

    /// Returns the inverted `RoaringBitmap` for an exact `(tenant_id, category_hash)` pair.
    #[inline]
    pub fn tenant_category_bitmap(
        &self,
        tenant_id: u32,
        category_hash: u32,
    ) -> Option<&RoaringBitmap> {
        self.by_tenant_category.get(&(tenant_id, category_hash))
    }

    /// Returns the inverted `RoaringBitmap` for `tenant_id`.
    #[inline]
    pub fn tenant_bitmap(&self, tenant_id: u32) -> Option<&RoaringBitmap> {
        self.by_tenant.get(&tenant_id)
    }

    /// Returns the inverted `RoaringBitmap` for `category_hash`.
    #[inline]
    pub fn category_bitmap(&self, category_hash: u32) -> Option<&RoaringBitmap> {
        self.by_category.get(&category_hash)
    }

    /// Validates a single candidate `row_id` against `predicate` in $O(1)$ (zero heap allocation).
    #[inline]
    pub fn matches_candidate(&self, row_id: u32, predicate: &FilterPredicate) -> bool {
        let Some(attrs) = self.attributes(row_id) else {
            return false;
        };
        attrs.matches(predicate)
    }

    /// Validates a 64-bit `doc_id` against `predicate` in $O(1)$ when `doc_id`s were registered.
    #[inline]
    pub fn matches_doc_id(&self, doc_id: u64, predicate: &FilterPredicate) -> bool {
        let Some(&row_id) = self.doc_id_to_row.get(&doc_id) else {
            return false;
        };
        self.matches_candidate(row_id, predicate)
    }

    /// Checks in $O(1)$ whether a 64-bit `doc_id` belongs to a pre-evaluated `eligible_rows` bitmap.
    #[inline]
    pub fn is_doc_eligible(&self, doc_id: u64, eligible_rows: &RoaringBitmap) -> bool {
        self.doc_id_to_row
            .get(&doc_id)
            .is_some_and(|&row_id| eligible_rows.contains(row_id))
    }

    /// Returns the stored filter attributes for a 64-bit `doc_id` in $O(1)$, if registered.
    #[inline]
    pub fn attributes_for_doc_id(&self, doc_id: u64) -> Option<&RowFilterAttributes> {
        self.doc_id_to_row
            .get(&doc_id)
            .and_then(|&row_id| self.attributes(row_id))
    }

    /// Checks in $O(1)$ whether `minipage_id` can contain any row matching `predicate`.
    /// Returns `true` if no Zone Map is recorded for `minipage_id` (safe conservative default).
    #[inline]
    pub fn minipage_might_match(&self, minipage_id: u32, predicate: &FilterPredicate) -> bool {
        match self.zone_maps.get(&minipage_id) {
            Some(zm) => zm.might_match(predicate),
            None => true,
        }
    }

    /// Filters `candidate_minipages` in $O(k)$ using Zone Maps prior to DP planning,
    /// appending surviving minipage IDs into `out_surviving`.
    pub fn prune_minipages(
        &self,
        candidate_minipages: &[u32],
        predicate: &FilterPredicate,
        out_surviving: &mut Vec<u32>,
    ) {
        out_surviving.clear();
        for &mp_id in candidate_minipages {
            if self.minipage_might_match(mp_id, predicate) {
                out_surviving.push(mp_id);
            }
        }
    }

    /// Evaluates `predicate` using inverted `RoaringBitmap`s and Zone Maps, returning the
    /// exact set of eligible `row_id`s.
    pub fn evaluate(&self, predicate: &FilterPredicate) -> RoaringBitmap {
        let mut plan = AndPlan::default();
        if !collect_and_plan(predicate, &mut plan) {
            return RoaringBitmap::new();
        }

        let base_bitmap: Option<&RoaringBitmap> = match (plan.tenant, plan.category) {
            (Some(t), Some(c)) => match self.by_tenant_category.get(&(t, c)) {
                Some(bm) => Some(bm),
                None => return RoaringBitmap::new(),
            },
            (Some(t), None) => match self.by_tenant.get(&t) {
                Some(bm) => Some(bm),
                None => return RoaringBitmap::new(),
            },
            (None, Some(c)) => match self.by_category.get(&c) {
                Some(bm) => Some(bm),
                None => return RoaringBitmap::new(),
            },
            (None, None) => None,
        };

        let ts_range = plan.ts_range;

        match (base_bitmap, ts_range) {
            (Some(bm), None) => bm.clone(),
            (Some(bm), Some((min_ts, max_ts))) => {
                let mut result = RoaringBitmap::new();
                for row_id in bm.iter() {
                    if let Ok(idx) = usize::try_from(row_id) {
                        if let Some(Some(mp_id)) = self.row_minipages.get(idx) {
                            if let Some(zm) = self.zone_maps.get(mp_id) {
                                if zm.max_timestamp < min_ts || zm.min_timestamp > max_ts {
                                    continue;
                                }
                            }
                        }
                        if let Some(attrs) = self.rows.get(idx) {
                            if attrs.timestamp_epoch_secs >= min_ts
                                && attrs.timestamp_epoch_secs <= max_ts
                            {
                                result.insert(row_id);
                            }
                        }
                    }
                }
                result
            }
            (None, None) => {
                let mut result = RoaringBitmap::new();
                if let Ok(len_u32) = u32::try_from(self.rows.len()) {
                    result.insert_range(0..len_u32);
                }
                result
            }
            (None, Some((min_ts, max_ts))) => {
                let mut result = RoaringBitmap::new();
                if !self.zone_maps.is_empty() && self.row_minipages.iter().all(Option::is_some) {
                    for (mp_id, zm) in &self.zone_maps {
                        if zm.max_timestamp < min_ts || zm.min_timestamp > max_ts {
                            continue;
                        }
                        if zm.min_timestamp >= min_ts && zm.max_timestamp <= max_ts {
                            if let Some(rows_bm) = self.minipage_rows.get(mp_id) {
                                result |= rows_bm;
                            }
                            continue;
                        }
                        if let Some(rows_bm) = self.minipage_rows.get(mp_id) {
                            for row_id in rows_bm.iter() {
                                if let Some(attrs) = self.attributes(row_id) {
                                    if attrs.timestamp_epoch_secs >= min_ts
                                        && attrs.timestamp_epoch_secs <= max_ts
                                    {
                                        result.insert(row_id);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    for (idx, attrs) in self.rows.iter().enumerate() {
                        if attrs.timestamp_epoch_secs >= min_ts
                            && attrs.timestamp_epoch_secs <= max_ts
                        {
                            if let Ok(row_id) = u32::try_from(idx) {
                                result.insert(row_id);
                            }
                        }
                    }
                }
                result
            }
        }
    }
}

#[derive(Default)]
struct AndPlan {
    tenant: Option<u32>,
    category: Option<u32>,
    ts_range: Option<(u64, u64)>,
}

fn collect_and_plan(predicate: &FilterPredicate, plan: &mut AndPlan) -> bool {
    match predicate {
        FilterPredicate::TenantEq(tid) => {
            if let Some(existing) = plan.tenant {
                if existing != *tid {
                    return false;
                }
            } else {
                plan.tenant = Some(*tid);
            }
            true
        }
        FilterPredicate::CategoryEq(cat) => {
            if let Some(existing) = plan.category {
                if existing != *cat {
                    return false;
                }
            } else {
                plan.category = Some(*cat);
            }
            true
        }
        FilterPredicate::TimestampRange { min, max } => {
            if *min > *max {
                return false;
            }
            match plan.ts_range {
                Some((cur_min, cur_max)) => {
                    let eff_min = cur_min.max(*min);
                    let eff_max = cur_max.min(*max);
                    if eff_min > eff_max {
                        return false;
                    }
                    plan.ts_range = Some((eff_min, eff_max));
                }
                None => plan.ts_range = Some((*min, *max)),
            }
            true
        }
        FilterPredicate::And(clauses) => {
            for clause in clauses {
                if !collect_and_plan(clause, plan) {
                    return false;
                }
            }
            true
        }
    }
}

fn extract_tenant_col(batch: &RecordBatch, col_idx: usize) -> OspreyResult<Vec<u32>> {
    let col = batch
        .columns()
        .get(col_idx)
        .ok_or_else(|| invalid_input(format!("tenant column index {col_idx} out of bounds")))?;
    let mut out = Vec::with_capacity(batch.num_rows());
    match col.data_type() {
        DataType::UInt32 => {
            let arr = col
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| invalid_data("expected UInt32Array for tenant"))?;
            for r in 0..arr.len() {
                out.push(if arr.is_null(r) { 0 } else { arr.value(r) });
            }
        }
        DataType::Int32 => {
            let arr = col
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| invalid_data("expected Int32Array for tenant"))?;
            for r in 0..arr.len() {
                if arr.is_null(r) {
                    out.push(0);
                } else {
                    let v = u32::try_from(arr.value(r))
                        .map_err(|_| invalid_data(format!("negative tenant_id at row {r}")))?;
                    out.push(v);
                }
            }
        }
        DataType::UInt64 => {
            let arr = col
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| invalid_data("expected UInt64Array for tenant"))?;
            for r in 0..arr.len() {
                if arr.is_null(r) {
                    out.push(0);
                } else {
                    let v = u32::try_from(arr.value(r))
                        .map_err(|_| invalid_data(format!("tenant_id overflow at row {r}")))?;
                    out.push(v);
                }
            }
        }
        DataType::Int64 => {
            let arr = col
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| invalid_data("expected Int64Array for tenant"))?;
            for r in 0..arr.len() {
                if arr.is_null(r) {
                    out.push(0);
                } else {
                    let v = u32::try_from(arr.value(r))
                        .map_err(|_| invalid_data(format!("invalid Int64 tenant_id at row {r}")))?;
                    out.push(v);
                }
            }
        }
        DataType::Utf8 => {
            let arr = col
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| invalid_data("expected StringArray for tenant"))?;
            for r in 0..arr.len() {
                out.push(if arr.is_null(r) {
                    0
                } else {
                    hash_category(arr.value(r))
                });
            }
        }
        DataType::LargeUtf8 => {
            let arr = col
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(|| invalid_data("expected LargeStringArray for tenant"))?;
            for r in 0..arr.len() {
                out.push(if arr.is_null(r) {
                    0
                } else {
                    hash_category(arr.value(r))
                });
            }
        }
        other => {
            return Err(invalid_data(format!(
                "unsupported tenant column type {other:?}"
            )));
        }
    }
    Ok(out)
}

fn extract_timestamp_col(batch: &RecordBatch, col_idx: usize) -> OspreyResult<Vec<u64>> {
    let col = batch
        .columns()
        .get(col_idx)
        .ok_or_else(|| invalid_input(format!("timestamp column index {col_idx} out of bounds")))?;
    let mut out = Vec::with_capacity(batch.num_rows());
    match col.data_type() {
        DataType::UInt64 => {
            let arr = col
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| invalid_data("expected UInt64Array for timestamp"))?;
            for r in 0..arr.len() {
                out.push(if arr.is_null(r) { 0 } else { arr.value(r) });
            }
        }
        DataType::Int64 => {
            let arr = col
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| invalid_data("expected Int64Array for timestamp"))?;
            for r in 0..arr.len() {
                out.push(if arr.is_null(r) {
                    0
                } else {
                    u64::try_from(arr.value(r).max(0)).unwrap_or(0)
                });
            }
        }
        DataType::Timestamp(unit, _) => match unit {
            TimeUnit::Second => {
                let arr = col
                    .as_any()
                    .downcast_ref::<TimestampSecondArray>()
                    .ok_or_else(|| invalid_data("expected TimestampSecondArray"))?;
                for r in 0..arr.len() {
                    out.push(if arr.is_null(r) {
                        0
                    } else {
                        u64::try_from(arr.value(r).max(0)).unwrap_or(0)
                    });
                }
            }
            TimeUnit::Millisecond => {
                let arr = col
                    .as_any()
                    .downcast_ref::<TimestampMillisecondArray>()
                    .ok_or_else(|| invalid_data("expected TimestampMillisecondArray"))?;
                for r in 0..arr.len() {
                    out.push(if arr.is_null(r) {
                        0
                    } else {
                        u64::try_from((arr.value(r) / 1_000).max(0)).unwrap_or(0)
                    });
                }
            }
            TimeUnit::Microsecond => {
                let arr = col
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .ok_or_else(|| invalid_data("expected TimestampMicrosecondArray"))?;
                for r in 0..arr.len() {
                    out.push(if arr.is_null(r) {
                        0
                    } else {
                        u64::try_from((arr.value(r) / 1_000_000).max(0)).unwrap_or(0)
                    });
                }
            }
            TimeUnit::Nanosecond => {
                let arr = col
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                    .ok_or_else(|| invalid_data("expected TimestampNanosecondArray"))?;
                for r in 0..arr.len() {
                    out.push(if arr.is_null(r) {
                        0
                    } else {
                        u64::try_from((arr.value(r) / 1_000_000_000).max(0)).unwrap_or(0)
                    });
                }
            }
        },
        other => {
            return Err(invalid_data(format!(
                "unsupported timestamp column type {other:?}"
            )));
        }
    }
    Ok(out)
}

fn extract_category_col(batch: &RecordBatch, col_idx: usize) -> OspreyResult<Vec<u32>> {
    let col = batch
        .columns()
        .get(col_idx)
        .ok_or_else(|| invalid_input(format!("category column index {col_idx} out of bounds")))?;
    let mut out = Vec::with_capacity(batch.num_rows());
    match col.data_type() {
        DataType::Utf8 => {
            let arr = col
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| invalid_data("expected StringArray for category"))?;
            for r in 0..arr.len() {
                out.push(if arr.is_null(r) {
                    0
                } else {
                    hash_category(arr.value(r))
                });
            }
        }
        DataType::LargeUtf8 => {
            let arr = col
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(|| invalid_data("expected LargeStringArray for category"))?;
            for r in 0..arr.len() {
                out.push(if arr.is_null(r) {
                    0
                } else {
                    hash_category(arr.value(r))
                });
            }
        }
        DataType::UInt32 => {
            let arr = col
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| invalid_data("expected UInt32Array for category_hash"))?;
            for r in 0..arr.len() {
                out.push(if arr.is_null(r) { 0 } else { arr.value(r) });
            }
        }
        DataType::Int32 => {
            let arr = col
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| invalid_data("expected Int32Array for category_hash"))?;
            for r in 0..arr.len() {
                out.push(if arr.is_null(r) {
                    0
                } else {
                    u32::try_from(arr.value(r))
                        .map_err(|_| invalid_data(format!("negative category_hash at row {r}")))?
                });
            }
        }
        DataType::UInt64 => {
            let arr = col
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| invalid_data("expected UInt64Array for category_hash"))?;
            for r in 0..arr.len() {
                out.push(if arr.is_null(r) {
                    0
                } else {
                    u32::try_from(arr.value(r))
                        .map_err(|_| invalid_data(format!("category_hash overflow at row {r}")))?
                });
            }
        }
        other => {
            return Err(invalid_data(format!(
                "unsupported category column type {other:?}"
            )));
        }
    }
    Ok(out)
}

// -----------------------------------------------------------------------------
// Out-of-Core Two-Pass Streaming Ingest Pipeline
// -----------------------------------------------------------------------------

/// Default training reservoir sample cap (`100_000` vectors) for bounded-RAM Pass 1.
pub const DEFAULT_MAX_TRAINING_SAMPLE_VECTORS: usize = 100_000;

/// Default Pass-2 streaming batch size (`16_384` rows per chunk).
pub const DEFAULT_OUT_OF_CORE_BATCH_ROWS: usize = 16_384;

/// Configuration for [`OutOfCoreIngestPipeline`].
#[derive(Clone, Debug)]
pub struct OutOfCoreIngestConfig {
    /// Maximum number of vectors retained in the Pass-1 reservoir sample (Algorithm R).
    pub max_training_sample_vectors: usize,
    /// Number of rows read per chunked batch during Pass-2 streaming ingestion.
    pub batch_size_rows: usize,
    /// Target number of `1-Bit Binary K-Means` clusters (clamped to `sample_count` if smaller).
    pub num_clusters: usize,
    /// Deterministic `fwht_seed` for orthogonal Rademacher + FWHT rotation.
    pub fwht_seed: u64,
    /// Deterministic `SplitMix64` seed for Algorithm R reservoir sampling.
    pub reservoir_seed: u64,
    /// Maximum 2-opt refinement passes for TSP centroid ordering.
    pub tsp_max_2opt_passes: usize,
    /// Column hints for Parquet schema inspection and projection pushdown.
    pub hints: IngestColumnHints,
    /// `.osv` slab writer configuration used during Pass-2 chunk emission.
    pub osv_config: OsvWriterConfig,
}

impl Default for OutOfCoreIngestConfig {
    fn default() -> Self {
        Self {
            max_training_sample_vectors: DEFAULT_MAX_TRAINING_SAMPLE_VECTORS,
            batch_size_rows: DEFAULT_OUT_OF_CORE_BATCH_ROWS,
            num_clusters: 64,
            fwht_seed: 0x4F53_5052_4559_3130,
            reservoir_seed: 0x9E37_79B9_7F4A_7C15,
            tsp_max_2opt_passes: 20,
            hints: IngestColumnHints::default(),
            osv_config: OsvWriterConfig::default(),
        }
    }
}

/// Trained `1-Bit Binary K-Means` codebook with TSP-ordered `f16` centroids.
#[derive(Clone, Debug)]
pub struct TrainedLakeCodebook {
    /// Original vector dimensionality.
    pub orig_dim: usize,
    /// FWHT-padded power-of-two dimensionality (`>= 64`).
    pub padded_dim: usize,
    /// Number of trained clusters `K`.
    pub num_clusters: usize,
    /// Deterministic FWHT seed used for rotation.
    pub fwht_seed: u64,
    /// TSP-ordered centroids in `half::f16` (`[num_clusters * padded_dim]`).
    pub centroids_f16: Vec<f16>,
    /// Pre-promoted `f32` copy of `centroids_f16` for fast Pass-2 assignment.
    centroids_f32: Vec<f32>,
    /// Precomputed Rademacher sign XOR masks for FWHT rotation.
    xor_masks: Vec<u32>,
    /// Bidirectional logical <-> TSP physical cluster permutation.
    pub permutation: ClusterLocalityPermutation,
}

/// Summary produced by [`OutOfCoreIngestPipeline`] across Pass 1 and Pass 2.
#[derive(Debug)]
pub struct OutOfCoreIngestReport {
    /// Detected source Data Lake schema.
    pub detected_schema: DetectedLakeSchema,
    /// Number of vectors actually retained in the Pass-1 reservoir (`<= max_training_sample_vectors`).
    pub total_vectors_sampled: usize,
    /// Total vectors observed across the Parquet dataset.
    pub total_vectors_seen: u64,
    /// Total vectors emitted to `OsvWriter` in Pass 2.
    pub total_vectors_indexed: u64,
    /// Number of streaming chunk batches processed in Pass 2.
    pub total_batches: usize,
    /// Trained and TSP-reordered codebook.
    pub codebook: TrainedLakeCodebook,
    /// Hybrid scalar filter index + per-minipage Zone Maps populated during Pass 2.
    pub filter_index: ScalarFilterIndex,
    /// Per-chunk `.osv` slab write reports produced during Pass 2.
    pub chunk_reports: Vec<OsvWriteReport>,
}

/// Two-pass out-of-core streaming ingest pipeline for Parquet tables larger than RAM.
#[derive(Clone, Debug, Default)]
pub struct OutOfCoreIngestPipeline {
    /// Pipeline configuration.
    pub config: OutOfCoreIngestConfig,
}

impl OutOfCoreIngestPipeline {
    /// Creates an [`OutOfCoreIngestPipeline`] with the given configuration.
    #[inline]
    pub const fn new(config: OutOfCoreIngestConfig) -> Self {
        Self { config }
    }

    /// Pass 1: Scans only the embedding column via `ProjectionMask::roots` and selects up to
    /// `max_training_sample_vectors` via Vitter's deterministic Algorithm R with `SplitMix64`.
    ///
    /// Peak RAM is strictly bounded by `O(max_training_sample_vectors * orig_dim * 4)`.
    pub fn run_pass1_reservoir_sampling<R: ChunkReader + 'static>(
        &self,
        reader: R,
    ) -> OspreyResult<(DetectedLakeSchema, Vec<f32>, usize, u64)> {
        let cap = self.config.max_training_sample_vectors.max(1);
        let batch_rows = self.config.batch_size_rows.max(1);

        let builder = ParquetRecordBatchReaderBuilder::try_new_with_options(
            reader,
            ArrowReaderOptions::new(),
        )
        .map_err(|err| invalid_data(format!("Pass 1 Parquet metadata error: {err}")))?;

        let detected = inspect_parquet_schema(builder.schema().as_ref(), &self.config.hints)?;
        let vector_only_mask = ProjectionMask::roots(
            builder.metadata().file_metadata().schema_descr(),
            [detected.vector_col_idx],
        );

        let mut vec_reader = builder
            .with_projection(vector_only_mask)
            .with_batch_size(batch_rows)
            .build()
            .map_err(|err| invalid_data(format!("Pass 1 Parquet reader build error: {err}")))?;

        let mut rng = SplitMix64::new(self.config.reservoir_seed);
        let mut inferred_dim = (detected.orig_dim > 0).then_some(detected.orig_dim);
        let mut reservoir: Vec<f32> = Vec::new();
        let mut sampled_count = 0_usize;
        let mut total_seen = 0_u64;

        for batch_res in &mut vec_reader {
            let batch =
                batch_res.map_err(|err| invalid_data(format!("Pass 1 read error: {err}")))?;
            if batch.num_rows() == 0 {
                continue;
            }
            let (batch_dim, batch_vecs) = extract_batch_vectors_f32(&batch, 0, inferred_dim)?;
            if inferred_dim.is_none() {
                inferred_dim = Some(batch_dim);
                reservoir.reserve(cap.min(batch.num_rows().max(cap)) * batch_dim);
            }

            let rows_in_batch = batch.num_rows();
            for r in 0..rows_in_batch {
                let start = r * batch_dim;
                let vec_slice = &batch_vecs[start..start + batch_dim];

                if sampled_count < cap {
                    reservoir.extend_from_slice(vec_slice);
                    sampled_count += 1;
                } else {
                    let j_u64 = rng.next_u64() % (total_seen + 1);
                    let j = usize::try_from(j_u64).unwrap_or(usize::MAX);
                    if j < cap {
                        let dst_start = j * batch_dim;
                        reservoir[dst_start..dst_start + batch_dim].copy_from_slice(vec_slice);
                    }
                }
                total_seen += 1;
            }
        }

        if sampled_count == 0 {
            return Err(invalid_data(
                "Parquet dataset contained zero vectors in Pass 1",
            ));
        }

        let mut final_schema = detected;
        if final_schema.orig_dim == 0 {
            final_schema.orig_dim = inferred_dim.unwrap_or(0);
        }

        Ok((final_schema, reservoir, sampled_count, total_seen))
    }

    /// Trains `1-Bit Binary K-Means` on the Pass-1 reservoir sample and solves TSP locality ordering.
    pub fn train_codebook(
        &self,
        samples_f32: &[f32],
        sample_count: usize,
        orig_dim: usize,
    ) -> OspreyResult<TrainedLakeCodebook> {
        if sample_count == 0 || orig_dim == 0 {
            return Err(OspreyError::InvalidDimension { dim: 0 });
        }
        let k = self.config.num_clusters.max(1).min(sample_count);
        let sk_config = BinaryKMeansConfig::new(k);
        let km = fit_binary_kmeans(
            samples_f32,
            sample_count,
            orig_dim,
            self.config.fwht_seed,
            &sk_config,
            &ScalarClusterKernel,
        )?;

        let padded_dim = padded_dimension(orig_dim)?;
        let mut counts = vec![0_u64; k];
        for &cid in &km.assignments {
            if let Ok(idx) = usize::try_from(cid) {
                if let Some(c) = counts.get_mut(idx) {
                    *c += 1;
                }
            }
        }

        let permutation = solve_tsp_locality_largest_first(
            km.centroids_f16.as_slice(),
            k,
            padded_dim,
            &counts,
            self.config.tsp_max_2opt_passes,
        )?;
        let centroids_f16 =
            permutation.reorder_centroids_f16(km.centroids_f16.as_slice(), padded_dim)?;
        let centroids_f32: Vec<f32> = centroids_f16.iter().map(|v| v.to_f32()).collect();
        let signs = RademacherSigns::from_seed(padded_dim, self.config.fwht_seed)?;

        Ok(TrainedLakeCodebook {
            orig_dim,
            padded_dim,
            num_clusters: k,
            fwht_seed: self.config.fwht_seed,
            centroids_f16,
            centroids_f32,
            xor_masks: signs.xor_masks().to_vec(),
            permutation,
        })
    }

    /// Rotates a single raw vector via FWHT and assigns it to its nearest TSP physical centroid.
    ///
    /// Reuses `scratch` to guarantee zero per-vector heap allocation in Pass 2.
    pub fn assign_tsp_cluster(
        &self,
        codebook: &TrainedLakeCodebook,
        vector_f32: &[f32],
        scratch: &mut Vec<f32>,
    ) -> OspreyResult<u32> {
        if vector_f32.len() != codebook.orig_dim {
            return Err(OspreyError::InvalidDimension {
                dim: vector_f32.len(),
            });
        }
        scratch.resize(codebook.padded_dim, 0.0_f32);
        prepare_and_pad_vector(vector_f32, scratch)?;
        ScalarClusterKernel.fwht_rotate_in_place(scratch, &codebook.xor_masks)?;

        let mut best_cluster = 0_u32;
        let mut best_dist_sq = f32::INFINITY;
        for c in 0..codebook.num_clusters {
            let start = c * codebook.padded_dim;
            let centroid = &codebook.centroids_f32[start..start + codebook.padded_dim];
            let mut dist_sq = 0.0_f32;
            for (&a, &b) in scratch.iter().zip(centroid.iter()) {
                let d = a - b;
                dist_sq += d * d;
            }
            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best_cluster =
                    u32::try_from(c).map_err(|_| OspreyError::InvalidDimension { dim: c })?;
            }
        }
        Ok(best_cluster)
    }

    /// Executes the full two-pass out-of-core pipeline against a Parquet source factory
    /// (`make_reader`), streaming each Pass-2 chunk through `OsvWriter` via `write_chunk` without
    /// retaining raw `f32` vectors across batches.
    pub fn ingest_chunked<R, MakeReader, WriteChunk>(
        &self,
        make_reader: MakeReader,
        mut write_chunk: WriteChunk,
    ) -> OspreyResult<OutOfCoreIngestReport>
    where
        R: ChunkReader + 'static,
        MakeReader: Fn() -> OspreyResult<R>,
        WriteChunk:
            FnMut(usize, &OsvWriter, &[OsvVectorRecord<'_>], usize) -> OspreyResult<OsvWriteReport>,
    {
        // Pass 1: Reservoir sample only the vector column.
        let r1 = make_reader()?;
        let (detected_schema, reservoir_f32, total_vectors_sampled, total_vectors_seen) =
            self.run_pass1_reservoir_sampling(r1)?;

        let codebook = self.train_codebook(
            &reservoir_f32,
            total_vectors_sampled,
            detected_schema.orig_dim,
        )?;
        drop(reservoir_f32);

        // Pass 2: Stream bounded chunks with projection pushdown.
        let r2 = make_reader()?;
        let batch_rows = self.config.batch_size_rows.max(1);
        let mut reader = ParquetBatchReader::from_chunk_reader(r2, &self.config.hints, batch_rows)?;

        let col_map = reader.projected_map().clone();
        let projected_schema = reader.projected_schema().clone();
        let filter_mapping = ScalarFilterColumnMapping::from_schema(projected_schema.as_ref());
        let osv_writer = OsvWriter::new(self.config.osv_config.clone());

        let mut filter_index = ScalarFilterIndex::new();
        let mut chunk_reports = Vec::new();
        let mut rot_scratch = vec![0.0_f32; codebook.padded_dim];
        let mut total_vectors_indexed = 0_u64;
        let mut total_batches = 0_usize;
        let mut minipage_offset = 0_u32;

        for batch_res in &mut reader {
            let batch = batch_res?;
            let n = batch.num_rows();
            if n == 0 {
                continue;
            }

            let doc_ids = extract_batch_doc_ids(&batch, col_map.id_batch_idx)?;
            let (dim, vectors_f32) = extract_batch_vectors_f32(
                &batch,
                col_map.vector_batch_idx,
                Some(codebook.orig_dim),
            )?;
            let payloads =
                build_batch_metadata_json_payloads(&batch, &col_map.metadata_batch_indices)?;

            let mut cluster_ids = Vec::with_capacity(n);
            let mut vectors_f16 = Vec::with_capacity(n * dim);
            for row in 0..n {
                let vec_slice = &vectors_f32[row * dim..(row + 1) * dim];
                cluster_ids.push(self.assign_tsp_cluster(
                    &codebook,
                    vec_slice,
                    &mut rot_scratch,
                )?);
                for &v in vec_slice {
                    vectors_f16.push(f16::from_f32(v));
                }
            }
            // Release raw f32 batch buffer before compressing and writing slabs.
            drop(vectors_f32);

            let mut records = Vec::with_capacity(n);
            for row in 0..n {
                records.push(OsvVectorRecord {
                    doc_id: doc_ids[row],
                    cluster_id: cluster_ids[row],
                    vector_f16: &vectors_f16[row * dim..(row + 1) * dim],
                    payload: &payloads[row],
                });
            }

            let report = write_chunk(total_batches, &osv_writer, &records, dim)?;
            let mut batch_minipages = Vec::with_capacity(n);
            for &doc_id in &doc_ids {
                let local_mp = report
                    .location_for_doc(doc_id)
                    .map(|(mp, _)| mp)
                    .unwrap_or(0);
                batch_minipages.push(minipage_offset.saturating_add(local_mp));
            }
            filter_index.insert_arrow_batch_with_doc_ids(
                &batch,
                &filter_mapping,
                Some(&doc_ids),
                Some(&batch_minipages),
            )?;

            let page_cnt = u32::try_from(report.directory.len()).map_err(|_| {
                OspreyError::InvalidDimension {
                    dim: report.directory.len(),
                }
            })?;
            minipage_offset = minipage_offset.saturating_add(page_cnt);
            total_vectors_indexed += u64::try_from(n).unwrap_or(0);
            total_batches += 1;
            chunk_reports.push(report);
        }

        Ok(OutOfCoreIngestReport {
            detected_schema,
            total_vectors_sampled,
            total_vectors_seen,
            total_vectors_indexed,
            total_batches,
            codebook,
            filter_index,
            chunk_reports,
        })
    }

    /// Convenience helper streaming a local Parquet file into chunked `.osv` slab files
    /// (`chunk_000000.osv`, `chunk_000001.osv`, ...) inside `out_dir`.
    pub fn ingest_parquet_file_to_dir(
        &self,
        parquet_path: impl AsRef<Path>,
        out_dir: impl AsRef<Path>,
    ) -> OspreyResult<OutOfCoreIngestReport> {
        let parquet_path = parquet_path.as_ref().to_path_buf();
        let out_dir = out_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&out_dir)?;

        self.ingest_chunked(
            || File::open(&parquet_path).map_err(OspreyError::from),
            |chunk_idx, writer, records, orig_dim| {
                let out_path = out_dir.join(format!("chunk_{chunk_idx:06}.osv"));
                let file = File::create(out_path)?;
                let mut buf = BufWriter::new(file);
                writer.write_slab(&mut buf, records, orig_dim, None)
            },
        )
    }
}

fn build_batch_metadata_json_payloads(
    batch: &RecordBatch,
    metadata_batch_indices: &[(usize, usize)],
) -> OspreyResult<Vec<Vec<u8>>> {
    let n = batch.num_rows();
    if metadata_batch_indices.is_empty() {
        return Ok(vec![b"{}".to_vec(); n]);
    }

    let schema = batch.schema();
    let mut payloads = Vec::with_capacity(n);
    for row in 0..n {
        let mut obj = serde_json::Map::with_capacity(metadata_batch_indices.len());
        for &(_, batch_col_idx) in metadata_batch_indices {
            let field_name = schema.field(batch_col_idx).name();
            let col = batch.column(batch_col_idx);
            if col.is_null(row) {
                obj.insert(field_name.clone(), serde_json::Value::Null);
                continue;
            }
            let val = match col.data_type() {
                DataType::UInt32 => col
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .map(|a| serde_json::Value::from(a.value(row)))
                    .unwrap_or(serde_json::Value::Null),
                DataType::Int32 => col
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .map(|a| serde_json::Value::from(a.value(row)))
                    .unwrap_or(serde_json::Value::Null),
                DataType::UInt64 => col
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .map(|a| serde_json::Value::from(a.value(row)))
                    .unwrap_or(serde_json::Value::Null),
                DataType::Int64 => col
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .map(|a| serde_json::Value::from(a.value(row)))
                    .unwrap_or(serde_json::Value::Null),
                DataType::Float32 => col
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .and_then(|a| serde_json::Number::from_f64(f64::from(a.value(row))))
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null),
                DataType::Float64 => col
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .and_then(|a| serde_json::Number::from_f64(a.value(row)))
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null),
                DataType::Boolean => col
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .map(|a| serde_json::Value::from(a.value(row)))
                    .unwrap_or(serde_json::Value::Null),
                DataType::Utf8 => col
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .map(|a| serde_json::Value::from(a.value(row)))
                    .unwrap_or(serde_json::Value::Null),
                DataType::LargeUtf8 => col
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .map(|a| serde_json::Value::from(a.value(row)))
                    .unwrap_or(serde_json::Value::Null),
                DataType::Timestamp(unit, _) => {
                    let secs = match unit {
                        TimeUnit::Second => col
                            .as_any()
                            .downcast_ref::<TimestampSecondArray>()
                            .map(|a| a.value(row)),
                        TimeUnit::Millisecond => col
                            .as_any()
                            .downcast_ref::<TimestampMillisecondArray>()
                            .map(|a| a.value(row) / 1_000),
                        TimeUnit::Microsecond => col
                            .as_any()
                            .downcast_ref::<TimestampMicrosecondArray>()
                            .map(|a| a.value(row) / 1_000_000),
                        TimeUnit::Nanosecond => col
                            .as_any()
                            .downcast_ref::<TimestampNanosecondArray>()
                            .map(|a| a.value(row) / 1_000_000_000),
                    };
                    secs.map(serde_json::Value::from)
                        .unwrap_or(serde_json::Value::Null)
                }
                _ => serde_json::Value::Null,
            };
            obj.insert(field_name.clone(), val);
        }
        let json_bytes = serde_json::to_vec(&obj)
            .map_err(|err| invalid_data(format!("JSON encode failed: {err}")))?;
        payloads.push(json_bytes);
    }
    Ok(payloads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        FixedSizeListBuilder, Float16Builder, Float32Builder, ListBuilder, StringBuilder,
        UInt32Builder, UInt64Builder,
    };
    use arrow::datatypes::Field;
    use bytes::Bytes;
    use futures::StreamExt;
    use half::f16;
    use object_store::memory::InMemory;
    use parquet::arrow::ArrowWriter;

    fn write_parquet_bytes(batch: &RecordBatch) -> OspreyResult<Bytes> {
        let mut buf = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut buf, batch.schema(), None)
            .map_err(|e| invalid_data(e.to_string()))?;
        writer
            .write(batch)
            .map_err(|e| invalid_data(e.to_string()))?;
        writer.close().map_err(|e| invalid_data(e.to_string()))?;
        Ok(Bytes::from(buf))
    }

    #[test]
    fn test_schema_inspector_auto_detection_f32_and_colbert() -> OspreyResult<()> {
        let schema = Schema::new(vec![
            Field::new("raw_html_blob", DataType::LargeUtf8, true),
            Field::new("doc_id", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 4),
                false,
            ),
            Field::new(
                "colbert_tokens",
                DataType::List(Arc::new(Field::new(
                    "item",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float16, false)),
                        128,
                    ),
                    true,
                ))),
                true,
            ),
            Field::new("tenant_id", DataType::UInt32, false),
            Field::new("category", DataType::Utf8, false),
        ]);

        let detected = inspect_parquet_schema(&schema, &IngestColumnHints::default())?;
        assert_eq!(detected.id_col_idx, 1);
        assert_eq!(detected.id_col_name, "doc_id");
        assert!(detected.id_is_string);
        assert_eq!(detected.vector_col_idx, 2);
        assert_eq!(detected.vector_col_name, "embedding");
        assert_eq!(detected.orig_dim, 4);
        assert!(!detected.is_f16);
        assert_eq!(detected.colbert_col_idx, Some(3));
        assert_eq!(detected.colbert_token_dim, Some(128));
        assert_eq!(detected.metadata_col_indices, vec![4, 5]);
        assert_eq!(detected.projected_indices(), vec![1, 2, 3, 4, 5]);
        Ok(())
    }

    #[test]
    fn test_schema_inspector_list_f16_and_int64_id() -> OspreyResult<()> {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new(
                "vec",
                DataType::List(Arc::new(Field::new("item", DataType::Float16, true))),
                false,
            ),
            Field::new(
                "multi_vec",
                DataType::List(Arc::new(Field::new(
                    "item",
                    DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                    true,
                ))),
                true,
            ),
            Field::new("unrequested_logs", DataType::Utf8, true),
        ]);

        let hints = IngestColumnHints {
            expected_vector_dim: Some(64),
            expected_colbert_dim: Some(128),
            ..Default::default()
        };
        let detected = inspect_parquet_schema(&schema, &hints)?;
        assert_eq!(detected.id_col_idx, 0);
        assert!(!detected.id_is_string);
        assert_eq!(detected.vector_col_idx, 1);
        assert_eq!(detected.orig_dim, 64);
        assert!(detected.is_f16);
        assert_eq!(detected.colbert_col_idx, Some(2));
        assert_eq!(detected.colbert_token_dim, Some(128));
        assert!(detected.metadata_col_indices.is_empty());
        assert_eq!(detected.projected_indices(), vec![0, 1, 2]);
        Ok(())
    }

    #[test]
    fn test_projection_pushdown_ignores_heavy_blobs_and_extracts_vectors() -> OspreyResult<()> {
        let mut html_builder = StringBuilder::new();
        let mut id_builder = StringBuilder::new();
        let mut vec_builder = FixedSizeListBuilder::new(Float32Builder::new(), 3);
        let mut colbert_builder =
            ListBuilder::new(FixedSizeListBuilder::new(Float16Builder::new(), 2));
        let mut tenant_builder = UInt32Builder::new();
        let mut ts_builder = UInt64Builder::new();
        let mut cat_builder = StringBuilder::new();

        for i in 0..4_u32 {
            html_builder.append_value(format!("<html><body>heavy-blob-{i}</body></html>"));
            let doc_key = format!("doc-{i}");
            id_builder.append_value(&doc_key);
            vec_builder.values().append_value(i as f32);
            vec_builder.values().append_value((i + 1) as f32);
            vec_builder.values().append_value((i + 2) as f32);
            vec_builder.append(true);

            // 2 tokens of dim 2 per doc
            colbert_builder
                .values()
                .values()
                .append_value(f16::from_f32(1.0));
            colbert_builder
                .values()
                .values()
                .append_value(f16::from_f32(0.5));
            colbert_builder.values().append(true);
            colbert_builder
                .values()
                .values()
                .append_value(f16::from_f32(i as f32));
            colbert_builder
                .values()
                .values()
                .append_value(f16::from_f32(-0.5));
            colbert_builder.values().append(true);
            colbert_builder.append(true);

            tenant_builder.append_value(if i < 2 { 10 } else { 20 });
            ts_builder.append_value(1_700_000_000 + u64::from(i) * 100);
            cat_builder.append_value(if i % 2 == 0 { "finance" } else { "legal" });
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("raw_html", DataType::Utf8, false),
            Field::new("doc_id", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 3),
                false,
            ),
            Field::new(
                "colbert",
                DataType::List(Arc::new(Field::new(
                    "item",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float16, true)),
                        2,
                    ),
                    true,
                ))),
                true,
            ),
            Field::new("tenant_id", DataType::UInt32, false),
            Field::new("timestamp_epoch_secs", DataType::UInt64, false),
            Field::new("category", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(html_builder.finish()),
                Arc::new(id_builder.finish()),
                Arc::new(vec_builder.finish()),
                Arc::new(colbert_builder.finish()),
                Arc::new(tenant_builder.finish()),
                Arc::new(ts_builder.finish()),
                Arc::new(cat_builder.finish()),
            ],
        )
        .map_err(|e| invalid_data(e.to_string()))?;

        let parquet_bytes = write_parquet_bytes(&batch)?;
        let mut reader =
            ParquetBatchReader::from_chunk_reader(parquet_bytes, &IngestColumnHints::default(), 2)?;

        // raw_html (col 0) must be completely excluded from the projected schema
        assert!(reader
            .projected_schema()
            .field_with_name("raw_html")
            .is_err());
        assert_eq!(reader.projected_schema().fields().len(), 6);

        let map = reader.projected_map().clone();
        let filter_mapping = ScalarFilterColumnMapping::from_schema(reader.projected_schema());
        let mut filter_index = ScalarFilterIndex::new();
        let mut all_ids = Vec::new();
        let mut all_vecs = Vec::new();
        let mut all_colbert = Vec::new();

        for batch_res in &mut reader {
            let b = batch_res?;
            let ids = extract_batch_doc_ids(&b, map.id_batch_idx)?;
            let (dim, vecs) = extract_batch_vectors_f32(&b, map.vector_batch_idx, Some(3))?;
            assert_eq!(dim, 3);
            if let Some(cb_idx) = map.colbert_batch_idx {
                let (tok_dim, toks) = extract_batch_colbert_tokens_f32(&b, cb_idx, Some(2))?;
                assert_eq!(tok_dim, 2);
                all_colbert.extend(toks);
            }
            // Assign first 2 rows to minipage 0, next 2 rows to minipage 1
            let mps: Vec<u32> = (0..b.num_rows())
                .map(|r| if all_ids.len() + r < 2 { 0 } else { 1 })
                .collect();
            filter_index.insert_arrow_batch(&b, &filter_mapping, Some(&mps))?;
            all_ids.extend(ids);
            all_vecs.extend(vecs);
        }

        assert_eq!(all_ids.len(), 4);
        assert_eq!(all_ids[0], map_string_id_to_u64("doc-0"));
        assert_eq!(all_ids[3], map_string_id_to_u64("doc-3"));
        assert_eq!(all_vecs.len(), 12);
        assert_eq!(all_colbert.len(), 4);
        assert_eq!(all_colbert[2], vec![1.0, 0.5, 2.0, -0.5]);

        // Verify scalar filter index + Zone Map pruning
        let pred = FilterPredicate::And(vec![
            FilterPredicate::TenantEq(10),
            FilterPredicate::category_eq_str("finance"),
            FilterPredicate::TimestampRange {
                min: 1_700_000_000,
                max: 1_700_000_050,
            },
        ]);
        let hits = filter_index.evaluate(&pred);
        assert_eq!(hits.len(), 1);
        assert!(hits.contains(0));
        assert!(filter_index.matches_candidate(0, &pred));
        assert!(!filter_index.matches_candidate(1, &pred));

        let mut surviving = Vec::new();
        filter_index.prune_minipages(&[0, 1], &pred, &mut surviving);
        assert_eq!(surviving, vec![0]);
        Ok(())
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)] // Uses tokio runtime + object_store InMemory clock_gettime
    async fn test_async_parquet_batch_stream_projection_pushdown() -> OspreyResult<()> {
        let mut blob_builder = StringBuilder::new();
        let mut id_builder = UInt64Builder::new();
        let mut vec_builder = FixedSizeListBuilder::new(Float32Builder::new(), 2);

        for i in 0..3_u64 {
            blob_builder.append_value("huge-irrelevant-log-line");
            id_builder.append_value(100 + i);
            vec_builder.values().append_value(i as f32);
            vec_builder.values().append_value(1.0);
            vec_builder.append(true);
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("logs", DataType::Utf8, false),
            Field::new("id", DataType::UInt64, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
                false,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(blob_builder.finish()),
                Arc::new(id_builder.finish()),
                Arc::new(vec_builder.finish()),
            ],
        )
        .map_err(|e| invalid_data(e.to_string()))?;

        let parquet_bytes = write_parquet_bytes(&batch)?;
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = ObjectPath::from("lake/part-000.parquet");
        store
            .put(&path, parquet_bytes.into())
            .await
            .map_err(|e| invalid_data(e.to_string()))?;

        let mut stream =
            ParquetBatchStream::from_object_store(store, &path, &IngestColumnHints::default(), 2)
                .await?;

        assert!(stream.projected_schema().field_with_name("logs").is_err());
        let map = stream.projected_map().clone();

        let mut collected_ids = Vec::new();
        while let Some(batch_res) = stream.next().await {
            let b = batch_res?;
            collected_ids.extend(extract_batch_doc_ids(&b, map.id_batch_idx)?);
        }
        assert_eq!(collected_ids, vec![100, 101, 102]);
        Ok(())
    }

    #[test]
    fn test_non_finite_vector_and_zone_map_pruning() -> OspreyResult<()> {
        let mut id_builder = UInt64Builder::new();
        let mut vec_builder = FixedSizeListBuilder::new(Float32Builder::new(), 2);
        id_builder.append_value(1);
        vec_builder.values().append_value(1.0);
        vec_builder.values().append_value(f32::NAN);
        vec_builder.append(true);

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
                false,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_builder.finish()),
                Arc::new(vec_builder.finish()),
            ],
        )
        .map_err(|e| invalid_data(e.to_string()))?;

        let err = extract_batch_vectors_f32(&batch, 1, Some(2));
        assert!(matches!(
            err,
            Err(OspreyError::NonFiniteEmbedding { index: 1 })
        ));

        let mut idx = ScalarFilterIndex::new();
        idx.insert_row(Some(10), RowFilterAttributes::with_category(1, 100, "news"))?;
        idx.insert_row(
            Some(10),
            RowFilterAttributes::with_category(2, 150, "sports"),
        )?;
        idx.insert_row(Some(11), RowFilterAttributes::with_category(3, 500, "tech"))?;

        let pred_ts = FilterPredicate::TimestampRange { min: 400, max: 600 };
        assert!(!idx.minipage_might_match(10, &pred_ts));
        assert!(idx.minipage_might_match(11, &pred_ts));

        let bm = idx.evaluate(&FilterPredicate::And(vec![
            FilterPredicate::TenantEq(2),
            FilterPredicate::category_eq_str("sports"),
        ]));
        assert_eq!(bm.iter().collect::<Vec<_>>(), vec![1]);

        let empty_conflict = idx.evaluate(&FilterPredicate::And(vec![
            FilterPredicate::TenantEq(1),
            FilterPredicate::TenantEq(2),
        ]));
        assert!(empty_conflict.is_empty());
        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Invokes zstd-sys C FFI and rayon threadpool
    fn test_out_of_core_reservoir_sampling_and_two_pass_osv_ingest() -> OspreyResult<()> {
        let total_rows = 200_usize;
        let dim = 8_usize;
        let max_sample = 32_usize;
        let chunk_rows = 64_usize;

        let mut id_builder = UInt64Builder::new();
        let mut vec_builder = FixedSizeListBuilder::new(Float32Builder::new(), 8);
        let mut tenant_builder = UInt32Builder::new();
        let mut ts_builder = UInt64Builder::new();
        let mut cat_builder = StringBuilder::new();
        let mut blob_builder = StringBuilder::new();

        for i in 0..total_rows {
            id_builder.append_value(1000 + i as u64);
            for d in 0..dim {
                let val = ((i * 7 + d) % 19) as f32 * 0.1 + 0.05;
                vec_builder.values().append_value(val);
            }
            vec_builder.append(true);
            tenant_builder.append_value(u32::try_from(i % 4).unwrap_or(0));
            ts_builder.append_value(1_700_000_000 + i as u64);
            cat_builder.append_value(if i % 2 == 0 { "alpha" } else { "beta" });
            blob_builder.append_value("ignored-heavy-column");
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 8),
                false,
            ),
            Field::new("tenant_id", DataType::UInt32, false),
            Field::new("timestamp_epoch_secs", DataType::UInt64, false),
            Field::new("category", DataType::Utf8, false),
            Field::new("raw_html", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_builder.finish()),
                Arc::new(vec_builder.finish()),
                Arc::new(tenant_builder.finish()),
                Arc::new(ts_builder.finish()),
                Arc::new(cat_builder.finish()),
                Arc::new(blob_builder.finish()),
            ],
        )
        .map_err(|e| invalid_data(e.to_string()))?;

        let parquet_bytes = write_parquet_bytes(&batch)?;

        let pipeline = OutOfCoreIngestPipeline::new(OutOfCoreIngestConfig {
            max_training_sample_vectors: max_sample,
            batch_size_rows: chunk_rows,
            num_clusters: 4,
            ..Default::default()
        });

        let mut slab_blobs: Vec<Vec<u8>> = Vec::new();
        let report = pipeline.ingest_chunked(
            || Ok(parquet_bytes.clone()),
            |_chunk_idx, writer, records, orig_dim| {
                let (bytes, rep) = writer.write_slab_to_bytes(records, orig_dim, None)?;
                slab_blobs.push(bytes);
                Ok(rep)
            },
        )?;

        assert_eq!(report.total_vectors_seen, total_rows as u64);
        assert_eq!(report.total_vectors_sampled, max_sample);
        assert_eq!(report.total_vectors_indexed, total_rows as u64);
        assert_eq!(report.total_batches, 4); // 200 / 64 -> 64 + 64 + 64 + 8
        assert_eq!(slab_blobs.len(), 4);
        assert_eq!(report.codebook.num_clusters, 4);
        assert_eq!(report.filter_index.len(), total_rows);

        // Verify scalar filter index + Zone Maps built in Pass 2
        let pred = FilterPredicate::And(vec![
            FilterPredicate::TenantEq(2),
            FilterPredicate::category_eq_str("alpha"),
        ]);
        let matching = report.filter_index.evaluate(&pred);
        assert_eq!(matching.len(), 50); // i % 4 == 2 implies i % 2 == 0 ("alpha") -> 200/4 = 50

        // Verify generated .osv slabs are valid and readable
        let mut aligned_slab = osprey_core::AlignedBuffer::<u8>::new(slab_blobs[0].len())?;
        aligned_slab.as_mut_slice().copy_from_slice(&slab_blobs[0]);
        let slab0 = crate::osv_slab::OsvSlabView::from_bytes(aligned_slab.as_slice())?;
        assert_eq!(slab0.header.total_vectors, chunk_rows as u64);
        Ok(())
    }
}
