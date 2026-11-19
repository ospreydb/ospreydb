//! Apache Arrow Flight gRPC server (`grpc://0.0.0.0:8815`) for zero-copy
//! batch ingestion (`DoPut`), hybrid vector search & `LUT-Rank` reranking (`DoExchange`),
//! and cluster segment metadata streaming (`DoGet` / `ListFlights`).

use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use arrow::array::{
    Array, BinaryArray, BinaryBuilder, FixedSizeListArray, Float32Array, Float32Builder,
    Float64Array, LargeListArray, ListArray, StringArray, StringBuilder, UInt32Array, UInt64Array,
    UInt64Builder,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow_flight::decode::FlightRecordBatchStream;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::error::FlightError;
use arrow_flight::flight_service_server::{FlightService, FlightServiceServer};
use arrow_flight::{
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightEndpoint, FlightInfo,
    HandshakeRequest, HandshakeResponse, PollInfo, PutResult, SchemaResult, Ticket,
};
use bytes::Bytes;
use futures::stream::{self, BoxStream, StreamExt, TryStreamExt};
use osprey_core::RawVectorRecord;
use serde_json::json;
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status, Streaming};

use crate::delta_buffer::HybridSearchEngine;
use crate::error::ProResult;
use crate::rerank_service::LutRankReranker;

/// Default Apache Arrow Flight gRPC bind address (`grpc://0.0.0.0:8815`).
pub const DEFAULT_ARROW_FLIGHT_ADDR: &str = "0.0.0.0:8815";

type TonicStream<T> = Pin<Box<dyn futures::Stream<Item = Result<T, Status>> + Send + 'static>>;

/// `osprey` Apache Arrow Flight gRPC service backed by [`HybridSearchEngine`] and [`LutRankReranker`].
#[derive(Clone, Debug)]
pub struct OspreyFlightService {
    engine: Arc<HybridSearchEngine>,
    reranker: Arc<LutRankReranker>,
    total_vectors: Arc<AtomicU64>,
    default_nprobe: usize,
    default_top_k: usize,
    read_only: bool,
}

impl OspreyFlightService {
    /// Creates a new [`OspreyFlightService`].
    #[must_use]
    pub fn new(engine: Arc<HybridSearchEngine>, reranker: Arc<LutRankReranker>) -> Self {
        Self::with_vector_counter(engine, reranker, Arc::new(AtomicU64::new(0)))
    }

    /// Creates a new [`OspreyFlightService`] sharing a vector counter with HTTP endpoints.
    #[must_use]
    pub fn with_vector_counter(
        engine: Arc<HybridSearchEngine>,
        reranker: Arc<LutRankReranker>,
        total_vectors: Arc<AtomicU64>,
    ) -> Self {
        Self {
            engine,
            reranker,
            total_vectors,
            default_nprobe: 16,
            default_top_k: 10,
            read_only: false,
        }
    }

    /// Marks this Arrow Flight service read-only (for `--role reader` nodes), rejecting `DoPut`.
    #[must_use]
    pub const fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    fn search_result_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("query_id", DataType::UInt64, false),
            Field::new("doc_id", DataType::UInt64, false),
            Field::new("score", DataType::Float32, false),
            Field::new("payload_json", DataType::Binary, false),
        ]))
    }

    fn rerank_result_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("query_id", DataType::UInt64, false),
            Field::new("doc_index", DataType::UInt64, false),
            Field::new("relevance_score", DataType::Float32, false),
            Field::new("document", DataType::Utf8, false),
        ]))
    }

    fn segments_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("segment_id", DataType::UInt64, false),
            Field::new("num_vectors", DataType::UInt64, false),
            Field::new("tombstone_count", DataType::UInt64, false),
            Field::new("osi_key", DataType::Utf8, false),
            Field::new("osv_key", DataType::Utf8, false),
        ]))
    }

    #[allow(clippy::result_large_err)]
    async fn build_segments_batch(&self) -> Result<RecordBatch, Status> {
        let snap = self
            .engine
            .manifest_store()
            .load()
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let mut seg_ids = UInt64Builder::new();
        let mut num_vecs = UInt64Builder::new();
        let mut tomb_counts = UInt64Builder::new();
        let mut osi_keys = StringBuilder::new();
        let mut osv_keys = StringBuilder::new();

        if let Some(s) = snap {
            for seg in &s.manifest.segments {
                seg_ids.append_value(seg.segment_id);
                num_vecs.append_value(seg.num_vectors);
                tomb_counts.append_value(seg.tombstone_count);
                osi_keys.append_value(&seg.osi_key);
                osv_keys.append_value(&seg.osv_key);
            }
        }

        RecordBatch::try_new(
            Self::segments_schema(),
            vec![
                Arc::new(seg_ids.finish()),
                Arc::new(num_vecs.finish()),
                Arc::new(tomb_counts.finish()),
                Arc::new(osi_keys.finish()),
                Arc::new(osv_keys.finish()),
            ],
        )
        .map_err(|e| Status::internal(e.to_string()))
    }

    #[allow(clippy::result_large_err)]
    async fn process_exchange_batch(&self, batch: &RecordBatch) -> Result<RecordBatch, Status> {
        if batch.column_by_name("vector").is_some() {
            return self.process_vector_search_batch(batch).await;
        }
        if batch.column_by_name("query").is_some() && batch.column_by_name("document").is_some() {
            return self.process_rerank_batch(batch);
        }
        Err(Status::invalid_argument(
            "DoExchange RecordBatch must contain either 'vector' (search) or ('query', 'document') (LUT-Rank)",
        ))
    }

    #[allow(clippy::result_large_err)]
    async fn process_vector_search_batch(
        &self,
        batch: &RecordBatch,
    ) -> Result<RecordBatch, Status> {
        let num_rows = batch.num_rows();
        let vector_col = batch
            .column_by_name("vector")
            .ok_or_else(|| Status::invalid_argument("missing 'vector' column"))?;
        let query_ids = batch
            .column_by_name("query_id")
            .and_then(|c| c.as_any().downcast_ref::<UInt64Array>());
        let top_k_col = batch
            .column_by_name("top_k")
            .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
        let nprobe_col = batch
            .column_by_name("nprobe")
            .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());

        let extracted = extract_vector_slices(vector_col.as_ref(), num_rows)?;

        let mut qid_builder = UInt64Builder::new();
        let mut doc_builder = UInt64Builder::new();
        let mut score_builder = Float32Builder::new();
        let mut payload_builder = BinaryBuilder::new();

        for row in 0..num_rows {
            let query_id = query_ids.map_or(row as u64, |arr| arr.value(row));
            let top_k = top_k_col.map_or(self.default_top_k, |arr| {
                usize::try_from(arr.value(row)).unwrap_or(10)
            });
            let nprobe = nprobe_col.map_or(self.default_nprobe, |arr| {
                usize::try_from(arr.value(row)).unwrap_or(16)
            });
            let top_m = top_k.saturating_mul(8).max(64);
            let q_slice = extracted.slice_at(row);

            let hits = self
                .engine
                .search_hybrid_async(q_slice, nprobe, top_m, top_k)
                .await
                .map_err(|e| Status::invalid_argument(e.to_string()))?;

            for hit in hits {
                qid_builder.append_value(query_id);
                doc_builder.append_value(hit.doc_id);
                score_builder.append_value(hit.exact_score);
                payload_builder.append_value(&hit.payload_json);
            }
        }

        RecordBatch::try_new(
            Self::search_result_schema(),
            vec![
                Arc::new(qid_builder.finish()),
                Arc::new(doc_builder.finish()),
                Arc::new(score_builder.finish()),
                Arc::new(payload_builder.finish()),
            ],
        )
        .map_err(|e| Status::internal(e.to_string()))
    }

    #[allow(clippy::result_large_err)]
    fn process_rerank_batch(&self, batch: &RecordBatch) -> Result<RecordBatch, Status> {
        let num_rows = batch.num_rows();
        let query_col = batch
            .column_by_name("query")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| Status::invalid_argument("missing Utf8 'query' column"))?;
        let doc_col = batch
            .column_by_name("document")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| Status::invalid_argument("missing Utf8 'document' column"))?;
        let query_ids = batch
            .column_by_name("query_id")
            .and_then(|c| c.as_any().downcast_ref::<UInt64Array>());
        let top_n_col = batch
            .column_by_name("top_n")
            .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());

        let mut qid_builder = UInt64Builder::new();
        let mut idx_builder = UInt64Builder::new();
        let mut score_builder = Float32Builder::new();
        let mut doc_builder = StringBuilder::new();

        if num_rows > 0 {
            let query_text = query_col.value(0);
            let qid = query_ids.map_or(0_u64, |a| a.value(0));
            let top_n = top_n_col.map(|a| usize::try_from(a.value(0)).unwrap_or(num_rows));
            let docs: Vec<&str> = (0..num_rows).map(|r| doc_col.value(r)).collect();

            let hits = self
                .reranker
                .rerank_texts(query_text, &docs, top_n, Some(self.engine.metrics()))
                .map_err(|e| Status::internal(e.to_string()))?;

            for hit in hits {
                qid_builder.append_value(qid);
                idx_builder.append_value(u64::try_from(hit.index).unwrap_or(u64::MAX));
                score_builder.append_value(hit.relevance_score);
                doc_builder.append_value(docs[hit.index]);
            }
        }

        RecordBatch::try_new(
            Self::rerank_result_schema(),
            vec![
                Arc::new(qid_builder.finish()),
                Arc::new(idx_builder.finish()),
                Arc::new(score_builder.finish()),
                Arc::new(doc_builder.finish()),
            ],
        )
        .map_err(|e| Status::internal(e.to_string()))
    }
}

enum ExtractedVectorColumn<'a> {
    BorrowedContiguous {
        values: &'a [f32],
        dim: usize,
    },
    BorrowedList {
        offsets: &'a [i32],
        values: &'a [f32],
    },
    BorrowedLargeList {
        offsets: &'a [i64],
        values: &'a [f32],
    },
    OwnedContiguous {
        values: Vec<f32>,
        dim: usize,
    },
    OwnedRagged {
        ranges: Vec<(usize, usize)>,
        values: Vec<f32>,
    },
}

impl<'a> ExtractedVectorColumn<'a> {
    fn slice_at(&self, row: usize) -> &[f32] {
        match self {
            Self::BorrowedContiguous { values, dim } => &values[row * *dim..(row + 1) * *dim],
            Self::BorrowedList { offsets, values } => {
                let start = usize::try_from(offsets[row]).unwrap_or(0);
                let end = usize::try_from(offsets[row + 1]).unwrap_or(start);
                &values[start..end]
            }
            Self::BorrowedLargeList { offsets, values } => {
                let start = usize::try_from(offsets[row]).unwrap_or(0);
                let end = usize::try_from(offsets[row + 1]).unwrap_or(start);
                &values[start..end]
            }
            Self::OwnedContiguous { values, dim } => &values[row * *dim..(row + 1) * *dim],
            Self::OwnedRagged { ranges, values } => {
                let (start, end) = ranges[row];
                &values[start..end]
            }
        }
    }
}

#[allow(clippy::cast_possible_truncation)]
fn f64_slice_to_f32(vals: &[f64]) -> Vec<f32> {
    vals.iter().map(|&v| v as f32).collect()
}

#[allow(clippy::result_large_err)]
fn extract_vector_slices<'a>(
    col: &'a dyn Array,
    num_rows: usize,
) -> Result<ExtractedVectorColumn<'a>, Status> {
    if col.null_count() > 0 {
        return Err(Status::invalid_argument(
            "'vector' column must not contain null entries",
        ));
    }

    if let Some(fsl) = col.as_any().downcast_ref::<FixedSizeListArray>() {
        let dim = usize::try_from(fsl.value_length()).unwrap_or(0);
        if let Some(f32_arr) = fsl.values().as_any().downcast_ref::<Float32Array>() {
            let values = f32_arr.values();
            if values.len() < num_rows * dim {
                return Err(Status::invalid_argument(
                    "truncated FixedSizeList<Float32> vector buffer",
                ));
            }
            return Ok(ExtractedVectorColumn::BorrowedContiguous { values, dim });
        }
        if let Some(f64_arr) = fsl.values().as_any().downcast_ref::<Float64Array>() {
            let raw = f64_arr.values();
            if raw.len() < num_rows * dim {
                return Err(Status::invalid_argument(
                    "truncated FixedSizeList<Float64> vector buffer",
                ));
            }
            return Ok(ExtractedVectorColumn::OwnedContiguous {
                values: f64_slice_to_f32(&raw[..num_rows * dim]),
                dim,
            });
        }
        return Err(Status::invalid_argument(
            "'vector' FixedSizeList must contain Float32 or Float64",
        ));
    }

    if let Some(list) = col.as_any().downcast_ref::<ListArray>() {
        let offsets = list.value_offsets();
        if offsets.len() < num_rows + 1 {
            return Err(Status::invalid_argument("truncated List offset buffer"));
        }
        if let Some(f32_arr) = list.values().as_any().downcast_ref::<Float32Array>() {
            return Ok(ExtractedVectorColumn::BorrowedList {
                offsets,
                values: f32_arr.values(),
            });
        }
        if let Some(f64_arr) = list.values().as_any().downcast_ref::<Float64Array>() {
            let ranges = (0..num_rows)
                .map(|r| {
                    let s = usize::try_from(offsets[r]).unwrap_or(0);
                    let e = usize::try_from(offsets[r + 1]).unwrap_or(s);
                    (s, e)
                })
                .collect();
            return Ok(ExtractedVectorColumn::OwnedRagged {
                ranges,
                values: f64_slice_to_f32(f64_arr.values()),
            });
        }
        return Err(Status::invalid_argument(
            "'vector' List must contain Float32 or Float64",
        ));
    }

    if let Some(list) = col.as_any().downcast_ref::<LargeListArray>() {
        let offsets = list.value_offsets();
        if offsets.len() < num_rows + 1 {
            return Err(Status::invalid_argument(
                "truncated LargeList offset buffer",
            ));
        }
        if let Some(f32_arr) = list.values().as_any().downcast_ref::<Float32Array>() {
            return Ok(ExtractedVectorColumn::BorrowedLargeList {
                offsets,
                values: f32_arr.values(),
            });
        }
        if let Some(f64_arr) = list.values().as_any().downcast_ref::<Float64Array>() {
            let ranges = (0..num_rows)
                .map(|r| {
                    let s = usize::try_from(offsets[r]).unwrap_or(0);
                    let e = usize::try_from(offsets[r + 1]).unwrap_or(s);
                    (s, e)
                })
                .collect();
            return Ok(ExtractedVectorColumn::OwnedRagged {
                ranges,
                values: f64_slice_to_f32(f64_arr.values()),
            });
        }
        return Err(Status::invalid_argument(
            "'vector' LargeList must contain Float32 or Float64",
        ));
    }

    Err(Status::invalid_argument(
        "'vector' column must be FixedSizeList<Float32/Float64> or List<Float32/Float64>",
    ))
}

#[tonic::async_trait]
impl FlightService for OspreyFlightService {
    type HandshakeStream = TonicStream<HandshakeResponse>;
    type ListFlightsStream = TonicStream<FlightInfo>;
    type DoGetStream = TonicStream<FlightData>;
    type DoPutStream = TonicStream<PutResult>;
    type DoExchangeStream = TonicStream<FlightData>;
    type DoActionStream = TonicStream<arrow_flight::Result>;
    type ListActionsStream = TonicStream<ActionType>;

    async fn handshake(
        &self,
        _request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        let s = stream::empty();
        Ok(Response::new(Box::pin(s)))
    }

    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        let schema = Self::segments_schema();
        let descriptor = FlightDescriptor::new_path(vec!["segments".to_owned()]);
        let info = FlightInfo::new()
            .try_with_schema(schema.as_ref())
            .map_err(|e| Status::internal(e.to_string()))?
            .with_descriptor(descriptor.clone())
            .with_endpoint(FlightEndpoint::new().with_ticket(Ticket::new("segments")));
        let s = stream::iter(vec![Ok(info)]);
        Ok(Response::new(Box::pin(s)))
    }

    async fn get_flight_info(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let descriptor = request.into_inner();
        let ticket_name = descriptor
            .path
            .first()
            .cloned()
            .unwrap_or_else(|| "segments".to_owned());
        let info = FlightInfo::new()
            .try_with_schema(Self::segments_schema().as_ref())
            .map_err(|e| Status::internal(e.to_string()))?
            .with_descriptor(descriptor)
            .with_endpoint(FlightEndpoint::new().with_ticket(Ticket::new(ticket_name)));
        Ok(Response::new(info))
    }

    async fn poll_flight_info(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<PollInfo>, Status> {
        let info = self.get_flight_info(request).await?.into_inner();
        Ok(Response::new(PollInfo::new().with_info(info)))
    }

    async fn get_schema(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        Err(Status::unimplemented(
            "use get_flight_info for schema discovery",
        ))
    }

    async fn do_get(
        &self,
        _request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let batch = self.build_segments_batch().await?;
        let schema = batch.schema();
        let batch_stream = stream::iter(vec![Ok(batch)]);
        let flight_stream = FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .build(batch_stream)
            .map_err(|e| Status::internal(e.to_string()));
        Ok(Response::new(Box::pin(flight_stream)))
    }

    async fn do_put(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        if self.read_only {
            return Err(Status::failed_precondition(
                "node is running in read-only (--role reader) mode; DoPut writes are not permitted",
            ));
        }
        let mut batch_stream = FlightRecordBatchStream::new_from_flight_data(
            request.into_inner().map_err(FlightError::Tonic),
        );

        let mut total_ingested = 0_usize;
        let mut last_seq = 0_u64;

        while let Some(batch_res) = batch_stream.next().await {
            let batch = batch_res.map_err(|e| Status::invalid_argument(e.to_string()))?;
            let num_rows = batch.num_rows();
            if num_rows == 0 {
                continue;
            }

            let doc_ids = batch
                .column_by_name("doc_id")
                .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
                .ok_or_else(|| Status::invalid_argument("DoPut requires UInt64 'doc_id' column"))?;
            let vector_col = batch
                .column_by_name("vector")
                .ok_or_else(|| Status::invalid_argument("DoPut requires 'vector' column"))?;
            let extracted = extract_vector_slices(vector_col.as_ref(), num_rows)?;

            let payload_bin = batch
                .column_by_name("payload_json")
                .and_then(|c| c.as_any().downcast_ref::<BinaryArray>());
            let payload_str = batch
                .column_by_name("payload_json")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());

            let mut raw_records = Vec::with_capacity(num_rows);
            for row in 0..num_rows {
                let payload_slice: &[u8] = if let Some(bin) = payload_bin {
                    bin.value(row)
                } else if let Some(s) = payload_str {
                    s.value(row).as_bytes()
                } else {
                    b"{}"
                };
                raw_records.push(RawVectorRecord {
                    doc_id: doc_ids.value(row),
                    vector: extracted.slice_at(row),
                    payload_json: payload_slice,
                });
            }

            let batch_len = u64::try_from(raw_records.len()).unwrap_or(0);

            last_seq = self
                .engine
                .delta_buffer()
                .ingest_batch(&raw_records)
                .map_err(|e| Status::invalid_argument(e.to_string()))?;
            self.engine.metrics().observe_wal_ingest(raw_records.len());
            self.total_vectors.fetch_add(batch_len, Ordering::Release);
            total_ingested = total_ingested.saturating_add(raw_records.len());
        }

        let meta_bytes = serde_json::to_vec(&json!({
            "ingested": total_ingested,
            "last_seq_no": last_seq
        }))
        .map_err(|e| Status::internal(e.to_string()))?;

        let s = stream::iter(vec![Ok(PutResult {
            app_metadata: Bytes::from(meta_bytes),
        })]);
        Ok(Response::new(Box::pin(s)))
    }

    async fn do_exchange(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        let mut batch_stream = FlightRecordBatchStream::new_from_flight_data(
            request.into_inner().map_err(FlightError::Tonic),
        );

        let mut output_batches = Vec::new();
        while let Some(batch_res) = batch_stream.next().await {
            let batch = batch_res.map_err(|e| Status::invalid_argument(e.to_string()))?;
            if batch.num_rows() == 0 {
                continue;
            }
            let out = self.process_exchange_batch(&batch).await?;
            output_batches.push(Ok(out));
        }

        let encoded_stream: BoxStream<'static, Result<FlightData, Status>> =
            if output_batches.is_empty() {
                Box::pin(stream::empty())
            } else {
                Box::pin(
                    FlightDataEncoderBuilder::new()
                        .build(stream::iter(output_batches))
                        .map_err(|e| Status::internal(e.to_string())),
                )
            };

        Ok(Response::new(encoded_stream))
    }

    async fn do_action(
        &self,
        _request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        let s = stream::empty();
        Ok(Response::new(Box::pin(s)))
    }

    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        let s = stream::empty();
        Ok(Response::new(Box::pin(s)))
    }
}

/// Binds and serves the Apache Arrow Flight gRPC server on `addr` until `shutdown` completes.
pub async fn serve_arrow_flight<F>(
    addr: SocketAddr,
    service: OspreyFlightService,
    shutdown: F,
) -> ProResult<()>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    let listener = TcpListener::bind(addr).await?;
    let incoming = TcpListenerStream::new(listener);
    tonic::transport::Server::builder()
        .add_service(FlightServiceServer::new(service))
        .serve_with_incoming_shutdown(incoming, shutdown)
        .await
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    Ok(())
}
