//! Apache Arrow Flight gRPC server (`grpc://0.0.0.0:8815`) for zero-copy
//! batch ingestion (`DoPut`), hybrid vector search & `LUT-Rank` reranking (`DoExchange`),
//! and cluster segment metadata streaming (`DoGet` / `ListFlights`).

use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;

use arrow::array::{
    Array, BinaryArray, BinaryBuilder, FixedSizeListArray, Float32Array, Float32Builder, ListArray,
    StringArray, StringBuilder, UInt32Array, UInt64Array, UInt64Builder,
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
use crate::error::{ProEngineError, ProResult};
use crate::license::VectorQuotaState;
use crate::rerank_service::LutRankReranker;

/// Default Apache Arrow Flight gRPC bind address (`grpc://0.0.0.0:8815`).
pub const DEFAULT_ARROW_FLIGHT_ADDR: &str = "0.0.0.0:8815";

type TonicStream<T> = Pin<Box<dyn futures::Stream<Item = Result<T, Status>> + Send + 'static>>;

/// `osprey-pro` Apache Arrow Flight gRPC service backed by [`HybridSearchEngine`] and [`LutRankReranker`].
#[derive(Clone, Debug)]
pub struct OspreyFlightService {
    engine: Arc<HybridSearchEngine>,
    reranker: Arc<LutRankReranker>,
    quota: VectorQuotaState,
    default_nprobe: usize,
    default_top_k: usize,
}

impl OspreyFlightService {
    /// Creates a new [`OspreyFlightService`].
    #[must_use]
    pub fn new(engine: Arc<HybridSearchEngine>, reranker: Arc<LutRankReranker>) -> Self {
        Self::with_quota(engine, reranker, VectorQuotaState::new())
    }

    /// Creates a new [`OspreyFlightService`] sharing a [`VectorQuotaState`] with HTTP endpoints.
    #[must_use]
    pub fn with_quota(
        engine: Arc<HybridSearchEngine>,
        reranker: Arc<LutRankReranker>,
        quota: VectorQuotaState,
    ) -> Self {
        Self {
            engine,
            reranker,
            quota,
            default_nprobe: 16,
            default_top_k: 10,
        }
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

enum BorrowedVectorColumn<'a> {
    Contiguous {
        values: &'a [f32],
        dim: usize,
    },
    Ragged {
        list: &'a ListArray,
        values: &'a [f32],
    },
}

impl<'a> BorrowedVectorColumn<'a> {
    fn slice_at(&self, row: usize) -> &'a [f32] {
        match self {
            Self::Contiguous { values, dim } => &values[row * *dim..(row + 1) * *dim],
            Self::Ragged { list, values } => {
                let offsets = list.value_offsets();
                let start = usize::try_from(offsets[row]).unwrap_or(0);
                let end = usize::try_from(offsets[row + 1]).unwrap_or(start);
                &values[start..end]
            }
        }
    }
}

#[allow(clippy::result_large_err)]
fn extract_vector_slices<'a>(
    col: &'a dyn Array,
    num_rows: usize,
) -> Result<BorrowedVectorColumn<'a>, Status> {
    if let Some(fsl) = col.as_any().downcast_ref::<FixedSizeListArray>() {
        let dim = usize::try_from(fsl.value_length()).unwrap_or(0);
        let f32_arr = fsl
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| {
                Status::invalid_argument("'vector' FixedSizeList must contain Float32")
            })?;
        let values = f32_arr.values();
        if values.len() < num_rows * dim {
            return Err(Status::invalid_argument(
                "truncated FixedSizeList vector buffer",
            ));
        }
        return Ok(BorrowedVectorColumn::Contiguous { values, dim });
    }
    if let Some(list) = col.as_any().downcast_ref::<ListArray>() {
        let f32_arr = list
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| Status::invalid_argument("'vector' List must contain Float32"))?;
        return Ok(BorrowedVectorColumn::Ragged {
            list,
            values: f32_arr.values(),
        });
    }
    Err(Status::invalid_argument(
        "'vector' column must be FixedSizeList<Float32> or List<Float32>",
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
            if let Err(limit_err) = self.quota.check_vector_limit(batch_len) {
                return Err(match limit_err {
                    ProEngineError::CommunityTierLimitReached { .. } => {
                        Status::resource_exhausted(limit_err.to_string())
                    }
                    other => Status::internal(other.to_string()),
                });
            }

            last_seq = self
                .engine
                .delta_buffer()
                .ingest_batch(&raw_records)
                .map_err(|e| Status::invalid_argument(e.to_string()))?;
            self.engine.metrics().observe_wal_ingest(raw_records.len());
            self.quota.increment_vector_count(batch_len);
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
