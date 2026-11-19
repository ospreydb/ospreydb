//! End-to-end Data Lake integration test:
//! - Generates a 2,000-document Apache Parquet table with `Utf8` doc IDs (`xxHash64`),
//!   multi-tenant attributes (`tenant_id`, `created_at`), anisotropic 128-D `f32` embeddings,
//!   16x128-D ColBERT token tensors, heavy `raw_html_blob` junk column, and JSON payloads;
//! - Auto-detects schema (`inspect_parquet_schema`), ignores `raw_html_blob` via
//!   `ProjectionMask::roots`, runs two-pass out-of-core reservoir sampling + `1-Bit Binary K-Means`
//!   + TSP locality ordering, and seals `.osi`, `.osv`, and `.osr` artifacts on disk + `ObjectStore`;
//! - Executes filtered hybrid search (`tenant_id == 20` + `created_at` range):
//!   Stage 1 (`.osi` 1-bit `RaBitQ` + `RoaringBitmap` filter -> Top-100) ->
//!   Stage 2 (DP range coalescing + `ObjectStore::get_ranges` `.osv` + exact `f16` refine -> Top-25) ->
//!   Stage 3 (`.osr` `LUT-Rank` `vpshufb`/`vtbl` ColBERT MaxSim -> Top-10);
//! - Validates 100% filter compliance, Top-1 retrieval of planted target, and `Recall@10 > 96.5%`.

use std::collections::HashSet;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::{
    FixedSizeListBuilder, Float32Builder, ListBuilder, StringBuilder, UInt32Builder, UInt64Builder,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use half::f16;
use object_store::memory::InMemory;
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;
use osprey_core::math::{padded_dimension, SplitMix64};
use osprey_core::{
    BinaryKMeansResult, OsiIndex, OspreyError, OspreyResult, OsrDocumentTokens, OsrStore,
    QuantizedQueryLut, VectorLocation, DEFAULT_LUTRANK_GAMMA, LUTRANK_TOKEN_DIM,
};
use osprey_simd::SimdEngine;
use osprey_storage::{
    extract_batch_colbert_tokens_f32, extract_batch_doc_ids, extract_batch_vectors_f32,
    inspect_parquet_schema, map_string_id_to_u64, FilterPredicate, IngestColumnHints,
    MinipageLruCache, OsvSlabReader, OsvVectorRecord, OsvWriter, OsvWriterConfig,
    OutOfCoreIngestConfig, OutOfCoreIngestPipeline, ParquetBatchReader, RangePlannerConfig,
    RefineStage, RemoteStoreConfig, ScalarFilterColumnMapping, ScalarFilterIndex,
};
use parquet::arrow::ArrowWriter;
use roaring::RoaringBitmap;

const NUM_DOCS: usize = 2_000;
const DIM: usize = 128;
const TOKENS_PER_DOC: usize = 16;
const NUM_CLUSTERS: usize = 16;
const FWHT_SEED: u64 = 0x4F53_5052_4559_1038;

fn invalid_data(msg: impl Into<String>) -> OspreyError {
    OspreyError::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        msg.into(),
    ))
}

struct TempWorkspaceDir {
    path: PathBuf,
}

impl TempWorkspaceDir {
    fn new(prefix: &str) -> OspreyResult<Self> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| invalid_data(e.to_string()))?
            .as_nanos();
        let path = std::env::temp_dir().join(format!("{prefix}_{}_{}", std::process::id(), nanos));
        std::fs::create_dir_all(&path)?;
        Ok(Self { path })
    }
}

impl Drop for TempWorkspaceDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn l2_normalize(v: &mut [f32]) {
    let norm_sq: f32 = v.iter().map(|&x| x * x).sum();
    let inv = if norm_sq > 1e-12 {
        1.0 / norm_sq.sqrt()
    } else {
        0.0
    };
    for x in v.iter_mut() {
        *x *= inv;
    }
}

fn pseudo_unit_scalar(rng: &mut SplitMix64) -> f32 {
    let u = ((rng.next_u64() >> 40) as f32) / ((1_u64 << 24) as f32);
    u * 2.0 - 1.0
}

#[derive(Clone)]
struct GroundTruthDoc {
    doc_id: u64,
    tenant_id: u32,
    created_at: u64,
    embedding: Vec<f32>,
    tokens: Vec<f32>,
}

struct GeneratedLakeDataset {
    gt_docs: Vec<GroundTruthDoc>,
    planted_idx: usize,
    planted_query_vec: Vec<f32>,
    planted_query_tokens: Vec<f32>,
}

fn generate_corporate_parquet(parquet_path: &PathBuf) -> OspreyResult<GeneratedLakeDataset> {
    let mut rng = SplitMix64::new(0xDEAD_BEEF_CAFE_1038);

    // Build 16 anisotropic cluster centers in R^128.
    let mut centers = vec![0.0_f32; NUM_CLUSTERS * DIM];
    for c in 0..NUM_CLUSTERS {
        let row = &mut centers[c * DIM..(c + 1) * DIM];
        for (d, val) in row.iter_mut().enumerate() {
            // Anisotropic coordinate scaling across dimensions
            let aniso = if d < 16 {
                3.0
            } else if d < 64 {
                1.2
            } else {
                0.35
            };
            *val = pseudo_unit_scalar(&mut rng) * aniso;
        }
        l2_normalize(row);
    }

    let mut id_builder = StringBuilder::new();
    let mut tenant_builder = UInt32Builder::new();
    let mut ts_builder = UInt64Builder::new();
    let mut vec_builder = FixedSizeListBuilder::new(Float32Builder::new(), 128);
    let mut colbert_builder =
        ListBuilder::new(FixedSizeListBuilder::new(Float32Builder::new(), 128));
    let mut html_builder = StringBuilder::new();
    let mut snippet_builder = StringBuilder::new();

    let heavy_html_template = "<html><body><article>".to_string()
        + &"lorem ipsum dolor sit amet consectetur adipiscing elit ".repeat(30)
        + "</article></body></html>";

    let mut gt_docs = Vec::with_capacity(NUM_DOCS);
    let planted_idx = 777_usize;
    let mut planted_query_vec = vec![0.0_f32; DIM];
    let mut planted_query_tokens = vec![0.0_f32; 4 * DIM];

    for i in 0..NUM_DOCS {
        let doc_str = format!("doc_{i:04}");
        let doc_id = map_string_id_to_u64(&doc_str);
        let tenant_id = match i % 3 {
            0 => 10_u32,
            1 => 20_u32,
            _ => 30_u32,
        };
        // Ensure planted doc 777 belongs to tenant 20 (777 % 3 == 0 -> override to keep deterministic)
        let tenant_id = if i == planted_idx { 20 } else { tenant_id };
        let created_at = if i == planted_idx {
            1_700_008_500_u64
        } else {
            1_700_000_000_u64 + u64::try_from((i * 17) % 20_000).unwrap_or(0)
        };

        let cluster = i % NUM_CLUSTERS;
        let center = &centers[cluster * DIM..(cluster + 1) * DIM];
        let mut emb = vec![0.0_f32; DIM];
        for (d, val) in emb.iter_mut().enumerate() {
            *val = center[d] + pseudo_unit_scalar(&mut rng) * 0.14;
        }
        l2_normalize(&mut emb);

        if i == planted_idx {
            planted_query_vec.copy_from_slice(&emb);
        }

        // Generate 16 ColBERT tokens aligned with `emb` (small token-level variations).
        let mut doc_tokens = Vec::with_capacity(TOKENS_PER_DOC * DIM);
        for t in 0..TOKENS_PER_DOC {
            let mut tok = vec![0.0_f32; DIM];
            for (d, val) in tok.iter_mut().enumerate() {
                let noise = if i == planted_idx && t < 4 {
                    pseudo_unit_scalar(&mut rng) * 0.02
                } else {
                    pseudo_unit_scalar(&mut rng) * 0.12
                };
                *val = emb[d] + noise;
            }
            l2_normalize(&mut tok);
            if i == planted_idx && t < 4 {
                planted_query_tokens[t * DIM..(t + 1) * DIM].copy_from_slice(&tok);
            }
            for &v in &tok {
                colbert_builder.values().values().append_value(v);
            }
            colbert_builder.values().append(true);
            doc_tokens.extend_from_slice(&tok);
        }
        colbert_builder.append(true);

        id_builder.append_value(&doc_str);
        tenant_builder.append_value(tenant_id);
        ts_builder.append_value(created_at);
        for &v in &emb {
            vec_builder.values().append_value(v);
        }
        vec_builder.append(true);

        html_builder.append_value(format!("{heavy_html_template}<!-- {i} -->"));
        snippet_builder.append_value(format!(
            "{{\"doc\":\"{doc_str}\",\"tenant\":{tenant_id},\"ts\":{created_at},\"title\":\"Record {i}\"}}"
        ));

        gt_docs.push(GroundTruthDoc {
            doc_id,
            tenant_id,
            created_at,
            embedding: emb,
            tokens: doc_tokens,
        });
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::Utf8, false),
        Field::new("tenant_id", DataType::UInt32, false),
        Field::new("created_at", DataType::UInt64, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 128),
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
        Field::new("raw_html_blob", DataType::Utf8, false),
        Field::new("title_and_snippet", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(id_builder.finish()),
            Arc::new(tenant_builder.finish()),
            Arc::new(ts_builder.finish()),
            Arc::new(vec_builder.finish()),
            Arc::new(colbert_builder.finish()),
            Arc::new(html_builder.finish()),
            Arc::new(snippet_builder.finish()),
        ],
    )
    .map_err(|e| invalid_data(e.to_string()))?;

    let file = File::create(parquet_path)?;
    let mut writer = ArrowWriter::try_new(BufWriter::new(file), schema, None)
        .map_err(|e| invalid_data(e.to_string()))?;
    // Write in 4 row-groups of 500 rows to exercise multi-row-group streaming.
    for offset in (0..NUM_DOCS).step_by(500) {
        let slice = batch.slice(offset, 500);
        writer
            .write(&slice)
            .map_err(|e| invalid_data(e.to_string()))?;
    }
    writer.close().map_err(|e| invalid_data(e.to_string()))?;

    Ok(GeneratedLakeDataset {
        gt_docs,
        planted_idx,
        planted_query_vec,
        planted_query_tokens,
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_lake_parquet_end_to_end_osi_osv_osr_hybrid_pipeline() -> OspreyResult<()> {
    let tmp = TempWorkspaceDir::new("osprey_lake_e2e")?;
    let parquet_path = tmp.path.join("corporate_lake.parquet");
    let osi_path = tmp.path.join("segment.osi");
    let osv_path = tmp.path.join("segment.osv");
    let osr_path = tmp.path.join("segment.osr");

    let GeneratedLakeDataset {
        gt_docs,
        planted_idx,
        planted_query_vec,
        planted_query_tokens,
    } = generate_corporate_parquet(&parquet_path)?;

    let hints = IngestColumnHints {
        metadata_columns: vec![
            "tenant_id".to_string(),
            "created_at".to_string(),
            "title_and_snippet".to_string(),
        ],
        ..Default::default()
    };

    // Step 1a: Pass-1 Out-of-Core Reservoir Sampling (embedding-only projection pushdown)
    // and Binary K-Means + TSP locality ordering.
    let ingest_pipeline = OutOfCoreIngestPipeline::new(OutOfCoreIngestConfig {
        max_training_sample_vectors: 1024,
        batch_size_rows: 512,
        num_clusters: NUM_CLUSTERS,
        fwht_seed: FWHT_SEED,
        hints: hints.clone(),
        ..Default::default()
    });
    let (detected_schema, reservoir_f32, sampled_count, total_seen) =
        ingest_pipeline.run_pass1_reservoir_sampling(File::open(&parquet_path)?)?;
    assert_eq!(total_seen, NUM_DOCS as u64);
    assert_eq!(sampled_count, 1024);
    assert!(detected_schema.id_is_string);
    assert_eq!(detected_schema.orig_dim, DIM);
    assert_eq!(detected_schema.colbert_token_dim, Some(LUTRANK_TOKEN_DIM));

    // Verify `raw_html_blob` (column 5) is strictly excluded from projection pushdown!
    let raw_html_idx = 5_usize;
    assert!(!detected_schema.projected_indices().contains(&raw_html_idx));

    let codebook = ingest_pipeline.train_codebook(&reservoir_f32, sampled_count, DIM)?;
    drop(reservoir_f32);

    // Step 1b: Pass-2 Chunked Streaming Extraction (never reading `raw_html_blob`).
    let mut reader = ParquetBatchReader::open(&parquet_path, &hints, 512)?;
    assert!(reader
        .projected_schema()
        .field_with_name("raw_html_blob")
        .is_err());

    let col_map = reader.projected_map().clone();
    let filter_mapping = ScalarFilterColumnMapping::from_schema(reader.projected_schema());
    let snippet_batch_idx = reader
        .projected_schema()
        .index_of("title_and_snippet")
        .map_err(|e| invalid_data(e.to_string()))?;

    let mut all_doc_ids = Vec::with_capacity(NUM_DOCS);
    let mut all_vectors_f32 = Vec::with_capacity(NUM_DOCS * DIM);
    let mut all_vectors_f16 = Vec::with_capacity(NUM_DOCS * DIM);
    let mut all_tsp_assignments = Vec::with_capacity(NUM_DOCS);
    let mut all_payloads: Vec<Vec<u8>> = Vec::with_capacity(NUM_DOCS);
    let mut all_colbert_tokens: Vec<Vec<f32>> = Vec::with_capacity(NUM_DOCS);
    let mut raw_filter_attrs = Vec::with_capacity(NUM_DOCS);

    let mut rot_scratch = vec![0.0_f32; codebook.padded_dim];
    let mut temp_filter = ScalarFilterIndex::new();

    for batch_res in &mut reader {
        let batch = batch_res?;
        let n = batch.num_rows();
        let doc_ids = extract_batch_doc_ids(&batch, col_map.id_batch_idx)?;
        let (dim, vecs_f32) =
            extract_batch_vectors_f32(&batch, col_map.vector_batch_idx, Some(DIM))?;
        assert_eq!(dim, DIM);

        let cb_idx = col_map
            .colbert_batch_idx
            .ok_or_else(|| invalid_data("missing ColBERT column in projected batch"))?;
        let (tok_dim, toks) =
            extract_batch_colbert_tokens_f32(&batch, cb_idx, Some(LUTRANK_TOKEN_DIM))?;
        assert_eq!(tok_dim, LUTRANK_TOKEN_DIM);

        let snippet_col = batch
            .column(snippet_batch_idx)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .ok_or_else(|| invalid_data("expected StringArray for title_and_snippet"))?;

        let prev_len = temp_filter.len();
        temp_filter.insert_arrow_batch_with_doc_ids(
            &batch,
            &filter_mapping,
            Some(&doc_ids),
            None,
        )?;

        for r in 0..n {
            let v_slice = &vecs_f32[r * DIM..(r + 1) * DIM];
            let cid = ingest_pipeline.assign_tsp_cluster(&codebook, v_slice, &mut rot_scratch)?;
            all_tsp_assignments.push(cid);
            for &val in v_slice {
                all_vectors_f16.push(f16::from_f32(val));
            }
            all_payloads.push(snippet_col.value(r).as_bytes().to_vec());
            let row_id = u32::try_from(prev_len + r).unwrap_or(0);
            let attrs = temp_filter
                .attributes(row_id)
                .copied()
                .ok_or_else(|| invalid_data("missing filter attrs"))?;
            raw_filter_attrs.push(attrs);
        }

        all_doc_ids.extend_from_slice(&doc_ids);
        all_vectors_f32.extend_from_slice(&vecs_f32);
        all_colbert_tokens.extend(toks);
    }
    assert_eq!(all_doc_ids.len(), NUM_DOCS);

    // Step 1c: Write `.osv` slab (16 KB Zstd minipages in TSP order) & upload to `ObjectStore`.
    let osv_records: Vec<OsvVectorRecord<'_>> = (0..NUM_DOCS)
        .map(|i| OsvVectorRecord {
            doc_id: all_doc_ids[i],
            cluster_id: all_tsp_assignments[i],
            vector_f16: &all_vectors_f16[i * DIM..(i + 1) * DIM],
            payload: &all_payloads[i],
        })
        .collect();

    let osv_writer = OsvWriter::new(OsvWriterConfig::default());
    let osv_file = File::create(&osv_path)?;
    let osv_report =
        osv_writer.write_slab(&mut BufWriter::new(osv_file), &osv_records, DIM, None)?;

    let osv_bytes = std::fs::read(&osv_path)?;
    let osv_reader = OsvSlabReader::open_mmap(&osv_path)?;
    let directory = osv_reader.directory().to_vec();
    let zstd_dict = if osv_reader.header().zstd_dict_len > 0 {
        let start = usize::try_from(osv_reader.header().zstd_dict_offset)
            .map_err(|_| invalid_data("dict offset overflow"))?;
        let len = usize::try_from(osv_reader.header().zstd_dict_len)
            .map_err(|_| invalid_data("dict len overflow"))?;
        Some(osv_bytes[start..start + len].to_vec())
    } else {
        None
    };

    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let osv_obj_path = ObjectPath::from("lake/segment.osv");
    store
        .put(&osv_obj_path, osv_bytes.into())
        .await
        .map_err(|e| invalid_data(e.to_string()))?;

    // Populate `ScalarFilterIndex` with exact `.osv` `minipage_id`s for Zone Map pruning!
    let mut filter_index = ScalarFilterIndex::with_capacity(NUM_DOCS);
    for i in 0..NUM_DOCS {
        let mp_id = osv_report
            .location_for_doc(all_doc_ids[i])
            .map(|(mp, _)| mp);
        filter_index.insert_row_with_doc_id(all_doc_ids[i], mp_id, raw_filter_attrs[i])?;
    }

    // Step 1d: Build & seal `.osi` index (with TSP-reordered centroids and exact `.osv` locators).
    let simd = SimdEngine::new_auto();
    let padded_dim = padded_dimension(DIM)?;
    let mut aligned_centroids =
        osprey_core::AlignedBuffer::<f16>::new(codebook.centroids_f16.len())?;
    aligned_centroids
        .as_mut_slice()
        .copy_from_slice(&codebook.centroids_f16);
    let mut binary_centroids = vec![0_u64; NUM_CLUSTERS * (padded_dim / 64)];
    for c in 0..NUM_CLUSTERS {
        let cent_slice = &codebook.centroids_f16[c * padded_dim..(c + 1) * padded_dim];
        let words_per_vec = padded_dim / 64;
        for (w, out_word) in binary_centroids[c * words_per_vec..(c + 1) * words_per_vec]
            .iter_mut()
            .enumerate()
        {
            let mut word = 0_u64;
            for b in 0..64 {
                if cent_slice[w * 64 + b].to_f32() >= 0.0 {
                    word |= 1_u64 << b;
                }
            }
            *out_word = word;
        }
    }
    let km_result = BinaryKMeansResult {
        centroids_f16: aligned_centroids,
        binary_centroids,
        assignments: all_tsp_assignments,
        final_hamming_inertia: 0,
        iterations_run: 1,
    };
    let locations: Vec<VectorLocation> = all_doc_ids
        .iter()
        .map(|&doc_id| {
            let (minipage_id, slot_idx) = osv_report.location_for_doc(doc_id).unwrap_or((0, 0));
            VectorLocation {
                minipage_id,
                slot_idx,
                slab_id: 0,
            }
        })
        .collect();

    let osi_index = OsiIndex::build(
        &all_vectors_f32,
        &all_doc_ids,
        &locations,
        &km_result,
        DIM,
        FWHT_SEED,
        RoaringBitmap::new(),
        &simd,
    )?;
    let osi_aligned = osi_index.to_aligned_bytes()?;
    std::fs::write(&osi_path, osi_aligned.as_slice())?;
    let loaded_osi = OsiIndex::from_bytes(osi_aligned.as_slice())?;

    // Step 1e: Build & seal `.osr` ColBERT store (nested lattice nibble-packed tokens) & mmap it.
    let osr_docs: Vec<OsrDocumentTokens<'_>> = (0..NUM_DOCS)
        .map(|i| OsrDocumentTokens {
            doc_id: all_doc_ids[i],
            tokens: &all_colbert_tokens[i],
            num_tokens: TOKENS_PER_DOC,
        })
        .collect();
    let osr_built = OsrStore::build(
        &osr_docs,
        LUTRANK_TOKEN_DIM,
        DEFAULT_LUTRANK_GAMMA,
        FWHT_SEED,
        &simd,
    )?;
    let osr_aligned = osr_built.to_aligned_bytes()?;
    std::fs::write(&osr_path, osr_aligned.as_slice())?;
    let osr_mmap = OsrStore::open_mmap(&osr_path)?;

    // Step 2: Execute hybrid filtered searches (tenant_id == 20 AND created_at in [1700004000, 1700014000]).
    let ts_min = 1_700_004_000_u64;
    let ts_max = 1_700_014_000_u64;
    let predicate = FilterPredicate::And(vec![
        FilterPredicate::TenantEq(20),
        FilterPredicate::TimestampRange {
            min: ts_min,
            max: ts_max,
        },
    ]);
    let eligible_rows = filter_index.evaluate(&predicate);
    assert!(!eligible_rows.is_empty());

    let refine_stage = RefineStage::new(0, RangePlannerConfig::default(), zstd_dict);
    let cache = MinipageLruCache::new(16 * 1024 * 1024);
    let remote_cfg = RemoteStoreConfig::default();

    // Build 10 filtered test queries (query 0 is the planted target doc `planted_idx`,
    // queries 1..10 are synthetic queries drawn from filtered tenant-20 docs).
    let filtered_indices: Vec<usize> = (0..NUM_DOCS)
        .filter(|&i| {
            gt_docs[i].tenant_id == 20
                && gt_docs[i].created_at >= ts_min
                && gt_docs[i].created_at <= ts_max
        })
        .collect();
    assert!(filtered_indices.len() >= 50);

    let mut query_indices = vec![planted_idx];
    for step in 1..10 {
        let pick = filtered_indices[(step * 17) % filtered_indices.len()];
        query_indices.push(pick);
    }

    let mut total_stage2_recall_hits = 0_usize;
    let total_recall_slots = query_indices.len() * 10;

    for (q_idx, &target_doc_idx) in query_indices.iter().enumerate() {
        let (q_vec, q_tokens, num_q_toks) = if q_idx == 0 {
            (&planted_query_vec, &planted_query_tokens, 4_usize)
        } else {
            (
                &gt_docs[target_doc_idx].embedding,
                &gt_docs[target_doc_idx].tokens,
                4_usize,
            )
        };

        // Ground truth Top-10 exact dense scores among filtered documents.
        let mut exact_dense: Vec<(u64, f32)> = filtered_indices
            .iter()
            .map(|&idx| {
                let dot: f32 = q_vec
                    .iter()
                    .zip(gt_docs[idx].embedding.iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                (gt_docs[idx].doc_id, dot)
            })
            .collect();
        exact_dense.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let gt_top10: HashSet<u64> = exact_dense.iter().take(10).map(|&(id, _)| id).collect();

        // Stage 1: `.osi` 1-bit RaBitQ scan + O(1) RoaringBitmap filter -> Top-100 candidates.
        let stage1_candidates = loaded_osi.search_candidates_filtered_with(
            q_vec,
            NUM_CLUSTERS, // probe all 16 clusters (tiny scan in RAM)
            100,
            &simd,
            |_entry_idx, doc_id| filter_index.is_doc_eligible(doc_id, &eligible_rows),
        )?;
        assert!(!stage1_candidates.is_empty());

        // Stage 2: DP range coalescing + async ObjectStore::get_ranges + exact f16 refine -> Top-25 candidates.
        let (refined_top25, _stats) = refine_stage
            .refine_scored_async(
                q_vec,
                &stage1_candidates,
                25,
                &directory,
                &cache,
                &store,
                &osv_obj_path,
                &remote_cfg,
            )
            .await?;
        assert!(!refined_top25.is_empty());

        // Measure Recall@10 of dense Stage-2 exact refinement against ground-truth filtered Top-10.
        let stage2_top10: HashSet<u64> = refined_top25.iter().take(10).map(|r| r.doc_id).collect();
        total_stage2_recall_hits += gt_top10.intersection(&stage2_top10).count();

        // Stage 3: ColBERT `LUT-Rank` MaxSim via `vpshufb`/`vtbl` on `.osr` mmap -> Top-10 final.
        let q_lut = QuantizedQueryLut::synthesize(
            &q_tokens[..num_q_toks * LUTRANK_TOKEN_DIM],
            num_q_toks,
            LUTRANK_TOKEN_DIM,
            DEFAULT_LUTRANK_GAMMA,
            FWHT_SEED,
            &simd,
        )?;
        let final_top10 =
            RefineStage::rerank_with_osr(&refined_top25, &osr_mmap, &q_lut, 10, &simd)?;
        assert_eq!(final_top10.len(), 10);

        // Assert 100% tenant_id == 20 & created_at filter compliance on final Top-10!
        for (res, _maxsim) in &final_top10 {
            let attrs = filter_index
                .attributes_for_doc_id(res.doc_id)
                .ok_or_else(|| invalid_data("returned doc_id missing in filter index"))?;
            assert_eq!(attrs.tenant_id, 20);
            assert!(attrs.timestamp_epoch_secs >= ts_min && attrs.timestamp_epoch_secs <= ts_max);

            let json_val: serde_json::Value = serde_json::from_slice(&res.payload_json)
                .map_err(|e| invalid_data(e.to_string()))?;
            assert_eq!(json_val["tenant"], 20);
        }

        // Assert planted target document ranks #1 (Top-1) after LUT-Rank reranking!
        if q_idx == 0 {
            assert_eq!(
                final_top10[0].0.doc_id, gt_docs[planted_idx].doc_id,
                "planted target document must rank #1 after LUT-Rank rerank"
            );
        }
    }

    let recall_at_10 = (total_stage2_recall_hits as f64) / (total_recall_slots as f64);
    println!(
        "Filtered Recall@10 across {} queries = {:.2}% ({}/{})",
        query_indices.len(),
        recall_at_10 * 100.0,
        total_stage2_recall_hits,
        total_recall_slots
    );
    assert!(
        recall_at_10 > 0.965,
        "expected filtered Recall@10 > 96.5%, got {:.2}%",
        recall_at_10 * 100.0
    );

    // Also verify `inspect_parquet_schema` directly on the projected/full schema and `OsiIndex::search_candidates_filtered`.
    let inspected_again = inspect_parquet_schema(reader.projected_schema().as_ref(), &hints)?;
    assert_eq!(inspected_again.orig_dim, DIM);

    let mut small_bm = RoaringBitmap::new();
    small_bm.insert(0);
    let _ =
        loaded_osi.search_candidates_filtered(&planted_query_vec, 4, 10, &simd, Some(&small_bm))?;

    Ok(())
}
