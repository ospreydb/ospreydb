//! Shared `LUT-Rank` multi-vector reranking service used by both the Cohere/Jina
//! drop-in `POST /v1/rerank` endpoint and the Apache Arrow Flight `do_exchange` server.

use std::cmp::Ordering;
use std::time::Instant;

use osprey_core::lutrank::{
    encode_token, synthesize_query_token_luts, NestedLatticeCodebooks, OsrTokenCode,
    QueryTokenLuts, LUTRANK_SUBSPACES, LUTRANK_TOKEN_DIM,
};
use osprey_core::math::{prepare_and_pad_vector, RademacherSigns};
use osprey_simd::{SimdBackend, SimdEngine};
use xxhash_rust::xxh64::xxh64;

use crate::error::ProResult;
use crate::metrics::ProMetricsRegistry;

/// Default deterministic FWHT seed for online `LUT-Rank` token projection.
pub const DEFAULT_LUTRANK_SEED: u64 = 0x4F53_5052_4C55;

/// Ranked document index and `MaxSim` relevance score returned by [`LutRankReranker`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RerankHit {
    /// Original 0-based document index in the request list.
    pub index: usize,
    /// `LUT-Rank` `MaxSim(Q, D)` relevance score.
    pub relevance_score: f32,
}

/// Stateless `LUT-Rank` multi-vector reranker using nested 2D lattice quantization
/// and runtime-dispatched `vpshufb`/`vtbl` SIMD `MaxSim` kernels.
pub struct LutRankReranker {
    codebooks: NestedLatticeCodebooks,
    signs: RademacherSigns,
    simd: SimdEngine,
}

impl std::fmt::Debug for LutRankReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LutRankReranker")
            .field("token_dim", &self.codebooks.token_dim)
            .field("gamma", &self.codebooks.gamma)
            .finish_non_exhaustive()
    }
}

impl LutRankReranker {
    /// Creates a canonical `LUT-Rank` reranker (`token_dim = 128`, `gamma = 0.28`).
    pub fn new(fwht_seed: u64) -> ProResult<Self> {
        let codebooks = NestedLatticeCodebooks::canonical()?;
        let signs = RademacherSigns::from_seed(LUTRANK_TOKEN_DIM, fwht_seed)?;
        Ok(Self {
            codebooks,
            signs,
            simd: SimdEngine::new_auto(),
        })
    }

    /// Reranks raw UTF-8 text documents against `query` by projecting tokens into 128-dim
    /// embeddings, quantizing via 2-layer nested lattice, and evaluating `LUT-Rank` `MaxSim`.
    pub fn rerank_texts(
        &self,
        query: &str,
        documents: &[&str],
        top_n: Option<usize>,
        metrics: Option<&ProMetricsRegistry>,
    ) -> ProResult<Vec<RerankHit>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let q_tokens = project_text_to_token_embeddings(query);
        let d_tokens: Vec<Vec<[f32; LUTRANK_TOKEN_DIM]>> = documents
            .iter()
            .map(|doc| project_text_to_token_embeddings(doc))
            .collect();

        let q_refs: Vec<&[f32]> = q_tokens.iter().map(|t| t.as_slice()).collect();
        let d_refs: Vec<Vec<&[f32]>> = d_tokens
            .iter()
            .map(|toks| toks.iter().map(|t| t.as_slice()).collect())
            .collect();

        self.rerank_token_embeddings(&q_refs, &d_refs, top_n, metrics)
    }

    /// Reranks multi-vector 128-dim token embeddings directly using `LUT-Rank` SIMD `MaxSim`.
    pub fn rerank_token_embeddings(
        &self,
        query_tokens: &[&[f32]],
        doc_tokens: &[Vec<&[f32]>],
        top_n: Option<usize>,
        metrics: Option<&ProMetricsRegistry>,
    ) -> ProResult<Vec<RerankHit>> {
        if doc_tokens.is_empty() || query_tokens.is_empty() {
            return Ok(Vec::new());
        }

        let start = Instant::now();
        let mut scratch_rot = [0.0_f32; LUTRANK_TOKEN_DIM];
        let mut query_luts = Vec::with_capacity(query_tokens.len());

        for &q_tok in query_tokens {
            prepare_and_pad_vector(q_tok, &mut scratch_rot)?;
            self.simd
                .fwht_rotate_in_place(&mut scratch_rot, self.signs.xor_masks())?;
            let mut luts = QueryTokenLuts::default();
            synthesize_query_token_luts(&scratch_rot, &self.codebooks, &mut luts)?;
            query_luts.push(luts);
        }

        let mut total_doc_tokens = 0_usize;
        let mut hits = Vec::with_capacity(doc_tokens.len());

        for (doc_idx, d_toks) in doc_tokens.iter().enumerate() {
            if d_toks.is_empty() {
                hits.push(RerankHit {
                    index: doc_idx,
                    relevance_score: 0.0,
                });
                continue;
            }
            let mut encoded_tokens: Vec<OsrTokenCode> = Vec::with_capacity(d_toks.len());
            for &tok in d_toks {
                let code = encode_token(
                    tok,
                    &self.signs,
                    &self.codebooks,
                    &self.simd,
                    &mut scratch_rot,
                )?;
                encoded_tokens.push(code);
            }
            total_doc_tokens = total_doc_tokens.saturating_add(encoded_tokens.len());

            let doc_bytes: &[u8] = bytemuck::cast_slice(&encoded_tokens);
            let mut maxsim_sum = 0.0_f32;
            for q_lut in &query_luts {
                let q1_i8: &[i8] = bytemuck::cast_slice(&q_lut.lut1);
                let q2_i8: &[i8] = bytemuck::cast_slice(&q_lut.lut2);
                maxsim_sum += self.simd.lutrank_doc_maxsim(
                    doc_bytes,
                    encoded_tokens.len(),
                    LUTRANK_SUBSPACES,
                    q1_i8,
                    q2_i8,
                    1,
                    q_lut.scale1,
                    q_lut.scale2,
                );
            }

            #[allow(clippy::cast_precision_loss)]
            let normalized = maxsim_sum / (query_luts.len() as f32);
            hits.push(RerankHit {
                index: doc_idx,
                relevance_score: normalized,
            });
        }

        hits.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.index.cmp(&b.index))
        });
        if let Some(limit) = top_n {
            hits.truncate(limit);
        }

        if let Some(reg) = metrics {
            let elapsed_us = u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX);
            reg.observe_lutrank(
                doc_tokens.len(),
                total_doc_tokens.saturating_mul(query_tokens.len()),
                elapsed_us,
            );
        }

        Ok(hits)
    }
}

/// Deterministically projects UTF-8 text into L2-normalized 128-dim token embeddings
/// using word/subword n-gram feature hashing (enabling zero-dependency Cohere/Jina text rerank).
fn project_text_to_token_embeddings(text: &str) -> Vec<[f32; LUTRANK_TOKEN_DIM]> {
    let words: Vec<&str> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();

    if words.is_empty() {
        let mut fallback = [0.0_f32; LUTRANK_TOKEN_DIM];
        fallback[0] = 1.0;
        return vec![fallback];
    }

    words
        .into_iter()
        .take(64)
        .map(|word| {
            let lower = word.to_lowercase();
            let bytes = lower.as_bytes();
            let mut vec = [0.0_f32; LUTRANK_TOKEN_DIM];

            // Exact token projection + Unicode character trigram hashing for morphological overlap.
            let h_word = xxh64(bytes, 0x9E37_79B9_7F4A_7C15);
            let idx_main = usize::try_from(h_word % 128).unwrap_or(0);
            vec[idx_main] += 2.0;
            let idx_sec = usize::try_from((h_word >> 16) % 128).unwrap_or(0);
            vec[idx_sec] += 1.0;

            let chars: Vec<char> = lower.chars().collect();
            if chars.len() >= 3 {
                for tri in chars.windows(3) {
                    let mut tri_buf = [0_u8; 12];
                    let mut pos = 0_usize;
                    for &ch in tri {
                        let encoded_len = ch.encode_utf8(&mut tri_buf[pos..]).len();
                        pos += encoded_len;
                    }
                    let h = xxh64(&tri_buf[..pos], 0xBF58_476D_1CE4_E5B9);
                    let dim_idx = usize::try_from(h % 128).unwrap_or(0);
                    let sign = if ((h >> 63) & 1) == 0 { 1.0 } else { -1.0 };
                    vec[dim_idx] += 0.5 * sign;
                }
            } else {
                let h = xxh64(bytes, 0x94D0_49BB_1331_11EB);
                let dim_idx = usize::try_from(h % 128).unwrap_or(0);
                vec[dim_idx] += 1.0;
            }

            let norm_sq: f32 = vec.iter().map(|v| v * v).sum();
            let inv_norm = if norm_sq > 1e-12 {
                1.0 / norm_sq.sqrt()
            } else {
                1.0
            };
            for v in &mut vec {
                *v *= inv_norm;
            }
            vec
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn lutrank_reranker_ranks_relevant_document_first() {
        let reranker = LutRankReranker::new(DEFAULT_LUTRANK_SEED).unwrap();
        let metrics = ProMetricsRegistry::new();

        let docs = [
            "chocolate cake recipe with vanilla frosting",
            "vector search on S3 slabs using RaBitQ and DP range coalescing",
            "kubernetes pod autoscaling for stateless web apps",
        ];
        let hits = reranker
            .rerank_texts(
                "S3 vector search DP range coalescing",
                &docs,
                Some(2),
                Some(&metrics),
            )
            .unwrap();

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].index, 1);
        assert!(hits[0].relevance_score > hits[1].relevance_score);
        assert!(metrics.snapshot().lutrank_tokens_total > 0);
    }

    #[test]
    fn lutrank_reranker_handles_unicode_accented_text_case_insensitively() {
        let reranker = LutRankReranker::new(DEFAULT_LUTRANK_SEED).unwrap();
        let docs = [
            "receita de bolo de chocolate com baunilha",
            "BUSCA SEMÂNTICA VETORIAL EM PRODUÇÃO COM ACENTUAÇÃO",
            "configuração de rede local sem fio",
        ];
        let hits = reranker
            .rerank_texts("busca semântica em produção", &docs, Some(3), None)
            .unwrap();
        assert_eq!(hits[0].index, 1);
        assert!(hits[0].relevance_score > hits[1].relevance_score);
    }
}
