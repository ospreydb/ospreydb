//! Lightweight async HTTP embedding client compatible with the standard
//! OpenAI `/v1/embeddings` contract (OpenAI, Voyage AI, Jina, Cohere, Ollama,
//! vLLM, LiteLLM). Zero ONNX/Torch dependencies.
//!
//! Configured via `OSPREY_EMBED_URL`, `OSPREY_EMBED_MODEL`, `OSPREY_EMBED_API_KEY`
//! environment variables or CLI flags. Includes automatic batching and exponential
//! retry with jitter for HTTP `429` (Rate Limit).

use std::sync::Arc;
use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tracing;

use crate::error::{ProEngineError, ProResult};

/// Default batch size for embedding API requests.
const DEFAULT_BATCH_SIZE: usize = 32;

/// Maximum concurrent embedding requests.
const MAX_CONCURRENT_REQUESTS: usize = 16;

/// Maximum retry attempts on 429 rate-limit responses.
const MAX_RETRIES: usize = 5;

/// Base backoff duration for exponential retry.
const BASE_BACKOFF_MS: u64 = 200;

/// Configuration for the `/v1/embeddings` HTTP embedding client.
#[derive(Clone, Debug)]
pub struct EmbedderConfig {
    /// Base URL of the OpenAI-compatible embeddings endpoint (e.g. `https://api.openai.com`).
    pub embed_url: String,
    /// Model name sent in the embedding request body.
    pub embed_model: String,
    /// Optional bearer API key (`Authorization: Bearer <key>`).
    pub api_key: Option<String>,
    /// Maximum texts per single `/v1/embeddings` request.
    pub batch_size: usize,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            embed_url: String::new(),
            embed_model: "text-embedding-3-small".to_owned(),
            api_key: None,
            batch_size: DEFAULT_BATCH_SIZE,
        }
    }
}

/// OpenAI-compatible `/v1/embeddings` request body.
#[derive(Clone, Debug, Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<&'a str>,
}

/// OpenAI-compatible usage object in the embedding response.
#[derive(Clone, Debug, Deserialize)]
struct EmbedUsage {
    #[allow(dead_code)]
    prompt_tokens: u64,
    #[serde(default)]
    total_tokens: u64,
}

/// Single embedding data entry in the response.
#[derive(Clone, Debug, Deserialize)]
struct EmbedData {
    embedding: Vec<f32>,
    index: usize,
}

/// OpenAI-compatible `/v1/embeddings` response body.
#[derive(Clone, Debug, Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
    #[serde(default)]
    usage: Option<EmbedUsage>,
}

/// Asynchronous, batched HTTP embedding client for the standard `/v1/embeddings` contract.
#[derive(Clone, Debug)]
pub struct Embedder {
    client: Client,
    config: EmbedderConfig,
    semaphore: Arc<Semaphore>,
}

impl Embedder {
    /// Creates a new [`Embedder`] with the given [`EmbedderConfig`].
    #[must_use]
    pub fn new(config: EmbedderConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .pool_max_idle_per_host(MAX_CONCURRENT_REQUESTS)
            .build()
            .unwrap_or_else(|_| Client::new());

        Self {
            client,
            config,
            semaphore: Arc::new(Semaphore::new(MAX_CONCURRENT_REQUESTS)),
        }
    }

    /// Returns `true` when an embedding endpoint is configured and ready.
    #[must_use]
    pub fn is_configured(&self) -> bool {
        !self.config.embed_url.is_empty()
            && !self.config.embed_model.is_empty()
    }

    /// Returns the configured embedding model name.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.config.embed_model
    }

    /// Returns the configured embedding dimensions by querying the API with a
    /// single-token probe or relying on caller-supplied dimensionality.
    ///
    /// Falls back to `default_dim` when the probe fails or the embedder is
    /// unconfigured.
    pub async fn probe_dimension(&self, default_dim: usize) -> usize {
        if !self.is_configured() {
            return default_dim;
        }
        let texts = vec!["probe".to_owned()];
        match self.embed_batch(&texts).await {
            Ok(embeddings) => embeddings.first().map_or(default_dim, |v| v.len()),
            Err(_) => default_dim,
        }
    }

    /// Embeds a batch of raw text strings into dense `f32` vectors by splitting
    /// into configured `batch_size` chunks, issuing HTTP requests with
    /// concurrency control, and applying exponential backoff on `429` status.
    pub async fn embed_batch(&self, texts: &[String]) -> ProResult<Vec<Vec<f32>>> {
        if !self.is_configured() {
            return Err(ProEngineError::InvalidConfig(
                "embedding endpoint not configured (OSPREY_EMBED_URL/OSPREY_EMBED_MODEL missing)",
            ));
        }
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = self.config.batch_size.max(1);
        let chunks: Vec<&[String]> = texts.chunks(batch_size).collect();
        let mut handles = Vec::with_capacity(chunks.len());

        for chunk in chunks {
            let embedder = self.clone();
            let chunk_owned: Vec<String> = chunk.to_vec();
            let handle = tokio::spawn(async move {
                embedder.embed_single_batch_with_retry(&chunk_owned).await
            });
            handles.push(handle);
        }

        let mut results: Vec<Vec<Vec<f32>>> = Vec::with_capacity(handles.len());
        for handle in handles {
            match handle.await {
                Ok(Ok(batch)) => results.push(batch),
                Ok(Err(err)) => return Err(err),
                Err(join_err) => {
                    return Err(ProEngineError::Io(std::io::Error::other(format!(
                        "embedder task panicked: {join_err}"
                    ))));
                }
            }
        }

        let total = texts.len();
        let mut flat = Vec::with_capacity(total);
        for batch in results {
            flat.extend(batch);
        }
        Ok(flat)
    }

    /// Embeds a single text string, returning a `Vec<f32>` embedding.
    pub async fn embed_one(&self, text: &str) -> ProResult<Vec<f32>> {
        let results = self.embed_batch(&[text.to_owned()]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| ProEngineError::Io(std::io::Error::other("empty embedding response")))
    }

    async fn embed_single_batch_with_retry(
        &self,
        texts: &[String],
    ) -> ProResult<Vec<Vec<f32>>> {
        let _permit = self.semaphore.acquire().await.map_err(|_| {
            ProEngineError::Io(std::io::Error::other("embedder semaphore closed"))
        })?;

        let endpoint = format!("{}/v1/embeddings", self.config.embed_url.trim_end_matches('/'));
        let body = EmbedRequest {
            model: &self.config.embed_model,
            input: texts,
            encoding_format: None,
        };

        for attempt in 0..MAX_RETRIES {
            let mut req = self
                .client
                .post(&endpoint)
                .json(&body);

            if let Some(ref key) = self.config.api_key {
                req = req.bearer_auth(key);
            }

            match req.send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        let embed_resp: EmbedResponse = resp.json().await.map_err(|err| {
                            ProEngineError::Io(std::io::Error::other(format!(
                                "failed to decode embedding response: {err}"
                            )))
                        })?;

                        tracing::debug!(
                            model = %self.config.embed_model,
                            num_texts = texts.len(),
                            tokens = embed_resp.usage.as_ref().map_or(0, |u| u.total_tokens),
                            "embedding batch completed"
                        );

                        let mut sorted: Vec<Option<Vec<f32>>> =
                            vec![None; embed_resp.data.len()];
                        for item in embed_resp.data {
                            if item.index < sorted.len() {
                                sorted[item.index] = Some(item.embedding);
                            }
                        }
                        let embeddings: Vec<Vec<f32>> =
                            sorted.into_iter().flatten().collect();
                        return Ok(embeddings);
                    }

                    if status.as_u16() == 429 {
                        if attempt + 1 < MAX_RETRIES {
                            let delay_ms = BASE_BACKOFF_MS
                                .saturating_mul(1 << attempt)
                                .min(30_000);
                            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                            let jitter = (f64::from(delay_ms as u32) * 0.25
                                * rand_factor())
                                as u64;
                            let wait = Duration::from_millis(delay_ms + jitter);
                            tracing::warn!(
                                attempt = attempt + 1,
                                delay_ms = delay_ms + jitter,
                                "embedding rate-limited (429), retrying"
                            );
                            tokio::time::sleep(wait).await;
                            continue;
                        }
                        return Err(ProEngineError::Io(std::io::Error::other(
                            format!(
                                "embedding endpoint rate-limited after {MAX_RETRIES} attempts"
                            ),
                        )));
                    }

                    let error_body = resp.text().await.unwrap_or_default();
                    return Err(ProEngineError::Io(std::io::Error::other(format!(
                        "embedding HTTP {status}: {error_body}"
                    ))));
                }
                Err(err) => {
                    if attempt + 1 < MAX_RETRIES && (err.is_timeout() || err.is_connect()) {
                        let delay_ms = BASE_BACKOFF_MS.saturating_mul(1 << attempt);
                        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                        continue;
                    }
                    return Err(ProEngineError::Io(std::io::Error::other(format!(
                        "embedding request failed: {err}"
                    ))));
                }
            }
        }

        Err(ProEngineError::Io(std::io::Error::other(
            "embedding exhausted all retries",
        )))
    }
}

fn rand_factor() -> f64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let h = RandomState::new().build_hasher().finish();
    (h as f64) / (u64::MAX as f64)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn embedder_is_unconfigured_by_default() {
        let embedder = Embedder::new(EmbedderConfig::default());
        assert!(!embedder.is_configured());
    }

    #[test]
    fn embedder_with_valid_config_is_configured() {
        let config = EmbedderConfig {
            embed_url: "https://api.openai.com".to_owned(),
            embed_model: "text-embedding-3-small".to_owned(),
            api_key: Some("sk-test".to_owned()),
            batch_size: 16,
        };
        let embedder = Embedder::new(config);
        assert!(embedder.is_configured());
        assert_eq!(embedder.model(), "text-embedding-3-small");
    }
}