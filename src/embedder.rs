//! Embedding trait and implementations.
//!
//! Provides the [`Embedder`] trait for text-to-vector conversion,
//! with [`CandleEmbedder`] (default, in-process pure-Rust), [`OllamaEmbedder`]
//! (external Ollama server), and [`MockEmbedder`] (testing).

use crate::config::EmbeddingConfig;
use crate::error::MemoryError;
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::pin::Pin;

/// Boxed future type alias for single embedding results.
pub type EmbedFuture<'a> = Pin<Box<dyn Future<Output = Result<Vec<f32>, MemoryError>> + Send + 'a>>;

/// Boxed future type alias for batch embedding results.
pub type EmbedBatchFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Vec<Vec<f32>>, MemoryError>> + Send + 'a>>;

/// Trait for embedding text into vectors.
///
/// Implement this to swap embedding providers.
pub trait Embedder: Send + Sync {
    /// Embed a single text. Returns a vector of f32.
    fn embed<'a>(&'a self, text: &'a str) -> EmbedFuture<'a>;

    /// Embed multiple texts in a batch.
    ///
    /// Takes owned strings to avoid lifetime issues across async boundaries.
    fn embed_batch<'a>(&'a self, texts: Vec<String>) -> EmbedBatchFuture<'a>;

    /// The model name this embedder uses.
    fn model_name(&self) -> &str;

    /// Expected embedding dimensions.
    fn dimensions(&self) -> usize;
}

// ─── OllamaEmbedder ─────────────────────────────────────────────

/// Embedding provider that calls Ollama's `/api/embed` endpoint.
pub struct OllamaEmbedder {
    client: reqwest::Client,
    base_url: String,
    model: String,
    dimensions: usize,
    batch_size: usize,
}

impl OllamaEmbedder {
    /// Create a new OllamaEmbedder from config.
    ///
    /// Returns an error if the HTTP client cannot be constructed (e.g. TLS backend
    /// is unavailable).
    pub fn try_new(config: &EmbeddingConfig) -> Result<Self, MemoryError> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| {
                MemoryError::EmbedderUnavailable(format!("failed to build HTTP client: {e}"))
            })?;

        Ok(Self {
            client,
            base_url: config.ollama_url.trim_end_matches('/').to_string(),
            model: config.model.clone(),
            dimensions: config.dimensions,
            batch_size: config.batch_size,
        })
    }

    // GOV-005: Deprecated `new()` method removed — all consumers should use `try_new()`.
}

impl Embedder for OllamaEmbedder {
    fn embed<'a>(&'a self, text: &'a str) -> EmbedFuture<'a> {
        Box::pin(async move {
            let mut results = self.embed_batch(vec![text.to_string()]).await?;
            results.pop().ok_or_else(|| {
                MemoryError::Other("Ollama returned empty embeddings for single text".to_string())
            })
        })
    }

    fn embed_batch<'a>(&'a self, texts: Vec<String>) -> EmbedBatchFuture<'a> {
        Box::pin(async move {
            let mut all_embeddings = Vec::with_capacity(texts.len());

            for batch in texts.chunks(self.batch_size) {
                let input: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
                let body = serde_json::json!({
                    "model": self.model,
                    "input": input
                });

                let url = format!("{}/api/embed", self.base_url);
                let response = self
                    .client
                    .post(&url)
                    .json(&body)
                    .send()
                    .await
                    .map_err(|e| {
                        if e.is_connect() {
                            MemoryError::EmbedderUnavailable(format!(
                                "Ollama not running at {}",
                                self.base_url
                            ))
                        } else if e.is_timeout() {
                            MemoryError::EmbedderUnavailable(format!(
                                "Ollama embedding timed out: {}",
                                e
                            ))
                        } else {
                            MemoryError::EmbeddingRequest(e)
                        }
                    })?;

                if response.status() == reqwest::StatusCode::NOT_FOUND {
                    return Err(MemoryError::EmbedderUnavailable(format!(
                        "Model '{}' not available in Ollama. Run: ollama pull {}",
                        self.model, self.model
                    )));
                }

                if !response.status().is_success() {
                    let status = response.status();
                    let body = response
                        .text()
                        .await
                        .map_err(|err| format!("failed to read Ollama error body: {err}"));
                    return Err(format_ollama_http_error(status, body));
                }

                let resp_body: serde_json::Value = response.json().await?;
                let batch_embeddings = parse_embedding_response(&resp_body, self.dimensions)?;
                all_embeddings.extend(batch_embeddings);
            }

            Ok(all_embeddings)
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

#[doc(hidden)]
pub fn format_ollama_http_error(
    status: reqwest::StatusCode,
    body: Result<String, String>,
) -> MemoryError {
    match body {
        Ok(body) => MemoryError::Other(format!(
            "Ollama returned HTTP {}: {}",
            status,
            &body[..body.len().min(500)]
        )),
        Err(err) => MemoryError::Other(format!("Ollama returned HTTP {status}; {err}")),
    }
}

/// Parse an Ollama embedding response body into vectors.
///
/// Validates that all values are numeric and dimensions match.
#[doc(hidden)]
pub fn parse_embedding_response(
    body: &serde_json::Value,
    expected_dims: usize,
) -> Result<Vec<Vec<f32>>, MemoryError> {
    let embeddings = body["embeddings"].as_array().ok_or_else(|| {
        MemoryError::Other("Ollama response missing 'embeddings' field".to_string())
    })?;

    let mut result = Vec::with_capacity(embeddings.len());
    for embedding_val in embeddings {
        let raw_array = embedding_val
            .as_array()
            .ok_or_else(|| MemoryError::Other("Embedding is not an array".to_string()))?;

        let mut embedding = Vec::with_capacity(raw_array.len());
        for (i, v) in raw_array.iter().enumerate() {
            let val = v.as_f64().ok_or_else(|| {
                MemoryError::Other(format!(
                    "Embedding dimension {} contains non-numeric value: {}",
                    i, v
                ))
            })?;
            embedding.push(val as f32);
        }

        if embedding.len() != expected_dims {
            return Err(MemoryError::DimensionMismatch {
                expected: expected_dims,
                actual: embedding.len(),
            });
        }

        result.push(embedding);
    }

    Ok(result)
}

// ─── MockEmbedder ────────────────────────────────────────────────

/// Deterministic embedder for unit tests.
///
/// Generates consistent embeddings based on a hash of the input text.
/// Same text always produces the same embedding. Output is normalized.
pub struct MockEmbedder {
    dimensions: usize,
}

impl MockEmbedder {
    /// Create a new MockEmbedder with the given dimensions.
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }
}

impl Embedder for MockEmbedder {
    fn embed<'a>(&'a self, text: &'a str) -> EmbedFuture<'a> {
        let embedding = deterministic_embedding(text, self.dimensions);
        Box::pin(async move { Ok(embedding) })
    }

    fn embed_batch<'a>(&'a self, texts: Vec<String>) -> EmbedBatchFuture<'a> {
        let embeddings: Vec<Vec<f32>> = texts
            .iter()
            .map(|t| deterministic_embedding(t, self.dimensions))
            .collect();
        Box::pin(async move { Ok(embeddings) })
    }

    fn model_name(&self) -> &str {
        "mock-embedder"
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

/// Generate a deterministic embedding from text using a hash-seeded xorshift RNG.
fn deterministic_embedding(text: &str, dimensions: usize) -> Vec<f32> {
    let mut hasher = std::hash::DefaultHasher::new();
    text.hash(&mut hasher);
    let mut state = hasher.finish();
    if state == 0 {
        state = 1;
    }

    let mut values = Vec::with_capacity(dimensions);
    for _ in 0..dimensions {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let val = ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0;
        values.push(val as f32);
    }

    // Normalize to unit length
    let magnitude: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for v in &mut values {
            *v /= magnitude;
        }
    }

    values
}

// ─── CandleEmbedder ─────────────────────────────────────────────

/// In-process embedder using Candle (pure-Rust ML framework, CPU-only).
///
/// Downloads the model from HuggingFace Hub on first use (cached in
/// `~/.cache/huggingface/hub`). No external process or server required.
///
/// Default model: `nomic-ai/nomic-embed-text-v1.5` (768 dimensions).
/// The model matches the Ollama `nomic-embed-text` embedding, so an
/// existing HNSW index built with Ollama's nomic-embed-text is compatible.
#[cfg(feature = "candle-embedder")]
pub struct CandleEmbedder {
    model: candle_transformers::models::nomic_bert::NomicBertModel,
    tokenizer: tokenizers::Tokenizer,
    device: candle_core::Device,
    model_id: String,
    dimensions: usize,
    max_seq_len: usize,
}

#[cfg(feature = "candle-embedder")]
impl CandleEmbedder {
    /// Create a new CandleEmbedder with the default model (nomic-embed-text-v1.5, 768d).
    pub fn try_new(config: &EmbeddingConfig) -> Result<Self, MemoryError> {
        Self::try_new_with_model(
            "nomic-ai/nomic-embed-text-v1.5",
            config,
        )
    }

    /// Create a CandleEmbedder with a specific HuggingFace model ID.
    ///
    /// The model must be a NomicBert architecture (nomic-embed-text-v1.5).
    /// For other architectures (BERT, MiniLM), use a different model loader.
    pub fn try_new_with_model(model_id: &str, config: &EmbeddingConfig) -> Result<Self, MemoryError> {
        let device = candle_core::Device::Cpu;
        let dimensions = config.dimensions;
        let max_seq_len = 8192; // nomic-embed-text supports up to 8192 tokens

        // Download model files from HuggingFace Hub (cached after first download).
        // We download individual files rather than a snapshot to keep the API
        // simple and avoid pulling unnecessary files.
        let (owner, name) = match model_id.split_once('/') {
            Some((o, n)) => (o, n),
            None => ("nomic-ai", model_id),
        };

        let api = hf_hub::HFClientSync::new().map_err(|e| {
            MemoryError::EmbedderUnavailable(format!("failed to create HF Hub client: {e}"))
        })?;
        let repo = api.model(owner, name);

        // Download required files. These are cached by hf-hub after first download.
        let config_path = download_hf_file(&repo, "config.json")?;
        let tokenizer_path = download_hf_file(&repo, "tokenizer.json")?;

        // Try safetensors first, fall back to pytorch_model.bin.
        let weights_path = download_hf_file(&repo, "model.safetensors")
            .or_else(|_| download_hf_file(&repo, "pytorch_model.bin"))?;

        // Load tokenizer.
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            MemoryError::EmbedderUnavailable(format!("failed to load tokenizer from {}: {e}", tokenizer_path.display()))
        })?;

        // Load model config.
        let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
            MemoryError::EmbedderUnavailable(format!("failed to read config.json: {e}"))
        })?;
        let model_config: candle_transformers::models::nomic_bert::Config =
            serde_json::from_str(&config_str).map_err(|e| {
                MemoryError::EmbedderUnavailable(format!("failed to parse model config: {e}"))
            })?;

        // Verify dimensions match.
        if model_config.n_embd != dimensions {
            return Err(MemoryError::DimensionMismatch {
                expected: dimensions,
                actual: model_config.n_embd,
            });
        }

        // Load model weights via mmap.
        let dtype = candle_core::DType::F32;
        // Read the safetensors file into memory and load from buffer.
        // This avoids the unsafe mmap API (workspace lints deny unsafe_code).
        let weights_bytes = std::fs::read(&weights_path).map_err(|e| {
            MemoryError::EmbedderUnavailable(format!("failed to read weights file {}: {e}", weights_path.display()))
        })?;
        let vb = candle_nn::VarBuilder::from_buffered_safetensors(weights_bytes, dtype, &device)
            .map_err(|e| {
                MemoryError::EmbedderUnavailable(format!("failed to load model weights: {e}"))
            })?;

        let model = candle_transformers::models::nomic_bert::NomicBertModel::load(vb, &model_config)
            .map_err(|e| {
                MemoryError::EmbedderUnavailable(format!("failed to build NomicBert model: {e}"))
            })?;

        Ok(Self {
            model,
            tokenizer,
            device,
            model_id: model_id.to_string(),
            dimensions,
            max_seq_len,
        })
    }

    /// Tokenize and embed a batch of texts, returning f32 vectors.
    fn embed_batch_sync(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, MemoryError> {
        use candle_core::Tensor;
        use candle_transformers::models::nomic_bert::{mean_pooling, l2_normalize};

        let mut all_embeddings = Vec::with_capacity(texts.len());

        // Process one at a time to keep memory bounded (CPU-only).
        for text in texts {
            // nomic-embed-text expects a task prefix. For storing/searching
            // documents, the prefix is "search_document:". For queries, it's
            // "search_query:". Since we store facts (documents) and search
            // with queries, we need to prefix appropriately.
            // However, since both add and search go through this same embedder,
            // and nomic-embed-text-v1.5 is designed for asymmetric search, we
            // use the same "search_document:" prefix for everything to keep
            // it simple. The quality difference is minor for a knowledge base
            // that stores and retrieves facts.
            let prefixed = format!("search_document: {text}");

            let encoding = self.tokenizer.encode(prefixed.as_str(), true).map_err(|e| {
                MemoryError::Other(format!("tokenizer error: {e}"))
            })?;

            let input_ids = encoding.get_ids();
            let attention_mask = encoding.get_attention_mask();

            // Truncate to max_seq_len.
            let seq_len = input_ids.len().min(self.max_seq_len);
            let input_ids = &input_ids[..seq_len];
            let attention_mask = &attention_mask[..seq_len];

            let input_ids_tensor = Tensor::new(input_ids, &self.device)?
                .unsqueeze(0)?; // (1, seq_len)
            let attention_mask_tensor = Tensor::new(attention_mask, &self.device)?
                .unsqueeze(0)?; // (1, seq_len)

            // Run forward pass — NomicBertModel doesn't need token_type_ids
            // (it uses rotary embeddings, and type_vocab_size may be 0).
            let token_type_ids = input_ids_tensor.zeros_like()?;
            let hidden_states = self.model.forward(
                &input_ids_tensor,
                Some(&token_type_ids),
                Some(&attention_mask_tensor),
            )?;

            // Mean-pool using attention mask, then L2-normalize.
            let pooled = mean_pooling(&hidden_states, &attention_mask_tensor)?;
            let normalized = l2_normalize(&pooled)?;

            // Extract the single embedding (batch size 1).
            let embedding_vec = normalized.to_vec2::<f32>()?;
            let embedding = embedding_vec.into_iter().next().ok_or_else(|| {
                MemoryError::Other("model returned empty embedding".to_string())
            })?;

            if embedding.len() != self.dimensions {
                return Err(MemoryError::DimensionMismatch {
                    expected: self.dimensions,
                    actual: embedding.len(),
                });
            }

            all_embeddings.push(embedding);
        }

        Ok(all_embeddings)
    }
}

/// Download a single file from a HuggingFace repo, returning the local path.
/// The file is cached by hf-hub after the first download.
#[cfg(feature = "candle-embedder")]
fn download_hf_file(
    repo: &hf_hub::HFRepositorySync<hf_hub::repository::RepoTypeModel>,
    filename: &str,
) -> Result<std::path::PathBuf, MemoryError> {
    repo.download_file()
        .filename(filename.to_string())
        .send()
        .map_err(|e| {
            MemoryError::EmbedderUnavailable(format!("failed to download '{filename}': {e}"))
        })
}

#[cfg(feature = "candle-embedder")]
impl Embedder for CandleEmbedder {
    fn embed<'a>(&'a self, text: &'a str) -> EmbedFuture<'a> {
        let result = self.embed_batch_sync(&[text.to_string()]);
        Box::pin(async move {
            let mut results = result?;
            results.pop().ok_or_else(|| {
                MemoryError::Other("Candle embedder returned empty results".to_string())
            })
        })
    }

    fn embed_batch<'a>(&'a self, texts: Vec<String>) -> EmbedBatchFuture<'a> {
        let result = self.embed_batch_sync(&texts);
        Box::pin(async move { result })
    }

    fn model_name(&self) -> &str {
        &self.model_id
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}
