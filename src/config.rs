use crate::tokenizer::TokenCounter;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

/// Configuration for the memory system.
#[derive(Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Base directory for all storage files (SQLite + HNSW sidecar files).
    /// Replaces the v0.1.0 `database_path` field.
    pub base_dir: PathBuf,

    /// Embedding provider configuration.
    pub embedding: EmbeddingConfig,

    /// Search tuning parameters.
    pub search: SearchConfig,

    /// Chunking parameters.
    pub chunking: ChunkingConfig,

    /// Custom token counter. None = use EstimateTokenCounter (chars / 4).
    #[serde(skip)]
    pub token_counter: Option<Arc<dyn TokenCounter>>,

    /// HNSW index configuration.
    #[cfg(feature = "hnsw")]
    #[serde(skip)]
    pub hnsw: crate::hnsw::HnswConfig,
}

impl std::fmt::Debug for MemoryConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("MemoryConfig");
        s.field("base_dir", &self.base_dir)
            .field("embedding", &self.embedding)
            .field("search", &self.search)
            .field("chunking", &self.chunking)
            .field(
                "token_counter",
                &self.token_counter.as_ref().map(|_| "custom"),
            );
        #[cfg(feature = "hnsw")]
        s.field("hnsw", &self.hnsw);
        s.finish()
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            base_dir: PathBuf::from("memory"),
            embedding: EmbeddingConfig::default(),
            search: SearchConfig::default(),
            chunking: ChunkingConfig::default(),
            token_counter: None,
            #[cfg(feature = "hnsw")]
            hnsw: crate::hnsw::HnswConfig::default(),
        }
    }
}

/// Embedding provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Ollama base URL.
    pub ollama_url: String,

    /// Embedding model name.
    pub model: String,

    /// Expected embedding dimensions.
    pub dimensions: usize,

    /// Maximum texts to embed in a single API call.
    pub batch_size: usize,

    /// Timeout for embedding requests in seconds.
    pub timeout_secs: u64,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            ollama_url: "http://localhost:11434".to_string(),
            model: "nomic-embed-text".to_string(),
            dimensions: 768,
            batch_size: 32,
            timeout_secs: 30,
        }
    }
}

/// Search tuning parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Weight for BM25 score in RRF fusion.
    pub bm25_weight: f64,

    /// Weight for vector similarity in RRF fusion.
    pub vector_weight: f64,

    /// RRF constant (k). Controls rank importance decay.
    pub rrf_k: f64,

    /// Number of candidates from each search method before fusion.
    pub candidate_pool_size: usize,

    /// Default number of results to return.
    pub default_top_k: usize,

    /// Minimum cosine similarity threshold for vector candidates.
    pub min_similarity: f64,

    /// Optional recency boost. If enabled, results are boosted based on how
    /// recently they were created/updated. The value is the half-life in days —
    /// a fact that is `recency_half_life_days` old gets 50% of the recency boost.
    /// None = no recency weighting (current behavior, default).
    pub recency_half_life_days: Option<f64>,

    /// Weight of the recency boost relative to BM25 and vector scores in RRF.
    /// Only used when recency_half_life_days is Some.
    /// Default: 0.5
    pub recency_weight: f64,

    /// When true, rerank top HNSW candidates using exact f32 cosine similarity
    /// from SQLite. Improves recall at the cost of one batched SQL query.
    /// Only applies when HNSW feature is enabled.
    /// Default: true
    pub rerank_from_f32: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            bm25_weight: 1.0,
            vector_weight: 1.0,
            rrf_k: 60.0,
            candidate_pool_size: 50,
            default_top_k: 5,
            min_similarity: 0.3,
            recency_half_life_days: None,
            recency_weight: 0.5,
            rerank_from_f32: true,
        }
    }
}

/// Text chunking parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    /// Target chunk size in characters.
    pub target_size: usize,

    /// Minimum chunk size. Chunks smaller than this are merged with neighbors.
    pub min_size: usize,

    /// Maximum chunk size. Chunks larger than this are force-split.
    pub max_size: usize,

    /// Overlap between adjacent chunks in characters.
    pub overlap: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            target_size: 1000,
            min_size: 100,
            max_size: 2000,
            overlap: 200,
        }
    }
}
