use crate::tokenizer::TokenCounter;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

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

    /// Connection pool configuration.
    pub pool: PoolConfig,

    /// Resource limits.
    pub limits: MemoryLimits,

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
            .field("pool", &self.pool)
            .field("limits", &self.limits)
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
            pool: PoolConfig::default(),
            limits: MemoryLimits::default(),
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

/// Connection pool configuration for SQLite.
///
/// Controls busy timeout and WAL checkpoint behavior. These defaults
/// are tuned for a single-process server on local SSD storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    /// SQLite busy timeout in milliseconds.
    /// Default: 5000 (5 seconds).
    pub busy_timeout_ms: u32,

    /// WAL auto-checkpoint threshold in pages.
    /// Default: 1000 (~4 MB with 4KB pages).
    pub wal_autocheckpoint: u32,

    /// Enable WAL mode. Should almost always be true.
    /// Default: true.
    pub enable_wal: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            busy_timeout_ms: 5000,
            wal_autocheckpoint: 1000,
            enable_wal: true,
        }
    }
}

/// Resource limits for the memory system.
///
/// Prevents runaway resource usage. All limits have defaults tuned for
/// a laptop-class server (8GB RAM, SSD storage).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLimits {
    /// Maximum number of facts per namespace.
    /// Default: 100_000.
    pub max_facts_per_namespace: usize,

    /// Maximum number of chunks per document.
    /// Default: 1_000.
    pub max_chunks_per_document: usize,

    /// Maximum content size in bytes for a single fact or message.
    /// Default: 1 MB (1_048_576 bytes).
    pub max_content_bytes: usize,

    /// Maximum number of concurrent embedding requests.
    /// Hard-capped at 32 regardless of config.
    /// Default: 8.
    pub max_embedding_concurrency: usize,

    /// Maximum total database size in bytes. 0 = unlimited.
    /// Default: 0 (unlimited).
    pub max_db_size_bytes: u64,

    /// Embedding request timeout.
    /// Default: 30 seconds.
    #[serde(with = "duration_secs")]
    pub embedding_timeout: Duration,
}

impl Default for MemoryLimits {
    fn default() -> Self {
        Self {
            max_facts_per_namespace: 100_000,
            max_chunks_per_document: 1_000,
            max_content_bytes: 1_048_576,
            max_embedding_concurrency: 8,
            max_db_size_bytes: 0,
            embedding_timeout: Duration::from_secs(30),
        }
    }
}

impl MemoryLimits {
    /// Validate and clamp limits to hard caps.
    pub fn validated(mut self) -> Self {
        // Hard cap: concurrency at 32
        if self.max_embedding_concurrency > 32 {
            self.max_embedding_concurrency = 32;
        }
        if self.max_embedding_concurrency == 0 {
            self.max_embedding_concurrency = 1;
        }
        self
    }
}

mod duration_secs {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(d.as_secs())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        let secs = u64::deserialize(d)?;
        Ok(Duration::from_secs(secs))
    }
}
