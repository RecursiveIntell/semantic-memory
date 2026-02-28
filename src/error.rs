/// Error types for the semantic-memory crate.
///
/// All errors flow through [`MemoryError`], using `#[from]` for automatic
/// conversion from rusqlite and reqwest errors.
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    /// SQLite / rusqlite error.
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    /// HTTP error from the embedding provider.
    #[error("Embedding request failed: {0}")]
    EmbeddingRequest(#[from] reqwest::Error),

    /// Embedding vector has wrong number of dimensions.
    #[error("Embedding provider returned {actual} dimensions, expected {expected}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Raw BLOB data is not a valid embedding.
    #[error("Invalid embedding data: expected {expected_bytes} bytes, got {actual_bytes}")]
    InvalidEmbedding {
        expected_bytes: usize,
        actual_bytes: usize,
    },

    /// Database was created with a different embedding model.
    #[error("Embedding model mismatch: database has '{stored}', config specifies '{configured}'")]
    ModelMismatch { stored: String, configured: String },

    /// Session with the given ID does not exist.
    #[error("Session not found: {0}")]
    SessionNotFound(String),

    /// Fact with the given ID does not exist.
    #[error("Fact not found: {0}")]
    FactNotFound(String),

    /// Document with the given ID does not exist.
    #[error("Document not found: {0}")]
    DocumentNotFound(String),

    /// Embedding provider is unreachable or misconfigured.
    #[error("Embedding provider unavailable: {0}")]
    EmbedderUnavailable(String),

    /// Database migration failed.
    #[error("Migration failed at version {version}: {reason}")]
    MigrationFailed { version: u32, reason: String },

    /// HNSW index error.
    #[error("HNSW index error: {0}")]
    HnswError(String),

    /// Invalid HNSW key format.
    #[error("Invalid HNSW key format: {0}")]
    InvalidKey(String),

    /// Quantization error.
    #[error("Quantization error: {0}")]
    QuantizationError(String),

    /// Storage path error.
    #[error("Storage path error: {0}")]
    StorageError(String),

    /// Index integrity check failed.
    #[error("Index integrity check failed: {in_sqlite_not_hnsw} items in SQLite but not HNSW, {in_hnsw_not_sqlite} items in HNSW but not SQLite")]
    IntegrityError {
        in_sqlite_not_hnsw: usize,
        in_hnsw_not_sqlite: usize,
    },

    /// Catch-all for other errors.
    #[error("{0}")]
    Other(String),
}
