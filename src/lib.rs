//! # semantic-memory
//!
//! Hybrid semantic search library backed by SQLite + HNSW.
//! Combines BM25 (FTS5) with approximate nearest neighbor search via Reciprocal Rank Fusion.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use semantic_memory::{MemoryConfig, MemoryStore};
//!
//! # async fn example() -> Result<(), semantic_memory::MemoryError> {
//! let store = MemoryStore::open(MemoryConfig::default())?;
//!
//! // Store a fact
//! store.add_fact("general", "Rust was first released in 2015", None, None).await?;
//!
//! // Search
//! let results = store.search("when was Rust released", None, None, None).await?;
//! # Ok(())
//! # }
//! ```

// At least one search backend must be enabled.
#[cfg(not(any(feature = "hnsw", feature = "brute-force")))]
compile_error!("At least one search backend feature must be enabled: 'hnsw' or 'brute-force'");

pub mod chunker;
pub mod config;
pub mod conversation;
pub mod db;
pub mod documents;
pub mod embedder;
pub mod error;
#[cfg(feature = "hnsw")]
pub mod hnsw;
pub mod knowledge;
pub mod quantize;
pub mod search;
pub mod storage;
pub mod tokenizer;
pub mod types;

// Re-export primary public types.
pub use config::{ChunkingConfig, EmbeddingConfig, MemoryConfig, SearchConfig};
pub use embedder::{Embedder, MockEmbedder, OllamaEmbedder};
pub use error::MemoryError;
#[cfg(feature = "hnsw")]
pub use hnsw::{HnswConfig, HnswHit, HnswIndex};
pub use quantize::{pack_quantized, unpack_quantized, QuantizedVector, Quantizer};
pub use storage::StoragePaths;
pub use tokenizer::{EstimateTokenCounter, TokenCounter};
pub use types::{
    Document, Fact, MemoryStats, Message, Role, SearchResult, SearchSource, SearchSourceType,
    Session, TextChunk,
};

use std::sync::{Arc, Mutex};

/// Thread-safe handle to the memory database.
///
/// Clone is cheap (Arc internals). `Send + Sync`.
#[derive(Clone)]
pub struct MemoryStore {
    inner: Arc<MemoryStoreInner>,
}

struct MemoryStoreInner {
    conn: Mutex<rusqlite::Connection>,
    embedder: Box<dyn Embedder>,
    config: MemoryConfig,
    paths: StoragePaths,
    token_counter: Arc<dyn TokenCounter>,
    #[cfg(feature = "hnsw")]
    hnsw_index: std::sync::RwLock<HnswIndex>,
}

#[cfg(feature = "hnsw")]
impl Drop for MemoryStoreInner {
    fn drop(&mut self) {
        let hnsw_guard = match self.hnsw_index.read() {
            Ok(g) => g,
            Err(_) => {
                tracing::warn!("HNSW RwLock poisoned on drop — skipping save");
                return;
            }
        };

        // hnsw_rs::file_dump panics if the directory no longer exists (e.g., TempDir
        // cleaned up before Drop runs). Catch the panic to avoid aborting.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            hnsw_guard.save(&self.paths.hnsw_dir, &self.paths.hnsw_basename)
        }));
        match result {
            Ok(Err(e)) => tracing::error!("Failed to save HNSW index on drop: {}", e),
            Err(_) => tracing::warn!("HNSW save panicked on drop (directory may have been removed)"),
            Ok(Ok(())) => {}
        }

        // Flush key mappings to SQLite
        if let Ok(conn) = self.conn.lock() {
            if let Err(e) = hnsw_guard.flush_keymap(&conn) {
                tracing::error!("Failed to flush HNSW keymap on drop: {}", e);
            }
        }
    }
}

/// Helper to convert `Option<&[&str]>` into owned data for `'static` closures,
/// and convert back to the reference form inside the closure.
fn to_owned_string_vec(opt: Option<&[&str]>) -> Option<Vec<String>> {
    opt.map(|s| s.iter().map(|v| v.to_string()).collect())
}

/// Convert `Option<Vec<String>>` back to `Option<Vec<&str>>` + `Option<&[&str]>`.
fn as_str_slice(opt: &Option<Vec<String>>) -> Option<Vec<&str>> {
    opt.as_ref().map(|v| v.iter().map(|s| s.as_str()).collect())
}

impl MemoryStore {
    /// Run a closure that needs the database connection on a blocking thread.
    ///
    /// This prevents SQLite I/O from stalling the tokio executor. The closure
    /// receives a reference to the Connection (already locked via Mutex).
    async fn with_conn<F, T>(&self, f: F) -> Result<T, MemoryError>
    where
        F: FnOnce(&rusqlite::Connection) -> Result<T, MemoryError> + Send + 'static,
        T: Send + 'static,
    {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let conn = inner.conn.lock().expect("mutex poisoned");
            f(&conn)
        })
        .await
        .map_err(|e| MemoryError::Other(format!("Blocking task panicked: {}", e)))?
    }

    /// Open or create a memory store at the configured base directory.
    ///
    /// Creates the directory if it doesn't exist, opens/creates SQLite,
    /// runs migrations, and initializes the HNSW index.
    pub fn open(config: MemoryConfig) -> Result<Self, MemoryError> {
        let embedder = Box::new(OllamaEmbedder::new(&config.embedding));
        Self::open_with_embedder(config, embedder)
    }

    /// Open with a custom embedder (for testing or non-Ollama providers).
    #[allow(unused_mut)] // `config` is mutated only when the `hnsw` feature is enabled
    pub fn open_with_embedder(
        mut config: MemoryConfig,
        embedder: Box<dyn Embedder>,
    ) -> Result<Self, MemoryError> {
        let paths = StoragePaths::new(&config.base_dir);

        // Create directory if needed
        std::fs::create_dir_all(&paths.base_dir).map_err(|e| {
            MemoryError::StorageError(format!(
                "Failed to create directory {}: {}",
                paths.base_dir.display(),
                e
            ))
        })?;

        let conn = db::open_database(&paths.sqlite_path)?;
        db::check_embedding_metadata(&conn, &config.embedding)?;

        // Ensure HNSW dimensions match the embedding config
        #[cfg(feature = "hnsw")]
        {
            config.hnsw.dimensions = config.embedding.dimensions;
        }

        let token_counter = config
            .token_counter
            .clone()
            .unwrap_or_else(tokenizer::default_token_counter);

        #[cfg(feature = "hnsw")]
        let hnsw_index = {
            let hnsw_config = config.hnsw.clone();

            let embeddings_dirty = db::is_embeddings_dirty(&conn)?;

            if embeddings_dirty {
                // Embedding model changed — old HNSW index is useless.
                // Create a fresh index; reembed_all() will rebuild it.
                tracing::warn!(
                    "Embedding model changed — creating fresh HNSW index (old index is stale)"
                );
                HnswIndex::new(hnsw_config)?
            } else if paths.hnsw_files_exist() {
                tracing::info!("Loading HNSW index from {:?}", paths.hnsw_dir);
                match HnswIndex::load(&paths.hnsw_dir, &paths.hnsw_basename, hnsw_config.clone()) {
                    Ok(index) => {
                        // Load key mappings from SQLite
                        if let Err(e) = index.load_keymap(&conn) {
                            tracing::warn!("Failed to load HNSW key mappings: {}. Mappings will be empty until rebuild.", e);
                        }
                        tracing::info!(
                            "HNSW index loaded ({} active keys)",
                            index.len()
                        );
                        index
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to load HNSW index: {}. Creating new empty index.",
                            e
                        );
                        HnswIndex::new(hnsw_config)?
                    }
                }
            } else {
                tracing::info!("Creating new empty HNSW index");
                HnswIndex::new(hnsw_config)?
            }
        };

        Ok(Self {
            inner: Arc::new(MemoryStoreInner {
                conn: Mutex::new(conn),
                embedder,
                config,
                paths,
                token_counter,
                #[cfg(feature = "hnsw")]
                hnsw_index: std::sync::RwLock::new(hnsw_index),
            }),
        })
    }

    // ─── HNSW Management ───────────────────────────────────────

    /// Rebuild the HNSW index from SQLite f32 embeddings.
    ///
    /// Call this if sidecar files are missing, corrupted, or after `reembed_all()`.
    #[cfg(feature = "hnsw")]
    pub async fn rebuild_hnsw_index(&self) -> Result<(), MemoryError> {
        tracing::info!("Rebuilding HNSW index from SQLite embeddings...");

        let hnsw_config = self.inner.config.hnsw.clone();
        let new_index = HnswIndex::new(hnsw_config)?;

        // Load all fact embeddings
        let fact_data: Vec<(String, Vec<u8>)> = self
            .with_conn(|conn| {
                let mut stmt =
                    conn.prepare("SELECT id, embedding FROM facts WHERE embedding IS NOT NULL")?;
                let result = stmt
                    .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(result)
            })
            .await?;

        for (fact_id, blob) in &fact_data {
            let embedding = db::bytes_to_embedding(blob)?;
            let key = format!("fact:{}", fact_id);
            new_index.insert(key, &embedding)?;
        }

        // Load all chunk embeddings
        let chunk_data: Vec<(String, Vec<u8>)> = self
            .with_conn(|conn| {
                let mut stmt =
                    conn.prepare("SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL")?;
                let result = stmt
                    .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(result)
            })
            .await?;

        for (chunk_id, blob) in &chunk_data {
            let embedding = db::bytes_to_embedding(blob)?;
            let key = format!("chunk:{}", chunk_id);
            new_index.insert(key, &embedding)?;
        }

        // Load all message embeddings
        let msg_data: Vec<(i64, Vec<u8>)> = self
            .with_conn(|conn| {
                let mut stmt = conn
                    .prepare("SELECT id, embedding FROM messages WHERE embedding IS NOT NULL")?;
                let result = stmt
                    .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(result)
            })
            .await?;

        for (msg_id, blob) in &msg_data {
            let embedding = db::bytes_to_embedding(blob)?;
            let key = format!("msg:{}", msg_id);
            new_index.insert(key, &embedding)?;
        }

        let total = fact_data.len() + chunk_data.len() + msg_data.len();
        tracing::info!(
            facts = fact_data.len(),
            chunks = chunk_data.len(),
            messages = msg_data.len(),
            total = total,
            "HNSW index rebuilt"
        );

        // Hot-swap: acquire write lock and replace the index
        {
            let mut guard = self.inner.hnsw_index.write().unwrap();
            *guard = new_index;
        }

        // Persist the new index (read lock is fine now)
        {
            let guard = self.inner.hnsw_index.read().unwrap();
            guard.save(&self.inner.paths.hnsw_dir, &self.inner.paths.hnsw_basename)?;
            let conn = self.inner.conn.lock().expect("mutex poisoned");
            guard.flush_keymap(&conn)?;
        }

        Ok(())
    }

    /// Persist the HNSW graph, vector data, and key mappings to disk.
    ///
    /// Called automatically on drop, but can be called explicitly for durability.
    #[cfg(feature = "hnsw")]
    pub fn flush_hnsw(&self) -> Result<(), MemoryError> {
        let guard = self.inner.hnsw_index.read().unwrap();
        guard.save(&self.inner.paths.hnsw_dir, &self.inner.paths.hnsw_basename)?;

        // Flush key mappings to SQLite
        let conn = self.inner.conn.lock().expect("mutex poisoned");
        guard.flush_keymap(&conn)?;
        Ok(())
    }

    /// Compact the HNSW index by rebuilding without tombstones.
    ///
    /// Only rebuilds if the deleted ratio exceeds the compaction threshold.
    #[cfg(feature = "hnsw")]
    pub async fn compact_hnsw(&self) -> Result<(), MemoryError> {
        if !self.inner.hnsw_index.read().unwrap().needs_compaction() {
            tracing::info!("HNSW compaction not needed (deleted ratio below threshold)");
            return Ok(());
        }
        self.rebuild_hnsw_index().await
    }

    // ─── Session Management ─────────────────────────────────────

    /// Create a new conversation session. Returns the session ID (UUID v4).
    pub async fn create_session(&self, channel: &str) -> Result<String, MemoryError> {
        let channel = channel.to_string();
        self.with_conn(move |conn| conversation::create_session(conn, &channel, None))
            .await
    }

    /// List recent sessions, newest first.
    pub async fn list_sessions(
        &self,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Session>, MemoryError> {
        self.with_conn(move |conn| conversation::list_sessions(conn, limit, offset))
            .await
    }

    /// Delete a session and all its messages.
    pub async fn delete_session(&self, session_id: &str) -> Result<(), MemoryError> {
        let session_id = session_id.to_string();
        self.with_conn(move |conn| conversation::delete_session(conn, &session_id))
            .await
    }

    // ─── Message Storage ────────────────────────────────────────

    /// Append a message to a session. Returns the message's auto-increment ID.
    pub async fn add_message(
        &self,
        session_id: &str,
        role: Role,
        content: &str,
        token_count: Option<u32>,
        metadata: Option<serde_json::Value>,
    ) -> Result<i64, MemoryError> {
        let effective_token_count =
            token_count.or_else(|| Some(self.inner.token_counter.count_tokens(content) as u32));
        let sid = session_id.to_string();
        let ct = content.to_string();
        let meta = metadata;
        self.with_conn(move |conn| {
            conversation::add_message(conn, &sid, role, &ct, effective_token_count, meta.as_ref())
        })
        .await
    }

    /// Get the most recent N messages from a session, in chronological order.
    pub async fn get_recent_messages(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<Message>, MemoryError> {
        let sid = session_id.to_string();
        self.with_conn(move |conn| conversation::get_recent_messages(conn, &sid, limit))
            .await
    }

    /// Get messages from a session up to `max_tokens` total.
    pub async fn get_messages_within_budget(
        &self,
        session_id: &str,
        max_tokens: u32,
    ) -> Result<Vec<Message>, MemoryError> {
        let sid = session_id.to_string();
        self.with_conn(move |conn| conversation::get_messages_within_budget(conn, &sid, max_tokens))
            .await
    }

    /// Get total token count for a session.
    pub async fn session_token_count(&self, session_id: &str) -> Result<u64, MemoryError> {
        let sid = session_id.to_string();
        self.with_conn(move |conn| conversation::session_token_count(conn, &sid))
            .await
    }

    // ─── Fact CRUD ──────────────────────────────────────────────

    /// Store a fact with automatic embedding. Returns the fact ID (UUID v4).
    pub async fn add_fact(
        &self,
        namespace: &str,
        content: &str,
        source: Option<&str>,
        metadata: Option<serde_json::Value>,
    ) -> Result<String, MemoryError> {
        let embedding = self.inner.embedder.embed(content).await?;
        let embedding_bytes = db::embedding_to_bytes(&embedding);
        let fact_id = uuid::Uuid::new_v4().to_string();

        // Quantize for storage
        let quantizer = Quantizer::new(self.inner.config.embedding.dimensions);
        let q8_bytes = quantizer.quantize(&embedding)
            .map(|qv| quantize::pack_quantized(&qv))
            .ok();

        let ns = namespace.to_string();
        let ct = content.to_string();
        let fid = fact_id.clone();
        let src = source.map(|s| s.to_string());
        let meta = metadata;
        self.with_conn(move |conn| {
            knowledge::insert_fact_with_fts_q8(
                conn,
                &fid,
                &ns,
                &ct,
                &embedding_bytes,
                q8_bytes.as_deref(),
                src.as_deref(),
                meta.as_ref(),
            )
        })
        .await?;

        // HNSW insert
        #[cfg(feature = "hnsw")]
        {
            let key = format!("fact:{}", fact_id);
            self.inner.hnsw_index.read().unwrap().insert(key, &embedding)?;
        }

        Ok(fact_id)
    }

    /// Store a fact with a pre-computed embedding.
    pub async fn add_fact_with_embedding(
        &self,
        namespace: &str,
        content: &str,
        embedding: &[f32],
        source: Option<&str>,
        metadata: Option<serde_json::Value>,
    ) -> Result<String, MemoryError> {
        let embedding_bytes = db::embedding_to_bytes(embedding);
        let fact_id = uuid::Uuid::new_v4().to_string();

        // Quantize for storage
        let quantizer = Quantizer::new(self.inner.config.embedding.dimensions);
        let q8_bytes = quantizer.quantize(embedding)
            .map(|qv| quantize::pack_quantized(&qv))
            .ok();

        let ns = namespace.to_string();
        let ct = content.to_string();
        let fid = fact_id.clone();
        let src = source.map(|s| s.to_string());
        let meta = metadata;
        self.with_conn(move |conn| {
            knowledge::insert_fact_with_fts_q8(
                conn,
                &fid,
                &ns,
                &ct,
                &embedding_bytes,
                q8_bytes.as_deref(),
                src.as_deref(),
                meta.as_ref(),
            )
        })
        .await?;

        // HNSW insert
        #[cfg(feature = "hnsw")]
        {
            let key = format!("fact:{}", fact_id);
            self.inner.hnsw_index.read().unwrap().insert(key, embedding)?;
        }

        Ok(fact_id)
    }

    /// Update a fact's content. Re-embeds automatically.
    pub async fn update_fact(&self, fact_id: &str, content: &str) -> Result<(), MemoryError> {
        let embedding = self.inner.embedder.embed(content).await?;
        let embedding_bytes = db::embedding_to_bytes(&embedding);

        let fid = fact_id.to_string();
        let ct = content.to_string();
        self.with_conn(move |conn| {
            knowledge::update_fact_with_fts(conn, &fid, &ct, &embedding_bytes)
        })
        .await?;

        // HNSW update
        #[cfg(feature = "hnsw")]
        {
            let key = format!("fact:{}", fact_id);
            self.inner.hnsw_index.read().unwrap().update(key, &embedding)?;
        }

        Ok(())
    }

    /// Delete a fact by ID.
    pub async fn delete_fact(&self, fact_id: &str) -> Result<(), MemoryError> {
        let fid = fact_id.to_string();
        self.with_conn(move |conn| knowledge::delete_fact_with_fts(conn, &fid))
            .await?;

        // HNSW delete
        #[cfg(feature = "hnsw")]
        {
            let key = format!("fact:{}", fact_id);
            self.inner.hnsw_index.read().unwrap().delete(&key)?;
        }

        Ok(())
    }

    /// Delete all facts in a namespace. Returns the count of deleted facts.
    pub async fn delete_namespace(&self, namespace: &str) -> Result<usize, MemoryError> {
        let ns = namespace.to_string();

        // Get fact IDs before deleting (for HNSW cleanup)
        #[cfg(feature = "hnsw")]
        let fact_ids: Vec<String> = {
            let ns_clone = ns.clone();
            self.with_conn(move |conn| {
                let mut stmt = conn.prepare("SELECT id FROM facts WHERE namespace = ?1")?;
                let ids = stmt
                    .query_map(rusqlite::params![ns_clone], |row| row.get(0))?
                    .collect::<Result<Vec<String>, _>>()?;
                Ok(ids)
            })
            .await?
        };

        let count = self
            .with_conn(move |conn| knowledge::delete_namespace(conn, &ns))
            .await?;

        // HNSW delete
        #[cfg(feature = "hnsw")]
        {
            for fact_id in &fact_ids {
                let key = format!("fact:{}", fact_id);
                self.inner.hnsw_index.read().unwrap().delete(&key)?;
            }
        }

        Ok(count)
    }

    /// Get a fact by ID.
    pub async fn get_fact(&self, fact_id: &str) -> Result<Option<Fact>, MemoryError> {
        let fid = fact_id.to_string();
        self.with_conn(move |conn| knowledge::get_fact(conn, &fid))
            .await
    }

    /// Get a fact's embedding vector.
    pub async fn get_fact_embedding(&self, fact_id: &str) -> Result<Option<Vec<f32>>, MemoryError> {
        let fid = fact_id.to_string();
        self.with_conn(move |conn| knowledge::get_fact_embedding(conn, &fid))
            .await
    }

    /// List all facts in a namespace.
    pub async fn list_facts(
        &self,
        namespace: &str,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Fact>, MemoryError> {
        let ns = namespace.to_string();
        self.with_conn(move |conn| knowledge::list_facts(conn, &ns, limit, offset))
            .await
    }

    // ─── Document Ingestion ─────────────────────────────────────

    /// Ingest a document: chunk, embed all chunks, store everything.
    pub async fn ingest_document(
        &self,
        title: &str,
        content: &str,
        namespace: &str,
        source_path: Option<&str>,
        metadata: Option<serde_json::Value>,
    ) -> Result<String, MemoryError> {
        let text_chunks = chunker::chunk_text(
            content,
            &self.inner.config.chunking,
            self.inner.token_counter.as_ref(),
        );

        let chunk_texts: Vec<String> = text_chunks.iter().map(|c| c.content.clone()).collect();
        let embeddings = self.inner.embedder.embed_batch(chunk_texts).await?;

        let quantizer = Quantizer::new(self.inner.config.embedding.dimensions);
        let chunks: Vec<documents::ChunkRow> = text_chunks
            .iter()
            .zip(embeddings.iter())
            .map(|(tc, emb)| {
                let q8 = quantizer.quantize(emb)
                    .map(|qv| quantize::pack_quantized(&qv))
                    .ok();
                (
                    tc.content.clone(),
                    db::embedding_to_bytes(emb),
                    q8,
                    tc.token_count_estimate,
                )
            })
            .collect();

        let doc_id = uuid::Uuid::new_v4().to_string();

        let did = doc_id.clone();
        let t = title.to_string();
        let ns = namespace.to_string();
        let sp = source_path.map(|s| s.to_string());
        let meta = metadata;

        // We need chunk IDs for HNSW, so get them from the insert
        #[cfg(feature = "hnsw")]
        let chunk_ids: Vec<String> = {
            // Generate chunk IDs ahead of time so we know them for HNSW
            let chunk_ids: Vec<String> = (0..chunks.len())
                .map(|_| uuid::Uuid::new_v4().to_string())
                .collect();
            let cids = chunk_ids.clone();

            let did_clone = did.clone();
            self.with_conn(move |conn| {
                documents::insert_document_with_chunks_and_ids(
                    conn,
                    &did_clone,
                    &t,
                    &ns,
                    sp.as_deref(),
                    meta.as_ref(),
                    &chunks,
                    &cids,
                )
            })
            .await?;

            chunk_ids
        };

        #[cfg(not(feature = "hnsw"))]
        {
            self.with_conn(move |conn| {
                documents::insert_document_with_chunks(
                    conn,
                    &did,
                    &t,
                    &ns,
                    sp.as_deref(),
                    meta.as_ref(),
                    &chunks,
                )
            })
            .await?;
        }

        // HNSW insert for each chunk
        #[cfg(feature = "hnsw")]
        {
            for (chunk_id, embedding) in chunk_ids.iter().zip(embeddings.iter()) {
                let key = format!("chunk:{}", chunk_id);
                self.inner.hnsw_index.read().unwrap().insert(key, embedding)?;
            }
        }

        Ok(doc_id)
    }

    /// Delete a document and all its chunks.
    pub async fn delete_document(&self, document_id: &str) -> Result<(), MemoryError> {
        // Get chunk IDs before deleting (for HNSW cleanup)
        #[cfg(feature = "hnsw")]
        let chunk_ids: Vec<String> = {
            let did = document_id.to_string();
            self.with_conn(move |conn| {
                let mut stmt =
                    conn.prepare("SELECT id FROM chunks WHERE document_id = ?1")?;
                let ids = stmt
                    .query_map(rusqlite::params![did], |row| row.get(0))?
                    .collect::<Result<Vec<String>, _>>()?;
                Ok(ids)
            })
            .await?
        };

        let did = document_id.to_string();
        self.with_conn(move |conn| documents::delete_document_with_chunks(conn, &did))
            .await?;

        // HNSW delete
        #[cfg(feature = "hnsw")]
        {
            for chunk_id in &chunk_ids {
                let key = format!("chunk:{}", chunk_id);
                self.inner.hnsw_index.read().unwrap().delete(&key)?;
            }
        }

        Ok(())
    }

    /// List documents in a namespace.
    pub async fn list_documents(
        &self,
        namespace: &str,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Document>, MemoryError> {
        let ns = namespace.to_string();
        self.with_conn(move |conn| documents::list_documents(conn, &ns, limit, offset))
            .await
    }

    // ─── Search ─────────────────────────────────────────────────

    /// Hybrid search across facts and document chunks.
    pub async fn search(
        &self,
        query: &str,
        top_k: Option<usize>,
        namespaces: Option<&[&str]>,
        source_types: Option<&[SearchSourceType]>,
    ) -> Result<Vec<SearchResult>, MemoryError> {
        let k = top_k.unwrap_or(self.inner.config.search.default_top_k);

        let query_embedding = self.inner.embedder.embed(query).await?;

        // HNSW-based vector search
        #[cfg(feature = "hnsw")]
        let hnsw_hits = {
            let guard = self.inner.hnsw_index.read().unwrap();
            if guard.needs_compaction() {
                tracing::warn!(
                    deleted_ratio = guard.deleted_ratio(),
                    "HNSW index has high tombstone ratio. Call compact_hnsw() to reclaim."
                );
            }
            let candidates = k * 3;
            guard.search(&query_embedding, candidates)?
        };

        let q = query.to_string();
        let config = self.inner.config.search.clone();
        let ns_owned = to_owned_string_vec(namespaces);
        let st_owned: Option<Vec<SearchSourceType>> = source_types.map(|s| s.to_vec());

        #[cfg(feature = "hnsw")]
        let hnsw_hits_owned = hnsw_hits;

        self.with_conn(move |conn| {
            if db::is_embeddings_dirty(conn)? {
                tracing::warn!(
                    "Embeddings are stale after model change — search quality is degraded. \
                     Call reembed_all() to regenerate embeddings."
                );
            }
            let ns_refs = as_str_slice(&ns_owned);
            let ns_slice: Option<&[&str]> = ns_refs.as_deref();
            let st_slice: Option<&[SearchSourceType]> = st_owned.as_deref();

            #[cfg(feature = "hnsw")]
            {
                search::hybrid_search_with_hnsw(
                    conn,
                    &q,
                    &query_embedding,
                    &config,
                    k,
                    ns_slice,
                    st_slice,
                    None,
                    &hnsw_hits_owned,
                )
            }
            #[cfg(not(feature = "hnsw"))]
            {
                search::hybrid_search(
                    conn,
                    &q,
                    &query_embedding,
                    &config,
                    k,
                    ns_slice,
                    st_slice,
                    None,
                )
            }
        })
        .await
    }

    /// Full-text search only (no embeddings needed).
    pub async fn search_fts_only(
        &self,
        query: &str,
        top_k: Option<usize>,
        namespaces: Option<&[&str]>,
        source_types: Option<&[SearchSourceType]>,
    ) -> Result<Vec<SearchResult>, MemoryError> {
        let k = top_k.unwrap_or(self.inner.config.search.default_top_k);
        let q = query.to_string();
        let config = self.inner.config.search.clone();
        let ns_owned = to_owned_string_vec(namespaces);
        let st_owned: Option<Vec<SearchSourceType>> = source_types.map(|s| s.to_vec());
        self.with_conn(move |conn| {
            let ns_refs = as_str_slice(&ns_owned);
            let ns_slice: Option<&[&str]> = ns_refs.as_deref();
            let st_slice: Option<&[SearchSourceType]> = st_owned.as_deref();
            search::fts_only_search(conn, &q, &config, k, ns_slice, st_slice, None)
        })
        .await
    }

    /// Vector similarity search only (no FTS).
    pub async fn search_vector_only(
        &self,
        query: &str,
        top_k: Option<usize>,
        namespaces: Option<&[&str]>,
        source_types: Option<&[SearchSourceType]>,
    ) -> Result<Vec<SearchResult>, MemoryError> {
        let k = top_k.unwrap_or(self.inner.config.search.default_top_k);
        let query_embedding = self.inner.embedder.embed(query).await?;

        // HNSW-based vector search
        #[cfg(feature = "hnsw")]
        let hnsw_hits = {
            let candidates = k * 3;
            self.inner.hnsw_index.read().unwrap().search(&query_embedding, candidates)?
        };

        let config = self.inner.config.search.clone();
        let ns_owned = to_owned_string_vec(namespaces);
        let st_owned: Option<Vec<SearchSourceType>> = source_types.map(|s| s.to_vec());

        #[cfg(feature = "hnsw")]
        let hnsw_hits_owned = hnsw_hits;

        self.with_conn(move |conn| {
            if db::is_embeddings_dirty(conn)? {
                tracing::warn!(
                    "Embeddings are stale after model change — search quality is degraded. \
                     Call reembed_all() to regenerate embeddings."
                );
            }
            let ns_refs = as_str_slice(&ns_owned);
            let ns_slice: Option<&[&str]> = ns_refs.as_deref();
            let st_slice: Option<&[SearchSourceType]> = st_owned.as_deref();

            #[cfg(feature = "hnsw")]
            {
                search::vector_only_search_with_hnsw(
                    conn, &config, k, ns_slice, st_slice, None, &hnsw_hits_owned,
                )
            }
            #[cfg(not(feature = "hnsw"))]
            {
                search::vector_only_search(conn, &query_embedding, &config, k, ns_slice, st_slice, None)
            }
        })
        .await
    }

    // ─── Conversation Search ───────────────────────────────────

    /// Append a message to a session with automatic embedding.
    pub async fn add_message_embedded(
        &self,
        session_id: &str,
        role: Role,
        content: &str,
        token_count: Option<u32>,
        metadata: Option<serde_json::Value>,
    ) -> Result<i64, MemoryError> {
        let effective_token_count =
            token_count.or_else(|| Some(self.inner.token_counter.count_tokens(content) as u32));

        let embedding = self.inner.embedder.embed(content).await?;
        let embedding_bytes = db::embedding_to_bytes(&embedding);

        // Quantize for storage
        let quantizer = Quantizer::new(self.inner.config.embedding.dimensions);
        let q8_bytes = quantizer.quantize(&embedding)
            .map(|qv| quantize::pack_quantized(&qv))
            .ok();

        let sid = session_id.to_string();
        let ct = content.to_string();
        let meta = metadata;
        let msg_id = self
            .with_conn(move |conn| {
                conversation::add_message_with_embedding_q8(
                    conn,
                    &sid,
                    role,
                    &ct,
                    effective_token_count,
                    meta.as_ref(),
                    &embedding_bytes,
                    q8_bytes.as_deref(),
                )
            })
            .await?;

        // HNSW insert
        #[cfg(feature = "hnsw")]
        {
            let key = format!("msg:{}", msg_id);
            self.inner.hnsw_index.read().unwrap().insert(key, &embedding)?;
        }

        Ok(msg_id)
    }

    /// Hybrid search over conversation messages only.
    pub async fn search_conversations(
        &self,
        query: &str,
        top_k: Option<usize>,
        session_ids: Option<&[&str]>,
    ) -> Result<Vec<SearchResult>, MemoryError> {
        let k = top_k.unwrap_or(self.inner.config.search.default_top_k);

        let query_embedding = self.inner.embedder.embed(query).await?;

        let q = query.to_string();
        let config = self.inner.config.search.clone();
        let sids_owned = to_owned_string_vec(session_ids);
        self.with_conn(move |conn| {
            let sids_refs = as_str_slice(&sids_owned);
            let sids_slice: Option<&[&str]> = sids_refs.as_deref();
            search::hybrid_search(
                conn,
                &q,
                &query_embedding,
                &config,
                k,
                None,
                Some(&[SearchSourceType::Messages]),
                sids_slice,
            )
        })
        .await
    }

    // ─── Utility ────────────────────────────────────────────────

    /// Chunk text using the configured strategy and token counter.
    pub fn chunk_text(&self, text: &str) -> Vec<TextChunk> {
        chunker::chunk_text(
            text,
            &self.inner.config.chunking,
            self.inner.token_counter.as_ref(),
        )
    }

    /// Embed a single text via the configured provider.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, MemoryError> {
        self.inner.embedder.embed(text).await
    }

    /// Embed multiple texts in a batch.
    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, MemoryError> {
        let owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        self.inner.embedder.embed_batch(owned).await
    }

    /// Get database statistics.
    pub async fn stats(&self) -> Result<MemoryStats, MemoryError> {
        let db_path = self.inner.paths.sqlite_path.clone();
        self.with_conn(move |conn| {
            let total_facts: u64 =
                conn.query_row("SELECT COUNT(*) FROM facts", [], |r| r.get(0))?;
            let total_documents: u64 =
                conn.query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0))?;
            let total_chunks: u64 =
                conn.query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))?;
            let total_sessions: u64 =
                conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))?;
            let total_messages: u64 =
                conn.query_row("SELECT COUNT(*) FROM messages", [], |r| r.get(0))?;

            let db_size = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);

            let (model, dims): (Option<String>, Option<usize>) = conn
                .query_row(
                    "SELECT model_name, dimensions FROM embedding_metadata WHERE id = 1",
                    [],
                    |r| Ok((Some(r.get(0)?), Some(r.get(1)?))),
                )
                .unwrap_or((None, None));

            Ok(MemoryStats {
                total_facts,
                total_documents,
                total_chunks,
                total_sessions,
                total_messages,
                database_size_bytes: db_size,
                embedding_model: model,
                embedding_dimensions: dims,
            })
        })
        .await
    }

    /// Check if embeddings need re-generation after a model change.
    pub async fn embeddings_are_dirty(&self) -> Result<bool, MemoryError> {
        self.with_conn(db::is_embeddings_dirty).await
    }

    /// Re-embed all facts, chunks, and messages. Call after changing embedding models.
    pub async fn reembed_all(&self) -> Result<usize, MemoryError> {
        let mut count = 0usize;
        let batch_size = self.inner.config.embedding.batch_size;
        let dims = self.inner.config.embedding.dimensions;

        // ─── Facts ──────────────────────────────────────────────────
        let fact_contents: Vec<(String, String)> = self
            .with_conn(|conn| {
                let mut stmt = conn.prepare("SELECT id, content FROM facts")?;
                let result = stmt
                    .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(result)
            })
            .await?;

        let mut fact_count = 0usize;
        for batch in fact_contents.chunks(batch_size) {
            let texts: Vec<String> = batch.iter().map(|(_, c)| c.clone()).collect();
            let embeddings = self.inner.embedder.embed_batch(texts).await?;

            let quantizer = Quantizer::new(dims);
            let updates: Vec<(String, Vec<u8>, Option<Vec<u8>>)> = batch
                .iter()
                .zip(embeddings.iter())
                .map(|((id, _), emb)| {
                    let q8 = quantizer.quantize(emb)
                        .map(|qv| quantize::pack_quantized(&qv))
                        .ok();
                    (id.clone(), db::embedding_to_bytes(emb), q8)
                })
                .collect();

            self.with_conn(move |conn| {
                db::with_transaction(conn, |tx| {
                    for (fid, bytes, q8) in &updates {
                        tx.execute(
                            "UPDATE facts SET embedding = ?1, embedding_q8 = ?2, updated_at = datetime('now') WHERE id = ?3",
                            rusqlite::params![bytes, q8.as_deref(), fid],
                        )?;
                    }
                    Ok(())
                })
            })
            .await?;

            fact_count += batch.len();
            count += batch.len();
            if fact_count % 100 < batch_size {
                tracing::info!(fact_count, "Re-embedded {} facts so far", fact_count);
            }
        }

        // ─── Chunks ─────────────────────────────────────────────────
        let chunk_data: Vec<(String, String)> = self
            .with_conn(|conn| {
                let mut stmt = conn.prepare("SELECT id, content FROM chunks")?;
                let result = stmt
                    .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(result)
            })
            .await?;

        let mut chunk_count = 0usize;
        for batch in chunk_data.chunks(batch_size) {
            let texts: Vec<String> = batch.iter().map(|(_, c)| c.clone()).collect();
            let embeddings = self.inner.embedder.embed_batch(texts).await?;

            let quantizer = Quantizer::new(dims);
            let updates: Vec<(String, Vec<u8>, Option<Vec<u8>>)> = batch
                .iter()
                .zip(embeddings.iter())
                .map(|((id, _), emb)| {
                    let q8 = quantizer.quantize(emb)
                        .map(|qv| quantize::pack_quantized(&qv))
                        .ok();
                    (id.clone(), db::embedding_to_bytes(emb), q8)
                })
                .collect();

            self.with_conn(move |conn| {
                db::with_transaction(conn, |tx| {
                    for (cid, bytes, q8) in &updates {
                        tx.execute(
                            "UPDATE chunks SET embedding = ?1, embedding_q8 = ?2 WHERE id = ?3",
                            rusqlite::params![bytes, q8.as_deref(), cid],
                        )?;
                    }
                    Ok(())
                })
            })
            .await?;

            chunk_count += batch.len();
            count += batch.len();
            if chunk_count % 100 < batch_size {
                tracing::info!(chunk_count, "Re-embedded {} chunks so far", chunk_count);
            }
        }

        // ─── Messages ───────────────────────────────────────────────
        let message_data: Vec<(i64, String)> = self
            .with_conn(|conn| {
                let mut stmt =
                    conn.prepare("SELECT id, content FROM messages WHERE embedding IS NOT NULL")?;
                let result = stmt
                    .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(result)
            })
            .await?;

        let mut msg_count = 0usize;
        for batch in message_data.chunks(batch_size) {
            let texts: Vec<String> = batch.iter().map(|(_, c)| c.clone()).collect();
            let embeddings = self.inner.embedder.embed_batch(texts).await?;

            let quantizer = Quantizer::new(dims);
            let updates: Vec<(i64, Vec<u8>, Option<Vec<u8>>)> = batch
                .iter()
                .zip(embeddings.iter())
                .map(|((id, _), emb)| {
                    let q8 = quantizer.quantize(emb)
                        .map(|qv| quantize::pack_quantized(&qv))
                        .ok();
                    (*id, db::embedding_to_bytes(emb), q8)
                })
                .collect();

            self.with_conn(move |conn| {
                db::with_transaction(conn, |tx| {
                    for (mid, bytes, q8) in &updates {
                        tx.execute(
                            "UPDATE messages SET embedding = ?1, embedding_q8 = ?2 WHERE id = ?3",
                            rusqlite::params![bytes, q8.as_deref(), mid],
                        )?;
                    }
                    Ok(())
                })
            })
            .await?;

            msg_count += batch.len();
            count += batch.len();
            if msg_count % 100 < batch_size {
                tracing::info!(msg_count, "Re-embedded {} messages so far", msg_count);
            }
        }

        // Clear the dirty flag
        self.with_conn(db::clear_embeddings_dirty).await?;

        tracing::info!(
            facts = fact_count,
            chunks = chunk_count,
            messages = msg_count,
            total = count,
            "Re-embedding complete"
        );

        // Rebuild HNSW after re-embedding
        #[cfg(feature = "hnsw")]
        {
            tracing::info!("Rebuilding HNSW index after re-embedding...");
            self.rebuild_hnsw_index().await?;
        }

        Ok(count)
    }

    /// Vacuum the database (reclaim space after deletions).
    pub async fn vacuum(&self) -> Result<(), MemoryError> {
        self.with_conn(|conn| {
            conn.execute_batch("VACUUM")?;
            Ok(())
        })
        .await
    }

    /// Execute raw SQL. For testing only — not part of the stable public API.
    #[cfg(any(test, feature = "testing"))]
    pub async fn raw_execute(&self, sql: &str, params: Vec<String>) -> Result<usize, MemoryError> {
        let sql = sql.to_string();
        self.with_conn(move |conn| {
            let param_refs: Vec<&dyn rusqlite::types::ToSql> = params
                .iter()
                .map(|s| s as &dyn rusqlite::types::ToSql)
                .collect();
            Ok(conn.execute(&sql, &*param_refs)?)
        })
        .await
    }
}
