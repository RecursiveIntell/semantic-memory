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
pub use config::{
    ChunkingConfig, EmbeddingConfig, MemoryConfig, MemoryLimits, PoolConfig, SearchConfig,
};
pub use db::{IntegrityReport, ReconcileAction, VerifyMode};
pub use embedder::{Embedder, MockEmbedder, OllamaEmbedder};
pub use error::MemoryError;
#[cfg(feature = "hnsw")]
pub use hnsw::{HnswConfig, HnswHit, HnswIndex};
pub use quantize::{pack_quantized, unpack_quantized, QuantizedVector, Quantizer};
pub use storage::StoragePaths;
pub use tokenizer::{EstimateTokenCounter, TokenCounter};
pub use types::{
    Document, EmbeddingDisplacement, EpisodeMeta, EpisodeOutcome, ExplainedResult, Fact,
    GraphDirection, GraphEdge, GraphEdgeType, GraphView, MemoryStats, Message, Role,
    ScoreBreakdown, SearchResult, SearchSource, SearchSourceType, Session, TextChunk,
    VerificationStatus,
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
            Err(_) => {
                tracing::warn!("HNSW save panicked on drop (directory may have been removed)")
            }
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

/// Rebuild an HNSW index from all embeddings stored in SQLite.
///
/// Used during startup when sidecar files are missing (FIX-9) or stale (FIX-10).
/// This is a synchronous function since it runs inside the blocking `open()` path.
#[cfg(feature = "hnsw")]
fn rebuild_hnsw_from_sqlite(
    conn: &rusqlite::Connection,
    config: &HnswConfig,
) -> Result<HnswIndex, MemoryError> {
    let new_index = HnswIndex::new(config.clone())?;

    // Load fact embeddings
    {
        let mut stmt =
            conn.prepare("SELECT id, embedding FROM facts WHERE embedding IS NOT NULL")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
        })?;
        for row in rows {
            let (id, blob) = row?;
            if let Ok(emb) = db::bytes_to_embedding(&blob) {
                let key = format!("fact:{}", id);
                if let Err(e) = new_index.insert(key.clone(), &emb) {
                    tracing::warn!("Failed to insert {} into HNSW: {}", key, e);
                }
            }
        }
    }

    // Load chunk embeddings
    {
        let mut stmt =
            conn.prepare("SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
        })?;
        for row in rows {
            let (id, blob) = row?;
            if let Ok(emb) = db::bytes_to_embedding(&blob) {
                let key = format!("chunk:{}", id);
                if let Err(e) = new_index.insert(key.clone(), &emb) {
                    tracing::warn!("Failed to insert {} into HNSW: {}", key, e);
                }
            }
        }
    }

    // Load message embeddings
    {
        let mut stmt =
            conn.prepare("SELECT id, embedding FROM messages WHERE embedding IS NOT NULL")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
        })?;
        for row in rows {
            let (id, blob) = row?;
            if let Ok(emb) = db::bytes_to_embedding(&blob) {
                let key = format!("msg:{}", id);
                if let Err(e) = new_index.insert(key.clone(), &emb) {
                    tracing::warn!("Failed to insert {} into HNSW: {}", key, e);
                }
            }
        }
    }

    Ok(new_index)
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
            let conn = inner.conn.lock().unwrap_or_else(|e| e.into_inner());
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

        let conn = db::open_database(&paths.sqlite_path, &config.pool)?;
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

                        // Stale index detection: compare HNSW entry count vs SQLite
                        // embedding count. A mismatch means the app crashed before
                        // flushing HNSW, or keys were lost.
                        let hnsw_count = index.len();
                        let sqlite_count: i64 = conn
                            .query_row(
                                "SELECT (SELECT COUNT(*) FROM facts WHERE embedding IS NOT NULL) +
                                    (SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL) +
                                    (SELECT COUNT(*) FROM messages WHERE embedding IS NOT NULL)",
                                [],
                                |row| row.get(0),
                            )
                            .unwrap_or(0);

                        let drift = (sqlite_count as i64 - hnsw_count as i64).abs();
                        if drift > 0 {
                            tracing::warn!(
                                hnsw_count,
                                sqlite_count,
                                drift,
                                "HNSW index is stale — {} entries differ from SQLite. \
                                 Likely caused by unclean shutdown. Triggering inline rebuild.",
                                drift
                            );
                            // Discard the stale index and rebuild from SQLite
                            let rebuilt = rebuild_hnsw_from_sqlite(&conn, &hnsw_config)?;
                            tracing::info!(
                                active = rebuilt.len(),
                                "HNSW index rebuilt after stale detection"
                            );
                            rebuilt
                        } else {
                            tracing::info!(
                                "HNSW index loaded ({} active keys, in sync with SQLite)",
                                hnsw_count
                            );
                            index
                        }
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
                // Check if SQLite has embeddings that should be in the index.
                // This happens when: sidecar files were deleted, data dir was
                // partially copied, app crashed before first flush, or HNSW was
                // added after data already existed.
                let orphan_count: i64 = conn
                    .query_row(
                        "SELECT (SELECT COUNT(*) FROM facts WHERE embedding IS NOT NULL) +
                            (SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL) +
                            (SELECT COUNT(*) FROM messages WHERE embedding IS NOT NULL)",
                        [],
                        |row| row.get(0),
                    )
                    .unwrap_or(0);

                if orphan_count > 0 {
                    tracing::warn!(
                        orphan_count,
                        "HNSW sidecar files missing but {} embeddings exist in SQLite — \
                         rebuilding index inline",
                        orphan_count
                    );
                    let new_index = rebuild_hnsw_from_sqlite(&conn, &hnsw_config)?;
                    tracing::info!(
                        active = new_index.len(),
                        "HNSW index rebuilt from SQLite embeddings"
                    );
                    new_index
                } else {
                    tracing::info!("Creating new empty HNSW index (no embeddings in SQLite)");
                    HnswIndex::new(hnsw_config)?
                }
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
                let mut stmt =
                    conn.prepare("SELECT id, embedding FROM messages WHERE embedding IS NOT NULL")?;
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
            let mut guard = self
                .inner
                .hnsw_index
                .write()
                .unwrap_or_else(|e| e.into_inner());
            *guard = new_index;
        }

        // Persist the new index (read lock is fine now)
        {
            let guard = self
                .inner
                .hnsw_index
                .read()
                .unwrap_or_else(|e| e.into_inner());
            guard.save(&self.inner.paths.hnsw_dir, &self.inner.paths.hnsw_basename)?;
            let conn = self.inner.conn.lock().unwrap_or_else(|e| e.into_inner());
            guard.flush_keymap(&conn)?;
        }

        Ok(())
    }

    /// Opportunistically flush HNSW if the configured interval has elapsed.
    ///
    /// Cheap no-op when `flush_interval_secs` is None or the interval hasn't
    /// elapsed yet (just an atomic load + epoch comparison).
    #[cfg(feature = "hnsw")]
    fn maybe_flush_hnsw(&self) {
        if let Some(interval) = self.inner.config.hnsw.flush_interval_secs {
            let guard = self
                .inner
                .hnsw_index
                .read()
                .unwrap_or_else(|e| e.into_inner());
            if guard.should_flush(interval) {
                drop(guard); // release read lock before flushing
                if let Err(e) = self.flush_hnsw() {
                    tracing::warn!("Opportunistic HNSW flush failed: {}", e);
                } else {
                    let guard = self
                        .inner
                        .hnsw_index
                        .read()
                        .unwrap_or_else(|e| e.into_inner());
                    guard.update_last_flush_epoch();
                    tracing::info!("Opportunistic HNSW flush completed");
                }
            }
        }
    }

    /// Persist the HNSW graph, vector data, and key mappings to disk.
    ///
    /// Called automatically on drop, but can be called explicitly for durability.
    #[cfg(feature = "hnsw")]
    pub fn flush_hnsw(&self) -> Result<(), MemoryError> {
        let guard = self
            .inner
            .hnsw_index
            .read()
            .unwrap_or_else(|e| e.into_inner());
        guard.save(&self.inner.paths.hnsw_dir, &self.inner.paths.hnsw_basename)?;

        // Flush key mappings to SQLite
        let conn = self.inner.conn.lock().unwrap_or_else(|e| e.into_inner());
        guard.flush_keymap(&conn)?;
        Ok(())
    }

    /// Compact the HNSW index by rebuilding without tombstones.
    ///
    /// Only rebuilds if the deleted ratio exceeds the compaction threshold.
    #[cfg(feature = "hnsw")]
    pub async fn compact_hnsw(&self) -> Result<(), MemoryError> {
        if !self
            .inner
            .hnsw_index
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .needs_compaction()
        {
            tracing::info!("HNSW compaction not needed (deleted ratio below threshold)");
            return Ok(());
        }
        self.rebuild_hnsw_index().await
    }

    // ─── Integrity & Diagnostics ────────────────────────────────

    /// Verify database integrity.
    ///
    /// In `Quick` mode, checks table existence and row counts.
    /// In `Full` mode, also verifies FTS consistency and runs SQLite integrity_check.
    pub async fn verify_integrity(
        &self,
        mode: db::VerifyMode,
    ) -> Result<db::IntegrityReport, MemoryError> {
        self.with_conn(move |conn| db::verify_integrity_sync(conn, mode))
            .await
    }

    /// Reconcile detected integrity issues.
    ///
    /// - `ReportOnly`: no-op, just returns the integrity report.
    /// - `RebuildFts`: rebuilds all FTS indexes from source data.
    /// - `ReEmbed`: not yet implemented (requires async embedding calls).
    pub async fn reconcile(
        &self,
        action: db::ReconcileAction,
    ) -> Result<db::IntegrityReport, MemoryError> {
        match action {
            db::ReconcileAction::ReportOnly => self.verify_integrity(db::VerifyMode::Full).await,
            db::ReconcileAction::RebuildFts => {
                self.with_conn(db::reconcile_fts).await?;
                self.verify_integrity(db::VerifyMode::Full).await
            }
            db::ReconcileAction::ReEmbed => {
                // Re-embedding requires the embedder (async). For now, just report.
                tracing::warn!("ReEmbed action not yet implemented — reporting only");
                self.verify_integrity(db::VerifyMode::Full).await
            }
        }
    }

    /// Get the current configuration.
    pub fn config(&self) -> &MemoryConfig {
        &self.inner.config
    }

    // ─── Session Management ─────────────────────────────────────

    /// Create a new conversation session. Returns the session ID (UUID v4).
    pub async fn create_session(&self, channel: &str) -> Result<String, MemoryError> {
        let channel = channel.to_string();
        self.with_conn(move |conn| conversation::create_session(conn, &channel, None))
            .await
    }

    /// Rename a session's channel (display name).
    pub async fn rename_session(
        &self,
        session_id: &str,
        new_channel: &str,
    ) -> Result<(), MemoryError> {
        let sid = session_id.to_string();
        let ch = new_channel.to_string();
        self.with_conn(move |conn| conversation::rename_session(conn, &sid, &ch))
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
    ///
    /// Cleans up HNSW entries for embedded messages before CASCADE delete.
    pub async fn delete_session(&self, session_id: &str) -> Result<(), MemoryError> {
        // Collect message IDs with embeddings for HNSW cleanup
        #[cfg(feature = "hnsw")]
        let message_ids: Vec<i64> = {
            let sid = session_id.to_string();
            self.with_conn(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT id FROM messages WHERE session_id = ?1 AND embedding IS NOT NULL",
                )?;
                let ids = stmt
                    .query_map(rusqlite::params![sid], |row| row.get(0))?
                    .collect::<Result<Vec<i64>, _>>()?;
                Ok(ids)
            })
            .await?
        };

        // Delete session (CASCADE handles messages, FTS cleanup inside transaction)
        let sid = session_id.to_string();
        self.with_conn(move |conn| conversation::delete_session(conn, &sid))
            .await?;

        // Remove orphaned HNSW entries
        #[cfg(feature = "hnsw")]
        {
            let guard = self
                .inner
                .hnsw_index
                .read()
                .unwrap_or_else(|e| e.into_inner());
            for msg_id in &message_ids {
                let key = format!("msg:{}", msg_id);
                if let Err(e) = guard.delete(&key) {
                    tracing::warn!("Failed to remove HNSW entry {}: {}", key, e);
                }
            }
        }

        Ok(())
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
        // Enforce content size limit
        let limits = &self.inner.config.limits;
        if content.len() > limits.max_content_bytes {
            return Err(MemoryError::ContentTooLarge {
                size: content.len(),
                limit: limits.max_content_bytes,
            });
        }

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

    /// Append a message to a session with FTS indexing but no embedding.
    ///
    /// Fallback path when embedding fails: messages still appear in conversation
    /// history and are findable via BM25 search, just not via vector search.
    pub async fn add_message_fts(
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
            conversation::add_message_with_fts(
                conn,
                &sid,
                role,
                &ct,
                effective_token_count,
                meta.as_ref(),
            )
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
        // Enforce content size limit
        let limits = &self.inner.config.limits;
        if content.len() > limits.max_content_bytes {
            return Err(MemoryError::ContentTooLarge {
                size: content.len(),
                limit: limits.max_content_bytes,
            });
        }

        // Enforce namespace fact count limit
        let ns_check = namespace.to_string();
        let max_facts = limits.max_facts_per_namespace;
        let count: usize = self
            .with_conn(move |conn| {
                let c: usize = conn
                    .query_row(
                        "SELECT COUNT(*) FROM facts WHERE namespace = ?1",
                        rusqlite::params![ns_check],
                        |row| row.get(0),
                    )
                    .unwrap_or(0);
                Ok(c)
            })
            .await?;
        if count >= max_facts {
            return Err(MemoryError::NamespaceFull {
                namespace: namespace.to_string(),
                count,
                limit: max_facts,
            });
        }

        let embedding = self.inner.embedder.embed(content).await?;
        let embedding_bytes = db::embedding_to_bytes(&embedding);
        let fact_id = uuid::Uuid::new_v4().to_string();

        // Quantize for storage
        let quantizer = Quantizer::new(self.inner.config.embedding.dimensions);
        let q8_bytes = quantizer
            .quantize(&embedding)
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
            self.inner
                .hnsw_index
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .insert(key, &embedding)?;
            self.maybe_flush_hnsw();
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
        let q8_bytes = quantizer
            .quantize(embedding)
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
            self.inner
                .hnsw_index
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .insert(key, embedding)?;
            self.maybe_flush_hnsw();
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
            self.inner
                .hnsw_index
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .update(key, &embedding)?;
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
            self.inner
                .hnsw_index
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .delete(&key)?;
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
                self.inner
                    .hnsw_index
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .delete(&key)?;
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

        let max_chunks = self.inner.config.limits.max_chunks_per_document;
        if text_chunks.len() > max_chunks {
            return Err(MemoryError::ContentTooLarge {
                size: text_chunks.len(),
                limit: max_chunks,
            });
        }

        let chunk_texts: Vec<String> = text_chunks.iter().map(|c| c.content.clone()).collect();
        let embeddings = self.inner.embedder.embed_batch(chunk_texts).await?;

        let quantizer = Quantizer::new(self.inner.config.embedding.dimensions);
        let chunks: Vec<documents::ChunkRow> = text_chunks
            .iter()
            .zip(embeddings.iter())
            .map(|(tc, emb)| {
                let q8 = quantizer
                    .quantize(emb)
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
                self.inner
                    .hnsw_index
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(key, embedding)?;
            }
            self.maybe_flush_hnsw();
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
                let mut stmt = conn.prepare("SELECT id FROM chunks WHERE document_id = ?1")?;
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
                self.inner
                    .hnsw_index
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .delete(&key)?;
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

    /// Count the number of chunks for a document.
    pub async fn count_chunks_for_document(&self, document_id: &str) -> Result<usize, MemoryError> {
        let did = document_id.to_string();
        self.with_conn(move |conn| documents::count_chunks_for_document(conn, &did))
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

        // HNSW-based vector search — non-fatal fallback to BM25-only if HNSW fails
        #[cfg(feature = "hnsw")]
        let hnsw_hits = {
            let guard = self
                .inner
                .hnsw_index
                .read()
                .unwrap_or_else(|e| e.into_inner());
            if guard.needs_compaction() {
                tracing::warn!(
                    deleted_ratio = guard.deleted_ratio(),
                    "HNSW index has high tombstone ratio. Call compact_hnsw() to reclaim."
                );
            }
            let candidates = k * 3;
            match guard.search(&query_embedding, candidates) {
                Ok(hits) => hits,
                Err(e) => {
                    tracing::error!("HNSW search failed, falling back to BM25-only: {}", e);
                    Vec::new()
                }
            }
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
            self.inner
                .hnsw_index
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .search(&query_embedding, candidates)?
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
                    conn,
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
                search::vector_only_search(
                    conn,
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
        let q8_bytes = quantizer
            .quantize(&embedding)
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
            self.inner
                .hnsw_index
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .insert(key, &embedding)?;
            self.maybe_flush_hnsw();
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

    // ─── Episodes ──────────────────────────────────────────────

    /// Ingest a causal episode attached to a document.
    ///
    /// The document must already exist. The episode is created with `Pending` outcome.
    pub async fn ingest_episode(
        &self,
        document_id: &str,
        meta: &types::EpisodeMeta,
    ) -> Result<(), MemoryError> {
        let doc_id = document_id.to_string();
        let cause_ids_json =
            serde_json::to_string(&meta.cause_ids).unwrap_or_else(|_| "[]".to_string());
        let effect_type = meta.effect_type.clone();
        let outcome = meta.outcome.as_str().to_string();
        let confidence = meta.confidence;
        let verification_json = serde_json::to_string(&meta.verification_status)
            .unwrap_or_else(|_| r#"{"status":"unverified"}"#.to_string());
        let experiment_id = meta.experiment_id.clone();

        self.with_conn(move |conn| {
            conn.execute(
                "INSERT OR REPLACE INTO episodes \
                 (document_id, cause_ids, effect_type, outcome, confidence, verification_status, experiment_id) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                rusqlite::params![
                    doc_id,
                    cause_ids_json,
                    effect_type,
                    outcome,
                    confidence,
                    verification_json,
                    experiment_id,
                ],
            )?;
            Ok(())
        })
        .await
    }

    /// Update the outcome of an existing episode.
    pub async fn update_episode_outcome(
        &self,
        document_id: &str,
        outcome: types::EpisodeOutcome,
        confidence: f32,
        experiment_id: Option<&str>,
    ) -> Result<(), MemoryError> {
        let doc_id = document_id.to_string();
        let outcome_str = outcome.as_str().to_string();
        let exp_id = experiment_id.map(|s| s.to_string());

        let updated = self
            .with_conn(move |conn| {
                let rows = conn.execute(
                    "UPDATE episodes SET outcome = ?1, confidence = ?2, experiment_id = COALESCE(?3, experiment_id) \
                     WHERE document_id = ?4",
                    rusqlite::params![outcome_str, confidence, exp_id, doc_id],
                )?;
                Ok(rows)
            })
            .await?;

        if updated == 0 {
            return Err(MemoryError::DocumentNotFound(document_id.to_string()));
        }
        Ok(())
    }

    /// Search for episodes by effect_type and/or outcome.
    pub async fn search_episodes(
        &self,
        effect_type: Option<&str>,
        outcome: Option<&types::EpisodeOutcome>,
        limit: usize,
    ) -> Result<Vec<(String, types::EpisodeMeta)>, MemoryError> {
        let et = effect_type.map(|s| s.to_string());
        let oc = outcome.map(|o| o.as_str().to_string());

        self.with_conn(move |conn| {
            let mut sql = String::from(
                "SELECT document_id, cause_ids, effect_type, outcome, confidence, \
                 verification_status, experiment_id FROM episodes WHERE 1=1",
            );
            let mut params_vec: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

            if let Some(ref et) = et {
                sql.push_str(&format!(" AND effect_type = ?{}", params_vec.len() + 1));
                params_vec.push(Box::new(et.clone()));
            }
            if let Some(ref oc) = oc {
                sql.push_str(&format!(" AND outcome = ?{}", params_vec.len() + 1));
                params_vec.push(Box::new(oc.clone()));
            }
            sql.push_str(&format!(" ORDER BY created_at DESC LIMIT {}", limit));

            let param_refs: Vec<&dyn rusqlite::types::ToSql> =
                params_vec.iter().map(|p| p.as_ref()).collect();

            let mut stmt = conn.prepare(&sql)?;
            let rows = stmt
                .query_map(&*param_refs, |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, f64>(4)?,
                        row.get::<_, String>(5)?,
                        row.get::<_, Option<String>>(6)?,
                    ))
                })?
                .collect::<Result<Vec<_>, _>>()?;

            let mut results = Vec::new();
            for (doc_id, cause_ids_json, effect_type, outcome_str, confidence, vs_json, exp_id) in
                rows
            {
                let cause_ids: Vec<String> =
                    serde_json::from_str(&cause_ids_json).unwrap_or_default();
                let outcome = types::EpisodeOutcome::from_str_value(&outcome_str)
                    .unwrap_or(types::EpisodeOutcome::Pending);
                let verification_status: types::VerificationStatus =
                    serde_json::from_str(&vs_json).unwrap_or(types::VerificationStatus::Unverified);

                results.push((
                    doc_id,
                    types::EpisodeMeta {
                        cause_ids,
                        effect_type,
                        outcome,
                        confidence: confidence as f32,
                        verification_status,
                        experiment_id: exp_id,
                    },
                ));
            }
            Ok(results)
        })
        .await
    }

    // ─── Explainable Search ───────────────────────────────────

    /// Search with full score breakdown for each result.
    pub async fn search_explained(
        &self,
        query: &str,
        top_k: Option<usize>,
        namespaces: Option<&[&str]>,
        source_types: Option<&[SearchSourceType]>,
    ) -> Result<Vec<types::ExplainedResult>, MemoryError> {
        // Delegate to the regular search and annotate with breakdowns
        let results = self.search(query, top_k, namespaces, source_types).await?;

        let explained: Vec<types::ExplainedResult> = results
            .into_iter()
            .map(|r| {
                let breakdown = types::ScoreBreakdown {
                    rrf_score: r.score,
                    bm25_score: r.bm25_rank.map(|rank| 1.0 / (60.0 + rank as f64)),
                    vector_score: r.cosine_similarity,
                    recency_score: None, // TODO: expose from search internals
                    bm25_rank: r.bm25_rank,
                    vector_rank: r.vector_rank,
                };
                types::ExplainedResult {
                    result: r,
                    breakdown,
                }
            })
            .collect();

        Ok(explained)
    }

    // ─── Embedding Displacement ───────────────────────────────

    /// Compute embedding displacement between two texts.
    pub async fn embedding_displacement(
        &self,
        text_a: &str,
        text_b: &str,
    ) -> Result<types::EmbeddingDisplacement, MemoryError> {
        let emb_a = self.inner.embedder.embed(text_a).await?;
        let emb_b = self.inner.embedder.embed(text_b).await?;
        Ok(Self::embedding_displacement_from_vecs(&emb_a, &emb_b))
    }

    /// Compute embedding displacement from pre-computed vectors.
    pub fn embedding_displacement_from_vecs(a: &[f32], b: &[f32]) -> types::EmbeddingDisplacement {
        let cosine_sim = search::cosine_similarity(a, b);

        let euclidean_dist: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt();

        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        types::EmbeddingDisplacement {
            cosine_similarity: cosine_sim,
            euclidean_distance: euclidean_dist,
            magnitude_a: mag_a,
            magnitude_b: mag_b,
        }
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
                    let q8 = quantizer
                        .quantize(emb)
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
                    let q8 = quantizer
                        .quantize(emb)
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
                    let q8 = quantizer
                        .quantize(emb)
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
