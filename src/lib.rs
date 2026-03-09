//! # semantic-memory
//!
//! Local-first semantic memory backed by authoritative SQLite state and an optional recoverable
//! HNSW sidecar.
//!
//! The crate stores facts, chunked documents, conversation messages, and searchable episodes in
//! SQLite. Search combines BM25 (FTS5) and vector retrieval with Reciprocal Rank Fusion, and
//! `search_explained()` returns the exact scoring breakdown from the live pipeline.
//!
//! Concurrency uses one writer connection plus a pool of WAL-enabled reader connections.
//! Durable writes are committed to SQLite first; any required HNSW sidecar mutations are journaled
//! in SQLite and replayed on open, flush, rebuild, or reconcile.
//!
//! `search()` targets facts, document chunks, and episodes by default. Message retrieval is
//! available through `search_conversations()` or by opting into
//! [`SearchSourceType::Messages`](crate::SearchSourceType::Messages).
//!
//! Integrity tooling is strict about malformed stored data: invalid roles, JSON, enums, embedding
//! blobs, quantized blobs, and sidecar drift are surfaced through `verify_integrity()` instead of
//! being silently converted into defaults. `reconcile()` can rebuild FTS or fully re-embed and
//! rebuild derived state from SQLite.
//!
//! `store.graph_view()` exposes a deterministic graph traversal layer over namespaces, facts,
//! documents, chunks, sessions, messages, episodes, and semantic/temporal/causal links derived
//! from SQLite state.
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
//!
//! ## Operational Notes
//!
//! - SQLite is authoritative for all durable records and embeddings.
//! - HNSW is an acceleration sidecar. Pending sidecar mutations are journaled in SQLite, so a
//!   sidecar failure does not imply the SQLite write rolled back.
//! - WAL mode plus pooled reader connections allows concurrent reads while writes serialize through
//!   the writer connection.
//! - `search_explained()` reflects the exact ranking math used by the active search pipeline,
//!   including reranking from exact f32 cosine similarity when configured.

// At least one search backend must be enabled.
#[cfg(not(any(feature = "hnsw", feature = "brute-force")))]
compile_error!("At least one search backend feature must be enabled: 'hnsw' or 'brute-force'");

pub mod chunker;
pub mod config;
pub(crate) mod conversation;
pub mod db;
pub(crate) mod documents;
pub mod embedder;
pub(crate) mod episodes;
pub mod error;
mod graph;
#[cfg(feature = "hnsw")]
pub mod hnsw;
pub(crate) mod knowledge;
mod pool;
pub mod projection_import;
pub(crate) mod projection_storage;
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
    GraphDirection, GraphEdge, GraphEdgeType, GraphView, MemoryStats, Message,
    ProjectionClaimVersion, ProjectionEntityAlias, ProjectionEpisode, ProjectionEvidenceRef,
    ProjectionQuery, ProjectionRelationVersion, Role, ScoreBreakdown, SearchResult, SearchSource,
    SearchSourceType, Session, TextChunk, VerificationStatus,
};

use forge_memory_bridge::{
    ContradictionStatus, ImportProjectionRecord, MergeDecision, ProjectionImportBatchV1,
    ReviewState, PROJECTION_IMPORT_BATCH_V1_SCHEMA,
};
use stack_ids::TraceCtx;
use std::sync::Arc;

/// Compatibility-only public access to retained legacy surfaces.
pub mod compat {
    pub mod legacy_import_envelope {
        pub use crate::projection_import::{
            ImportEnvelope, ImportReceipt, ImportRecord, ImportStatus, ProjectionFreshness,
        };
        pub use stack_ids::EnvelopeId;
    }

    #[allow(deprecated)]
    pub mod compat_trace_id {
        pub use crate::types::TraceId;
    }
}

fn encode_merge_decision(decision: &MergeDecision) -> String {
    match decision {
        MergeDecision::PendingReview => "pending_review".into(),
        _ => serde_json::to_string(decision).unwrap_or_else(|_| "\"pending_review\"".into()),
    }
}

fn encode_review_state(state: &ReviewState) -> String {
    match state {
        ReviewState::Unreviewed => "unreviewed".into(),
        ReviewState::PendingReview => "pending_review".into(),
        _ => serde_json::to_string(state).unwrap_or_else(|_| "\"unreviewed\"".into()),
    }
}

const EXPORT_ENVELOPE_V1_JSON_COMPAT: &str = "export_envelope_v1";
const JSON_COMPAT_DEFAULT_TIMESTAMP: &str = "1970-01-01T00:00:00Z";

fn json_compat_invalid(reason: impl Into<String>) -> MemoryError {
    MemoryError::ImportInvalid {
        reason: format!("invalid batch JSON: {}", reason.into()),
    }
}

fn insert_default_json_field(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    default: Option<&serde_json::Value>,
) {
    if !obj.contains_key(key) {
        if let Some(value) = default {
            obj.insert(key.to_string(), value.clone());
        }
    }
}

fn validate_json_compat_field<T>(
    obj: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<(), MemoryError>
where
    T: serde::de::DeserializeOwned,
{
    if let Some(value) = obj.get(field) {
        serde_json::from_value::<T>(value.clone())
            .map(|_| ())
            .map_err(|err| json_compat_invalid(format!("{field}: {err}")))?;
    }
    Ok(())
}

fn decode_projection_batch_json_compat(
    batch_json: &str,
) -> Result<ProjectionImportBatchV1, MemoryError> {
    let mut value: serde_json::Value =
        serde_json::from_str(batch_json).map_err(|e| json_compat_invalid(e.to_string()))?;
    let root = value
        .as_object_mut()
        .ok_or_else(|| json_compat_invalid("top-level payload must be an object"))?;

    let original_schema_version = root
        .get("schema_version")
        .and_then(|value| value.as_str())
        .map(|value| value.to_string());
    match original_schema_version.as_deref() {
        Some(EXPORT_ENVELOPE_V1_JSON_COMPAT) => {
            root.entry("export_schema_version".to_string())
                .or_insert_with(|| serde_json::json!(EXPORT_ENVELOPE_V1_JSON_COMPAT));
            root.insert(
                "schema_version".to_string(),
                serde_json::json!(PROJECTION_IMPORT_BATCH_V1_SCHEMA),
            );
        }
        Some(PROJECTION_IMPORT_BATCH_V1_SCHEMA) | None => {}
        Some(other) => {
            return Err(MemoryError::ImportInvalid {
                reason: format!(
                    "unsupported schema_version: {}; expected {} or {}",
                    other, PROJECTION_IMPORT_BATCH_V1_SCHEMA, EXPORT_ENVELOPE_V1_JSON_COMPAT
                ),
            });
        }
    }

    root.entry("source_exported_at".to_string())
        .or_insert_with(|| serde_json::json!(JSON_COMPAT_DEFAULT_TIMESTAMP));
    root.entry("transformed_at".to_string())
        .or_insert_with(|| serde_json::json!(JSON_COMPAT_DEFAULT_TIMESTAMP));

    let default_source_envelope_id = root.get("source_envelope_id").cloned();
    let default_source_authority = root.get("source_authority").cloned();
    let default_scope_key = root.get("scope_key").cloned();
    let default_trace_ctx = root.get("trace_ctx").cloned();

    if let Some(records) = root
        .get_mut("records")
        .and_then(|value| value.as_array_mut())
    {
        for record in records {
            let Some(obj) = record.as_object_mut() else {
                continue;
            };

            match obj.get("kind").and_then(|value| value.as_str()) {
                Some("claim_version") => {
                    insert_default_json_field(obj, "scope_key", default_scope_key.as_ref());
                    insert_default_json_field(
                        obj,
                        "source_envelope_id",
                        default_source_envelope_id.as_ref(),
                    );
                    insert_default_json_field(
                        obj,
                        "source_authority",
                        default_source_authority.as_ref(),
                    );
                    insert_default_json_field(obj, "trace_ctx", default_trace_ctx.as_ref());
                    insert_default_json_field(
                        obj,
                        "contradiction_status",
                        Some(&serde_json::json!(ContradictionStatus::None)),
                    );
                }
                Some("relation_version") => {
                    insert_default_json_field(obj, "scope_key", default_scope_key.as_ref());
                    insert_default_json_field(
                        obj,
                        "source_envelope_id",
                        default_source_envelope_id.as_ref(),
                    );
                    insert_default_json_field(
                        obj,
                        "source_authority",
                        default_source_authority.as_ref(),
                    );
                    insert_default_json_field(obj, "trace_ctx", default_trace_ctx.as_ref());
                    insert_default_json_field(
                        obj,
                        "contradiction_status",
                        Some(&serde_json::json!(ContradictionStatus::None)),
                    );
                }
                Some("episode") => {
                    insert_default_json_field(obj, "cause_ids", Some(&serde_json::json!([])));
                    insert_default_json_field(
                        obj,
                        "source_envelope_id",
                        default_source_envelope_id.as_ref(),
                    );
                    insert_default_json_field(
                        obj,
                        "source_authority",
                        default_source_authority.as_ref(),
                    );
                    insert_default_json_field(obj, "trace_ctx", default_trace_ctx.as_ref());
                }
                Some("entity_alias") => {
                    insert_default_json_field(obj, "scope", default_scope_key.as_ref());
                    insert_default_json_field(
                        obj,
                        "source_envelope_id",
                        default_source_envelope_id.as_ref(),
                    );
                    insert_default_json_field(
                        obj,
                        "review_state",
                        Some(&serde_json::json!(ReviewState::Unreviewed)),
                    );
                    insert_default_json_field(
                        obj,
                        "is_human_confirmed",
                        Some(&serde_json::json!(false)),
                    );
                    insert_default_json_field(
                        obj,
                        "is_human_confirmed_final",
                        Some(&serde_json::json!(false)),
                    );
                    validate_json_compat_field::<MergeDecision>(obj, "merge_decision")?;
                    validate_json_compat_field::<ReviewState>(obj, "review_state")?;
                }
                Some("evidence_ref") => {
                    insert_default_json_field(
                        obj,
                        "source_envelope_id",
                        default_source_envelope_id.as_ref(),
                    );
                }
                _ => {}
            }
        }
    }

    serde_json::from_value(value).map_err(|e| json_compat_invalid(e.to_string()))
}

/// Result of a projection batch import (V11+).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProjectionImportResult {
    /// Source envelope ID.
    pub source_envelope_id: String,
    /// Import status: "complete" or "already_imported".
    pub status: String,
    /// Number of records in the batch.
    pub record_count: usize,
    /// Whether this was a duplicate (idempotent no-op).
    pub was_duplicate: bool,
}

/// Public view of a V11 projection import log entry.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProjectionImportLogEntry {
    pub batch_id: String,
    pub source_envelope_id: String,
    /// Import-side batch schema version recorded at the memory boundary.
    pub schema_version: String,
    /// Source export schema version preserved as provenance when provided.
    pub export_schema_version: Option<String>,
    pub content_digest: String,
    pub source_authority: String,
    pub scope_namespace: String,
    pub record_count: usize,
    pub claim_count: usize,
    pub relation_count: usize,
    pub episode_count: usize,
    pub alias_count: usize,
    pub evidence_count: usize,
    pub status: String,
    pub source_exported_at: Option<String>,
    pub transformed_at: Option<String>,
    pub imported_at: String,
}

/// Thread-safe handle to the memory database.
///
/// Clone is cheap (Arc internals). `Send + Sync`.
#[derive(Clone)]
pub struct MemoryStore {
    inner: Arc<MemoryStoreInner>,
}

struct MemoryStoreInner {
    pool: pool::SqlitePool,
    embedder: Box<dyn Embedder>,
    embedding_permits: Arc<tokio::sync::Semaphore>,
    config: MemoryConfig,
    paths: StoragePaths,
    token_counter: Arc<dyn TokenCounter>,
    #[cfg(feature = "hnsw")]
    hnsw_index: std::sync::RwLock<HnswIndex>,
}

#[cfg(feature = "hnsw")]
impl Drop for MemoryStoreInner {
    fn drop(&mut self) {
        let pending_ops = match self.pool.with_read_conn(db::pending_index_op_count) {
            Ok(count) => count,
            Err(err) => {
                tracing::warn!("Failed to inspect pending HNSW work on drop: {}", err);
                0
            }
        };

        if pending_ops > 0 {
            if let Err(err) = recover_hnsw_sidecar_sync(&self.pool, &self.paths, &self.config.hnsw)
            {
                tracing::error!("Failed to recover and flush HNSW on drop: {}", err);
            }
            return;
        }

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
        if let Err(e) = self
            .pool
            .with_write_conn(|conn| hnsw_guard.flush_keymap(conn))
        {
            tracing::error!("Failed to flush HNSW keymap on drop: {}", e);
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

fn merge_trace_ctx(
    metadata: Option<serde_json::Value>,
    trace_ctx: Option<&TraceCtx>,
) -> Option<serde_json::Value> {
    let Some(trace_ctx) = trace_ctx else {
        return metadata;
    };

    match metadata {
        Some(serde_json::Value::Object(mut map)) => {
            map.insert(
                "trace_id".to_string(),
                serde_json::Value::String(trace_ctx.trace_id.clone()),
            );
            Some(serde_json::Value::Object(map))
        }
        Some(existing) => Some(serde_json::json!({
            "trace_id": trace_ctx.trace_id,
            "payload": existing,
        })),
        None => Some(serde_json::json!({
            "trace_id": trace_ctx.trace_id,
        })),
    }
}

fn describe_verification_status(status: &VerificationStatus) -> String {
    match status {
        VerificationStatus::Unverified => "unverified".to_string(),
        VerificationStatus::Verified { method, at } => {
            format!("verified via {method} at {at}")
        }
        VerificationStatus::Failed { reason, at } => {
            format!("verification failed at {at}: {reason}")
        }
    }
}

fn truncate_chars(input: &str, max_chars: usize) -> String {
    input.chars().take(max_chars).collect()
}

fn build_episode_search_text(
    document_title: &str,
    document_context: &str,
    meta: &EpisodeMeta,
) -> String {
    let cause_text = if meta.cause_ids.is_empty() {
        "none".to_string()
    } else {
        meta.cause_ids.join(" ")
    };
    let experiment_text = meta.experiment_id.as_deref().unwrap_or("none");
    let verification_text = describe_verification_status(&meta.verification_status);
    let context_excerpt = truncate_chars(document_context, 2_000);

    format!(
        "document {document_title}\n\
         effect {effect}\n\
         outcome {outcome}\n\
         confidence {confidence:.3}\n\
         verification {verification}\n\
         experiment {experiment}\n\
         causes {causes}\n\
         context {context}",
        effect = meta.effect_type,
        outcome = meta.outcome.as_str(),
        confidence = meta.confidence,
        verification = verification_text,
        experiment = experiment_text,
        causes = cause_text,
        context = context_excerpt,
    )
}

fn verification_status_for_outcome(
    outcome: &EpisodeOutcome,
    experiment_id: Option<&str>,
) -> VerificationStatus {
    match outcome {
        EpisodeOutcome::Pending => VerificationStatus::Unverified,
        EpisodeOutcome::Confirmed | EpisodeOutcome::Refuted | EpisodeOutcome::Inconclusive => {
            VerificationStatus::Verified {
                method: experiment_id
                    .map(|id| format!("experiment:{id}"))
                    .unwrap_or_else(|| "manual_outcome_update".to_string()),
                at: chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            }
        }
    }
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

    // Load episode embeddings (keyed by episode_id)
    {
        let mut stmt =
            conn.prepare("SELECT episode_id, embedding FROM episodes WHERE embedding IS NOT NULL")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
        })?;
        for row in rows {
            let (episode_id, blob) = row?;
            if let Ok(emb) = db::bytes_to_embedding(&blob) {
                let key = episodes::episode_item_key(&episode_id);
                if let Err(e) = new_index.insert(key.clone(), &emb) {
                    tracing::warn!("Failed to insert {} into HNSW: {}", key, e);
                }
            }
        }
    }

    Ok(new_index)
}

#[cfg(feature = "hnsw")]
fn sync_pending_hnsw_sidecar(inner: &MemoryStoreInner) -> Result<usize, MemoryError> {
    inner.pool.with_write_conn(|conn| {
        let pending_ops = db::list_pending_index_ops(conn)?;
        if pending_ops.is_empty() {
            if !db::is_sidecar_dirty(conn)? {
                return Ok(0);
            }

            let guard = inner.hnsw_index.read().unwrap_or_else(|e| e.into_inner());
            guard.save(&inner.paths.hnsw_dir, &inner.paths.hnsw_basename)?;
            guard.flush_keymap(conn)?;
            guard.update_last_flush_epoch();
            db::set_sidecar_dirty(conn, false)?;
            return Ok(0);
        }

        let result: Result<usize, MemoryError> = (|| {
            let guard = inner.hnsw_index.write().unwrap_or_else(|e| e.into_inner());

            for op in &pending_ops {
                match op.op_kind {
                    db::IndexOpKind::Upsert => {
                        match db::load_embedding_for_index_key(conn, &op.item_key)? {
                            Some(embedding) => guard.insert(op.item_key.clone(), &embedding)?,
                            None => {
                                // Source row no longer exists; the desired end-state is absence.
                                guard.delete(&op.item_key)?;
                            }
                        }
                    }
                    db::IndexOpKind::Delete => {
                        guard.delete(&op.item_key)?;
                    }
                }
            }

            let processed_keys: Vec<String> =
                pending_ops.iter().map(|op| op.item_key.clone()).collect();
            guard.save(&inner.paths.hnsw_dir, &inner.paths.hnsw_basename)?;
            guard.flush_keymap(conn)?;
            guard.update_last_flush_epoch();
            db::clear_pending_index_ops(conn, &processed_keys)?;
            db::set_sidecar_dirty(conn, false)?;
            Ok(pending_ops.len())
        })();

        if let Err(err) = result {
            let err_text = err.to_string();
            let keys: Vec<String> = pending_ops.iter().map(|op| op.item_key.clone()).collect();
            let _ = db::mark_pending_index_ops_failed(conn, &keys, &err_text);
            return Err(err);
        }

        result
    })
}

#[cfg(feature = "hnsw")]
fn recover_hnsw_sidecar_sync(
    pool: &pool::SqlitePool,
    paths: &StoragePaths,
    config: &HnswConfig,
) -> Result<HnswIndex, MemoryError> {
    let recovered = pool.with_read_conn(|conn| rebuild_hnsw_from_sqlite(conn, config))?;
    recovered.save(&paths.hnsw_dir, &paths.hnsw_basename)?;
    pool.with_write_conn(|conn| {
        recovered.flush_keymap(conn)?;
        db::clear_all_pending_index_ops(conn)?;
        db::set_sidecar_dirty(conn, false)?;
        Ok(())
    })?;
    Ok(recovered)
}

impl MemoryStore {
    /// Run read-only work on a pooled reader connection on a blocking thread.
    ///
    /// This prevents SQLite I/O from stalling the tokio executor while allowing
    /// multiple concurrent readers under WAL mode.
    async fn with_read_conn<F, T>(&self, f: F) -> Result<T, MemoryError>
    where
        F: FnOnce(&rusqlite::Connection) -> Result<T, MemoryError> + Send + 'static,
        T: Send + 'static,
    {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || -> Result<T, MemoryError> {
            inner.pool.with_read_conn(f)
        })
        .await
        .map_err(|e| MemoryError::Other(format!("Blocking task panicked: {}", e)))?
    }

    /// Run write-capable work on the single writer connection on a blocking thread.
    async fn with_write_conn<F, T>(&self, f: F) -> Result<T, MemoryError>
    where
        F: FnOnce(&rusqlite::Connection) -> Result<T, MemoryError> + Send + 'static,
        T: Send + 'static,
    {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || -> Result<T, MemoryError> {
            inner.pool.with_write_conn(f)
        })
        .await
        .map_err(|e| MemoryError::Other(format!("Blocking task panicked: {}", e)))?
    }

    #[cfg(feature = "hnsw")]
    fn sync_pending_hnsw_ops_blocking(&self) -> Result<usize, MemoryError> {
        sync_pending_hnsw_sidecar(&self.inner)
    }

    #[cfg(feature = "hnsw")]
    async fn sync_pending_hnsw_ops(&self) -> Result<usize, MemoryError> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || sync_pending_hnsw_sidecar(&inner))
            .await
            .map_err(|e| MemoryError::Other(format!("Blocking task panicked: {}", e)))?
    }

    #[cfg(feature = "hnsw")]
    async fn sync_pending_hnsw_ops_best_effort(&self, operation: &'static str) {
        if let Err(err) = self.sync_pending_hnsw_ops().await {
            tracing::warn!(
                operation,
                error = %err,
                "SQLite write committed but HNSW sidecar sync is still pending"
            );
        } else {
            self.maybe_flush_hnsw();
        }
    }

    /// Open or create a memory store at the configured base directory.
    ///
    /// Creates the directory if it doesn't exist, opens/creates SQLite,
    /// runs migrations, and initializes the HNSW index.
    pub fn open(config: MemoryConfig) -> Result<Self, MemoryError> {
        let config = config.normalize_and_validate()?;
        let embedder = Box::new(OllamaEmbedder::try_new(&config.embedding)?);
        Self::open_with_embedder(config, embedder)
    }

    /// Open with a custom embedder (for testing or non-Ollama providers).
    #[allow(unused_mut)] // `config` is mutated only when the `hnsw` feature is enabled
    pub fn open_with_embedder(
        mut config: MemoryConfig,
        embedder: Box<dyn Embedder>,
    ) -> Result<Self, MemoryError> {
        config = config.normalize_and_validate()?;
        if embedder.dimensions() != config.embedding.dimensions {
            return Err(MemoryError::DimensionMismatch {
                expected: config.embedding.dimensions,
                actual: embedder.dimensions(),
            });
        }
        config.embedding.model = embedder.model_name().to_string();

        let paths = StoragePaths::new(&config.base_dir);

        // Create directory if needed
        std::fs::create_dir_all(&paths.base_dir).map_err(|e| {
            MemoryError::StorageError(format!(
                "Failed to create directory {}: {}",
                paths.base_dir.display(),
                e
            ))
        })?;

        let pool = pool::SqlitePool::open(&paths.sqlite_path, &config.pool, &config.limits)?;
        pool.with_write_conn(|conn| db::check_embedding_metadata(conn, &config.embedding))?;

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

            let embeddings_dirty = pool.with_write_conn(db::is_embeddings_dirty)?;
            let pending_index_ops = pool.with_read_conn(db::pending_index_op_count)?;

            if embeddings_dirty {
                // Embedding model changed — old HNSW index is useless.
                // Create a fresh index; reembed_all() will rebuild it.
                tracing::warn!(
                    "Embedding model changed — creating fresh HNSW index (old index is stale)"
                );
                pool.with_write_conn(|conn| {
                    db::clear_all_pending_index_ops(conn)?;
                    db::set_sidecar_dirty(conn, false)?;
                    Ok(())
                })?;
                HnswIndex::new(hnsw_config)?
            } else if pending_index_ops > 0 || pool.with_read_conn(db::is_sidecar_dirty)? {
                tracing::warn!(
                    pending_index_ops,
                    "Recovering HNSW sidecar from SQLite because durable sidecar work exists"
                );
                recover_hnsw_sidecar_sync(&pool, &paths, &hnsw_config)?
            } else if paths.hnsw_files_exist() {
                tracing::info!("Loading HNSW index from {:?}", paths.hnsw_dir);
                match HnswIndex::load(&paths.hnsw_dir, &paths.hnsw_basename, hnsw_config.clone()) {
                    Ok(index) => {
                        // Load key mappings from SQLite
                        if let Err(e) = pool.with_write_conn(|conn| index.load_keymap(conn)) {
                            tracing::warn!("Failed to load HNSW key mappings: {}. Mappings will be empty until rebuild.", e);
                        }

                        // Stale index detection: compare HNSW entry count vs SQLite
                        // embedding count. A mismatch means the app crashed before
                        // flushing HNSW, or keys were lost.
                        let hnsw_count = index.len();
                        let sqlite_count: i64 = pool.with_write_conn(|conn| {
                            Ok(conn
                                .query_row(
                                    "SELECT (SELECT COUNT(*) FROM facts WHERE embedding IS NOT NULL) +
                                        (SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL) +
                                        (SELECT COUNT(*) FROM messages WHERE embedding IS NOT NULL) +
                                        (SELECT COUNT(*) FROM episodes WHERE embedding IS NOT NULL)",
                                    [],
                                    |row| row.get(0),
                                )
                                .unwrap_or(0))
                        })?;

                        let drift = (sqlite_count - hnsw_count as i64).abs();
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
                            let rebuilt = recover_hnsw_sidecar_sync(&pool, &paths, &hnsw_config)?;
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
                let orphan_count: i64 = pool.with_write_conn(|conn| {
                    Ok(conn
                        .query_row(
                            "SELECT (SELECT COUNT(*) FROM facts WHERE embedding IS NOT NULL) +
                                (SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL) +
                                (SELECT COUNT(*) FROM messages WHERE embedding IS NOT NULL) +
                                (SELECT COUNT(*) FROM episodes WHERE embedding IS NOT NULL)",
                            [],
                            |row| row.get(0),
                        )
                        .unwrap_or(0))
                })?;

                if orphan_count > 0 {
                    tracing::warn!(
                        orphan_count,
                        "HNSW sidecar files missing but {} embeddings exist in SQLite — \
                         rebuilding index inline",
                        orphan_count
                    );
                    let new_index = recover_hnsw_sidecar_sync(&pool, &paths, &hnsw_config)?;
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

        let store = Self {
            inner: Arc::new(MemoryStoreInner {
                pool,
                embedder,
                embedding_permits: Arc::new(tokio::sync::Semaphore::new(
                    config.limits.max_embedding_concurrency,
                )),
                config,
                paths,
                token_counter,
                #[cfg(feature = "hnsw")]
                hnsw_index: std::sync::RwLock::new(hnsw_index),
            }),
        };

        #[cfg(feature = "hnsw")]
        if let Err(err) = store.sync_pending_hnsw_ops_blocking() {
            tracing::warn!(
                error = %err,
                "Failed to reconcile pending HNSW sidecar ops during open; sidecar replay remains pending"
            );
        }

        Ok(store)
    }

    async fn with_embedding_permit(
        &self,
    ) -> Result<tokio::sync::OwnedSemaphorePermit, MemoryError> {
        self.inner
            .embedding_permits
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| MemoryError::Other("embedding semaphore closed".to_string()))
    }

    async fn embed_text_internal(&self, text: &str) -> Result<Vec<f32>, MemoryError> {
        let _permit = self.with_embedding_permit().await?;
        self.inner.embedder.embed(text).await
    }

    async fn embed_batch_internal(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, MemoryError> {
        let _permit = self.with_embedding_permit().await?;
        self.inner.embedder.embed_batch(texts).await
    }

    fn validate_embedding_dimensions(&self, embedding: &[f32]) -> Result<(), MemoryError> {
        let expected = self.inner.config.embedding.dimensions;
        if embedding.len() != expected {
            return Err(MemoryError::DimensionMismatch {
                expected,
                actual: embedding.len(),
            });
        }
        Ok(())
    }

    fn validate_content(&self, field: &'static str, content: &str) -> Result<(), MemoryError> {
        if content.is_empty() {
            return Err(MemoryError::InvalidConfig {
                field,
                reason: "content must not be empty".to_string(),
            });
        }

        let limit = self.inner.config.limits.max_content_bytes;
        if content.len() > limit {
            return Err(MemoryError::ContentTooLarge {
                size: content.len(),
                limit,
            });
        }

        Ok(())
    }

    fn validate_confidence(confidence: f32) -> Result<(), MemoryError> {
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(MemoryError::InvalidConfig {
                field: "episodes.confidence",
                reason: "confidence must be finite and within [0.0, 1.0]".to_string(),
            });
        }
        Ok(())
    }

    // ─── HNSW Management ───────────────────────────────────────

    /// Rebuild the HNSW index from SQLite f32 embeddings.
    ///
    /// Call this if sidecar files are missing, corrupted, or after `reembed_all()`.
    #[cfg(feature = "hnsw")]
    pub async fn rebuild_hnsw_index(&self) -> Result<(), MemoryError> {
        tracing::info!("Rebuilding HNSW index from SQLite embeddings...");
        let hnsw_config = self.inner.config.hnsw.clone();
        let new_index = self
            .with_read_conn(move |conn| rebuild_hnsw_from_sqlite(conn, &hnsw_config))
            .await?;

        {
            let mut guard = self
                .inner
                .hnsw_index
                .write()
                .unwrap_or_else(|e| e.into_inner());
            *guard = new_index.clone();
        }

        new_index.save(&self.inner.paths.hnsw_dir, &self.inner.paths.hnsw_basename)?;
        self.inner.pool.with_write_conn(|conn| {
            new_index.flush_keymap(conn)?;
            db::clear_all_pending_index_ops(conn)?;
            db::set_sidecar_dirty(conn, false)?;
            Ok(())
        })?;

        tracing::info!(active = new_index.len(), "HNSW index rebuilt");

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
        let pending_ops = self.inner.pool.with_read_conn(db::pending_index_op_count)?;
        if pending_ops > 0 {
            tracing::info!(
                pending_ops,
                "Flushing HNSW via authoritative SQLite rebuild because pending durable sidecar work exists"
            );
            let rebuilt = recover_hnsw_sidecar_sync(
                &self.inner.pool,
                &self.inner.paths,
                &self.inner.config.hnsw,
            )?;
            let mut guard = self
                .inner
                .hnsw_index
                .write()
                .unwrap_or_else(|e| e.into_inner());
            *guard = rebuilt;
            return Ok(());
        }

        let guard = self
            .inner
            .hnsw_index
            .read()
            .unwrap_or_else(|e| e.into_inner());
        guard.save(&self.inner.paths.hnsw_dir, &self.inner.paths.hnsw_basename)?;

        // Flush key mappings to SQLite
        self.inner.pool.with_write_conn(|conn| {
            guard.flush_keymap(conn)?;
            db::clear_all_pending_index_ops(conn)?;
            db::set_sidecar_dirty(conn, false)?;
            Ok(())
        })?;
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
        let mut report = self
            .with_read_conn(move |conn| db::verify_integrity_sync(conn, mode))
            .await?;

        #[cfg(feature = "hnsw")]
        {
            let embedding_count: i64 = self
                .with_read_conn(|conn| {
                    Ok(conn.query_row(
                        "SELECT (SELECT COUNT(*) FROM facts WHERE embedding IS NOT NULL) +
                            (SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL) +
                            (SELECT COUNT(*) FROM messages WHERE embedding IS NOT NULL) +
                            (SELECT COUNT(*) FROM episodes WHERE embedding IS NOT NULL)",
                        [],
                        |row| row.get(0),
                    )?)
                })
                .await?;

            if embedding_count > 0 && !self.inner.paths.hnsw_files_exist() {
                report.issues.push(format!(
                    "HNSW sidecar files are missing while {} embedded rows exist in SQLite",
                    embedding_count
                ));
            }

            let keymap_count: i64 = self
                .with_read_conn(|conn| {
                    Ok(conn
                        .query_row(
                            "SELECT COUNT(*) FROM hnsw_keymap WHERE deleted = 0",
                            [],
                            |row| row.get(0),
                        )
                        .unwrap_or(0))
                })
                .await?;

            if keymap_count != embedding_count {
                report.issues.push(format!(
                    "HNSW keymap drift: {} active keymap rows vs {} embedded SQLite rows",
                    keymap_count, embedding_count
                ));
            }
        }

        report.ok = report.issues.is_empty();
        Ok(report)
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
                self.with_write_conn(db::reconcile_fts).await?;
                #[cfg(feature = "hnsw")]
                self.sync_pending_hnsw_ops_best_effort("reconcile_rebuild_fts")
                    .await;
                self.verify_integrity(db::VerifyMode::Full).await
            }
            db::ReconcileAction::ReEmbed => {
                self.reembed_all().await?;
                self.verify_integrity(db::VerifyMode::Full).await
            }
        }
    }

    /// Get the current configuration.
    pub fn config(&self) -> &MemoryConfig {
        &self.inner.config
    }

    /// View the store as a derived graph over documents, chunks, facts, sessions, messages,
    /// episodes, namespaces, and semantic similarity edges.
    pub fn graph_view(&self) -> Arc<dyn GraphView> {
        graph::graph_view(self.inner.clone())
    }

    // ─── Session Management ─────────────────────────────────────

    /// Create a new conversation session. Returns the session ID (UUID v4).
    pub async fn create_session(&self, channel: &str) -> Result<String, MemoryError> {
        let channel = channel.to_string();
        self.with_write_conn(move |conn| conversation::create_session(conn, &channel, None))
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
        self.with_write_conn(move |conn| conversation::rename_session(conn, &sid, &ch))
            .await
    }

    /// List recent sessions, newest first.
    pub async fn list_sessions(
        &self,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Session>, MemoryError> {
        self.with_read_conn(move |conn| conversation::list_sessions(conn, limit, offset))
            .await
    }

    /// Delete a session and all its messages.
    ///
    /// Cleans up HNSW entries for embedded messages before CASCADE delete.
    pub async fn delete_session(&self, session_id: &str) -> Result<(), MemoryError> {
        // Delete session (CASCADE handles messages, FTS cleanup inside transaction)
        let sid = session_id.to_string();
        self.with_write_conn(move |conn| conversation::delete_session(conn, &sid))
            .await?;

        #[cfg(feature = "hnsw")]
        self.sync_pending_hnsw_ops_best_effort("delete_session")
            .await;

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
        self.add_message_with_trace(session_id, role, content, token_count, metadata, None)
            .await
    }

    /// Append a message to a session with optional trace metadata.
    pub async fn add_message_with_trace(
        &self,
        session_id: &str,
        role: Role,
        content: &str,
        token_count: Option<u32>,
        metadata: Option<serde_json::Value>,
        trace_ctx: Option<&TraceCtx>,
    ) -> Result<i64, MemoryError> {
        self.add_message_embedded_with_trace(
            session_id,
            role,
            content,
            token_count,
            metadata,
            trace_ctx,
        )
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
        self.add_message_fts_with_trace(session_id, role, content, token_count, metadata, None)
            .await
    }

    /// Append a message with FTS indexing and optional trace metadata.
    pub async fn add_message_fts_with_trace(
        &self,
        session_id: &str,
        role: Role,
        content: &str,
        token_count: Option<u32>,
        metadata: Option<serde_json::Value>,
        trace_ctx: Option<&TraceCtx>,
    ) -> Result<i64, MemoryError> {
        self.validate_content("message.content", content)?;

        let effective_token_count =
            token_count.or_else(|| Some(self.inner.token_counter.count_tokens(content) as u32));
        let sid = session_id.to_string();
        let ct = content.to_string();
        let meta = merge_trace_ctx(metadata, trace_ctx);
        self.with_write_conn(move |conn| {
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
        self.with_read_conn(move |conn| conversation::get_recent_messages(conn, &sid, limit))
            .await
    }

    /// Get messages from a session up to `max_tokens` total.
    pub async fn get_messages_within_budget(
        &self,
        session_id: &str,
        max_tokens: u32,
    ) -> Result<Vec<Message>, MemoryError> {
        let sid = session_id.to_string();
        self.with_read_conn(move |conn| {
            conversation::get_messages_within_budget(conn, &sid, max_tokens)
        })
        .await
    }

    /// Get total token count for a session.
    pub async fn session_token_count(&self, session_id: &str) -> Result<u64, MemoryError> {
        let sid = session_id.to_string();
        self.with_read_conn(move |conn| conversation::session_token_count(conn, &sid))
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
        self.add_fact_with_trace(namespace, content, source, metadata, None)
            .await
    }

    /// Store a fact with automatic embedding and optional trace metadata.
    pub async fn add_fact_with_trace(
        &self,
        namespace: &str,
        content: &str,
        source: Option<&str>,
        metadata: Option<serde_json::Value>,
        trace_ctx: Option<&TraceCtx>,
    ) -> Result<String, MemoryError> {
        self.validate_content("fact.content", content)?;

        let embedding = self.embed_text_internal(content).await?;
        self.validate_embedding_dimensions(&embedding)?;
        let embedding_bytes = db::embedding_to_bytes(&embedding);
        let fact_id = uuid::Uuid::new_v4().to_string();
        let max_facts_per_namespace = self.inner.config.limits.max_facts_per_namespace;

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
        let meta = merge_trace_ctx(metadata, trace_ctx);
        self.with_write_conn(move |conn| {
            let current_count: usize = conn.query_row(
                "SELECT COUNT(*) FROM facts WHERE namespace = ?1",
                rusqlite::params![&ns],
                |row| row.get(0),
            )?;
            if current_count >= max_facts_per_namespace {
                return Err(MemoryError::NamespaceFull {
                    namespace: ns.clone(),
                    count: current_count,
                    limit: max_facts_per_namespace,
                });
            }
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

        // SQLite is authoritative. Sidecar failures are journaled, not surfaced as
        // false write failures after the SQLite commit succeeds.
        #[cfg(feature = "hnsw")]
        self.sync_pending_hnsw_ops_best_effort("add_fact").await;

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
        self.add_fact_with_embedding_and_trace(
            namespace, content, embedding, source, metadata, None,
        )
        .await
    }

    /// Store a fact with a pre-computed embedding and optional trace metadata.
    pub async fn add_fact_with_embedding_and_trace(
        &self,
        namespace: &str,
        content: &str,
        embedding: &[f32],
        source: Option<&str>,
        metadata: Option<serde_json::Value>,
        trace_ctx: Option<&TraceCtx>,
    ) -> Result<String, MemoryError> {
        self.validate_content("fact.content", content)?;
        self.validate_embedding_dimensions(embedding)?;
        let embedding_bytes = db::embedding_to_bytes(embedding);
        let fact_id = uuid::Uuid::new_v4().to_string();
        let max_facts_per_namespace = self.inner.config.limits.max_facts_per_namespace;

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
        let meta = merge_trace_ctx(metadata, trace_ctx);
        self.with_write_conn(move |conn| {
            let current_count: usize = conn.query_row(
                "SELECT COUNT(*) FROM facts WHERE namespace = ?1",
                rusqlite::params![&ns],
                |row| row.get(0),
            )?;
            if current_count >= max_facts_per_namespace {
                return Err(MemoryError::NamespaceFull {
                    namespace: ns.clone(),
                    count: current_count,
                    limit: max_facts_per_namespace,
                });
            }
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

        // SQLite is authoritative. Sidecar failures are journaled, not surfaced as
        // false write failures after the SQLite commit succeeds.
        #[cfg(feature = "hnsw")]
        self.sync_pending_hnsw_ops_best_effort("add_fact_with_embedding")
            .await;

        Ok(fact_id)
    }

    /// Update a fact's content. Re-embeds automatically.
    pub async fn update_fact(&self, fact_id: &str, content: &str) -> Result<(), MemoryError> {
        self.validate_content("fact.content", content)?;
        let embedding = self.embed_text_internal(content).await?;
        self.validate_embedding_dimensions(&embedding)?;
        let embedding_bytes = db::embedding_to_bytes(&embedding);
        let q8_bytes = Quantizer::new(self.inner.config.embedding.dimensions)
            .quantize(&embedding)
            .map(|qv| quantize::pack_quantized(&qv))
            .ok();

        let fid = fact_id.to_string();
        let ct = content.to_string();
        self.with_write_conn(move |conn| {
            knowledge::update_fact_with_fts(conn, &fid, &ct, &embedding_bytes, q8_bytes.as_deref())
        })
        .await?;

        // SQLite is authoritative. Sidecar failures are journaled, not surfaced as
        // false write failures after the SQLite commit succeeds.
        #[cfg(feature = "hnsw")]
        self.sync_pending_hnsw_ops_best_effort("update_fact").await;

        Ok(())
    }

    /// Delete a fact by ID.
    pub async fn delete_fact(&self, fact_id: &str) -> Result<(), MemoryError> {
        let fid = fact_id.to_string();
        self.with_write_conn(move |conn| knowledge::delete_fact_with_fts(conn, &fid))
            .await?;

        // SQLite is authoritative. Sidecar failures are journaled, not surfaced as
        // false write failures after the SQLite commit succeeds.
        #[cfg(feature = "hnsw")]
        self.sync_pending_hnsw_ops_best_effort("delete_fact").await;

        Ok(())
    }

    /// Delete all facts in a namespace. Returns the count of deleted facts.
    pub async fn delete_namespace(&self, namespace: &str) -> Result<usize, MemoryError> {
        let ns = namespace.to_string();

        let count = self
            .with_write_conn(move |conn| knowledge::delete_namespace(conn, &ns))
            .await?;

        #[cfg(feature = "hnsw")]
        self.sync_pending_hnsw_ops_best_effort("delete_namespace")
            .await;

        Ok(count)
    }

    /// Get a fact by ID.
    pub async fn get_fact(&self, fact_id: &str) -> Result<Option<Fact>, MemoryError> {
        let fid = fact_id.to_string();
        self.with_read_conn(move |conn| knowledge::get_fact(conn, &fid))
            .await
    }

    /// Get a fact's embedding vector.
    pub async fn get_fact_embedding(&self, fact_id: &str) -> Result<Option<Vec<f32>>, MemoryError> {
        let fid = fact_id.to_string();
        self.with_read_conn(move |conn| knowledge::get_fact_embedding(conn, &fid))
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
        self.with_read_conn(move |conn| knowledge::list_facts(conn, &ns, limit, offset))
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
        self.ingest_document_with_trace(title, content, namespace, source_path, metadata, None)
            .await
    }

    /// Ingest a document with optional trace metadata.
    pub async fn ingest_document_with_trace(
        &self,
        title: &str,
        content: &str,
        namespace: &str,
        source_path: Option<&str>,
        metadata: Option<serde_json::Value>,
        trace_ctx: Option<&TraceCtx>,
    ) -> Result<String, MemoryError> {
        self.validate_content("document.content", content)?;

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
        let embeddings = self.embed_batch_internal(chunk_texts).await?;
        for embedding in &embeddings {
            self.validate_embedding_dimensions(embedding)?;
        }

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
        let meta = merge_trace_ctx(metadata, trace_ctx);

        self.with_write_conn(move |conn| {
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

        #[cfg(feature = "hnsw")]
        self.sync_pending_hnsw_ops_best_effort("ingest_document")
            .await;

        Ok(doc_id)
    }

    /// Delete a document and all its chunks.
    pub async fn delete_document(&self, document_id: &str) -> Result<(), MemoryError> {
        let did = document_id.to_string();
        self.with_write_conn(move |conn| documents::delete_document_with_chunks(conn, &did))
            .await?;

        #[cfg(feature = "hnsw")]
        self.sync_pending_hnsw_ops_best_effort("delete_document")
            .await;

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
        self.with_read_conn(move |conn| documents::list_documents(conn, &ns, limit, offset))
            .await
    }

    /// Count the number of chunks for a document.
    pub async fn count_chunks_for_document(&self, document_id: &str) -> Result<usize, MemoryError> {
        let did = document_id.to_string();
        self.with_read_conn(move |conn| documents::count_chunks_for_document(conn, &did))
            .await
    }

    // ─── Search ─────────────────────────────────────────────────

    /// Hybrid search across facts, document chunks, and searchable episodes.
    pub async fn search(
        &self,
        query: &str,
        top_k: Option<usize>,
        namespaces: Option<&[&str]>,
        source_types: Option<&[SearchSourceType]>,
    ) -> Result<Vec<SearchResult>, MemoryError> {
        let k = top_k.unwrap_or(self.inner.config.search.default_top_k);

        let query_embedding = self.embed_text_internal(query).await?;

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
            let candidates = self.inner.config.search.candidate_pool_size.max(k * 3);
            match guard.search(&query_embedding, candidates) {
                Ok(hits) => hits,
                Err(e) => {
                    tracing::error!(
                        "HNSW search failed, falling back to brute-force vector search: {}",
                        e
                    );
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

        self.with_read_conn(move |conn| {
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
                if hnsw_hits_owned.is_empty() {
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
                } else {
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
        self.with_read_conn(move |conn| {
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
        let query_embedding = self.embed_text_internal(query).await?;

        #[cfg(feature = "hnsw")]
        let hnsw_hits = {
            let candidates = self.inner.config.search.candidate_pool_size.max(k * 3);
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

        self.with_read_conn(move |conn| {
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
                if hnsw_hits_owned.is_empty() {
                    search::vector_only_search(
                        conn,
                        &query_embedding,
                        &config,
                        k,
                        ns_slice,
                        st_slice,
                        None,
                    )
                } else {
                    search::vector_only_search_with_hnsw(
                        conn,
                        &query_embedding,
                        &config,
                        k,
                        ns_slice,
                        st_slice,
                        None,
                        &hnsw_hits_owned,
                    )
                }
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

    /// Append a message to a session with automatic embedding and FTS indexing.
    pub async fn add_message_embedded(
        &self,
        session_id: &str,
        role: Role,
        content: &str,
        token_count: Option<u32>,
        metadata: Option<serde_json::Value>,
    ) -> Result<i64, MemoryError> {
        self.add_message_embedded_with_trace(session_id, role, content, token_count, metadata, None)
            .await
    }

    /// Append an embedded message with optional trace metadata.
    pub async fn add_message_embedded_with_trace(
        &self,
        session_id: &str,
        role: Role,
        content: &str,
        token_count: Option<u32>,
        metadata: Option<serde_json::Value>,
        trace_ctx: Option<&TraceCtx>,
    ) -> Result<i64, MemoryError> {
        self.validate_content("message.content", content)?;

        let effective_token_count =
            token_count.or_else(|| Some(self.inner.token_counter.count_tokens(content) as u32));

        let embedding = self.embed_text_internal(content).await?;
        self.validate_embedding_dimensions(&embedding)?;
        let embedding_bytes = db::embedding_to_bytes(&embedding);

        // Quantize for storage
        let quantizer = Quantizer::new(self.inner.config.embedding.dimensions);
        let q8_bytes = quantizer
            .quantize(&embedding)
            .map(|qv| quantize::pack_quantized(&qv))
            .ok();

        let sid = session_id.to_string();
        let ct = content.to_string();
        let meta = merge_trace_ctx(metadata, trace_ctx);
        let msg_id = self
            .with_write_conn(move |conn| {
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

        // SQLite is authoritative. Sidecar failures are journaled, not surfaced as
        // false write failures after the SQLite commit succeeds.
        #[cfg(feature = "hnsw")]
        self.sync_pending_hnsw_ops_best_effort("add_message_embedded")
            .await;

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

        let query_embedding = self.embed_text_internal(query).await?;

        #[cfg(feature = "hnsw")]
        let hnsw_hits = {
            let guard = self
                .inner
                .hnsw_index
                .read()
                .unwrap_or_else(|e| e.into_inner());
            let candidates = self.inner.config.search.candidate_pool_size.max(k * 3);
            match guard.search(&query_embedding, candidates) {
                Ok(hits) => hits,
                Err(err) => {
                    tracing::error!(
                        "HNSW conversation search failed, falling back to brute-force message search: {}",
                        err
                    );
                    Vec::new()
                }
            }
        };

        let q = query.to_string();
        let config = self.inner.config.search.clone();
        let sids_owned = to_owned_string_vec(session_ids);

        #[cfg(feature = "hnsw")]
        let hnsw_hits_owned = hnsw_hits;

        self.with_read_conn(move |conn| {
            let sids_refs = as_str_slice(&sids_owned);
            let sids_slice: Option<&[&str]> = sids_refs.as_deref();
            #[cfg(feature = "hnsw")]
            {
                if hnsw_hits_owned.is_empty() {
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
                } else {
                    search::hybrid_search_with_hnsw(
                        conn,
                        &q,
                        &query_embedding,
                        &config,
                        k,
                        None,
                        Some(&[SearchSourceType::Messages]),
                        sids_slice,
                        &hnsw_hits_owned,
                    )
                }
            }
            #[cfg(not(feature = "hnsw"))]
            {
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
            }
        })
        .await
    }

    // ─── Episodes ──────────────────────────────────────────────

    /// Ingest or update a causal episode attached to a document.
    ///
    /// The document must already exist. Existing episodes keep their original `created_at`
    /// timestamp while their searchable text, outcome state, verification metadata, embeddings,
    /// and `updated_at` are refreshed.
    pub async fn ingest_episode(
        &self,
        document_id: &str,
        meta: &types::EpisodeMeta,
    ) -> Result<String, MemoryError> {
        self.ingest_episode_with_trace(document_id, meta, None)
            .await
    }

    /// Ingest a causal episode with optional trace metadata. Returns the episode_id.
    pub async fn ingest_episode_with_trace(
        &self,
        document_id: &str,
        meta: &types::EpisodeMeta,
        trace_ctx: Option<&TraceCtx>,
    ) -> Result<String, MemoryError> {
        self.validate_content("episodes.effect_type", &meta.effect_type)?;
        Self::validate_confidence(meta.confidence)?;
        let doc_id = document_id.to_string();
        let meta = meta.clone();
        let (document_title, document_context) = self
            .with_read_conn(move |conn| episodes::load_episode_context(conn, &doc_id))
            .await?;
        let search_text = build_episode_search_text(&document_title, &document_context, &meta);
        let embedding = self.embed_text_internal(&search_text).await?;
        self.validate_embedding_dimensions(&embedding)?;
        let embedding_bytes = db::embedding_to_bytes(&embedding);
        let q8_bytes = Quantizer::new(self.inner.config.embedding.dimensions)
            .quantize(&embedding)
            .map(|vector| quantize::pack_quantized(&vector))
            .ok();
        let trace_id_owned = trace_ctx.map(|value| value.trace_id.clone());

        let doc_id = document_id.to_string();
        let episode_id = self
            .with_write_conn(move |conn| {
                episodes::upsert_episode(
                    conn,
                    &doc_id,
                    &meta,
                    &search_text,
                    &embedding_bytes,
                    q8_bytes.as_deref(),
                    trace_id_owned.as_deref(),
                )
            })
            .await?;

        #[cfg(feature = "hnsw")]
        self.sync_pending_hnsw_ops_best_effort("ingest_episode")
            .await;

        Ok(episode_id)
    }

    /// Create a new episode with an explicit episode_id. Returns the episode_id.
    pub async fn create_episode(
        &self,
        episode_id: &str,
        document_id: &str,
        meta: &types::EpisodeMeta,
    ) -> Result<String, MemoryError> {
        self.create_episode_with_trace(episode_id, document_id, meta, None)
            .await
    }

    /// Create a new episode with an explicit episode_id and optional trace metadata.
    pub async fn create_episode_with_trace(
        &self,
        episode_id: &str,
        document_id: &str,
        meta: &types::EpisodeMeta,
        trace_ctx: Option<&TraceCtx>,
    ) -> Result<String, MemoryError> {
        self.validate_content("episodes.effect_type", &meta.effect_type)?;
        Self::validate_confidence(meta.confidence)?;
        let doc_id = document_id.to_string();
        let meta = meta.clone();
        let (document_title, document_context) = self
            .with_read_conn(move |conn| episodes::load_episode_context(conn, &doc_id))
            .await?;
        let search_text = build_episode_search_text(&document_title, &document_context, &meta);
        let embedding = self.embed_text_internal(&search_text).await?;
        self.validate_embedding_dimensions(&embedding)?;
        let embedding_bytes = db::embedding_to_bytes(&embedding);
        let q8_bytes = Quantizer::new(self.inner.config.embedding.dimensions)
            .quantize(&embedding)
            .map(|vector| quantize::pack_quantized(&vector))
            .ok();
        let trace_id_owned = trace_ctx.map(|value| value.trace_id.clone());

        let ep_id = episode_id.to_string();
        let doc_id = document_id.to_string();
        let created_ep_id = self
            .with_write_conn(move |conn| {
                episodes::create_episode(
                    conn,
                    &ep_id,
                    &doc_id,
                    &meta,
                    &search_text,
                    &embedding_bytes,
                    q8_bytes.as_deref(),
                    trace_id_owned.as_deref(),
                )
            })
            .await?;

        #[cfg(feature = "hnsw")]
        self.sync_pending_hnsw_ops_best_effort("create_episode")
            .await;

        Ok(created_ep_id)
    }

    /// Retrieve an episode by its episode_id.
    pub async fn get_episode(
        &self,
        episode_id: &str,
    ) -> Result<Option<(String, types::EpisodeMeta)>, MemoryError> {
        let ep_id = episode_id.to_string();
        self.with_read_conn(move |conn| episodes::get_episode(conn, &ep_id))
            .await
    }

    /// Update the outcome of an episode by its episode_id.
    pub async fn update_episode_outcome_by_id(
        &self,
        episode_id: &str,
        outcome: types::EpisodeOutcome,
        confidence: f32,
        experiment_id: Option<&str>,
    ) -> Result<(), MemoryError> {
        Self::validate_confidence(confidence)?;
        let ep_id = episode_id.to_string();
        let ep_id_clone = ep_id.clone();
        // Get the document_id from the episode to load context
        let (doc_id, current_meta) = self
            .with_read_conn(move |conn| {
                episodes::get_episode(conn, &ep_id_clone)?
                    .ok_or_else(|| MemoryError::EpisodeNotFound(ep_id_clone.clone()))
            })
            .await?;

        let experiment_id_owned = experiment_id.map(|value| value.to_string());
        let verification_status =
            verification_status_for_outcome(&outcome, experiment_id_owned.as_deref());
        let updated_meta = EpisodeMeta {
            cause_ids: current_meta.cause_ids,
            effect_type: current_meta.effect_type,
            outcome: outcome.clone(),
            confidence,
            verification_status: verification_status.clone(),
            experiment_id: experiment_id_owned.clone().or(current_meta.experiment_id),
        };

        let (document_title, document_context) = self
            .with_read_conn(move |conn| episodes::load_episode_context(conn, &doc_id))
            .await?;
        let search_text =
            build_episode_search_text(&document_title, &document_context, &updated_meta);
        let embedding = self.embed_text_internal(&search_text).await?;
        self.validate_embedding_dimensions(&embedding)?;
        let embedding_bytes = db::embedding_to_bytes(&embedding);
        let q8_bytes = Quantizer::new(self.inner.config.embedding.dimensions)
            .quantize(&embedding)
            .map(|vector| quantize::pack_quantized(&vector))
            .ok();

        self.with_write_conn(move |conn| {
            episodes::update_episode_outcome_by_id(
                conn,
                &ep_id,
                outcome,
                confidence,
                experiment_id_owned.as_deref(),
                &verification_status,
                &search_text,
                &embedding_bytes,
                q8_bytes.as_deref(),
            )
        })
        .await?;

        #[cfg(feature = "hnsw")]
        self.sync_pending_hnsw_ops_best_effort("update_episode_outcome_by_id")
            .await;

        Ok(())
    }

    /// Update the outcome of an existing episode.
    pub async fn update_episode_outcome(
        &self,
        document_id: &str,
        outcome: types::EpisodeOutcome,
        confidence: f32,
        experiment_id: Option<&str>,
    ) -> Result<(), MemoryError> {
        Self::validate_confidence(confidence)?;
        let doc_id = document_id.to_string();
        let current_meta = self
            .with_read_conn(move |conn| episodes::load_episode_meta(conn, &doc_id))
            .await?
            .ok_or_else(|| MemoryError::DocumentNotFound(document_id.to_string()))?;

        let experiment_id_owned = experiment_id.map(|value| value.to_string());
        let verification_status =
            verification_status_for_outcome(&outcome, experiment_id_owned.as_deref());
        let updated_meta = EpisodeMeta {
            cause_ids: current_meta.cause_ids,
            effect_type: current_meta.effect_type,
            outcome: outcome.clone(),
            confidence,
            verification_status: verification_status.clone(),
            experiment_id: experiment_id_owned.clone().or(current_meta.experiment_id),
        };

        let doc_id = document_id.to_string();
        let (document_title, document_context) = self
            .with_read_conn(move |conn| episodes::load_episode_context(conn, &doc_id))
            .await?;
        let search_text =
            build_episode_search_text(&document_title, &document_context, &updated_meta);
        let embedding = self.embed_text_internal(&search_text).await?;
        self.validate_embedding_dimensions(&embedding)?;
        let embedding_bytes = db::embedding_to_bytes(&embedding);
        let q8_bytes = Quantizer::new(self.inner.config.embedding.dimensions)
            .quantize(&embedding)
            .map(|vector| quantize::pack_quantized(&vector))
            .ok();

        let doc_id = document_id.to_string();
        self.with_write_conn(move |conn| {
            episodes::update_episode_outcome(
                conn,
                &doc_id,
                outcome,
                confidence,
                experiment_id_owned.as_deref(),
                &verification_status,
                &search_text,
                &embedding_bytes,
                q8_bytes.as_deref(),
            )
        })
        .await?;

        #[cfg(feature = "hnsw")]
        self.sync_pending_hnsw_ops_best_effort("update_episode_outcome")
            .await;

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
        let outcome_owned = outcome.cloned();

        self.with_read_conn(move |conn| {
            episodes::search_episodes(conn, et.as_deref(), outcome_owned.as_ref(), limit)
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
        let k = top_k.unwrap_or(self.inner.config.search.default_top_k);
        let query_embedding = self.embed_text_internal(query).await?;

        #[cfg(feature = "hnsw")]
        let hnsw_hits = {
            let guard = self
                .inner
                .hnsw_index
                .read()
                .unwrap_or_else(|e| e.into_inner());
            let candidates = self.inner.config.search.candidate_pool_size.max(k * 3);
            match guard.search(&query_embedding, candidates) {
                Ok(hits) => hits,
                Err(err) => {
                    tracing::error!(
                        "HNSW search failed during explained search, falling back to brute-force path: {}",
                        err
                    );
                    Vec::new()
                }
            }
        };

        let q = query.to_string();
        let config = self.inner.config.search.clone();
        let ns_owned = to_owned_string_vec(namespaces);
        let st_owned: Option<Vec<SearchSourceType>> = source_types.map(|value| value.to_vec());

        #[cfg(feature = "hnsw")]
        let hnsw_hits_owned = hnsw_hits;

        self.with_read_conn(move |conn| {
            let ns_refs = as_str_slice(&ns_owned);
            let ns_slice: Option<&[&str]> = ns_refs.as_deref();
            let st_slice: Option<&[SearchSourceType]> = st_owned.as_deref();

            #[cfg(feature = "hnsw")]
            {
                if hnsw_hits_owned.is_empty() {
                    search::hybrid_search_detailed(
                        conn,
                        &q,
                        &query_embedding,
                        &config,
                        k,
                        ns_slice,
                        st_slice,
                        None,
                    )
                } else {
                    search::hybrid_search_with_hnsw_detailed(
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
            }
            #[cfg(not(feature = "hnsw"))]
            {
                search::hybrid_search_detailed(
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

    // ─── Embedding Displacement ───────────────────────────────

    /// Compute embedding displacement between two texts.
    pub async fn embedding_displacement(
        &self,
        text_a: &str,
        text_b: &str,
    ) -> Result<types::EmbeddingDisplacement, MemoryError> {
        let emb_a = self.embed_text_internal(text_a).await?;
        let emb_b = self.embed_text_internal(text_b).await?;
        Self::embedding_displacement_from_vecs(&emb_a, &emb_b)
    }

    /// Compute embedding displacement from pre-computed vectors.
    pub fn embedding_displacement_from_vecs(
        a: &[f32],
        b: &[f32],
    ) -> Result<types::EmbeddingDisplacement, MemoryError> {
        if a.len() != b.len() {
            return Err(MemoryError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }
        let cosine_sim = search::cosine_similarity(a, b);

        let euclidean_dist: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt();

        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        Ok(types::EmbeddingDisplacement {
            cosine_similarity: cosine_sim,
            euclidean_distance: euclidean_dist,
            magnitude_a: mag_a,
            magnitude_b: mag_b,
        })
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
        self.embed_text_internal(text).await
    }

    /// Embed multiple texts in a batch.
    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, MemoryError> {
        let owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        self.embed_batch_internal(owned).await
    }

    /// Get database statistics.
    pub async fn stats(&self) -> Result<MemoryStats, MemoryError> {
        let db_path = self.inner.paths.sqlite_path.clone();
        self.with_read_conn(move |conn| {
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
        self.with_read_conn(db::is_embeddings_dirty).await
    }

    /// Re-embed all facts, chunks, messages, and episodes. Call after changing embedding models.
    pub async fn reembed_all(&self) -> Result<usize, MemoryError> {
        let mut count = 0usize;
        let batch_size = self.inner.config.embedding.batch_size;
        let dims = self.inner.config.embedding.dimensions;

        // ─── Facts ──────────────────────────────────────────────────
        let fact_contents: Vec<(String, String)> = self
            .with_read_conn(|conn| {
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
            let embeddings = self.embed_batch_internal(texts).await?;
            for embedding in &embeddings {
                self.validate_embedding_dimensions(embedding)?;
            }

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

            self.with_write_conn(move |conn| {
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
            .with_read_conn(|conn| {
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
            let embeddings = self.embed_batch_internal(texts).await?;
            for embedding in &embeddings {
                self.validate_embedding_dimensions(embedding)?;
            }

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

            self.with_write_conn(move |conn| {
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
            .with_read_conn(|conn| {
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
            let embeddings = self.embed_batch_internal(texts).await?;
            for embedding in &embeddings {
                self.validate_embedding_dimensions(embedding)?;
            }

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

            self.with_write_conn(move |conn| {
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

        // ─── Episodes ───────────────────────────────────────────────
        let episode_data: Vec<(String, String)> = self
            .with_read_conn(|conn| {
                let mut stmt = conn.prepare("SELECT episode_id, search_text FROM episodes")?;
                let result = stmt
                    .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(result)
            })
            .await?;

        let mut episode_count = 0usize;
        for batch in episode_data.chunks(batch_size) {
            let texts: Vec<String> = batch.iter().map(|(_, text)| text.clone()).collect();
            let embeddings = self.embed_batch_internal(texts).await?;
            for embedding in &embeddings {
                self.validate_embedding_dimensions(embedding)?;
            }

            let quantizer = Quantizer::new(dims);
            let updates: Vec<(String, Vec<u8>, Option<Vec<u8>>)> = batch
                .iter()
                .zip(embeddings.iter())
                .map(|((episode_id, _), embedding)| {
                    let q8 = quantizer
                        .quantize(embedding)
                        .map(|vector| quantize::pack_quantized(&vector))
                        .ok();
                    (episode_id.clone(), db::embedding_to_bytes(embedding), q8)
                })
                .collect();

            self.with_write_conn(move |conn| {
                db::with_transaction(conn, |tx| {
                    for (episode_id, bytes, q8) in &updates {
                        tx.execute(
                            "UPDATE episodes
                             SET embedding = ?1,
                                 embedding_q8 = ?2,
                                 updated_at = datetime('now')
                             WHERE episode_id = ?3",
                            rusqlite::params![bytes, q8.as_deref(), episode_id],
                        )?;
                        #[cfg(feature = "hnsw")]
                        db::queue_pending_index_op(
                            tx,
                            &episodes::episode_item_key(episode_id),
                            "episode",
                            db::IndexOpKind::Upsert,
                        )?;
                    }
                    Ok(())
                })
            })
            .await?;

            episode_count += batch.len();
            count += batch.len();
            if episode_count % 100 < batch_size {
                tracing::info!(
                    episode_count,
                    "Re-embedded {} episodes so far",
                    episode_count
                );
            }
        }

        // Clear the dirty flag
        self.with_write_conn(db::clear_embeddings_dirty).await?;

        tracing::info!(
            facts = fact_count,
            chunks = chunk_count,
            messages = msg_count,
            episodes = episode_count,
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
        self.with_write_conn(|conn| {
            conn.execute_batch("VACUUM")?;
            Ok(())
        })
        .await
    }

    // ─── Projection Import ─────────────────────────────────────

    /// Import a projection envelope atomically (V10 legacy path).
    ///
    /// ## Phase status: compatibility / migration-only
    ///
    /// This method is the V10 legacy import path. New integrations should use
    /// [`import_projection_batch()`](Self::import_projection_batch) instead,
    /// which accepts the canonical `ProjectionImportBatchV1` format from
    /// `forge-memory-bridge`.
    ///
    /// **Removal condition**: removed when all callers migrate to the bridge pipeline.
    ///
    /// **Idempotent**: re-importing the same envelope (same `envelope_id` +
    /// `schema_version` + `content_digest`) returns a receipt with
    /// `was_duplicate = true` and does not modify data.
    ///
    /// **Atomic**: all records are committed in a single transaction. On any
    /// failure the entire import is rolled back — no partial visibility.
    ///
    /// **Provenance**: each imported record's metadata is tagged with the
    /// envelope_id and source_authority for traceability.
    #[allow(deprecated)]
    pub async fn import_envelope(
        &self,
        envelope: &projection_import::ImportEnvelope,
    ) -> Result<projection_import::ImportReceipt, MemoryError> {
        envelope.validate()?;

        let (eid, sv, cd) = envelope.dedupe_key();
        let eid = eid.to_string();
        let sv = sv.to_string();
        let cd = cd.to_string();

        // Fast-path idempotency check (reader conn)
        let already = {
            let eid_c = eid.clone();
            let sv_c = sv.clone();
            let cd_c = cd.clone();
            self.with_read_conn(move |conn| {
                projection_import::check_import_exists(conn, &eid_c, &sv_c, &cd_c)
            })
            .await?
        };
        if already {
            return Ok(projection_import::ImportReceipt {
                envelope_id: envelope.envelope_id.clone(),
                schema_version: envelope.schema_version.clone(),
                content_digest: envelope.content_digest.clone(),
                status: projection_import::ImportStatus::AlreadyImported,
                record_count: envelope.records.len(),
                imported_at: String::new(),
                was_duplicate: true,
                trace_id: envelope.trace_id.clone(),
            });
        }

        // Compute embeddings for all records outside the transaction.
        type PreparedImportRecord = (usize, Vec<f32>, Vec<u8>, Option<Vec<u8>>);
        let mut prepared: Vec<PreparedImportRecord> = Vec::new();
        for (i, record) in envelope.records.iter().enumerate() {
            let text = record.content_text();
            let embedding = self.embed_text_internal(text).await?;
            self.validate_embedding_dimensions(&embedding)?;
            let embedding_bytes = db::embedding_to_bytes(&embedding);
            let q8_bytes = quantize::Quantizer::new(self.inner.config.embedding.dimensions)
                .quantize(&embedding)
                .map(|qv| quantize::pack_quantized(&qv))
                .ok();
            prepared.push((i, embedding, embedding_bytes, q8_bytes));
        }

        // Clone data needed for the write transaction closure
        let envelope_c = envelope.clone();
        let record_count = envelope.records.len();

        let receipt = self.with_write_conn(move |conn| {
            db::with_transaction(conn, |tx| {
                // Double-check idempotency under write lock
                if projection_import::check_import_exists(
                    tx,
                    envelope_c.envelope_id.as_str(),
                    &envelope_c.schema_version,
                    &envelope_c.content_digest,
                )? {
                    return Ok(projection_import::ImportReceipt {
                        envelope_id: envelope_c.envelope_id.clone(),
                        schema_version: envelope_c.schema_version.clone(),
                        content_digest: envelope_c.content_digest.clone(),
                        status: projection_import::ImportStatus::AlreadyImported,
                        record_count,
                        imported_at: String::new(),
                        was_duplicate: true,
                        trace_id: envelope_c.trace_id.clone(),
                    });
                }

                // Insert each record
                for (i, _embedding, embedding_bytes, q8_bytes) in &prepared {
                    let record = &envelope_c.records[*i];
                    // Build provenance metadata
                    let provenance_meta = serde_json::json!({
                        "import_envelope_id": envelope_c.envelope_id.as_str(),
                        "import_source_authority": envelope_c.source_authority,
                        "import_schema_version": envelope_c.schema_version,
                    });

                    match record {
                        projection_import::ImportRecord::Fact {
                            content,
                            source,
                            metadata,
                        } => {
                            let fact_id = uuid::Uuid::new_v4().to_string();
                            let mut meta = metadata.clone().unwrap_or(serde_json::json!({}));
                            if let Some(obj) = meta.as_object_mut() {
                                obj.insert("_import".into(), provenance_meta.clone());
                            }
                            if let Some(trace_id) = &envelope_c.trace_id {
                                if let Some(obj) = meta.as_object_mut() {
                                    obj.insert(
                                        "trace_id".into(),
                                        serde_json::Value::String(trace_id.0.clone()),
                                    );
                                }
                            }

                            knowledge::insert_fact_in_tx(
                                tx,
                                &fact_id,
                                &envelope_c.namespace,
                                content,
                                embedding_bytes,
                                q8_bytes.as_deref(),
                                source.as_deref(),
                                Some(&meta),
                            )?;
                        }
                        projection_import::ImportRecord::Episode {
                            document_id,
                            meta,
                        } => {
                            let episode_id = uuid::Uuid::new_v4().to_string();
                            let meta_json = serde_json::to_value(meta).map_err(|e| {
                                MemoryError::ImportInvalid {
                                    reason: format!("failed to serialize episode meta: {e}"),
                                }
                            })?;
                            let cause_ids_json =
                                serde_json::to_string(&meta.cause_ids).unwrap_or_default();
                            let verification_json =
                                serde_json::to_string(&meta.verification_status)
                                    .unwrap_or_default();
                            let search_text = format!(
                                "{} {} {} {}",
                                meta.effect_type,
                                meta.outcome.as_str(),
                                meta.experiment_id.as_deref().unwrap_or(""),
                                cause_ids_json
                            );

                            // Avoid unused variable warning
                            let _ = meta_json;

                            tx.execute(
                                "INSERT INTO episodes
                                 (episode_id, document_id, cause_ids, effect_type, outcome,
                                  confidence, verification_status, experiment_id,
                                  search_text, embedding, embedding_q8, trace_id)
                                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                                rusqlite::params![
                                    episode_id,
                                    document_id,
                                    cause_ids_json,
                                    meta.effect_type,
                                    meta.outcome.as_str(),
                                    meta.confidence,
                                    verification_json,
                                    meta.experiment_id,
                                    search_text,
                                    embedding_bytes,
                                    q8_bytes.as_deref(),
                                    envelope_c
                                        .trace_id
                                        .as_ref()
                                        .map(|t| t.as_str().to_string()),
                                ],
                            )?;

                            // FTS entry
                            tx.execute(
                                "INSERT INTO episodes_rowid_map (episode_id, document_id) VALUES (?1, ?2)",
                                rusqlite::params![episode_id, document_id],
                            )?;
                            let fts_rowid = tx.last_insert_rowid();
                            tx.execute(
                                "INSERT INTO episodes_fts(rowid, content) VALUES (?1, ?2)",
                                rusqlite::params![fts_rowid, search_text],
                            )?;

                            // Causal edges
                            for (ordinal, cause_id) in meta.cause_ids.iter().enumerate() {
                                tx.execute(
                                    "INSERT OR IGNORE INTO episode_causes (episode_id, cause_node_id, ordinal)
                                     VALUES (?1, ?2, ?3)",
                                    rusqlite::params![episode_id, cause_id, ordinal as i64],
                                )?;
                            }

                            // HNSW queue
                            #[cfg(feature = "hnsw")]
                            db::enqueue_pending_index_op(
                                tx,
                                &format!("episode:{}", episode_id),
                                "episode",
                                db::PendingIndexOpKind::Upsert,
                            )?;
                        }
                    }
                }

                // Log the import
                projection_import::insert_import_log(
                    tx,
                    &envelope_c,
                    &projection_import::ImportStatus::Complete,
                    record_count,
                )?;

                Ok(projection_import::ImportReceipt {
                    envelope_id: envelope_c.envelope_id.clone(),
                    schema_version: envelope_c.schema_version.clone(),
                    content_digest: envelope_c.content_digest.clone(),
                    status: projection_import::ImportStatus::Complete,
                    record_count,
                    imported_at: chrono::Utc::now().to_rfc3339(),
                    was_duplicate: false,
                    trace_id: envelope_c.trace_id.clone(),
                })
            })
        })
        .await?;

        #[cfg(feature = "hnsw")]
        self.sync_pending_hnsw_ops_best_effort("import_envelope")
            .await;

        Ok(receipt)
    }

    /// Check whether an envelope has already been imported.
    #[allow(deprecated)]
    pub async fn import_status(
        &self,
        envelope_id: &projection_import::EnvelopeId,
    ) -> Result<Vec<projection_import::ImportReceipt>, MemoryError> {
        let eid = envelope_id.0.clone();
        self.with_read_conn(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT envelope_id, schema_version, content_digest, status,
                        record_count, imported_at, trace_id
                 FROM import_log
                 WHERE envelope_id = ?1
                 ORDER BY imported_at DESC",
            )?;
            let rows = stmt
                .query_map(rusqlite::params![eid], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, i64>(4)?,
                        row.get::<_, String>(5)?,
                        row.get::<_, Option<String>>(6)?,
                    ))
                })?
                .collect::<Result<Vec<_>, _>>()?;

            Ok(rows
                .into_iter()
                .map(|(eid, sv, cd, status, rc, ts, tid)| {
                    let status_parsed = projection_import::ImportStatus::from_str_value(&status);
                    let was_dup = matches!(
                        status_parsed,
                        projection_import::ImportStatus::AlreadyImported
                    );
                    projection_import::ImportReceipt {
                        envelope_id: projection_import::EnvelopeId(eid),
                        schema_version: sv,
                        content_digest: cd,
                        status: status_parsed,
                        record_count: rc as usize,
                        imported_at: ts,
                        was_duplicate: was_dup,
                        trace_id: tid.map(crate::types::TraceId::new),
                    }
                })
                .collect())
        })
        .await
    }

    /// List recent imports, optionally filtered by namespace.
    #[allow(deprecated)]
    pub async fn list_imports(
        &self,
        namespace: Option<&str>,
        limit: usize,
    ) -> Result<Vec<projection_import::ImportReceipt>, MemoryError> {
        let ns = namespace.map(|s| s.to_string());
        self.with_read_conn(move |conn| {
            projection_import::query_import_log(conn, ns.as_deref(), limit)
        })
        .await
    }

    /// Get the most recent successful import timestamp for a namespace.
    pub async fn last_import_at(&self, namespace: &str) -> Result<Option<String>, MemoryError> {
        let ns = namespace.to_string();
        self.with_read_conn(move |conn| projection_import::last_import_at(conn, &ns))
            .await
    }

    // ── Projection batch import (V11+, canonical stack v4) ──────────

    /// Import a projection batch from `forge-memory-bridge`.
    ///
    /// This is the canonical in-process import path for the stack:
    /// `ExportEnvelopeV1 -> forge-memory-bridge -> ProjectionImportBatchV1
    /// -> semantic-memory import transaction`.
    ///
    /// The old `import_envelope()` method remains functional for backward
    /// compatibility during the migration cycle. JSON parsing is retained
    /// only via [`import_projection_batch_json_compat()`](Self::import_projection_batch_json_compat).
    pub async fn import_projection_batch(
        &self,
        batch: &ProjectionImportBatchV1,
    ) -> Result<ProjectionImportResult, MemoryError> {
        if batch.schema_version != PROJECTION_IMPORT_BATCH_V1_SCHEMA {
            return Err(MemoryError::ImportInvalid {
                reason: format!(
                    "unsupported schema_version: {}; expected {}",
                    batch.schema_version, PROJECTION_IMPORT_BATCH_V1_SCHEMA
                ),
            });
        }

        let source_envelope_id = batch.source_envelope_id.as_str().to_string();
        let schema_version = batch.schema_version.clone();
        let export_schema_version = batch.export_schema_version.clone();
        let content_digest = batch.content_digest.hex().to_string();
        let source_authority = batch.source_authority.clone();
        let scope_namespace = batch.scope_key.namespace.clone();
        let scope_domain = batch.scope_key.domain.clone();
        let scope_workspace_id = batch.scope_key.workspace_id.clone();
        let scope_repo_id = batch.scope_key.repo_id.clone();
        let trace_id = batch.trace_ctx.as_ref().map(|ctx| ctx.trace_id.clone());
        let source_exported_at = Some(batch.source_exported_at.clone());
        let transformed_at = Some(batch.transformed_at.clone());

        // Fast-path idempotency check
        let sei_c = source_envelope_id.clone();
        let sv_c = schema_version.clone();
        let cd_c = content_digest.clone();
        let record_len = batch.records.len();
        let already = self
            .with_read_conn(move |conn| {
                projection_storage::check_projection_import_exists(conn, &sei_c, &sv_c, &cd_c)
            })
            .await?;

        if already {
            return Ok(ProjectionImportResult {
                source_envelope_id,
                status: "already_imported".into(),
                record_count: record_len,
                was_duplicate: true,
            });
        }

        // Process records via typed payloads
        let mut claim_count = 0usize;
        let mut relation_count = 0usize;
        let mut episode_count = 0usize;
        let mut alias_count = 0usize;
        let mut evidence_count = 0usize;

        let mut claim_rows = Vec::new();
        let mut relation_rows = Vec::new();
        let mut alias_rows = Vec::new();
        let mut evidence_rows = Vec::new();
        let mut episode_rows = Vec::new();

        for record in &batch.records {
            match record {
                ImportProjectionRecord::ClaimVersion(cv) => {
                    claim_count += 1;
                    claim_rows.push(projection_storage::ClaimVersionRow {
                        claim_version_id: cv.claim_version_id.as_str().to_string(),
                        claim_id: cv.claim_id.as_str().to_string(),
                        claim_state: cv.claim_state.as_str().to_string(),
                        projection_family: cv.projection_family.clone(),
                        subject_entity_id: cv.subject_entity_id.as_str().to_string(),
                        predicate: cv.predicate.clone(),
                        object_anchor: cv.object_anchor.to_string(),
                        scope_namespace: cv.scope_key.namespace.clone(),
                        scope_domain: cv.scope_key.domain.clone(),
                        scope_workspace_id: cv.scope_key.workspace_id.clone(),
                        scope_repo_id: cv.scope_key.repo_id.clone(),
                        valid_from: cv.valid_from.clone(),
                        valid_to: cv.valid_to.clone(),
                        recorded_at: String::new(),
                        preferred_open: cv.preferred_open,
                        source_envelope_id: cv.source_envelope_id.as_str().to_string(),
                        source_authority: cv.source_authority.clone(),
                        trace_id: cv.trace_ctx.as_ref().map(|ctx| ctx.trace_id.clone()),
                        freshness: cv.freshness.as_str().to_string(),
                        contradiction_status: serde_json::to_string(&cv.contradiction_status)
                            .unwrap_or_else(|_| "\"none\"".into()),
                        supersedes_claim_version_id: cv
                            .supersedes_claim_version_id
                            .as_ref()
                            .map(|id| id.as_str().to_string()),
                        content: cv.content.clone(),
                        confidence: cv.confidence,
                        content_digest: Some(batch.content_digest.hex().to_string()),
                        metadata: cv.metadata.as_ref().map(|v| v.to_string()),
                    });
                }
                ImportProjectionRecord::RelationVersion(rv) => {
                    relation_count += 1;
                    relation_rows.push(projection_storage::RelationVersionRow {
                        relation_version_id: rv.relation_version_id.as_str().to_string(),
                        subject_entity_id: rv.subject_entity_id.as_str().to_string(),
                        predicate: rv.predicate.clone(),
                        object_anchor: rv.object_anchor.to_string(),
                        scope_namespace: rv.scope_key.namespace.clone(),
                        scope_domain: rv.scope_key.domain.clone(),
                        scope_workspace_id: rv.scope_key.workspace_id.clone(),
                        scope_repo_id: rv.scope_key.repo_id.clone(),
                        claim_id: rv.claim_id.as_ref().map(|id| id.as_str().to_string()),
                        source_episode_id: rv
                            .source_episode_id
                            .as_ref()
                            .map(|id| id.as_str().to_string()),
                        valid_from: rv.valid_from.clone(),
                        valid_to: rv.valid_to.clone(),
                        recorded_at: String::new(),
                        preferred_open: rv.preferred_open,
                        supersedes_relation_version_id: rv
                            .supersedes_relation_version_id
                            .as_ref()
                            .map(|id| id.as_str().to_string()),
                        contradiction_status: serde_json::to_string(&rv.contradiction_status)
                            .unwrap_or_else(|_| "\"none\"".into()),
                        source_confidence: rv.source_confidence,
                        projection_family: rv.projection_family.clone(),
                        source_envelope_id: rv.source_envelope_id.as_str().to_string(),
                        source_authority: rv.source_authority.clone(),
                        trace_id: rv.trace_ctx.as_ref().map(|ctx| ctx.trace_id.clone()),
                        freshness: rv.freshness.as_str().to_string(),
                        metadata: rv.metadata.as_ref().map(|v| v.to_string()),
                    });
                }
                ImportProjectionRecord::EntityAlias(ea) => {
                    alias_count += 1;
                    alias_rows.push(projection_storage::EntityAliasRow {
                        canonical_entity_id: ea.canonical_entity_id.as_str().to_string(),
                        alias_text: ea.alias_text.clone(),
                        alias_source: ea.alias_source.clone(),
                        match_evidence: ea.match_evidence.as_ref().map(|v| v.to_string()),
                        confidence: ea.confidence,
                        merge_decision: encode_merge_decision(&ea.merge_decision),
                        scope_namespace: ea.scope.namespace.clone(),
                        scope_domain: ea.scope.domain.clone(),
                        scope_workspace_id: ea.scope.workspace_id.clone(),
                        scope_repo_id: ea.scope.repo_id.clone(),
                        review_state: encode_review_state(&ea.review_state),
                        is_human_confirmed: ea.is_human_confirmed,
                        is_human_confirmed_final: ea.is_human_confirmed_final,
                        superseded_by_entity_id: ea
                            .superseded_by_entity_id
                            .as_ref()
                            .map(|id| id.as_str().to_string()),
                        split_from_entity_id: ea
                            .split_from_entity_id
                            .as_ref()
                            .map(|id| id.as_str().to_string()),
                        source_envelope_id: ea.source_envelope_id.as_str().to_string(),
                        recorded_at: String::new(),
                    });
                }
                ImportProjectionRecord::EvidenceRef(er) => {
                    evidence_count += 1;
                    evidence_rows.push(projection_storage::EvidenceRefRow {
                        claim_id: er.claim_id.as_str().to_string(),
                        claim_version_id: er
                            .claim_version_id
                            .as_ref()
                            .map(|id| id.as_str().to_string()),
                        fetch_handle: er.fetch_handle.clone(),
                        source_authority: er.source_authority.clone(),
                        source_envelope_id: er.source_envelope_id.as_str().to_string(),
                        recorded_at: String::new(),
                        metadata: er.metadata.as_ref().map(|v| v.to_string()),
                    });
                }
                ImportProjectionRecord::Episode(ep) => {
                    episode_count += 1;
                    episode_rows.push(projection_storage::EpisodeLinkRow {
                        episode_id: ep.episode_id.as_str().to_string(),
                        document_id: ep.document_id.clone(),
                        cause_ids: serde_json::to_string(&ep.cause_ids)
                            .unwrap_or_else(|_| "[]".into()),
                        effect_type: ep.effect_type.clone(),
                        outcome: ep.outcome.clone(),
                        confidence: ep.confidence,
                        experiment_id: ep.experiment_id.clone(),
                        source_envelope_id: ep.source_envelope_id.as_str().to_string(),
                        source_authority: ep.source_authority.clone(),
                        trace_id: ep.trace_ctx.as_ref().map(|ctx| ctx.trace_id.clone()),
                        recorded_at: String::new(),
                        metadata: ep.metadata.as_ref().map(|v| v.to_string()),
                    });
                }
            }
        }

        let total_count = record_len;
        let batch_id = uuid::Uuid::new_v4().to_string();

        self.with_write_conn(move |conn| {
            let mut claim_rows = claim_rows;
            let mut relation_rows = relation_rows;
            let mut alias_rows = alias_rows;
            let mut evidence_rows = evidence_rows;
            let mut episode_rows = episode_rows;
            db::with_transaction(conn, |tx| {
                // Double-check idempotency under write lock
                if projection_storage::check_projection_import_exists(
                    tx,
                    &source_envelope_id,
                    &schema_version,
                    &content_digest,
                )? {
                    return Ok(ProjectionImportResult {
                        source_envelope_id,
                        status: "already_imported".into(),
                        record_count: total_count,
                        was_duplicate: true,
                    });
                }

                let imported_at = chrono::Utc::now().to_rfc3339();
                for cv in &mut claim_rows {
                    cv.recorded_at = imported_at.clone();
                }
                for rv in &mut relation_rows {
                    rv.recorded_at = imported_at.clone();
                }
                for ea in &mut alias_rows {
                    ea.recorded_at = imported_at.clone();
                }
                for er in &mut evidence_rows {
                    er.recorded_at = imported_at.clone();
                }
                for el in &mut episode_rows {
                    el.recorded_at = imported_at.clone();
                }

                let log_row = projection_storage::ProjectionImportLogRow {
                    batch_id: batch_id.clone(),
                    source_envelope_id: source_envelope_id.clone(),
                    schema_version: schema_version.clone(),
                    export_schema_version: export_schema_version.clone(),
                    content_digest: content_digest.clone(),
                    source_authority: source_authority.clone(),
                    scope_namespace: scope_namespace.clone(),
                    scope_domain: scope_domain.clone(),
                    scope_workspace_id: scope_workspace_id.clone(),
                    scope_repo_id: scope_repo_id.clone(),
                    trace_id: trace_id.clone(),
                    record_count: total_count,
                    claim_count,
                    relation_count,
                    episode_count,
                    alias_count,
                    evidence_count,
                    status: "complete".into(),
                    source_exported_at: source_exported_at.clone(),
                    transformed_at: transformed_at.clone(),
                    imported_at: imported_at.clone(),
                };

                // Insert claim versions
                for cv in &claim_rows {
                    projection_storage::insert_claim_version(tx, cv)?;
                }

                // Insert relation versions
                for rv in &relation_rows {
                    projection_storage::insert_relation_version(tx, rv)?;
                }

                // Insert entity aliases
                for ea in &alias_rows {
                    projection_storage::insert_entity_alias(tx, ea)?;
                }

                // Insert evidence refs
                for er in &evidence_rows {
                    projection_storage::insert_evidence_ref(tx, er)?;
                }

                // Insert episode links
                for el in &episode_rows {
                    projection_storage::insert_episode_link(tx, el)?;
                }

                // Insert derivation edges for provenance lineage.
                // Evidence refs derive from claim versions (Issue #2 fix:
                // use claim_version_id as target when available, fall back
                // to claim_id with target_kind="claim").
                for er in &evidence_rows {
                    let (target_kind, target_id) = if let Some(cvid) = &er.claim_version_id {
                        ("claim_version", cvid.as_str())
                    } else {
                        ("claim", er.claim_id.as_str())
                    };
                    projection_storage::insert_derivation_edge(
                        tx,
                        "evidence_ref",
                        &er.fetch_handle,
                        target_kind,
                        target_id,
                        "supports",
                        "on_source_change",
                    )?;
                }

                // Insert import log
                projection_storage::insert_projection_import_log(tx, &log_row)?;

                Ok(ProjectionImportResult {
                    source_envelope_id,
                    status: "complete".into(),
                    record_count: total_count,
                    was_duplicate: false,
                })
            })
        })
        .await
    }

    /// Deserialize and import a projection batch from JSON.
    ///
    /// This is a compatibility boundary for callers that still cross the
    /// in-process seam as serialized JSON. New code should pass
    /// `ProjectionImportBatchV1` directly to `import_projection_batch()`.
    pub async fn import_projection_batch_json_compat(
        &self,
        batch_json: &str,
    ) -> Result<ProjectionImportResult, MemoryError> {
        let batch = decode_projection_batch_json_compat(batch_json)?;
        self.import_projection_batch(&batch).await
    }

    // ── Projection query APIs (V11+) ──────────────────────────────

    /// Query the V11 projection import log.
    pub async fn query_projection_imports(
        &self,
        scope_namespace: Option<&str>,
        limit: usize,
    ) -> Result<Vec<ProjectionImportLogEntry>, MemoryError> {
        let ns = scope_namespace.map(|s| s.to_string());
        self.with_read_conn(move |conn| {
            let rows = projection_storage::query_projection_import_log(conn, ns.as_deref(), limit)?;
            Ok(rows
                .into_iter()
                .map(|r| ProjectionImportLogEntry {
                    batch_id: r.batch_id,
                    source_envelope_id: r.source_envelope_id,
                    schema_version: r.schema_version,
                    export_schema_version: r.export_schema_version,
                    content_digest: r.content_digest,
                    source_authority: r.source_authority,
                    scope_namespace: r.scope_namespace,
                    record_count: r.record_count,
                    claim_count: r.claim_count,
                    relation_count: r.relation_count,
                    episode_count: r.episode_count,
                    alias_count: r.alias_count,
                    evidence_count: r.evidence_count,
                    status: r.status,
                    source_exported_at: r.source_exported_at,
                    transformed_at: r.transformed_at,
                    imported_at: r.imported_at,
                })
                .collect())
        })
        .await
    }

    /// Query imported claim projection rows through the supported public read surface.
    pub async fn query_claim_versions(
        &self,
        query: ProjectionQuery,
    ) -> Result<Vec<ProjectionClaimVersion>, MemoryError> {
        self.with_read_conn(move |conn| projection_storage::query_claim_versions(conn, &query))
            .await
    }

    /// Query imported relation projection rows through the supported public read surface.
    pub async fn query_relation_versions(
        &self,
        query: ProjectionQuery,
    ) -> Result<Vec<ProjectionRelationVersion>, MemoryError> {
        self.with_read_conn(move |conn| projection_storage::query_relation_versions(conn, &query))
            .await
    }

    /// Query imported episode projection rows through the supported public read surface.
    pub async fn query_episodes(
        &self,
        query: ProjectionQuery,
    ) -> Result<Vec<ProjectionEpisode>, MemoryError> {
        self.with_read_conn(move |conn| projection_storage::query_episode_rows(conn, &query))
            .await
    }

    /// Query imported entity-alias rows through the supported public read surface.
    pub async fn query_entity_aliases(
        &self,
        query: ProjectionQuery,
    ) -> Result<Vec<ProjectionEntityAlias>, MemoryError> {
        self.with_read_conn(move |conn| projection_storage::query_entity_aliases(conn, &query))
            .await
    }

    /// Query imported evidence-reference rows through the supported public read surface.
    pub async fn query_evidence_refs(
        &self,
        query: ProjectionQuery,
    ) -> Result<Vec<ProjectionEvidenceRef>, MemoryError> {
        self.with_read_conn(move |conn| projection_storage::query_evidence_refs(conn, &query))
            .await
    }

    /// Invalidate derivation edges matching a trigger mode, bounded by source artifact.
    ///
    /// Returns the number of edges invalidated. This enables bounded recomputation:
    /// only derived artifacts downstream of the specified source are affected.
    pub async fn invalidate_derivations(
        &self,
        source_kind: &str,
        source_id: &str,
        trigger_mode: &str,
        reason: &str,
    ) -> Result<usize, MemoryError> {
        let sk = source_kind.to_string();
        let si = source_id.to_string();
        let tm = trigger_mode.to_string();
        let r = reason.to_string();
        self.with_write_conn(move |conn| {
            projection_storage::invalidate_derivation_edges(conn, &sk, &si, &tm, &r)
        })
        .await
    }

    /// Execute raw SQL. For testing only — not part of the stable public API.
    #[cfg(any(test, feature = "testing"))]
    pub async fn raw_execute(&self, sql: &str, params: Vec<String>) -> Result<usize, MemoryError> {
        let sql = sql.to_string();
        self.with_write_conn(move |conn| {
            let param_refs: Vec<&dyn rusqlite::types::ToSql> = params
                .iter()
                .map(|s| s as &dyn rusqlite::types::ToSql)
                .collect();
            Ok(conn.execute(&sql, &*param_refs)?)
        })
        .await
    }
}
