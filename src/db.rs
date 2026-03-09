//! Database initialization, migrations, integrity checks, and durable sidecar state.

use crate::config::{EmbeddingConfig, MemoryLimits, PoolConfig};
use crate::error::MemoryError;
use crate::quantize::unpack_quantized;
use crate::types::{EpisodeOutcome, Role, VerificationStatus};
use rusqlite::{params, Connection, OpenFlags};
use std::path::Path;

/// V1 migration: full schema.
const MIGRATION_V1: &str = r#"
-- CONVERSATIONS
CREATE TABLE sessions (
    id          TEXT PRIMARY KEY,
    channel     TEXT NOT NULL DEFAULT 'repl',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    metadata    TEXT
);

CREATE INDEX idx_sessions_updated ON sessions(updated_at DESC);

CREATE TABLE messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    content     TEXT NOT NULL,
    token_count INTEGER,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    metadata    TEXT
);

CREATE INDEX idx_messages_session ON messages(session_id, created_at ASC);
CREATE INDEX idx_messages_created ON messages(created_at DESC);

-- KNOWLEDGE (Facts)
CREATE TABLE facts (
    id          TEXT PRIMARY KEY,
    namespace   TEXT NOT NULL DEFAULT 'general',
    content     TEXT NOT NULL,
    source      TEXT,
    embedding   BLOB,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    metadata    TEXT
);

CREATE INDEX idx_facts_namespace ON facts(namespace);
CREATE INDEX idx_facts_updated ON facts(updated_at DESC);

CREATE TABLE facts_rowid_map (
    rowid       INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id     TEXT NOT NULL UNIQUE REFERENCES facts(id) ON DELETE CASCADE
);

CREATE VIRTUAL TABLE facts_fts USING fts5(
    content,
    content='',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

-- DOCUMENTS (Chunked content)
CREATE TABLE documents (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    source_path TEXT,
    namespace   TEXT NOT NULL DEFAULT 'general',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    metadata    TEXT
);

CREATE TABLE chunks (
    id          TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content     TEXT NOT NULL,
    token_count INTEGER,
    embedding   BLOB,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_chunks_document ON chunks(document_id, chunk_index ASC);

CREATE TABLE chunks_rowid_map (
    rowid       INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id    TEXT NOT NULL UNIQUE REFERENCES chunks(id) ON DELETE CASCADE
);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content,
    content='',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

-- EMBEDDING METADATA
CREATE TABLE embedding_metadata (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    model_name  TEXT NOT NULL,
    dimensions  INTEGER NOT NULL,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"#;

/// V2 migration: message embeddings for conversation search.
const MIGRATION_V2: &str = r#"
ALTER TABLE messages ADD COLUMN embedding BLOB;

CREATE TABLE messages_rowid_map (
    rowid       INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id  INTEGER NOT NULL UNIQUE REFERENCES messages(id) ON DELETE CASCADE
);

CREATE VIRTUAL TABLE messages_fts USING fts5(
    content,
    content='',
    content_rowid='rowid',
    tokenize='porter unicode61'
);
"#;

/// V3 migration: embedding staleness tracking.
const MIGRATION_V3: &str = r#"
ALTER TABLE embedding_metadata ADD COLUMN embeddings_dirty INTEGER NOT NULL DEFAULT 0;
"#;

/// V4 migration: HNSW metadata tracking.
const MIGRATION_V4: &str = r#"
CREATE TABLE IF NOT EXISTS hnsw_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"#;

/// V5 migration: quantized embeddings + HNSW keymap persistence.
const MIGRATION_V5: &str = r#"
ALTER TABLE facts ADD COLUMN embedding_q8 BLOB;
ALTER TABLE chunks ADD COLUMN embedding_q8 BLOB;
ALTER TABLE messages ADD COLUMN embedding_q8 BLOB;

CREATE TABLE IF NOT EXISTS hnsw_keymap (
    node_id     INTEGER PRIMARY KEY,
    item_key    TEXT NOT NULL UNIQUE,
    deleted     INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX idx_hnsw_keymap_key ON hnsw_keymap(item_key);
"#;

/// V6 migration: episodes table for causal tracking.
const MIGRATION_V6: &str = r#"
CREATE TABLE IF NOT EXISTS episodes (
    document_id TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    cause_ids TEXT NOT NULL,
    effect_type TEXT NOT NULL,
    outcome TEXT NOT NULL DEFAULT 'pending',
    confidence REAL NOT NULL DEFAULT 0.0,
    verification_status TEXT NOT NULL DEFAULT '{"status":"unverified"}',
    experiment_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_episodes_effect_type ON episodes(effect_type);
CREATE INDEX IF NOT EXISTS idx_episodes_outcome ON episodes(outcome);
CREATE INDEX IF NOT EXISTS idx_episodes_experiment_id ON episodes(experiment_id);
"#;

/// V7 migration: searchable episodes + durable sidecar journal.
const MIGRATION_V7: &str = r#"
ALTER TABLE episodes ADD COLUMN updated_at TEXT NOT NULL DEFAULT (datetime('now'));
ALTER TABLE episodes ADD COLUMN search_text TEXT NOT NULL DEFAULT '';
ALTER TABLE episodes ADD COLUMN embedding BLOB;
ALTER TABLE episodes ADD COLUMN embedding_q8 BLOB;

CREATE TABLE IF NOT EXISTS episodes_rowid_map (
    rowid       INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL UNIQUE REFERENCES episodes(document_id) ON DELETE CASCADE
);

CREATE VIRTUAL TABLE episodes_fts USING fts5(
    content,
    content='',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS pending_index_ops (
    item_key      TEXT PRIMARY KEY,
    entity_type   TEXT NOT NULL,
    op_kind       TEXT NOT NULL CHECK (op_kind IN ('upsert', 'delete')),
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_error    TEXT,
    updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT OR IGNORE INTO hnsw_metadata (key, value) VALUES ('sidecar_dirty', '0');

UPDATE episodes
SET search_text = TRIM(
    COALESCE(effect_type, '') || ' ' ||
    COALESCE(outcome, '') || ' ' ||
    COALESCE(experiment_id, '') || ' ' ||
    COALESCE(cause_ids, '')
)
WHERE search_text = '';

INSERT OR IGNORE INTO episodes_rowid_map (document_id)
SELECT document_id FROM episodes;

INSERT INTO episodes_fts (rowid, content)
SELECT rm.rowid, e.search_text
FROM episodes_rowid_map rm
JOIN episodes e ON e.document_id = rm.document_id;
"#;

/// V8 migration: durable episode trace IDs.
const MIGRATION_V8: &str = r#"
ALTER TABLE episodes ADD COLUMN trace_id TEXT;
"#;

/// V9 migration: first-class episode identity + normalized causal edge table.
///
/// Rebuilds the episodes table so `episode_id` is the primary key while
/// `document_id` becomes a non-unique FK allowing multiple episodes per doc.
/// Adds `episode_causes` for normalized causal backlinks.
///
/// Applied via `run_migration_v9()` because it requires table rebuild.
const MIGRATION_V9: &str = "";

/// Ordered list of migrations.
const MIGRATIONS: &[(u32, &str)] = &[
    (1, MIGRATION_V1),
    (2, MIGRATION_V2),
    (3, MIGRATION_V3),
    (4, MIGRATION_V4),
    (5, MIGRATION_V5),
    (6, MIGRATION_V6),
    (7, MIGRATION_V7),
    (8, MIGRATION_V8),
    (9, MIGRATION_V9),
    (10, crate::projection_import::MIGRATION_V10),
    (11, crate::projection_storage::MIGRATION_V11),
    (12, crate::projection_storage::MIGRATION_V12),
    (13, crate::projection_storage::MIGRATION_V13),
];

/// Maximum schema version this build supports.
pub const MAX_SCHEMA_VERSION: u32 = 13;

/// Procedural migration for V9: rebuild episodes table with episode_id PK.
fn run_migration_v9(conn: &Connection) -> Result<(), MemoryError> {
    // Check if episodes table exists (fresh DBs won't have it yet at V6)
    let episodes_exist: bool = conn
        .query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='episodes'",
            [],
            |row| row.get(0),
        )
        .unwrap_or(false);

    if !episodes_exist {
        // No episodes table to migrate; create the target schema directly
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS episode_causes (
                 episode_id    TEXT NOT NULL,
                 cause_node_id TEXT NOT NULL,
                 ordinal       INTEGER NOT NULL DEFAULT 0,
                 PRIMARY KEY (episode_id, cause_node_id)
             );
             CREATE INDEX IF NOT EXISTS idx_episode_causes_cause ON episode_causes(cause_node_id);",
        )?;
        return Ok(());
    }

    // Disable foreign keys for table rebuild
    conn.execute_batch("PRAGMA foreign_keys = OFF;")?;

    conn.execute_batch(
        "CREATE TABLE episodes_new (
             episode_id  TEXT PRIMARY KEY,
             document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
             cause_ids   TEXT NOT NULL,
             effect_type TEXT NOT NULL,
             outcome     TEXT NOT NULL DEFAULT 'pending',
             confidence  REAL NOT NULL DEFAULT 0.0,
             verification_status TEXT NOT NULL DEFAULT '{\"status\":\"unverified\"}',
             experiment_id TEXT,
             created_at  TEXT NOT NULL DEFAULT (datetime('now')),
             updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
             search_text TEXT NOT NULL DEFAULT '',
             embedding   BLOB,
             embedding_q8 BLOB,
             trace_id    TEXT
         )",
    )?;

    // Migrate existing data with deterministic episode_id
    conn.execute_batch(
        "INSERT INTO episodes_new
             (episode_id, document_id, cause_ids, effect_type, outcome, confidence,
              verification_status, experiment_id, created_at, updated_at,
              search_text, embedding, embedding_q8, trace_id)
         SELECT
             document_id || '-ep0',
             document_id, cause_ids, effect_type, outcome, confidence,
             verification_status, experiment_id, created_at, updated_at,
             search_text, embedding, embedding_q8, trace_id
         FROM episodes",
    )?;

    conn.execute_batch("DROP TABLE episodes")?;
    conn.execute_batch("ALTER TABLE episodes_new RENAME TO episodes")?;

    conn.execute_batch(
        "CREATE INDEX idx_episodes_document_id ON episodes(document_id);
         CREATE INDEX idx_episodes_effect_type ON episodes(effect_type);
         CREATE INDEX idx_episodes_outcome ON episodes(outcome);
         CREATE INDEX idx_episodes_experiment_id ON episodes(experiment_id);",
    )?;

    // Rebuild episodes_rowid_map with episode_id
    conn.execute_batch(
        "DROP TABLE IF EXISTS episodes_rowid_map;
         CREATE TABLE episodes_rowid_map (
             rowid       INTEGER PRIMARY KEY AUTOINCREMENT,
             episode_id  TEXT NOT NULL UNIQUE,
             document_id TEXT
         );
         INSERT INTO episodes_rowid_map (episode_id, document_id)
         SELECT episode_id, document_id FROM episodes;",
    )?;

    // Rebuild episodes FTS
    conn.execute_batch(
        "DROP TABLE IF EXISTS episodes_fts;
         CREATE VIRTUAL TABLE episodes_fts USING fts5(
             content,
             content='',
             content_rowid='rowid',
             tokenize='porter unicode61'
         );
         INSERT INTO episodes_fts (rowid, content)
         SELECT rm.rowid, e.search_text
         FROM episodes_rowid_map rm
         JOIN episodes e ON e.episode_id = rm.episode_id;",
    )?;

    // Normalized causal edge table
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS episode_causes (
             episode_id    TEXT NOT NULL,
             cause_node_id TEXT NOT NULL,
             ordinal       INTEGER NOT NULL DEFAULT 0,
             PRIMARY KEY (episode_id, cause_node_id)
         );
         CREATE INDEX IF NOT EXISTS idx_episode_causes_cause ON episode_causes(cause_node_id);",
    )?;

    // Populate edge table from existing JSON cause_ids
    conn.execute_batch(
        "INSERT OR IGNORE INTO episode_causes (episode_id, cause_node_id, ordinal)
         SELECT e.episode_id, je.value, CAST(je.key AS INTEGER)
         FROM episodes e, json_each(e.cause_ids) je;",
    )?;

    conn.execute_batch("PRAGMA foreign_keys = ON;")?;

    Ok(())
}

/// How thorough the integrity check should be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyMode {
    /// Quick: counts and basic metadata only.
    Quick,
    /// Full: includes FTS, JSON/enum decoding, blobs, and SQLite integrity_check.
    Full,
}

/// Result of an integrity verification.
#[derive(Debug, Clone)]
pub struct IntegrityReport {
    pub ok: bool,
    pub schema_version: u32,
    pub fact_count: usize,
    pub chunk_count: usize,
    pub message_count: usize,
    pub facts_missing_embeddings: usize,
    pub chunks_missing_embeddings: usize,
    pub issues: Vec<String>,
}

/// Action to take when integrity issues are found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconcileAction {
    ReportOnly,
    RebuildFts,
    ReEmbed,
}

/// Desired HNSW sidecar mutation queued in SQLite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IndexOpKind {
    Upsert,
    Delete,
}

impl IndexOpKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Upsert => "upsert",
            Self::Delete => "delete",
        }
    }

    fn parse(raw: &str, item_key: &str) -> Result<Self, MemoryError> {
        match raw {
            "upsert" => Ok(Self::Upsert),
            "delete" => Ok(Self::Delete),
            other => Err(MemoryError::CorruptData {
                table: "pending_index_ops",
                row_id: item_key.to_string(),
                detail: format!("invalid op_kind '{other}'"),
            }),
        }
    }
}

/// Durable sidecar repair record.
#[derive(Debug, Clone)]
pub(crate) struct PendingIndexOp {
    pub item_key: String,
    pub entity_type: String,
    pub op_kind: IndexOpKind,
    pub attempt_count: u32,
    pub last_error: Option<String>,
}

/// Run a closure inside an unchecked transaction, committing on success.
pub fn with_transaction<F, T>(conn: &Connection, f: F) -> Result<T, MemoryError>
where
    F: FnOnce(&rusqlite::Transaction<'_>) -> Result<T, MemoryError>,
{
    let tx = conn.unchecked_transaction()?;
    let result = f(&tx)?;
    tx.commit()?;
    Ok(result)
}

/// Open or create a SQLite database, configure pragmas, and run migrations.
pub fn open_database(
    path: &Path,
    pool: &PoolConfig,
    limits: &MemoryLimits,
) -> Result<Connection, MemoryError> {
    open_database_internal(path, pool, limits.max_db_size_bytes, true)
}

/// Open a SQLite connection with pragmas applied but without running migrations.
pub fn open_database_connection(
    path: &Path,
    pool: &PoolConfig,
    limits: &MemoryLimits,
) -> Result<Connection, MemoryError> {
    open_database_internal(path, pool, limits.max_db_size_bytes, false)
}

pub(crate) fn open_database_internal(
    path: &Path,
    pool: &PoolConfig,
    max_db_size_bytes: u64,
    run_schema_migrations: bool,
) -> Result<Connection, MemoryError> {
    create_parent_dirs(path)?;
    let conn = Connection::open(path)?;
    configure_connection(&conn, path, pool, max_db_size_bytes, false)?;
    if run_schema_migrations {
        run_migrations(&conn)?;
    }
    Ok(conn)
}

pub(crate) fn open_pool_member_connection(
    path: &Path,
    pool: &PoolConfig,
    limits: &MemoryLimits,
    query_only: bool,
) -> Result<Connection, MemoryError> {
    create_parent_dirs(path)?;
    let flags = OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE;
    let conn = Connection::open_with_flags(path, flags)?;
    configure_connection(&conn, path, pool, limits.max_db_size_bytes, query_only)?;
    Ok(conn)
}

fn create_parent_dirs(path: &Path) -> Result<(), MemoryError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| {
                MemoryError::StorageError(format!(
                    "failed to create database directory {}: {}",
                    parent.display(),
                    e
                ))
            })?;
        }
    }
    Ok(())
}

fn configure_connection(
    conn: &Connection,
    path: &Path,
    pool: &PoolConfig,
    max_db_size_bytes: u64,
    query_only: bool,
) -> Result<(), MemoryError> {
    let journal_mode = if pool.enable_wal { "WAL" } else { "DELETE" };
    conn.execute_batch(&format!(
        "PRAGMA journal_mode = {};
         PRAGMA foreign_keys = ON;
         PRAGMA busy_timeout = {};
         PRAGMA synchronous = NORMAL;
         PRAGMA wal_autocheckpoint = {};",
        journal_mode, pool.busy_timeout_ms, pool.wal_autocheckpoint,
    ))?;

    if query_only {
        conn.execute_batch("PRAGMA query_only = ON;")?;
    }

    let actual_journal_mode: String =
        conn.query_row("PRAGMA journal_mode", [], |row| row.get(0))?;
    let expected_journal_mode = if pool.enable_wal { "wal" } else { "delete" };
    if actual_journal_mode.to_lowercase() != expected_journal_mode {
        return Err(MemoryError::StorageError(format!(
            "SQLite journal mode mismatch for {}: requested {}, got {}",
            path.display(),
            expected_journal_mode,
            actual_journal_mode
        )));
    }

    if max_db_size_bytes > 0 {
        let page_size: u64 = conn.query_row("PRAGMA page_size", [], |row| row.get(0))?;
        let max_page_count = max_db_size_bytes.div_ceil(page_size);
        let actual_max_page_count: u64 = conn.query_row(
            &format!("PRAGMA max_page_count = {}", max_page_count),
            [],
            |row| row.get(0),
        )?;
        let page_count: u64 = conn.query_row("PRAGMA page_count", [], |row| row.get(0))?;

        if page_count > actual_max_page_count {
            return Err(MemoryError::DatabaseSizeLimitExceeded {
                current: page_count.saturating_mul(page_size),
                limit: max_db_size_bytes,
            });
        }
    }

    Ok(())
}

/// Run all pending migrations.
pub fn run_migrations(conn: &Connection) -> Result<(), MemoryError> {
    let user_version: u32 = conn
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .unwrap_or(0);

    if user_version > MAX_SCHEMA_VERSION {
        return Err(MemoryError::SchemaAhead {
            found: user_version,
            supported: MAX_SCHEMA_VERSION,
        });
    }

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS _schema_version (
            version     INTEGER PRIMARY KEY,
            applied_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );",
    )?;

    for &(version, sql) in MIGRATIONS {
        let current_version: u32 = conn
            .query_row(
                "SELECT COALESCE(MAX(version), 0) FROM _schema_version",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        if current_version >= version {
            continue;
        }

        // V9 requires a procedural migration (table rebuild)
        if version == 9 {
            run_migration_v9(conn).map_err(|e| MemoryError::MigrationFailed {
                version: 9,
                reason: e.to_string(),
            })?;
            conn.execute(
                "INSERT INTO _schema_version (version) VALUES (?1)",
                params![version],
            )
            .map_err(|e| MemoryError::MigrationFailed {
                version,
                reason: e.to_string(),
            })?;
            tracing::info!("Applied migration V{}", version);
            continue;
        }

        with_transaction(conn, |tx| {
            tx.execute_batch(sql)
                .map_err(|e| MemoryError::MigrationFailed {
                    version,
                    reason: e.to_string(),
                })?;
            tx.execute(
                "INSERT INTO _schema_version (version) VALUES (?1)",
                params![version],
            )
            .map_err(|e| MemoryError::MigrationFailed {
                version,
                reason: e.to_string(),
            })?;
            Ok(())
        })?;

        tracing::info!("Applied migration V{}", version);
    }

    let final_version: u32 = conn
        .query_row(
            "SELECT COALESCE(MAX(version), 0) FROM _schema_version",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);
    conn.execute_batch(&format!("PRAGMA user_version = {};", final_version))?;

    Ok(())
}

/// Check and update the embedding metadata singleton row.
pub fn check_embedding_metadata(
    conn: &Connection,
    config: &EmbeddingConfig,
) -> Result<(), MemoryError> {
    let existing: Option<(String, usize)> = conn
        .query_row(
            "SELECT model_name, dimensions FROM embedding_metadata WHERE id = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .ok();

    match existing {
        Some((model, dims)) => {
            if model != config.model || dims != config.dimensions {
                tracing::warn!(
                    stored_model = %model,
                    stored_dims = dims,
                    configured_model = %config.model,
                    configured_dims = config.dimensions,
                    "Embedding model changed. Existing embeddings are stale."
                );
                conn.execute(
                    "UPDATE embedding_metadata
                     SET model_name = ?1,
                         dimensions = ?2,
                         embeddings_dirty = 1,
                         updated_at = datetime('now')
                     WHERE id = 1",
                    params![config.model, config.dimensions],
                )?;
            }
        }
        None => {
            conn.execute(
                "INSERT INTO embedding_metadata (id, model_name, dimensions) VALUES (1, ?1, ?2)",
                params![config.model, config.dimensions],
            )?;
        }
    }

    Ok(())
}

/// Encode an f32 slice as bytes for SQLite BLOB storage.
pub fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for value in embedding {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Decode a SQLite embedding BLOB back to f32 values.
#[allow(clippy::manual_is_multiple_of)]
pub fn bytes_to_embedding(bytes: &[u8]) -> Result<Vec<f32>, MemoryError> {
    if bytes.len() % 4 != 0 {
        return Err(MemoryError::InvalidEmbedding {
            expected_bytes: bytes.len() - (bytes.len() % 4),
            actual_bytes: bytes.len(),
        });
    }

    match bytemuck::try_cast_slice::<u8, f32>(bytes) {
        Ok(slice) => Ok(slice.to_vec()),
        Err(_) => {
            let mut embedding = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                embedding.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(embedding)
        }
    }
}

pub fn is_embeddings_dirty(conn: &Connection) -> Result<bool, MemoryError> {
    let dirty: i32 = conn
        .query_row(
            "SELECT COALESCE(embeddings_dirty, 0) FROM embedding_metadata WHERE id = 1",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);
    Ok(dirty != 0)
}

pub fn clear_embeddings_dirty(conn: &Connection) -> Result<(), MemoryError> {
    conn.execute(
        "UPDATE embedding_metadata SET embeddings_dirty = 0 WHERE id = 1",
        [],
    )?;
    Ok(())
}

#[cfg(feature = "hnsw")]
pub(crate) fn queue_pending_index_op(
    tx: &rusqlite::Transaction<'_>,
    item_key: &str,
    entity_type: &str,
    op_kind: IndexOpKind,
) -> Result<(), MemoryError> {
    tx.execute(
        "INSERT INTO pending_index_ops (item_key, entity_type, op_kind, attempt_count, last_error, updated_at)
         VALUES (?1, ?2, ?3, 0, NULL, datetime('now'))
         ON CONFLICT(item_key) DO UPDATE SET
             entity_type = excluded.entity_type,
             op_kind = excluded.op_kind,
             attempt_count = 0,
             last_error = NULL,
             updated_at = datetime('now')",
        params![item_key, entity_type, op_kind.as_str()],
    )?;
    mark_sidecar_dirty(tx)?;
    Ok(())
}

#[cfg(feature = "hnsw")]
pub(crate) use IndexOpKind as PendingIndexOpKind;

#[cfg(feature = "hnsw")]
pub(crate) fn enqueue_pending_index_op(
    tx: &rusqlite::Transaction<'_>,
    item_key: &str,
    entity_type: &str,
    op_kind: PendingIndexOpKind,
) -> Result<(), MemoryError> {
    queue_pending_index_op(tx, item_key, entity_type, op_kind)
}

pub(crate) fn list_pending_index_ops(
    conn: &Connection,
) -> Result<Vec<PendingIndexOp>, MemoryError> {
    let table_exists: bool = conn
        .query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='pending_index_ops'",
            [],
            |row| row.get(0),
        )
        .unwrap_or(false);
    if !table_exists {
        return Ok(Vec::new());
    }

    let mut stmt = conn.prepare(
        "SELECT item_key, entity_type, op_kind, attempt_count, last_error
         FROM pending_index_ops
         ORDER BY updated_at ASC, item_key ASC",
    )?;
    let rows = stmt
        .query_map([], |row| {
            let item_key: String = row.get(0)?;
            let op_kind: String = row.get(2)?;
            Ok(PendingIndexOp {
                item_key: item_key.clone(),
                entity_type: row.get(1)?,
                op_kind: IndexOpKind::parse(&op_kind, &item_key).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        2,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    )
                })?,
                attempt_count: row.get::<_, i64>(3)? as u32,
                last_error: row.get(4)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

#[cfg(feature = "hnsw")]
pub(crate) fn pending_index_op_count(conn: &Connection) -> Result<usize, MemoryError> {
    let table_exists: bool = conn
        .query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='pending_index_ops'",
            [],
            |row| row.get(0),
        )
        .unwrap_or(false);
    if !table_exists {
        return Ok(0);
    }

    let count: i64 = conn.query_row("SELECT COUNT(*) FROM pending_index_ops", [], |row| {
        row.get(0)
    })?;
    Ok(count as usize)
}

#[cfg(feature = "hnsw")]
pub(crate) fn mark_pending_index_ops_failed(
    conn: &Connection,
    item_keys: &[String],
    error: &str,
) -> Result<(), MemoryError> {
    with_transaction(conn, |tx| {
        for item_key in item_keys {
            tx.execute(
                "UPDATE pending_index_ops
                 SET attempt_count = attempt_count + 1,
                     last_error = ?1,
                     updated_at = datetime('now')
                 WHERE item_key = ?2",
                params![error, item_key],
            )?;
        }
        Ok(())
    })
}

#[cfg(feature = "hnsw")]
pub(crate) fn clear_pending_index_ops(
    conn: &Connection,
    item_keys: &[String],
) -> Result<(), MemoryError> {
    with_transaction(conn, |tx| {
        for item_key in item_keys {
            tx.execute(
                "DELETE FROM pending_index_ops WHERE item_key = ?1",
                params![item_key],
            )?;
        }
        Ok(())
    })
}

#[cfg(feature = "hnsw")]
pub(crate) fn clear_all_pending_index_ops(conn: &Connection) -> Result<(), MemoryError> {
    conn.execute("DELETE FROM pending_index_ops", [])?;
    Ok(())
}

#[cfg(feature = "hnsw")]
pub(crate) fn load_embedding_for_index_key(
    conn: &Connection,
    item_key: &str,
) -> Result<Option<Vec<f32>>, MemoryError> {
    let Some((domain, raw_id)) = item_key.split_once(':') else {
        return Err(MemoryError::InvalidKey(item_key.to_string()));
    };

    let blob_result: Result<Option<Vec<u8>>, rusqlite::Error> = match domain {
        "fact" => conn.query_row(
            "SELECT embedding FROM facts WHERE id = ?1",
            params![raw_id],
            |row| row.get(0),
        ),
        "chunk" => conn.query_row(
            "SELECT embedding FROM chunks WHERE id = ?1",
            params![raw_id],
            |row| row.get(0),
        ),
        "msg" => {
            let message_id = raw_id
                .parse::<i64>()
                .map_err(|_| MemoryError::InvalidKey(item_key.to_string()))?;
            conn.query_row(
                "SELECT embedding FROM messages WHERE id = ?1",
                params![message_id],
                |row| row.get(0),
            )
        }
        "episode" => conn.query_row(
            "SELECT embedding FROM episodes WHERE episode_id = ?1",
            params![raw_id],
            |row| row.get(0),
        ),
        _ => return Err(MemoryError::InvalidKey(item_key.to_string())),
    };

    let blob = match blob_result {
        Ok(blob) => blob,
        Err(rusqlite::Error::QueryReturnedNoRows) => None,
        Err(err) => return Err(err.into()),
    };

    blob.map(|bytes| bytes_to_embedding(&bytes)).transpose()
}

#[cfg(feature = "hnsw")]
fn mark_sidecar_dirty(tx: &rusqlite::Transaction<'_>) -> Result<(), MemoryError> {
    tx.execute(
        "INSERT INTO hnsw_metadata (key, value) VALUES ('sidecar_dirty', '1')
         ON CONFLICT(key) DO UPDATE SET value = '1'",
        [],
    )?;
    Ok(())
}

#[cfg(feature = "hnsw")]
pub(crate) fn is_sidecar_dirty(conn: &Connection) -> Result<bool, MemoryError> {
    let dirty: Option<String> = conn
        .query_row(
            "SELECT value FROM hnsw_metadata WHERE key = 'sidecar_dirty'",
            [],
            |row| row.get(0),
        )
        .ok();
    Ok(matches!(dirty.as_deref(), Some("1")))
}

#[cfg(feature = "hnsw")]
pub(crate) fn set_sidecar_dirty(conn: &Connection, dirty: bool) -> Result<(), MemoryError> {
    conn.execute(
        "INSERT INTO hnsw_metadata (key, value) VALUES ('sidecar_dirty', ?1)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![if dirty { "1" } else { "0" }],
    )?;
    Ok(())
}

pub(crate) fn parse_optional_json(
    table: &'static str,
    row_id: &str,
    field: &'static str,
    raw: Option<&str>,
) -> Result<Option<serde_json::Value>, MemoryError> {
    match raw {
        Some(raw) => serde_json::from_str(raw)
            .map(Some)
            .map_err(|e| MemoryError::CorruptData {
                table,
                row_id: row_id.to_string(),
                detail: format!("invalid {field}: {e}"),
            }),
        None => Ok(None),
    }
}

pub(crate) fn parse_string_list_json(
    table: &'static str,
    row_id: &str,
    field: &'static str,
    raw: &str,
) -> Result<Vec<String>, MemoryError> {
    serde_json::from_str(raw).map_err(|e| MemoryError::CorruptData {
        table,
        row_id: row_id.to_string(),
        detail: format!("invalid {field}: {e}"),
    })
}

pub(crate) fn parse_role(
    table: &'static str,
    row_id: &str,
    raw: &str,
) -> Result<Role, MemoryError> {
    Role::from_str_value(raw).ok_or_else(|| MemoryError::CorruptData {
        table,
        row_id: row_id.to_string(),
        detail: format!("invalid role '{raw}'"),
    })
}

pub(crate) fn parse_episode_outcome(
    row_id: &str,
    raw: &str,
) -> Result<EpisodeOutcome, MemoryError> {
    EpisodeOutcome::from_str_value(raw).ok_or_else(|| MemoryError::CorruptData {
        table: "episodes",
        row_id: row_id.to_string(),
        detail: format!("invalid outcome '{raw}'"),
    })
}

pub(crate) fn parse_verification_status(
    row_id: &str,
    raw: &str,
) -> Result<VerificationStatus, MemoryError> {
    serde_json::from_str(raw).map_err(|e| MemoryError::CorruptData {
        table: "episodes",
        row_id: row_id.to_string(),
        detail: format!("invalid verification_status: {e}"),
    })
}

/// Run integrity verification on the database.
pub fn verify_integrity_sync(
    conn: &Connection,
    mode: VerifyMode,
) -> Result<IntegrityReport, MemoryError> {
    let mut issues = Vec::new();

    let schema_version: u32 = conn
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .unwrap_or_else(|e| {
            issues.push(format!("failed to read schema version: {e}"));
            0
        });
    if schema_version > MAX_SCHEMA_VERSION {
        issues.push(format!(
            "schema version {} is ahead of supported {}",
            schema_version, MAX_SCHEMA_VERSION
        ));
    }

    let fact_count: usize = conn
        .query_row("SELECT COUNT(*) FROM facts", [], |row| row.get(0))
        .unwrap_or_else(|e| {
            issues.push(format!("failed to count facts: {e}"));
            0
        });
    let chunk_count: usize = conn
        .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
        .unwrap_or_else(|e| {
            issues.push(format!("failed to count chunks: {e}"));
            0
        });
    let message_count: usize = conn
        .query_row("SELECT COUNT(*) FROM messages", [], |row| row.get(0))
        .unwrap_or_else(|e| {
            issues.push(format!("failed to count messages: {e}"));
            0
        });
    let episode_count: usize = conn
        .query_row("SELECT COUNT(*) FROM episodes", [], |row| row.get(0))
        .unwrap_or_else(|e| {
            issues.push(format!("failed to count episodes: {e}"));
            0
        });

    let facts_missing_embeddings: usize = conn
        .query_row(
            "SELECT COUNT(*) FROM facts WHERE embedding IS NULL",
            [],
            |row| row.get(0),
        )
        .unwrap_or_else(|e| {
            issues.push(format!("failed to count facts missing embeddings: {e}"));
            0
        });
    let chunks_missing_embeddings: usize = conn
        .query_row(
            "SELECT COUNT(*) FROM chunks WHERE embedding IS NULL",
            [],
            |row| row.get(0),
        )
        .unwrap_or_else(|e| {
            issues.push(format!("failed to count chunks missing embeddings: {e}"));
            0
        });
    let episodes_missing_embeddings: usize = conn
        .query_row(
            "SELECT COUNT(*) FROM episodes WHERE embedding IS NULL",
            [],
            |row| row.get(0),
        )
        .unwrap_or_else(|e| {
            issues.push(format!("failed to count episodes missing embeddings: {e}"));
            0
        });

    if facts_missing_embeddings > 0 {
        issues.push(format!(
            "{} facts missing embeddings",
            facts_missing_embeddings
        ));
    }
    if chunks_missing_embeddings > 0 {
        issues.push(format!(
            "{} chunks missing embeddings",
            chunks_missing_embeddings
        ));
    }
    if episodes_missing_embeddings > 0 {
        issues.push(format!(
            "{} episodes missing embeddings",
            episodes_missing_embeddings
        ));
    }

    let pending_ops = list_pending_index_ops(conn).unwrap_or_default();
    if !pending_ops.is_empty() {
        issues.push(format!(
            "{} pending HNSW sidecar ops queued in SQLite",
            pending_ops.len()
        ));
        for op in pending_ops.iter().take(5) {
            let op_kind = op.op_kind.as_str();
            let detail = match &op.last_error {
                Some(last_error) => format!(
                    "{} {} {} (attempts: {}, last_error: {})",
                    op.entity_type,
                    op.op_kind.as_str(),
                    op.item_key,
                    op.attempt_count,
                    last_error
                ),
                None => format!(
                    "{} {} {} (attempts: {})",
                    op.entity_type, op_kind, op.item_key, op.attempt_count
                ),
            };
            issues.push(format!("pending sidecar op: {detail}"));
        }
    }

    if mode == VerifyMode::Full {
        let dims: usize = conn
            .query_row(
                "SELECT dimensions FROM embedding_metadata WHERE id = 1",
                [],
                |row| row.get(0),
            )
            .unwrap_or_else(|e| {
                issues.push(format!("failed to read embedding dimensions: {e}"));
                0
            });

        verify_fts_drift(conn, "facts", "facts_rowid_map", fact_count, &mut issues);
        verify_fts_drift(conn, "chunks", "chunks_rowid_map", chunk_count, &mut issues);
        verify_fts_drift(
            conn,
            "messages",
            "messages_rowid_map",
            message_count,
            &mut issues,
        );
        verify_fts_drift(
            conn,
            "episodes",
            "episodes_rowid_map",
            episode_count,
            &mut issues,
        );

        verify_blob_table(conn, "facts", "id", "embedding", dims, &mut issues)?;
        verify_blob_table(conn, "chunks", "id", "embedding", dims, &mut issues)?;
        verify_blob_table(conn, "messages", "id", "embedding", dims, &mut issues)?;
        verify_blob_table(
            conn,
            "episodes",
            "episode_id",
            "embedding",
            dims,
            &mut issues,
        )?;

        verify_quantized_table(conn, "facts", "id", dims, &mut issues)?;
        verify_quantized_table(conn, "chunks", "id", dims, &mut issues)?;
        verify_quantized_table(conn, "messages", "id", dims, &mut issues)?;
        verify_quantized_table(conn, "episodes", "episode_id", dims, &mut issues)?;

        verify_session_rows(conn, &mut issues)?;
        verify_message_rows(conn, &mut issues)?;
        verify_fact_rows(conn, &mut issues)?;
        verify_document_rows(conn, &mut issues)?;
        verify_episode_rows(conn, &mut issues)?;

        let integrity_check: String = conn
            .query_row("PRAGMA integrity_check", [], |row| row.get(0))
            .unwrap_or_else(|_| "error".to_string());
        if integrity_check != "ok" {
            issues.push(format!("SQLite integrity_check: {}", integrity_check));
        }
    }

    Ok(IntegrityReport {
        ok: issues.is_empty(),
        schema_version,
        fact_count,
        chunk_count,
        message_count,
        facts_missing_embeddings,
        chunks_missing_embeddings,
        issues,
    })
}

/// Reconcile FTS indexes by rebuilding them from source data.
pub fn reconcile_fts(conn: &Connection) -> Result<(), MemoryError> {
    with_transaction(conn, |tx| {
        tx.execute_batch("DROP TABLE IF EXISTS facts_fts")?;
        tx.execute_batch("DELETE FROM facts_rowid_map")?;
        tx.execute_batch(
            "CREATE VIRTUAL TABLE facts_fts USING fts5(
                content,
                content='',
                content_rowid='rowid',
                tokenize='porter unicode61'
            )",
        )?;
        tx.execute_batch("INSERT INTO facts_rowid_map (fact_id) SELECT id FROM facts")?;
        tx.execute_batch(
            "INSERT INTO facts_fts (rowid, content)
             SELECT rm.rowid, f.content
             FROM facts_rowid_map rm
             JOIN facts f ON f.id = rm.fact_id",
        )?;

        tx.execute_batch("DROP TABLE IF EXISTS chunks_fts")?;
        tx.execute_batch("DELETE FROM chunks_rowid_map")?;
        tx.execute_batch(
            "CREATE VIRTUAL TABLE chunks_fts USING fts5(
                content,
                content='',
                content_rowid='rowid',
                tokenize='porter unicode61'
            )",
        )?;
        tx.execute_batch("INSERT INTO chunks_rowid_map (chunk_id) SELECT id FROM chunks")?;
        tx.execute_batch(
            "INSERT INTO chunks_fts (rowid, content)
             SELECT rm.rowid, c.content
             FROM chunks_rowid_map rm
             JOIN chunks c ON c.id = rm.chunk_id",
        )?;

        tx.execute_batch("DROP TABLE IF EXISTS messages_fts")?;
        tx.execute_batch("DELETE FROM messages_rowid_map")?;
        tx.execute_batch(
            "CREATE VIRTUAL TABLE messages_fts USING fts5(
                content,
                content='',
                content_rowid='rowid',
                tokenize='porter unicode61'
            )",
        )?;
        tx.execute_batch("INSERT INTO messages_rowid_map (message_id) SELECT id FROM messages")?;
        tx.execute_batch(
            "INSERT INTO messages_fts (rowid, content)
             SELECT rm.rowid, m.content
             FROM messages_rowid_map rm
             JOIN messages m ON m.id = rm.message_id",
        )?;

        tx.execute_batch("DROP TABLE IF EXISTS episodes_fts")?;
        tx.execute_batch("DELETE FROM episodes_rowid_map")?;
        tx.execute_batch(
            "CREATE VIRTUAL TABLE episodes_fts USING fts5(
                content,
                content='',
                content_rowid='rowid',
                tokenize='porter unicode61'
            )",
        )?;
        tx.execute_batch(
            "INSERT INTO episodes_rowid_map (episode_id, document_id) SELECT episode_id, document_id FROM episodes",
        )?;
        tx.execute_batch(
            "INSERT INTO episodes_fts (rowid, content)
             SELECT rm.rowid, e.search_text
             FROM episodes_rowid_map rm
             JOIN episodes e ON e.episode_id = rm.episode_id",
        )?;

        Ok(())
    })?;

    tracing::info!("FTS indexes reconciled");
    Ok(())
}

fn verify_fts_drift(
    conn: &Connection,
    label: &str,
    map_table: &str,
    source_count: usize,
    issues: &mut Vec<String>,
) {
    let table_exists: bool = conn
        .query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name = ?1",
            params![map_table],
            |row| row.get(0),
        )
        .unwrap_or(false);
    if !table_exists {
        if source_count > 0 {
            issues.push(format!("{} rows exist but {} is missing", label, map_table));
        }
        return;
    }

    let sql = format!("SELECT COUNT(*) FROM {}", map_table);
    let indexed_count: usize = conn.query_row(&sql, [], |row| row.get(0)).unwrap_or(0);
    if indexed_count != source_count {
        issues.push(format!(
            "FTS {} index drift: {} rows in map vs {} source rows",
            label, indexed_count, source_count
        ));
    }
}

fn verify_blob_table(
    conn: &Connection,
    table: &'static str,
    id_column: &'static str,
    blob_column: &'static str,
    expected_dims: usize,
    issues: &mut Vec<String>,
) -> Result<(), MemoryError> {
    if expected_dims == 0 {
        return Ok(());
    }

    let sql = format!(
        "SELECT CAST({id_column} AS TEXT), {blob_column} FROM {table} WHERE {blob_column} IS NOT NULL"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
    })?;

    for row in rows {
        let (row_id, blob) = row?;
        match bytes_to_embedding(&blob) {
            Ok(embedding) if embedding.len() != expected_dims => issues.push(format!(
                "{}({}) has embedding dimension {} but expected {}",
                table,
                row_id,
                embedding.len(),
                expected_dims
            )),
            Ok(_) => {}
            Err(err) => issues.push(format!(
                "{}({}) invalid embedding blob: {}",
                table, row_id, err
            )),
        }
    }

    Ok(())
}

fn verify_quantized_table(
    conn: &Connection,
    table: &'static str,
    id_column: &'static str,
    expected_dims: usize,
    issues: &mut Vec<String>,
) -> Result<(), MemoryError> {
    if expected_dims == 0 {
        return Ok(());
    }

    let sql = format!(
        "SELECT CAST({id_column} AS TEXT), embedding_q8 FROM {table} WHERE embedding IS NOT NULL"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, Option<Vec<u8>>>(1)?))
    })?;

    for row in rows {
        let (row_id, blob) = row?;
        match blob {
            Some(blob) => {
                if let Err(err) = unpack_quantized(&blob, expected_dims) {
                    issues.push(format!(
                        "{}({}) invalid quantized embedding: {}",
                        table, row_id, err
                    ));
                }
            }
            None => issues.push(format!("{}({}) missing quantized embedding", table, row_id)),
        }
    }

    Ok(())
}

fn verify_session_rows(conn: &Connection, issues: &mut Vec<String>) -> Result<(), MemoryError> {
    let mut stmt = conn.prepare("SELECT id, metadata FROM sessions WHERE metadata IS NOT NULL")?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    for row in rows {
        let (id, metadata) = row?;
        if let Err(err) = parse_optional_json("sessions", &id, "metadata", Some(&metadata)) {
            issues.push(err.to_string());
        }
    }
    Ok(())
}

fn verify_message_rows(conn: &Connection, issues: &mut Vec<String>) -> Result<(), MemoryError> {
    let mut stmt = conn.prepare("SELECT id, role, metadata FROM messages")?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, Option<String>>(2)?,
        ))
    })?;
    for row in rows {
        let (id, role, metadata) = row?;
        let row_id = id.to_string();
        if let Err(err) = parse_role("messages", &row_id, &role) {
            issues.push(err.to_string());
        }
        if let Err(err) = parse_optional_json("messages", &row_id, "metadata", metadata.as_deref())
        {
            issues.push(err.to_string());
        }
    }
    Ok(())
}

fn verify_fact_rows(conn: &Connection, issues: &mut Vec<String>) -> Result<(), MemoryError> {
    let mut stmt = conn.prepare("SELECT id, metadata FROM facts WHERE metadata IS NOT NULL")?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    for row in rows {
        let (id, metadata) = row?;
        if let Err(err) = parse_optional_json("facts", &id, "metadata", Some(&metadata)) {
            issues.push(err.to_string());
        }
    }
    Ok(())
}

fn verify_document_rows(conn: &Connection, issues: &mut Vec<String>) -> Result<(), MemoryError> {
    let mut stmt = conn.prepare("SELECT id, metadata FROM documents WHERE metadata IS NOT NULL")?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    for row in rows {
        let (id, metadata) = row?;
        if let Err(err) = parse_optional_json("documents", &id, "metadata", Some(&metadata)) {
            issues.push(err.to_string());
        }
    }
    Ok(())
}

fn verify_episode_rows(conn: &Connection, issues: &mut Vec<String>) -> Result<(), MemoryError> {
    let mut stmt = conn.prepare(
        "SELECT episode_id, cause_ids, outcome, verification_status
         FROM episodes",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
        ))
    })?;
    for row in rows {
        let (episode_id, cause_ids, outcome, verification_status) = row?;
        if let Err(err) = parse_string_list_json("episodes", &episode_id, "cause_ids", &cause_ids) {
            issues.push(err.to_string());
        }
        if let Err(err) = parse_episode_outcome(&episode_id, &outcome) {
            issues.push(err.to_string());
        }
        if let Err(err) = parse_verification_status(&episode_id, &verification_status) {
            issues.push(err.to_string());
        }
    }
    Ok(())
}
