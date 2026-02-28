//! Database initialization, migrations, and connection management.

use crate::config::EmbeddingConfig;
use crate::error::MemoryError;
use rusqlite::{params, Connection};
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
-- V2: Message embeddings for conversation search
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
-- V3: Embedding staleness tracking
ALTER TABLE embedding_metadata ADD COLUMN embeddings_dirty INTEGER NOT NULL DEFAULT 0;
"#;

/// V4 migration: HNSW metadata tracking.
const MIGRATION_V4: &str = r#"
-- V4: HNSW index metadata
CREATE TABLE IF NOT EXISTS hnsw_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"#;

/// V5 migration: quantized embeddings + HNSW keymap persistence.
const MIGRATION_V5: &str = r#"
-- V5: Quantized embeddings + HNSW keymap persistence
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

/// Run a closure inside an `unchecked_transaction`, committing on success.
///
/// SAFETY: We hold &Connection (not &mut) via Mutex::lock(). unchecked_transaction()
/// is required because transaction() needs &mut self. The Mutex serializes all access,
/// preventing concurrent transaction nesting.
pub fn with_transaction<F, T>(conn: &Connection, f: F) -> Result<T, MemoryError>
where
    F: FnOnce(&rusqlite::Transaction<'_>) -> Result<T, MemoryError>,
{
    let tx = conn.unchecked_transaction()?;
    let result = f(&tx)?;
    tx.commit()?;
    Ok(result)
}

/// Open or create a SQLite database at `path`, configure pragmas, and run migrations.
pub fn open_database(path: &Path) -> Result<Connection, MemoryError> {
    // Create parent directories if needed
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| {
                MemoryError::Other(format!(
                    "Failed to create database directory {}: {}",
                    parent.display(),
                    e
                ))
            })?;
        }
    }

    let conn = Connection::open(path)?;

    // Set pragmas BEFORE any other operation
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA foreign_keys = ON;
         PRAGMA busy_timeout = 5000;
         PRAGMA synchronous = NORMAL;",
    )?;

    run_migrations(&conn)?;

    Ok(conn)
}

/// Ordered list of migrations. Each entry is (version, SQL).
const MIGRATIONS: &[(u32, &str)] = &[
    (1, MIGRATION_V1),
    (2, MIGRATION_V2),
    (3, MIGRATION_V3),
    (4, MIGRATION_V4),
    (5, MIGRATION_V5),
];

/// Run all pending migrations.
pub fn run_migrations(conn: &Connection) -> Result<(), MemoryError> {
    // Create version table if it doesn't exist
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

    Ok(())
}

/// Check and update the embedding_metadata singleton row.
///
/// If the row exists and model/dimensions don't match, warn and update.
/// If no row exists, insert one. If it matches, no-op.
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
                    "Embedding model changed. Existing embeddings are invalid. Call reembed_all() to re-embed."
                );
                conn.execute(
                    "UPDATE embedding_metadata SET model_name = ?1, dimensions = ?2, \
                     embeddings_dirty = 1, updated_at = datetime('now') WHERE id = 1",
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
    for &val in embedding {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

/// Check if embeddings are stale after a model change.
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

/// Clear the dirty flag after re-embedding.
pub fn clear_embeddings_dirty(conn: &Connection) -> Result<(), MemoryError> {
    conn.execute(
        "UPDATE embedding_metadata SET embeddings_dirty = 0 WHERE id = 1",
        [],
    )?;
    Ok(())
}

/// Decode a SQLite BLOB back to f32 values.
///
/// Returns an error if the byte length is not divisible by 4.
/// Uses `bytemuck::try_cast_slice` for zero-copy decoding when alignment permits,
/// falling back to manual decode otherwise.
#[allow(clippy::manual_is_multiple_of)] // MSRV 1.75: is_multiple_of stabilized later
pub fn bytes_to_embedding(bytes: &[u8]) -> Result<Vec<f32>, MemoryError> {
    if bytes.len() % 4 != 0 {
        return Err(MemoryError::InvalidEmbedding {
            expected_bytes: bytes.len() - (bytes.len() % 4),
            actual_bytes: bytes.len(),
        });
    }
    // Heap-allocated Vec<u8> from SQLite is aligned, so cast_slice succeeds.
    // If alignment is off (shouldn't happen), fall back to manual decode.
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
