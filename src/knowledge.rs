//! Fact CRUD with FTS5 synchronization.
//!
//! Every fact operation that touches `facts_fts` is transactional.
//! See SPEC.md §8.3 for the insert/update/delete procedures.

use crate::db::{bytes_to_embedding, with_transaction};
use crate::error::MemoryError;
use crate::types::Fact;
use rusqlite::{params, Connection};

/// Insert a fact and its FTS entry in a transaction.
///
/// Called after embedding is already computed.
pub fn insert_fact_with_fts(
    conn: &Connection,
    fact_id: &str,
    namespace: &str,
    content: &str,
    embedding_bytes: &[u8],
    source: Option<&str>,
    metadata: Option<&serde_json::Value>,
) -> Result<(), MemoryError> {
    insert_fact_with_fts_q8(conn, fact_id, namespace, content, embedding_bytes, None, source, metadata)
}

/// Insert a fact with both f32 and quantized embeddings.
#[allow(clippy::too_many_arguments)]
pub fn insert_fact_with_fts_q8(
    conn: &Connection,
    fact_id: &str,
    namespace: &str,
    content: &str,
    embedding_bytes: &[u8],
    q8_bytes: Option<&[u8]>,
    source: Option<&str>,
    metadata: Option<&serde_json::Value>,
) -> Result<(), MemoryError> {
    let metadata_str = metadata.map(|m| m.to_string());
    with_transaction(conn, |tx| {
        // 1. Insert into facts table
        tx.execute(
            "INSERT INTO facts (id, namespace, content, source, embedding, embedding_q8, metadata) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![fact_id, namespace, content, source, embedding_bytes, q8_bytes, metadata_str],
        )?;

        // 2. Insert into rowid bridge
        tx.execute(
            "INSERT INTO facts_rowid_map (fact_id) VALUES (?1)",
            params![fact_id],
        )?;
        let fts_rowid = tx.last_insert_rowid();

        // 3. Insert into FTS
        tx.execute(
            "INSERT INTO facts_fts(rowid, content) VALUES (?1, ?2)",
            params![fts_rowid, content],
        )?;

        Ok(())
    })
}

/// Delete a fact and its FTS entry in a transaction.
///
/// Must read content BEFORE deleting from the main table (contentless FTS
/// requires the original content for delete).
pub fn delete_fact_with_fts(conn: &Connection, fact_id: &str) -> Result<(), MemoryError> {
    with_transaction(conn, |tx| {
        // 1. Get FTS rowid from bridge
        let fts_rowid: i64 = tx
            .query_row(
                "SELECT rowid FROM facts_rowid_map WHERE fact_id = ?1",
                params![fact_id],
                |row| row.get(0),
            )
            .map_err(|_| MemoryError::FactNotFound(fact_id.to_string()))?;

        // 2. Get content (needed for FTS delete)
        let content: String = tx
            .query_row(
                "SELECT content FROM facts WHERE id = ?1",
                params![fact_id],
                |row| row.get(0),
            )
            .map_err(|_| MemoryError::FactNotFound(fact_id.to_string()))?;

        // 3. Delete from FTS (contentless FTS delete syntax)
        tx.execute(
            "INSERT INTO facts_fts(facts_fts, rowid, content) VALUES('delete', ?1, ?2)",
            params![fts_rowid, content],
        )?;

        // 4. Delete from bridge
        tx.execute(
            "DELETE FROM facts_rowid_map WHERE fact_id = ?1",
            params![fact_id],
        )?;

        // 5. Delete from facts
        tx.execute("DELETE FROM facts WHERE id = ?1", params![fact_id])?;

        Ok(())
    })
}

/// Update a fact's content and embedding, with FTS sync.
///
/// Called after new embedding is already computed.
pub fn update_fact_with_fts(
    conn: &Connection,
    fact_id: &str,
    new_content: &str,
    new_embedding_bytes: &[u8],
) -> Result<(), MemoryError> {
    with_transaction(conn, |tx| {
        // 1. Get old FTS rowid and content
        let (fts_rowid, old_content): (i64, String) = tx
            .query_row(
                "SELECT fm.rowid, f.content FROM facts f
                 JOIN facts_rowid_map fm ON fm.fact_id = f.id
                 WHERE f.id = ?1",
                params![fact_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(|_| MemoryError::FactNotFound(fact_id.to_string()))?;

        // 2. Delete old FTS entry
        tx.execute(
            "INSERT INTO facts_fts(facts_fts, rowid, content) VALUES('delete', ?1, ?2)",
            params![fts_rowid, old_content],
        )?;

        // 3. Update facts table
        tx.execute(
            "UPDATE facts SET content = ?1, embedding = ?2, updated_at = datetime('now') WHERE id = ?3",
            params![new_content, new_embedding_bytes, fact_id],
        )?;

        // 4. Insert new FTS entry (reuse same rowid)
        tx.execute(
            "INSERT INTO facts_fts(rowid, content) VALUES (?1, ?2)",
            params![fts_rowid, new_content],
        )?;

        Ok(())
    })
}

/// Delete all facts in a namespace atomically. Returns count of deleted facts.
///
/// All deletions happen in a single transaction so the operation is atomic —
/// either all facts are deleted or none are (no partial namespace deletion on crash).
pub fn delete_namespace(conn: &Connection, namespace: &str) -> Result<usize, MemoryError> {
    with_transaction(conn, |tx| {
        // 1. Get all fact IDs, FTS rowids, and content in the namespace
        let mut stmt = tx.prepare(
            "SELECT f.id, fm.rowid, f.content
             FROM facts f
             JOIN facts_rowid_map fm ON fm.fact_id = f.id
             WHERE f.namespace = ?1",
        )?;
        let facts: Vec<(String, i64, String)> = stmt
            .query_map(params![namespace], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        let count = facts.len();

        // 2. Delete FTS entries and bridge rows for each fact
        for (fact_id, fts_rowid, content) in &facts {
            tx.execute(
                "INSERT INTO facts_fts(facts_fts, rowid, content) VALUES('delete', ?1, ?2)",
                params![fts_rowid, content],
            )?;
            tx.execute(
                "DELETE FROM facts_rowid_map WHERE fact_id = ?1",
                params![fact_id],
            )?;
        }

        // 3. Delete all facts in the namespace
        tx.execute(
            "DELETE FROM facts WHERE namespace = ?1",
            params![namespace],
        )?;

        Ok(count)
    })
}

/// Get a fact by ID.
pub fn get_fact(conn: &Connection, fact_id: &str) -> Result<Option<Fact>, MemoryError> {
    let result = conn.query_row(
        "SELECT id, namespace, content, source, created_at, updated_at, metadata
         FROM facts WHERE id = ?1",
        params![fact_id],
        |row| {
            let metadata_str: Option<String> = row.get(6)?;
            Ok(Fact {
                id: row.get(0)?,
                namespace: row.get(1)?,
                content: row.get(2)?,
                source: row.get(3)?,
                created_at: row.get(4)?,
                updated_at: row.get(5)?,
                metadata: metadata_str.and_then(|s| serde_json::from_str(&s).ok()),
            })
        },
    );

    match result {
        Ok(fact) => Ok(Some(fact)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(MemoryError::Database(e)),
    }
}

/// Get a fact's raw embedding bytes.
pub fn get_fact_embedding(
    conn: &Connection,
    fact_id: &str,
) -> Result<Option<Vec<f32>>, MemoryError> {
    let result: Result<Option<Vec<u8>>, _> = conn.query_row(
        "SELECT embedding FROM facts WHERE id = ?1",
        params![fact_id],
        |row| row.get(0),
    );

    match result {
        Ok(Some(bytes)) => Ok(Some(bytes_to_embedding(&bytes)?)),
        Ok(None) => Ok(None),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(MemoryError::Database(e)),
    }
}

/// List facts in a namespace with pagination.
pub fn list_facts(
    conn: &Connection,
    namespace: &str,
    limit: usize,
    offset: usize,
) -> Result<Vec<Fact>, MemoryError> {
    let mut stmt = conn.prepare(
        "SELECT id, namespace, content, source, created_at, updated_at, metadata
         FROM facts
         WHERE namespace = ?1
         ORDER BY updated_at DESC
         LIMIT ?2 OFFSET ?3",
    )?;

    let facts = stmt
        .query_map(params![namespace, limit as i64, offset as i64], |row| {
            let metadata_str: Option<String> = row.get(6)?;
            Ok(Fact {
                id: row.get(0)?,
                namespace: row.get(1)?,
                content: row.get(2)?,
                source: row.get(3)?,
                created_at: row.get(4)?,
                updated_at: row.get(5)?,
                metadata: metadata_str.and_then(|s| serde_json::from_str(&s).ok()),
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    Ok(facts)
}
