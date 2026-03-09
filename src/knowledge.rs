//! Fact CRUD with FTS5 synchronization.
//!
//! Every fact operation that touches `facts_fts` is transactional.

use crate::db::{bytes_to_embedding, parse_optional_json, with_transaction};
#[cfg(feature = "hnsw")]
use crate::db::{enqueue_pending_index_op, PendingIndexOpKind};
use crate::error::MemoryError;
use crate::types::Fact;
use rusqlite::{params, Connection};

/// Insert a fact and its FTS entry in a transaction.
#[allow(dead_code)]
pub fn insert_fact_with_fts(
    conn: &Connection,
    fact_id: &str,
    namespace: &str,
    content: &str,
    embedding_bytes: &[u8],
    source: Option<&str>,
    metadata: Option<&serde_json::Value>,
) -> Result<(), MemoryError> {
    insert_fact_with_fts_q8(
        conn,
        fact_id,
        namespace,
        content,
        embedding_bytes,
        None,
        source,
        metadata,
    )
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
        tx.execute(
            "INSERT INTO facts (id, namespace, content, source, embedding, embedding_q8, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                fact_id,
                namespace,
                content,
                source,
                embedding_bytes,
                q8_bytes,
                metadata_str
            ],
        )?;

        tx.execute(
            "INSERT INTO facts_rowid_map (fact_id) VALUES (?1)",
            params![fact_id],
        )?;
        let fts_rowid = tx.last_insert_rowid();

        tx.execute(
            "INSERT INTO facts_fts(rowid, content) VALUES (?1, ?2)",
            params![fts_rowid, content],
        )?;

        #[cfg(feature = "hnsw")]
        enqueue_pending_index_op(
            tx,
            &format!("fact:{}", fact_id),
            "fact",
            PendingIndexOpKind::Upsert,
        )?;

        Ok(())
    })
}

/// Insert a fact within an existing transaction (no nested transaction).
///
/// Used by the import boundary where the outer transaction is already active.
#[allow(clippy::too_many_arguments)]
pub fn insert_fact_in_tx(
    tx: &rusqlite::Transaction<'_>,
    fact_id: &str,
    namespace: &str,
    content: &str,
    embedding_bytes: &[u8],
    q8_bytes: Option<&[u8]>,
    source: Option<&str>,
    metadata: Option<&serde_json::Value>,
) -> Result<(), MemoryError> {
    let metadata_str = metadata.map(|m| m.to_string());
    tx.execute(
        "INSERT INTO facts (id, namespace, content, source, embedding, embedding_q8, metadata)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            fact_id,
            namespace,
            content,
            source,
            embedding_bytes,
            q8_bytes,
            metadata_str
        ],
    )?;

    tx.execute(
        "INSERT INTO facts_rowid_map (fact_id) VALUES (?1)",
        params![fact_id],
    )?;
    let fts_rowid = tx.last_insert_rowid();

    tx.execute(
        "INSERT INTO facts_fts(rowid, content) VALUES (?1, ?2)",
        params![fts_rowid, content],
    )?;

    #[cfg(feature = "hnsw")]
    enqueue_pending_index_op(
        tx,
        &format!("fact:{}", fact_id),
        "fact",
        PendingIndexOpKind::Upsert,
    )?;

    Ok(())
}

/// Delete a fact and its FTS entry in a transaction.
pub fn delete_fact_with_fts(conn: &Connection, fact_id: &str) -> Result<(), MemoryError> {
    with_transaction(conn, |tx| {
        let fts_rowid: i64 = tx
            .query_row(
                "SELECT rowid FROM facts_rowid_map WHERE fact_id = ?1",
                params![fact_id],
                |row| row.get(0),
            )
            .map_err(|_| MemoryError::FactNotFound(fact_id.to_string()))?;

        let content: String = tx
            .query_row(
                "SELECT content FROM facts WHERE id = ?1",
                params![fact_id],
                |row| row.get(0),
            )
            .map_err(|_| MemoryError::FactNotFound(fact_id.to_string()))?;

        tx.execute(
            "INSERT INTO facts_fts(facts_fts, rowid, content) VALUES('delete', ?1, ?2)",
            params![fts_rowid, content],
        )?;
        tx.execute(
            "DELETE FROM facts_rowid_map WHERE fact_id = ?1",
            params![fact_id],
        )?;
        tx.execute("DELETE FROM facts WHERE id = ?1", params![fact_id])?;

        #[cfg(feature = "hnsw")]
        enqueue_pending_index_op(
            tx,
            &format!("fact:{}", fact_id),
            "fact",
            PendingIndexOpKind::Delete,
        )?;

        Ok(())
    })
}

/// Update a fact's content and embeddings, with FTS synchronization.
pub fn update_fact_with_fts(
    conn: &Connection,
    fact_id: &str,
    new_content: &str,
    new_embedding_bytes: &[u8],
    new_q8_bytes: Option<&[u8]>,
) -> Result<(), MemoryError> {
    with_transaction(conn, |tx| {
        let (fts_rowid, old_content): (i64, String) = tx
            .query_row(
                "SELECT fm.rowid, f.content
                 FROM facts f
                 JOIN facts_rowid_map fm ON fm.fact_id = f.id
                 WHERE f.id = ?1",
                params![fact_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(|_| MemoryError::FactNotFound(fact_id.to_string()))?;

        tx.execute(
            "INSERT INTO facts_fts(facts_fts, rowid, content) VALUES('delete', ?1, ?2)",
            params![fts_rowid, old_content],
        )?;

        tx.execute(
            "UPDATE facts
             SET content = ?1,
                 embedding = ?2,
                 embedding_q8 = ?3,
                 updated_at = datetime('now')
             WHERE id = ?4",
            params![new_content, new_embedding_bytes, new_q8_bytes, fact_id],
        )?;

        tx.execute(
            "INSERT INTO facts_fts(rowid, content) VALUES (?1, ?2)",
            params![fts_rowid, new_content],
        )?;

        #[cfg(feature = "hnsw")]
        enqueue_pending_index_op(
            tx,
            &format!("fact:{}", fact_id),
            "fact",
            PendingIndexOpKind::Upsert,
        )?;

        Ok(())
    })
}

/// Delete all facts in a namespace atomically. Returns the deleted count.
pub fn delete_namespace(conn: &Connection, namespace: &str) -> Result<usize, MemoryError> {
    with_transaction(conn, |tx| {
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

        for (fact_id, fts_rowid, content) in &facts {
            tx.execute(
                "INSERT INTO facts_fts(facts_fts, rowid, content) VALUES('delete', ?1, ?2)",
                params![fts_rowid, content],
            )?;
            tx.execute(
                "DELETE FROM facts_rowid_map WHERE fact_id = ?1",
                params![fact_id],
            )?;

            #[cfg(feature = "hnsw")]
            enqueue_pending_index_op(
                tx,
                &format!("fact:{}", fact_id),
                "fact",
                PendingIndexOpKind::Delete,
            )?;
        }

        tx.execute("DELETE FROM facts WHERE namespace = ?1", params![namespace])?;
        Ok(facts.len())
    })
}

/// Get a fact by ID.
pub fn get_fact(conn: &Connection, fact_id: &str) -> Result<Option<Fact>, MemoryError> {
    let result = conn.query_row(
        "SELECT id, namespace, content, source, created_at, updated_at, metadata
         FROM facts WHERE id = ?1",
        params![fact_id],
        |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, Option<String>>(6)?,
            ))
        },
    );

    match result {
        Ok((id, namespace, content, source, created_at, updated_at, metadata_raw)) => {
            Ok(Some(Fact {
                metadata: parse_optional_json("facts", &id, "metadata", metadata_raw.as_deref())?,
                id,
                namespace,
                content,
                source,
                created_at,
                updated_at,
            }))
        }
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(err) => Err(MemoryError::Database(err)),
    }
}

/// Get a fact embedding vector.
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
        Err(err) => Err(MemoryError::Database(err)),
    }
}

/// List facts within a namespace.
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
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, Option<String>>(6)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(
            |(id, namespace, content, source, created_at, updated_at, metadata_raw)| {
                Ok(Fact {
                    metadata: parse_optional_json(
                        "facts",
                        &id,
                        "metadata",
                        metadata_raw.as_deref(),
                    )?,
                    id,
                    namespace,
                    content,
                    source,
                    created_at,
                    updated_at,
                })
            },
        )
        .collect::<Result<Vec<_>, MemoryError>>()?;

    Ok(facts)
}
