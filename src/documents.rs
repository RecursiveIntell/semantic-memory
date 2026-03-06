//! Document ingestion pipeline: chunk, embed, store.

use crate::db::with_transaction;
use crate::error::MemoryError;
use crate::types::Document;
use rusqlite::{params, Connection};

/// A single chunk to insert: (content, embedding_bytes, q8_bytes, token_count_estimate).
pub type ChunkRow = (String, Vec<u8>, Option<Vec<u8>>, usize);

/// Insert a document and all its chunks + FTS entries in a single transaction.
///
/// `chunks` is a vec of (content, embedding_bytes, q8_bytes, token_count_estimate).
pub fn insert_document_with_chunks(
    conn: &Connection,
    doc_id: &str,
    title: &str,
    namespace: &str,
    source_path: Option<&str>,
    metadata: Option<&serde_json::Value>,
    chunks: &[ChunkRow],
) -> Result<(), MemoryError> {
    let metadata_str = metadata.map(|m| m.to_string());
    with_transaction(conn, |tx| {
        // Insert document
        tx.execute(
            "INSERT INTO documents (id, title, source_path, namespace, metadata) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![doc_id, title, source_path, namespace, metadata_str],
        )?;

        // Insert each chunk
        for (chunk_index, (content, embedding_bytes, q8_bytes, token_count)) in
            chunks.iter().enumerate()
        {
            let chunk_id = uuid::Uuid::new_v4().to_string();

            tx.execute(
                "INSERT INTO chunks (id, document_id, chunk_index, content, token_count, embedding, embedding_q8) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![chunk_id, doc_id, chunk_index as i64, content, *token_count as i64, embedding_bytes, q8_bytes.as_deref()],
            )?;

            // Insert into rowid bridge
            tx.execute(
                "INSERT INTO chunks_rowid_map (chunk_id) VALUES (?1)",
                params![chunk_id],
            )?;
            let fts_rowid = tx.last_insert_rowid();

            // Insert into FTS
            tx.execute(
                "INSERT INTO chunks_fts(rowid, content) VALUES (?1, ?2)",
                params![fts_rowid, content],
            )?;
        }

        Ok(())
    })
}

/// Insert a document and all its chunks + FTS entries in a single transaction,
/// using pre-generated chunk IDs (for HNSW key mapping).
///
/// `chunks` is a vec of (content, embedding_bytes, q8_bytes, token_count_estimate).
/// `chunk_ids` must have the same length as `chunks`.
#[allow(clippy::too_many_arguments)]
pub fn insert_document_with_chunks_and_ids(
    conn: &Connection,
    doc_id: &str,
    title: &str,
    namespace: &str,
    source_path: Option<&str>,
    metadata: Option<&serde_json::Value>,
    chunks: &[ChunkRow],
    chunk_ids: &[String],
) -> Result<(), MemoryError> {
    assert_eq!(
        chunks.len(),
        chunk_ids.len(),
        "chunks and chunk_ids must have same length"
    );
    let metadata_str = metadata.map(|m| m.to_string());
    with_transaction(conn, |tx| {
        // Insert document
        tx.execute(
            "INSERT INTO documents (id, title, source_path, namespace, metadata) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![doc_id, title, source_path, namespace, metadata_str],
        )?;

        // Insert each chunk with pre-generated ID
        for (chunk_index, ((content, embedding_bytes, q8_bytes, token_count), chunk_id)) in
            chunks.iter().zip(chunk_ids.iter()).enumerate()
        {
            tx.execute(
                "INSERT INTO chunks (id, document_id, chunk_index, content, token_count, embedding, embedding_q8) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![chunk_id, doc_id, chunk_index as i64, content, *token_count as i64, embedding_bytes, q8_bytes.as_deref()],
            )?;

            // Insert into rowid bridge
            tx.execute(
                "INSERT INTO chunks_rowid_map (chunk_id) VALUES (?1)",
                params![chunk_id],
            )?;
            let fts_rowid = tx.last_insert_rowid();

            // Insert into FTS
            tx.execute(
                "INSERT INTO chunks_fts(rowid, content) VALUES (?1, ?2)",
                params![fts_rowid, content],
            )?;
        }

        Ok(())
    })
}

/// Delete a document and all its chunks + FTS entries in a transaction.
pub fn delete_document_with_chunks(
    conn: &Connection,
    document_id: &str,
) -> Result<(), MemoryError> {
    with_transaction(conn, |tx| {
        // Get all chunks for this document (collect before mutating)
        let chunk_data: Vec<(String, String, i64)> = {
            let mut stmt = tx.prepare(
                "SELECT c.id, c.content, cm.rowid
                 FROM chunks c
                 JOIN chunks_rowid_map cm ON cm.chunk_id = c.id
                 WHERE c.document_id = ?1",
            )?;
            let result = stmt
                .query_map(params![document_id], |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            result
        };

        // Delete FTS entries for each chunk
        for (chunk_id, content, fts_rowid) in &chunk_data {
            tx.execute(
                "INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', ?1, ?2)",
                params![fts_rowid, content],
            )?;
            tx.execute(
                "DELETE FROM chunks_rowid_map WHERE chunk_id = ?1",
                params![chunk_id],
            )?;
        }

        // Delete chunks and document
        tx.execute(
            "DELETE FROM chunks WHERE document_id = ?1",
            params![document_id],
        )?;
        let affected = tx.execute("DELETE FROM documents WHERE id = ?1", params![document_id])?;

        if affected == 0 {
            return Err(MemoryError::DocumentNotFound(document_id.to_string()));
        }

        Ok(())
    })
}

/// Count the number of chunks for a document.
pub fn count_chunks_for_document(
    conn: &Connection,
    document_id: &str,
) -> Result<usize, MemoryError> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM chunks WHERE document_id = ?1",
        params![document_id],
        |row| row.get(0),
    )?;
    Ok(count as usize)
}

/// List documents in a namespace with chunk counts.
pub fn list_documents(
    conn: &Connection,
    namespace: &str,
    limit: usize,
    offset: usize,
) -> Result<Vec<Document>, MemoryError> {
    let mut stmt = conn.prepare(
        "SELECT d.id, d.title, d.source_path, d.namespace, d.created_at, d.metadata,
                (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id) AS chunk_count
         FROM documents d
         WHERE d.namespace = ?1
         ORDER BY d.created_at DESC
         LIMIT ?2 OFFSET ?3",
    )?;

    let docs = stmt
        .query_map(params![namespace, limit as i64, offset as i64], |row| {
            let metadata_str: Option<String> = row.get(5)?;
            let chunk_count: i64 = row.get(6)?;
            Ok(Document {
                id: row.get(0)?,
                title: row.get(1)?,
                source_path: row.get(2)?,
                namespace: row.get(3)?,
                created_at: row.get(4)?,
                metadata: metadata_str.and_then(|s| serde_json::from_str(&s).ok()),
                chunk_count: chunk_count as u32,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    Ok(docs)
}
