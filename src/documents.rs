//! Document ingestion pipeline: chunk, embed, store, and queue sidecar updates.

use crate::chunker;
use crate::db;
#[cfg(feature = "hnsw")]
use crate::db::IndexOpKind;
use crate::error::MemoryError;
use crate::quantize::{self, Quantizer};
use crate::types::Document;
use crate::{merge_trace_ctx, MemoryStore};
use rusqlite::{params, Connection};
use stack_ids::TraceCtx;

/// A single chunk to insert: `(content, embedding_bytes, q8_bytes, token_count_estimate)`.
pub type ChunkRow = (String, Vec<u8>, Option<Vec<u8>>, usize);

pub fn insert_document_with_chunks(
    conn: &Connection,
    doc_id: &str,
    title: &str,
    namespace: &str,
    source_path: Option<&str>,
    metadata: Option<&serde_json::Value>,
    chunks: &[ChunkRow],
) -> Result<Vec<String>, MemoryError> {
    let chunk_ids: Vec<String> = (0..chunks.len())
        .map(|_| uuid::Uuid::new_v4().to_string())
        .collect();
    insert_document_with_chunks_and_ids(
        conn,
        doc_id,
        title,
        namespace,
        source_path,
        metadata,
        chunks,
        &chunk_ids,
    )?;
    Ok(chunk_ids)
}

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
    if chunks.len() != chunk_ids.len() {
        return Err(MemoryError::Other(
            "chunks and chunk_ids must have the same length".to_string(),
        ));
    }

    let metadata_str = metadata.map(|value| value.to_string());
    db::with_transaction(conn, |tx| {
        tx.execute(
            "INSERT INTO documents (id, title, source_path, namespace, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![doc_id, title, source_path, namespace, metadata_str],
        )?;

        for (chunk_index, ((content, embedding_bytes, q8_bytes, token_count), chunk_id)) in
            chunks.iter().zip(chunk_ids.iter()).enumerate()
        {
            tx.execute(
                "INSERT INTO chunks (id, document_id, chunk_index, content, token_count, embedding, embedding_q8)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    chunk_id,
                    doc_id,
                    chunk_index as i64,
                    content,
                    *token_count as i64,
                    embedding_bytes,
                    q8_bytes.as_deref()
                ],
            )?;

            tx.execute(
                "INSERT INTO chunks_rowid_map (chunk_id) VALUES (?1)",
                params![chunk_id],
            )?;
            let fts_rowid = tx.last_insert_rowid();
            tx.execute(
                "INSERT INTO chunks_fts (rowid, content) VALUES (?1, ?2)",
                params![fts_rowid, content],
            )?;

            #[cfg(feature = "hnsw")]
            db::queue_pending_index_op(
                tx,
                &format!("chunk:{}", chunk_id),
                "chunk",
                IndexOpKind::Upsert,
            )?;
        }

        Ok(())
    })
}

pub fn delete_document_with_chunks(
    conn: &Connection,
    document_id: &str,
) -> Result<Vec<String>, MemoryError> {
    db::with_transaction(conn, |tx| {
        let mut stmt = tx.prepare(
            "SELECT c.id, c.content, cm.rowid
             FROM chunks c
             JOIN chunks_rowid_map cm ON cm.chunk_id = c.id
             WHERE c.document_id = ?1",
        )?;
        let chunk_rows: Vec<(String, String, i64)> = stmt
            .query_map(params![document_id], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        let chunk_ids: Vec<String> = chunk_rows.iter().map(|(id, _, _)| id.clone()).collect();

        for (chunk_id, content, fts_rowid) in &chunk_rows {
            tx.execute(
                "INSERT INTO chunks_fts (chunks_fts, rowid, content) VALUES ('delete', ?1, ?2)",
                params![fts_rowid, content],
            )?;
            tx.execute(
                "DELETE FROM chunks_rowid_map WHERE chunk_id = ?1",
                params![chunk_id],
            )?;
            #[cfg(feature = "hnsw")]
            db::queue_pending_index_op(
                tx,
                &format!("chunk:{}", chunk_id),
                "chunk",
                IndexOpKind::Delete,
            )?;
        }

        tx.execute(
            "DELETE FROM chunks WHERE document_id = ?1",
            params![document_id],
        )?;
        let affected = tx.execute("DELETE FROM documents WHERE id = ?1", params![document_id])?;
        if affected == 0 {
            return Err(MemoryError::DocumentNotFound(document_id.to_string()));
        }

        Ok(chunk_ids)
    })
}

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

    let rows = stmt
        .query_map(params![namespace, limit as i64, offset as i64], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, Option<String>>(5)?,
                row.get::<_, i64>(6)? as u32,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;

    rows.into_iter()
        .map(
            |(id, title, source_path, namespace, created_at, metadata_raw, chunk_count)| {
                Ok(Document {
                    metadata: db::parse_optional_json(
                        "documents",
                        &id,
                        "metadata",
                        metadata_raw.as_deref(),
                    )?,
                    id,
                    title,
                    source_path,
                    namespace,
                    created_at,
                    chunk_count,
                })
            },
        )
        .collect()
}

impl MemoryStore {
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
        let chunks: Vec<ChunkRow> = text_chunks
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
            insert_document_with_chunks(conn, &did, &t, &ns, sp.as_deref(), meta.as_ref(), &chunks)
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
        self.with_write_conn(move |conn| delete_document_with_chunks(conn, &did))
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
        self.with_read_conn(move |conn| list_documents(conn, &ns, limit, offset))
            .await
    }

    /// Count the number of chunks for a document.
    pub async fn count_chunks_for_document(&self, document_id: &str) -> Result<usize, MemoryError> {
        let did = document_id.to_string();
        self.with_read_conn(move |conn| count_chunks_for_document(conn, &did))
            .await
    }
}
