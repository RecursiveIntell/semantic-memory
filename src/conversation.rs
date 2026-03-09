//! Session and message CRUD for conversation storage.

#[cfg(feature = "hnsw")]
use crate::db::{enqueue_pending_index_op, PendingIndexOpKind};
use crate::db::{parse_optional_json, parse_role, with_transaction};
use crate::error::MemoryError;
use crate::types::{Message, Role, Session};
use rusqlite::{params, Connection};

/// Create a new conversation session and return its UUID.
pub fn create_session(
    conn: &Connection,
    channel: &str,
    metadata: Option<&serde_json::Value>,
) -> Result<String, MemoryError> {
    let id = uuid::Uuid::new_v4().to_string();
    let metadata_str = metadata.map(|m| m.to_string());
    conn.execute(
        "INSERT INTO sessions (id, channel, metadata) VALUES (?1, ?2, ?3)",
        params![id, channel, metadata_str],
    )?;
    Ok(id)
}

/// Append a message to a session without search indexes.
#[allow(dead_code)]
pub fn add_message(
    conn: &Connection,
    session_id: &str,
    role: Role,
    content: &str,
    token_count: Option<u32>,
    metadata: Option<&serde_json::Value>,
) -> Result<i64, MemoryError> {
    let exists: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sessions WHERE id = ?1)",
        params![session_id],
        |row| row.get(0),
    )?;
    if !exists {
        return Err(MemoryError::SessionNotFound(session_id.to_string()));
    }

    let metadata_str = metadata.map(|m| m.to_string());
    with_transaction(conn, |tx| {
        tx.execute(
            "INSERT INTO messages (session_id, role, content, token_count, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                session_id,
                role.as_str(),
                content,
                token_count,
                metadata_str
            ],
        )?;
        let msg_id = tx.last_insert_rowid();
        tx.execute(
            "UPDATE sessions SET updated_at = datetime('now') WHERE id = ?1",
            params![session_id],
        )?;
        Ok(msg_id)
    })
}

/// Get the most recent N messages from a session in chronological order.
pub fn get_recent_messages(
    conn: &Connection,
    session_id: &str,
    limit: usize,
) -> Result<Vec<Message>, MemoryError> {
    let mut stmt = conn.prepare(
        "SELECT id, session_id, role, content, token_count, created_at, metadata
         FROM messages
         WHERE session_id = ?1
         ORDER BY created_at DESC, id DESC
         LIMIT ?2",
    )?;

    let mut messages: Vec<Message> = stmt
        .query_map(params![session_id, limit as i64], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, Option<u32>>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, Option<String>>(6)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(
            |(id, session_id, role_raw, content, token_count, created_at, metadata_raw)| {
                Ok(Message {
                    role: parse_role("messages", &id.to_string(), &role_raw)?,
                    metadata: parse_optional_json(
                        "messages",
                        &id.to_string(),
                        "metadata",
                        metadata_raw.as_deref(),
                    )?,
                    id,
                    session_id,
                    content,
                    token_count,
                    created_at,
                })
            },
        )
        .collect::<Result<Vec<_>, MemoryError>>()?;

    messages.reverse();
    Ok(messages)
}

/// Get messages from a session while staying under the token budget.
pub fn get_messages_within_budget(
    conn: &Connection,
    session_id: &str,
    max_tokens: u32,
) -> Result<Vec<Message>, MemoryError> {
    let mut stmt = conn.prepare(
        "SELECT id, session_id, role, content, token_count, created_at, metadata
         FROM messages
         WHERE session_id = ?1
         ORDER BY created_at DESC, id DESC",
    )?;

    let all_messages: Vec<Message> = stmt
        .query_map(params![session_id], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, Option<u32>>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, Option<String>>(6)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(
            |(id, session_id, role_raw, content, token_count, created_at, metadata_raw)| {
                Ok(Message {
                    role: parse_role("messages", &id.to_string(), &role_raw)?,
                    metadata: parse_optional_json(
                        "messages",
                        &id.to_string(),
                        "metadata",
                        metadata_raw.as_deref(),
                    )?,
                    id,
                    session_id,
                    content,
                    token_count,
                    created_at,
                })
            },
        )
        .collect::<Result<Vec<_>, MemoryError>>()?;

    let mut collected = Vec::new();
    let mut total_tokens = 0u32;
    for msg in all_messages {
        let msg_tokens = msg.token_count.unwrap_or(0);
        if total_tokens + msg_tokens > max_tokens && !collected.is_empty() {
            break;
        }
        total_tokens += msg_tokens;
        collected.push(msg);
    }

    collected.reverse();
    Ok(collected)
}

/// Get the total token count for a session.
pub fn session_token_count(conn: &Connection, session_id: &str) -> Result<u64, MemoryError> {
    let count: i64 = conn.query_row(
        "SELECT COALESCE(SUM(token_count), 0) FROM messages WHERE session_id = ?1",
        params![session_id],
        |row| row.get(0),
    )?;
    Ok(count as u64)
}

/// Append a message with embedding + q8 + FTS entries.
#[allow(clippy::too_many_arguments)]
pub fn add_message_with_embedding_q8(
    conn: &Connection,
    session_id: &str,
    role: Role,
    content: &str,
    token_count: Option<u32>,
    metadata: Option<&serde_json::Value>,
    embedding_bytes: &[u8],
    q8_bytes: Option<&[u8]>,
) -> Result<i64, MemoryError> {
    let exists: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sessions WHERE id = ?1)",
        params![session_id],
        |row| row.get(0),
    )?;
    if !exists {
        return Err(MemoryError::SessionNotFound(session_id.to_string()));
    }

    let metadata_str = metadata.map(|m| m.to_string());
    with_transaction(conn, |tx| {
        tx.execute(
            "INSERT INTO messages (session_id, role, content, token_count, metadata, embedding, embedding_q8)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                session_id,
                role.as_str(),
                content,
                token_count,
                metadata_str,
                embedding_bytes,
                q8_bytes
            ],
        )?;
        let msg_id = tx.last_insert_rowid();

        tx.execute(
            "INSERT INTO messages_rowid_map (message_id) VALUES (?1)",
            params![msg_id],
        )?;
        let fts_rowid = tx.last_insert_rowid();
        tx.execute(
            "INSERT INTO messages_fts(rowid, content) VALUES (?1, ?2)",
            params![fts_rowid, content],
        )?;

        #[cfg(feature = "hnsw")]
        enqueue_pending_index_op(
            tx,
            &format!("msg:{}", msg_id),
            "message",
            PendingIndexOpKind::Upsert,
        )?;

        tx.execute(
            "UPDATE sessions SET updated_at = datetime('now') WHERE id = ?1",
            params![session_id],
        )?;

        Ok(msg_id)
    })
}

/// Backward-compatible wrapper for embedded messages without q8 input.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn add_message_with_embedding(
    conn: &Connection,
    session_id: &str,
    role: Role,
    content: &str,
    token_count: Option<u32>,
    metadata: Option<&serde_json::Value>,
    embedding_bytes: &[u8],
) -> Result<i64, MemoryError> {
    add_message_with_embedding_q8(
        conn,
        session_id,
        role,
        content,
        token_count,
        metadata,
        embedding_bytes,
        None,
    )
}

/// Append a message with FTS indexing but no embedding.
pub fn add_message_with_fts(
    conn: &Connection,
    session_id: &str,
    role: Role,
    content: &str,
    token_count: Option<u32>,
    metadata: Option<&serde_json::Value>,
) -> Result<i64, MemoryError> {
    let exists: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sessions WHERE id = ?1)",
        params![session_id],
        |row| row.get(0),
    )?;
    if !exists {
        return Err(MemoryError::SessionNotFound(session_id.to_string()));
    }

    let metadata_str = metadata.map(|m| m.to_string());
    with_transaction(conn, |tx| {
        tx.execute(
            "INSERT INTO messages (session_id, role, content, token_count, metadata, embedding, embedding_q8)
             VALUES (?1, ?2, ?3, ?4, ?5, NULL, NULL)",
            params![session_id, role.as_str(), content, token_count, metadata_str],
        )?;
        let msg_id = tx.last_insert_rowid();

        tx.execute(
            "INSERT INTO messages_rowid_map (message_id) VALUES (?1)",
            params![msg_id],
        )?;
        let fts_rowid = tx.last_insert_rowid();
        tx.execute(
            "INSERT INTO messages_fts(rowid, content) VALUES (?1, ?2)",
            params![fts_rowid, content],
        )?;
        tx.execute(
            "UPDATE sessions SET updated_at = datetime('now') WHERE id = ?1",
            params![session_id],
        )?;

        Ok(msg_id)
    })
}

/// Delete FTS entries for a single message.
#[allow(dead_code)]
pub fn delete_message_fts(conn: &Connection, message_id: i64) -> Result<(), MemoryError> {
    let result: Result<(String, i64), _> = conn.query_row(
        "SELECT m.content, mm.rowid
         FROM messages m
         JOIN messages_rowid_map mm ON mm.message_id = m.id
         WHERE m.id = ?1",
        params![message_id],
        |row| Ok((row.get(0)?, row.get(1)?)),
    );

    if let Ok((content, fts_rowid)) = result {
        conn.execute(
            "INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', ?1, ?2)",
            params![fts_rowid, content],
        )?;
        conn.execute(
            "DELETE FROM messages_rowid_map WHERE message_id = ?1",
            params![message_id],
        )?;
    }

    Ok(())
}

/// Delete a session and all its messages.
pub fn delete_session(conn: &Connection, session_id: &str) -> Result<(), MemoryError> {
    with_transaction(conn, |tx| {
        let fts_data: Vec<(i64, String, i64, bool)> = {
            let mut stmt = tx.prepare(
                "SELECT m.id, m.content, mm.rowid, m.embedding IS NOT NULL
                 FROM messages m
                 JOIN messages_rowid_map mm ON mm.message_id = m.id
                 WHERE m.session_id = ?1",
            )?;
            let rows = stmt.query_map(params![session_id], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })?;
            rows.collect::<Result<Vec<_>, _>>()?
        };

        for (msg_id, content, fts_rowid, has_embedding) in &fts_data {
            tx.execute(
                "INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', ?1, ?2)",
                params![fts_rowid, content],
            )?;

            #[cfg(feature = "hnsw")]
            if *has_embedding {
                enqueue_pending_index_op(
                    tx,
                    &format!("msg:{}", msg_id),
                    "message",
                    PendingIndexOpKind::Delete,
                )?;
            }

            #[cfg(not(feature = "hnsw"))]
            {
                let _ = msg_id;
                let _ = has_embedding;
            }
        }

        let affected = tx.execute("DELETE FROM sessions WHERE id = ?1", params![session_id])?;
        if affected == 0 {
            return Err(MemoryError::SessionNotFound(session_id.to_string()));
        }

        Ok(())
    })
}

/// List recent sessions with message counts.
pub fn list_sessions(
    conn: &Connection,
    limit: usize,
    offset: usize,
) -> Result<Vec<Session>, MemoryError> {
    let mut stmt = conn.prepare(
        "SELECT s.id, s.channel, s.created_at, s.updated_at, s.metadata,
                COUNT(m.id) AS message_count
         FROM sessions s
         LEFT JOIN messages m ON m.session_id = s.id
         GROUP BY s.id
         ORDER BY s.updated_at DESC
         LIMIT ?1 OFFSET ?2",
    )?;

    let sessions = stmt
        .query_map(params![limit as i64, offset as i64], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, i64>(5)? as u32,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(
            |(id, channel, created_at, updated_at, metadata_raw, message_count)| {
                Ok(Session {
                    metadata: parse_optional_json(
                        "sessions",
                        &id,
                        "metadata",
                        metadata_raw.as_deref(),
                    )?,
                    id,
                    channel,
                    created_at,
                    updated_at,
                    message_count,
                })
            },
        )
        .collect::<Result<Vec<_>, MemoryError>>()?;

    Ok(sessions)
}

/// Update a session channel.
pub fn rename_session(
    conn: &Connection,
    session_id: &str,
    new_channel: &str,
) -> Result<(), MemoryError> {
    let affected = conn.execute(
        "UPDATE sessions SET channel = ?1, updated_at = datetime('now') WHERE id = ?2",
        params![new_channel, session_id],
    )?;
    if affected == 0 {
        return Err(MemoryError::SessionNotFound(session_id.to_string()));
    }
    Ok(())
}
