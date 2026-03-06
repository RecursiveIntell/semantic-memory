//! Session and message CRUD for conversation storage.

use crate::db::with_transaction;
use crate::error::MemoryError;
use crate::types::{Message, Role, Session};
use rusqlite::{params, Connection};

/// Create a new conversation session.
///
/// Returns the session ID (UUID v4).
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

/// Append a message to a session.
///
/// Updates the session's `updated_at` timestamp. Returns the message's auto-increment ID.
/// This does NOT add the message to the FTS or vector index — use
/// `add_message_with_fts` or `add_message_with_embedding` for searchable messages.
pub fn add_message(
    conn: &Connection,
    session_id: &str,
    role: Role,
    content: &str,
    token_count: Option<u32>,
    metadata: Option<&serde_json::Value>,
) -> Result<i64, MemoryError> {
    // Verify session exists
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
            "INSERT INTO messages (session_id, role, content, token_count, metadata) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![session_id, role.as_str(), content, token_count, metadata_str],
        )?;
        let msg_id = tx.last_insert_rowid();

        tx.execute(
            "UPDATE sessions SET updated_at = datetime('now') WHERE id = ?1",
            params![session_id],
        )?;

        Ok(msg_id)
    })
}

/// Get the most recent N messages from a session, in chronological order.
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
            let role_str: String = row.get(2)?;
            let metadata_str: Option<String> = row.get(6)?;
            Ok(Message {
                id: row.get(0)?,
                session_id: row.get(1)?,
                role: Role::from_str_value(&role_str).unwrap_or(Role::User),
                content: row.get(3)?,
                token_count: row.get(4)?,
                created_at: row.get(5)?,
                metadata: metadata_str.and_then(|s| serde_json::from_str(&s).ok()),
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    // Reverse to chronological order (we fetched newest-first)
    messages.reverse();
    Ok(messages)
}

/// Get messages from a session up to `max_tokens` total.
///
/// Walks backward from newest, accumulating token counts, stops when
/// the budget is exceeded. Returns messages in chronological order.
///
/// **Edge case:** The first (most recent) message is always included even
/// if it alone exceeds `max_tokens`. This ensures the method never returns
/// an empty Vec for a non-empty session. Callers that need strict budget
/// enforcement should check the total token count of returned messages.
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
            let role_str: String = row.get(2)?;
            let metadata_str: Option<String> = row.get(6)?;
            Ok(Message {
                id: row.get(0)?,
                session_id: row.get(1)?,
                role: Role::from_str_value(&role_str).unwrap_or(Role::User),
                content: row.get(3)?,
                token_count: row.get(4)?,
                created_at: row.get(5)?,
                metadata: metadata_str.and_then(|s| serde_json::from_str(&s).ok()),
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    let mut collected = Vec::new();
    let mut total_tokens: u32 = 0;

    for msg in all_messages {
        let msg_tokens = msg.token_count.unwrap_or(0);
        if total_tokens + msg_tokens > max_tokens && !collected.is_empty() {
            break;
        }
        total_tokens += msg_tokens;
        collected.push(msg);
    }

    // Reverse to chronological order
    collected.reverse();
    Ok(collected)
}

/// Get total token count for a session.
pub fn session_token_count(conn: &Connection, session_id: &str) -> Result<u64, MemoryError> {
    let count: i64 = conn.query_row(
        "SELECT COALESCE(SUM(token_count), 0) FROM messages WHERE session_id = ?1",
        params![session_id],
        |row| row.get(0),
    )?;
    Ok(count as u64)
}

/// Append a message to a session with a pre-computed embedding and FTS entry.
///
/// Same as `add_message` but also stores the embedding BLOB and inserts into
/// the FTS bridge + FTS table. Wrap in a transaction.
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

/// Append an embedded message with optional quantized embedding.
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
    // Verify session exists
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
        // INSERT into messages (with embedding + q8 BLOBs)
        tx.execute(
            "INSERT INTO messages (session_id, role, content, token_count, metadata, embedding, embedding_q8) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![session_id, role.as_str(), content, token_count, metadata_str, embedding_bytes, q8_bytes],
        )?;
        let msg_id = tx.last_insert_rowid();

        // INSERT into messages_rowid_map
        tx.execute(
            "INSERT INTO messages_rowid_map (message_id) VALUES (?1)",
            params![msg_id],
        )?;
        let fts_rowid = tx.last_insert_rowid();

        // INSERT into messages_fts
        tx.execute(
            "INSERT INTO messages_fts(rowid, content) VALUES (?1, ?2)",
            params![fts_rowid, content],
        )?;

        // UPDATE sessions SET updated_at
        tx.execute(
            "UPDATE sessions SET updated_at = datetime('now') WHERE id = ?1",
            params![session_id],
        )?;

        Ok(msg_id)
    })
}

/// Append a message to a session with FTS indexing but no embedding.
///
/// Same as `add_message` but also inserts into `messages_rowid_map` and
/// `messages_fts`. The embedding and embedding_q8 columns are set to NULL.
/// This is a fallback path when embedding fails: the message is still
/// findable via BM25 search, just not via vector search.
pub fn add_message_with_fts(
    conn: &Connection,
    session_id: &str,
    role: Role,
    content: &str,
    token_count: Option<u32>,
    metadata: Option<&serde_json::Value>,
) -> Result<i64, MemoryError> {
    // Verify session exists
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
        // INSERT into messages (embedding = NULL, embedding_q8 = NULL)
        tx.execute(
            "INSERT INTO messages (session_id, role, content, token_count, metadata, embedding, embedding_q8) VALUES (?1, ?2, ?3, ?4, ?5, NULL, NULL)",
            params![session_id, role.as_str(), content, token_count, metadata_str],
        )?;
        let msg_id = tx.last_insert_rowid();

        // INSERT into messages_rowid_map
        tx.execute(
            "INSERT INTO messages_rowid_map (message_id) VALUES (?1)",
            params![msg_id],
        )?;
        let fts_rowid = tx.last_insert_rowid();

        // INSERT into messages_fts
        tx.execute(
            "INSERT INTO messages_fts(rowid, content) VALUES (?1, ?2)",
            params![fts_rowid, content],
        )?;

        // UPDATE sessions SET updated_at
        tx.execute(
            "UPDATE sessions SET updated_at = datetime('now') WHERE id = ?1",
            params![session_id],
        )?;

        Ok(msg_id)
    })
}

/// Delete FTS entries for a single message. Needed for cleanup.
pub fn delete_message_fts(conn: &Connection, message_id: i64) -> Result<(), MemoryError> {
    // Get content and FTS rowid
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
    // If no FTS entry exists (non-embedded message), that's fine — nothing to clean up.

    Ok(())
}

/// Delete a session and all its messages (CASCADE).
///
/// Cleans up message FTS entries before CASCADE to avoid ghost entries.
pub fn delete_session(conn: &Connection, session_id: &str) -> Result<(), MemoryError> {
    with_transaction(conn, |tx| {
        // Clean up message FTS entries before CASCADE
        let fts_data: Vec<(i64, String, i64)> = {
            let mut stmt = tx.prepare(
                "SELECT m.id, m.content, mm.rowid
                 FROM messages m
                 JOIN messages_rowid_map mm ON mm.message_id = m.id
                 WHERE m.session_id = ?1",
            )?;
            let result = stmt
                .query_map(params![session_id], |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            result
        };

        for (_msg_id, content, fts_rowid) in &fts_data {
            tx.execute(
                "INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', ?1, ?2)",
                params![fts_rowid, content],
            )?;
        }

        // Now delete session (CASCADE handles messages + messages_rowid_map)
        let affected = tx.execute("DELETE FROM sessions WHERE id = ?1", params![session_id])?;
        if affected == 0 {
            return Err(MemoryError::SessionNotFound(session_id.to_string()));
        }

        Ok(())
    })
}

/// List recent sessions with message counts, newest first.
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
            let metadata_str: Option<String> = row.get(4)?;
            let message_count: i64 = row.get(5)?;
            Ok(Session {
                id: row.get(0)?,
                channel: row.get(1)?,
                created_at: row.get(2)?,
                updated_at: row.get(3)?,
                metadata: metadata_str.and_then(|s| serde_json::from_str(&s).ok()),
                message_count: message_count as u32,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    Ok(sessions)
}

/// Update a session's channel (display name).
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
