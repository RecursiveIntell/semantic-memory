use crate::db;
#[cfg(feature = "hnsw")]
use crate::db::IndexOpKind;
use crate::error::MemoryError;
use crate::types::{EpisodeMeta, EpisodeOutcome, VerificationStatus};
use rusqlite::{params, Connection};

// ─── Centralized episode identity helpers ──────────────────────────────

/// Canonical HNSW item key for an episode.
pub(crate) fn episode_item_key(episode_id: &str) -> String {
    format!("episode:{episode_id}")
}

/// Canonical graph node ID for an episode.
pub(crate) fn episode_node_id(episode_id: &str) -> String {
    format!("episode:{episode_id}")
}

/// Resolve the primary (first-created) episode_id for a document.
/// This is **legacy compatibility** behavior for APIs that still target
/// a single episode by document_id. Canonical code should use episode_id directly.
pub(crate) fn resolve_primary_episode_id_legacy(
    conn: &Connection,
    document_id: &str,
) -> Result<Option<String>, MemoryError> {
    match conn.query_row(
        "SELECT episode_id FROM episodes WHERE document_id = ?1 ORDER BY created_at ASC LIMIT 1",
        params![document_id],
        |row| row.get(0),
    ) {
        Ok(id) => Ok(Some(id)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(err) => Err(MemoryError::Database(err)),
    }
}

/// List all episode_ids for a document, ordered by creation time.
pub(crate) fn list_document_episode_ids(
    conn: &Connection,
    document_id: &str,
) -> Result<Vec<String>, MemoryError> {
    let mut stmt = conn.prepare(
        "SELECT episode_id FROM episodes WHERE document_id = ?1 ORDER BY created_at ASC",
    )?;
    let ids = stmt
        .query_map(params![document_id], |row| row.get::<_, String>(0))?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ids)
}

/// Insert a new episode with an explicit episode_id (canonical path).
/// Returns the episode_id.
#[allow(clippy::too_many_arguments)]
pub(crate) fn create_episode(
    conn: &Connection,
    episode_id: &str,
    document_id: &str,
    meta: &EpisodeMeta,
    search_text: &str,
    embedding_bytes: &[u8],
    q8_bytes: Option<&[u8]>,
    trace_id: Option<&str>,
) -> Result<String, MemoryError> {
    let cause_ids_json =
        serde_json::to_string(&meta.cause_ids).map_err(|e| MemoryError::Other(e.to_string()))?;
    let verification_json = serde_json::to_string(&meta.verification_status)
        .map_err(|e| MemoryError::Other(e.to_string()))?;
    #[cfg(feature = "hnsw")]
    let item_key = episode_item_key(episode_id);

    db::with_transaction(conn, |tx| {
        let exists: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM documents WHERE id = ?1)",
            params![document_id],
            |row| row.get(0),
        )?;
        if !exists {
            return Err(MemoryError::DocumentNotFound(document_id.to_string()));
        }

        tx.execute(
            "INSERT INTO episodes
                (episode_id, document_id, cause_ids, effect_type, outcome, confidence,
                 verification_status, experiment_id, search_text, embedding, embedding_q8,
                 trace_id, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, datetime('now'))",
            params![
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
                q8_bytes,
                trace_id
            ],
        )?;

        // Insert FTS mapping
        tx.execute(
            "INSERT INTO episodes_rowid_map (episode_id, document_id) VALUES (?1, ?2)",
            params![episode_id, document_id],
        )?;
        let fts_rowid: i64 = tx.query_row(
            "SELECT rowid FROM episodes_rowid_map WHERE episode_id = ?1",
            params![episode_id],
            |row| row.get(0),
        )?;
        tx.execute(
            "INSERT INTO episodes_fts (rowid, content) VALUES (?1, ?2)",
            params![fts_rowid, search_text],
        )?;

        // Populate normalized causal edges
        sync_causal_edges(tx, episode_id, &meta.cause_ids)?;

        #[cfg(feature = "hnsw")]
        db::queue_pending_index_op(tx, &item_key, "episode", IndexOpKind::Upsert)?;
        Ok(episode_id.to_string())
    })
}

/// Legacy compatibility: upsert the primary episode for a document.
///
/// If an episode already exists for this document, updates the first one.
/// Otherwise creates a new one with a deterministic `{document_id}-ep0` episode_id.
/// Canonical callers should use `create_episode()` with an explicit episode_id instead.
#[allow(clippy::too_many_arguments)]
pub(crate) fn upsert_episode(
    conn: &Connection,
    document_id: &str,
    meta: &EpisodeMeta,
    search_text: &str,
    embedding_bytes: &[u8],
    q8_bytes: Option<&[u8]>,
    trace_id: Option<&str>,
) -> Result<String, MemoryError> {
    let cause_ids_json =
        serde_json::to_string(&meta.cause_ids).map_err(|e| MemoryError::Other(e.to_string()))?;
    let verification_json = serde_json::to_string(&meta.verification_status)
        .map_err(|e| MemoryError::Other(e.to_string()))?;

    // Legacy compat: resolve the primary episode for this document
    let existing_episode_id = resolve_primary_episode_id_legacy(conn, document_id)?;

    let episode_id = existing_episode_id.unwrap_or_else(|| format!("{}-ep0", document_id));

    #[cfg(feature = "hnsw")]
    let item_key = episode_item_key(&episode_id);

    db::with_transaction(conn, |tx| {
        let old_search_text: Option<String> = tx
            .query_row(
                "SELECT search_text FROM episodes WHERE episode_id = ?1",
                params![episode_id],
                |row| row.get(0),
            )
            .ok();
        let exists: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM documents WHERE id = ?1)",
            params![document_id],
            |row| row.get(0),
        )?;
        if !exists {
            return Err(MemoryError::DocumentNotFound(document_id.to_string()));
        }

        if old_search_text.is_some() {
            // Update existing episode
            tx.execute(
                "UPDATE episodes SET
                    cause_ids = ?1,
                    effect_type = ?2,
                    outcome = ?3,
                    confidence = ?4,
                    verification_status = ?5,
                    experiment_id = ?6,
                    search_text = ?7,
                    embedding = ?8,
                    embedding_q8 = ?9,
                    trace_id = COALESCE(?10, trace_id),
                    updated_at = datetime('now')
                 WHERE episode_id = ?11",
                params![
                    cause_ids_json,
                    meta.effect_type,
                    meta.outcome.as_str(),
                    meta.confidence,
                    verification_json,
                    meta.experiment_id,
                    search_text,
                    embedding_bytes,
                    q8_bytes,
                    trace_id,
                    episode_id
                ],
            )?;

            // Update FTS
            let fts_rowid: i64 = tx.query_row(
                "SELECT rowid FROM episodes_rowid_map WHERE episode_id = ?1",
                params![episode_id],
                |row| row.get(0),
            )?;
            if let Some(old_text) = old_search_text {
                tx.execute(
                    "INSERT INTO episodes_fts (episodes_fts, rowid, content) VALUES ('delete', ?1, ?2)",
                    params![fts_rowid, old_text],
                )?;
            }
            tx.execute(
                "INSERT INTO episodes_fts (rowid, content) VALUES (?1, ?2)",
                params![fts_rowid, search_text],
            )?;
        } else {
            // Insert new episode
            tx.execute(
                "INSERT INTO episodes
                    (episode_id, document_id, cause_ids, effect_type, outcome, confidence,
                     verification_status, experiment_id, search_text, embedding, embedding_q8,
                     trace_id, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, datetime('now'))",
                params![
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
                    q8_bytes,
                    trace_id
                ],
            )?;

            // Insert FTS mapping
            tx.execute(
                "INSERT INTO episodes_rowid_map (episode_id, document_id) VALUES (?1, ?2)",
                params![episode_id, document_id],
            )?;
            let fts_rowid: i64 = tx.query_row(
                "SELECT rowid FROM episodes_rowid_map WHERE episode_id = ?1",
                params![episode_id],
                |row| row.get(0),
            )?;
            tx.execute(
                "INSERT INTO episodes_fts (rowid, content) VALUES (?1, ?2)",
                params![fts_rowid, search_text],
            )?;
        }

        // Sync normalized causal edges
        sync_causal_edges(tx, &episode_id, &meta.cause_ids)?;

        #[cfg(feature = "hnsw")]
        db::queue_pending_index_op(tx, &item_key, "episode", IndexOpKind::Upsert)?;
        Ok(episode_id.to_string())
    })
}

/// Synchronize the episode_causes table with the given cause_ids.
fn sync_causal_edges(
    tx: &rusqlite::Transaction<'_>,
    episode_id: &str,
    cause_ids: &[String],
) -> Result<(), MemoryError> {
    tx.execute(
        "DELETE FROM episode_causes WHERE episode_id = ?1",
        params![episode_id],
    )?;
    for (ordinal, cause_id) in cause_ids.iter().enumerate() {
        tx.execute(
            "INSERT OR IGNORE INTO episode_causes (episode_id, cause_node_id, ordinal)
             VALUES (?1, ?2, ?3)",
            params![episode_id, cause_id, ordinal as i64],
        )?;
    }
    Ok(())
}

/// Legacy compatibility: update the primary episode's outcome for a document.
/// Resolves the first-created episode and delegates to `update_episode_outcome_by_id`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn update_episode_outcome(
    conn: &Connection,
    document_id: &str,
    outcome: EpisodeOutcome,
    confidence: f32,
    experiment_id: Option<&str>,
    verification_status: &VerificationStatus,
    search_text: &str,
    embedding_bytes: &[u8],
    q8_bytes: Option<&[u8]>,
) -> Result<(), MemoryError> {
    // Legacy compat: resolve the primary episode for this document
    let episode_id = resolve_primary_episode_id_legacy(conn, document_id)?
        .ok_or_else(|| MemoryError::DocumentNotFound(document_id.to_string()))?;

    update_episode_outcome_by_id(
        conn,
        &episode_id,
        outcome,
        confidence,
        experiment_id,
        verification_status,
        search_text,
        embedding_bytes,
        q8_bytes,
    )
}

/// Update the outcome of an episode by its episode_id (canonical path).
#[allow(clippy::too_many_arguments)]
pub(crate) fn update_episode_outcome_by_id(
    conn: &Connection,
    episode_id: &str,
    outcome: EpisodeOutcome,
    confidence: f32,
    experiment_id: Option<&str>,
    verification_status: &VerificationStatus,
    search_text: &str,
    embedding_bytes: &[u8],
    q8_bytes: Option<&[u8]>,
) -> Result<(), MemoryError> {
    let verification_json = serde_json::to_string(verification_status)
        .map_err(|e| MemoryError::Other(e.to_string()))?;
    #[cfg(feature = "hnsw")]
    let item_key = episode_item_key(episode_id);

    db::with_transaction(conn, |tx| {
        let old_search_text: String = tx
            .query_row(
                "SELECT search_text FROM episodes WHERE episode_id = ?1",
                params![episode_id],
                |row| row.get(0),
            )
            .map_err(|_| MemoryError::EpisodeNotFound(episode_id.to_string()))?;
        let fts_rowid: i64 = tx.query_row(
            "SELECT rowid FROM episodes_rowid_map WHERE episode_id = ?1",
            params![episode_id],
            |row| row.get(0),
        )?;

        tx.execute(
            "INSERT INTO episodes_fts (episodes_fts, rowid, content) VALUES ('delete', ?1, ?2)",
            params![fts_rowid, old_search_text],
        )?;
        tx.execute(
            "UPDATE episodes
             SET outcome = ?1,
                 confidence = ?2,
                 experiment_id = COALESCE(?3, experiment_id),
                 verification_status = ?4,
                 search_text = ?5,
                 embedding = ?6,
                 embedding_q8 = ?7,
                 updated_at = datetime('now')
             WHERE episode_id = ?8",
            params![
                outcome.as_str(),
                confidence,
                experiment_id,
                verification_json,
                search_text,
                embedding_bytes,
                q8_bytes,
                episode_id
            ],
        )?;
        tx.execute(
            "INSERT INTO episodes_fts (rowid, content) VALUES (?1, ?2)",
            params![fts_rowid, search_text],
        )?;

        #[cfg(feature = "hnsw")]
        db::queue_pending_index_op(tx, &item_key, "episode", IndexOpKind::Upsert)?;
        Ok(())
    })
}

pub(crate) fn search_episodes(
    conn: &Connection,
    effect_type: Option<&str>,
    outcome: Option<&EpisodeOutcome>,
    limit: usize,
) -> Result<Vec<(String, EpisodeMeta)>, MemoryError> {
    let effect_type = effect_type.map(ToOwned::to_owned);
    let outcome = outcome.map(|value| value.as_str().to_string());

    let mut sql = String::from(
        "SELECT episode_id, document_id, cause_ids, effect_type, outcome, confidence, verification_status, experiment_id
         FROM episodes
         WHERE 1 = 1",
    );
    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

    if let Some(effect_type) = &effect_type {
        sql.push_str(&format!(" AND effect_type = ?{}", params.len() + 1));
        params.push(Box::new(effect_type.clone()));
    }
    if let Some(outcome) = &outcome {
        sql.push_str(&format!(" AND outcome = ?{}", params.len() + 1));
        params.push(Box::new(outcome.clone()));
    }
    sql.push_str(&format!(" ORDER BY updated_at DESC LIMIT {}", limit));

    let param_refs: Vec<&dyn rusqlite::types::ToSql> =
        params.iter().map(|value| value.as_ref()).collect();
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt
        .query_map(&*param_refs, |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, f32>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, Option<String>>(7)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;

    rows.into_iter()
        .map(
            |(
                _episode_id,
                document_id,
                cause_ids_raw,
                effect_type,
                outcome_raw,
                confidence,
                verification_status_raw,
                experiment_id,
            )| {
                Ok((
                    document_id.clone(),
                    EpisodeMeta {
                        cause_ids: db::parse_string_list_json(
                            "episodes",
                            &document_id,
                            "cause_ids",
                            &cause_ids_raw,
                        )?,
                        effect_type,
                        outcome: db::parse_episode_outcome(&document_id, &outcome_raw)?,
                        confidence,
                        verification_status: db::parse_verification_status(
                            &document_id,
                            &verification_status_raw,
                        )?,
                        experiment_id,
                    },
                ))
            },
        )
        .collect()
}

/// Get episode by episode_id.
pub(crate) fn get_episode(
    conn: &Connection,
    episode_id: &str,
) -> Result<Option<(String, EpisodeMeta)>, MemoryError> {
    let row = conn.query_row(
        "SELECT document_id, cause_ids, effect_type, outcome, confidence, verification_status, experiment_id
         FROM episodes
         WHERE episode_id = ?1",
        params![episode_id],
        |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, f32>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, Option<String>>(6)?,
            ))
        },
    );

    match row {
        Ok((
            document_id,
            cause_ids_raw,
            effect_type,
            outcome_raw,
            confidence,
            verification_status_raw,
            experiment_id,
        )) => Ok(Some((
            document_id.clone(),
            EpisodeMeta {
                cause_ids: db::parse_string_list_json(
                    "episodes",
                    &document_id,
                    "cause_ids",
                    &cause_ids_raw,
                )?,
                effect_type,
                outcome: db::parse_episode_outcome(&document_id, &outcome_raw)?,
                confidence,
                verification_status: db::parse_verification_status(
                    &document_id,
                    &verification_status_raw,
                )?,
                experiment_id,
            },
        ))),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(err) => Err(MemoryError::Database(err)),
    }
}

/// Legacy compatibility: load the primary episode's metadata for a document.
/// Returns the first-created episode's metadata, or None if no episodes exist.
pub(crate) fn load_episode_meta(
    conn: &Connection,
    document_id: &str,
) -> Result<Option<EpisodeMeta>, MemoryError> {
    let row = conn.query_row(
        "SELECT cause_ids, effect_type, outcome, confidence, verification_status, experiment_id
         FROM episodes
         WHERE document_id = ?1
         ORDER BY created_at ASC
         LIMIT 1",
        params![document_id],
        |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, f32>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, Option<String>>(5)?,
            ))
        },
    );

    match row {
        Ok((
            cause_ids_raw,
            effect_type,
            outcome_raw,
            confidence,
            verification_status_raw,
            experiment_id,
        )) => Ok(Some(EpisodeMeta {
            cause_ids: db::parse_string_list_json(
                "episodes",
                document_id,
                "cause_ids",
                &cause_ids_raw,
            )?,
            effect_type,
            outcome: db::parse_episode_outcome(document_id, &outcome_raw)?,
            confidence,
            verification_status: db::parse_verification_status(
                document_id,
                &verification_status_raw,
            )?,
            experiment_id,
        })),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(err) => Err(MemoryError::Database(err)),
    }
}

/// Legacy compatibility: load the first episode_id for a document.
/// Delegates to `resolve_primary_episode_id_legacy`.
#[allow(dead_code)]
pub(crate) fn load_primary_episode_id_legacy(
    conn: &Connection,
    document_id: &str,
) -> Result<Option<String>, MemoryError> {
    resolve_primary_episode_id_legacy(conn, document_id)
}

pub(crate) fn load_episode_context(
    conn: &Connection,
    document_id: &str,
) -> Result<(String, String), MemoryError> {
    let title: String = conn
        .query_row(
            "SELECT title FROM documents WHERE id = ?1",
            params![document_id],
            |row| row.get(0),
        )
        .map_err(|_| MemoryError::DocumentNotFound(document_id.to_string()))?;

    let mut stmt =
        conn.prepare("SELECT content FROM chunks WHERE document_id = ?1 ORDER BY chunk_index ASC")?;
    let chunks = stmt
        .query_map(params![document_id], |row| row.get::<_, String>(0))?
        .collect::<Result<Vec<_>, _>>()?;

    Ok((title, chunks.join("\n")))
}
