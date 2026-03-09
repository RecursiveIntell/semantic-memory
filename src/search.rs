//! Hybrid search engine: BM25 + vector similarity + Reciprocal Rank Fusion.

use crate::config::SearchConfig;
use crate::episodes;
use crate::error::MemoryError;
use crate::types::{ExplainedResult, ScoreBreakdown, SearchResult, SearchSource, SearchSourceType};
use rusqlite::types::Value as SqlValue;
use rusqlite::Connection;
use std::collections::{HashMap, HashSet};

/// Per-table row count above which vector search emits a warning.
const VECTOR_SCAN_WARN_THRESHOLD: usize = 50_000;

/// Sanitize a raw query string for FTS5 MATCH syntax.
///
/// Strips FTS5 operators, splits on whitespace, and returns `None` if nothing remains.
pub fn sanitize_fts_query(raw: &str) -> Option<String> {
    let cleaned: String = raw
        .chars()
        .map(|c| {
            if matches!(
                c,
                '"' | '*' | '+' | '-' | '(' | ')' | '^' | '{' | '}' | '~' | ':'
            ) {
                ' '
            } else {
                c
            }
        })
        .collect();

    let tokens: Vec<&str> = cleaned
        .split_whitespace()
        .filter(|t| !matches!(t.to_uppercase().as_str(), "AND" | "OR" | "NOT" | "NEAR"))
        .collect();

    if tokens.is_empty() {
        None
    } else {
        Some(tokens.join(" OR "))
    }
}

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "embedding dimension mismatch");
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn days_since(iso_timestamp: &str) -> Option<f64> {
    let dt = chrono::NaiveDateTime::parse_from_str(iso_timestamp, "%Y-%m-%d %H:%M:%S").ok()?;
    let now = chrono::Utc::now().naive_utc();
    let duration = now - dt;
    Some(duration.num_seconds() as f64 / 86_400.0)
}

fn recency_contribution(config: &SearchConfig, updated_at: Option<&str>) -> Option<f64> {
    match (config.recency_half_life_days, updated_at) {
        (Some(half_life), Some(ts)) if half_life > 0.0 => {
            let age_days = days_since(ts).unwrap_or(0.0).max(0.0);
            let decay = 2.0_f64.powf(-age_days / half_life);
            Some(config.recency_weight * decay / (config.rrf_k + 1.0))
        }
        _ => None,
    }
}

fn source_dedup_key(source: &SearchSource) -> (u8, String) {
    match source {
        SearchSource::Fact { fact_id, .. } => (0, fact_id.clone()),
        SearchSource::Chunk { chunk_id, .. } => (1, chunk_id.clone()),
        SearchSource::Message { message_id, .. } => (2, message_id.to_string()),
        SearchSource::Episode { episode_id, .. } => (3, episode_id.clone()),
        SearchSource::Projection { projection_id, .. } => (4, projection_id.clone()),
    }
}

/// A BM25 search hit from FTS5.
#[derive(Debug, Clone)]
pub struct Bm25Hit {
    /// Search item key such as `fact:{uuid}` or `episode:{episode_id}`.
    pub id: String,
    /// Text content returned to callers.
    pub content: String,
    /// Source info.
    pub source: SearchSource,
    /// Raw BM25 score reported by SQLite FTS5.
    pub raw_score: f64,
    /// Timestamp used for recency scoring.
    pub updated_at: Option<String>,
}

/// A vector search hit.
#[derive(Debug, Clone)]
pub struct VectorHit {
    /// Search item key such as `fact:{uuid}` or `episode:{episode_id}`.
    pub id: String,
    /// Text content returned to callers.
    pub content: String,
    /// Source info.
    pub source: SearchSource,
    /// Final similarity used for vector ranking.
    pub similarity: f64,
    /// Timestamp used for recency scoring.
    pub updated_at: Option<String>,
    /// Rank from the underlying retrieval stage before exact reranking.
    pub source_rank: Option<usize>,
    /// Similarity from the underlying retrieval stage before exact reranking.
    pub source_similarity: Option<f64>,
    /// Whether exact f32 reranking changed or confirmed this candidate ordering.
    pub reranked_from_f32: bool,
}

struct VectorRow {
    id: String,
    content: String,
    blob: Vec<u8>,
    updated_at: Option<String>,
    source: SearchSource,
}

struct RrfCandidate {
    content: String,
    source: SearchSource,
    updated_at: Option<String>,
    bm25_score: Option<f64>,
    bm25_rank: Option<usize>,
    vector_score: Option<f64>,
    vector_rank: Option<usize>,
    vector_source_rank: Option<usize>,
    vector_source_score: Option<f64>,
    vector_reranked_from_f32: bool,
}

impl RrfCandidate {
    fn explained(self, config: &SearchConfig) -> ExplainedResult {
        let bm25_contribution = self
            .bm25_rank
            .map(|rank| config.bm25_weight / (config.rrf_k + rank as f64));
        let vector_contribution = self
            .vector_rank
            .map(|rank| config.vector_weight / (config.rrf_k + rank as f64));
        let recency_score = recency_contribution(config, self.updated_at.as_deref());
        let rrf_score = bm25_contribution.unwrap_or(0.0)
            + vector_contribution.unwrap_or(0.0)
            + recency_score.unwrap_or(0.0);

        let breakdown = ScoreBreakdown {
            rrf_score,
            bm25_score: self.bm25_score,
            vector_score: self.vector_score,
            recency_score,
            bm25_rank: self.bm25_rank,
            vector_rank: self.vector_rank,
            vector_source_rank: self.vector_source_rank,
            vector_source_score: self.vector_source_score,
            bm25_contribution,
            vector_contribution,
            vector_reranked_from_f32: self.vector_reranked_from_f32,
            bm25_weight: config.bm25_weight,
            vector_weight: config.vector_weight,
            recency_weight: config.recency_half_life_days.map(|_| config.recency_weight),
            rrf_k: config.rrf_k,
        };

        ExplainedResult {
            result: SearchResult {
                content: self.content,
                source: self.source,
                score: rrf_score,
                bm25_rank: breakdown.bm25_rank,
                vector_rank: breakdown.vector_rank,
                cosine_similarity: breakdown.vector_score,
            },
            breakdown,
        }
    }
}

fn scan_vector_rows(
    rows: impl Iterator<Item = Result<VectorRow, rusqlite::Error>>,
    query_embedding: &[f32],
    min_similarity: f64,
    table_label: &str,
) -> Result<(Vec<VectorHit>, usize), MemoryError> {
    let expected_dims = query_embedding.len();
    let mut hits = Vec::new();
    let mut row_count = 0usize;

    for row in rows {
        let row = row?;
        row_count += 1;

        if row.blob.len() % 4 != 0 {
            tracing::warn!(
                "Skipping {} with invalid embedding length: {}",
                table_label,
                row.blob.len()
            );
            continue;
        }

        let stored_embedding = bytemuck::try_cast_slice::<u8, f32>(&row.blob).map_err(|_| {
            MemoryError::InvalidEmbedding {
                expected_bytes: row.blob.len() - (row.blob.len() % 4),
                actual_bytes: row.blob.len(),
            }
        })?;

        if stored_embedding.len() != expected_dims {
            tracing::warn!(
                expected = expected_dims,
                actual = stored_embedding.len(),
                "Skipping {} with wrong embedding dimensions",
                table_label
            );
            continue;
        }

        let similarity = cosine_similarity(query_embedding, stored_embedding) as f64;
        if similarity >= min_similarity {
            hits.push(VectorHit {
                id: row.id,
                content: row.content,
                source: row.source,
                similarity,
                updated_at: row.updated_at,
                source_rank: None,
                source_similarity: None,
                reranked_from_f32: false,
            });
        }
    }

    Ok((hits, row_count))
}

fn rank_vector_hits(mut hits: Vec<VectorHit>, pool_size: usize) -> Vec<VectorHit> {
    hits.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (idx, hit) in hits.iter_mut().enumerate() {
        hit.source_rank = Some(idx + 1);
        hit.source_similarity = Some(hit.similarity);
    }

    hits.truncate(pool_size);
    hits
}

/// Run BM25 search over facts_fts, chunks_fts, episodes_fts, and optionally messages_fts.
pub(crate) fn bm25_search(
    conn: &Connection,
    sanitized_query: &str,
    pool_size: usize,
    namespaces: Option<&[&str]>,
    source_types: Option<&[SearchSourceType]>,
    session_ids: Option<&[&str]>,
) -> Result<Vec<Bm25Hit>, MemoryError> {
    let mut hits = Vec::new();

    let search_facts = source_types
        .map(|st| st.contains(&SearchSourceType::Facts))
        .unwrap_or(true);
    let search_chunks = source_types
        .map(|st| st.contains(&SearchSourceType::Chunks))
        .unwrap_or(true);
    let search_messages = source_types
        .map(|st| st.contains(&SearchSourceType::Messages))
        .unwrap_or(false);
    let search_episodes = source_types
        .map(|st| st.contains(&SearchSourceType::Episodes))
        .unwrap_or(true);

    if search_facts {
        let (ns_clause, ns_params) = build_filter_clause("f.namespace", namespaces, 3);
        let sql = format!(
            "SELECT fm.fact_id, f.content, f.namespace, bm25(facts_fts) AS score, f.updated_at
             FROM facts_fts
             JOIN facts_rowid_map fm ON facts_fts.rowid = fm.rowid
             JOIN facts f ON f.id = fm.fact_id
             WHERE facts_fts MATCH ?1 {}
             ORDER BY score ASC
             LIMIT ?2",
            ns_clause
        );

        let mut params = vec![
            SqlValue::Text(sanitized_query.to_string()),
            SqlValue::Integer(pool_size as i64),
        ];
        params.extend(ns_params);

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(&params), |row| {
            let fact_id: String = row.get(0)?;
            let content: String = row.get(1)?;
            let namespace: String = row.get(2)?;
            let raw_score: f64 = row.get(3)?;
            let updated_at: Option<String> = row.get(4)?;
            Ok(Bm25Hit {
                id: format!("fact:{fact_id}"),
                content,
                source: SearchSource::Fact { fact_id, namespace },
                raw_score,
                updated_at,
            })
        })?;

        for row in rows {
            hits.push(row?);
        }
    }

    if search_chunks {
        let (ns_clause, ns_params) = build_filter_clause("d.namespace", namespaces, 3);
        let sql = format!(
            "SELECT cm.chunk_id, c.content, c.document_id, d.title, c.chunk_index,
                    bm25(chunks_fts) AS score, c.created_at
             FROM chunks_fts
             JOIN chunks_rowid_map cm ON chunks_fts.rowid = cm.rowid
             JOIN chunks c ON c.id = cm.chunk_id
             JOIN documents d ON d.id = c.document_id
             WHERE chunks_fts MATCH ?1 {}
             ORDER BY score ASC
             LIMIT ?2",
            ns_clause
        );

        let mut params = vec![
            SqlValue::Text(sanitized_query.to_string()),
            SqlValue::Integer(pool_size as i64),
        ];
        params.extend(ns_params);

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(&params), |row| {
            let chunk_id: String = row.get(0)?;
            let content: String = row.get(1)?;
            let document_id: String = row.get(2)?;
            let document_title: String = row.get(3)?;
            let chunk_index: i64 = row.get(4)?;
            let raw_score: f64 = row.get(5)?;
            let updated_at: Option<String> = row.get(6)?;
            Ok(Bm25Hit {
                id: format!("chunk:{chunk_id}"),
                content,
                source: SearchSource::Chunk {
                    chunk_id,
                    document_id,
                    document_title,
                    chunk_index: chunk_index as usize,
                },
                raw_score,
                updated_at,
            })
        })?;

        for row in rows {
            hits.push(row?);
        }
    }

    if search_messages {
        let (sid_clause, sid_params) = build_filter_clause("m.session_id", session_ids, 3);
        let sql = format!(
            "SELECT mm.message_id, m.content, m.session_id, m.role,
                    bm25(messages_fts) AS score, m.created_at
             FROM messages_fts
             JOIN messages_rowid_map mm ON messages_fts.rowid = mm.rowid
             JOIN messages m ON m.id = mm.message_id
             WHERE messages_fts MATCH ?1 {}
             ORDER BY score ASC
             LIMIT ?2",
            sid_clause
        );

        let mut params = vec![
            SqlValue::Text(sanitized_query.to_string()),
            SqlValue::Integer(pool_size as i64),
        ];
        params.extend(sid_params);

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(&params), |row| {
            let message_id: i64 = row.get(0)?;
            let content: String = row.get(1)?;
            let session_id: String = row.get(2)?;
            let role: String = row.get(3)?;
            let raw_score: f64 = row.get(4)?;
            let updated_at: Option<String> = row.get(5)?;
            Ok(Bm25Hit {
                id: format!("msg:{message_id}"),
                content,
                source: SearchSource::Message {
                    message_id,
                    session_id,
                    role,
                },
                raw_score,
                updated_at,
            })
        })?;

        for row in rows {
            hits.push(row?);
        }
    }

    if search_episodes {
        let (ns_clause, ns_params) = build_filter_clause("d.namespace", namespaces, 3);
        let sql = format!(
            "SELECT e.episode_id, e.document_id, e.search_text, e.effect_type, e.outcome,
                    bm25(episodes_fts) AS score, e.updated_at
             FROM episodes_fts
             JOIN episodes_rowid_map rm ON episodes_fts.rowid = rm.rowid
             JOIN episodes e ON e.episode_id = rm.episode_id
             JOIN documents d ON d.id = e.document_id
             WHERE episodes_fts MATCH ?1 {}
             ORDER BY score ASC
             LIMIT ?2",
            ns_clause
        );

        let mut params = vec![
            SqlValue::Text(sanitized_query.to_string()),
            SqlValue::Integer(pool_size as i64),
        ];
        params.extend(ns_params);

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(&params), |row| {
            let episode_id: String = row.get(0)?;
            let document_id: String = row.get(1)?;
            let content: String = row.get(2)?;
            let effect_type: String = row.get(3)?;
            let outcome: String = row.get(4)?;
            let raw_score: f64 = row.get(5)?;
            let updated_at: Option<String> = row.get(6)?;
            Ok(Bm25Hit {
                id: episodes::episode_item_key(&episode_id),
                content,
                source: SearchSource::Episode {
                    episode_id,
                    document_id,
                    effect_type,
                    outcome,
                },
                raw_score,
                updated_at,
            })
        })?;

        for row in rows {
            hits.push(row?);
        }
    }

    Ok(hits)
}

/// Run brute-force vector search over facts, chunks, episodes, and optionally messages.
pub(crate) fn vector_search(
    conn: &Connection,
    query_embedding: &[f32],
    pool_size: usize,
    min_similarity: f64,
    namespaces: Option<&[&str]>,
    source_types: Option<&[SearchSourceType]>,
    session_ids: Option<&[&str]>,
) -> Result<Vec<VectorHit>, MemoryError> {
    let mut hits = Vec::new();

    let search_facts = source_types
        .map(|st| st.contains(&SearchSourceType::Facts))
        .unwrap_or(true);
    let search_chunks = source_types
        .map(|st| st.contains(&SearchSourceType::Chunks))
        .unwrap_or(true);
    let search_messages = source_types
        .map(|st| st.contains(&SearchSourceType::Messages))
        .unwrap_or(false);
    let search_episodes = source_types
        .map(|st| st.contains(&SearchSourceType::Episodes))
        .unwrap_or(true);

    if search_facts {
        let (ns_clause, ns_params) = build_filter_clause("namespace", namespaces, 1);
        let sql = format!(
            "SELECT id, content, namespace, embedding, updated_at
             FROM facts
             WHERE embedding IS NOT NULL {}",
            ns_clause
        );

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(&ns_params), |row| {
            let id: String = row.get(0)?;
            let content: String = row.get(1)?;
            let namespace: String = row.get(2)?;
            let blob: Vec<u8> = row.get(3)?;
            let updated_at: Option<String> = row.get(4)?;
            Ok(VectorRow {
                id: format!("fact:{id}"),
                content,
                blob,
                updated_at,
                source: SearchSource::Fact {
                    fact_id: id,
                    namespace,
                },
            })
        })?;

        let (fact_hits, fact_count) =
            scan_vector_rows(rows, query_embedding, min_similarity, "fact")?;
        hits.extend(fact_hits);

        if fact_count > VECTOR_SCAN_WARN_THRESHOLD {
            tracing::warn!(
                count = fact_count,
                "facts table exceeds vector scan threshold ({} rows)",
                fact_count
            );
        }
    }

    if search_chunks {
        let (ns_clause, ns_params) = build_filter_clause("d.namespace", namespaces, 1);
        let sql = format!(
            "SELECT c.id, c.content, c.document_id, d.title, c.chunk_index, c.embedding, c.created_at
             FROM chunks c
             JOIN documents d ON d.id = c.document_id
             WHERE c.embedding IS NOT NULL {}",
            ns_clause
        );

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(&ns_params), |row| {
            let id: String = row.get(0)?;
            let content: String = row.get(1)?;
            let document_id: String = row.get(2)?;
            let document_title: String = row.get(3)?;
            let chunk_index: i64 = row.get(4)?;
            let blob: Vec<u8> = row.get(5)?;
            let updated_at: Option<String> = row.get(6)?;
            Ok(VectorRow {
                id: format!("chunk:{id}"),
                content,
                blob,
                updated_at,
                source: SearchSource::Chunk {
                    chunk_id: id,
                    document_id,
                    document_title,
                    chunk_index: chunk_index as usize,
                },
            })
        })?;

        let (chunk_hits, chunk_count) =
            scan_vector_rows(rows, query_embedding, min_similarity, "chunk")?;
        hits.extend(chunk_hits);

        if chunk_count > VECTOR_SCAN_WARN_THRESHOLD {
            tracing::warn!(
                count = chunk_count,
                "chunks table exceeds vector scan threshold ({} rows)",
                chunk_count
            );
        }
    }

    if search_messages {
        let (sid_clause, sid_params) = build_filter_clause("m.session_id", session_ids, 1);
        let sql = format!(
            "SELECT m.id, m.content, m.session_id, m.role, m.embedding, m.created_at
             FROM messages m
             WHERE m.embedding IS NOT NULL {}",
            sid_clause
        );

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(&sid_params), |row| {
            let message_id: i64 = row.get(0)?;
            let content: String = row.get(1)?;
            let session_id: String = row.get(2)?;
            let role: String = row.get(3)?;
            let blob: Vec<u8> = row.get(4)?;
            let updated_at: Option<String> = row.get(5)?;
            Ok(VectorRow {
                id: format!("msg:{message_id}"),
                content,
                blob,
                updated_at,
                source: SearchSource::Message {
                    message_id,
                    session_id,
                    role,
                },
            })
        })?;

        let (message_hits, message_count) =
            scan_vector_rows(rows, query_embedding, min_similarity, "message")?;
        hits.extend(message_hits);

        if message_count > VECTOR_SCAN_WARN_THRESHOLD {
            tracing::warn!(
                count = message_count,
                "messages table exceeds vector scan threshold ({} rows)",
                message_count
            );
        }
    }

    if search_episodes {
        let (ns_clause, ns_params) = build_filter_clause("d.namespace", namespaces, 1);
        let sql = format!(
            "SELECT e.episode_id, e.document_id, e.search_text, e.effect_type, e.outcome, e.embedding, e.updated_at
             FROM episodes e
             JOIN documents d ON d.id = e.document_id
             WHERE e.embedding IS NOT NULL {}",
            ns_clause
        );

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(&ns_params), |row| {
            let episode_id: String = row.get(0)?;
            let document_id: String = row.get(1)?;
            let content: String = row.get(2)?;
            let effect_type: String = row.get(3)?;
            let outcome: String = row.get(4)?;
            let blob: Vec<u8> = row.get(5)?;
            let updated_at: Option<String> = row.get(6)?;
            Ok(VectorRow {
                id: episodes::episode_item_key(&episode_id),
                content,
                blob,
                updated_at,
                source: SearchSource::Episode {
                    episode_id,
                    document_id,
                    effect_type,
                    outcome,
                },
            })
        })?;

        let (episode_hits, episode_count) =
            scan_vector_rows(rows, query_embedding, min_similarity, "episode")?;
        hits.extend(episode_hits);

        if episode_count > VECTOR_SCAN_WARN_THRESHOLD {
            tracing::warn!(
                count = episode_count,
                "episodes table exceeds vector scan threshold ({} rows)",
                episode_count
            );
        }
    }

    Ok(rank_vector_hits(hits, pool_size))
}

fn rrf_fuse_detailed(
    bm25_hits: &[Bm25Hit],
    vector_hits: &[VectorHit],
    config: &SearchConfig,
    top_k: usize,
) -> Vec<ExplainedResult> {
    let mut candidates: HashMap<(u8, String), RrfCandidate> = HashMap::new();

    for (rank_0, hit) in bm25_hits.iter().enumerate() {
        let key = source_dedup_key(&hit.source);
        let rank = rank_0 + 1;
        candidates
            .entry(key)
            .and_modify(|candidate| {
                candidate.bm25_rank = Some(rank);
                candidate.bm25_score = Some(hit.raw_score);
                if candidate.updated_at.is_none() {
                    candidate.updated_at = hit.updated_at.clone();
                }
            })
            .or_insert_with(|| RrfCandidate {
                content: hit.content.clone(),
                source: hit.source.clone(),
                updated_at: hit.updated_at.clone(),
                bm25_score: Some(hit.raw_score),
                bm25_rank: Some(rank),
                vector_score: None,
                vector_rank: None,
                vector_source_rank: None,
                vector_source_score: None,
                vector_reranked_from_f32: false,
            });
    }

    for (rank_0, hit) in vector_hits.iter().enumerate() {
        let key = source_dedup_key(&hit.source);
        let rank = rank_0 + 1;
        candidates
            .entry(key)
            .and_modify(|candidate| {
                candidate.vector_rank = Some(rank);
                candidate.vector_score = Some(hit.similarity);
                candidate.vector_source_rank = hit.source_rank.or(Some(rank));
                candidate.vector_source_score = hit.source_similarity.or(Some(hit.similarity));
                candidate.vector_reranked_from_f32 = hit.reranked_from_f32;
                if candidate.updated_at.is_none() {
                    candidate.updated_at = hit.updated_at.clone();
                }
            })
            .or_insert_with(|| RrfCandidate {
                content: hit.content.clone(),
                source: hit.source.clone(),
                updated_at: hit.updated_at.clone(),
                bm25_score: None,
                bm25_rank: None,
                vector_score: Some(hit.similarity),
                vector_rank: Some(rank),
                vector_source_rank: hit.source_rank.or(Some(rank)),
                vector_source_score: hit.source_similarity.or(Some(hit.similarity)),
                vector_reranked_from_f32: hit.reranked_from_f32,
            });
    }

    let mut explained: Vec<ExplainedResult> = candidates
        .into_values()
        .map(|candidate| candidate.explained(config))
        .collect();

    explained.sort_by(|a, b| {
        b.result
            .score
            .partial_cmp(&a.result.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                source_dedup_key(&a.result.source).cmp(&source_dedup_key(&b.result.source))
            })
    });
    explained.truncate(top_k);
    explained
}

/// Fuse BM25 and vector results via Reciprocal Rank Fusion.
pub fn rrf_fuse(
    bm25_hits: &[Bm25Hit],
    vector_hits: &[VectorHit],
    config: &SearchConfig,
    top_k: usize,
) -> Vec<SearchResult> {
    rrf_fuse_detailed(bm25_hits, vector_hits, config, top_k)
        .into_iter()
        .map(|result| result.result)
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn hybrid_search_detailed(
    conn: &Connection,
    query: &str,
    query_embedding: &[f32],
    config: &SearchConfig,
    top_k: usize,
    namespaces: Option<&[&str]>,
    source_types: Option<&[SearchSourceType]>,
    session_ids: Option<&[&str]>,
) -> Result<Vec<ExplainedResult>, MemoryError> {
    let bm25_hits = match sanitize_fts_query(query) {
        Some(sanitized) => bm25_search(
            conn,
            &sanitized,
            config.candidate_pool_size,
            namespaces,
            source_types,
            session_ids,
        )?,
        None => Vec::new(),
    };

    let vector_hits = vector_search(
        conn,
        query_embedding,
        config.candidate_pool_size,
        config.min_similarity,
        namespaces,
        source_types,
        session_ids,
    )?;

    Ok(rrf_fuse_detailed(&bm25_hits, &vector_hits, config, top_k))
}

/// Perform a hybrid search and return the exact score decomposition.
#[allow(clippy::too_many_arguments)]
pub fn hybrid_search_explained(
    conn: &Connection,
    query: &str,
    query_embedding: &[f32],
    config: &SearchConfig,
    top_k: usize,
    namespaces: Option<&[&str]>,
    source_types: Option<&[SearchSourceType]>,
    session_ids: Option<&[&str]>,
) -> Result<Vec<ExplainedResult>, MemoryError> {
    hybrid_search_detailed(
        conn,
        query,
        query_embedding,
        config,
        top_k,
        namespaces,
        source_types,
        session_ids,
    )
}

/// Perform a hybrid search (BM25 + vector + RRF).
#[allow(clippy::too_many_arguments)]
pub fn hybrid_search(
    conn: &Connection,
    query: &str,
    query_embedding: &[f32],
    config: &SearchConfig,
    top_k: usize,
    namespaces: Option<&[&str]>,
    source_types: Option<&[SearchSourceType]>,
    session_ids: Option<&[&str]>,
) -> Result<Vec<SearchResult>, MemoryError> {
    Ok(hybrid_search_detailed(
        conn,
        query,
        query_embedding,
        config,
        top_k,
        namespaces,
        source_types,
        session_ids,
    )?
    .into_iter()
    .map(|result| result.result)
    .collect())
}

#[cfg(feature = "hnsw")]
#[derive(Clone)]
struct HnswCandidateSeed {
    source_rank: usize,
    source_similarity: f64,
}

#[cfg(feature = "hnsw")]
#[allow(clippy::type_complexity)]
fn resolve_hnsw_hits_batched(
    conn: &Connection,
    query_embedding: &[f32],
    config: &SearchConfig,
    namespaces: Option<&[&str]>,
    source_types: Option<&[SearchSourceType]>,
    session_ids: Option<&[&str]>,
    hnsw_hits: &[crate::hnsw::HnswHit],
) -> Result<Vec<VectorHit>, MemoryError> {
    let search_facts = source_types
        .map(|st| st.contains(&SearchSourceType::Facts))
        .unwrap_or(true);
    let search_chunks = source_types
        .map(|st| st.contains(&SearchSourceType::Chunks))
        .unwrap_or(true);
    let search_messages = source_types
        .map(|st| st.contains(&SearchSourceType::Messages))
        .unwrap_or(false);
    let search_episodes = source_types
        .map(|st| st.contains(&SearchSourceType::Episodes))
        .unwrap_or(true);

    let mut fact_entries: HashMap<String, HnswCandidateSeed> = HashMap::new();
    let mut chunk_entries: HashMap<String, HnswCandidateSeed> = HashMap::new();
    let mut message_entries: HashMap<i64, HnswCandidateSeed> = HashMap::new();
    let mut episode_entries: HashMap<String, HnswCandidateSeed> = HashMap::new();

    for (rank_0, hit) in hnsw_hits.iter().enumerate() {
        let similarity = hit.similarity() as f64;
        if similarity < config.min_similarity {
            continue;
        }

        let (domain, raw_id) = hit.parse_key()?;
        let seed = HnswCandidateSeed {
            source_rank: rank_0 + 1,
            source_similarity: similarity,
        };

        match domain {
            "fact" if search_facts => {
                fact_entries.entry(raw_id.to_string()).or_insert(seed);
            }
            "chunk" if search_chunks => {
                chunk_entries.entry(raw_id.to_string()).or_insert(seed);
            }
            "msg" if search_messages => {
                if let Ok(message_id) = raw_id.parse::<i64>() {
                    message_entries.entry(message_id).or_insert(seed);
                }
            }
            "episode" if search_episodes => {
                episode_entries.entry(raw_id.to_string()).or_insert(seed);
            }
            _ => {}
        }
    }

    let mut hits = Vec::new();
    batch_load_fact_hits(
        conn,
        query_embedding,
        config,
        namespaces,
        &fact_entries,
        &mut hits,
    )?;
    batch_load_chunk_hits(
        conn,
        query_embedding,
        config,
        namespaces,
        &chunk_entries,
        &mut hits,
    )?;
    batch_load_message_hits(
        conn,
        query_embedding,
        config,
        session_ids,
        &message_entries,
        &mut hits,
    )?;
    batch_load_episode_hits(
        conn,
        query_embedding,
        config,
        namespaces,
        &episode_entries,
        &mut hits,
    )?;

    hits.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.source_rank
                    .unwrap_or(usize::MAX)
                    .cmp(&b.source_rank.unwrap_or(usize::MAX))
            })
    });
    hits.truncate(config.candidate_pool_size);
    Ok(hits)
}

#[cfg(feature = "hnsw")]
fn exact_similarity_from_blob(
    query_embedding: &[f32],
    blob: &[u8],
) -> Result<Option<f64>, MemoryError> {
    if blob.is_empty() {
        return Ok(None);
    }
    let stored = crate::db::bytes_to_embedding(blob)?;
    if stored.len() != query_embedding.len() {
        return Ok(None);
    }
    Ok(Some(cosine_similarity(query_embedding, &stored) as f64))
}

#[cfg(feature = "hnsw")]
#[allow(clippy::too_many_arguments)]
fn build_ranked_vector_hit(
    id: String,
    content: String,
    source: SearchSource,
    updated_at: Option<String>,
    embedding_blob: Option<Vec<u8>>,
    seed: &HnswCandidateSeed,
    query_embedding: &[f32],
    config: &SearchConfig,
) -> Result<Option<VectorHit>, MemoryError> {
    let similarity = if config.rerank_from_f32 {
        match embedding_blob {
            Some(blob) => exact_similarity_from_blob(query_embedding, &blob)?,
            None => None,
        }
        .unwrap_or(seed.source_similarity)
    } else {
        seed.source_similarity
    };

    if similarity < config.min_similarity {
        return Ok(None);
    }

    Ok(Some(VectorHit {
        id,
        content,
        source,
        similarity,
        updated_at,
        source_rank: Some(seed.source_rank),
        source_similarity: Some(seed.source_similarity),
        reranked_from_f32: config.rerank_from_f32,
    }))
}

#[cfg(feature = "hnsw")]
fn batch_load_fact_hits(
    conn: &Connection,
    query_embedding: &[f32],
    config: &SearchConfig,
    namespaces: Option<&[&str]>,
    entries: &HashMap<String, HnswCandidateSeed>,
    output: &mut Vec<VectorHit>,
) -> Result<(), MemoryError> {
    if entries.is_empty() {
        return Ok(());
    }

    let placeholders = (1..=entries.len())
        .map(|idx| format!("?{idx}"))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT id, content, namespace, updated_at, embedding
         FROM facts
         WHERE id IN ({placeholders})"
    );
    let params: Vec<SqlValue> = entries
        .keys()
        .map(|id| SqlValue::Text(id.clone()))
        .collect();
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(&params), |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, Option<String>>(3)?,
            row.get::<_, Option<Vec<u8>>>(4)?,
        ))
    })?;

    for row in rows {
        let (fact_id, content, namespace, updated_at, embedding_blob) = row?;
        if let Some(filter) = namespaces {
            if !filter.contains(&namespace.as_str()) {
                continue;
            }
        }
        if let Some(seed) = entries.get(&fact_id) {
            if let Some(hit) = build_ranked_vector_hit(
                format!("fact:{fact_id}"),
                content,
                SearchSource::Fact { fact_id, namespace },
                updated_at,
                embedding_blob,
                seed,
                query_embedding,
                config,
            )? {
                output.push(hit);
            }
        }
    }

    Ok(())
}

#[cfg(feature = "hnsw")]
fn batch_load_chunk_hits(
    conn: &Connection,
    query_embedding: &[f32],
    config: &SearchConfig,
    namespaces: Option<&[&str]>,
    entries: &HashMap<String, HnswCandidateSeed>,
    output: &mut Vec<VectorHit>,
) -> Result<(), MemoryError> {
    if entries.is_empty() {
        return Ok(());
    }

    let placeholders = (1..=entries.len())
        .map(|idx| format!("?{idx}"))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT c.id, c.content, c.document_id, d.title, c.chunk_index, c.created_at, d.namespace, c.embedding
         FROM chunks c
         JOIN documents d ON d.id = c.document_id
         WHERE c.id IN ({placeholders})"
    );
    let params: Vec<SqlValue> = entries
        .keys()
        .map(|id| SqlValue::Text(id.clone()))
        .collect();
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(&params), |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, i64>(4)?,
            row.get::<_, Option<String>>(5)?,
            row.get::<_, String>(6)?,
            row.get::<_, Option<Vec<u8>>>(7)?,
        ))
    })?;

    for row in rows {
        let (
            chunk_id,
            content,
            document_id,
            document_title,
            chunk_index,
            updated_at,
            namespace,
            embedding_blob,
        ) = row?;
        if let Some(filter) = namespaces {
            if !filter.contains(&namespace.as_str()) {
                continue;
            }
        }
        if let Some(seed) = entries.get(&chunk_id) {
            if let Some(hit) = build_ranked_vector_hit(
                format!("chunk:{chunk_id}"),
                content,
                SearchSource::Chunk {
                    chunk_id,
                    document_id,
                    document_title,
                    chunk_index: chunk_index as usize,
                },
                updated_at,
                embedding_blob,
                seed,
                query_embedding,
                config,
            )? {
                output.push(hit);
            }
        }
    }

    Ok(())
}

#[cfg(feature = "hnsw")]
fn batch_load_message_hits(
    conn: &Connection,
    query_embedding: &[f32],
    config: &SearchConfig,
    session_ids: Option<&[&str]>,
    entries: &HashMap<i64, HnswCandidateSeed>,
    output: &mut Vec<VectorHit>,
) -> Result<(), MemoryError> {
    if entries.is_empty() {
        return Ok(());
    }

    let placeholders = (1..=entries.len())
        .map(|idx| format!("?{idx}"))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT id, content, session_id, role, created_at, embedding
         FROM messages
         WHERE id IN ({placeholders})"
    );
    let params: Vec<SqlValue> = entries.keys().map(|id| SqlValue::Integer(*id)).collect();
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(&params), |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, Option<String>>(4)?,
            row.get::<_, Option<Vec<u8>>>(5)?,
        ))
    })?;

    for row in rows {
        let (message_id, content, session_id, role, updated_at, embedding_blob) = row?;
        if let Some(filter) = session_ids {
            if !filter.contains(&session_id.as_str()) {
                continue;
            }
        }
        if let Some(seed) = entries.get(&message_id) {
            if let Some(hit) = build_ranked_vector_hit(
                format!("msg:{message_id}"),
                content,
                SearchSource::Message {
                    message_id,
                    session_id,
                    role,
                },
                updated_at,
                embedding_blob,
                seed,
                query_embedding,
                config,
            )? {
                output.push(hit);
            }
        }
    }

    Ok(())
}

#[cfg(feature = "hnsw")]
fn batch_load_episode_hits(
    conn: &Connection,
    query_embedding: &[f32],
    config: &SearchConfig,
    namespaces: Option<&[&str]>,
    entries: &HashMap<String, HnswCandidateSeed>,
    output: &mut Vec<VectorHit>,
) -> Result<(), MemoryError> {
    if entries.is_empty() {
        return Ok(());
    }

    let placeholders = (1..=entries.len())
        .map(|idx| format!("?{idx}"))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT e.episode_id, e.document_id, e.search_text, e.effect_type, e.outcome, e.updated_at, d.namespace, e.embedding
         FROM episodes e
         JOIN documents d ON d.id = e.document_id
         WHERE e.episode_id IN ({placeholders})"
    );
    let params: Vec<SqlValue> = entries
        .keys()
        .map(|id| SqlValue::Text(id.clone()))
        .collect();
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(&params), |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, String>(4)?,
            row.get::<_, Option<String>>(5)?,
            row.get::<_, String>(6)?,
            row.get::<_, Option<Vec<u8>>>(7)?,
        ))
    })?;

    for row in rows {
        let (
            episode_id,
            document_id,
            content,
            effect_type,
            outcome,
            updated_at,
            namespace,
            embedding_blob,
        ) = row?;
        if let Some(filter) = namespaces {
            if !filter.contains(&namespace.as_str()) {
                continue;
            }
        }
        if let Some(seed) = entries.get(&episode_id) {
            if let Some(hit) = build_ranked_vector_hit(
                episodes::episode_item_key(&episode_id),
                content,
                SearchSource::Episode {
                    episode_id,
                    document_id,
                    effect_type,
                    outcome,
                },
                updated_at,
                embedding_blob,
                seed,
                query_embedding,
                config,
            )? {
                output.push(hit);
            }
        }
    }

    Ok(())
}

/// Perform a hybrid search using pre-computed HNSW hits for the vector component.
#[cfg(feature = "hnsw")]
#[allow(clippy::too_many_arguments)]
pub fn hybrid_search_with_hnsw(
    conn: &Connection,
    query: &str,
    query_embedding: &[f32],
    config: &SearchConfig,
    top_k: usize,
    namespaces: Option<&[&str]>,
    source_types: Option<&[SearchSourceType]>,
    session_ids: Option<&[&str]>,
    hnsw_hits: &[crate::hnsw::HnswHit],
) -> Result<Vec<SearchResult>, MemoryError> {
    Ok(hybrid_search_with_hnsw_detailed(
        conn,
        query,
        query_embedding,
        config,
        top_k,
        namespaces,
        source_types,
        session_ids,
        hnsw_hits,
    )?
    .into_iter()
    .map(|result| result.result)
    .collect())
}

#[cfg(feature = "hnsw")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn hybrid_search_with_hnsw_detailed(
    conn: &Connection,
    query: &str,
    query_embedding: &[f32],
    config: &SearchConfig,
    top_k: usize,
    namespaces: Option<&[&str]>,
    source_types: Option<&[SearchSourceType]>,
    session_ids: Option<&[&str]>,
    hnsw_hits: &[crate::hnsw::HnswHit],
) -> Result<Vec<ExplainedResult>, MemoryError> {
    let bm25_hits = match sanitize_fts_query(query) {
        Some(sanitized) => bm25_search(
            conn,
            &sanitized,
            config.candidate_pool_size,
            namespaces,
            source_types,
            session_ids,
        )?,
        None => Vec::new(),
    };

    let vector_hits = resolve_hnsw_hits_batched(
        conn,
        query_embedding,
        config,
        namespaces,
        source_types,
        session_ids,
        hnsw_hits,
    )?;

    Ok(rrf_fuse_detailed(&bm25_hits, &vector_hits, config, top_k))
}

/// Perform a hybrid HNSW-backed search and return the exact score decomposition.
#[cfg(feature = "hnsw")]
#[allow(clippy::too_many_arguments)]
pub fn hybrid_search_explained_with_hnsw(
    conn: &Connection,
    query: &str,
    query_embedding: &[f32],
    config: &SearchConfig,
    top_k: usize,
    namespaces: Option<&[&str]>,
    source_types: Option<&[SearchSourceType]>,
    session_ids: Option<&[&str]>,
    hnsw_hits: &[crate::hnsw::HnswHit],
) -> Result<Vec<ExplainedResult>, MemoryError> {
    hybrid_search_with_hnsw_detailed(
        conn,
        query,
        query_embedding,
        config,
        top_k,
        namespaces,
        source_types,
        session_ids,
        hnsw_hits,
    )
}

pub(crate) fn fts_only_search_detailed(
    conn: &Connection,
    query: &str,
    config: &SearchConfig,
    top_k: usize,
    namespaces: Option<&[&str]>,
    source_types: Option<&[SearchSourceType]>,
    session_ids: Option<&[&str]>,
) -> Result<Vec<ExplainedResult>, MemoryError> {
    let sanitized = match sanitize_fts_query(query) {
        Some(value) => value,
        None => return Ok(Vec::new()),
    };
    let bm25_hits = bm25_search(
        conn,
        &sanitized,
        top_k,
        namespaces,
        source_types,
        session_ids,
    )?;
    Ok(rrf_fuse_detailed(&bm25_hits, &[], config, top_k))
}

/// Full-text search only (no embeddings needed). Synchronous.
pub fn fts_only_search(
    conn: &Connection,
    query: &str,
    config: &SearchConfig,
    top_k: usize,
    namespaces: Option<&[&str]>,
    source_types: Option<&[SearchSourceType]>,
    session_ids: Option<&[&str]>,
) -> Result<Vec<SearchResult>, MemoryError> {
    Ok(fts_only_search_detailed(
        conn,
        query,
        config,
        top_k,
        namespaces,
        source_types,
        session_ids,
    )?
    .into_iter()
    .map(|result| result.result)
    .collect())
}

pub(crate) fn vector_only_search_detailed(
    conn: &Connection,
    query_embedding: &[f32],
    config: &SearchConfig,
    top_k: usize,
    namespaces: Option<&[&str]>,
    source_types: Option<&[SearchSourceType]>,
    session_ids: Option<&[&str]>,
) -> Result<Vec<ExplainedResult>, MemoryError> {
    let vector_hits = vector_search(
        conn,
        query_embedding,
        top_k,
        config.min_similarity,
        namespaces,
        source_types,
        session_ids,
    )?;
    Ok(rrf_fuse_detailed(&[], &vector_hits, config, top_k))
}

/// Vector-only search. Called after embedding the query.
pub fn vector_only_search(
    conn: &Connection,
    query_embedding: &[f32],
    config: &SearchConfig,
    top_k: usize,
    namespaces: Option<&[&str]>,
    source_types: Option<&[SearchSourceType]>,
    session_ids: Option<&[&str]>,
) -> Result<Vec<SearchResult>, MemoryError> {
    Ok(vector_only_search_detailed(
        conn,
        query_embedding,
        config,
        top_k,
        namespaces,
        source_types,
        session_ids,
    )?
    .into_iter()
    .map(|result| result.result)
    .collect())
}

/// Vector-only search using pre-computed HNSW hits.
#[cfg(feature = "hnsw")]
#[allow(clippy::too_many_arguments)]
pub fn vector_only_search_with_hnsw(
    conn: &Connection,
    query_embedding: &[f32],
    config: &SearchConfig,
    top_k: usize,
    namespaces: Option<&[&str]>,
    source_types: Option<&[SearchSourceType]>,
    session_ids: Option<&[&str]>,
    hnsw_hits: &[crate::hnsw::HnswHit],
) -> Result<Vec<SearchResult>, MemoryError> {
    Ok(vector_only_search_with_hnsw_detailed(
        conn,
        query_embedding,
        config,
        top_k,
        namespaces,
        source_types,
        session_ids,
        hnsw_hits,
    )?
    .into_iter()
    .map(|result| result.result)
    .collect())
}

#[cfg(feature = "hnsw")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn vector_only_search_with_hnsw_detailed(
    conn: &Connection,
    query_embedding: &[f32],
    config: &SearchConfig,
    top_k: usize,
    namespaces: Option<&[&str]>,
    source_types: Option<&[SearchSourceType]>,
    session_ids: Option<&[&str]>,
    hnsw_hits: &[crate::hnsw::HnswHit],
) -> Result<Vec<ExplainedResult>, MemoryError> {
    let vector_hits = resolve_hnsw_hits_batched(
        conn,
        query_embedding,
        config,
        namespaces,
        source_types,
        session_ids,
        hnsw_hits,
    )?;
    Ok(rrf_fuse_detailed(&[], &vector_hits, config, top_k))
}

fn build_filter_clause(
    column: &str,
    values: Option<&[&str]>,
    param_offset: usize,
) -> (String, Vec<SqlValue>) {
    match values {
        Some(values) if !values.is_empty() => {
            let placeholders = (0..values.len())
                .map(|idx| format!("?{}", param_offset + idx))
                .collect::<Vec<_>>();
            let clause = format!(" AND {} IN ({})", column, placeholders.join(", "));
            let params = values
                .iter()
                .map(|value| SqlValue::Text((*value).to_string()))
                .collect();
            (clause, params)
        }
        _ => (String::new(), Vec::new()),
    }
}

/// Deduplicate results by (source_type, source_id), keeping the first occurrence.
pub fn deduplicate_results(results: Vec<SearchResult>) -> Vec<SearchResult> {
    let mut seen = HashSet::new();
    results
        .into_iter()
        .filter(|result| seen.insert(source_dedup_key(&result.source)))
        .collect()
}
