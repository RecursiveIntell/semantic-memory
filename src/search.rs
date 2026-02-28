//! Hybrid search engine: BM25 + vector similarity + Reciprocal Rank Fusion.

use crate::config::SearchConfig;
use crate::error::MemoryError;
use crate::types::{SearchResult, SearchSource, SearchSourceType};
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
    // Filter out bare FTS5 boolean operators that would cause query errors
    let tokens: Vec<&str> = cleaned
        .split_whitespace()
        .filter(|t| !matches!(t.to_uppercase().as_str(), "AND" | "OR" | "NOT" | "NEAR"))
        .collect();
    if tokens.is_empty() {
        None
    } else {
        Some(tokens.join(" "))
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

/// Compute the number of days since an ISO 8601 timestamp (SQLite format).
fn days_since(iso_timestamp: &str) -> Option<f64> {
    let dt = chrono::NaiveDateTime::parse_from_str(iso_timestamp, "%Y-%m-%d %H:%M:%S").ok()?;
    let now = chrono::Utc::now().naive_utc();
    let duration = now - dt;
    Some(duration.num_seconds() as f64 / 86400.0)
}

/// An RRF candidate accumulating scores from BM25 and vector search.
struct RrfCandidate {
    content: String,
    source: SearchSource,
    bm25_rank: Option<usize>,
    vector_rank: Option<usize>,
    cosine_similarity: Option<f64>,
    updated_at: Option<String>,
}

impl RrfCandidate {
    fn score(&self, config: &SearchConfig) -> f64 {
        let bm25_score = self
            .bm25_rank
            .map(|r| config.bm25_weight / (config.rrf_k + r as f64))
            .unwrap_or(0.0);
        let vector_score = self
            .vector_rank
            .map(|r| config.vector_weight / (config.rrf_k + r as f64))
            .unwrap_or(0.0);

        let recency_score = match (config.recency_half_life_days, &self.updated_at) {
            (Some(half_life), Some(ts)) if half_life > 0.0 => {
                let age_days = days_since(ts).unwrap_or(0.0).max(0.0);
                let decay = 2.0_f64.powf(-age_days / half_life);
                config.recency_weight * decay / (config.rrf_k + 1.0)
            }
            (Some(half_life), _) if half_life <= 0.0 => {
                tracing::warn!("recency_half_life_days <= 0, ignoring recency boost");
                0.0
            }
            _ => 0.0,
        };

        bm25_score + vector_score + recency_score
    }
}

/// A BM25 search hit from FTS5.
pub struct Bm25Hit {
    /// Item ID (fact_id or chunk_id).
    pub id: String,
    /// Text content.
    pub content: String,
    /// Source info.
    pub source: SearchSource,
    /// Timestamp for recency scoring.
    pub updated_at: Option<String>,
}

/// A vector search hit.
pub struct VectorHit {
    /// Item ID (fact_id or chunk_id).
    pub id: String,
    /// Text content.
    pub content: String,
    /// Source info.
    pub source: SearchSource,
    /// Cosine similarity score.
    pub similarity: f64,
    /// Timestamp for recency scoring.
    pub updated_at: Option<String>,
}

/// Row data extracted from a SQLite query for vector similarity scoring.
struct VectorRow {
    id: String,
    content: String,
    blob: Vec<u8>,
    updated_at: Option<String>,
    source: SearchSource,
}

/// Decode embedding BLOBs and compute cosine similarity for a set of rows.
///
/// Shared logic for the facts, chunks, and messages vector search loops.
/// Returns the matching hits and the total row count scanned.
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
        let stored_embedding: &[f32] =
            bytemuck::try_cast_slice(&row.blob).map_err(|_| MemoryError::InvalidEmbedding {
                expected_bytes: row.blob.len() - (row.blob.len() % 4),
                actual_bytes: row.blob.len(),
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

        let sim = cosine_similarity(query_embedding, stored_embedding) as f64;
        if sim >= min_similarity {
            hits.push(VectorHit {
                id: row.id,
                content: row.content,
                source: row.source,
                similarity: sim,
                updated_at: row.updated_at,
            });
        }
    }

    Ok((hits, row_count))
}

/// Run BM25 search over facts_fts, chunks_fts, and optionally messages_fts.
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
    // Messages are NOT included by default — only when explicitly requested
    let search_messages = source_types
        .map(|st| st.contains(&SearchSourceType::Messages))
        .unwrap_or(false);

    // Search facts
    if search_facts {
        let (ns_clause, ns_params) = build_namespace_clause("f.namespace", namespaces, 3);
        let sql = format!(
            "SELECT fm.fact_id, f.content, f.namespace, bm25(facts_fts) AS score, f.updated_at
             FROM facts_fts
             JOIN facts_rowid_map fm ON facts_fts.rowid = fm.rowid
             JOIN facts f ON f.id = fm.fact_id
             WHERE facts_fts MATCH ?1 {}
             ORDER BY bm25(facts_fts)
             LIMIT ?2",
            ns_clause
        );

        let mut all_params: Vec<SqlValue> = vec![
            SqlValue::Text(sanitized_query.to_string()),
            SqlValue::Integer(pool_size as i64),
        ];
        all_params.extend(ns_params.clone());

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(&all_params), |row| {
            let fact_id: String = row.get(0)?;
            let content: String = row.get(1)?;
            let namespace: String = row.get(2)?;
            let updated_at: Option<String> = row.get(4)?;
            Ok(Bm25Hit {
                id: fact_id.clone(),
                content,
                source: SearchSource::Fact { fact_id, namespace },
                updated_at,
            })
        })?;

        for row in rows {
            hits.push(row?);
        }
    }

    // Search chunks
    if search_chunks {
        let (ns_clause, ns_params) = build_namespace_clause("d.namespace", namespaces, 3);
        let sql = format!(
            "SELECT cm.chunk_id, c.content, c.document_id, d.title, c.chunk_index, bm25(chunks_fts) AS score, c.created_at
             FROM chunks_fts
             JOIN chunks_rowid_map cm ON chunks_fts.rowid = cm.rowid
             JOIN chunks c ON c.id = cm.chunk_id
             JOIN documents d ON d.id = c.document_id
             WHERE chunks_fts MATCH ?1 {}
             ORDER BY bm25(chunks_fts)
             LIMIT ?2",
            ns_clause
        );

        let mut all_params: Vec<SqlValue> = vec![
            SqlValue::Text(sanitized_query.to_string()),
            SqlValue::Integer(pool_size as i64),
        ];
        all_params.extend(ns_params.clone());

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(&all_params), |row| {
            let chunk_id: String = row.get(0)?;
            let content: String = row.get(1)?;
            let document_id: String = row.get(2)?;
            let document_title: String = row.get(3)?;
            let chunk_index: i64 = row.get(4)?;
            let updated_at: Option<String> = row.get(6)?;
            Ok(Bm25Hit {
                id: chunk_id.clone(),
                content,
                source: SearchSource::Chunk {
                    chunk_id,
                    document_id,
                    document_title,
                    chunk_index: chunk_index as usize,
                },
                updated_at,
            })
        })?;

        for row in rows {
            hits.push(row?);
        }
    }

    // Search messages (only when explicitly requested)
    if search_messages {
        let (sid_clause, sid_params) = build_namespace_clause("m.session_id", session_ids, 3);
        let sql = format!(
            "SELECT mm.message_id, m.content, m.session_id, m.role, bm25(messages_fts) AS score, m.created_at
             FROM messages_fts
             JOIN messages_rowid_map mm ON messages_fts.rowid = mm.rowid
             JOIN messages m ON m.id = mm.message_id
             WHERE messages_fts MATCH ?1 {}
             ORDER BY bm25(messages_fts)
             LIMIT ?2",
            sid_clause
        );

        let mut all_params: Vec<SqlValue> = vec![
            SqlValue::Text(sanitized_query.to_string()),
            SqlValue::Integer(pool_size as i64),
        ];
        all_params.extend(sid_params.clone());

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(&all_params), |row| {
            let message_id: i64 = row.get(0)?;
            let content: String = row.get(1)?;
            let session_id: String = row.get(2)?;
            let role: String = row.get(3)?;
            let updated_at: Option<String> = row.get(5)?;
            Ok(Bm25Hit {
                id: format!("msg:{}", message_id),
                content,
                source: SearchSource::Message {
                    message_id,
                    session_id,
                    role,
                },
                updated_at,
            })
        })?;

        for row in rows {
            hits.push(row?);
        }
    }

    Ok(hits)
}

/// Run vector similarity search over facts, chunks, and optionally messages.
///
/// Uses buffer reuse to avoid per-row allocations during the brute-force scan.
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

    // Vector search over facts
    if search_facts {
        let (ns_clause, ns_params) = build_namespace_clause("namespace", namespaces, 1);
        let sql = format!(
            "SELECT id, content, namespace, embedding, updated_at FROM facts WHERE embedding IS NOT NULL {}",
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
                id: id.clone(),
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
                "facts table exceeds vector scan threshold ({} rows). \
                 Consider namespace partitioning or pruning old data.",
                fact_count
            );
        }
    }

    // Vector search over chunks
    if search_chunks {
        let (ns_clause, ns_params) = build_namespace_clause("d.namespace", namespaces, 1);
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
                id: id.clone(),
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
                "chunks table exceeds vector scan threshold ({} rows). \
                 Consider namespace partitioning or pruning old data.",
                chunk_count
            );
        }
    }

    // Vector search over messages (only when explicitly requested)
    if search_messages {
        let (sid_clause, sid_params) = build_namespace_clause("m.session_id", session_ids, 1);
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
                id: format!("msg:{}", message_id),
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

        let (msg_hits, msg_count) = scan_vector_rows(
            rows,
            query_embedding,
            min_similarity,
            "message",
        )?;
        hits.extend(msg_hits);

        if msg_count > VECTOR_SCAN_WARN_THRESHOLD {
            tracing::warn!(
                count = msg_count,
                "messages table exceeds vector scan threshold ({} rows). \
                 Consider pruning old sessions.",
                msg_count
            );
        }
    }

    // Sort by similarity descending, take top pool_size
    hits.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits.truncate(pool_size);

    Ok(hits)
}

/// Fuse BM25 and vector results via Reciprocal Rank Fusion.
pub fn rrf_fuse(
    bm25_hits: &[Bm25Hit],
    vector_hits: &[VectorHit],
    config: &SearchConfig,
    top_k: usize,
) -> Vec<SearchResult> {
    let mut candidates: HashMap<String, RrfCandidate> = HashMap::new();

    // Walk BM25 results (ranks are 1-based)
    for (rank_0, hit) in bm25_hits.iter().enumerate() {
        let rank = rank_0 + 1;
        candidates
            .entry(hit.id.clone())
            .and_modify(|c| {
                c.bm25_rank = Some(rank);
                // Prefer the most recent timestamp if both sources provide one
                if c.updated_at.is_none() {
                    c.updated_at = hit.updated_at.clone();
                }
            })
            .or_insert(RrfCandidate {
                content: hit.content.clone(),
                source: hit.source.clone(),
                bm25_rank: Some(rank),
                vector_rank: None,
                cosine_similarity: None,
                updated_at: hit.updated_at.clone(),
            });
    }

    // Walk vector results (ranks are 1-based)
    for (rank_0, hit) in vector_hits.iter().enumerate() {
        let rank = rank_0 + 1;
        candidates
            .entry(hit.id.clone())
            .and_modify(|c| {
                c.vector_rank = Some(rank);
                c.cosine_similarity = Some(hit.similarity);
                if c.updated_at.is_none() {
                    c.updated_at = hit.updated_at.clone();
                }
            })
            .or_insert(RrfCandidate {
                content: hit.content.clone(),
                source: hit.source.clone(),
                bm25_rank: None,
                vector_rank: Some(rank),
                cosine_similarity: Some(hit.similarity),
                updated_at: hit.updated_at.clone(),
            });
    }

    // Score, sort, truncate
    let mut results: Vec<SearchResult> = candidates
        .into_values()
        .map(|c| {
            let score = c.score(config);
            SearchResult {
                content: c.content,
                source: c.source,
                score,
                bm25_rank: c.bm25_rank,
                vector_rank: c.vector_rank,
                cosine_similarity: c.cosine_similarity,
            }
        })
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(top_k);
    results
}

/// Perform a hybrid search (BM25 + vector + RRF).
///
/// This is the main search entry point. Embed query first (async outside this fn),
/// then call this with the connection locked.
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
    // BM25 search
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

    // Vector search
    let vector_hits = vector_search(
        conn,
        query_embedding,
        config.candidate_pool_size,
        config.min_similarity,
        namespaces,
        source_types,
        session_ids,
    )?;

    // RRF fusion + dedup
    let results = rrf_fuse(&bm25_hits, &vector_hits, config, top_k);
    Ok(deduplicate_results(results))
}

/// Perform a hybrid search using pre-computed HNSW hits for the vector component.
///
/// Instead of brute-force scanning all rows, this takes the HNSW nearest neighbor
/// results and looks up their content from SQLite. The rest (BM25 + RRF fusion)
/// is identical to `hybrid_search`.
#[cfg(feature = "hnsw")]
#[allow(clippy::too_many_arguments)]
pub fn hybrid_search_with_hnsw(
    conn: &Connection,
    query: &str,
    _query_embedding: &[f32],
    config: &SearchConfig,
    top_k: usize,
    namespaces: Option<&[&str]>,
    source_types: Option<&[SearchSourceType]>,
    session_ids: Option<&[&str]>,
    hnsw_hits: &[crate::hnsw::HnswHit],
) -> Result<Vec<SearchResult>, MemoryError> {
    // BM25 search (same as hybrid_search)
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

    // Convert HNSW hits to VectorHits via batched SQLite lookups
    let vector_hits = resolve_hnsw_hits_batched(
        conn, config, namespaces, source_types, session_ids, hnsw_hits,
    )?;

    // RRF fusion + dedup
    let results = rrf_fuse(&bm25_hits, &vector_hits, config, top_k);
    Ok(deduplicate_results(results))
}

/// Resolve HNSW hits to VectorHits using batched SQL queries (one per domain).
///
/// Replaces the N+1 query pattern with at most 3 batch queries.
#[cfg(feature = "hnsw")]
fn resolve_hnsw_hits_batched(
    conn: &Connection,
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

    // Partition HNSW hits by domain
    let mut fact_entries: Vec<(String, f64)> = Vec::new();
    let mut chunk_entries: Vec<(String, f64)> = Vec::new();
    let mut msg_entries: Vec<(i64, f64)> = Vec::new();

    for hit in hnsw_hits {
        let similarity = hit.similarity() as f64;
        if similarity < config.min_similarity {
            continue;
        }
        match hit.key.split_once(':') {
            Some(("fact", id)) if search_facts => fact_entries.push((id.to_string(), similarity)),
            Some(("chunk", id)) if search_chunks => chunk_entries.push((id.to_string(), similarity)),
            Some(("msg", id)) if search_messages => {
                if let Ok(mid) = id.parse::<i64>() {
                    msg_entries.push((mid, similarity));
                }
            }
            _ => continue,
        }
    }

    let mut vector_hits = Vec::new();

    // Batch load facts
    if !fact_entries.is_empty() {
        let sim_map: HashMap<String, f64> = fact_entries.iter().cloned().collect();
        let placeholders: String = (1..=fact_entries.len())
            .map(|i| format!("?{}", i))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT id, content, namespace, updated_at FROM facts WHERE id IN ({})",
            placeholders
        );
        let params: Vec<SqlValue> = fact_entries
            .iter()
            .map(|(id, _)| SqlValue::Text(id.clone()))
            .collect();

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(&params), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
            ))
        })?;

        for row in rows {
            let (fact_id, content, namespace, updated_at) = row?;
            if let Some(ns) = namespaces {
                if !ns.contains(&namespace.as_str()) {
                    continue;
                }
            }
            if let Some(&similarity) = sim_map.get(&fact_id) {
                vector_hits.push(VectorHit {
                    id: fact_id.clone(),
                    content,
                    source: SearchSource::Fact { fact_id, namespace },
                    similarity,
                    updated_at,
                });
            }
        }
    }

    // Batch load chunks
    if !chunk_entries.is_empty() {
        let sim_map: HashMap<String, f64> = chunk_entries.iter().cloned().collect();
        let placeholders: String = (1..=chunk_entries.len())
            .map(|i| format!("?{}", i))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT c.id, c.content, c.document_id, d.title, c.chunk_index, c.created_at, d.namespace
             FROM chunks c JOIN documents d ON d.id = c.document_id
             WHERE c.id IN ({})",
            placeholders
        );
        let params: Vec<SqlValue> = chunk_entries
            .iter()
            .map(|(id, _)| SqlValue::Text(id.clone()))
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
            ))
        })?;

        for row in rows {
            let (chunk_id, content, document_id, document_title, chunk_index, updated_at, doc_ns) = row?;
            if let Some(ns) = namespaces {
                if !ns.contains(&doc_ns.as_str()) {
                    continue;
                }
            }
            if let Some(&similarity) = sim_map.get(&chunk_id) {
                vector_hits.push(VectorHit {
                    id: chunk_id.clone(),
                    content,
                    source: SearchSource::Chunk {
                        chunk_id,
                        document_id,
                        document_title,
                        chunk_index: chunk_index as usize,
                    },
                    similarity,
                    updated_at,
                });
            }
        }
    }

    // Batch load messages
    if !msg_entries.is_empty() {
        let sim_map: HashMap<i64, f64> = msg_entries.iter().cloned().collect();
        let placeholders: String = (1..=msg_entries.len())
            .map(|i| format!("?{}", i))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT id, content, session_id, role, created_at FROM messages WHERE id IN ({})",
            placeholders
        );
        let params: Vec<SqlValue> = msg_entries
            .iter()
            .map(|(id, _)| SqlValue::Integer(*id))
            .collect();

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(&params), |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, Option<String>>(4)?,
            ))
        })?;

        for row in rows {
            let (message_id, content, session_id, role, updated_at) = row?;
            if let Some(sids) = session_ids {
                if !sids.contains(&session_id.as_str()) {
                    continue;
                }
            }
            if let Some(&similarity) = sim_map.get(&message_id) {
                vector_hits.push(VectorHit {
                    id: format!("msg:{}", message_id),
                    content,
                    source: SearchSource::Message {
                        message_id,
                        session_id,
                        role,
                    },
                    similarity,
                    updated_at,
                });
            }
        }
    }

    // Sort by similarity descending
    vector_hits.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    vector_hits.truncate(config.candidate_pool_size);

    Ok(vector_hits)
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
    let sanitized = match sanitize_fts_query(query) {
        Some(s) => s,
        None => return Ok(Vec::new()),
    };

    let hits = bm25_search(
        conn,
        &sanitized,
        top_k,
        namespaces,
        source_types,
        session_ids,
    )?;

    let results: Vec<SearchResult> = hits
        .into_iter()
        .enumerate()
        .map(|(rank_0, hit)| SearchResult {
            content: hit.content,
            source: hit.source,
            score: config.bm25_weight / (config.rrf_k + (rank_0 + 1) as f64),
            bm25_rank: Some(rank_0 + 1),
            vector_rank: None,
            cosine_similarity: None,
        })
        .collect();

    Ok(deduplicate_results(results))
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
    let hits = vector_search(
        conn,
        query_embedding,
        top_k,
        config.min_similarity,
        namespaces,
        source_types,
        session_ids,
    )?;

    let results: Vec<SearchResult> = hits
        .into_iter()
        .enumerate()
        .map(|(rank_0, hit)| SearchResult {
            content: hit.content,
            source: hit.source,
            score: config.vector_weight / (config.rrf_k + (rank_0 + 1) as f64),
            bm25_rank: None,
            vector_rank: Some(rank_0 + 1),
            cosine_similarity: Some(hit.similarity),
        })
        .collect();

    Ok(deduplicate_results(results))
}

/// Vector-only search using pre-computed HNSW hits.
///
/// Skips BM25 entirely. Uses batched SQL lookups via `resolve_hnsw_hits_batched`.
#[cfg(feature = "hnsw")]
#[allow(clippy::too_many_arguments)]
pub fn vector_only_search_with_hnsw(
    conn: &Connection,
    config: &SearchConfig,
    top_k: usize,
    namespaces: Option<&[&str]>,
    source_types: Option<&[SearchSourceType]>,
    session_ids: Option<&[&str]>,
    hnsw_hits: &[crate::hnsw::HnswHit],
) -> Result<Vec<SearchResult>, MemoryError> {
    let mut vector_hits = resolve_hnsw_hits_batched(
        conn, config, namespaces, source_types, session_ids, hnsw_hits,
    )?;
    vector_hits.truncate(top_k);

    let results: Vec<SearchResult> = vector_hits
        .into_iter()
        .enumerate()
        .map(|(rank_0, hit)| SearchResult {
            content: hit.content,
            source: hit.source,
            score: config.vector_weight / (config.rrf_k + (rank_0 + 1) as f64),
            bm25_rank: None,
            vector_rank: Some(rank_0 + 1),
            cosine_similarity: Some(hit.similarity),
        })
        .collect();

    Ok(deduplicate_results(results))
}

/// Extract a dedupe key from a search source: (source type discriminant, primary ID).
///
/// This preserves provenance — the same text in a fact and a chunk are kept,
/// but the same source appearing via both BM25 and vector paths is deduplicated.
fn source_dedup_key(source: &SearchSource) -> (u8, String) {
    match source {
        SearchSource::Fact { fact_id, .. } => (0, fact_id.clone()),
        SearchSource::Chunk { chunk_id, .. } => (1, chunk_id.clone()),
        SearchSource::Message { message_id, .. } => (2, message_id.to_string()),
    }
}

/// Deduplicate results by (source_type, source_id), keeping the first (highest-scored) occurrence.
fn deduplicate_results(results: Vec<SearchResult>) -> Vec<SearchResult> {
    let mut seen = HashSet::new();
    results
        .into_iter()
        .filter(|r| seen.insert(source_dedup_key(&r.source)))
        .collect()
}

/// Build a parameterized namespace filter SQL fragment.
///
/// Returns a tuple of (SQL clause, parameter values). The `param_offset` sets
/// the starting numbered placeholder (e.g., if existing query uses ?1 and ?2,
/// pass `param_offset = 3`).
fn build_namespace_clause(
    column: &str,
    namespaces: Option<&[&str]>,
    param_offset: usize,
) -> (String, Vec<SqlValue>) {
    match namespaces {
        Some(ns) if !ns.is_empty() => {
            let placeholders: Vec<String> = (0..ns.len())
                .map(|i| format!("?{}", param_offset + i))
                .collect();
            let clause = format!("AND {} IN ({})", column, placeholders.join(", "));
            let values: Vec<SqlValue> = ns.iter().map(|n| SqlValue::Text(n.to_string())).collect();
            (clause, values)
        }
        _ => (String::new(), vec![]),
    }
}
