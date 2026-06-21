//! First-class stored graph edges — durable, typed relationships between
//! any two nodes in the knowledge graph.
//!
//! Unlike the derived edges in [`crate::graph`] (which are computed on-the-fly
//! from SQLite state), these edges are explicitly created by agents or users
//! and persisted in the `graph_edges` table (migration V27).
//!
//! ## Authority
//!
//! `semantic-memory` is authoritative for graph edge state. These edges are
//! first-class knowledge relationships, not projection lineage (which lives in
//! `derivation_edges`).
//!
//! ## Bitemporal semantics
//!
//! Edges are append-only. Invalidation is a separate operation that sets
//! `is_invalidated = 1` with a timestamp and reason — the original row is
//! never deleted. This follows the append-plus-supersession doctrine.
//!
//! ## Idempotent insertion
//!
//! Each edge carries a `content_digest` (blake3 of source + target + edge_type
//! + weight + metadata). Inserting the same edge twice returns the existing
//! row ID without creating a duplicate.

use crate::error::MemoryError;
use crate::types::{GraphEdge, GraphEdgeType};
use chrono::Utc;
use rusqlite::{params, Connection};
use stack_ids::DigestBuilder;

/// A stored graph edge row.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StoredGraphEdge {
    /// Unique edge ID (UUID v4).
    pub id: String,
    /// Source node ID (prefixed, e.g. `fact:<uuid>`).
    pub source: String,
    /// Target node ID (prefixed, e.g. `fact:<uuid>`).
    pub target: String,
    /// Serialized GraphEdgeType JSON.
    pub edge_type: String,
    /// Deserialized edge type.
    #[serde(skip)]
    pub edge_type_parsed: Option<GraphEdgeType>,
    /// Edge weight.
    pub weight: f64,
    /// Optional metadata JSON.
    pub metadata: Option<String>,
    /// Content digest (blake3 of source+target+edge_type+weight+metadata).
    pub content_digest: String,
    /// Recorded timestamp (microsecond precision).
    pub recorded_at: String,
    /// Whether this edge has been invalidated.
    pub is_invalidated: bool,
    /// Invalidation timestamp.
    pub invalidated_at: Option<String>,
    /// Invalidation reason.
    pub invalidation_reason: Option<String>,
}

/// Parameters for creating a graph edge.
#[derive(Debug, Clone)]
pub struct AddGraphEdgeParams {
    /// Source node ID (prefixed, e.g. `fact:<uuid>`, `namespace:<name>`).
    pub source: String,
    /// Target node ID (prefixed).
    pub target: String,
    /// Edge type (semantic, temporal, causal, entity).
    pub edge_type: GraphEdgeType,
    /// Edge weight (interpretation depends on edge_type).
    pub weight: f64,
    /// Optional metadata.
    pub metadata: Option<serde_json::Value>,
}

/// Insert a graph edge. Idempotent on content_digest.
pub(crate) fn insert_graph_edge(
    conn: &Connection,
    params: &AddGraphEdgeParams,
) -> Result<StoredGraphEdge, MemoryError> {
    let edge_type_json = serde_json::to_string(&params.edge_type)
        .map_err(|e| MemoryError::Other(format!("failed to serialize edge_type: {e}")))?;

    let metadata_json = match &params.metadata {
        Some(v) => Some(serde_json::to_string(v)
            .map_err(|e| MemoryError::Other(format!("failed to serialize metadata: {e}")))?),
        None => None,
    };

    let content_digest = compute_edge_digest(
        &params.source,
        &params.target,
        &edge_type_json,
        params.weight,
        &metadata_json,
    );

    // Check for existing edge with same digest (idempotent).
    let existing: Option<(String, String)> = conn
        .query_row(
            "SELECT id, recorded_at FROM graph_edges WHERE content_digest = ?1 AND is_invalidated = 0",
            params![&content_digest],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .ok();

    if let Some((id, recorded_at)) = existing {
        return Ok(StoredGraphEdge {
            id,
            source: params.source.clone(),
            target: params.target.clone(),
            edge_type: edge_type_json,
            edge_type_parsed: Some(params.edge_type.clone()),
            weight: params.weight,
            metadata: metadata_json,
            content_digest,
            recorded_at,
            is_invalidated: false,
            invalidated_at: None,
            invalidation_reason: None,
        });
    }

    let id = uuid::Uuid::new_v4().to_string();
    let recorded_at = Utc::now().format("%Y-%m-%d %H:%M:%S%.6f").to_string();

    conn.execute(
        "INSERT INTO graph_edges (id, source, target, edge_type, weight, metadata, content_digest, recorded_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            &id,
            &params.source,
            &params.target,
            &edge_type_json,
            params.weight,
            metadata_json.as_deref(),
            &content_digest,
            &recorded_at,
        ],
    )
    .map_err(|e| MemoryError::Database(e))?;

    Ok(StoredGraphEdge {
        id,
        source: params.source.clone(),
        target: params.target.clone(),
        edge_type: edge_type_json,
        edge_type_parsed: Some(params.edge_type.clone()),
        weight: params.weight,
        metadata: metadata_json,
        content_digest,
        recorded_at,
        is_invalidated: false,
        invalidated_at: None,
        invalidation_reason: None,
    })
}

/// List all stored graph edges involving a given node (either as source or
/// target), excluding invalidated edges.
pub(crate) fn list_graph_edges_for_node(
    conn: &Connection,
    node_id: &str,
) -> Result<Vec<StoredGraphEdge>, MemoryError> {
    let mut stmt = conn.prepare(
        "SELECT id, source, target, edge_type, weight, metadata, content_digest, recorded_at,
                is_invalidated, invalidated_at, invalidation_reason
         FROM graph_edges
         WHERE (source = ?1 OR target = ?1) AND is_invalidated = 0
         ORDER BY recorded_at ASC",
    )?;
    let rows = stmt
        .query_map(params![node_id], |row| {
            Ok(StoredGraphEdge {
                id: row.get(0)?,
                source: row.get(1)?,
                target: row.get(2)?,
                edge_type: row.get(3)?,
                edge_type_parsed: None,
                weight: row.get(4)?,
                metadata: row.get(5)?,
                content_digest: row.get(6)?,
                recorded_at: row.get(7)?,
                is_invalidated: row.get::<_, i64>(8)? != 0,
                invalidated_at: row.get(9)?,
                invalidation_reason: row.get(10)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// List ALL stored graph edges, excluding invalidated ones.
pub(crate) fn list_all_graph_edges(conn: &Connection) -> Result<Vec<StoredGraphEdge>, MemoryError> {
    let mut stmt = conn.prepare(
        "SELECT id, source, target, edge_type, weight, metadata, content_digest, recorded_at,
                is_invalidated, invalidated_at, invalidation_reason
         FROM graph_edges
         WHERE is_invalidated = 0
         ORDER BY recorded_at ASC",
    )?;
    let rows = stmt
        .query_map([], |row| {
            Ok(StoredGraphEdge {
                id: row.get(0)?,
                source: row.get(1)?,
                target: row.get(2)?,
                edge_type: row.get(3)?,
                edge_type_parsed: None,
                weight: row.get(4)?,
                metadata: row.get(5)?,
                content_digest: row.get(6)?,
                recorded_at: row.get(7)?,
                is_invalidated: row.get::<_, i64>(8)? != 0,
                invalidated_at: row.get(9)?,
                invalidation_reason: row.get(10)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// Invalidate a graph edge by ID. Append-only — does not delete the row.
pub(crate) fn invalidate_graph_edge(
    conn: &Connection,
    edge_id: &str,
    reason: &str,
) -> Result<(), MemoryError> {
    let invalidated_at = Utc::now().format("%Y-%m-%d %H:%M:%S%.6f").to_string();
    let count = conn.execute(
        "UPDATE graph_edges SET is_invalidated = 1, invalidated_at = ?1, invalidation_reason = ?2
         WHERE id = ?3 AND is_invalidated = 0",
        params![&invalidated_at, reason, edge_id],
    )?;
    if count == 0 {
        return Err(MemoryError::Other(format!(
            "graph edge {} not found or already invalidated",
            edge_id
        )));
    }
    Ok(())
}

/// Load stored graph edges for a node and convert them to GraphEdge objects
/// for the derived graph view. Only non-invalidated edges are included.
pub(crate) fn stored_edges_for_node(
    conn: &Connection,
    node_id: &str,
) -> Result<Vec<GraphEdge>, MemoryError> {
    let rows = list_graph_edges_for_node(conn, node_id)?;
    let mut edges = Vec::new();
    for row in rows {
        let edge_type: GraphEdgeType = serde_json::from_str(&row.edge_type)
            .map_err(|e| MemoryError::Other(format!("failed to deserialize edge_type: {e}")))?;
        let metadata: Option<serde_json::Value> = match &row.metadata {
            Some(s) => Some(serde_json::from_str(s).map_err(|e| {
                MemoryError::Other(format!("failed to deserialize metadata: {e}"))
            })?),
            None => None,
        };
        edges.push(GraphEdge {
            source: row.source,
            target: row.target,
            edge_type,
            weight: row.weight,
            metadata,
        });
    }
    Ok(edges)
}

/// Load stored outgoing edges for a node (where node is the source).
/// Only non-invalidated edges are included.
pub(crate) fn stored_outgoing_edges(
    conn: &Connection,
    node_id: &str,
) -> Result<Vec<GraphEdge>, MemoryError> {
    let mut stmt = conn.prepare(
        "SELECT id, source, target, edge_type, weight, metadata, content_digest, recorded_at,
                is_invalidated, invalidated_at, invalidation_reason
         FROM graph_edges
         WHERE source = ?1 AND is_invalidated = 0
         ORDER BY recorded_at ASC",
    )?;
    let rows = stmt
        .query_map(params![node_id], |row| {
            Ok(StoredGraphEdge {
                id: row.get(0)?,
                source: row.get(1)?,
                target: row.get(2)?,
                edge_type: row.get(3)?,
                edge_type_parsed: None,
                weight: row.get(4)?,
                metadata: row.get(5)?,
                content_digest: row.get(6)?,
                recorded_at: row.get(7)?,
                is_invalidated: row.get::<_, i64>(8)? != 0,
                invalidated_at: row.get(9)?,
                invalidation_reason: row.get(10)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    rows_to_graph_edges(rows)
}

/// Load stored incoming edges for a node (where node is the target).
/// Only non-invalidated edges are included.
pub(crate) fn stored_incoming_edges(
    conn: &Connection,
    node_id: &str,
) -> Result<Vec<GraphEdge>, MemoryError> {
    let mut stmt = conn.prepare(
        "SELECT id, source, target, edge_type, weight, metadata, content_digest, recorded_at,
                is_invalidated, invalidated_at, invalidation_reason
         FROM graph_edges
         WHERE target = ?1 AND is_invalidated = 0
         ORDER BY recorded_at ASC",
    )?;
    let rows = stmt
        .query_map(params![node_id], |row| {
            Ok(StoredGraphEdge {
                id: row.get(0)?,
                source: row.get(1)?,
                target: row.get(2)?,
                edge_type: row.get(3)?,
                edge_type_parsed: None,
                weight: row.get(4)?,
                metadata: row.get(5)?,
                content_digest: row.get(6)?,
                recorded_at: row.get(7)?,
                is_invalidated: row.get::<_, i64>(8)? != 0,
                invalidated_at: row.get(9)?,
                invalidation_reason: row.get(10)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    rows_to_graph_edges(rows)
}

/// Convert StoredGraphEdge rows into GraphEdge objects.
fn rows_to_graph_edges(rows: Vec<StoredGraphEdge>) -> Result<Vec<GraphEdge>, MemoryError> {
    let mut edges = Vec::new();
    for row in rows {
        let edge_type: GraphEdgeType = serde_json::from_str(&row.edge_type)
            .map_err(|e| MemoryError::Other(format!("failed to deserialize edge_type: {e}")))?;
        let metadata: Option<serde_json::Value> = match &row.metadata {
            Some(s) => Some(serde_json::from_str(s).map_err(|e| {
                MemoryError::Other(format!("failed to deserialize metadata: {e}"))
            })?),
            None => None,
        };
        edges.push(GraphEdge {
            source: row.source,
            target: row.target,
            edge_type,
            weight: row.weight,
            metadata,
        });
    }
    Ok(edges)
}
/// Count total stored edges (non-invalidated).
pub(crate) fn count_graph_edges(conn: &Connection) -> Result<usize, MemoryError> {
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM graph_edges WHERE is_invalidated = 0",
            [],
            |row| row.get(0),
        )
        .map_err(|e| MemoryError::Database(e))?;
    Ok(count as usize)
}

fn compute_edge_digest(
    source: &str,
    target: &str,
    edge_type_json: &str,
    weight: f64,
    metadata_json: &Option<String>,
) -> String {
    let mut builder = stack_ids::DigestBuilder::new();
    builder.update(source.as_bytes());
    builder.update(target.as_bytes());
    builder.update(edge_type_json.as_bytes());
    builder.update(&weight.to_le_bytes());
    if let Some(meta) = metadata_json {
        builder.update(meta.as_bytes());
    }
    builder.finalize().0
}