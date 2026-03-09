#![allow(deprecated)]

use crate::error::MemoryError;
use serde::{Deserialize, Serialize};
use stack_ids::{
    ClaimId, ClaimVersionId, EntityId, EnvelopeId, EpisodeId, RelationVersionId, ScopeKey,
};

/// Stable trace identifier used for cross-crate correlation and auditability.
///
/// ## Phase status: compatibility / migration-only
///
/// This is a crate-local `TraceId` retained for backward compatibility.
/// The canonical replacement is `stack_ids::TraceCtx`. Use
/// `TraceCtx::from_legacy_trace_id()` to convert.
///
/// **Removal condition**: removed when all internal usage migrates to `TraceCtx`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CompatTraceId(pub String);

#[deprecated(since = "0.5.0", note = "Use stack_ids::TraceCtx instead")]
pub type TraceId = CompatTraceId;

impl CompatTraceId {
    /// Create a trace ID from any owned string-like input.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Borrow the trace ID as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for CompatTraceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for CompatTraceId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for CompatTraceId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

/// Role of a message in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System prompt / instructions.
    System,
    /// User message.
    User,
    /// Assistant (LLM) response.
    Assistant,
    /// Tool call result.
    Tool,
}

impl Role {
    /// Convert to the string stored in SQLite.
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }

    /// Parse from the string stored in SQLite.
    pub fn from_str_value(s: &str) -> Option<Self> {
        match s {
            "system" => Some(Role::System),
            "user" => Some(Role::User),
            "assistant" => Some(Role::Assistant),
            "tool" => Some(Role::Tool),
            _ => None,
        }
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Role {
    type Err = MemoryError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_str_value(s).ok_or_else(|| MemoryError::Other(format!("Unknown role: '{}'", s)))
    }
}

/// Indicates whether a search result came from a fact, document chunk, message, or episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchSourceType {
    /// Result is from the facts table.
    Facts,
    /// Result is from the chunks table.
    Chunks,
    /// Result is from the messages table.
    Messages,
    /// Result is from the episodes table.
    Episodes,
}

/// Common filter surface for imported projection queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionQuery {
    /// Full scope to enforce.
    pub scope: ScopeKey,
    /// Optional free-text query applied to the projection's searchable fields.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_query: Option<String>,
    /// Valid-time as-of filter for versioned projection rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_at: Option<String>,
    /// Transaction-time cutoff for imported rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recorded_at_or_before: Option<String>,
    /// Optional subject-entity filter for claim/relation queries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject_entity_id: Option<EntityId>,
    /// Optional canonical-entity filter for alias queries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canonical_entity_id: Option<EntityId>,
    /// Optional claim-state filter for claim-version queries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claim_state: Option<String>,
    /// Optional claim filter for claim/evidence queries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claim_id: Option<ClaimId>,
    /// Optional claim-version filter for evidence queries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claim_version_id: Option<ClaimVersionId>,
    /// Final result limit.
    pub limit: usize,
}

impl ProjectionQuery {
    pub fn new(scope: ScopeKey) -> Self {
        Self {
            scope,
            text_query: None,
            valid_at: None,
            recorded_at_or_before: None,
            subject_entity_id: None,
            canonical_entity_id: None,
            claim_state: None,
            claim_id: None,
            claim_version_id: None,
            limit: 10,
        }
    }
}

/// Public read shape for imported claim projection rows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionClaimVersion {
    pub claim_version_id: ClaimVersionId,
    pub claim_id: ClaimId,
    pub claim_state: String,
    pub projection_family: String,
    pub subject_entity_id: EntityId,
    pub predicate: String,
    pub object_anchor: serde_json::Value,
    pub scope_key: ScopeKey,
    pub valid_from: Option<String>,
    pub valid_to: Option<String>,
    pub recorded_at: String,
    pub preferred_open: bool,
    pub source_envelope_id: EnvelopeId,
    pub source_authority: String,
    pub trace_id: Option<String>,
    pub freshness: String,
    pub contradiction_status: String,
    pub supersedes_claim_version_id: Option<ClaimVersionId>,
    pub content: String,
    pub confidence: f32,
    pub metadata: Option<serde_json::Value>,
    pub source_exported_at: Option<String>,
    pub transformed_at: Option<String>,
}

/// Public read shape for imported relation projection rows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionRelationVersion {
    pub relation_version_id: RelationVersionId,
    pub subject_entity_id: EntityId,
    pub predicate: String,
    pub object_anchor: serde_json::Value,
    pub scope_key: ScopeKey,
    pub claim_id: Option<ClaimId>,
    pub source_episode_id: Option<EpisodeId>,
    pub valid_from: Option<String>,
    pub valid_to: Option<String>,
    pub recorded_at: String,
    pub preferred_open: bool,
    pub supersedes_relation_version_id: Option<RelationVersionId>,
    pub contradiction_status: String,
    pub source_confidence: f32,
    pub projection_family: String,
    pub source_envelope_id: EnvelopeId,
    pub source_authority: String,
    pub trace_id: Option<String>,
    pub freshness: String,
    pub metadata: Option<serde_json::Value>,
    pub source_exported_at: Option<String>,
    pub transformed_at: Option<String>,
}

/// Public read shape for imported episode projection rows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionEpisode {
    pub episode_id: EpisodeId,
    pub document_id: String,
    pub cause_ids: Vec<String>,
    pub effect_type: String,
    pub outcome: String,
    pub confidence: f32,
    pub experiment_id: Option<String>,
    pub scope_key: ScopeKey,
    pub source_envelope_id: EnvelopeId,
    pub source_authority: String,
    pub trace_id: Option<String>,
    pub recorded_at: String,
    pub metadata: Option<serde_json::Value>,
    pub source_exported_at: Option<String>,
    pub transformed_at: Option<String>,
}

/// Public read shape for imported entity-alias rows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionEntityAlias {
    pub canonical_entity_id: EntityId,
    pub alias_text: String,
    pub alias_source: String,
    pub match_evidence: Option<serde_json::Value>,
    pub confidence: f32,
    pub merge_decision: String,
    pub scope_key: ScopeKey,
    pub review_state: String,
    pub is_human_confirmed: bool,
    pub is_human_confirmed_final: bool,
    pub superseded_by_entity_id: Option<EntityId>,
    pub split_from_entity_id: Option<EntityId>,
    pub source_envelope_id: EnvelopeId,
    pub recorded_at: String,
    pub source_exported_at: Option<String>,
    pub transformed_at: Option<String>,
}

/// Public read shape for imported evidence-reference rows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionEvidenceRef {
    pub claim_id: ClaimId,
    pub claim_version_id: Option<ClaimVersionId>,
    pub fetch_handle: String,
    pub source_authority: String,
    pub source_envelope_id: EnvelopeId,
    pub scope_key: ScopeKey,
    pub recorded_at: String,
    pub metadata: Option<serde_json::Value>,
    pub source_exported_at: Option<String>,
    pub transformed_at: Option<String>,
}

/// A conversation session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// UUID v4.
    pub id: String,
    /// Channel identifier (e.g. "repl", "telegram").
    pub channel: String,
    /// ISO 8601 timestamp.
    pub created_at: String,
    /// ISO 8601 timestamp.
    pub updated_at: String,
    /// Optional JSON metadata.
    pub metadata: Option<serde_json::Value>,
    /// Number of messages (populated on list queries).
    pub message_count: u32,
}

/// A single message within a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Auto-increment ID.
    pub id: i64,
    /// Session this message belongs to.
    pub session_id: String,
    /// Role of the speaker.
    pub role: Role,
    /// Message text.
    pub content: String,
    /// Estimated token count (caller-provided).
    pub token_count: Option<u32>,
    /// ISO 8601 timestamp.
    pub created_at: String,
    /// Optional JSON metadata.
    pub metadata: Option<serde_json::Value>,
}

/// A discrete fact in the knowledge store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    /// UUID v4.
    pub id: String,
    /// Categorization namespace.
    pub namespace: String,
    /// The fact text.
    pub content: String,
    /// Where this fact came from.
    pub source: Option<String>,
    /// ISO 8601 timestamp.
    pub created_at: String,
    /// ISO 8601 timestamp.
    pub updated_at: String,
    /// Optional JSON metadata.
    pub metadata: Option<serde_json::Value>,
}

/// A source document that has been chunked and embedded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// UUID v4.
    pub id: String,
    /// Document title.
    pub title: String,
    /// File path, URL, or identifier.
    pub source_path: Option<String>,
    /// Categorization namespace.
    pub namespace: String,
    /// ISO 8601 timestamp.
    pub created_at: String,
    /// Optional JSON metadata.
    pub metadata: Option<serde_json::Value>,
    /// Number of chunks (populated on list queries).
    pub chunk_count: u32,
}

/// A chunk produced by the text splitter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextChunk {
    /// Position in the original document (0-based).
    pub index: usize,
    /// The chunk text.
    pub content: String,
    /// Rough token estimate (chars / 4).
    pub token_count_estimate: usize,
}

/// A single search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The matched text content.
    pub content: String,

    /// Where this result came from.
    pub source: SearchSource,

    /// Combined RRF score. Higher = more relevant.
    pub score: f64,

    /// BM25 rank (1-based) if this result appeared in BM25 results.
    pub bm25_rank: Option<usize>,

    /// Vector rank (1-based) if this result appeared in vector results.
    pub vector_rank: Option<usize>,

    /// Cosine similarity score if computed.
    pub cosine_similarity: Option<f64>,
}

/// Source information for a search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchSource {
    /// Result came from the facts table.
    Fact {
        /// Fact UUID.
        fact_id: String,
        /// Fact namespace.
        namespace: String,
    },
    /// Result came from a document chunk.
    Chunk {
        /// Chunk UUID.
        chunk_id: String,
        /// Parent document UUID.
        document_id: String,
        /// Parent document title.
        document_title: String,
        /// Position within the document (0-based).
        chunk_index: usize,
    },
    /// Result came from a conversation message.
    Message {
        /// Message auto-increment ID.
        message_id: i64,
        /// Session UUID.
        session_id: String,
        /// Message role (user, assistant, etc.).
        role: String,
    },
    /// Result came from an episode (causal record). SearchSource::Episode variant.
    Episode {
        /// First-class episode identity (V9+). Falls back to `document_id + "-ep0"`
        /// for legacy data.
        episode_id: String,
        /// Document ID the episode is attached to.
        document_id: String,
        /// Type of effect (e.g. "test_failure", "regression").
        effect_type: String,
        /// Current outcome.
        outcome: String,
    },
    /// Result came from an imported projection row.
    Projection {
        /// Projection row family, such as `claim_version` or `relation_version`.
        projection_kind: String,
        /// Stable projection-row identity.
        projection_id: String,
        /// Full scope carried by the imported row.
        scope_key: ScopeKey,
        /// Validity start for versioned projections, if any.
        valid_from: Option<String>,
        /// Validity end for versioned projections, if any.
        valid_to: Option<String>,
        /// Authoritative importer-assigned recorded_at.
        recorded_at: String,
        /// Source envelope provenance.
        source_envelope_id: String,
        /// Source authority provenance.
        source_authority: String,
    },
}

// ─── Episode Types ─────────────────────────────────────────────

/// Metadata for a causal episode (PRIMITIVES_CONTRACT §4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeMeta {
    /// IDs of the facts/chunks/messages that caused this episode.
    pub cause_ids: Vec<String>,
    /// Type of effect (e.g. "test_failure", "regression", "improvement").
    pub effect_type: String,
    /// Current outcome assessment.
    pub outcome: EpisodeOutcome,
    /// Confidence in the causal link (0.0 to 1.0).
    pub confidence: f32,
    /// Verification status.
    pub verification_status: VerificationStatus,
    /// Links to an EvidenceBundle.run_id (if experimentally verified).
    pub experiment_id: Option<String>,
}

/// Outcome of an episode's causal hypothesis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EpisodeOutcome {
    /// Causal link confirmed by experiment.
    Confirmed,
    /// Causal link refuted by experiment.
    Refuted,
    /// Evidence is inconclusive.
    Inconclusive,
    /// Not yet tested.
    Pending,
}

impl EpisodeOutcome {
    /// Convert to the string stored in SQLite.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Confirmed => "confirmed",
            Self::Refuted => "refuted",
            Self::Inconclusive => "inconclusive",
            Self::Pending => "pending",
        }
    }

    /// Parse from the string stored in SQLite.
    pub fn from_str_value(s: &str) -> Option<Self> {
        match s {
            "confirmed" => Some(Self::Confirmed),
            "refuted" => Some(Self::Refuted),
            "inconclusive" => Some(Self::Inconclusive),
            "pending" => Some(Self::Pending),
            _ => None,
        }
    }
}

impl std::fmt::Display for EpisodeOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Verification status for an episode.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum VerificationStatus {
    /// Not yet verified.
    Unverified,
    /// Successfully verified.
    Verified {
        /// Method used for verification.
        method: String,
        /// When verification occurred (ISO 8601).
        at: String,
    },
    /// Verification attempt failed.
    Failed {
        /// Reason for failure.
        reason: String,
        /// When verification was attempted (ISO 8601).
        at: String,
    },
}

// ─── Score Breakdown ───────────────────────────────────────────

/// Detailed score breakdown for explainable search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    /// Final fused RRF score.
    pub rrf_score: f64,
    /// Raw BM25 score reported by SQLite FTS5 (lower is better).
    pub bm25_score: Option<f64>,
    /// Raw vector similarity used for the final vector ordering.
    pub vector_score: Option<f64>,
    /// Recency contribution added during fusion.
    pub recency_score: Option<f64>,
    /// BM25 rank (1-based).
    pub bm25_rank: Option<usize>,
    /// Vector rank (1-based).
    pub vector_rank: Option<usize>,
    /// Rank from the underlying vector retrieval source before any exact rerank.
    pub vector_source_rank: Option<usize>,
    /// Similarity score from the underlying vector retrieval source before rerank.
    pub vector_source_score: Option<f64>,
    /// BM25 RRF contribution to the final score.
    pub bm25_contribution: Option<f64>,
    /// Vector RRF contribution to the final score.
    pub vector_contribution: Option<f64>,
    /// Whether the vector ordering was reranked with exact f32 cosine similarity.
    pub vector_reranked_from_f32: bool,
    /// Configured BM25 fusion weight.
    pub bm25_weight: f64,
    /// Configured vector fusion weight.
    pub vector_weight: f64,
    /// Configured recency weight when recency is enabled.
    pub recency_weight: Option<f64>,
    /// Configured RRF decay constant.
    pub rrf_k: f64,
}

/// Search result with full score explanation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplainedResult {
    /// The search result.
    pub result: SearchResult,
    /// Score breakdown.
    pub breakdown: ScoreBreakdown,
}

// ─── Graph Types (PRIMITIVES_CONTRACT §8) ──────────────────────

/// Trait for querying the memory store as a graph.
pub trait GraphView: Send + Sync {
    /// Find neighboring nodes up to `max_depth` hops away.
    fn neighbors(
        &self,
        node_id: &str,
        direction: GraphDirection,
        max_depth: usize,
    ) -> Result<Vec<GraphEdge>, MemoryError>;

    /// Find a path between two nodes (BFS, max depth).
    fn path(
        &self,
        from: &str,
        to: &str,
        max_depth: usize,
    ) -> Result<Option<Vec<String>>, MemoryError>;
}

/// Direction for graph traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphDirection {
    /// Follow outgoing edges.
    Outgoing,
    /// Follow incoming edges.
    Incoming,
    /// Follow edges in both directions.
    Both,
}

/// An edge in the memory graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Source node ID.
    pub source: String,
    /// Target node ID.
    pub target: String,
    /// Type of relationship.
    pub edge_type: GraphEdgeType,
    /// Edge weight (interpretation depends on edge_type).
    pub weight: f64,
    /// Optional metadata.
    pub metadata: Option<serde_json::Value>,
}

/// Type of relationship between graph nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphEdgeType {
    /// Semantic similarity. GraphEdgeType::Semantic variant.
    Semantic {
        /// Cosine similarity between embeddings.
        cosine_similarity: f32,
    },
    /// Temporal proximity. GraphEdgeType::Temporal variant.
    Temporal {
        /// Time delta in seconds.
        delta_secs: u64,
    },
    /// Causal relationship. GraphEdgeType::Causal variant.
    Causal {
        /// Confidence in the causal link.
        confidence: f32,
        /// EvidenceBundle run_ids supporting this link.
        evidence_ids: Vec<String>,
    },
    /// Entity co-occurrence. GraphEdgeType::Entity variant.
    Entity {
        /// Relationship type (e.g. "mentions", "modifies").
        relation: String,
    },
}

/// Embedding displacement between two text embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingDisplacement {
    /// Cosine similarity between the two embeddings.
    pub cosine_similarity: f32,
    /// Euclidean distance between the two embeddings.
    pub euclidean_distance: f32,
    /// Magnitude of the first embedding.
    pub magnitude_a: f32,
    /// Magnitude of the second embedding.
    pub magnitude_b: f32,
}

/// Database statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Total number of facts.
    pub total_facts: u64,
    /// Total number of documents.
    pub total_documents: u64,
    /// Total number of chunks across all documents.
    pub total_chunks: u64,
    /// Total number of conversation sessions.
    pub total_sessions: u64,
    /// Total number of messages across all sessions.
    pub total_messages: u64,
    /// Database file size in bytes.
    pub database_size_bytes: u64,
    /// Currently configured embedding model.
    pub embedding_model: Option<String>,
    /// Currently configured embedding dimensions.
    pub embedding_dimensions: Option<usize>,
}
