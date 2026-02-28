use crate::error::MemoryError;
use serde::{Deserialize, Serialize};

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

/// Indicates whether a search result came from a fact, document chunk, or message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchSourceType {
    /// Result is from the facts table.
    Facts,
    /// Result is from the chunks table.
    Chunks,
    /// Result is from the messages table.
    Messages,
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
