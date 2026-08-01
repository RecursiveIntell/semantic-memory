//! LLM receipt ingestion adapter for semantic-memory.
//!
//! This module provides a bridge for storing llm-pipeline execution receipts
//! as evidence in semantic-memory. It does NOT depend on llm-pipeline directly —
//! it accepts receipt JSON and digests as opaque strings, keeping the dependency
//! boundary clean.
//!
//! The adapter stores:
//! - The receipt digest (SHA-256 of canonical JSON) as an evidence reference
//! - The traceparent for trace correlation
//! - The model/provider/pipeline IDs as metadata
//! - The integrity verification status
//!
//! Usage:
//! ```no_run
//! use semantic_memory::MemoryStore;
//! use semantic_memory::llm_receipt_ingest::LlmReceiptEvidence;
//!
//! # async fn example(store: MemoryStore) -> Result<(), Box<dyn std::error::Error>> {
//! let evidence = LlmReceiptEvidence {
//!     receipt_digest: "sha256:abc123...".to_string(),
//!     traceparent: Some("00-trace-id-span-01".to_string()),
//!     pipeline_id: "pipeline-001".to_string(),
//!     provider: "ollama".to_string(),
//!     model: "llama3.2:3b".to_string(),
//!     integrity_verified: true,
//!     receipt_json: r#"{"receipt_id":"..."}"#.to_string(),
//! };
//! let fact_id = store.add_fact(
//!     "llm-executions",
//!     "LLM call completed: ollama/llama3.2:3b",
//!     Some("llm-pipeline"),
//!     Some(serde_json::to_value(&evidence)?),
//! ).await?;
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};

/// Evidence metadata for an LLM execution receipt stored in semantic-memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmReceiptEvidence {
    /// SHA-256 digest of the canonical receipt JSON.
    pub receipt_digest: String,
    /// W3C traceparent for trace correlation.
    pub traceparent: Option<String>,
    /// Pipeline execution ID from the receipt.
    pub pipeline_id: String,
    /// Provider name (e.g., "ollama", "openai", "anthropic").
    pub provider: String,
    /// Model identifier used for the call.
    pub model: String,
    /// Whether the receipt's HMAC integrity tag was verified before ingestion.
    pub integrity_verified: bool,
    /// The full canonical receipt JSON, stored for replay.
    pub receipt_json: String,
}

impl LlmReceiptEvidence {
    /// Create evidence from canonical receipt JSON and a verification result.
    pub fn new(
        receipt_json: &str,
        digest: &str,
        integrity_verified: bool,
        traceparent: Option<String>,
        pipeline_id: &str,
        provider: &str,
        model: &str,
    ) -> Self {
        Self {
            receipt_digest: digest.to_string(),
            traceparent,
            pipeline_id: pipeline_id.to_string(),
            provider: provider.to_string(),
            model: model.to_string(),
            integrity_verified,
            receipt_json: receipt_json.to_string(),
        }
    }

    /// Compute the SHA-256 digest of canonical receipt JSON.
    pub fn compute_digest(receipt_json: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(receipt_json.as_bytes());
        format!("sha256:{}", hex::encode(hasher.finalize()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_receipt_evidence_serialization() {
        let evidence = LlmReceiptEvidence::new(
            r#"{"receipt_id":"test-001"}"#,
            "sha256:abc123",
            true,
            Some("00-trace-id-span-01".to_string()),
            "pipeline-001",
            "ollama",
            "llama3.2:3b",
        );
        let json = serde_json::to_string(&evidence).unwrap();
        let restored: LlmReceiptEvidence = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.receipt_digest, "sha256:abc123");
        assert_eq!(restored.provider, "ollama");
        assert_eq!(restored.model, "llama3.2:3b");
        assert!(restored.integrity_verified);
        assert!(restored.traceparent.is_some());
    }

    #[test]
    fn test_compute_digest_deterministic() {
        let json = r#"{"a":1,"b":2}"#;
        let d1 = LlmReceiptEvidence::compute_digest(json);
        let d2 = LlmReceiptEvidence::compute_digest(json);
        assert_eq!(d1, d2);
        assert!(d1.starts_with("sha256:"));
    }

    #[test]
    fn test_compute_digest_different_inputs() {
        let d1 = LlmReceiptEvidence::compute_digest(r#"{"a":1}"#);
        let d2 = LlmReceiptEvidence::compute_digest(r#"{"a":2}"#);
        assert_ne!(d1, d2);
    }
}
