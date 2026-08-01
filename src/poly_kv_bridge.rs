//! poly-kv bridge for semantic-memory integration.
//!
//! This module is gated behind `poly-kv-codec`. `poly-kv` represents a
//! model-specific KV cache, while semantic-memory stores standalone embeddings.
//! The bridge therefore refuses to manufacture a KV pool from embeddings: doing
//! so would attach a misleading compressed-KV receipt to data with no KV shape,
//! model fingerprint, tokenizer fingerprint, or exact fallback.

use crate::error::MemoryError;

/// Returns a typed rejection because arbitrary semantic embeddings are not
/// sufficient input to construct a truthful `poly_kv::SharedKvPool`.
///
/// A real pool must be built by `poly-kv` from authenticated `ExactKvBlock`s,
/// shape metadata, model/tokenizer fingerprints, and an exact fallback.
pub fn build_pool_from_embeddings(
    _embeddings: &[(String, Vec<f32>)],
    _embedding_dim: usize,
    _seed: u64,
) -> Result<(poly_kv::SharedKvPool, poly_kv::PoolBuildReceiptV1), MemoryError> {
    Err(MemoryError::Other(
        "cannot build a poly-kv KV-cache pool from semantic embeddings; supply authenticated ExactKvBlock input to poly-kv directly"
            .to_string(),
    ))
}

/// Convert a real poly-kv build receipt into the semantic-memory generation
/// record without inventing a codec result.
pub fn receipt_to_generation(
    receipt: &poly_kv::PoolBuildReceiptV1,
    snapshot_digest: &str,
) -> crate::types::ProveKvPoolGenerationV1 {
    crate::types::ProveKvPoolGenerationV1 {
        schema_version: "provekv_pool_generation_v1".into(),
        generation_id: receipt.manifest_digest.to_string(),
        embedding_snapshot_digest: snapshot_digest.to_string(),
        source_digest: receipt.input_digest.to_string(),
        pool_manifest_digest: receipt.manifest_digest.to_string(),
        codec_family: "poly-kv".to_string(),
        codec_profile: "alpha_reference".to_string(),
        vector_dim: 0,
        item_count: receipt.block_count as usize,
        payload_bytes: receipt.encoded_bytes as u64,
        created_at: chrono::Utc::now(),
    }
}

/// Serialization is deliberately unavailable through the bridge because
/// `SharedKvPool` keeps its internal fallback and encoded block state private.
/// Persist the canonical manifest and receipt provided by poly-kv instead.
pub fn serialize_pool(_pool: &poly_kv::SharedKvPool) -> Result<Vec<u8>, MemoryError> {
    Err(MemoryError::Other(
        "poly-kv pool serialization is not exposed by this bridge; persist canonical poly-kv artifacts at the owning boundary"
            .to_string(),
    ))
}

/// Deserialization is deliberately unavailable through the bridge. Rehydrate
/// pools through a verified poly-kv artifact loader at the owning boundary.
pub fn deserialize_pool(_data: &[u8]) -> Result<poly_kv::SharedKvPool, MemoryError> {
    Err(MemoryError::Other(
        "poly-kv pool deserialization is not exposed by this bridge; load verified poly-kv artifacts at the owning boundary"
            .to_string(),
    ))
}

/// Similarity search over a KV cache is not equivalent to semantic embedding
/// retrieval. This bridge refuses the operation rather than pretending that a
/// cache-layer decoder provides a semantic nearest-neighbour index.
pub fn search_pool_layer(
    _pool: &poly_kv::SharedKvPool,
    _layer_idx: usize,
    _query: &[f32],
    _top_k: usize,
) -> Result<Vec<(usize, f32)>, MemoryError> {
    Err(MemoryError::Other(
        "poly-kv pool-layer search is not a semantic retrieval primitive".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeddings_cannot_be_relabelled_as_a_kv_pool() {
        let error = build_pool_from_embeddings(&[("doc".to_string(), vec![1.0])], 1, 0)
            .expect_err("semantic embeddings cannot form a KV pool");
        assert!(error.to_string().contains("cannot build"));
    }

    #[test]
    fn pool_deserialization_fails_closed_without_a_poly_kv_loader() {
        let error = deserialize_pool(b"not-a-poly-kv-artifact")
            .expect_err("the bridge must not invent a pool from bytes");
        assert!(error.to_string().contains("deserialization is not exposed"));
    }
}
