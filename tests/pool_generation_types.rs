use chrono::Utc;
use semantic_memory::{
    DerivedCandidateReceiptV1, ProveKvPoolArtifactBuildReceiptV1, ProveKvPoolArtifactStatusV1,
    ProveKvPoolGenerationStatus, ProveKvPoolGenerationV1, ProveKvPoolItemMapEntryV1,
};

#[test]
fn provekv_pool_generation_types_roundtrip() {
    let created_at = Utc::now();
    let generation = ProveKvPoolGenerationV1 {
        schema_version: "semantic_memory_provekv_pool_generation_v1".to_string(),
        generation_id: "gen-1".to_string(),
        embedding_snapshot_digest: "blake3:snapshot".to_string(),
        source_digest: "blake3:source".to_string(),
        pool_manifest_digest: "blake3:manifest".to_string(),
        codec_family: "provekv_pool".to_string(),
        codec_profile: "semantic-memory-f32-derived-candidate-v1".to_string(),
        vector_dim: 3,
        item_count: 2,
        payload_bytes: 7,
        created_at,
    };

    let json = serde_json::to_string(&generation).unwrap();
    let decoded: ProveKvPoolGenerationV1 = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, generation);
    assert_eq!(
        decoded.schema_version,
        "semantic_memory_provekv_pool_generation_v1"
    );

    let entry = ProveKvPoolItemMapEntryV1 {
        generation_id: generation.generation_id.clone(),
        item_id: "fact:f1".to_string(),
        source_type: "fact".to_string(),
        pool_index: 0,
        embedding_digest: "blake3:item".to_string(),
    };
    let entry_json = serde_json::to_string(&entry).unwrap();
    assert_eq!(
        serde_json::from_str::<ProveKvPoolItemMapEntryV1>(&entry_json).unwrap(),
        entry
    );

    let status = ProveKvPoolArtifactStatusV1 {
        status: ProveKvPoolGenerationStatus::Ready,
        generation_id: Some(generation.generation_id.clone()),
        embedding_snapshot_digest: Some(generation.embedding_snapshot_digest.clone()),
        pool_manifest_digest: Some(generation.pool_manifest_digest.clone()),
        item_count: generation.item_count,
        payload_bytes: generation.payload_bytes,
        reason: None,
    };
    assert_eq!(serde_json::to_string(&status.status).unwrap(), "\"ready\"");
    assert_eq!(
        serde_json::from_str::<ProveKvPoolArtifactStatusV1>(
            &serde_json::to_string(&status).unwrap()
        )
        .unwrap(),
        status
    );
}

#[test]
fn provekv_pool_receipt_types_roundtrip() {
    let build = ProveKvPoolArtifactBuildReceiptV1 {
        schema_version: "semantic_memory_provekv_pool_build_receipt_v1".to_string(),
        generation_id: "gen-1".to_string(),
        embedding_snapshot_digest: "blake3:snapshot".to_string(),
        source_digest: "blake3:source".to_string(),
        pool_manifest_digest: "blake3:manifest".to_string(),
        codec_family: "provekv_pool".to_string(),
        codec_profile: "semantic-memory-f32-derived-candidate-v1".to_string(),
        vector_dim: 2,
        item_count: 1,
        payload_bytes: 0,
        exact_rerank_required: true,
        created_at: Utc::now(),
    };
    assert_eq!(
        serde_json::from_str::<ProveKvPoolArtifactBuildReceiptV1>(
            &serde_json::to_string(&build).unwrap()
        )
        .unwrap(),
        build
    );

    let receipt = DerivedCandidateReceiptV1 {
        candidate_backend: "provekv_pool_candidate_then_exact_f32".to_string(),
        codec_family: Some("provekv_pool".to_string()),
        generation_id: None,
        embedding_snapshot_digest: None,
        pool_manifest_digest: None,
        exact_rerank: true,
        approximate: false,
        fallback: Some("provekv_pool_generation_not_materialized".to_string()),
        raw_candidate_count: 3,
        post_filter_count: 2,
        final_result_count: 1,
    };
    assert_eq!(
        serde_json::from_str::<DerivedCandidateReceiptV1>(
            &serde_json::to_string(&receipt).unwrap()
        )
        .unwrap(),
        receipt
    );
}
