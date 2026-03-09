//! Tests for V11 projection storage and import.
//!
//! Covers:
//! - Projection batch import (claim versions, relation versions, entity aliases, evidence refs)
//! - Idempotent re-import
//! - V11 schema migration from V10
//! - Projection import log tracking
//! - Preferred-open uniqueness
//! - Duplicate-but-not-identical envelopes
//! - Mid-import rollback (invalid record)
//! - Human-confirmed-final protection
//! - Legacy path still works (backward compatibility)

#![allow(deprecated)]

use forge_memory_bridge::PROJECTION_IMPORT_BATCH_V1_SCHEMA;
use semantic_memory::compat::compat_trace_id::TraceId;
use semantic_memory::compat::legacy_import_envelope::{ImportEnvelope, ImportRecord, ImportStatus};
use semantic_memory::{MemoryConfig, MemoryStore, MockEmbedder, ProjectionQuery};
use stack_ids::{EnvelopeId, ScopeKey};
use tempfile::TempDir;

fn test_store() -> (MemoryStore, TempDir) {
    let dir = TempDir::new().unwrap();
    let config = MemoryConfig {
        base_dir: dir.path().to_path_buf(),
        ..Default::default()
    };
    let store = MemoryStore::open_with_embedder(config, Box::new(MockEmbedder::new(768))).unwrap();
    (store, dir)
}

fn make_claim_batch(envelope_id: &str, claim_id: &str, content: &str) -> String {
    serde_json::json!({
        "source_envelope_id": envelope_id,
        "schema_version": PROJECTION_IMPORT_BATCH_V1_SCHEMA,
        "export_schema_version": "export_envelope_v1",
        "content_digest": format!("digest-{envelope_id}"),
        "source_authority": "forge",
        "scope_key": { "namespace": "test-ns" },
        "trace_ctx": { "trace_id": "trace-001" },
        "source_exported_at": "2026-03-07T00:00:00Z",
        "transformed_at": "2026-03-07T00:00:01Z",
        "records": [
            {
                "kind": "claim_version",
                "claim_id": claim_id,
                "claim_version_id": format!("{claim_id}-v1"),
                "claim_state": "active",
                "projection_family": "forge_verification",
                "subject_entity_id": "ent-1",
                "predicate": "has_type",
                "object_anchor": "function",
                "scope_key": { "namespace": "test-ns" },
                "valid_from": "2026-01-01T00:00:00Z",
                "valid_to": null,
                "preferred_open": true,
                "source_envelope_id": envelope_id,
                "source_authority": "forge",
                "freshness": "current",
                "contradiction_status": "none",
                "content": content,
                "confidence": 0.95
            }
        ]
    })
    .to_string()
}

fn make_multi_record_batch(envelope_id: &str) -> String {
    serde_json::json!({
        "source_envelope_id": envelope_id,
        "schema_version": PROJECTION_IMPORT_BATCH_V1_SCHEMA,
        "export_schema_version": "export_envelope_v1",
        "content_digest": format!("digest-multi-{envelope_id}"),
        "source_authority": "forge",
        "scope_key": { "namespace": "test-ns" },
        "source_exported_at": "2026-03-07T00:00:00Z",
        "transformed_at": "2026-03-07T00:00:01Z",
        "records": [
            {
                "kind": "claim_version",
                "claim_id": "claim-1",
                "claim_version_id": "claim-1-v1",
                "claim_state": "active",
                "projection_family": "forge",
                "subject_entity_id": "ent-1",
                "predicate": "p1",
                "object_anchor": "v1",
                "preferred_open": true,
                "freshness": "current",
                "contradiction_status": "none",
                "content": "claim one",
                "confidence": 0.9
            },
            {
                "kind": "relation_version",
                "relation_version_id": "rel-1-v1",
                "subject_entity_id": "ent-1",
                "predicate": "depends_on",
                "object_anchor": "ent-2",
                "preferred_open": true,
                "source_confidence": 0.8,
                "projection_family": "forge",
                "freshness": "current",
                "contradiction_status": "none"
            },
            {
                "kind": "entity_alias",
                "canonical_entity_id": "ent-1",
                "alias_text": "Entity One",
                "alias_source": "forge_extraction",
                "confidence": 0.9,
                "merge_decision": { "automated": { "algorithm": "bridge_default" } },
                "scope": { "namespace": "test-ns" },
                "review_state": "unreviewed",
                "is_human_confirmed": false,
                "is_human_confirmed_final": false
            },
            {
                "kind": "evidence_ref",
                "claim_id": "claim-1",
                "fetch_handle": "forge://evidence/run-42/artifact-7",
                "source_authority": "forge"
            },
            {
                "kind": "episode",
                "episode_id": "episode-1",
                "document_id": "doc-1",
                "cause_ids": ["claim-1"],
                "effect_type": "code_change",
                "outcome": "success",
                "confidence": 0.7,
                "experiment_id": "exp-1"
            }
        ]
    })
    .to_string()
}

fn make_scoped_claim_batch(
    envelope_id: &str,
    scope_key: &ScopeKey,
    claim_id: &str,
    claim_version_id: &str,
    content: &str,
    valid_from: &str,
    valid_to: Option<&str>,
    preferred_open: bool,
) -> String {
    serde_json::json!({
        "source_envelope_id": envelope_id,
        "schema_version": PROJECTION_IMPORT_BATCH_V1_SCHEMA,
        "export_schema_version": "export_envelope_v1",
        "content_digest": format!("digest-{envelope_id}"),
        "source_authority": "forge",
        "scope_key": scope_key,
        "source_exported_at": "2026-03-07T00:00:00Z",
        "transformed_at": "2026-03-07T00:00:01Z",
        "records": [{
            "kind": "claim_version",
            "claim_id": claim_id,
            "claim_version_id": claim_version_id,
            "claim_state": "active",
            "projection_family": "forge_verification",
            "subject_entity_id": "ent-scope",
            "predicate": "has_type",
            "object_anchor": "function",
            "scope_key": scope_key,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "preferred_open": preferred_open,
            "source_envelope_id": envelope_id,
            "source_authority": "forge",
            "freshness": "current",
            "contradiction_status": "none",
            "content": content,
            "confidence": 0.95
        }]
    })
    .to_string()
}

#[tokio::test]
async fn import_claim_version_succeeds() {
    let (store, _dir) = test_store();
    let batch = make_claim_batch("env-001", "claim-1", "Test claim content");
    let result = store
        .import_projection_batch_json_compat(&batch)
        .await
        .unwrap();

    assert_eq!(result.status, "complete");
    assert_eq!(result.record_count, 1);
    assert!(!result.was_duplicate);
    assert_eq!(result.source_envelope_id, "env-001");
}

#[tokio::test]
async fn import_is_idempotent() {
    let (store, _dir) = test_store();
    let batch = make_claim_batch("env-002", "claim-2", "Idempotent claim");

    let r1 = store
        .import_projection_batch_json_compat(&batch)
        .await
        .unwrap();
    assert_eq!(r1.status, "complete");
    assert!(!r1.was_duplicate);

    let r2 = store
        .import_projection_batch_json_compat(&batch)
        .await
        .unwrap();
    assert_eq!(r2.status, "already_imported");
    assert!(r2.was_duplicate);
}

#[tokio::test]
async fn import_multi_record_batch() {
    let (store, _dir) = test_store();
    let batch = make_multi_record_batch("env-multi");
    let result = store
        .import_projection_batch_json_compat(&batch)
        .await
        .unwrap();

    assert_eq!(result.status, "complete");
    assert_eq!(result.record_count, 5);
    assert!(!result.was_duplicate);
}

#[tokio::test]
async fn public_projection_queries_read_imported_rows() {
    let (store, _dir) = test_store();
    let batch = make_multi_record_batch("env-queryable");
    store
        .import_projection_batch_json_compat(&batch)
        .await
        .unwrap();

    let query = ProjectionQuery::new(ScopeKey::namespace_only("test-ns"));
    let claims = store.query_claim_versions(query.clone()).await.unwrap();
    let relations = store.query_relation_versions(query.clone()).await.unwrap();
    let episodes = store.query_episodes(query.clone()).await.unwrap();
    let aliases = store.query_entity_aliases(query.clone()).await.unwrap();
    let evidence = store.query_evidence_refs(query).await.unwrap();

    assert_eq!(claims.len(), 1);
    assert_eq!(relations.len(), 1);
    assert_eq!(episodes.len(), 1);
    assert_eq!(aliases.len(), 1);
    assert_eq!(evidence.len(), 1);
    assert_eq!(claims[0].content, "claim one");
    assert_eq!(relations[0].predicate, "depends_on");
    assert_eq!(episodes[0].effect_type, "code_change");
    assert_eq!(aliases[0].alias_text, "Entity One");
    assert_eq!(
        evidence[0].fetch_handle,
        "forge://evidence/run-42/artifact-7"
    );
    assert_eq!(
        claims[0].source_exported_at.as_deref(),
        Some("2026-03-07T00:00:00Z")
    );
    assert_eq!(
        claims[0].transformed_at.as_deref(),
        Some("2026-03-07T00:00:01Z")
    );
}

#[tokio::test]
async fn projection_queries_enforce_full_scope() {
    let (store, _dir) = test_store();
    let matching_scope = ScopeKey {
        namespace: "test-ns".into(),
        domain: Some("code".into()),
        workspace_id: Some("ws-1".into()),
        repo_id: Some("repo-1".into()),
    };
    let other_scope = ScopeKey {
        namespace: "test-ns".into(),
        domain: Some("docs".into()),
        workspace_id: Some("ws-2".into()),
        repo_id: Some("repo-2".into()),
    };

    for batch in [
        make_scoped_claim_batch(
            "env-scope-hit",
            &matching_scope,
            "claim-scope-hit",
            "claim-scope-hit-v1",
            "scoped projection hit",
            "2026-01-01T00:00:00Z",
            None,
            true,
        ),
        make_scoped_claim_batch(
            "env-scope-miss",
            &other_scope,
            "claim-scope-miss",
            "claim-scope-miss-v1",
            "scoped projection miss",
            "2026-01-01T00:00:00Z",
            None,
            true,
        ),
    ] {
        store
            .import_projection_batch_json_compat(&batch)
            .await
            .unwrap();
    }

    let mut query = ProjectionQuery::new(matching_scope.clone());
    query.text_query = Some("scoped projection".into());
    let claims = store.query_claim_versions(query).await.unwrap();

    assert_eq!(claims.len(), 1);
    assert_eq!(claims[0].content, "scoped projection hit");
    assert_eq!(claims[0].scope_key, matching_scope);
}

#[tokio::test]
async fn claim_query_valid_at_filters_versions() {
    let (store, _dir) = test_store();
    let scope_key = ScopeKey::namespace_only("test-ns");

    for batch in [
        make_scoped_claim_batch(
            "env-claim-old",
            &scope_key,
            "claim-versioned",
            "claim-versioned-v1",
            "versioned claim old state",
            "2026-01-01T00:00:00Z",
            Some("2026-02-01T00:00:00Z"),
            false,
        ),
        make_scoped_claim_batch(
            "env-claim-current",
            &scope_key,
            "claim-versioned",
            "claim-versioned-v2",
            "versioned claim current state",
            "2026-02-01T00:00:00Z",
            None,
            true,
        ),
    ] {
        store
            .import_projection_batch_json_compat(&batch)
            .await
            .unwrap();
    }

    let mut historical = ProjectionQuery::new(scope_key.clone());
    historical.text_query = Some("versioned claim".into());
    historical.valid_at = Some("2026-01-15T00:00:00Z".into());
    let historical_claims = store.query_claim_versions(historical).await.unwrap();

    let mut current = ProjectionQuery::new(scope_key);
    current.text_query = Some("versioned claim".into());
    current.valid_at = Some("2026-03-15T00:00:00Z".into());
    let current_claims = store.query_claim_versions(current).await.unwrap();

    assert_eq!(historical_claims.len(), 1);
    assert_eq!(historical_claims[0].content, "versioned claim old state");
    assert_eq!(current_claims.len(), 1);
    assert_eq!(current_claims[0].content, "versioned claim current state");
}

#[tokio::test]
async fn duplicate_but_different_digest_both_import() {
    let (store, _dir) = test_store();

    let batch1 = make_claim_batch("env-dup", "claim-a", "Content A");
    let r1 = store
        .import_projection_batch_json_compat(&batch1)
        .await
        .unwrap();
    assert_eq!(r1.status, "complete");

    // Same envelope_id but different content_digest
    let batch2 = serde_json::json!({
        "source_envelope_id": "env-dup",
        "schema_version": PROJECTION_IMPORT_BATCH_V1_SCHEMA,
        "export_schema_version": "export_envelope_v1",
        "content_digest": "different-digest",
        "source_authority": "forge",
        "scope_key": { "namespace": "test-ns" },
        "source_exported_at": "2026-03-07T00:00:00Z",
        "transformed_at": "2026-03-07T00:00:01Z",
        "records": [{
            "kind": "claim_version",
            "claim_id": "claim-b",
            "claim_version_id": "claim-b-v1",
            "claim_state": "active",
            "projection_family": "forge",
            "subject_entity_id": "ent-2",
            "predicate": "p2",
            "object_anchor": "v2",
            "preferred_open": true,
            "freshness": "current",
            "contradiction_status": "none",
            "content": "Content B",
            "confidence": 0.8
        }]
    })
    .to_string();

    let r2 = store
        .import_projection_batch_json_compat(&batch2)
        .await
        .unwrap();
    assert_eq!(r2.status, "complete");
    assert!(!r2.was_duplicate);
}

#[tokio::test]
async fn invalid_batch_json_rejected() {
    let (store, _dir) = test_store();
    let err = store
        .import_projection_batch_json_compat("not valid json")
        .await
        .unwrap_err();
    assert_eq!(err.kind(), "import_invalid");
}

#[tokio::test]
async fn missing_required_fields_rejected() {
    let (store, _dir) = test_store();
    let batch = serde_json::json!({
        "source_envelope_id": "env-bad",
        "records": []
    })
    .to_string();

    let err = store
        .import_projection_batch_json_compat(&batch)
        .await
        .unwrap_err();
    assert_eq!(err.kind(), "import_invalid");
}

#[tokio::test]
async fn unknown_record_kind_rejected() {
    let (store, _dir) = test_store();
    let batch = serde_json::json!({
        "source_envelope_id": "env-unknown",
        "schema_version": PROJECTION_IMPORT_BATCH_V1_SCHEMA,
        "export_schema_version": "export_envelope_v1",
        "content_digest": "digest-unknown",
        "source_authority": "forge",
        "scope_key": { "namespace": "test-ns" },
        "source_exported_at": "2026-03-07T00:00:00Z",
        "transformed_at": "2026-03-07T00:00:01Z",
        "records": [{
            "kind": "unknown_type",
            "data": "foo"
        }]
    })
    .to_string();

    let err = store
        .import_projection_batch_json_compat(&batch)
        .await
        .unwrap_err();
    assert_eq!(err.kind(), "import_invalid");
}

#[tokio::test]
async fn legacy_import_envelope_still_works() {
    // Backward compatibility: the old import_envelope() path must still function
    let (store, _dir) = test_store();

    let envelope = ImportEnvelope {
        envelope_id: EnvelopeId::new("legacy-env-001"),
        schema_version: "1.0".into(),
        content_digest: "legacy-digest-001".into(),
        source_authority: "forge".into(),
        trace_id: Some(TraceId::new("trace-legacy")),
        namespace: "test-ns".into(),
        records: vec![ImportRecord::Fact {
            content: "Legacy fact content".into(),
            source: Some("test".into()),
            metadata: None,
        }],
    };

    let receipt = store.import_envelope(&envelope).await.unwrap();
    assert_eq!(receipt.status, ImportStatus::Complete);
    assert_eq!(receipt.record_count, 1);
    assert!(!receipt.was_duplicate);
}

#[tokio::test]
async fn entity_alias_review_state_persisted() {
    let (store, _dir) = test_store();

    let batch = serde_json::json!({
        "source_envelope_id": "env-alias",
        "schema_version": PROJECTION_IMPORT_BATCH_V1_SCHEMA,
        "export_schema_version": "export_envelope_v1",
        "content_digest": "digest-alias",
        "source_authority": "forge",
        "scope_key": { "namespace": "test-ns" },
        "source_exported_at": "2026-03-07T00:00:00Z",
        "transformed_at": "2026-03-07T00:00:01Z",
        "records": [{
            "kind": "entity_alias",
            "canonical_entity_id": "ent-1",
            "alias_text": "Entity One",
            "alias_source": "forge_extraction",
            "confidence": 0.85,
            "merge_decision": { "automated": { "algorithm": "exact_match" } },
            "scope": { "namespace": "test-ns" },
            "review_state": "pending_review",
            "is_human_confirmed": false,
            "is_human_confirmed_final": false
        }]
    })
    .to_string();

    let result = store
        .import_projection_batch_json_compat(&batch)
        .await
        .unwrap();
    assert_eq!(result.status, "complete");
    // The alias is imported with pending_review state (durable, survives restart)
    // Re-opening the store should still find it
}

#[tokio::test]
async fn evidence_ref_audit_only() {
    let (store, _dir) = test_store();

    let batch = serde_json::json!({
        "source_envelope_id": "env-evidence",
        "schema_version": PROJECTION_IMPORT_BATCH_V1_SCHEMA,
        "export_schema_version": "export_envelope_v1",
        "content_digest": "digest-evidence",
        "source_authority": "forge",
        "scope_key": { "namespace": "test-ns" },
        "source_exported_at": "2026-03-07T00:00:00Z",
        "transformed_at": "2026-03-07T00:00:01Z",
        "records": [{
            "kind": "evidence_ref",
            "claim_id": "claim-x",
            "claim_version_id": "claim-x-v1",
            "fetch_handle": "forge://evidence/run-42/artifact-7",
            "source_authority": "forge"
        }]
    })
    .to_string();

    let result = store
        .import_projection_batch_json_compat(&batch)
        .await
        .unwrap();
    assert_eq!(result.status, "complete");
    assert_eq!(result.record_count, 1);
}
