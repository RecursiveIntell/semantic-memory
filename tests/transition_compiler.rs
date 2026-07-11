use semantic_memory::{
    AssertionDraftV1, AuthorityFaultStage, AuthorityPermit, MemoryConfig, MemoryStore,
    MemoryTransitionCandidateV1, MemoryTransitionOutcomeV1, MockEmbedder, SourceArtifactV1,
    SourceSpanRefV1, SupersessionDraftV1, TransitionDisposition, TransitionOperation,
};
use tempfile::TempDir;

fn test_store() -> (MemoryStore, TempDir) {
    let tmp = TempDir::new().unwrap();
    let store = MemoryStore::open_with_embedder(
        MemoryConfig {
            base_dir: tmp.path().to_path_buf(),
            ..Default::default()
        },
        Box::new(MockEmbedder::new(768)),
    )
    .unwrap();
    (store, tmp)
}

fn permit(capability: &str) -> AuthorityPermit {
    AuthorityPermit::operator_system("principal:test", "caller:test", capability)
}

fn artifact(id: &str, content: &str) -> SourceArtifactV1 {
    SourceArtifactV1::new(id, content).unwrap()
}

fn span(artifact_id: &str, start: usize, end: usize) -> SourceSpanRefV1 {
    SourceSpanRefV1::new(artifact_id, start, end).unwrap()
}

fn append_candidate(
    candidate_id: &str,
    artifact_id: &str,
    evidence: &str,
    assertion: &str,
) -> MemoryTransitionCandidateV1 {
    let source_span = span(artifact_id, 0, evidence.len());
    MemoryTransitionCandidateV1::new(
        candidate_id,
        vec![artifact(artifact_id, evidence)],
        vec![source_span.clone()],
        vec![AssertionDraftV1::new(
            "assertion-1",
            "general",
            assertion,
            vec![source_span],
            vec![],
        )
        .unwrap()],
        TransitionOperation::Append {
            assertion_id: "assertion-1".into(),
        },
        vec![],
    )
    .unwrap()
}

#[test]
fn source_span_and_candidate_contracts_fail_closed() {
    assert!(SourceSpanRefV1::new("", 0, 1).is_err());
    assert!(SourceSpanRefV1::new("artifact", 1, 1).is_err());
    assert!(SourceSpanRefV1::new("artifact", 2, 1).is_err());

    let assertion =
        AssertionDraftV1::new("assertion", "general", "claim", vec![], vec![]).unwrap_err();
    assert!(assertion.contains("source span"));

    let missing = SourceSpanRefV1::new("artifact:missing", 0, 5).unwrap();
    let candidate = MemoryTransitionCandidateV1::new(
        "candidate-missing-source",
        vec![artifact("artifact:present", "claim")],
        vec![missing.clone()],
        vec![
            AssertionDraftV1::new("assertion", "general", "claim", vec![missing], vec![]).unwrap(),
        ],
        TransitionOperation::Append {
            assertion_id: "assertion".into(),
        },
        vec![],
    );
    assert!(candidate.unwrap_err().contains("missing source artifact"));

    let outside = SourceSpanRefV1::new("artifact:present", 0, 6).unwrap();
    let candidate = MemoryTransitionCandidateV1::new(
        "candidate-invalid-range",
        vec![artifact("artifact:present", "claim")],
        vec![outside.clone()],
        vec![
            AssertionDraftV1::new("assertion", "general", "claim", vec![outside], vec![]).unwrap(),
        ],
        TransitionOperation::Append {
            assertion_id: "assertion".into(),
        },
        vec![],
    );
    assert!(candidate.unwrap_err().contains("UTF-8 boundaries"));
}

#[tokio::test]
async fn exact_source_backed_candidate_commits_and_preserves_raw_evidence() {
    let (store, _tmp) = test_store();
    let authority = store.authority();
    let raw = "The verified launch date is 2026-07-10.";
    let candidate = append_candidate("candidate-exact", "artifact:launch", raw, raw);

    let outcome = authority
        .verify_and_commit(
            permit(AuthorityPermit::APPEND_CAPABILITY),
            "transition-exact".into(),
            candidate.clone(),
        )
        .await
        .unwrap();
    let MemoryTransitionOutcomeV1::Committed {
        verification,
        authority_receipt,
        ..
    } = outcome
    else {
        panic!("exact candidate must commit")
    };
    assert_eq!(
        verification.schema_version,
        "memory_transition_verification_v1"
    );
    assert_eq!(verification.disposition, TransitionDisposition::Commit);
    assert_eq!(verification.coverage.satisfied, verification.coverage.total);
    assert_eq!(
        verification.faithfulness.satisfied,
        verification.faithfulness.total
    );
    assert!(verification.omitted_spans.is_empty());
    assert!(verification.unsupported_spans.is_empty());
    assert_eq!(authority_receipt.after_epoch.0, 1);

    let current = store.list_facts("general", 10, 0).await.unwrap();
    assert_eq!(current.len(), 1);
    assert_eq!(current[0].content, raw);
    let record = authority
        .get_transition_by_idempotency_key("transition-exact")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(record.candidate, candidate);
    assert_eq!(record.candidate.source_artifacts[0].content, raw);
    assert_eq!(
        record.authority_receipt_id,
        Some(authority_receipt.receipt_id)
    );
    let overwrite = store
        .raw_execute(
            "UPDATE memory_transition_records SET candidate_json = '{}' WHERE caller_idempotency_key = ?1",
            vec!["transition-exact".into()],
        )
        .await;
    assert!(
        overwrite.is_err(),
        "raw transition evidence must be immutable"
    );
}

#[tokio::test]
async fn unsupported_and_omitted_spans_are_distinct_and_quarantined() {
    let (store, _tmp) = test_store();
    let authority = store.authority();
    let evidence = "supported statement";
    let unsupported = append_candidate(
        "candidate-unsupported",
        "artifact:one",
        evidence,
        "supported statement plus invention",
    );
    let outcome = authority
        .verify_and_commit(
            permit(AuthorityPermit::APPEND_CAPABILITY),
            "transition-unsupported".into(),
            unsupported,
        )
        .await
        .unwrap();
    let MemoryTransitionOutcomeV1::Quarantined { record } = outcome else {
        panic!("unsupported content must quarantine")
    };
    assert_eq!(
        record.verification.disposition,
        TransitionDisposition::Quarantine
    );
    assert!(!record.verification.unsupported_spans.is_empty());
    assert!(record.verification.omitted_spans.is_empty());
    assert!(store.list_facts("general", 10, 0).await.unwrap().is_empty());

    let mut omitted = append_candidate(
        "candidate-omitted",
        "artifact:two",
        "first required span; second required span",
        "first required span",
    );
    omitted.required_source_spans = vec![span("artifact:two", 0, 19), span("artifact:two", 21, 41)];
    omitted.assertions[0].source_spans = vec![span("artifact:two", 0, 19)];
    let outcome = authority
        .verify_and_commit(
            permit(AuthorityPermit::APPEND_CAPABILITY),
            "transition-omitted".into(),
            omitted,
        )
        .await
        .unwrap();
    let MemoryTransitionOutcomeV1::Quarantined { record } = outcome else {
        panic!("omitted required evidence must quarantine")
    };
    assert_eq!(record.verification.omitted_spans.len(), 1);
    assert!(record.verification.unsupported_spans.is_empty());
    assert!(store.list_facts("general", 10, 0).await.unwrap().is_empty());
}

#[tokio::test]
async fn active_head_preservation_and_dependency_effects_are_simulated_before_commit() {
    let (store, _tmp) = test_store();
    let authority = store.authority();
    let first = authority
        .append(
            permit(AuthorityPermit::APPEND_CAPABILITY),
            "seed-one".into(),
            "general".into(),
            "old service color".into(),
            Some("seed".into()),
        )
        .await
        .unwrap();
    let old_id = first.affected_ids[0].clone();
    let second = authority
        .append(
            permit(AuthorityPermit::APPEND_CAPABILITY),
            "seed-two".into(),
            "general".into(),
            "still valid region".into(),
            Some("seed".into()),
        )
        .await
        .unwrap();
    let preserved_id = second.affected_ids[0].clone();
    let dependency = store
        .add_graph_edge(
            &format!("fact:{preserved_id}"),
            &format!("fact:{old_id}"),
            semantic_memory::GraphEdgeType::Entity {
                relation: "depends_on".into(),
            },
            1.0,
            None,
        )
        .await
        .unwrap();

    let raw = "new service color";
    let source_span = span("artifact:color", 0, raw.len());
    let assertion = AssertionDraftV1::new(
        "replacement",
        "general",
        raw,
        vec![source_span.clone()],
        vec![preserved_id.clone()],
    )
    .unwrap();
    let candidate = MemoryTransitionCandidateV1::new(
        "candidate-replace",
        vec![artifact("artifact:color", raw)],
        vec![source_span],
        vec![assertion],
        TransitionOperation::Supersede {
            draft: SupersessionDraftV1::new(&old_id, "replacement").unwrap(),
        },
        vec![],
    )
    .unwrap();
    let outcome = authority
        .verify_and_commit(
            permit(AuthorityPermit::SUPERSEDE_CAPABILITY),
            "transition-preservation-fail".into(),
            candidate.clone(),
        )
        .await
        .unwrap();
    let MemoryTransitionOutcomeV1::Quarantined { record } = outcome else {
        panic!("unaccounted active assertion must quarantine")
    };
    assert_eq!(
        record.verification.preservation.satisfied + 1,
        record.verification.preservation.total
    );
    assert_eq!(
        record.verification.omitted_active_fact_ids,
        vec![preserved_id.clone()]
    );

    let mut preserved = candidate;
    preserved.candidate_id = "candidate-replace-preserved".into();
    preserved.preserved_active_fact_ids = vec![preserved_id.clone()];
    let outcome = authority
        .verify_and_commit(
            permit(AuthorityPermit::SUPERSEDE_CAPABILITY),
            "transition-preservation-pass".into(),
            preserved,
        )
        .await
        .unwrap();
    let MemoryTransitionOutcomeV1::Committed { verification, .. } = outcome else {
        panic!("explicitly preserved active assertion must commit")
    };
    assert!(verification.active_head_simulation.consistent);
    assert_eq!(
        verification.active_head_simulation.replaced_fact_id,
        Some(old_id)
    );
    assert!(verification
        .dependency_simulation
        .preserved_edge_ids
        .contains(&dependency.id));
}

#[tokio::test]
async fn source_backed_retraction_simulates_and_commits_through_existing_redaction_lane() {
    let (store, _tmp) = test_store();
    let authority = store.authority();
    let seeded = authority
        .append(
            permit(AuthorityPermit::APPEND_CAPABILITY),
            "seed-retract".into(),
            "general".into(),
            "sensitive assertion".into(),
            Some("seed".into()),
        )
        .await
        .unwrap();
    let target_id = seeded.affected_ids[0].clone();
    let evidence = "operator-approved retraction";
    let evidence_span = span("artifact:retraction", 0, evidence.len());
    let candidate = MemoryTransitionCandidateV1::new(
        "candidate-retraction",
        vec![artifact("artifact:retraction", evidence)],
        vec![evidence_span.clone()],
        vec![],
        TransitionOperation::Retract {
            target_fact_id: target_id.clone(),
            reason: evidence.into(),
            source_spans: vec![evidence_span],
        },
        vec![],
    )
    .unwrap();

    let outcome = authority
        .verify_and_commit(
            permit(AuthorityPermit::REDACT_CAPABILITY),
            "transition-retraction".into(),
            candidate,
        )
        .await
        .unwrap();
    let MemoryTransitionOutcomeV1::Committed { verification, .. } = outcome else {
        panic!("source-backed retraction must commit")
    };
    assert_eq!(
        verification.active_head_simulation.replaced_fact_id,
        Some(target_id)
    );
    assert!(verification.active_head_simulation.consistent);
    assert_eq!(
        store.list_facts("general", 10, 0).await.unwrap()[0].content,
        "[REDACTED]"
    );
}

#[tokio::test]
async fn quarantine_replay_is_stable_and_conflicting_retry_fails_closed() {
    let (store, _tmp) = test_store();
    let authority = store.authority();
    let candidate = append_candidate(
        "candidate-replay",
        "artifact:replay",
        "source",
        "unsupported",
    );
    let first = authority
        .verify_and_commit(
            permit(AuthorityPermit::APPEND_CAPABILITY),
            "transition-replay".into(),
            candidate.clone(),
        )
        .await
        .unwrap();
    let replay = authority
        .verify_and_commit(
            permit(AuthorityPermit::APPEND_CAPABILITY),
            "transition-replay".into(),
            candidate,
        )
        .await
        .unwrap();
    assert_eq!(first, replay);

    let conflict = authority
        .verify_and_commit(
            permit(AuthorityPermit::APPEND_CAPABILITY),
            "transition-replay".into(),
            append_candidate("different", "artifact:other", "other", "other"),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        conflict,
        semantic_memory::MemoryError::AuthorityIdempotencyConflict { .. }
    ));
}

#[tokio::test]
async fn fault_after_canonical_append_rolls_back_transition_and_evidence_record() {
    let (store, _tmp) = test_store();
    let authority = store.authority();
    authority.set_fault(Some(AuthorityFaultStage::AfterAppend));
    let result = authority
        .verify_and_commit(
            permit(AuthorityPermit::APPEND_CAPABILITY),
            "transition-fault".into(),
            append_candidate("candidate-fault", "artifact:fault", "fact", "fact"),
        )
        .await;
    assert!(matches!(
        result,
        Err(semantic_memory::MemoryError::AuthorityFaultInjected {
            stage: AuthorityFaultStage::AfterAppend
        })
    ));
    assert!(store.list_facts("general", 10, 0).await.unwrap().is_empty());
    assert!(authority
        .get_transition_by_idempotency_key("transition-fault")
        .await
        .unwrap()
        .is_none());
}
