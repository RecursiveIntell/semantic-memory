use semantic_memory::{
    answer_policy_for, resolve_dependency_states, AnswerDisposition, MemoryConfig, MemoryStore,
    MockEmbedder, PremiseStatus, ResolvedMemoryAnswerV1, StateDependencyEdgeV1,
    StateResolutionMode, StateResolutionReceiptV1, StateView,
};
use tempfile::TempDir;

fn store() -> (MemoryStore, TempDir) {
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

#[test]
fn receipt_is_versioned_and_deterministic_and_declares_state_view() {
    let receipt = StateResolutionReceiptV1::new_at(
        "request-1",
        StateView::HistoricalAt("2026-07-10T12:00:00Z".into()),
        StateResolutionMode::HistoricalAt("2026-07-10T12:00:00Z".into()),
        PremiseStatus::Supported,
        AnswerDisposition::Answer,
        vec!["fact-1".into()],
        vec![],
        true,
        false,
        false,
        false,
        "2026-07-10T12:01:00Z",
    )
    .unwrap();
    let same = StateResolutionReceiptV1::new_at(
        "request-1",
        StateView::HistoricalAt("2026-07-10T12:00:00Z".into()),
        StateResolutionMode::HistoricalAt("2026-07-10T12:00:00Z".into()),
        PremiseStatus::Supported,
        AnswerDisposition::Answer,
        vec!["fact-1".into()],
        vec![],
        true,
        false,
        false,
        false,
        "2026-07-10T12:01:00Z",
    )
    .unwrap();
    assert_eq!(receipt.schema_version, "state_resolution_receipt_v1");
    assert_eq!(
        receipt.state_view,
        StateView::HistoricalAt("2026-07-10T12:00:00Z".into())
    );
    assert_eq!(receipt.receipt_digest, same.receipt_digest);
    assert_eq!(
        serde_json::from_str::<StateResolutionReceiptV1>(&serde_json::to_string(&receipt).unwrap())
            .unwrap(),
        receipt
    );
}

#[test]
fn premise_statuses_have_explicit_safe_dispositions() {
    assert_eq!(
        answer_policy_for(PremiseStatus::Supported, true, false, false, false).disposition,
        AnswerDisposition::Answer
    );
    assert_eq!(
        answer_policy_for(PremiseStatus::Stale, true, false, false, false).disposition,
        AnswerDisposition::CorrectPremise
    );
    assert_eq!(
        answer_policy_for(PremiseStatus::Contradicted, true, false, false, false).disposition,
        AnswerDisposition::DiscloseConflict
    );
    assert_eq!(
        answer_policy_for(PremiseStatus::Unsupported, false, false, false, false).disposition,
        AnswerDisposition::RequestEvidence
    );
    assert_eq!(
        answer_policy_for(PremiseStatus::Ambiguous, true, true, false, false).disposition,
        AnswerDisposition::DiscloseConflict
    );
    for decision in [
        answer_policy_for(PremiseStatus::Unsupported, false, false, false, false),
        answer_policy_for(PremiseStatus::Ambiguous, true, true, false, false),
        answer_policy_for(PremiseStatus::Supported, true, false, true, false),
        answer_policy_for(PremiseStatus::Supported, true, false, false, true),
    ] {
        assert!(!decision.confident_negative);
        assert_ne!(decision.disposition, AnswerDisposition::Answer);
    }
}

#[test]
fn dependency_closure_is_typed_and_fails_closed_on_cycles_and_conflicting_heads() {
    let edges = vec![
        StateDependencyEdgeV1::invalidates("new", "old"),
        StateDependencyEdgeV1::derived_from_state("derived", "new"),
        StateDependencyEdgeV1::requires_reevaluation("derived", "old"),
    ];
    let resolution = resolve_dependency_states(&edges, &["old".into(), "derived".into()], 16);
    assert_eq!(
        resolution.status("old"),
        Some(semantic_memory::DependencyState::Invalid)
    );
    assert!(resolution.status("derived").is_some());
    assert!(!resolution.invalid_lineage);

    let cycle = vec![
        StateDependencyEdgeV1::derived_from_state("a", "b"),
        StateDependencyEdgeV1::derived_from_state("b", "a"),
    ];
    let cyclic = resolve_dependency_states(&cycle, &["a".into()], 16);
    assert!(cyclic.invalid_lineage);
    assert_eq!(cyclic.premise_status, PremiseStatus::Ambiguous);
}

#[test]
fn local_stale_fixture_marks_replaced_premise_without_confident_negative() {
    // STALE-style local fixture: a later state invalidates an earlier assertion.
    let edges = vec![StateDependencyEdgeV1::invalidates("state:new", "state:old")];
    let resolution = resolve_dependency_states(&edges, &["state:old".into()], 8);
    let decision = answer_policy_for(
        resolution.premise_status,
        true,
        resolution.unresolved_conflict,
        resolution.invalid_lineage,
        resolution.budget_exhausted,
    );
    assert_eq!(decision.disposition, AnswerDisposition::CorrectPremise);
    assert!(!decision.confident_negative);
}

#[test]
fn local_a_tma_and_memtrace_fixtures_keep_time_and_trajectory_explicit() {
    // A-TMA/LTP-style local temporal fixture.
    let historical = StateResolutionMode::HistoricalAt("2026-01-01T00:00:00Z".into());
    assert_eq!(
        historical.state_view(),
        StateView::HistoricalAt("2026-01-01T00:00:00Z".into())
    );

    // MemTrace-style local trajectory fixture: the transition request is explicit and does not
    // silently collapse into a current answer.
    let trajectory = StateResolutionMode::Trajectory {
        points: vec!["2026-01-01T00:00:00Z".into(), "2026-02-01T00:00:00Z".into()],
    };
    assert_eq!(trajectory.state_view(), StateView::IncludeSuperseded);
    assert_ne!(historical, trajectory);
}

#[tokio::test]
async fn resolved_answer_uses_existing_store_and_always_carries_receipt() {
    let (store, _tmp) = store();
    let _ = store
        .add_fact(
            "general",
            "Rust was first released in 2015",
            Some("fixture:stale-a-tma-memtrace-local"),
            None,
        )
        .await
        .unwrap();

    let answer: ResolvedMemoryAnswerV1 = store
        .resolve_memory(
            "when was Rust released",
            Some(4),
            None,
            StateResolutionMode::Current,
            16,
        )
        .await
        .unwrap();
    assert_eq!(answer.state_view, StateView::Current);
    assert_eq!(answer.receipt.schema_version, "state_resolution_receipt_v1");
    assert_eq!(answer.receipt.state_view, answer.state_view);
    assert!(!answer.receipt.receipt_digest.is_empty());
    // Raw fixture insertion does not advance governed authority state; the witness must report
    // the actual epoch rather than inventing a governed mutation.
    assert_eq!(
        answer.retrieval_witness.retrieval_epoch,
        semantic_memory::RetrievalEpoch(0)
    );
    assert_eq!(
        answer.retrieval_witness.ordered_result_ids,
        answer
            .assertions
            .iter()
            .map(|assertion| assertion.memory_id.clone())
            .collect::<Vec<_>>()
    );
    assert!(answer.answer.is_some());
}

#[tokio::test]
async fn dependency_edges_use_existing_graph_storage_and_mark_stale_results() {
    let (store, _tmp) = store();
    let old = store
        .add_fact(
            "general",
            "service color was blue",
            Some("fixture:stale"),
            None,
        )
        .await
        .unwrap();
    let new = store
        .add_fact(
            "general",
            "service color is green",
            Some("fixture:stale"),
            None,
        )
        .await
        .unwrap();
    store
        .add_state_dependency_edge(
            StateDependencyEdgeV1::invalidates(format!("fact:{new}"), format!("fact:{old}")),
            1.0,
        )
        .await
        .unwrap();

    let answer = store
        .resolve_memory(
            "service color was blue",
            Some(4),
            None,
            StateResolutionMode::Current,
            16,
        )
        .await
        .unwrap();
    assert!(answer
        .assertions
        .iter()
        .any(|assertion| assertion.premise_status == PremiseStatus::Stale));
    assert_eq!(answer.answer_disposition, AnswerDisposition::CorrectPremise);
}
