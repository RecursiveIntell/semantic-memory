use semantic_memory::{
    compare_shadow_execution_v1, evaluate_shadow_policy_promotion_v1, ActiveShadowPolicyV1,
    AuthorityFaultStage, MemoryConfig, MemoryStore, MockEmbedder, PromotionDecisionReceiptV1,
    PromotionDispositionV1, PromotionEvidenceV1, ShadowEvaluationWindowV1, ShadowPolicyKindV1,
    ShadowPolicyPromotionPermitV1, ShadowPolicyProposalV1, ShadowPolicyProvenanceV1,
    ShadowPolicyRiskV1, ShadowPolicyStatusV1,
};
use serde_json::json;
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

fn proposal(principal: &str, key: &str) -> ShadowPolicyProposalV1 {
    ShadowPolicyProposalV1::new(
        ShadowPolicyKindV1::Routing,
        principal,
        ShadowPolicyProvenanceV1::new("offline-evaluator", "fixture:held-out-v1"),
        ShadowEvaluationWindowV1::new(
            "2026-07-01T00:00:00Z",
            "2026-07-05T00:00:00Z",
            "2026-07-06T00:00:00Z",
            "2026-07-09T00:00:00Z",
            "blake3:held-out-inputs",
        ),
        "blake3:features",
        json!({"rerank_fine": 0.5}),
        json!({"rerank_fine": 0.1}),
        ShadowPolicyRiskV1::new(0.1, vec!["bounded-test-risk"]),
        "2999-01-01T00:00:00Z",
        key,
    )
}

fn evidence(proposal: &ShadowPolicyProposalV1) -> PromotionEvidenceV1 {
    PromotionEvidenceV1::new(
        proposal,
        0.08,
        false,
        false,
        json!({"n": 100, "quality": 0.91}),
        vec!["blake3:held-out-inputs".into()],
        "none",
    )
}

fn permit(principal: &str) -> ShadowPolicyPromotionPermitV1 {
    ShadowPolicyPromotionPermitV1::elevated(principal, "operator:shadow", "shadow-policy-test")
}

#[tokio::test]
async fn proposal_is_shadow_only_and_idempotent() {
    let (store, _tmp) = store();
    let p = proposal("tenant:a", "proposal-1");
    let first = store
        .submit_shadow_policy_proposal(p.clone())
        .await
        .unwrap();
    let retry = store.submit_shadow_policy_proposal(p).await.unwrap();
    assert_eq!(first, retry);
    assert_eq!(store.stats().await.unwrap().total_facts, 0);
    assert!(store
        .get_active_shadow_policy("tenant:a", ShadowPolicyKindV1::Routing)
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn conflicting_retry_and_principal_isolation_fail_closed() {
    let (store, _tmp) = store();
    store
        .submit_shadow_policy_proposal(proposal("tenant:a", "proposal-1"))
        .await
        .unwrap();
    let mut conflict = proposal("tenant:a", "proposal-1");
    conflict.proposed_delta = json!({"rerank_fine": 0.9});
    assert!(store.submit_shadow_policy_proposal(conflict).await.is_err());
    assert!(store
        .get_shadow_policy_proposal("proposal-1", "tenant:b")
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn promotion_requires_verified_evidence_and_versions_active_policy() {
    let (store, _tmp) = store();
    let p = proposal("tenant:a", "proposal-1");
    store
        .submit_shadow_policy_proposal(p.clone())
        .await
        .unwrap();
    let receipt = store
        .promote_shadow_policy(
            permit("tenant:a"),
            "promote-1",
            p.proposal_id.clone(),
            evidence(&p),
        )
        .await
        .unwrap();
    assert_eq!(receipt.disposition, PromotionDispositionV1::Promoted);
    assert_eq!(receipt.status, ShadowPolicyStatusV1::Promoted);
    let active: ActiveShadowPolicyV1 = store
        .get_active_shadow_policy("tenant:a", ShadowPolicyKindV1::Routing)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(active.version, 1);
    assert_eq!(active.policy["rerank_fine"], json!(0.6));
}

#[tokio::test]
async fn poisoned_fabricated_stale_and_unbounded_proposals_do_not_promote() {
    let (store, _tmp) = store();
    let mut poisoned = proposal("tenant:a", "poisoned");
    poisoned.risk = ShadowPolicyRiskV1::new(0.99, vec!["poison"]);
    poisoned.proposal_digest = poisoned.compute_digest();
    store
        .submit_shadow_policy_proposal(poisoned.clone())
        .await
        .unwrap();
    let mut fabricated = evidence(&poisoned);
    fabricated.held_out_improvement = 999.0;
    let receipt = store
        .promote_shadow_policy(
            permit("tenant:a"),
            "promote-poisoned",
            poisoned.proposal_id,
            fabricated,
        )
        .await
        .unwrap();
    assert!(matches!(
        receipt.disposition,
        PromotionDispositionV1::Rejected
            | PromotionDispositionV1::Quarantined
            | PromotionDispositionV1::Deferred
    ));
    assert!(store
        .get_active_shadow_policy("tenant:a", ShadowPolicyKindV1::Routing)
        .await
        .unwrap()
        .is_none());

    let mut stale = proposal("tenant:a", "stale");
    stale.expires_at = "2000-01-01T00:00:00Z".into();
    stale.proposal_digest = stale.compute_digest();
    store
        .submit_shadow_policy_proposal(stale.clone())
        .await
        .unwrap();
    let receipt = store
        .promote_shadow_policy(
            permit("tenant:a"),
            "promote-stale",
            stale.proposal_id.clone(),
            evidence(&stale),
        )
        .await
        .unwrap();
    assert_eq!(receipt.disposition, PromotionDispositionV1::Expired);
}

#[tokio::test]
async fn promotion_retry_is_idempotent_and_conflicting_retry_is_rejected() {
    let (store, _tmp) = store();
    let p = proposal("tenant:a", "proposal-1");
    store
        .submit_shadow_policy_proposal(p.clone())
        .await
        .unwrap();
    let first = store
        .promote_shadow_policy(
            permit("tenant:a"),
            "promote-1",
            p.proposal_id.clone(),
            evidence(&p),
        )
        .await
        .unwrap();
    let retry = store
        .promote_shadow_policy(
            permit("tenant:a"),
            "promote-1",
            p.proposal_id.clone(),
            evidence(&p),
        )
        .await
        .unwrap();
    assert_eq!(first, retry);
    let mut changed = evidence(&p);
    changed.held_out_improvement = 0.2;
    assert!(store
        .promote_shadow_policy(permit("tenant:a"), "promote-1", p.proposal_id, changed)
        .await
        .is_err());
}

#[tokio::test]
async fn rollback_restores_previous_version_without_touching_memory() {
    let (store, _tmp) = store();
    let first = proposal("tenant:a", "p1");
    store
        .submit_shadow_policy_proposal(first.clone())
        .await
        .unwrap();
    store
        .promote_shadow_policy(
            permit("tenant:a"),
            "promote-1",
            first.proposal_id.clone(),
            evidence(&first),
        )
        .await
        .unwrap();
    let mut second = proposal("tenant:a", "p2");
    second.baseline_policy = json!({"rerank_fine": 0.6});
    second.proposed_delta = json!({"rerank_fine": 0.2});
    second.baseline_policy_digest =
        semantic_memory::shadow_policy::shadow_policy_digest(&second.baseline_policy);
    second.proposal_digest = second.compute_digest();
    store
        .submit_shadow_policy_proposal(second.clone())
        .await
        .unwrap();
    let mut second_evidence = evidence(&second);
    second_evidence.rollback_target = "1".into();
    store
        .promote_shadow_policy(
            permit("tenant:a"),
            "promote-2",
            second.proposal_id,
            second_evidence,
        )
        .await
        .unwrap();
    let receipt = store
        .rollback_shadow_policy(
            permit("tenant:a"),
            "rollback-1",
            "tenant:a",
            ShadowPolicyKindV1::Routing,
            1,
        )
        .await
        .unwrap();
    assert_eq!(receipt.disposition, PromotionDispositionV1::RolledBack);
    assert_eq!(
        store
            .get_active_shadow_policy("tenant:a", ShadowPolicyKindV1::Routing)
            .await
            .unwrap()
            .unwrap()
            .version,
        1
    );
    assert_eq!(store.stats().await.unwrap().total_facts, 0);
}

#[tokio::test]
async fn promotion_fault_rolls_back_active_version_and_receipt() {
    let (store, _tmp) = store();
    let p = proposal("tenant:a", "faulty");
    store
        .submit_shadow_policy_proposal(p.clone())
        .await
        .unwrap();
    store
        .authority()
        .set_fault(Some(AuthorityFaultStage::AfterShadowPromotion));
    let error = store
        .promote_shadow_policy(
            permit("tenant:a"),
            "promote-fault",
            p.proposal_id.clone(),
            evidence(&p),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        semantic_memory::MemoryError::AuthorityFaultInjected {
            stage: AuthorityFaultStage::AfterShadowPromotion
        }
    ));
    assert!(store
        .get_active_shadow_policy("tenant:a", ShadowPolicyKindV1::Routing)
        .await
        .unwrap()
        .is_none());
    assert!(store
        .get_shadow_policy_promotion_receipt("promote-fault", "tenant:a")
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn fabricated_metrics_and_stale_evidence_are_not_admitted() {
    let (store, _tmp) = store();
    let p = proposal("tenant:a", "metrics");
    store
        .submit_shadow_policy_proposal(p.clone())
        .await
        .unwrap();
    let mut fabricated = evidence(&p);
    fabricated.metrics = json!({"n": 1, "quality": 1000.0});
    let receipt = store
        .promote_shadow_policy(
            permit("tenant:a"),
            "promote-metrics",
            p.proposal_id.clone(),
            fabricated,
        )
        .await
        .unwrap();
    assert_eq!(receipt.disposition, PromotionDispositionV1::Quarantined);

    let p = proposal("tenant:a", "old-evidence");
    store
        .submit_shadow_policy_proposal(p.clone())
        .await
        .unwrap();
    let mut stale = evidence(&p);
    stale.evaluated_at = "2000-01-01T00:00:00Z".into();
    let decision = evaluate_shadow_policy_promotion_v1(&p, &stale, &permit("tenant:a"), None);
    assert_eq!(decision.disposition, PromotionDispositionV1::Deferred);
}

#[tokio::test]
async fn all_policy_kinds_remain_shadow_only_until_promoted() {
    let (store, _tmp) = store();
    for (index, kind) in [
        ShadowPolicyKindV1::Routing,
        ShadowPolicyKindV1::WriteAdmission,
        ShadowPolicyKindV1::Retention,
        ShadowPolicyKindV1::RerankWeights,
    ]
    .into_iter()
    .enumerate()
    {
        let mut p = proposal("tenant:a", &format!("kind-{index}"));
        p.policy_kind = kind;
        p.proposal_digest = p.compute_digest();
        store.submit_shadow_policy_proposal(p).await.unwrap();
    }
    assert_eq!(store.stats().await.unwrap().total_facts, 0);
    for kind in [
        ShadowPolicyKindV1::Routing,
        ShadowPolicyKindV1::WriteAdmission,
        ShadowPolicyKindV1::Retention,
        ShadowPolicyKindV1::RerankWeights,
    ] {
        assert!(store
            .get_active_shadow_policy("tenant:a", kind)
            .await
            .unwrap()
            .is_none());
    }
}

#[tokio::test]
async fn shadow_ledger_rows_are_append_only() {
    let (store, _tmp) = store();
    let p = proposal("tenant:a", "append-only");
    store
        .submit_shadow_policy_proposal(p.clone())
        .await
        .unwrap();
    assert!(store
        .raw_execute(
            "UPDATE shadow_policy_proposals SET status = 'promoted' WHERE proposal_id = ?1",
            vec![p.proposal_id.clone()],
        )
        .await
        .is_err());
    assert!(store
        .raw_execute(
            "DELETE FROM shadow_policy_proposals WHERE proposal_id = ?1",
            vec![p.proposal_id],
        )
        .await
        .is_err());
}

#[test]
fn shadow_proposals_persist_across_store_reopen() {
    let tmp = TempDir::new().unwrap();
    let config = MemoryConfig {
        base_dir: tmp.path().to_path_buf(),
        ..Default::default()
    };
    let p = proposal("tenant:a", "persisted");
    let store =
        MemoryStore::open_with_embedder(config.clone(), Box::new(MockEmbedder::new(768))).unwrap();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime
        .block_on(store.submit_shadow_policy_proposal(p.clone()))
        .unwrap();
    drop(store);
    let reopened =
        MemoryStore::open_with_embedder(config, Box::new(MockEmbedder::new(768))).unwrap();
    let loaded = runtime
        .block_on(reopened.get_shadow_policy_proposal(&p.proposal_id, "tenant:a"))
        .unwrap()
        .unwrap();
    assert_eq!(loaded.proposal_digest, p.proposal_digest);
}

#[test]
fn shadow_execution_comparison_has_no_serving_effect() {
    let comparison = compare_shadow_execution_v1(
        "tenant:a",
        ShadowPolicyKindV1::Routing,
        "blake3:inputs",
        vec![(
            "case-1",
            json!({"route": "bm25"}),
            json!({"route": "hybrid"}),
        )],
    )
    .unwrap();
    assert!(!comparison.served);
    assert!(!comparison.canonical_mutation);
    assert_eq!(comparison.changed_cases, 1);
}

#[test]
fn receipts_are_versioned_and_serde_stable() {
    assert_eq!(
        PromotionDecisionReceiptV1::SCHEMA_VERSION,
        "promotion_decision_receipt_v1"
    );
}
