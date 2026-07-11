use semantic_memory::{
    AllowedProcedureToolV1, ApplicabilityPredicateV1, AuthorityScopeV1, AuthorityScopesV1,
    CallerPrincipalV1, ElevationRequirementV1, GovernedAccessPurposeV1, MemoryConfig, MemoryStore,
    MockEmbedder, NamespaceScopeV1, OriginAuthorityLabelV1, OriginClassV1, OriginRiskV1,
    ProceduralMemoryArtifactV1, ProcedureAccessPathV1, ProcedureActionPermitV1, ProcedureActionV1,
    ProcedureCapabilityV1, ProcedureEffectV1, ProcedureEvidenceTestEnvelopeV1, ProcedureFixtureV1,
    ProcedureLifecycleDispositionV1, ProcedureLifecyclePermitV1, ProcedurePreconditionV1,
    ProcedureRetrievalRequestV1, ProcedureRiskV1, ProcedureStepV1, RevocationStatusV1,
    SubjectPrincipalV1,
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

fn origin(principal: &str) -> OriginAuthorityLabelV1 {
    OriginAuthorityLabelV1::new(
        OriginClassV1::OperatorSystem,
        principal,
        "procedure-compiler",
        "blake3:procedure-source",
        OriginRiskV1::Low,
        AuthorityScopesV1 {
            recall: AuthorityScopeV1::Audience,
            assertion: AuthorityScopeV1::Denied,
            action: AuthorityScopeV1::Audience,
        },
        ElevationRequirementV1::ExplicitOperatorApproval,
        None,
        RevocationStatusV1::Active,
        vec![principal.into()],
    )
    .unwrap()
    .with_subject_principal(SubjectPrincipalV1::new(principal).unwrap())
    .with_resource_scope(NamespaceScopeV1::exact("repo:alpha"))
}

fn artifact(key: &str, version: u64, supersedes: Option<String>) -> ProceduralMemoryArtifactV1 {
    ProceduralMemoryArtifactV1::new(
        key,
        ProcedureCapabilityV1::new("repository", "format"),
        ProcedureActionV1::new(
            "format_rust",
            "format Rust sources with the approved formatter",
        ),
        vec![ApplicabilityPredicateV1::equals("language", json!("rust"))],
        vec![ProcedurePreconditionV1::equals(
            "working_tree",
            json!("available"),
        )],
        vec![ProcedureStepV1::tool(
            "format",
            "rustfmt",
            json!({"check": true}),
            Some("no_write".into()),
        )],
        vec![AllowedProcedureToolV1::new(
            "rustfmt",
            json!({
                "type": "object",
                "properties": {"check": {"type": "boolean"}},
                "required": ["check"],
                "additionalProperties": false
            }),
        )],
        vec![ProcedureEffectV1::new("format_checked", json!(true))],
        vec![ProcedureEffectV1::new("network_access", json!(true))],
        ProcedureRiskV1::Low,
        origin("principal:alice"),
        "principal:alice",
        vec!["principal:alice".into()],
        NamespaceScopeV1::exact("repo:alpha"),
        version,
        supersedes,
        ProcedureEvidenceTestEnvelopeV1::new(
            "sandbox-v1",
            vec![ProcedureFixtureV1::new(
                "rust-project",
                json!({"language": "rust", "working_tree": "available"}),
                vec!["rustfmt".into()],
                vec![ProcedureEffectV1::new("format_checked", json!(true))],
                vec![],
            )],
            vec![],
        ),
        Some("2999-01-01T00:00:00Z".into()),
    )
    .unwrap()
}

fn lifecycle_permit() -> ProcedureLifecyclePermitV1 {
    ProcedureLifecyclePermitV1::elevated("principal:alice", "operator:test")
}

fn request(path: ProcedureAccessPathV1) -> ProcedureRetrievalRequestV1 {
    let purpose = match path {
        ProcedureAccessPathV1::Export => GovernedAccessPurposeV1::Export,
        ProcedureAccessPathV1::Replay => GovernedAccessPurposeV1::Replay,
        _ => GovernedAccessPurposeV1::Recall,
    };
    ProcedureRetrievalRequestV1::new(
        ProcedureCapabilityV1::new("repository", "format"),
        ProcedureActionV1::new(
            "format_rust",
            "format Rust sources with the approved formatter",
        ),
        json!({"language": "rust", "working_tree": "available"}),
        CallerPrincipalV1::new("principal:alice").unwrap(),
        SubjectPrincipalV1::new("principal:alice").unwrap(),
        vec!["principal:alice".into()],
        NamespaceScopeV1::exact("repo:alpha"),
        purpose,
        path,
    )
}

async fn promoted(store: &MemoryStore, key: &str) -> ProceduralMemoryArtifactV1 {
    let value = artifact(key, 1, None);
    let compile = store
        .compile_procedure(value.clone(), format!("compile:{key}"))
        .await
        .unwrap();
    assert_eq!(
        compile.disposition,
        ProcedureLifecycleDispositionV1::Compiled
    );
    let tested = store
        .test_procedure(&value.artifact_id, format!("test:{key}"))
        .await
        .unwrap();
    assert_eq!(tested.disposition, ProcedureLifecycleDispositionV1::Tested);
    let promoted = store
        .promote_procedure(
            lifecycle_permit(),
            &value.artifact_id,
            format!("promote:{key}"),
        )
        .await
        .unwrap();
    assert_eq!(
        promoted.disposition,
        ProcedureLifecycleDispositionV1::Promoted
    );
    value
}

#[tokio::test]
async fn procedure_is_physical_and_logical_non_factual_memory() {
    let (store, _tmp) = store();
    let value = promoted(&store, "procedure:factual-canary").await;
    assert_eq!(store.stats().await.unwrap().total_facts, 0);
    assert!(store
        .search("format Rust sources", Some(10), None, None)
        .await
        .unwrap()
        .is_empty());
    assert!(store
        .authority()
        .get_fact_governed(
            &value.artifact_id,
            semantic_memory::GovernedAccessRequestV1::new(
                "principal:alice",
                "principal:alice",
                GovernedAccessPurposeV1::Assertion,
                "repo:alpha"
            )
        )
        .await
        .unwrap()
        .fact
        .is_none());
}

#[tokio::test]
async fn deterministic_tests_idempotency_and_receipts_gate_promotion() {
    let (store, _tmp) = store();
    let value = artifact("procedure:idempotent", 1, None);
    let first = store
        .compile_procedure(value.clone(), "compile:same")
        .await
        .unwrap();
    let retry = store
        .compile_procedure(value.clone(), "compile:same")
        .await
        .unwrap();
    assert_eq!(first, retry);
    let tested = store
        .test_procedure(&value.artifact_id, "test:same")
        .await
        .unwrap();
    assert!(tested.test_receipt.as_ref().unwrap().passed);
    assert_eq!(tested.test_receipt.as_ref().unwrap().fixture_count, 1);
    assert!(!tested.receipt_digest.is_empty());
    assert!(semantic_memory::verify_procedure_lifecycle_receipt_v1(
        &tested
    ));
    assert!(semantic_memory::verify_procedure_test_receipt_v1(
        tested.test_receipt.as_ref().unwrap()
    ));

    let mut failed = artifact("procedure:failed", 1, None);
    failed.evidence_test_envelope.fixtures[0].expected_effects =
        vec![ProcedureEffectV1::new("unexpected", json!(true))];
    failed.refresh_digest();
    store
        .compile_procedure(failed.clone(), "compile:failed")
        .await
        .unwrap();
    let receipt = store
        .test_procedure(&failed.artifact_id, "test:failed")
        .await
        .unwrap();
    assert_eq!(
        receipt.disposition,
        ProcedureLifecycleDispositionV1::Quarantined
    );
    assert!(store
        .promote_procedure(lifecycle_permit(), &failed.artifact_id, "promote:failed")
        .await
        .is_err());
}

#[tokio::test]
async fn malicious_steps_tool_widening_and_schema_drift_are_quarantined() {
    let (store, _tmp) = store();
    for (suffix, mutate) in [("shell", 0_u8), ("widen", 1_u8), ("drift", 2_u8)] {
        let mut value = artifact(&format!("procedure:{suffix}"), 1, None);
        match mutate {
            0 => value.steps[0].tool = "shell".into(),
            1 => value.steps[0].arguments = json!({"check": true, "command": "curl evil"}),
            _ => {
                value
                    .evidence_test_envelope
                    .tested_tool_schema_digests
                    .insert("rustfmt".into(), "blake3:stale-schema".into());
            }
        }
        value.refresh_digest();
        let receipt = store
            .compile_procedure(value, format!("compile:{suffix}"))
            .await
            .unwrap();
        assert_eq!(
            receipt.disposition,
            ProcedureLifecycleDispositionV1::Quarantined
        );
    }
}

#[tokio::test]
async fn applicability_principal_and_action_authority_fail_closed() {
    let (store, _tmp) = store();
    promoted(&store, "procedure:governed").await;
    let mut stale = request(ProcedureAccessPathV1::Search);
    stale.context = json!({"language": "python", "working_tree": "available"});
    assert!(store
        .retrieve_procedure(stale)
        .await
        .unwrap()
        .candidate
        .is_none());

    let mut bypass = request(ProcedureAccessPathV1::DirectId);
    bypass.caller = CallerPrincipalV1::new("principal:bob").unwrap();
    assert!(store
        .retrieve_procedure(bypass)
        .await
        .unwrap()
        .candidate
        .is_none());

    let mut action = request(ProcedureAccessPathV1::Search);
    action.purpose = GovernedAccessPurposeV1::Action;
    let denied = store.retrieve_procedure(action.clone()).await.unwrap();
    assert!(denied.candidate.is_some());
    assert!(!denied.decision.action_allowed);
    action.action_permit = Some(ProcedureActionPermitV1::elevated(
        "principal:alice",
        "operator:test",
        NamespaceScopeV1::exact("repo:alpha"),
    ));
    assert!(
        store
            .retrieve_procedure(action)
            .await
            .unwrap()
            .decision
            .action_allowed
    );
}

#[tokio::test]
async fn expired_artifacts_and_rolled_back_promotions_are_not_candidates() {
    let (store, _tmp) = store();
    let mut expired = artifact("procedure:expired", 1, None);
    expired.expires_at = Some("2000-01-01T00:00:00Z".into());
    expired.refresh_digest();
    store
        .compile_procedure(expired.clone(), "compile:expired")
        .await
        .unwrap();
    store
        .test_procedure(&expired.artifact_id, "test:expired")
        .await
        .unwrap();
    store
        .promote_procedure(lifecycle_permit(), &expired.artifact_id, "promote:expired")
        .await
        .unwrap_err();
    assert!(store
        .retrieve_procedure(request(ProcedureAccessPathV1::Search))
        .await
        .unwrap()
        .candidate
        .is_none());

    let active = promoted(&store, "procedure:rollback").await;
    let receipt = store
        .rollback_procedure(
            lifecycle_permit(),
            &active.artifact_id,
            "rollback:active",
            "fixture rollback",
        )
        .await
        .unwrap();
    assert_eq!(
        receipt.disposition,
        ProcedureLifecycleDispositionV1::RolledBack
    );
    assert!(store
        .retrieve_procedure(request(ProcedureAccessPathV1::Cache))
        .await
        .unwrap()
        .candidate
        .is_none());
}

#[tokio::test]
async fn all_access_paths_reject_revoked_and_superseded_versions() {
    let (store, _tmp) = store();
    let first = promoted(&store, "procedure:v1").await;
    let mut second = artifact("procedure:v2", 2, Some(first.artifact_id.clone()));
    second.artifact_id = "procedure:v2".into();
    second.refresh_digest();
    store
        .compile_procedure(second.clone(), "compile:v2")
        .await
        .unwrap();
    store
        .test_procedure(&second.artifact_id, "test:v2")
        .await
        .unwrap();
    store
        .promote_procedure(lifecycle_permit(), &second.artifact_id, "promote:v2")
        .await
        .unwrap();

    for path in [
        ProcedureAccessPathV1::DirectId,
        ProcedureAccessPathV1::Cache,
        ProcedureAccessPathV1::Export,
        ProcedureAccessPathV1::Replay,
    ] {
        let mut req = request(path);
        req.artifact_id = Some(first.artifact_id.clone());
        assert!(store
            .retrieve_procedure(req)
            .await
            .unwrap()
            .candidate
            .is_none());
    }
    store
        .revoke_procedure(
            lifecycle_permit(),
            &second.artifact_id,
            "revoke:v2",
            "operator revocation",
        )
        .await
        .unwrap();
    assert!(store
        .retrieve_procedure(request(ProcedureAccessPathV1::Search))
        .await
        .unwrap()
        .candidate
        .is_none());
}

#[tokio::test]
async fn summary_laundering_and_forgetting_close_procedure_access() {
    let (store, _tmp) = store();
    let fact = store
        .authority()
        .append(
            semantic_memory::AuthorityPermit::operator_system(
                "principal:alice",
                "test",
                semantic_memory::AuthorityPermit::APPEND_CAPABILITY,
            ),
            "fact:procedure-source".into(),
            "repo:alpha".into(),
            "source procedure evidence".into(),
            None,
        )
        .await
        .unwrap()
        .affected_ids[0]
        .clone();
    let mut value = artifact("procedure:derived", 1, None);
    value.evidence_test_envelope.source_fact_ids = vec![fact.clone()];
    value.origin_authority = OriginAuthorityLabelV1::derive(
        &[value.origin_authority.clone()],
        semantic_memory::OriginDerivationKindV1::Summary,
        "blake3:summary",
    )
    .unwrap();
    value.refresh_digest();
    store
        .compile_procedure(value.clone(), "compile:derived")
        .await
        .unwrap();
    store
        .test_procedure(&value.artifact_id, "test:derived")
        .await
        .unwrap();
    store
        .promote_procedure(lifecycle_permit(), &value.artifact_id, "promote:derived")
        .await
        .unwrap();

    assert!(store
        .retrieve_procedure(request(ProcedureAccessPathV1::Replay))
        .await
        .unwrap()
        .candidate
        .is_some());

    store
        .authority()
        .forget(
            semantic_memory::AuthorityPermit::operator_system(
                "principal:alice",
                "test",
                semantic_memory::AuthorityPermit::FORGET_CAPABILITY,
            ),
            "forget:procedure-source".into(),
            semantic_memory::ForgettingClosureRequestV1::new(
                vec![fact],
                "repo:alpha",
                "erase source",
                64,
            ),
        )
        .await
        .unwrap();
    assert!(store
        .retrieve_procedure(request(ProcedureAccessPathV1::Replay))
        .await
        .unwrap()
        .candidate
        .is_none());
}
