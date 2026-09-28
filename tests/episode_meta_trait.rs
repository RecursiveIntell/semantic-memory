//! The mirror exposes the owner-owned EpisodeMeta equality trait without changing its wire form.
use semantic_memory::{EpisodeMeta, EpisodeOutcome, VerificationStatus};
use serde_json::{json, Value};

#[test]
fn episode_meta_equality_and_serialization_are_distinct_contracts(
) -> Result<(), Box<dyn std::error::Error>> {
    let meta = EpisodeMeta {
        cause_ids: vec!["cause:1".to_owned()],
        effect_type: "test_failure".to_owned(),
        outcome: EpisodeOutcome::Pending,
        confidence: 0.5,
        verification_status: VerificationStatus::Unverified,
        experiment_id: None,
        valid_time: None,
        fact_digest: None,
    };
    let expected: Value = json!({
        "cause_ids": ["cause:1"],
        "effect_type": "test_failure",
        "outcome": "pending",
        "confidence": 0.5,
        "verification_status": {"status": "unverified"},
        "experiment_id": null,
        "valid_time": null,
        "fact_digest": null
    });
    assert_eq!(serde_json::to_value(&meta)?, expected);
    let restored: EpisodeMeta = serde_json::from_value(expected)?;
    assert_eq!(meta, restored);
    let mut changed = restored.clone();
    changed.effect_type = "regression".to_owned();
    assert_ne!(meta, changed);
    Ok(())
}
