use semantic_memory::hostile_benchmark::{
    evaluate_receipt, BenchmarkReceipt, MetricThresholds, ScenarioReceipt, ScenarioStatus,
};

fn scenario(name: &str, status: ScenarioStatus) -> ScenarioReceipt {
    ScenarioReceipt {
        name: name.into(),
        status,
        latency_us: 1,
        detail: String::new(),
    }
}

#[test]
fn not_tested_is_excluded_and_counts_are_self_consistent() {
    let mut receipt = BenchmarkReceipt::new("subject", MetricThresholds::declared());
    receipt.scenarios = vec![
        scenario("pass", ScenarioStatus::Pass),
        scenario("fail", ScenarioStatus::Fail),
        scenario("api-gap", ScenarioStatus::NotTested),
    ];
    evaluate_receipt(&mut receipt);
    assert_eq!(receipt.summary.tested, 2);
    assert_eq!(receipt.summary.passed, 1);
    assert_eq!(receipt.summary.failed, 1);
    assert_eq!(receipt.summary.not_tested, 1);
    assert_eq!(receipt.summary.pass_rate, 0.5);
    assert!(!receipt.summary.thresholds_met);
}

#[test]
fn perfect_tested_receipt_meets_predeclared_threshold() {
    let mut receipt = BenchmarkReceipt::new("subject", MetricThresholds::declared());
    receipt.scenarios = vec![
        scenario("a", ScenarioStatus::Pass),
        scenario("gap", ScenarioStatus::NotTested),
    ];
    receipt.metrics.replay_equivalent = true;
    evaluate_receipt(&mut receipt);
    assert_eq!(receipt.thresholds.minimum_pass_rate, 1.0);
    assert!(receipt.summary.thresholds_met);
}

#[test]
fn empty_receipt_never_passes() {
    let mut receipt = BenchmarkReceipt::new("subject", MetricThresholds::declared());
    evaluate_receipt(&mut receipt);
    assert_eq!(receipt.summary.pass_rate, 0.0);
    assert!(!receipt.summary.thresholds_met);
}
