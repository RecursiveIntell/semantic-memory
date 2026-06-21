//! Adaptive routing benchmark harness.
//!
//! Measures routing quality: does the router make the RIGHT decision
//! about which stages to run? Compares adaptive routing vs static
//! (always-run-all-stages) on quality, latency, and token efficiency.
//!
//! This is a synthetic benchmark — it doesn't require a real corpus.
//! It uses synthetic result sets with known ground truth to measure
//! whether the router's decisions improve outcomes.
//!
//! Behind `#[cfg(feature = "routing")]`.

use serde::{Deserialize, Serialize};
use std::time::Instant;

// ─── Benchmark types ────────────────────────────────────────────────────

/// A single benchmark case: query + expected ground truth.
#[derive(Debug, Clone)]
pub struct BenchmarkCase {
    /// The query string.
    pub query: String,
    /// Ground truth: which stages SHOULD be activated for this query?
    pub expected: ExpectedRouting,
    /// Simulated results if all stages are run (quality score 0..1).
    pub full_retrieval_quality: f64,
    /// Simulated results if no retrieval is used (quality score 0..1).
    pub no_retrieval_quality: f64,
    /// Simulated latency if all stages run (ms).
    pub full_retrieval_latency_ms: u64,
    /// Simulated latency if no stages run (ms).
    pub no_retrieval_latency_ms: u64,
    /// Simulated token cost if all stages run.
    pub full_retrieval_tokens: usize,
    /// Simulated token cost if no stages run.
    pub no_retrieval_tokens: usize,
}

/// Expected routing decision for a benchmark case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedRouting {
    pub bm25_coarse: bool,
    pub vector_medium: bool,
    pub rerank_fine: bool,
    pub graph_expansion: bool,
    pub decoder: bool,
    pub discord: bool,
    pub no_retrieval: bool,
}

impl ExpectedRouting {
    /// Check if a routing decision matches this expected routing.
    pub fn matches(&self, actual: &crate::routing::RoutingDecision) -> bool {
        self.bm25_coarse == actual.bm25_coarse
            && self.vector_medium == actual.vector_medium
            && self.rerank_fine == actual.rerank_fine
            && self.graph_expansion == actual.graph_expansion
            && self.decoder == actual.decoder
            && self.discord == actual.discord
            && self.no_retrieval == actual.no_retrieval
    }
}

/// Result of a single benchmark case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseResult {
    pub query: String,
    pub routing_correct: bool,
    /// Quality gained by adaptive routing vs full retrieval.
    /// Positive = adaptive is better; negative = full is better.
    pub quality_delta: f64,
    /// Latency saved by adaptive routing (ms).
    /// Positive = adaptive is faster; negative = adaptive is slower.
    pub latency_saved_ms: f64,
    /// Tokens saved by adaptive routing.
    pub tokens_saved: usize,
    pub reasoning: String,
}

/// Aggregate benchmark report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub total_cases: usize,
    pub routing_accuracy: f64,
    /// Average quality delta (positive = adaptive better).
    pub avg_quality_delta: f64,
    /// Average latency saved (ms, positive = adaptive faster).
    pub avg_latency_saved_ms: f64,
    /// Average tokens saved (positive = adaptive saves).
    pub avg_tokens_saved: f64,
    /// Number of cases where adaptive routing was correct.
    pub correct_routes: usize,
    /// Number of cases where adaptive routing was incorrect.
    pub incorrect_routes: usize,
    /// Number of cases where adaptive routing skipped retrieval when it shouldn't have.
    pub retrieval_underuse: usize,
    /// Number of cases where adaptive routing used retrieval when it shouldn't have.
    pub retrieval_overuse: usize,
    /// Per-case results.
    pub cases: Vec<CaseResult>,
    /// Wall-clock time to run the benchmark.
    pub elapsed_ms: u64,
}

// ─── Benchmark suite ────────────────────────────────────────────────────

/// The default benchmark suite: 12 cases covering routing edge cases.
pub fn default_suite() -> Vec<BenchmarkCase> {
    vec![
        // 1. Short greeting — should skip retrieval.
        BenchmarkCase {
            query: "hi".to_string(),
            expected: ExpectedRouting {
                bm25_coarse: false,
                vector_medium: false,
                rerank_fine: false,
                graph_expansion: false,
                decoder: false,
                discord: false,
                no_retrieval: true,
            },
            full_retrieval_quality: 0.1,
            no_retrieval_quality: 0.9,
            full_retrieval_latency_ms: 100,
            no_retrieval_latency_ms: 1,
            full_retrieval_tokens: 500,
            no_retrieval_tokens: 10,
        },
        // 2. Normal factual query — should use BM25 + vector + rerank.
        BenchmarkCase {
            query: "what is the architecture of semantic memory".to_string(),
            expected: ExpectedRouting {
                bm25_coarse: true,
                vector_medium: true,
                rerank_fine: true,
                graph_expansion: false,
                decoder: false,
                discord: false,
                no_retrieval: false,
            },
            full_retrieval_quality: 0.95,
            no_retrieval_quality: 0.3,
            full_retrieval_latency_ms: 350,
            no_retrieval_latency_ms: 1,
            full_retrieval_tokens: 300,
            no_retrieval_tokens: 10,
        },
        // 3. Contradiction query — should enable decoder.
        BenchmarkCase {
            query: "compare rust vs python performance differences".to_string(),
            expected: ExpectedRouting {
                bm25_coarse: true,
                vector_medium: true,
                rerank_fine: true,
                graph_expansion: false,
                decoder: true,
                discord: false,
                no_retrieval: false,
            },
            full_retrieval_quality: 0.85,
            no_retrieval_quality: 0.2,
            full_retrieval_latency_ms: 450,
            no_retrieval_latency_ms: 1,
            full_retrieval_tokens: 400,
            no_retrieval_tokens: 10,
        },
        // 4. Provenance query — should note provenance in reasoning.
        BenchmarkCase {
            query: "what is the source of the turbo-quant compression algorithm".to_string(),
            expected: ExpectedRouting {
                bm25_coarse: true,
                vector_medium: true,
                rerank_fine: true,
                graph_expansion: false,
                decoder: false,
                discord: false,
                no_retrieval: false,
            },
            full_retrieval_quality: 0.9,
            no_retrieval_quality: 0.25,
            full_retrieval_latency_ms: 350,
            no_retrieval_latency_ms: 1,
            full_retrieval_tokens: 350,
            no_retrieval_tokens: 10,
        },
        // 5. Temporal query — should note temporal in reasoning.
        BenchmarkCase {
            query: "what are the latest developments in vector search".to_string(),
            expected: ExpectedRouting {
                bm25_coarse: true,
                vector_medium: true,
                rerank_fine: true,
                graph_expansion: false,
                decoder: false,
                discord: false,
                no_retrieval: false,
            },
            full_retrieval_quality: 0.88,
            no_retrieval_quality: 0.2,
            full_retrieval_latency_ms: 350,
            no_retrieval_latency_ms: 1,
            full_retrieval_tokens: 320,
            no_retrieval_tokens: 10,
        },
        // 6. Entity query with dense corpus — should enable graph expansion + discord.
        BenchmarkCase {
            query: "how does Semantic-Memory integrate with Turbo-Quant".to_string(),
            expected: ExpectedRouting {
                bm25_coarse: true,
                vector_medium: true,
                rerank_fine: true,
                graph_expansion: true,
                decoder: false,
                discord: true,
                no_retrieval: false,
            },
            full_retrieval_quality: 0.92,
            no_retrieval_quality: 0.15,
            full_retrieval_latency_ms: 450,
            no_retrieval_latency_ms: 1,
            full_retrieval_tokens: 400,
            no_retrieval_tokens: 10,
        },
        // 7. Entity query with discord enabled — should enable graph expansion + discord.
        BenchmarkCase {
            query: "how does AiDENs work with Recall".to_string(),
            expected: ExpectedRouting {
                bm25_coarse: true,
                vector_medium: true,
                rerank_fine: true,
                graph_expansion: true,
                decoder: false,
                discord: true,
                no_retrieval: false,
            },
            full_retrieval_quality: 0.9,
            no_retrieval_quality: 0.2,
            full_retrieval_latency_ms: 400,
            no_retrieval_latency_ms: 1,
            full_retrieval_tokens: 350,
            no_retrieval_tokens: 10,
        },
        // 8. Broad query — 3 tokens, specificity = 0.15 >= 0.15, so vector IS on (borderline).
        BenchmarkCase {
            query: "a b c".to_string(),
            expected: ExpectedRouting {
                bm25_coarse: true,
                vector_medium: true,
                rerank_fine: true,
                graph_expansion: false,
                decoder: false,
                discord: false,
                no_retrieval: false,
            },
            full_retrieval_quality: 0.3,
            no_retrieval_quality: 0.1,
            full_retrieval_latency_ms: 350,
            no_retrieval_latency_ms: 1,
            full_retrieval_tokens: 300,
            no_retrieval_tokens: 10,
        },
        // 9. Complex multi-intent query — all features (has entities, contradiction, provenance, temporal).
        BenchmarkCase {
            query: "compare the latest source evidence for Rust vs Python".to_string(),
            expected: ExpectedRouting {
                bm25_coarse: true,
                vector_medium: true,
                rerank_fine: true,
                graph_expansion: true,
                decoder: true,
                discord: true,
                no_retrieval: false,
            },
            full_retrieval_quality: 0.93,
            no_retrieval_quality: 0.15,
            full_retrieval_latency_ms: 500,
            no_retrieval_latency_ms: 1,
            full_retrieval_tokens: 450,
            no_retrieval_tokens: 10,
        },
        // 10. Very short non-retrieval query.
        BenchmarkCase {
            query: "ok".to_string(),
            expected: ExpectedRouting {
                bm25_coarse: false,
                vector_medium: false,
                rerank_fine: false,
                graph_expansion: false,
                decoder: false,
                discord: false,
                no_retrieval: true,
            },
            full_retrieval_quality: 0.05,
            no_retrieval_quality: 0.95,
            full_retrieval_latency_ms: 100,
            no_retrieval_latency_ms: 1,
            full_retrieval_tokens: 500,
            no_retrieval_tokens: 5,
        },
        // 11. Single word — 1 token < min_query_length(3) → no_retrieval.
        BenchmarkCase {
            query: "turbo-quant".to_string(),
            expected: ExpectedRouting {
                bm25_coarse: false,
                vector_medium: false,
                rerank_fine: false,
                graph_expansion: false,
                decoder: false,
                discord: false,
                no_retrieval: true,
            },
            full_retrieval_quality: 0.4,
            no_retrieval_quality: 0.15,
            full_retrieval_latency_ms: 350,
            no_retrieval_latency_ms: 1,
            full_retrieval_tokens: 300,
            no_retrieval_tokens: 10,
        },
        // 12. Long specific query — all stages.
        BenchmarkCase {
            query: "what is the exact mechanism by which the provenance semiring combines confidence scores across multiple retrieval stages in the semantic memory system".to_string(),
            expected: ExpectedRouting {
                bm25_coarse: true,
                vector_medium: true,
                rerank_fine: true,
                graph_expansion: false,
                decoder: false,
                discord: false,
                no_retrieval: false,
            },
            full_retrieval_quality: 0.97,
            no_retrieval_quality: 0.1,
            full_retrieval_latency_ms: 350,
            no_retrieval_latency_ms: 1,
            full_retrieval_tokens: 300,
            no_retrieval_tokens: 10,
        },
    ]
}

// ─── Benchmark runner ───────────────────────────────────────────────────

/// Run the benchmark suite with a given router configuration.
pub fn run_benchmark(
    router: &crate::routing::RetrievalRouter,
    cases: &[BenchmarkCase],
) -> BenchmarkReport {
    let start = Instant::now();
    let mut results = Vec::with_capacity(cases.len());
    let mut correct = 0usize;
    let mut incorrect = 0usize;
    let mut underuse = 0usize;
    let mut overuse = 0usize;
    let mut total_quality_delta = 0.0;
    let mut total_latency_saved = 0.0f64;
    let mut total_tokens_saved = 0.0f64;

    for case in cases {
        let decision = router.route_query(&case.query);
        let routing_correct = case.expected.matches(&decision);

        if routing_correct {
            correct += 1;
        } else {
            incorrect += 1;
        }

        // Detect retrieval underuse/overuse.
        let actual_no_retrieval = decision.no_retrieval;
        let expected_no_retrieval = case.expected.no_retrieval;
        if expected_no_retrieval && !actual_no_retrieval {
            overuse += 1;
        }
        if !expected_no_retrieval && actual_no_retrieval {
            underuse += 1;
        }

        // Simulate quality/latency/tokens based on routing decision.
        let (quality, latency, tokens) = if decision.no_retrieval {
            (case.no_retrieval_quality, case.no_retrieval_latency_ms, case.no_retrieval_tokens)
        } else {
            // Partial savings from skipping stages.
            let stages_active = [
                decision.bm25_coarse,
                decision.vector_medium,
                decision.rerank_fine,
                decision.graph_expansion,
            ].iter().filter(|&&b| b).count() as f64;
            let total_stages = 4.0;
            let stage_ratio = stages_active / total_stages;
            let quality = case.full_retrieval_quality * stage_ratio.max(0.3);
            let latency = (case.full_retrieval_latency_ms as f64 * stage_ratio) as u64;
            let tokens = (case.full_retrieval_tokens as f64 * stage_ratio) as usize;
            (quality, latency, tokens)
        };

        // Quality delta: adaptive quality vs full retrieval quality.
        let quality_delta = quality - case.full_retrieval_quality;
        let latency_saved = case.full_retrieval_latency_ms as f64 - latency as f64;
        let tokens_saved = case.full_retrieval_tokens as f64 - tokens as f64;

        total_quality_delta += quality_delta;
        total_latency_saved += latency_saved;
        total_tokens_saved += tokens_saved;

        results.push(CaseResult {
            query: case.query.clone(),
            routing_correct,
            quality_delta,
            latency_saved_ms: latency_saved,
            tokens_saved: tokens_saved as usize,
            reasoning: decision.reasoning,
        });
    }

    let n = cases.len() as f64;
    BenchmarkReport {
        total_cases: cases.len(),
        routing_accuracy: correct as f64 / n,
        avg_quality_delta: total_quality_delta / n,
        avg_latency_saved_ms: total_latency_saved / n,
        avg_tokens_saved: total_tokens_saved / n,
        correct_routes: correct,
        incorrect_routes: incorrect,
        retrieval_underuse: underuse,
        retrieval_overuse: overuse,
        cases: results,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }
}

/// Run the benchmark with a router configured for all features enabled.
pub fn run_default_benchmark() -> BenchmarkReport {
    let router = crate::routing::RetrievalRouter {
        decoder_enabled: true,
        discord_enabled: true,
        corpus_density: 0.7,
        ..Default::default()
    };
    let cases = default_suite();
    run_benchmark(&router, &cases)
}

/// Generate a human-readable benchmark summary.
pub fn format_report(report: &BenchmarkReport) -> String {
    let mut out = String::new();
    out.push_str("=== RAGRouter-Bench Report ===\n\n");
    out.push_str(&format!("Total cases:      {}\n", report.total_cases));
    out.push_str(&format!("Routing accuracy: {:.1}% ({} correct, {} incorrect)\n",
        report.routing_accuracy * 100.0,
        report.correct_routes,
        report.incorrect_routes,
    ));
    out.push_str(&format!("Avg quality delta: {:.4} (positive = adaptive better)\n", report.avg_quality_delta));
    out.push_str(&format!("Avg latency saved: {:.1} ms\n", report.avg_latency_saved_ms));
    out.push_str(&format!("Avg tokens saved:  {:.1}\n", report.avg_tokens_saved));
    out.push_str(&format!("Retrieval underuse: {} (should have retrieved, didn't)\n", report.retrieval_underuse));
    out.push_str(&format!("Retrieval overuse:  {} (shouldn't have retrieved, did)\n", report.retrieval_overuse));
    out.push_str(&format!("Benchmark elapsed:  {} ms\n\n", report.elapsed_ms));
    out.push_str("--- Per-case results ---\n");
    for (i, case) in report.cases.iter().enumerate() {
        let status = if case.routing_correct { "OK" } else { "MISS" };
        out.push_str(&format!(
            "{}. [{}] q=\"{}\" dq={:.3} dl={:.0}ms dt={}\n",
            i + 1, status, case.query, case.quality_delta, case.latency_saved_ms, case.tokens_saved
        ));
    }
    out
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_runs_all_cases() {
        let report = run_default_benchmark();
        assert_eq!(report.total_cases, 12, "default suite has 12 cases");
    }

    #[test]
    fn benchmark_routing_accuracy_above_threshold() {
        let report = run_default_benchmark();
        assert!(
            report.routing_accuracy >= 0.75,
            "routing accuracy should be >= 75%, got {:.1}%",
            report.routing_accuracy * 100.0
        );
    }

    #[test]
    fn benchmark_latency_saved_positive() {
        let report = run_default_benchmark();
        // On average, adaptive routing should save latency (skip stages).
        assert!(
            report.avg_latency_saved_ms > 0.0,
            "adaptive routing should save latency on average, got {:.1}ms",
            report.avg_latency_saved_ms
        );
    }

    #[test]
    fn benchmark_tokens_saved_positive() {
        let report = run_default_benchmark();
        assert!(
            report.avg_tokens_saved > 0.0,
            "adaptive routing should save tokens on average, got {:.1}",
            report.avg_tokens_saved
        );
    }

    #[test]
    fn benchmark_no_retrieval_underuse() {
        let report = run_default_benchmark();
        // No cases where the router skipped retrieval when it should have used it.
        assert_eq!(
            report.retrieval_underuse, 0,
            "no retrieval underuse expected (router should not skip retrieval when needed)"
        );
    }

    #[test]
    fn benchmark_report_is_serializable() {
        let report = run_default_benchmark();
        let json = serde_json::to_string(&report).unwrap();
        let back: BenchmarkReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back.total_cases, report.total_cases);
    }

    #[test]
    fn benchmark_format_report_has_content() {
        let report = run_default_benchmark();
        let text = format_report(&report);
        assert!(text.contains("RAGRouter-Bench Report"));
        assert!(text.contains("Routing accuracy"));
        assert!(text.contains("Per-case results"));
    }

    #[test]
    fn benchmark_short_query_routes_correctly() {
        let report = run_default_benchmark();
        // Case 1: "hi" should be routed to no_retrieval.
        let case1 = &report.cases[0];
        assert!(case1.routing_correct, "short query should route correctly");
        assert!(case1.latency_saved_ms > 0.0, "short query should save latency");
    }

    #[test]
    fn benchmark_contradiction_query_routes_correctly() {
        let report = run_default_benchmark();
        // Case 3: contradiction query should enable decoder.
        let case3 = &report.cases[2];
        assert!(case3.routing_correct, "contradiction query should route correctly");
    }
}