// Example: run the RAGRouter-Bench and print the report + JSON manifest.
// Requires `--features benchmark` to compile.

#[cfg(feature = "benchmark")]
fn main() {
    let report = semantic_memory::benchmark::run_default_benchmark();
    let text = semantic_memory::benchmark::format_report(&report);
    println!("{}", text);

    let json = serde_json::to_string_pretty(&report).unwrap();
    println!("\n=== JSON Reproducibility Manifest ===\n{}", json);
}

#[cfg(not(feature = "benchmark"))]
fn main() {
    eprintln!("This example requires the `benchmark` feature. Run with:");
    eprintln!("  cargo run --features benchmark --example run_bench");
}