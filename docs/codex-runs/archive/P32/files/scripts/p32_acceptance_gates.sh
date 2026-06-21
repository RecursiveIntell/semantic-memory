#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-docs/audits/p32-research-max-retrieval-runtime-20260513}"
mkdir -p "$OUT_DIR"
: > "$OUT_DIR/gates.tsv"

run_gate() {
  local name="$1"; shift
  echo "== $name =="
  echo "$*" > "$OUT_DIR/${name}.cmd"
  if "$@" > "$OUT_DIR/${name}.log" 2>&1; then
    echo -e "$name\tPASS" >> "$OUT_DIR/gates.tsv"
  else
    echo -e "$name\tFAIL" >> "$OUT_DIR/gates.tsv"
  fi
}

run_gate cargo_fmt cargo fmt --all --check
run_gate cargo_check_workspace cargo check --workspace --all-targets --all-features
run_gate cargo_test_workspace cargo test --workspace --all-features
run_gate cargo_clippy_workspace cargo clippy --workspace --all-targets --all-features -- -D warnings
run_gate turbo_quant_tests cargo test -p turbo-quant
run_gate turbo_quant_wire cargo test -p turbo-quant wire_format
run_gate turbo_quant_malformed cargo test -p turbo-quant malformed_artifacts
run_gate semantic_memory_turbo_check cargo check -p semantic-memory --features turbo-quant-codec --all-targets
run_gate semantic_memory_turbo_tests cargo test -p semantic-memory --features turbo-quant-codec search_tests::turbo_quant -- --nocapture
run_gate semantic_memory_vector_codec cargo test -p semantic-memory --features turbo-quant-codec --test vector_codec -- --nocapture
run_gate semantic_memory_hnsw_persistence cargo test -p semantic-memory --features hnsw --test hnsw_persistence -- --nocapture

if [[ -x scripts/p32_retrieval_benchmark_gate.sh ]]; then
  run_gate p32_benchmark scripts/p32_retrieval_benchmark_gate.sh
elif [[ -x scripts/p31_turboquant_benchmark_gate.sh ]]; then
  run_gate p31_benchmark_legacy scripts/p31_turboquant_benchmark_gate.sh
else
  echo -e "benchmark_gate\tMISSING" >> "$OUT_DIR/gates.tsv"
fi

cat "$OUT_DIR/gates.tsv"
