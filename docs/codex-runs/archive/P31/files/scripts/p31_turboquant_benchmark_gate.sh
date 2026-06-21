#!/usr/bin/env bash
set -euo pipefail

audit_dir="docs/audits/p31-turboquant-evidence-grade-retrieval-20260512"
mkdir -p "$audit_dir"

cargo run -p semantic-memory --features turbo-quant-codec --example turboquant_benchmark_gate -- \
  "$audit_dir/turboquant-benchmark-summary.json"

printf '%s\n' "$audit_dir/turboquant-benchmark-summary.json"
