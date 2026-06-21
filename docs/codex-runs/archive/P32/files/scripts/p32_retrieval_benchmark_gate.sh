#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

out="docs/audits/p32-research-max-retrieval-runtime-20260513/turboquant-smoke-benchmark-summary.json"
(cd .. && cargo run -p semantic-memory --features turbo-quant-codec --example turboquant_benchmark_gate -- "semantic-memory/${out}")

classification="$(python - "$out" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
print(data.get("classification", "missing"))
PY
)"

if [ "$classification" != "green" ]; then
  echo "P32 smoke benchmark classification: ${classification}" >&2
  exit 1
fi

echo "P32 smoke benchmark gate passed: ${out}"
