#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== P32 active run =="
grep -n "P32_RESEARCH_MAX_RETRIEVAL_RUNTIME" docs/codex-runs/CURRENT_RUN.md

echo "== Public claim scan =="
claim_hits="$(grep -RIn \
  --exclude-dir=archive \
  "zero accuracy loss\|default-ready\|production-ready\|no dataset-specific calibration\|ICLR 2026\|AISTATS 2026\|AAAI 2025" \
  ../turbo-quant README.md docs || true)"
unsafe_claim_hits="$(printf '%s\n' "$claim_hits" | grep -v "Removed public" | grep -v "Do not claim" || true)"
if [ -n "$unsafe_claim_hits" ]; then
  printf '%s\n' "$unsafe_claim_hits" >&2
  echo "Unsafe active public claim text remains." >&2
  exit 1
fi

echo "== TurboQuant wire gates =="
(cd .. && cargo test -p turbo-quant --test wire_format --test malformed_artifacts)

echo "== semantic-memory feature check =="
(cd .. && cargo check -p semantic-memory --features turbo-quant-codec --all-targets)

echo "P32 runtime gates passed."
