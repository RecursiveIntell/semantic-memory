#!/usr/bin/env bash
set -euo pipefail

output="$(cargo test -p semantic-memory --features testing -- --list 2>&1)"
required=(
  authority_transactions
  transition_compiler
  forgetting_closure
  shadow_policy
  procedural_memory
)

for suite in "${required[@]}"; do
  if ! grep -q "tests/${suite}.rs" <<<"${output}"; then
    printf 'required governance suite missing from testing gate: %s\n' "${suite}" >&2
    exit 1
  fi
done

printf '%s\n' 'governance test gate contains all required suites'
