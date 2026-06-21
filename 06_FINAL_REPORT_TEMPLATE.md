# Codex Pass Final Report Template

## Pass identity

- Pass name:
- Branch/worktree:
- Starting commit:
- Ending commit:
- Date/time:
- Agent/model:

## Objective

State the exact packet or issue IDs addressed.

## Summary

Short statement of what changed and why.

## Files changed

| File | Change type | Notes |
|---|---|---|
| | | |

## Tests added or changed

| Test | Purpose | Expected failure before fix |
|---|---|---|
| | | |

## Commands run

| Command | Result | Notes |
|---|---|---|
| `cargo fmt --all --check` | pass/fail/not run | |
| `cargo check --workspace --all-targets --all-features` | pass/fail/not run | |
| `cargo test --workspace --all-features` | pass/fail/not run | |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` | pass/fail/not run | |
| `cargo doc --workspace --all-features --no-deps` | pass/fail/not run | |

## Issue closure

| Issue ID | Status | Evidence |
|---|---|---|
| GP- | fixed/not_a_bug/deferred | |

## Receipt/degradation behavior

Describe any new receipt, explanation, fallback, degradation, or diagnostic behavior.

## Authority impact

Confirm:

- [ ] No derived artifact became authoritative truth.
- [ ] Raw/reference path remains available where required.
- [ ] Sidecars/indexes remain rebuildable.
- [ ] Boundary defaults are explicit where touched.

## TurboQuant eligibility

One of:

- Not eligible yet.
- Prototype eligible.
- Default retrieval eligible.

Rationale:

## Remaining risks

List concrete unresolved risks, not vague caveats.

## Next pass recommendation

Name the next packet(s) to run.
