# Release gates

The five authority/governance integration suites declared with `required-features = ["testing"]`
are release-critical. Run the checked executable gate:

```bash
bash scripts/check_governance_test_gate.sh
cargo test --features testing
```

The script fails when any of `authority_transactions`, `transition_compiler`,
`forgetting_closure`, `shadow_policy`, or `procedural_memory` is absent from Cargo's executable
test list. This documents a local/release gate; it does not assert that hosted CI executed it.
