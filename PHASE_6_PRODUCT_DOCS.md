# Phase 6 - Product/docs/portfolio pass

## Objective

Make the architecture visible and market-readable without drowning users in internal doctrine.

## Public thesis

Use this language:

> **Local-first AI memory with receipts.**

Longer:

> semantic-memory is a Rust substrate for local-first AI memory: SQLite/FTS/vector search, episode identity, source-grounded projection imports, receipt-ready retrieval, and deterministic compressed-vector codecs such as TurboQuant.

TurboQuant integration pitch:

> TurboQuant integration turns semantic-memory into a compressed retrieval substrate: vectors are encoded deterministically from a codec profile, searched approximately, checked against raw reference scoring, and reported through search receipts so approximation is visible rather than hidden.

## Docs to add/update

```text
README.md
semantic-memory/README.md
docs/giga-pass/ARCHITECTURE.md
docs/giga-pass/RECEIPTS.md
docs/giga-pass/TURBOQUANT_INTEGRATION.md
docs/giga-pass/CONFORMANCE_AND_BENCHMARKS.md
```

## README sections

1. What this is
2. Why receipts matter
3. Architecture diagram
4. Quick start
5. Search modes
6. Compression modes
7. Why this result? receipt example
8. Conformance/release gates
9. Non-goals

## Minimal product demo script

The demo should answer:

```text
1. Add memory/source.
2. Generate embedding.
3. Search memory.
4. Show result.
5. Ask "why this result?"
6. Display source, embedding receipt, search backend, approximate/exact status, fallback/degradation, replay handle.
7. Switch TurboQuant profile on.
8. Show drift/receipt difference vs raw reference.
```

## What not to put in public docs yet

Avoid leading with:

- v11/v12 spec language
- constitutional runtime
- hypergraph decoder
- lawful subtraction
- generalized bitemporal database theory
- full causal attribution stack

Mention only if there is a working demo.

## Codex prompt

```text
Run Phase 6: product/docs/portfolio pass.

Update README/docs so the project is legible as "local-first AI memory with receipts" and "compressed retrieval with visible approximation." Do not expose internal spec sprawl as the main pitch. Include architecture diagrams, quick-start examples, search receipt examples, TurboQuant integration explanation, and conformance/release gates.

Create a demo script or documented scenario showing: ingest/source -> embedding -> search -> why-this-result receipt -> optional TurboQuant compressed search -> drift/fallback visibility.

Do not add major implementation in this phase unless docs reveal a small missing API example.
```

---
