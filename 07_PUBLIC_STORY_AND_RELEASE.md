# Public Story and Release Layer

## Public thesis

Do not headline provenance. Use it as the spine.

Public line:

> **Local-first AI memory with receipts.**

Longer line:

> **A Rust memory substrate for agents and developers: source-grounded recall, compressed retrieval readiness, and replayable search/execution receipts.**

Career thesis:

> **I build autonomous AI systems that can prove what they did.**

## Flagship release target

The visible flagship should be Recall / semantic-memory, not an abstract provenance runtime.

### Demo requirement

A user should be able to:

1. Add/import memory.
2. Search memory.
3. Click “Why this result?”
4. See:
   - source trail,
   - embedding/search path,
   - index/backend used,
   - approximation/fallback status,
   - receipt/replay ID.

## README pitch for semantic-memory

```text
semantic-memory is a Rust substrate for trustworthy local-first AI memory: SQLite + FTS + vector search, episode identity, source-grounded imports, compressed-vector readiness, and receipt-backed retrieval. It is designed to support compressed vector codecs such as TurboQuant without letting indexes or compressed vectors become the source of truth.
```

## README pitch for TurboQuant integration

```text
TurboQuant integration turns semantic-memory into a compressed retrieval substrate: vectors are encoded deterministically from a codec profile, searched approximately, checked against raw reference scoring, and reported through search receipts so approximation is visible rather than hidden.
```

## Suggested public posts

### Post 1 — Why AI Memory Needs Receipts

Audience: product/AI leaders.

Outline:

1. AI memory fails when it cannot explain why it remembers something.
2. Source trails are table stakes.
3. Agent/tool execution creates memory and must leave receipts.
4. Approximate retrieval should disclose approximation/fallback.
5. Demo: result with receipt.

### Post 2 — Compressed Semantic Memory with TurboQuant in Rust

Audience: engineers.

Outline:

1. Problem: embedding storage/search cost.
2. Why deterministic data-oblivious codecs fit local-first memory.
3. Raw reference vs compressed scoring.
4. Rank drift and recall metrics.
5. Results table.

### Post 3 — Building Agents That Can Testify

Audience: infrastructure/security/agent teams.

Outline:

1. Agents need logs, but logs are not enough.
2. Receipts: tool calls, searches, imports, fallbacks.
3. Replay and exactness labels.
4. How this helps debugging, trust, and governance.

## What not to publish yet

Do not lead with:

- v11 constitutional runtime,
- hypergraph decoder kernel,
- lawful subtraction runtime,
- federated settlement,
- proof-governed promotion,
- broad causal attribution.

Those can stay in research/specs. The public artifact must ship first.

## Release checklist

- [x] P0 HNSW integrity fixed.
- [x] SearchContext deterministic ranking added.
- [x] Search receipt skeleton exists.
- [x] Raw reference scorer available.
- [x] VectorCodec abstraction exists.
- [x] TurboQuant backend feature-gated or clearly roadmap-only.
- [x] “Why this result?” API exists.
- [ ] Benchmark harness exists or roadmap is honest.
- [x] README uses product language.
- [ ] Portfolio/case-study copy ready.
