# Pass Plan and Dependency Graph

## North-star release definition

A successful giga-pass produces a semantic-memory stack that can say:

```text
For any search result, I can explain:
  what source/memory row it came from,
  how it was embedded,
  which vector/index path produced it,
  whether approximation or fallback occurred,
  which codec/index profile was used,
  and how to rebuild or replay the path.
```

TurboQuant becomes eligible only when this substrate is stable.

## Tier model

### Tier 0 — Governance and proof discipline

Purpose: prevent clever work from creating hidden authority.

Deliverables:

- `AGENTS.md`
- audit ledger
- status JSON
- pass report template
- CI/acceptance commands

Exit gate:

```text
Codex knows what not to do, and every subsequent pass has a report target.
```

### Tier 1 — P0 semantic-memory integrity

Purpose: make current vector/index/search paths lawful.

Primary risks addressed:

- count-only HNSW parity,
- shallow sidecar validation,
- persisted `usize`,
- filtered HNSW under-return,
- pending mutation ambiguity,
- invalid vectors reaching storage/indexing.

Exit gate:

```text
The system can prove HNSW/keymap/database coherence or rebuild/degrade safely.
```

### Tier 2 — Replay-clean search and receipts

Purpose: make search explainable and deterministic under a supplied context.

Primary risks addressed:

- hidden wall-clock ranking,
- fallback invisibility,
- approximate retrieval invisibility,
- explainable search not durable enough for future receipts.

Exit gate:

```text
A search can emit a receipt-like object with evaluation time, backend, candidate counts, fallbacks, exact rerank state, and result IDs.
```

### Tier 3 — Boundary hardening

Purpose: prevent bridge/import/export from inventing meaning.

Primary risks addressed:

- `unwrap_or_default` at semantic boundaries,
- permissive JSON import behavior,
- silent missing fields,
- legacy compat masking malformed input.

Exit gate:

```text
Boundary defaults are explicit policy decisions, typed errors, or degradation records.
```

### Tier 4 — Codec abstraction

Purpose: prepare for TurboQuant without letting compression become truth.

Primary risks addressed:

- overloading `embedding_q8`,
- lack of codec profile identity,
- lack of raw reference scorer,
- lack of profile mismatch rejection.

Exit gate:

```text
RawF32 and SQ8 go through the same codec interface; profile digests are stable; mismatch fails closed.
```

### Tier 5 — TurboQuant backend

Purpose: add compressed retrieval as a derived codec path.

Primary risks addressed:

- premature fusion,
- hidden rank drift,
- approximate scores used as if exact,
- missing codec identity.

Exit gate:

```text
TurboQuant can encode/score deterministically, report rank drift vs raw reference, and emit search receipts.
```

### Tier 6 — Product-facing trust surface

Purpose: turn provenance into visible value.

Exit gate:

```text
API/UI/docs can answer “Why this result?” without exposing internal doctrine.
```

### Tier 7 — Public release and portfolio

Purpose: convert technical work into career-visible proof.

Exit gate:

```text
README, demo, benchmark, and case-study language present the stack as local-first AI memory with receipts.
```

## Dependency graph

```text
Tier 0
  ├── Tier 1A HNSW key parity
  ├── Tier 1B sidecar format/rebuild
  ├── Tier 1C vector validation
  └── Tier 1D filtered fallback
        ↓
Tier 2 deterministic search + receipts
        ↓
Tier 3 boundary/defaulting cleanup
        ↓
Tier 4 codec abstraction
        ↓
Tier 5 TurboQuant backend
        ↓
Tier 6 product-facing receipt API
        ↓
Tier 7 public release/docs
```

## Parallelization plan

Codex can run multiple tasks in separate worktrees, but avoid overlapping files.

### Safe-ish parallel lanes after baseline

| Lane | Focus | Likely files | Can run parallel with |
|---|---|---|---|
| A | HNSW key parity | `hnsw.rs`, `hnsw_ops.rs`, tests | C, D |
| B | HNSW sidecar header/rebuild | `hnsw.rs`, `hnsw_ops.rs` | C, D but not A without coordination |
| C | SearchContext/recency | `search.rs`, `lib.rs`, tests | A, B, D |
| D | Boundary/defaulting | forge/bridge/import files | A, B, C |
| E | Codec abstraction | `quantize.rs`, new codec module | after A-C green |
| F | TurboQuant backend | new optional feature/module | after E green |
| G | Docs/product story | README/docs | any time, but final after code gates |

### Avoid parallel overlap

Do not run these at the same time unless merging manually:

- A and B both changing HNSW persistence internals.
- C and E both changing search result/scoring types.
- E and F both changing codec tables/traits.
- D and forge bridge schema changes while import tests are being rewritten.

## Phase gates

### Gate G0 — Toolchain and baseline

```bash
cargo --version
rustc --version
cargo fmt --all --check
cargo check --workspace --all-targets --all-features
cargo test --workspace --all-features
```

### Gate G1 — Vector/index truth

Must pass:

- key-level parity tests,
- stale key tests,
- swapped key tests,
- wrong-domain key tests,
- corrupted/old sidecar tests,
- filtered fallback tests.

### Gate G2 — Search replay and receipts

Must pass:

- frozen evaluation time tests,
- receipt emitted tests,
- fallback/degradation receipt tests,
- exact/rerank disclosure tests.

### Gate G3 — Boundary discipline

Must pass:

- ugly import tests,
- missing-field tests,
- defaulting policy tests,
- bridge no-invention tests.

### Gate G4 — Codec abstraction

Must pass:

- raw codec determinism,
- sq8 codec determinism,
- profile digest stability,
- profile mismatch fail-closed,
- raw oracle scoring retained.

### Gate G5 — TurboQuant eligibility

Must pass:

- optional feature compiles,
- profile digest stable,
- encode deterministic,
- score deterministic,
- raw-vs-TurboQuant drift harness,
- receipt path works.

### Gate G6 — Release proof

Must pass:

- docs updated,
- examples updated,
- benchmark fixtures checked in or documented,
- final report complete,
- no P0/P1 open without explicit hard blocker.
