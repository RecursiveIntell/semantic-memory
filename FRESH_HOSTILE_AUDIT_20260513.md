# Fresh Hostile Audit — semantic-memory / TurboQuant latest snapshot

**Source package:** `semantic-memory-semantic-memory-next-codex-context-20260513.zip`  
**Audit date:** 2026-05-13  
**Mode:** static source review + attempted local gates + research-gap mapping  
**Local gate caveat:** this execution environment could not reach `index.crates.io`; cargo gates that need dependency resolution failed before build/test. `cargo fmt --all --check` passed locally.

## Executive verdict

The stack is materially improved and now earns the internal label:

```text
tq-live-feature-gated-exact-rerank-ready
```

It does **not** earn:

```text
turbo-quant-default-ready
v11A-conformant-core
v11B-conformant-runtime
release-clean
```

P31 did the right kind of work: BLAKE3/`stack-ids` digests, deterministic TurboQuant wire bytes, malformed-code rejection, derived artifact storage, HNSW sidecar manifest, exact f32 rerank, receipts, and a feature-gated live TurboQuant candidate path. The remaining work is no longer “add TurboQuant.” It is “make the retrieval runtime evidence-grade, scalable, conformance-gated, and honest under failure.”

## Package-level evidence

The 2026-05-13 certifier report says:

- strict package mode;
- 189 included files;
- 103 Rust source files;
- 22 TurboQuant files;
- 0 validation findings;
- archive hash `e6f37d481715c1cb0c9e81beeca54990e0e860334679761652bac89dd8118b87`.

The package is clean as a context bundle. That is not the same as release readiness.

## What is now solid

### 1. TurboQuant is a real workspace surface now

The package includes `turbo-quant` as a workspace member and the P31 final report claims TurboQuant tests, semantic-memory TurboQuant feature tests, and benchmark gates were run. The active package also includes new TurboQuant hardening surfaces:

- `turbo-quant/src/wire.rs`
- `turbo-quant/tests/wire_format.rs`
- `turbo-quant/tests/malformed_artifacts.rs`
- `turbo-quant/benches/turbo_quant_search.rs`

### 2. Derived vector artifacts are real, not implied

`semantic-memory/src/db.rs` defines `derived_vector_artifacts` with:

- `item_key`
- `codec_family`
- `codec_profile_digest`
- `source_embedding_digest`
- `encoded_digest`
- `artifact_digest`
- `encoding`
- `dim`
- `encoded`
- lifecycle status

Raw f32 embeddings remain authoritative; TurboQuant artifacts are rebuildable derived state.

### 3. Digests moved from cache-grade to evidence-compatible

The prior FNV-style identity surface appears replaced by `stack_ids::ContentDigest` / `DigestBuilder`, with digests rendered as `blake3:<64-hex>`. Static grep found no remaining `fnv` / `FNV` / `fnv1a` references in `semantic-memory/src` or `turbo-quant/src`.

### 4. HNSW graph/data sidecars are now manifest-bound

`HnswSidecarManifestV1` now binds:

- schema version;
- generation id;
- graph/data file names;
- graph/data BLAKE3 digests;
- dimensions;
- vector count;
- sidecar format version;
- source SQLite epoch;
- creation time.

Load validates digests before graph/data use. Legacy sidecar loading remains allowed but disclosed.

### 5. Receipts are meaningfully richer

`VectorSearchReceiptV1` now includes execution/evidence hooks:

- trace id;
- attempt family id;
- attempt id;
- replay id;
- query text/input/filter digests;
- redaction state;
- budget id;
- deadline;
- codec family/profile;
- artifact manifest/counts;
- stale/missing/corrupt counts;
- approximate candidate count;
- exact rerank count;
- fallback reason;
- degradation notes.

This is v11A-compatible scaffolding, not v11A compliance.

## Hostile findings

### H1 — Run provenance is still wrong

`semantic-memory/docs/codex-runs/CURRENT_RUN.md` says:

```text
Current run: `P30`
Updated UTC: `2026-05-13T02:36:02Z`
```

But the active docs clearly contain `P31_TURBOQUANT_READINESS_FINAL.md`, and P31 artifacts were archived. This is provenance rot. It makes the artifact story harder to trust.

**Fix:** P32 must set the active run to `P32_RESEARCH_MAX_RETRIEVAL_RUNTIME`, preserve P31 as historical evidence, and classify P31 audit artifacts under P31 rather than `unclassified` where possible.

### H2 — P31 evidence logs were archived out of the active package

The sidecar says 28 stale run artifacts were moved into `docs/codex-runs/archive/...`, but the report says `docs/codex-runs/archive` is pruned because `codex-archive-disabled`. The active package contains summaries, but not the underlying command logs, final status, benchmark summary, dirty tree, or scripts.

**Fix:** include a compact active `RUN_EVIDENCE_INDEX.md` and `RUN_EVIDENCE_SUMMARY.json` that preserve digest pointers to archived evidence, or include selected final gate logs in active package scope.

### H3 — The TurboQuant README still overclaims

`turbo-quant/README.md` still says:

```text
3-8 bits per value with zero accuracy loss and no dataset-specific calibration
```

P31 final report says this wording was removed from TurboQuant crate docs, but it remains in the active README. That is a public-claim blocker.

**Fix:** replace with conservative language: “compressed-vector candidate generation with exact-rerank evaluation; no default production claim without accepted corpus gates.”

### H4 — Wire format does not validate the seed header

`TurboCodeWireV1::encode` writes `profile.seed()` into the header, but `decode` reads it into `_seed` and does not compare it to `profile.seed()`.

This is not catastrophic because semantic-memory binds artifacts to a profile digest outside the wire artifact, but a deterministic wire format should reject profile-mismatched header claims internally.

**Fix:** reject seed mismatch and add `seed_mismatch_rejected` test.

### H5 — Wire decoder permits non-canonical QJL padding bits

QJL signs are packed into bytes. If projection count is not a multiple of 8, unused high bits in the last byte are currently ignored. That allows multiple byte encodings for the same logical sketch. For evidence-grade artifacts, canonical bytes matter.

**Fix:** reject non-zero unused padding bits and add a malformed-wire test.

### H6 — Live TurboQuant path still validates every artifact against raw state inside query execution

The live candidate path loads each authoritative raw row and recomputes source embedding digest during candidate generation. This is safe, but it turns a compressed candidate backend into a DB-heavy validation loop.

**Fix:** add artifact-generation manifests and generation epochs so query-time can trust a complete current artifact generation and only exact-rerank selected candidates against raw f32.

### H7 — Candidate generation sorts the full corpus

The candidate path sorts `scored` for all artifacts and only then truncates. For large corpora this is unnecessary.

**Fix:** use a bounded top-k heap / partial selection keyed by approximate score. Record `approximate_scanned_count` separately from `approximate_returned_count`.

### H8 — Candidate backend naming is misleading

The successful TurboQuant path uses:

```text
turbo_quant_then_brute_force_f32
```

That reads like fallback or brute-force path. The actual behavior is approximate TurboQuant candidate generation followed by exact f32 rerank.

**Fix:** rename successful path to:

```text
turbo_quant_candidate_then_exact_f32
```

Reserve `*_then_brute_force_f32` only for fallback.

### H9 — SQL-filtered TurboQuant still always degrades to raw f32

The readiness doc explicitly says SQL filters cause fallback to brute-force raw f32. This is safe but blocks default eligibility and reduces practical utility.

**Fix:** add filter-aware candidate generation. Either materialize filter metadata alongside derived artifacts or perform adaptive oversampling with post-filter candidate guarantees.

### H10 — Benchmarks are useful but still too narrow

P31 reports one deterministic gate:

```text
dim=384 corpus=1000 queries=50 recall@10=1.000 ndcg@10=1.000 encoded_bytes_per_vector=1022.0 raw_bytes_per_vector=1536.0 classification=green
```

The Criterion run covers more sizes, but the release gate is still too narrow. There are no accepted thresholds for 100k corpus, filtered queries, stale artifacts, p95/p99 latency, rebuild cost, or memory footprint.

**Fix:** P32 must define and enforce benchmark classes: smoke, internal, release-candidate, and default-eligibility.

### H11 — Cargo gates could not be freshly reproduced in this environment

Fresh local attempts:

- `cargo fmt --all --check`: passed.
- `cargo check/test/clippy`: failed before build because crates.io was unreachable and required dependencies were not available in local cache.

This is not a source failure. It is a local audit limitation. The P31 final report claims scoped gates passed; P32 should make those logs active or reproducibly packaged.

## Release recommendation

### Accept

- internal feature-gated evaluation;
- shadow-mode / benchmark-mode use;
- exact-reranked TurboQuant candidate generation;
- receipt and degradation conformance hardening;
- research-max P32 pass.

### Reject

- default TurboQuant retrieval;
- public “zero accuracy loss” claim;
- release-clean claim;
- v11A/v11B compliance claim;
- full production compression claim.

## Correct next label

```text
p32-research-max-retrieval-runtime-needed
```

