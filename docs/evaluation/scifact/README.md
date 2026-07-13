# BEIR SciFact retrieval evaluation

Receipt-bearing evaluation of `semantic-memory` retrieval APIs on the official BEIR SciFact test corpus.

![BEIR SciFact evaluation workflow](../../assets/scifact-evaluation.svg)

The harness builds one persisted store, executes a deterministic calibration/held-out split, emits raw per-query rows plus aggregate receipts, and validates those artifacts independently. The checked-in repository contains the harness and validator; it does **not** contain a frozen score artifact.

## Claim boundary

A validated run can support claims about retrieval quality and local latency for the exact:

- executable and source revision;
- SciFact corpus and payload hashes;
- embedding model and dimensions;
- persisted store and document mapping;
- retrieval mode and configuration;
- split definition and selected query IDs;
- raw per-query rows and aggregate receipt.

It does not establish:

- superiority over another system;
- general-domain retrieval quality;
- embedding-model quality in isolation;
- graph or factor-graph retrieval quality;
- native sparse/SPLADE retrieval;
- token-level late interaction;
- Matryoshka retrieval quality;
- production latency on another host.

Do not publish a bare metric table. Publish the validated artifact bundle and state what it does not prove.

## Frozen evaluation contract

| Property | Value enforced by source |
| --- | --- |
| Dataset | BEIR-hosted SciFact archive; source hashes recorded, but no expected archive digest is pinned in the builder |
| Corpus shape | 5,183 documents |
| Test queries | 300 |
| Default embedding fixture | `all-minilm:latest` through local Ollama; configurable by builder argument or environment |
| Maximum semantic text length | 700 Unicode scalar values before the query/document role prefix is added; complete prompts may be longer |
| Store namespace | `beir-scifact` |
| Retrieval depth | `top_k = 10` |
| Calibration split | First 100 query IDs after sorting by `(SHA-256(UTF-8 query_id), query_id)` |
| Held-out split | Remaining 200 query IDs |
| Modes | `fts_only`, `vector_only`, `hybrid` |
| Row schema | `semantic-memory-scifact-query-v1` |
| Aggregate schema | `semantic-memory-scifact-aggregate-v1` |

The builder rejects any corpus that does not match the expected 5,183-document/300-query shape. The runner refuses unknown query text rather than silently re-embedding through another model.

## Retrieval modes

| Mode | API | Exactness and lane policy |
| --- | --- | --- |
| `fts_only` | `search_fts_only_with_context` | Lexical retrieval only |
| `vector_only` | `search_vector_only_with_context` | `ExactnessProfile::PreferExact`; receipt must disclose backend evidence |
| `hybrid` | `search_explained_with_context` | Baseline BM25+dense RRF with component score evidence |

The frozen baseline sets:

- `sparse_weight = 0`;
- dense-derived sparse retrieval off;
- late-interaction weight to zero;
- `candidate_dims = None`;
- recency off;
- derived-vector backend disabled.

SciFact ingestion creates no graph edges. The harness therefore cannot support graph, factor-graph, native sparse, token-level late-interaction, or Matryoshka claims.

## Artifact flow

```text
BEIR-hosted SciFact archive
  -> shape and path-safety checks
  -> normalized document/query embeddings with append-only fsync cache
  -> persisted semantic-memory fact store + BEIR-ID mapping
  -> deterministic calibration or held-out selection
  -> one JSONL + one aggregate JSON per retrieval mode
  -> independent metric/hash/split/executable validation
```

## Prerequisites

Run from the Libraries workspace root.

- Python 3
- Python package `requests`
- local Ollama
- `all-minilm:latest`
- Rust 1.75 or newer
- a compatible C++ toolchain for the default usearch build, or the explicit `brute-force` feature command shown below

```bash
python3 -m pip install requests
ollama pull all-minilm:latest
```

## 1. Build the corpus fixture

```bash
python3 -u semantic-memory/tools/scifact_eval/build_scifact_semantic_memory.py \
  --out semantic-memory/target/scifact-eval/scifact-all-minilm-corpus.json \
  --work-dir semantic-memory/target/scifact-eval/build
```

The builder:

1. downloads the archive from the source URL embedded in the output fixture; the builder records source hashes but does not compare against a pinned expected archive digest;
2. rejects absolute or parent-traversing ZIP members;
3. verifies corpus and query counts;
4. constructs the semantic text used for embedding;
5. normalizes every vector and rejects empty, zero-norm, non-finite, or dimension-drifting embeddings;
6. stores new embeddings in an append-only JSONL cache;
7. flushes and `fsync`s every newly cached vector so interrupted builds can resume;
8. records source, payload, embedding, and truncation metadata in the fixture.

The 700-character ceiling applies to semantic text before `search_document:` or `search_query:` is added. It is not a tokenizer-aware prompt limit. If Ollama rejects a pathological complete prompt, the builder retries deterministic 500/300/120-character prompt lengths and discloses that policy in corpus metadata.

`all-minilm:latest` is the documented default, not an immutable harness invariant. If `--model` or `SCIFACT_EMBED_MODEL` changes it, retain that model identity in the fixture and every public claim.

## 2. Run calibration

Calibration is the only split intended for diagnosis or configuration selection.

```bash
SCIFACT_EVAL_LAUNCHER='cargo run -p semantic-memory --example scifact_retrieval_eval -- ...' \
  cargo run -p semantic-memory --example scifact_retrieval_eval -- \
  --corpus semantic-memory/target/scifact-eval/scifact-all-minilm-corpus.json \
  --store-dir semantic-memory/target/scifact-eval/store \
  --output-dir semantic-memory/target/scifact-eval/results \
  --split calibration \
  --mode all
```

`SCIFACT_EVAL_LAUNCHER` is optional. Set it when the exact outer launcher command must appear in the receipt; the runner always records the resolved executable invocation.

To run the evaluator without the default usearch/C++ dependency, select the exact brute-force feature explicitly:

```bash
cargo run -p semantic-memory \
  --no-default-features \
  --features brute-force \
  --example scifact_retrieval_eval -- \
  --corpus semantic-memory/target/scifact-eval/scifact-all-minilm-corpus.json \
  --store-dir semantic-memory/target/scifact-eval/store \
  --output-dir semantic-memory/target/scifact-eval/results \
  --split calibration \
  --mode all
```

The runner builds or resumes one persisted fact store. Documents are inserted with fixture vectors and are not re-embedded. A sidecar mapping links official BEIR document IDs to generated fact IDs; document-ID markers are not inserted into searchable text.

Request IDs are deterministic by mode, split, and query. Repeating the same mode/split against an existing receipt-bearing store can collide with an earlier run whose evaluation timestamp differs. Use a fresh store path for each independently reportable run, or remove the prior evaluation store before rerunning.

Failures remain represented as query rows with empty rankings and count against aggregate metrics.

## 3. Validate calibration artifacts

Each mode produces:

```text
scifact-<mode>-calibration.jsonl
scifact-<mode>-calibration.aggregate.json
```

Validate every aggregate:

```bash
for mode in fts_only vector_only hybrid; do
  python3 semantic-memory/tools/scifact_eval/validate_receipt.py \
    "semantic-memory/target/scifact-eval/results/scifact-${mode}-calibration.aggregate.json"
done
```

The independent validator recomputes or verifies:

- per-query and aggregate metrics;
- row count and JSONL hash;
- deterministic split membership;
- query and qrels identity;
- corpus source/payload hashes;
- ranked-result and result-object agreement;
- unique result IDs and contiguous ranks;
- result-count distribution;
- unique retrieved-document count;
- repeated top-1 frequency;
- executable, runner, and store-mapping hashes when those artifacts remain available.

A nonzero validator exit means the artifact is not eligible for a public metric claim.

The validator does **not** currently prove backend/exactness policy, validate per-query search-receipt digests, reconstruct every effective `SearchConfig` field, verify embedding values against the declared model, re-hash the live database against its after-run digest, require a clean Git state, or record machine identity. Those remain separate publication gates.

## 4. Freeze configuration

Before held-out execution, record and stop changing:

- source revision and dirty-state policy;
- corpus fixture hash;
- embedding model and dimensions;
- store path and mapping;
- retrieval modes;
- all search weights, thresholds, candidate settings, and exactness policy;
- calibration count and split algorithm;
- command line and host/runtime metadata required by the intended claim.

Never select weights, thresholds, modes, or other configuration from held-out outcomes.

The aggregate serializes the baseline's selected disabled options and binds the executable/source identity, but it does not serialize every effective search weight or threshold. Retain a complete effective configuration alongside any published result.

## 5. Run held-out

Use the same corpus and persisted store after freezing configuration:

```bash
SCIFACT_EVAL_LAUNCHER='cargo run -p semantic-memory --example scifact_retrieval_eval -- ...' \
  cargo run -p semantic-memory --example scifact_retrieval_eval -- \
  --corpus semantic-memory/target/scifact-eval/scifact-all-minilm-corpus.json \
  --store-dir semantic-memory/target/scifact-eval/store \
  --output-dir semantic-memory/target/scifact-eval/results \
  --split heldout \
  --mode all
```

Validate the three held-out aggregates:

```bash
for mode in fts_only vector_only hybrid; do
  python3 semantic-memory/tools/scifact_eval/validate_receipt.py \
    "semantic-memory/target/scifact-eval/results/scifact-${mode}-heldout.aggregate.json"
done
```

`--split all` is available only for a final all-query receipt after held-out evaluation. It must not become a new tuning surface.

## Metrics

The aggregate receipt contains averages over every selected query for:

- nDCG@10;
- Recall@1, @5, and @10;
- MRR@10;
- MAP@10;
- Success@1, @5, and @10;
- mean, p50, p95, and maximum local latency;
- failure count;
- result-count distribution;
- unique retrieved documents;
- most frequently repeated top-1 document.

Latency is scoped to the recorded local execution. It is not portable across machines, builds, stores, thermal state, or concurrent workload.

## Receipt contents

The aggregate receipt binds the run to:

| Family | Recorded evidence |
| --- | --- |
| Executable | Resolved path and SHA-256 |
| Source | Runner path/hash, crate and workspace commits, dirty-state digests |
| Corpus | Path, file hash, source hashes, payload hashes |
| Embedding | Model, dimensions, normalization metadata |
| Store | Database path/hash, namespace, mapping path/hash, document count |
| Retrieval | Mode, `top_k`, source types, frozen configuration, backend/exactness evidence |
| Split | Algorithm, calibration count, total count, selected-query-ID hash |
| Capabilities | Explicit unavailable reasons for sparse, graph, Matryoshka, and late-interaction claims |
| Claim boundary | The precise interpretation allowed for the run |

Per-query rows carry query identity, qrels, ordered document IDs, result details, latency, errors, computed metrics, and the search receipt returned by the selected retrieval API.

## Test the harness

```bash
cargo test -p semantic-memory --example scifact_retrieval_eval \
  --no-default-features --features brute-force

python3 -m py_compile \
  semantic-memory/tools/scifact_eval/build_scifact_semantic_memory.py \
  semantic-memory/tools/scifact_eval/validate_receipt.py

python3 semantic-memory/tools/scifact_eval/validate_receipt.py --self-test
```

The Rust unit tests cover binary and graded metric behavior, nearest-rank latency percentiles, and deterministic disjoint splitting. They do not run an end-to-end 5,183-document SciFact evaluation. The Python self-test creates a tiny corpus and validates its raw and aggregate receipts independently.

## Before publishing results

- [ ] Configuration was frozen before held-out execution.
- [ ] All three mode aggregates passed the independent validator.
- [ ] Raw JSONL and aggregate JSON files are retained together.
- [ ] Corpus, executable, source, and store-mapping hashes resolve.
- [ ] The reportable run starts from a clean Git worktree; dirty-state fields are still retained as evidence.
- [ ] Failures remain counted.
- [ ] Latency host/runtime context is included.
- [ ] A complete effective search configuration is retained outside the partial aggregate fields.
- [ ] Backend/exactness expectations and per-query receipt digests are checked separately.
- [ ] The claim names the exact mode, split, model, configuration, and revision.
- [ ] The claim repeats the unavailable capability boundaries.
- [ ] No general superiority or production claim is inferred from SciFact alone.
