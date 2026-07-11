# Official BEIR SciFact retrieval evaluation

This harness evaluates semantic-memory's production retrieval APIs on the official BEIR SciFact test corpus. It produces raw per-query JSONL plus aggregate, provenance, backend, and exactness receipts. The checked-in files contain no measured quality claims or benchmark results.

## Claim boundary

The harness can support claims about retrieval quality and local latency for the exact executable, corpus, embedding model, store, mode, split, and configuration recorded in a receipt. It does not establish superiority, general-domain quality, model quality, graph retrieval, native sparse/SPLADE retrieval, token-level late interaction, or Matryoshka quality.

The SciFact ingestion creates no graph edges, so factor-graph evaluation is unavailable. `all-minilm:latest` supplies dense vectors only. Dense-derived sparse vectors are explicitly disabled and are never labeled native sparse or SPLADE. The frozen baseline also disables proxy late interaction, Matryoshka candidate truncation, recency, and derived-vector candidate backends.

Never use held-out results to select weights, thresholds, modes, or other configuration. The deterministic calibration split is the only split intended for diagnosis. Freeze the configuration before running `heldout`.

## Prerequisites

- Python 3 with `requests`
- local Ollama with `all-minilm:latest`
- a Rust toolchain compatible with the workspace

From the workspace root:

```bash
ollama pull all-minilm:latest

python3 -u semantic-memory/tools/scifact_eval/build_scifact_semantic_memory.py \
  --out semantic-memory/target/scifact-eval/scifact-all-minilm-corpus.json \
  --work-dir semantic-memory/target/scifact-eval/build
```

The builder downloads the official archive from the URL embedded in the output, checks the expected 5,183-document/300-test-query shape, truncates each embedding prompt to at most 700 Unicode scalar values without breaking UTF-8, and calls Ollama's single-prompt `/api/embeddings` endpoint. Its append-only JSONL cache is flushed and fsync'd after every new embedding, so rerunning resumes completed work.

## Run calibration

```bash
cargo run -p semantic-memory --example scifact_retrieval_eval -- \
  --corpus semantic-memory/target/scifact-eval/scifact-all-minilm-corpus.json \
  --store-dir semantic-memory/target/scifact-eval/store \
  --output-dir semantic-memory/target/scifact-eval/results \
  --split calibration \
  --mode all
```

The default split sorts test query IDs by `(SHA-256(UTF-8 query_id), query_id)`. The first 100 are calibration and the remaining 200 are held out. This rule depends only on query IDs, not text, qrels, or observed metrics.

The runner builds or resumes one persisted fact store. Documents are inserted with the supplied vectors; ingestion never re-embeds them. Query lookup is backed only by the supplied query-vector fixture, so vector and hybrid modes cannot silently call another model. A sidecar maps BEIR document IDs to generated semantic-memory fact IDs; ID markers are not inserted into searchable text.

The supported modes are:

- `fts_only`: `search_fts_only_with_context`
- `vector_only`: `search_vector_only_with_context` with `PreferExact`; receipts must expose the actual backend/exactness evidence
- `hybrid`: the baseline `search_explained_with_context` pipeline with component score breakdowns

Each mode gets its own JSONL and aggregate JSON file. Failures remain represented as query rows with empty rankings and count against aggregate quality metrics.

## Validate receipts

Run the independent Python validator for each aggregate:

```bash
python3 semantic-memory/tools/scifact_eval/validate_receipt.py \
  semantic-memory/target/scifact-eval/results/scifact-vector_only-calibration.aggregate.json

python3 semantic-memory/tools/scifact_eval/validate_receipt.py --self-test
```

The validator recomputes every query and aggregate metric from raw JSONL. It also checks row count, deterministic split membership, query hashes, qrels against the corpus, corpus/payload/JSONL hashes, rank continuity, result-count distribution, unique-document count, repeated-top-1 frequency, and the executable hash when the recorded binary still exists. Any mismatch exits nonzero.

## Freeze and run held-out

After calibration diagnosis is complete and the configuration is frozen, use the same corpus and store:

```bash
cargo run -p semantic-memory --example scifact_retrieval_eval -- \
  --corpus semantic-memory/target/scifact-eval/scifact-all-minilm-corpus.json \
  --store-dir semantic-memory/target/scifact-eval/store \
  --output-dir semantic-memory/target/scifact-eval/results \
  --split heldout \
  --mode all
```

Validate all three held-out aggregate files before interpreting them. `--split all` is available for a final all-query receipt after the held-out run; it must not become a new tuning surface.

## Tests

```bash
cargo test -p semantic-memory --example scifact_retrieval_eval \
  --no-default-features --features brute-force

python3 -m py_compile \
  semantic-memory/tools/scifact_eval/build_scifact_semantic_memory.py \
  semantic-memory/tools/scifact_eval/validate_receipt.py

python3 semantic-memory/tools/scifact_eval/validate_receipt.py --self-test
```

The Rust tests cover binary/graded metric behavior, nearest-rank latency percentiles, and deterministic disjoint splitting. The Python self-test constructs a tiny corpus and independently validates its raw and aggregate receipts.
