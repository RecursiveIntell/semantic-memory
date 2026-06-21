# Regression Test Matrix

## Embedding invariants

- Fake embedder returns fewer embeddings than requested for document ingest.
- Fake embedder returns more embeddings than requested for document ingest.
- Same mismatch cases for facts/chunks/messages/episodes reembedding.
- NaN/+Inf/-Inf rejected through every public write path.
- Wrong dimension rejected through every public write path.
- Invalid persisted vector blob rejected without panic.

## Deletion / stale-state

- Delete document with chunks + episodes + FTS + HNSW hit; assert no stale search hit.
- Delete fact referenced by episode cause; assert no dangling episode_causes.
- Delete namespace; assert structured report counts all affected rows.
- Full integrity after deletes reports clean.

## Integrity

- Equal-count but wrong FTS rowid map fails.
- Equal-count but wrong HNSW keymap fails.
- q8 missing state follows chosen mandatory/optional policy.
- Repair removes stale derived state and/or backfills missing acceleration rows.

## HNSW persistence

- Corrupt header.
- Truncated graph/data/keymap.
- Huge declared byte_len.
- Dimension mismatch.
- Keymap missing.
- Save failure leaves pending ops replayable.
- Repeated upsert does not produce duplicate stale hits.
- Concurrent save/write does not corrupt sidecar.

## Search

- Mismatched cosine inputs fail.
- NaN scores do not enter ranking.
- Namespace/session/source filtered HNSW falls back when ANN candidates are filtered out.
- top_k/candidate/rerank caps enforced.
- deterministic tie order.
- conversation vector search runs via blocking helper.

## Packaging

- Clean extraction builds/tests/clippy/docs without parent workspace.
- Certifier returns zero findings.
- Archive contains no stale generated sidecars, no prior archives, no target/target-* dirs.
