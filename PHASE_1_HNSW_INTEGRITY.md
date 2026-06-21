# Phase 1 - HNSW and vector/index integrity hardening

## Objective

Make HNSW a lawful rebuildable acceleration artifact. The goal is not maximum performance. The goal is that semantic-memory can prove the HNSW sidecar/keymap is coherent with live embedded rows, or fall back/rebuild when it cannot.

## P0 issues covered

- F-001: key-level parity
- F-002: sidecar validation
- F-003: fixed-width sidecar dimensions
- F-004: filtered under-return fallback, if search module overlap is practical
- F-005: pending-op durability/rebuild-on-failure

## Design target

Add or harden an integrity API equivalent to:

```rust
pub enum IntegrityScope {
    Fast,
    Full,
}

pub struct HnswIntegrityReport {
    pub checked_at: DateTime<Utc>,
    pub live_embedding_rows: usize,
    pub active_keymap_rows: usize,
    pub missing_keys: Vec<String>,
    pub stale_keys: Vec<String>,
    pub malformed_keys: Vec<String>,
    pub wrong_domain_keys: Vec<String>,
    pub dimension_mismatches: Vec<String>,
    pub sidecar_status: SidecarStatus,
    pub recommended_action: IntegrityAction,
}

pub enum IntegrityAction {
    Accept,
    RebuildHnsw,
    DisableHnswAndFallback,
    Error,
}
```

Names can differ. Semantics cannot.

## Required key-level parity

For every live embedded row:

```text
expected_key = structured domain prefix + stable row/episode ID
expected_key exists in hnsw_keymap with deleted = false
mapped HNSW node/key points back to this row
embedding exists
embedding dimension matches configured dimension/profile
embedding is finite
```

For every active HNSW keymap row:

```text
key parses under structured key parser
key domain is one of known domains
corresponding live SQLite row exists
corresponding row has valid embedding
not tombstoned/deleted
```

## Sidecar header target

If sidecar format is changed, use a fixed-width header. Persisted `usize` is forbidden.

Minimum header fields:

```text
magic:          u32 or fixed bytes
version:        u16
header_len:     u16
dim:            u32
vector_count:   u64
format_flags:   u32
profile_digest: optional fixed digest or reserved bytes
```

Rules:

- Old/unknown sidecar versions must fail cleanly and rebuild, not panic.
- Wrong dimension must rebuild or disable HNSW.
- Wrong count must trigger full key-level validation or rebuild.
- Empty graph sidecar is not enough validation.

## Pending-op failure policy

Implement the simplest safe policy:

```text
If applying pending HNSW ops mutates the in-memory index but durable sidecar/keymap save fails:
  mark HNSW dirty/untrusted
  discard or quarantine in-memory index
  require rebuild from SQLite before serving HNSW search
  fall back to exact/brute-force search until rebuild succeeds
```

## Tests to add

Add integration/unit tests for:

1. Missing keymap entry for a live embedded row.
2. Active keymap entry pointing to a missing/deleted row.
3. Swapped keymap IDs with counts still equal.
4. Wrong domain prefix, e.g. document key used where episode key is expected.
5. Sidecar with wrong dimension.
6. Sidecar with unsupported version.
7. Empty graph sidecar.
8. Save failure/rebuild-on-failure if injectable via test seam.
9. HNSW disabled/fallback path when integrity fails.

## Acceptance criteria

- Full integrity check catches count-equal corruption.
- Fixed-width sidecar header exists, or old sidecar is explicitly treated as cache-only with rebuild-on-suspicion.
- Persisted dimensions no longer use `usize`.
- Dirty/untrusted sidecar cannot serve HNSW silently.
- Tests cover swapped IDs/stale keys/wrong domains.
- `cargo test` passes for HNSW/integrity tests.

## Codex prompt

```text
Run Phase 1: HNSW and vector/index integrity hardening.

Focus only on HNSW/keymap/sidecar/rebuild integrity. Do not add TurboQuant yet.

Implement key-level parity validation between live embedded SQLite rows and active HNSW keymap rows. Replace any persisted usize sidecar dimension/count format with fixed-width versioned fields, or add a rebuild-on-suspicion compatibility path if migration is too risky. Strengthen sidecar validation beyond non-empty graph files. Add tests that catch count-equal corruption: swapped key IDs, stale keys, missing keys, wrong domain prefixes, deleted rows, and dimension mismatch.

If pending HNSW ops can leave memory mutated after durable save failure, add a rebuild-on-failure or dirty/untrusted policy.

Run targeted tests plus cargo fmt/check. Report exact commands and residual risks.
```

---
