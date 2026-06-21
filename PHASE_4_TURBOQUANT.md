# Phase 4 - TurboQuant backend prototype

## Objective

Add TurboQuant as an optional vector codec backend without letting it become source-of-truth or the default until drift/conformance gates pass.

## Preconditions

Phase 4 should not start until:

- Phase 1 key-level HNSW integrity exists.
- Phase 2 SearchContext/receipt skeleton exists.
- Phase 3 VectorCodec abstraction exists.
- Raw reference scoring is available.

## Dependency strategy

Prefer an optional feature:

```toml
[features]
turbo-quant-codec = ["dep:turbo-quant"]

[dependencies]
turbo-quant = { path = "../turbo-quant", optional = true }
```

If repo topology requires a git dependency, document it, but avoid pinning to a floating branch in release mode. Prefer path for local monorepo/Codex work, then later formalize versioning.

## TurboQuant profile mapping

Profile fields:

```text
codec_family:    turbo_quant
codec_version:   turbo-quant crate version or git SHA marker
dim:             u32
bits:            u8
projections:     u32
seed:            u64
score_semantics: inner_product_estimate or l2_distance_estimate
normalization:   raw or unit_norm
```

## Implementation target

Add `TurboQuantCodec` roughly equivalent to:

```rust
#[cfg(feature = "turbo-quant-codec")]
pub struct TurboQuantCodec {
    quantizer: turbo_quant::TurboQuantizer,
    profile: VectorCodecProfileV1,
}
```

Implement:

- `encode(&[f32]) -> TurboCodeBytes` or typed `TurboCode`
- `score_inner_product(code, query)`
- `score_l2(code, query)`
- `decode_approx(code)` if useful
- `profile()`
- deterministic encoded digest

## Serialization

Do not invent a sloppy byte format. Choose one:

1. serde JSON canonicalized and digested; simple but larger
2. bincode/postcard with explicit format version; compact but dependency-sensitive
3. custom versioned bytes; most control, more implementation risk

For prototype, serde JSON is acceptable if documented and digested consistently. For production, define a compact format.

## Drift harness

Add a fixed-seed test harness that compares raw reference scoring to TurboQuant scoring.

Metrics to report:

```text
recall@1
recall@5
recall@10
mean rank drift
max top-k loss
mean absolute score error
p95 absolute score error
storage bytes per vector
latency rough timing if practical
```

## Tests to add

1. Feature-gated compile test.
2. Same profile + same vector = same TurboCode digest.
3. Different seed changes code digest.
4. Wrong profile/dim rejects scoring.
5. Turbo score can be compared to raw reference on fixed corpus.
6. Receipt records TurboQuant profile when used.
7. Fallback to raw reference works when TurboQuant artifact missing/malformed.

## Acceptance criteria

- TurboQuant backend is optional.
- No existing raw/SQ8 behavior regresses.
- TurboQuant code digests are deterministic.
- Drift harness exists and emits metrics.
- Search receipts include codec profile and approximation status.
- TurboQuant is not default until Phase 5 gates pass.

## Codex prompt

```text
Run Phase 4: TurboQuant backend prototype.

Add an optional turbo-quant-codec feature and implement TurboQuantCodec behind the VectorCodec interface from Phase 3. Do not make TurboQuant default. Do not remove raw embeddings. Do not let TurboQuant artifacts become authoritative truth.

Persist or encode TurboCode with an explicit format/profile. Add deterministic encoded digest tests. Add raw-reference comparison tests and a drift harness that reports recall@k, rank drift, score error, and storage bytes per vector on fixed-seed corpora.

Search receipts must record codec family/profile and approximate scoring when TurboQuant is used.

Run cargo check/test without the feature and with --features turbo-quant-codec. Report exact commands, metrics, and remaining risks.
```

---
