# Phase 3 - Vector codec abstraction and artifact profiles

## Objective

Decouple vector representation from memory truth. This creates the seam where SQ8, raw f32, and TurboQuant can coexist without corrupting authority semantics.

## P1 issues covered

- F-008: avoid overloading `embedding_q8`
- F-009: make search receipts codec-aware
- TurboQuant precondition: codec metadata and profile digest

## Design target

Introduce a `VectorCodec` trait or equivalent service boundary.

Suggested trait:

```rust
pub trait VectorCodec {
    type Code;

    fn profile(&self) -> VectorCodecProfileV1;
    fn encode(&self, raw: &[f32]) -> Result<Self::Code>;
    fn score_inner_product(&self, code: &Self::Code, query: &[f32]) -> Result<f32>;
    fn score_l2(&self, code: &Self::Code, query: &[f32]) -> Result<f32>;
    fn decode_approx(&self, code: &Self::Code) -> Result<Option<Vec<f32>>>;
}
```

Names can differ. The semantic boundary cannot.

## Required codec families

Implement at least:

```text
RawF32Codec - exact/reference path
Sq8Codec    - wrapper around existing semantic-memory quantize module
```

Reserve but do not implement yet:

```text
TurboQuantCodec - Phase 4
PolarQuantCodec - optional
QjlSketchCodec  - optional
```

## Vector codec profile

Suggested struct:

```rust
pub struct VectorCodecProfileV1 {
    pub profile_id: String,
    pub codec_family: String,
    pub codec_version: String,
    pub dim: u32,
    pub bits: Option<u8>,
    pub projections: Option<u32>,
    pub seed: Option<u64>,
    pub score_semantics: String,
    pub normalization: String,
    pub canonical_json: String,
    pub profile_digest: String,
}
```

Profile digest must be deterministic.

## Vector artifact table/schema

Add storage only if the current DB architecture can absorb it cleanly. Otherwise add in-memory structs and TODO migration doc.

Preferred minimal table:

```sql
CREATE TABLE IF NOT EXISTS vector_codec_profiles (
  profile_id TEXT PRIMARY KEY,
  codec_family TEXT NOT NULL,
  codec_version TEXT NOT NULL,
  dim INTEGER NOT NULL,
  bits INTEGER,
  projections INTEGER,
  seed TEXT,
  score_semantics TEXT NOT NULL,
  normalization TEXT NOT NULL,
  canonical_json TEXT NOT NULL,
  profile_digest TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vector_artifacts (
  artifact_id TEXT PRIMARY KEY,
  source_kind TEXT NOT NULL,
  source_id TEXT NOT NULL,
  source_embedding_digest TEXT NOT NULL,
  profile_id TEXT NOT NULL,
  encoded_bytes BLOB NOT NULL,
  encoded_digest TEXT NOT NULL,
  generation INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(profile_id) REFERENCES vector_codec_profiles(profile_id)
);
```

## Profile mismatch behavior

Scoring must reject mismatches:

- wrong dim
- wrong codec family
- wrong profile digest
- incompatible score semantics
- malformed code bytes
- non-finite query values

## Tests to add

1. RawF32 profile digest is deterministic.
2. SQ8 profile digest is deterministic.
3. Same profile + same vector = same encoded digest.
4. Wrong dimension rejects scoring.
5. Wrong profile rejects scoring.
6. Malformed code rejects scoring.
7. Raw reference score matches existing cosine/dot expectations.
8. SQ8 path works through the same codec interface.

## Acceptance criteria

- Codec profiles exist and are deterministic.
- Existing SQ8 behavior is not overloaded or reinterpreted.
- Raw f32 reference codec is available.
- Profile mismatch fails loudly.
- Search receipt can reference codec profile even before TurboQuant.

## Codex prompt

```text
Run Phase 3: vector codec abstraction and artifact profiles.

Add a VectorCodec boundary with RawF32 and SQ8/current-quantize implementations. Add deterministic VectorCodecProfileV1 metadata and encoded digest helpers. Do not implement TurboQuant yet. Do not rename embedding_q8 to mean TurboQuant. Keep raw/reference scoring available for conformance.

If a storage migration is safe, add vector_codec_profiles and vector_artifacts tables. If not, add typed structs and a migration plan doc, but still implement deterministic profile digests and codec mismatch rejection.

Add tests for deterministic profile digests, same-vector same-code digest, wrong-dimension/profile rejection, malformed code rejection, and raw/SQ8 scoring through the same interface.

Run targeted codec tests plus cargo fmt/check. Report exact commands and remaining risks.
```

---
