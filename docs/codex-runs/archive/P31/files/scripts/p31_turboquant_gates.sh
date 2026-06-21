#!/usr/bin/env bash
set -euo pipefail

cargo fmt --all --check
cargo check --workspace --all-features
cargo test -p turbo-quant
cargo test -p semantic-memory --features turbo-quant-codec
cargo test --workspace --all-features
cargo clippy -p turbo-quant --all-targets --all-features -- -D warnings
cargo clippy -p semantic-memory --all-targets --all-features -- -D warnings
cargo clippy --workspace --all-targets --all-features -- -D warnings
