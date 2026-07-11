# State Epistemics Local Fixtures — 2026-07-10

This receipt describes deterministic, CPU-only local fixtures for the P0-2
state-resolution kernel. The cases are STALE-style supersession, A-TMA/LTP-
style historical resolution, and MemTrace-style trajectory resolution.

They exercise typed dependency closure, explicit premise status, safe answer
disposition, versioned resolution receipts, and the existing authority/graph
storage path. They are not upstream STALE, A-TMA/LTP, or MemTrace benchmark
runs, and make no external benchmark or comparative-performance claim.

Receipt-producing test:

```text
cargo test --test state_epistemics --no-default-features --features 'brute-force,testing'
```
