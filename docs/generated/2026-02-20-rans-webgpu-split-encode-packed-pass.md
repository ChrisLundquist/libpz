# 2026-02-20 rANS WebGPU Split Encode Packed Submission Pass

## Goal

Move split shared-table encode closer to decode’s packed model by consolidating multi-block work into one encode dispatch/readback cycle when the block count is high.

## Code Changes

In `src/webgpu/rans.rs`:

1. Added `RANS_PACKED_SHARED_ENCODE_MIN_INPUTS` gate (currently `8` non-empty inputs).
2. Added packed shared-table encode path:
   - `rans_encode_chunked_payload_gpu_batched_shared_table_packed(...)`
   - packs independent blocks into one combined input/chunk-metadata layout
   - executes one shared-table encode dispatch
   - performs one words/states readback
   - reconstructs per-block CPU wire-format payloads.
3. Wired `rans_encode_chunked_payload_gpu_batched_impl(...)` to try packed shared-table encode first (when gated conditions are met), then fall back to the prior ring-based per-input path.
4. Added a small type alias (`RansEncodedBatch`) to keep encode batch signatures clippy-clean under `-D warnings`.

In `src/webgpu/tests.rs`:

1. Added `test_rans_chunked_encode_gpu_batched_shared_table_packed_cpu_decode_round_trip` (16 blocks) to force packed encode path and verify CPU decode parity.

## Validation

1. `cargo fmt --check` passed.
2. `cargo clippy --features webgpu -- -D warnings` passed.
3. `cargo test --features webgpu batched` passed (17 tests).

## Measurement Note

No trusted new GPU throughput numbers were collected in this non-approved sandbox pass because profile runs may fall back to CPU when WebGPU adapter creation is unavailable. This pass is recorded as correctness-complete implementation pending stable-GPU remeasurement.

## Next

1. Re-run split encode/decode stage profiles in a stable WebGPU environment to quantify packed encode contribution.
2. Tune packed encode gate threshold versus per-input fallback based on block count and input size.
