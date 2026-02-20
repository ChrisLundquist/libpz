# 2026-02-20 rANS WebGPU Split Encode Shared-Table Reuse Pass

## Goal

Reduce encode-side overhead in nvCOMP-style split mode by avoiding per-input rebuild/upload of identical shared tables.

## Code Changes

In `src/webgpu/rans.rs`:

1. Added `create_rans_encode_tables_buffer(...)` to build a GPU table buffer from a normalized table once.
2. Added `rans_encode_chunked_gpu_with_tables(...)` to run encode using a provided GPU table buffer.
3. Added `submit_rans_chunked_payload_encode_with_norm_and_tables(...)` to submit encode work while retaining the shared table buffer lifetime.
4. Updated `rans_encode_chunked_payload_gpu_batched_impl(...)`:
   - when `shared_norm` is provided, create one shared table buffer per batch call
   - clone that handle into each submitted payload instead of rebuilding per-input tables.

No wire format changes were made.

## Validation

1. `cargo fmt --check` passed.
2. `cargo clippy --features webgpu -- -D warnings` passed.
3. `cargo test --features webgpu batched` passed (16 tests).

## Measurement Note

No new trusted GPU stage numbers were collected in this non-approved sandbox pass because direct profile runs may fall back to CPU when a WebGPU adapter is unavailable. This pass is therefore recorded as a code-level optimization milestone pending stable-GPU remeasurement.

## Next

1. Re-run split encode/decode profile commands on stable WebGPU hardware to quantify encode-side delta from shared-table buffer reuse.
2. Continue toward packed split encode submission/readback symmetry (single packed dispatch/readback model, matching decode direction).
