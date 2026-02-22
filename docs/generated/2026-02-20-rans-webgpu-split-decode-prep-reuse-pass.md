# 2026-02-20 rANS WebGPU Split Decode Prep Reuse Pass

## Goal

Amortize split decode preparation across repeated runs so independent-block decode profiling is less dominated by CPU-side parse/pack setup.

## Code Changes

1. `src/webgpu/rans.rs`
   - Refactored packed shared-table split decode into two phases:
     - prepare: `prepare_rans_decode_chunked_payload_gpu_batched_shared_table_packed(...)`
     - execute: `execute_rans_decode_chunked_payload_gpu_batched_shared_table_packed(...)`
   - Added `rans_decode_chunked_payload_gpu_batched_shared_table_repeated(...)`:
     - builds shared decode tables once
     - prepares packed split decode inputs once (when packed path is eligible)
     - executes decode for `iterations` runs without reparsing payloads each call
   - Updated one-shot shared-table decode API to call the repeated path with `iterations=1`, preserving existing behavior.

2. `examples/profile.rs`
   - Split decode profiling now calls:
     - `rans_decode_chunked_payload_gpu_batched_shared_table_repeated(&decode_inputs, data, iterations)`
   - This exercises steady-state decode runs with prep reuse.

3. `src/webgpu/tests.rs`
   - Added `test_rans_chunked_decode_gpu_batched_shared_table_repeated_round_trip`.

## Validation

1. `cargo fmt --check` passed.
2. `cargo clippy --features webgpu -- -D warnings` passed.
3. `cargo test --features webgpu batched` passed (16 tests).

## Measurement Note

Non-approved sandbox profiling runs produced artifacts:

- `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-defaults-prep-cache-pass.txt`
- `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-independent-blocks-256k-prep-cache-pass.txt`
- `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-independent-blocks-64k-prep-cache-pass.txt`

These captures did not print the expected WebGPU stage-path banners and therefore likely exercised CPU fallback. They are retained for traceability, but not used as GPU gate evidence.

## Next

1. Re-run the same decode stage commands in a stable WebGPU-capable environment to quantify true GPU delta from prep reuse.
2. Apply the same prepare/execute split to the independent-block encode path so both directions can amortize setup.
