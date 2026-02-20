# 2026-02-20 rANS WebGPU Async Pipelining Pass

## Goal

Increase in-flight async GPU work for chunked rANS encode/decode by relaxing the batched pending-ring depth cap.

## Code Changes

1. `src/webgpu/rans.rs`
   - Added `RANS_MAX_PENDING_RING_DEPTH = 8`.
   - Updated both ring-depth calculators:
     - `rans_decode_pending_ring_depth()`
     - `rans_encode_pending_ring_depth()`
   - Previous hard cap: `3` slots.
   - New cap: up to `8` slots (still bounded by memory budget and batch size).
   - Added batched decode completion/readback path:
     - `complete_rans_chunked_payload_decode_batch()`
     - `rans_decode_chunked_payload_gpu_batched()` now drains completed work in bounded batches and reads multiple outputs back in a single submit/map cycle.

2. `examples/profile.rs`
   - Updated `DEFAULT_RANS_GPU_BATCH` from `3` -> `6`.
   - Retune sequence:
     - `3 -> 5` after ring-depth increase sweeps
     - `5 -> 6` after batched-readback post-sweep (`decode` peak shifted upward)

## Validation

- Build check: `cargo check --features webgpu --example profile` (pass)
- Targeted batched-path tests: `cargo test --features webgpu batched` (pass, 12 tests)

## Measurements

### Baseline before this pass (1MB, 300 iters, chunk=2048, batch=3)

From `docs/generated/2026-02-20-profile-direct-rans-encode-1mb-webgpu-escalated.txt` and `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-escalated.txt`:

- Encode: 49.3 MB/s
- Decode: 65.7 MB/s

### Post-change sweeps (1MB, 200 iters)

- Pre-change reference sweep: `docs/generated/2026-02-20-rans-webgpu-decode-batch-extended-1mb.tsv`
- Post-change decode sweep: `docs/generated/2026-02-20-rans-webgpu-decode-batch-extended-1mb-after-ring8.tsv`
- Post-change encode sweep: `docs/generated/2026-02-20-rans-webgpu-encode-batch-extended-1mb-after-ring8.tsv`

Observed trend after ring-depth increase:

- Throughput improves for `gpu_batch` beyond 3 (especially 4-8).
- Peak/near-peak values cluster in `gpu_batch=5..8`.
- Very high batch counts (`10+`) regress.

### Post-change default profile (1MB, 300 iters, defaults)

From:

- `docs/generated/2026-02-20-profile-direct-rans-encode-1mb-webgpu-defaults-after-ring8.txt`
- `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-defaults-after-ring8.txt`

Results:

- Encode: 50.6 MB/s
- Decode: 66.2 MB/s

Delta vs baseline (`batch=3`):

- Encode: +1.3 MB/s (~+2.6%)
- Decode: +0.5 MB/s (~+0.8%)

### Post-change default profile after batched decode readback (1MB, 300 iters, defaults)

From:

- `docs/generated/2026-02-20-profile-direct-rans-encode-1mb-webgpu-defaults-after-batched-readback-final-seq.txt`
- `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-defaults-after-batched-readback-final-seq.txt`

Results:

- Encode: 49.9 MB/s
- Decode: 70.2 MB/s

Delta vs ring-depth-only defaults:

- Encode: -0.7 MB/s (~-1.4%)
- Decode: +4.0 MB/s (~+6.0%)

### Post-change sweeps after batched decode readback (1MB, 200 iters)

- `docs/generated/2026-02-20-rans-webgpu-decode-batch-post-batched-readback-1mb.tsv`
- `docs/generated/2026-02-20-rans-webgpu-encode-batch-post-batched-readback-1mb.tsv`

Observed trend:

- Decode continues scaling through `gpu_batch=6` (peak in this sweep).
- Encode remains near peak across `gpu_batch=5..8`.
- Final default retuned to `gpu_batch=6` for better decode-weighted balance.

### Final default profile (1MB, 300 iters, defaults with `DEFAULT_RANS_GPU_BATCH=6`)

From:

- `docs/generated/2026-02-20-profile-direct-rans-encode-1mb-webgpu-defaults-final-batch6-seq.txt`
- `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-defaults-final-batch6-seq.txt`

Results:

- Encode: 51.0 MB/s
- Decode: 73.4 MB/s

Delta vs original baseline (`batch=3`):

- Encode: +1.7 MB/s (~+3.4%)
- Decode: +7.7 MB/s (~+11.7%)

## Conclusion

- Allowing deeper in-flight rANS work improves async overlap opportunities and makes larger batch settings useful.
- Batched decode readback improves decode throughput materially on this host/device while encode remains stable to slightly improved.
- Slice 4 perf gate is still open; decode remains materially below chunked CPU decode in stage benches.
