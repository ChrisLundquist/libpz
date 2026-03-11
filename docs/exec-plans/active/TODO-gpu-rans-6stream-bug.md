# RESOLVED: GPU rANS encode routing bug for LzSeqR

**Status:** Fixed (2026-03-10)

## Problem

LzSeqR (6-stream) round-trip failed with `InvalidInput` when compressed with
multi-threaded GPU backend. The bug was mischaracterized as "GPU rANS
interleaved decode fails with 6-stream LzSeqR" — the actual root cause was
on the encode side.

## Root Cause

**Routing inconsistency between single-block and parallel compress paths:**

- `entropy_encode` (blocks.rs:134) — used by single-block/single-thread paths:
  LzSeqR always uses `stage_rans_encode_with_options` (CPU rANS)

- `run_compress_stage` (stages.rs:911) — used by the parallel scheduler:
  LzSeqR routed to `stage_rans_encode_webgpu` when WebGpu backend was active

`stage_rans_encode_webgpu` uses `rans_encode_chunked_payload_gpu_batched`
which produces a **chunked payload format** incompatible with the standard
`rans::decode_interleaved` decoder. The data would encode successfully but
no decoder (CPU or GPU) could decode it.

LzssR didn't have this bug because `run_compress_stage` for LzssR (line 909)
always used `stage_rans_encode_with_options`.

## Evidence

Diagnostic results (192KB input, 64KB block size):
```
size=65536:  OK   (1 block, single-block fast path)
size=131072: FAIL (2 blocks, parallel path → stage_rans_encode_webgpu)
192KB threads=1: OK (sequential path → stage_rans_encode_with_options)
192KB threads=2: FAIL (parallel path → stage_rans_encode_webgpu)
```

## Fix

Changed `run_compress_stage` for `(Pipeline::LzSeqR, 1)` to always use
`stage_rans_encode_with_options`, matching the `entropy_encode` path.

## Remaining TODO

The GPU rANS encode path (`stage_rans_encode_webgpu`) produces a chunked
payload wire format that the standard rANS decoder doesn't understand.
If GPU rANS encode is ever re-enabled for LzSeqR, the chunked decode path
must be wired into `stage_rans_decode_webgpu`. However, since GPU rANS
entropy is known to be slower than CPU (0.54-0.77x), this is low priority.

## Files

- `src/pipeline/stages.rs:911` — fix: removed GPU routing for LzSeqR stage 1
- `src/pipeline/blocks.rs:134` — reference: entropy_encode always uses CPU
- `src/pipeline/tests.rs` — new test: `test_gpu_rans_interleaved_decode_lzseqr_6stream`
