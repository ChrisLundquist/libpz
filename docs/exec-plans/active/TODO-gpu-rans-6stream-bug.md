# TODO: GPU rANS interleaved decode fails with 6-stream LzSeqR

## Problem

GPU rANS interleaved decode works correctly for 4-stream pipelines (LzssR)
but fails for 6-stream pipelines (LzSeqR). CPU rANS interleaved decode
works correctly for both 4 and 6 streams.

## Evidence

- `test_gpu_rans_interleaved_decode_round_trip` originally used `Pipeline::Lzr`
  (3 streams). After Lzr removal, switching to `Pipeline::LzSeqR` (6 streams)
  caused the test to fail with `InvalidInput`.
- Switching to `Pipeline::LzssR` (4 streams) passes.
- CPU rANS interleaved encode/decode with LzSeqR works fine.
- The rANS encode/decode code in `src/pipeline/stages.rs` is stream-count
  agnostic — each stream is encoded/decoded independently.

## Workaround

Test uses `Pipeline::LzssR` (4-stream) instead of `Pipeline::LzSeqR` (6-stream).
See `src/pipeline/tests.rs:test_gpu_rans_interleaved_decode_round_trip`.

## Investigation directions

- LzSeq's `offset_extra` and `length_extra` streams can be very small or empty.
  GPU buffer sizing or dispatch dimensions may misbehave with near-zero streams.
- Check if the GPU rANS decode path (`stage_rans_decode_webgpu`) has alignment
  assumptions that break with 6 streams.
- Compare the per-stream byte sizes between LzssR (4 streams, all non-trivial)
  and LzSeqR (6 streams, some potentially empty) to find the divergence point.
- Test with synthetic 6-stream data where all streams are non-trivially sized.

## Files

- `src/pipeline/stages.rs` — `stage_rans_decode_webgpu`, `stage_rans_encode_with_options`
- `src/pipeline/tests.rs` — `test_gpu_rans_interleaved_decode_round_trip`
- `src/webgpu/rans.rs` — GPU rANS implementation
