# 2026-02-20 rANS WebGPU Decode Hotspot Pass

## Goal

Address CPU-side decode preparation hotspots before committing the current nvCOMP-style split path work.

## Code Changes

In `src/webgpu/rans.rs`:

1. Added `copy_u16_words_from_le_bytes(...)` to copy lane words directly from payload bytes into a `u16` destination buffer.
2. Added `pack_u16_words(...)` to bulk-pack `u16` words into the packed `u32` format expected by GPU buffers.
3. In `parse_rans_chunked_payload_decode(...)`, replaced repeated per-lane `write_packed_u16_slice(...)` calls with:
   - direct `u16` lane placement using `copy_u16_words_from_le_bytes(...)`
   - one final bulk pack step via `pack_u16_words(...)`
4. In packed shared-table decode assembly, replaced `write_packed_u16_slice(...)` with bulk `pack_u16_words(...)`.
5. In `rans_decode_chunked_payload_gpu_batched_shared_table(...)`, removed per-call seed-frequency counting and derive the normalized table from the first non-empty payload header.

## Profiling Evidence

`samply` top-symbol comparison:

1. Before pass (`docs/generated/2026-02-20-samply-rans-decode-split64k-1mb-top35.txt`), `write_packed_u16_slice` and `simd::avx2::byte_frequencies` were prominent.
2. After pass (`docs/generated/2026-02-20-samply-rans-decode-split64k-1mb-hotspot-pass-top35.txt`), those symbols no longer appear in top rows.
3. Default-path comparison shows the same trend:
   - before: `docs/generated/2026-02-20-samply-rans-decode-default-1mb-top35.txt`
   - after: `docs/generated/2026-02-20-samply-rans-decode-default-1mb-hotspot-pass-top35.txt`

## Throughput Snapshot (decode, 1MB, 300 iterations)

Current pass files:

1. Defaults:
   - `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-defaults-hotspot-pass.txt`: 69.1 MB/s
   - rerun: `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-defaults-hotspot-pass-rerun.txt`: 69.0 MB/s
2. Split 256KB (4 blocks):
   - `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-independent-blocks-256k-hotspot-pass.txt`: 71.7 MB/s
   - reruns: 71.5 / 53.2 / 56.5 MB/s
3. Split 64KB (16 blocks):
   - `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-independent-blocks-64k-hotspot-pass.txt`: 34.5 MB/s
   - reruns: 29.1 / 30.1 MB/s

Observed spread:

1. Defaults are stable (~69 MB/s).
2. Split paths are highly variable under current system load/contention.

## Conclusion

1. The targeted CPU hotspots were removed from top sampled symbols.
2. End-to-end decode throughput is still constrained by broader overheads/variance, especially for split decode.
3. Next optimization focus remains amortizing split decode prep and extending packed submission/readback symmetry to encode.
