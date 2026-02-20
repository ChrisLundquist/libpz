# 2026-02-20 rANS WebGPU Independent-Block Split Probe

## Scope

Initial nvCOMP-style experiment: split a single 1MB input into independent blocks and run GPU chunked rANS batched encode/decode over those blocks.

Implementation for this probe:

- Added `--rans-independent-block-bytes N` to `examples/profile.rs`.
- When set, stage profiling path splits input into independent blocks and uses:
  - `rans_encode_chunked_payload_gpu_batched` (encode)
  - `rans_decode_chunked_payload_gpu_batched` (decode)
- Follow-up optimization pass:
  - `src/webgpu/rans.rs` batched encode path now supports batched completion/readback (`complete_rans_chunked_payload_encode_batch`) to reduce per-block readback overhead.
- Additional follow-up pass:
  - added shared-table split encode API (`rans_encode_chunked_payload_gpu_batched_shared_table`) and wired profile split mode to seed one normalized table from the full input.
- Additional follow-up pass:
  - added shared-table split decode API (`rans_decode_chunked_payload_gpu_batched_shared_table`) to reuse one precomputed GPU table across independent-block decode batches.
- Additional follow-up pass:
  - added packed shared-table decode submission path in `src/webgpu/rans.rs`:
    - packs split payload words/states/chunk metadata into one decode dispatch/readback cycle when block count is high enough.
    - currently gated to `>= 8` non-empty payloads so smaller split sets keep the prior per-block decode path.
- Additional follow-up pass:
  - added split decode prep reuse path:
    - `rans_decode_chunked_payload_gpu_batched_shared_table_repeated(...)` now reuses shared-table decode setup and packed split decode preparation across repeated runs.
    - profiling split decode loop in `examples/profile.rs` now uses the repeated API.

No wire format changes were made in this probe.

## Commands

1. Baseline defaults (for comparison):
   - `cargo run --profile profiling --example profile --features webgpu -- --stage rans --size 1048576 --iterations 300`
   - `cargo run --profile profiling --example profile --features webgpu -- --stage rans --decompress --size 1048576 --iterations 300`
2. Independent 256KB blocks (4 blocks):
   - `cargo run --profile profiling --example profile --features webgpu -- --stage rans --size 1048576 --iterations 300 --rans-independent-block-bytes 262144`
   - `cargo run --profile profiling --example profile --features webgpu -- --stage rans --decompress --size 1048576 --iterations 300 --rans-independent-block-bytes 262144`
3. Independent 64KB blocks (16 blocks):
   - `cargo run --profile profiling --example profile --features webgpu -- --stage rans --size 1048576 --iterations 300 --rans-independent-block-bytes 65536`
   - `cargo run --profile profiling --example profile --features webgpu -- --stage rans --decompress --size 1048576 --iterations 300 --rans-independent-block-bytes 65536`
4. Independent split with shared-table seed (same stage flags on current head):
   - `cargo run --profile profiling --example profile --features webgpu -- --stage rans --size 1048576 --iterations 300 --rans-independent-block-bytes 262144`
   - `cargo run --profile profiling --example profile --features webgpu -- --stage rans --decompress --size 1048576 --iterations 300 --rans-independent-block-bytes 262144`
   - `cargo run --profile profiling --example profile --features webgpu -- --stage rans --size 1048576 --iterations 300 --rans-independent-block-bytes 65536`
   - `cargo run --profile profiling --example profile --features webgpu -- --stage rans --decompress --size 1048576 --iterations 300 --rans-independent-block-bytes 65536`

## Results (1MB, 300 iterations)

### Baseline defaults (batch=6, no independent split)

- Encode: 51.0 MB/s
- Decode: 73.4 MB/s

Source files:
- `docs/generated/2026-02-20-profile-direct-rans-encode-1mb-webgpu-defaults-final-batch6-seq.txt`
- `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-defaults-final-batch6-seq.txt`

### Independent split: 256KB blocks (4 blocks)

Before batched-encode readback:

- Encode: 30.3 MB/s
- Decode: 57.6 MB/s

After batched-encode readback:

- Encode: 35.8 MB/s
- Decode: 58.2 MB/s

After shared-table seed (+ decode table reuse):

- Encode: 33.8 MB/s (rerun: 34.0 MB/s)
- Decode: 55.2 MB/s (rerun: 55.7 MB/s)

After packed decode submission (gated policy active):

- Encode: 34.6 MB/s
- Decode: 56.2-56.5 MB/s typical (one high outlier run at 68.9 MB/s)

Source files:
- before:
  - `docs/generated/2026-02-20-profile-direct-rans-encode-1mb-webgpu-independent-blocks-256k.txt`
  - `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-independent-blocks-256k.txt`
- after:
  - `docs/generated/2026-02-20-profile-direct-rans-encode-1mb-webgpu-independent-blocks-256k-after-encode-batch-readback.txt`
  - `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-independent-blocks-256k-after-encode-batch-readback.txt`
- shared table:
  - `docs/generated/2026-02-20-profile-direct-rans-encode-1mb-webgpu-independent-blocks-256k-shared-table.txt`
  - `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-independent-blocks-256k-shared-table.txt`
  - `docs/generated/2026-02-20-profile-direct-rans-encode-1mb-webgpu-independent-blocks-256k-shared-table-rerun.txt`
  - `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-independent-blocks-256k-shared-table-rerun.txt`
- packed decode (gated):
  - `docs/generated/2026-02-20-profile-direct-rans-encode-1mb-webgpu-independent-blocks-256k-shared-table-packed-gated.txt`
  - `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-independent-blocks-256k-shared-table-packed-gated.txt`
  - `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-independent-blocks-256k-shared-table-packed-gated-rerun.txt`
  - `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-independent-blocks-256k-shared-table-packed-gated-rerun2.txt`

### Independent split: 64KB blocks (16 blocks)

Before batched-encode readback:

- Encode: 9.9 MB/s
- Decode: 34.2 MB/s

After batched-encode readback:

- Encode: 20.8 MB/s
- Decode: 34.5 MB/s

After shared-table seed (+ decode table reuse):

- Encode: 19.9 MB/s (rerun: 19.9 MB/s)
- Decode: 32.9 MB/s (rerun: 33.0 MB/s)

After packed decode submission (gated policy active):

- Encode: 21.3 MB/s
- Decode: 35.3-35.4 MB/s

Source files:
- before:
  - `docs/generated/2026-02-20-profile-direct-rans-encode-1mb-webgpu-independent-blocks-64k-seq.txt`
  - `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-independent-blocks-64k-seq.txt`
- after:
  - `docs/generated/2026-02-20-profile-direct-rans-encode-1mb-webgpu-independent-blocks-64k-after-encode-batch-readback.txt`
  - `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-independent-blocks-64k-after-encode-batch-readback.txt`
- shared table:
  - `docs/generated/2026-02-20-profile-direct-rans-encode-1mb-webgpu-independent-blocks-64k-shared-table.txt`
  - `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-independent-blocks-64k-shared-table.txt`
  - `docs/generated/2026-02-20-profile-direct-rans-encode-1mb-webgpu-independent-blocks-64k-shared-table-rerun.txt`
  - `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-independent-blocks-64k-shared-table-rerun.txt`
- packed decode (gated):
  - `docs/generated/2026-02-20-profile-direct-rans-encode-1mb-webgpu-independent-blocks-64k-shared-table-packed-gated.txt`
  - `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-independent-blocks-64k-shared-table-packed-gated.txt`
  - `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-independent-blocks-64k-shared-table-packed-gated-rerun.txt`

## Hotspot Follow-up Pass (Decode Prep)

Additional decode-prep optimization pass in `src/webgpu/rans.rs`:

1. Replaced repeated per-lane `write_packed_u16_slice(...)` calls with direct `u16` buffer placement plus one bulk `u16 -> u32` pack.
2. Removed per-call shared-table seed frequency counting from split decode setup and reused normalized table data derived from payload header.
3. Kept split wire format unchanged.

Follow-up decode measurements (1MB, 300 iterations):

1. Defaults: 69.0-69.1 MB/s
   - `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-defaults-hotspot-pass.txt`
   - `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-defaults-hotspot-pass-rerun.txt`
2. 256KB split (4 blocks): 53.2-71.7 MB/s (high variance)
   - `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-independent-blocks-256k-hotspot-pass.txt`
   - `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-independent-blocks-256k-hotspot-pass-rerun1.txt`
   - `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-independent-blocks-256k-hotspot-pass-rerun2.txt`
   - `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-independent-blocks-256k-hotspot-pass-rerun3.txt`
3. 64KB split (16 blocks): 29.1-34.5 MB/s
   - `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-independent-blocks-64k-hotspot-pass.txt`
   - `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-independent-blocks-64k-hotspot-pass-rerun.txt`
   - `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-independent-blocks-64k-hotspot-pass-rerun2.txt`

`samply` symbol diff for split decode indicates the targeted CPU hotspots were removed from top rows:

1. Before: `docs/generated/2026-02-20-samply-rans-decode-split64k-1mb-top35.txt`
2. After: `docs/generated/2026-02-20-samply-rans-decode-split64k-1mb-hotspot-pass-top35.txt`

## Prep Reuse Follow-up Pass

Follow-up implementation (roadmap item: amortize split decode prep across iterations):

1. Refactored packed shared-table split decode into prepare + execute phases in `src/webgpu/rans.rs`.
2. Added `rans_decode_chunked_payload_gpu_batched_shared_table_repeated(...)` and routed profile split decode through it.
3. Added GPU round-trip coverage for repeated shared-table decode in `src/webgpu/tests.rs`.
4. Detailed notes: `docs/generated/2026-02-20-rans-webgpu-split-decode-prep-reuse-pass.md`.

Measurement status:

1. Non-approved sandbox runs produced `*prep-cache-pass.txt` artifacts.
2. Those captures appear to have fallen back to CPU path (missing WebGPU path banners), so they are not used as GPU gate evidence.

## Interpretation

1. Independent-block splitting is still below the non-split default path on this host/device.
2. Batched encode readback materially improves split encode throughput (especially at higher block counts), confirming per-block readback overhead was a major contributor.
3. Shared-table seeding plus decode-side table reuse alone did not recover split throughput; table setup overhead is not the only remaining bottleneck.
4. Decode-prep hotspot pass removed `write_packed_u16_slice`/`byte_frequencies` from sampled top symbols, but end-to-end split decode remains variable and often below default path.
5. Packed multi-block decode submission helps the high-block-count (64KB/16-block) case and is now gated so smaller split sets keep the prior path.
6. Next wins likely require amortizing split decode preparation across iterations and adding analogous packed submission for encode.

## Next Implementation Targets

1. Re-run split decode profiling for the prep-reuse pass on stable WebGPU hardware to quantify true GPU delta.
2. Extend packed submission + prep reuse to split encode path so both directions share the same amortized model.
3. Add a low-noise benchmark mode for split decode (longer runs or isolated host) before changing scheduler defaults.
4. Keep block splitting as a scheduler-level option, with thresholds tied to block count/device break-even data.
