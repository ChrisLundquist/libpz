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

## Interpretation

1. Independent-block splitting is still below the non-split default path on this host/device.
2. Batched encode readback materially improves split encode throughput (especially at higher block counts), confirming per-block readback overhead was a major contributor.
3. Shared-table seeding plus decode-side table reuse alone did not recover split throughput; table setup overhead is not the primary remaining bottleneck.
4. Packed multi-block decode submission helps the high-block-count (64KB/16-block) case and is now gated so smaller split sets keep the prior path.
5. Split decode remains below non-split defaults overall; next wins likely require reducing split decode preparation/packing CPU overhead and adding analogous packed submission for encode.

## Next Implementation Targets

1. Add packed split decode preparation cache so payload parsing/state packing are amortized across iterations (not rebuilt every decode call).
2. Extend packed submission to split encode path (single metadata/input staging + consolidated readback) so both directions share the same model.
3. Keep block splitting as a scheduler-level option, with thresholds tied to block count/device break-even data.
