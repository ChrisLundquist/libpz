# 2026-02-20 Bench: Post Packed Split Encode Pass

## Scope

Benchmark current `codex/advance-roadmap` head after:

1. split decode prep reuse
2. shared-table encode table-buffer reuse
3. packed shared-table split encode path

## Commands

1. `cargo run --profile profiling --example profile --features webgpu -- --stage rans --size 1048576 --iterations 300`
2. `cargo run --profile profiling --example profile --features webgpu -- --stage rans --decompress --size 1048576 --iterations 300`
3. `cargo run --profile profiling --example profile --features webgpu -- --stage rans --size 1048576 --iterations 300 --rans-independent-block-bytes 262144`
4. `cargo run --profile profiling --example profile --features webgpu -- --stage rans --decompress --size 1048576 --iterations 300 --rans-independent-block-bytes 262144`
5. `cargo run --profile profiling --example profile --features webgpu -- --stage rans --size 1048576 --iterations 300 --rans-independent-block-bytes 65536`
6. `cargo run --profile profiling --example profile --features webgpu -- --stage rans --decompress --size 1048576 --iterations 300 --rans-independent-block-bytes 65536`
7. `cargo bench --bench stages_rans --features webgpu`

## Results (captured)

Profile runs (1MB, 300 iters):

1. encode defaults: 207.1 MB/s
2. decode defaults: 180.4 MB/s
3. encode split 256KB: 211.6 MB/s
4. decode split 256KB: 179.1 MB/s
5. encode split 64KB: 217.2 MB/s
6. decode split 64KB: 180.7 MB/s

Criterion `stages_rans` highlights (4MB):

1. `rans/encode/4194304`: 177.08-190.20 MiB/s
2. `rans/decode/4194304`: 170.98-176.41 MiB/s
3. `rans/encode_chunked_cpu/4194304`: 72.624-73.850 MiB/s
4. `rans/decode_chunked_cpu/4194304`: 205.88-210.59 MiB/s

## Important Caveat

The profile outputs above do not include the expected WebGPU stage banners (`using webgpu chunked rANS path`), indicating likely CPU fallback in this environment for these direct profile commands.

The `cargo bench --bench stages_rans --features webgpu` output also does not report `encode_chunked_gpu`/`decode_chunked_gpu` rows in this run.

These numbers are therefore recorded as environment trace data, not as trusted GPU gate evidence for the roadmap.

## Artifacts

1. `docs/generated/2026-02-20-profile-direct-rans-encode-1mb-webgpu-defaults-post-packed-encode-pass.txt`
2. `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-defaults-post-packed-encode-pass.txt`
3. `docs/generated/2026-02-20-profile-direct-rans-encode-1mb-webgpu-independent-blocks-256k-post-packed-encode-pass.txt`
4. `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-independent-blocks-256k-post-packed-encode-pass.txt`
5. `docs/generated/2026-02-20-profile-direct-rans-encode-1mb-webgpu-independent-blocks-64k-post-packed-encode-pass.txt`
6. `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-independent-blocks-64k-post-packed-encode-pass.txt`
7. `docs/generated/2026-02-20-cargo-bench-stages-rans-webgpu-post-packed-encode-pass.txt`
