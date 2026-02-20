# 2026-02-20 Master Baseline (rANS / LZR focus)

## Context

- Date: 2026-02-20
- Commit: `4a4f721`
- Branch: `codex/advance-roadmap` (fast-forwarded to `origin/master`)
- CPU: Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz
- Logical CPUs: 16

## Commands Executed

1. `./scripts/profile.sh --stage rans --size 1048576 --iterations 300 --webgpu`
2. `./scripts/profile.sh --stage rans --decompress --size 1048576 --iterations 300 --webgpu`
3. Fallback (because `profile.sh`/samply failed in this environment):
   - `cargo run --profile profiling --example profile --features webgpu -- --stage rans --size 1048576 --iterations 300 --rans-interleaved --rans-interleaved-states 4 --rans-chunked --rans-chunk-bytes 2048 --rans-gpu-batch 3`
   - `cargo run --profile profiling --example profile --features webgpu -- --stage rans --decompress --size 1048576 --iterations 300 --rans-interleaved --rans-interleaved-states 4 --rans-chunked --rans-chunk-bytes 2048 --rans-gpu-batch 3`
4. `cargo bench --bench stages_rans --features webgpu`
5. `./scripts/bench.sh -n 5 -p lzr,lzf,deflate`

## Artifacts

- `docs/generated/2026-02-20-profile-sh-rans-1mb-webgpu.txt`
- `docs/generated/2026-02-20-profile-sh-rans-decode-1mb-webgpu.txt`
- `docs/generated/2026-02-20-profile-direct-rans-encode-1mb-webgpu-escalated.txt`
- `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-escalated.txt`
- `docs/generated/2026-02-20-cargo-bench-stages-rans-webgpu-escalated.txt`
- `docs/generated/2026-02-20-bench-sh-n5-lzr-lzf-deflate.txt`

## Key Results

### A. rANS stage profile (1MB, 300 iters, WebGPU path)

From direct profile fallback:

- Encode: **49.3 MB/s**
- Decode: **65.7 MB/s**

Note: `scripts/profile.sh` failed due `samply` recording failure in this environment, so direct profile binary output is used for throughput values.

### B. `stages_rans` (Criterion, with `webgpu` feature)

Throughput medians from `docs/generated/2026-02-20-cargo-bench-stages-rans-webgpu-escalated.txt`:

| Case | 8KB | 64KB | 4MB |
|---|---:|---:|---:|
| `rans/encode` | 137.28 MiB/s | 127.24 MiB/s | 123.10 MiB/s |
| `rans/decode` | 117.56 MiB/s | 106.46 MiB/s | 113.01 MiB/s |
| `rans/encode_chunked_cpu` | 47.649 MiB/s | 46.477 MiB/s | 47.357 MiB/s |
| `rans/decode_chunked_cpu` | 212.10 MiB/s | 131.23 MiB/s | 136.30 MiB/s |
| `rans/encode_chunked_gpu` | 914.93 KiB/s | 5.4243 MiB/s | 48.326 MiB/s |
| `rans/decode_chunked_gpu` | 1.5955 MiB/s | 11.448 MiB/s | 76.964 MiB/s |

### C. Corpus benchmark (`bench.sh -n 5 -p lzr,lzf,deflate`)

Compression:

- `gzip`: 3.81 MB (28.6%), 5.9 MB/s
- `pz-lzr`: 5.41 MB (40.6%), 20.3 MB/s
- `pz-lzf`: 5.56 MB (41.7%), 21.0 MB/s
- `pz-deflate`: 5.54 MB (41.6%), 24.4 MB/s

Decompression:

- `gzip`: 33.7 MB/s
- `pz-lzr`: 26.9 MB/s
- `pz-lzf`: 25.1 MB/s
- `pz-deflate`: 26.0 MB/s

## Notes for Next Phase (P0-A Sprint)

1. Stage parity remains intact, but GPU chunked rANS throughput is still below CPU chunked decode at tested sizes.
2. 4MB encode shows near parity (`encode_chunked_gpu` 48.326 MiB/s vs `encode_chunked_cpu` 47.357 MiB/s), but decode remains materially behind (`76.964` vs `136.30` MiB/s).
3. This keeps the Slice 4 perf gate open; next work should target host overhead and decode-path kernel throughput.
