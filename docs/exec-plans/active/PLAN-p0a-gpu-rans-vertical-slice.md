# P0-A Plan: GPU Chunked rANS Vertical Slice (Highest-Leverage Track)

## Decision

Selected track: **P0-A GPU chunked rANS kernels**.

Rationale: this is the critical path dependency for full GPU LZR. Without GPU rANS entropy, the architecture remains CPU-heavy even with strong GPU LZ77, and downstream scheduler/policy work cannot realize meaningful end-to-end gains.

## Leverage Evaluation

| Track | Expected Throughput Impact | Dependency Unlock | Execution Risk | Net Leverage |
|---|---:|---:|---:|---:|
| P0-A GPU chunked rANS kernels | High | Very High | Medium/High | **Highest** |
| P0-B GPU demux + chaining | High | High | Medium | High (but partly blocked by P0-A for LZR) |
| P0-C Scheduler GPU batch integration | Medium | Medium | Medium | Medium |
| P0-D Policy/crossover automation | Low/Medium | Low | Low | Medium/Low |

Interpretation:

1. P0-B/C/D are valuable, but their practical payoff is capped until P0-A exists for the LZR path.
2. P0-A is the most leveraged first move despite higher implementation risk.

## Scope of This Plan

In scope:

1. GPU chunked rANS decode + encode kernels that match CPU chunked wire format.
2. `WebGpuEngine` APIs for chunked rANS stage operations.
3. CPU/GPU parity tests and stage benchmarks.

Out of scope for this slice:

1. Scheduler `GpuBatch` integration.
2. End-to-end policy defaults.
3. Full on-device LZ77->demux->rANS chaining.

## Execution Status (2026-02-17)

Current implementation status:

1. Slice 0 completed: WebGPU rANS kernels/module scaffolding added and lazy pipeline registration wired.
2. Slice 1 completed: GPU decode-first path implemented for CPU-compatible interleaved and chunked rANS payloads.
3. Slice 2 completed: GPU chunked encode implemented and serialized into CPU chunked wire format (`rans::encode_chunked` compatible), plus fallback-compatible behavior for chunk metadata overflow.
4. Slice 3 completed: CPU/GPU parity matrix added in WebGPU tests (lanes 4/8, chunk sizes 1KB/4KB/8KB/16KB, repetitive/text/binary/small-edge patterns).
5. Slice 4 measured (provisional): profiling harness now exercises WebGPU rANS stage path when available; current results do **not** show a stage-level gain vs CPU chunked reference.

Latest stage numbers (1MB, 300 iterations):

1. GPU chunked rANS default stage profile path (`lanes=4`, `chunk=2048`, `--rans-gpu-batch=3`): encode 57.9 MB/s, decode 103.0 MB/s (`a488474` + perf tuning, 2026-02-17).
2. Prior rebased master default (`lanes=4`, `chunk=8192`, `--rans-gpu-batch=2`): encode 35.2 MB/s, decode 87.1 MB/s.
3. CPU chunked interleaved rANS reference (`--no-default-features --rans-interleaved --rans-chunked --rans-interleaved-states 4 --rans-chunk-bytes 8192`): encode 75.5 MB/s, decode 191.7 MB/s.
4. Relative vs CPU reference: GPU is ~0.77x CPU on encode and ~0.54x CPU on decode for this host/device setup.
5. Recent perf deltas:
   - stage baseline improvement vs prior master default: encode +64.5%, decode +18.3%
   - larger host-side wins came from bulk LE pack/unpack in `src/webgpu/rans.rs` and lane-specialized WGSL entry points (`wg4`/`wg8`/`wg64`)
   - chunk/batch sweep indicates a throughput sweet spot near `chunk=2048` with `batch=3` for this host/device

Interim Go/No-Go:

1. Parity gate (Slice 3): PASS.
2. Stage perf gate (Slice 4): FAIL (no clear GPU advantage yet).
3. Recommendation: hold promotion to P0-B until GPU stage throughput improves materially.

### Execution Update (2026-02-20, commit `4a4f721`)

1. Re-ran Slice 4 baseline commands on latest master-aligned head and captured dated artifacts:
   - `docs/generated/2026-02-20-master-baseline.md`
   - `docs/generated/2026-02-20-cargo-bench-stages-rans-webgpu-escalated.txt`
   - `docs/generated/2026-02-20-bench-sh-n5-lzr-lzf-deflate.txt`
2. `scripts/profile.sh` stage commands still fail in this environment due `samply` recorder constraints; direct profile binary fallback was used for throughput values.
3. Direct 1MB WebGPU rANS stage profile (300 iterations, lanes=4, chunk=2048, batch=3):
   - encode: 49.3 MB/s
   - decode: 65.7 MB/s
4. Performed a compact chunk/batch sweep (`docs/generated/2026-02-20-rans-webgpu-sweep-1mb.tsv`) at 1MB, 200 iterations:
   - best encode: 58.5 MB/s (`chunk=2048`, `batch=3`)
   - best decode: 83.5 MB/s (`chunk=4096`, `batch=3`)
   - this sweep was taken before async depth retuning (ring depth still capped at 3).
5. `cargo bench --bench stages_rans --features webgpu` now includes GPU rows outside sandbox:
   - `encode_chunked_gpu/4194304`: 48.326 MiB/s
   - `decode_chunked_gpu/4194304`: 76.964 MiB/s
   - versus chunked CPU at 4MB: 47.357 MiB/s encode, 136.30 MiB/s decode.
6. Implemented async pipelining depth pass:
   - increased rANS batched pending-ring cap from 3 to 8 in `src/webgpu/rans.rs` (`RANS_MAX_PENDING_RING_DEPTH`).
7. Re-ran extended batch sweeps after ring-depth increase:
   - `docs/generated/2026-02-20-rans-webgpu-decode-batch-extended-1mb-after-ring8.tsv`
   - `docs/generated/2026-02-20-rans-webgpu-encode-batch-extended-1mb-after-ring8.tsv`
   - best region shifted to `gpu_batch=5..8` (higher than pre-change cap).
8. Retuned profiling defaults in `examples/profile.rs`:
   - `DEFAULT_RANS_GPU_BATCH`: 3 -> 6 (via intermediate 5)
   - `DEFAULT_RANS_GPU_CHUNK_BYTES`: unchanged at 2048.
9. Direct 1MB default-profile check after retune (`docs/generated/2026-02-20-rans-webgpu-async-pipelining-pass.md`):
   - final defaults (`batch=6`): encode 51.0 MB/s, decode 73.4 MB/s
   - vs original baseline (`batch=3`): encode +3.4%, decode +11.7%
10. Targeted correctness verification after async-depth change:
   - `cargo test --features webgpu batched` passed (12/12 tests).
11. Implemented async completion pass (decode path):
   - `rans_decode_chunked_payload_gpu_batched()` now batches completed work and performs batched output readback via `complete_rans_chunked_payload_decode_batch()`.
   - This reduces per-output submit/map/poll overhead in decode-heavy batched runs while keeping memory bounded by ring-sized drains.
12. Direct 1MB default-profile checks after batched decode readback:
   - ring-depth-only defaults: encode 50.6 MB/s, decode 66.2 MB/s
   - after readback + final retune: encode 51.0 MB/s, decode 73.4 MB/s
   - decode uplift over ring-depth-only defaults: +10.9%
   - artifact: `docs/generated/2026-02-20-rans-webgpu-async-pipelining-pass.md`
13. Priority order for remaining work is now explicit:
   - first: async submission/completion improvements (this pass)
   - second: nvCOMP-style independent stream/block splitting
   - third: additional Huffman GPU expansion only if decode-heavy benchmarks justify it
14. Started nvCOMP-style independent-block split probe (profiling-harness stage path):
   - added `--rans-independent-block-bytes` in `examples/profile.rs` to split one input into independent blocks and run batched GPU encode/decode over them.
   - probe artifact: `docs/generated/2026-02-20-rans-webgpu-independent-block-split-probe.md`
   - initial result: naive split path regressed vs current defaults on this host/device.
   - follow-up: batched encode completion/readback in `src/webgpu/rans.rs` improved split encode throughput materially (notably for 16-block case), but split decode remains below non-split defaults.
15. Added shared-table split encode path:
   - new API in `src/webgpu/rans.rs`: `rans_encode_chunked_payload_gpu_batched_shared_table(...)`.
   - profiling split mode now seeds one normalized table from full input and reuses it across independent blocks.
16. Added shared-table split decode path:
   - new API in `src/webgpu/rans.rs`: `rans_decode_chunked_payload_gpu_batched_shared_table(...)`.
   - split decode profile path now reuses one precomputed GPU table across the independent-block batch.
17. Added packed shared-table decode submission path:
   - shared-table decode now has a packed mode that consolidates split payloads into one decode dispatch/readback cycle.
   - packed mode is currently gated to `>= 8` non-empty payloads (`RANS_PACKED_SHARED_DECODE_MIN_PAYLOADS`) to avoid regressions on small split sets.
18. Latest split results after packed-decode gating:
   - 256KB split (4 blocks; falls back to prior path): encode 34.6 MB/s; decode typically 56.2-56.5 MB/s (one outlier run at 68.9 MB/s), still below the earlier 58.2 MB/s readback-only split result.
   - 64KB split (16 blocks; packed decode active): encode 21.3 MB/s; decode 35.3-35.4 MB/s, improving over 32.9-33.0 MB/s shared-table decode and slightly above 34.5 MB/s readback-only split decode.
19. Interim conclusion unchanged: Slice 4 perf gate remains open; do not promote to P0-B yet.
20. Decode hotspot follow-up pass (2026-02-20):
   - in `src/webgpu/rans.rs`, replaced repeated decode-side `write_packed_u16_slice(...)` packing with direct `u16` placement + one bulk pack, and removed per-call seed-frequency counting in shared-table split decode setup.
   - new hotspot report: `docs/generated/2026-02-20-rans-webgpu-hotspot-pass.md`
   - targeted sampled hotspots (`write_packed_u16_slice`, `simd::avx2::byte_frequencies`) dropped out of split decode top-symbol output after this change.
21. Latest decode reruns after hotspot pass (1MB, 300 iters):
   - defaults: 69.0-69.1 MB/s (stable on these runs).
   - 256KB split (4 blocks): 53.2-71.7 MB/s (high variance under current host contention).
   - 64KB split (16 blocks): 29.1-34.5 MB/s.
   - interpretation: hotspot fixes are necessary but not sufficient; split decode still needs prep amortization and lower-overhead submission/completion symmetry.
22. Implemented split decode prep reuse pass (2026-02-20):
   - refactored packed shared-table split decode in `src/webgpu/rans.rs` into prepare + execute phases.
   - added `rans_decode_chunked_payload_gpu_batched_shared_table_repeated(...)` so repeated runs reuse table setup and packed split preparation.
   - updated `examples/profile.rs` split decode path to use the repeated API.
   - added test coverage: `test_rans_chunked_decode_gpu_batched_shared_table_repeated_round_trip`.
23. Prep reuse pass validation status:
   - `cargo fmt --check`, `cargo clippy --features webgpu -- -D warnings`, and `cargo test --features webgpu batched` all pass.
   - implementation notes: `docs/generated/2026-02-20-rans-webgpu-split-decode-prep-reuse-pass.md`.
24. Non-approved sandbox profile reruns were captured but likely CPU fallback (missing WebGPU banners), so these are retained for traceability only and not used as Slice 4 GPU gate evidence:
   - `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-defaults-prep-cache-pass.txt`
   - `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-independent-blocks-256k-prep-cache-pass.txt`
   - `docs/generated/2026-02-20-profile-direct-rans-decode-1mb-webgpu-independent-blocks-64k-prep-cache-pass.txt`

## Existing Assets We Reuse

1. CPU chunked rANS reference in `src/rans.rs` (`encode_chunked`, `decode_chunked`).
2. Pipeline chunked framing support and flag bit in `src/pipeline/stages.rs` (`RANS_CHUNKED_FLAG`).
3. Established WebGPU pipeline patterns in `src/webgpu/fse.rs` and `src/webgpu/lz77.rs`.
4. Benchmark/profiling guardrails in `scripts/bench.sh` and `scripts/profile.sh`.

## Execution Plan (Vertical Slices)

## Slice 0: GPU Entropy Module Scaffolding

Files:

1. `kernels/rans_encode.wgsl` (skeleton)
2. `kernels/rans_decode.wgsl` (skeleton)
3. `src/webgpu/mod.rs` (pipeline registration)
4. `src/webgpu/rans.rs` (new module with API stubs)

Exit criteria:

1. Shaders compile.
2. Engine initializes and lazy pipeline creation works.
3. No behavioral changes yet.

## Slice 1: GPU Decode First (Parity Anchor)

Why decode first:

1. It validates table interpretation, chunk metadata parsing, lane sequencing, and renormalization with lower framing complexity than encode output construction.
2. It gives early CPU->GPU parity signal using existing CPU chunked encoder.

Work:

1. Implement chunked rANS decode kernel path (`rans_decode_chunked_gpu`).
2. Host-side packing/unpacking for metadata and lane word streams.
3. Add tests: CPU encode_chunked -> GPU decode == input.

Exit criteria:

1. Parity tests pass across chunk sizes and lane counts.
2. Invalid metadata is rejected safely.

## Slice 2: GPU Encode

Work:

1. Implement chunked rANS encode kernel path (`rans_encode_chunked_gpu`).
2. Serialize output exactly in CPU chunked wire format.
3. Add tests: GPU encode -> CPU decode_chunked == input.

Exit criteria:

1. Bit-compatible output format accepted by CPU decoder.
2. Round-trip and cross-path parity tests pass.

## Slice 3: CPU/GPU Cross-Parity Matrix

Matrix:

1. CPU encode -> CPU decode
2. CPU encode -> GPU decode
3. GPU encode -> CPU decode
4. GPU encode -> GPU decode

Coverage dimensions:

1. chunk sizes: 1KB, 4KB, 8KB, 16KB
2. state counts: 4, 8
3. data patterns: repetitive, mixed text, binary/high-entropy, small-edge inputs

Exit criteria:

1. Full matrix green.
2. No format divergence.

## Slice 4: Stage-Level Performance Sweep

Commands:

```bash
cargo bench --bench stages_rans --features webgpu
./scripts/profile.sh --stage rans --size 1048576 --iterations 300 --webgpu
./scripts/profile.sh --stage rans --decompress --size 1048576 --iterations 300 --webgpu
```

Exit criteria:

1. GPU stage shows clear large-input advantage vs CPU chunked reference.
2. Ratio deltas per chunk size documented.
3. Default candidate chunk size selected (or explicitly deferred).

## Technical Constraints (Non-Negotiable)

1. Preserve current chunked wire format as CPU/GPU common contract.
2. No GPU-only format fork.
3. Keep CPU default conservative until stage and pipeline evidence support promotion.
4. Fail safely on device unavailability and keep CPU fallback correct.

## Risks and Mitigations

1. Risk: sequential dependence per lane limits occupancy.
   - Mitigation: chunk-parallel dispatch and multi-lane batching; tune chunk sizes only after parity is solid.
2. Risk: host packing/unpacking overhead erodes gains.
   - Mitigation: optimize layout after parity; do not pre-optimize before correctness gates.
3. Risk: debugging complexity in WGSL arithmetic path.
   - Mitigation: decode-first approach and small deterministic fixtures before large corpus tests.

## Go/No-Go Criteria for Continuing to P0-B

Proceed to P0-B (GPU demux + chaining) only if all are true:

1. Slice 3 parity matrix fully green.
2. Slice 4 shows clear stage-level GPU gain at large sizes.
3. No correctness regressions in existing CPU paths.

If any fail, hold this track and fix parity/perf before integrating scheduler/policy work.
