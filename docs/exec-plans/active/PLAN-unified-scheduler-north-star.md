# Unified Scheduler North Star: CPU/GPU Full-Pipeline Compression

**Created:** 2026-02-15
**Status:** Partially complete — see status table below
**Priority:** P1
**Owner:** Engineering team

## Status (updated 2026-03-01)

| Phase | Description | Status | Notes |
|-------|-------------|--------|-------|
| Phase 1 | Chunked GPU rANS kernels | **DONE** (Slices 0-3 PASS, Slice 4 FAIL) | Kernels work but are 0.77x CPU encode, 0.54x CPU decode. See PLAN-p0a. |
| Phase 2 | GPU demux kernel | **DEFERRED** | Blocked on Phase 1 perf gate. Low value without competitive GPU entropy. |
| Phase 3 | Scheduler GpuBatch task type | **DONE** (via PR #101) | Implemented as `UnifiedTask::StageGpu` + `FusedGpu` variants with dedicated GPU coordinator thread. |
| Phase 4 | Partitioning policy | **PARTIALLY DONE** | Auto GPU/CPU routing works. Formal crossover benchmarking not done. |
| Phase 5 | On-device pipeline chaining | **NOT STARTED** | Requires Phase 2 demux kernel. |

**Key finding:** GPU rANS is not competitive with CPU at current block sizes (256KB-1MB). The serial data dependency in rANS limits GPU thread count to ~300 per stream, well below the ~8K-16K needed for saturation. Independent block splitting was tried extensively (15+ iterations) and regressed to 0.3-0.6x CPU. See `PLAN-p0a-gpu-rans-vertical-slice.md` for full details.

**Current strategy:** GPU does LZ77 match-finding (where it excels), CPU does entropy coding (where it's faster). The unified scheduler overlaps them. See `docs/design-docs/gpu-strategy.md` for the full analysis.

## Problem

The current GPU pipeline hands LZ77 matches back to CPU for entropy encoding, leaving ~40% of pipeline time on CPU even when GPU hardware is available. The unified scheduler prototype (PR #91) demonstrates cross-stage work sharing but is CPU-only. To reach nvCOMP-class GPU throughput, the full pipeline (transform + entropy) must execute on device, orchestrated by a scheduler that can partition work between CPU and GPU workers.

## Objective

1. ~~GPU rANS kernels (encode + decode) that produce byte-identical output to CPU interleaved rANS.~~ **DONE** — kernels work, but perf gate failed (0.77x CPU).
2. GPU demux kernel to eliminate the CPU roundtrip between LZ77 and entropy stages. **DEFERRED.**
3. ~~Unified scheduler extended with GPU task types for on-device pipelines.~~ **DONE** — `StageGpu`, `FusedGpu` variants in `compress_parallel_unified` (PR #101).
4. Automatic CPU/GPU partitioning policy based on device occupancy and input characteristics. **PARTIALLY DONE.**

## Key Insights

**Scheduler compatibility.** The unified scheduler (PR #101) uses a `Mutex<VecDeque<UnifiedTask>>` + `Condvar` with a dedicated GPU coordinator thread. The `UnifiedTask` enum has three variants: `Stage` (CPU), `StageGpu` (single GPU stage), and `FusedGpu` (multi-stage GPU execution). Workers use `try_send()` on a bounded `SyncSender` to avoid deadlock, with CPU fallback when the channel is full.

**GPU occupancy requires chunk-level parallelism.** Sequential entropy coding (rANS, FSE) on GPU gives one thread per lane. Even with N=32 interleaved lanes x 3 semantic streams x 10 batched blocks = 960 threads — far below the ~8K-16K needed for GPU saturation. The existing GPU FSE kernels (`fse_encode.wgsl`, `fse_decode.wgsl`) use `@workgroup_size(1)` with one thread per lane for the same reason — entropy is inherently sequential per-state. This was confirmed empirically: GPU rANS at 0.77x CPU on encode and 0.54x on decode (PLAN-p0a Slice 4).

**GPU value comes from LZ77, not entropy.** The cooperative-stitch kernel (`lz77_coop.wgsl`) runs 1,788 parallel probes per position. On large files (4MB+), GPU LZ77 outperforms CPU. On small files (<256KB), GPU dispatch overhead dominates.

## Existing Assets

| Asset | Location | Status |
|-------|----------|--------|
| GPU LZ77 (4 variants + decode) | `kernels/lz77_*.wgsl` | Production |
| GPU Huffman encode | `kernels/huffman_encode.wgsl` | Production |
| GPU FSE encode/decode | `kernels/fse_encode.wgsl`, `fse_decode.wgsl` | Production |
| GPU rANS encode/decode | `kernels/rans_encode.wgsl`, `kernels/rans_decode.wgsl` | Functional, slower than CPU |
| CPU interleaved rANS | `src/rans.rs` (`encode_interleaved_n`, `decode_interleaved`) | Production |
| Unified scheduler (with GPU coordinator) | `src/pipeline/parallel.rs` (`compress_parallel_unified`) | Production (PR #101) |
| GPU buffer ring | `src/webgpu/lz77.rs` (`BufferRing`) | Production |

## Implementation Phases

### Phase 1: Chunked GPU rANS Kernels — DONE (perf gate FAIL)

GPU rANS encode and decode kernels implemented with chunked + lane-interleaved dispatch. CPU/GPU parity tests pass across all chunk sizes and lane counts. However, GPU is slower than CPU:

- GPU encode: 57.9 MB/s vs CPU 75.5 MB/s (0.77x)
- GPU decode: 103.0 MB/s vs CPU 191.7 MB/s (0.54x)

Independent block splitting was extensively explored (15+ iterations) to increase GPU occupancy. Results: 64KB split regressed to 35 MB/s decode (-50%), 256KB split was -7-15%. The overhead of per-block metadata, table normalization, and transfer costs outweighed occupancy gains.

Go/no-go result: Slice 3 (parity) PASS. Slice 4 (performance) FAIL.

See `PLAN-p0a-gpu-rans-vertical-slice.md` for full execution history.

### Phase 2: GPU Demux Kernel — DEFERRED

Blocked on Phase 1 performance gate. Without competitive GPU entropy, on-device demux provides minimal benefit (saves ~1-2ms CPU demux time per block while entropy takes 50-400ms).

LzSeq pipelines already have an on-device fused match+demux path via `lzseq_encode_gpu()`.

### Phase 3: Scheduler GPU Task Type — DONE (PR #101)

Implemented as the unified scheduler refactor:
- `UnifiedTask::StageGpu(stage_idx, block_idx)` for single GPU stages
- `UnifiedTask::FusedGpu(start, end, block_idx)` for multi-stage GPU execution
- Dedicated GPU coordinator thread with batch-collect for LZ77 ring-buffered overlap
- GPU-to-CPU fallback on failure (re-enqueue as `Stage(0, block_idx)`)
- Bounded channel with `try_send()` to prevent deadlock

### Phase 4: Partitioning Policy — PARTIALLY DONE

Current behavior:
- GPU routing is automatic when `webgpu_engine` is present in `CompressOptions`
- `gpu_fused_span()` returns `Some((0, 1))` for Lzr and LzSeqR (both stages on GPU)
- Workers fall back to CPU when GPU channel is full
- GPU failures trigger CPU retry

Not yet done:
- Formal `SchedulerPolicy` enum (`CpuOnly`, `GpuOnly`, `Auto`)
- Benchmark-driven crossover thresholds
- CLI exposure via `--scheduler-policy`
- Consideration: `gpu_fused_span()` may be counterproductive since GPU entropy is slower than CPU

### Phase 5: On-Device Pipeline Chaining — NOT STARTED

Requires Phase 2 demux kernel. Would eliminate host-device copies between LZ77 and entropy stages via single command buffer submission. Currently blocked and low priority given Phase 1 perf gate failure.

## Benchmark and Validation Protocol

1. GPU rANS stage: `cargo bench --bench stages_rans --features webgpu`
2. Full pipeline: `./scripts/bench.sh --webgpu -n 5 -p lzr`
3. Cross-device equivalence: `cargo test --features webgpu` (CPU encode -> GPU decode and vice versa)
4. Profile harness: `./scripts/profile.sh --stage rans --size 1048576 --features webgpu`
5. Memory budget: `./scripts/gpu-meminfo.sh` before and after each phase

## Guardrails

1. CPU/GPU output must be byte-identical per algorithm path — no "GPU-only" format variants.
2. CPU path performance must not regress (unified scheduler overhead budget: <1% on CPU-only runs).
3. GPU path must degrade gracefully to CPU when device is unavailable or submit fails.

## Risks

1. **rANS encode backward-pass is hard to parallelize within a single lane.** Confirmed: chunking provides chunk x lane parallelism, but this caps at ~300 threads per stream — insufficient for GPU saturation. Independent block splitting was tried and failed.
2. **Chunk size is a ratio/throughput tradeoff.** Confirmed: shared frequency table across chunks mitigates ratio loss. Best encode throughput at chunk=2048, best decode at chunk=4096.
3. **Demux kernel memory amplification.** Unconfirmed (Phase 2 deferred). Two-pass approach planned.
4. **Partitioning heuristic is workload-dependent.** The current approach (GPU for LZ77, CPU for entropy) avoids this problem by not requiring block-level partitioning decisions.

## Relationship to Other Plans

- **PLAN-p0a-gpu-rans-vertical-slice.md**: Full execution history of GPU rANS kernel development with benchmarks.
- **gpu-strategy.md**: High-level GPU compression strategy document.
- **PLAN-competitive-roadmap.md** Phase 4 (nvCOMP-style batch track): this roadmap is the enabling work for that throughput target.

## Recommended Next Actions

1. **Reconsider `gpu_fused_span()` for Lzr/LzSeqR** — fusing entropy on GPU is currently slower than letting CPU handle it. The FusedGpu path should only activate when GPU entropy becomes competitive.
2. **Improve GPU match quality** — closing the ratio gap (41% -> 35%) would make GPU pipelines more useful. Shared-memory far-window probes and hash-guided long matches are candidates.
3. **Defer full on-device chaining** until GPU entropy crosses the CPU parity threshold.
