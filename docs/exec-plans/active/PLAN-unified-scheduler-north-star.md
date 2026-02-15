# Unified Scheduler North Star: CPU/GPU Full-Pipeline Compression

**Created:** 2026-02-15
**Status:** Planned
**Priority:** P1
**Owner:** Engineering team

## Problem

The current GPU pipeline hands LZ77 matches back to CPU for entropy encoding, leaving ~40% of pipeline time on CPU even when GPU hardware is available. The unified scheduler prototype (PR #91) demonstrates cross-stage work sharing but is CPU-only. To reach nvCOMP-class GPU throughput, the full pipeline (transform + entropy) must execute on device, orchestrated by a scheduler that can partition work between CPU and GPU workers.

## Objective

1. GPU rANS kernels (encode + decode) that produce byte-identical output to CPU interleaved rANS.
2. GPU demux kernel to eliminate the CPU roundtrip between LZ77 and entropy stages.
3. Unified scheduler extended with a `GpuBatch` task type for full on-device pipelines.
4. Automatic CPU/GPU partitioning policy based on device occupancy and input characteristics.

## Key Insight

If the GPU owns all stages on device (LZ77 → demux → rANS), a "GPU block" is just another synchronous task from the scheduler's perspective: one worker submits a batch, blocks on device fence, gets back compressed output. The existing `Mutex<VecDeque>` + `Condvar` task model survives — it just needs a `GpuBatch(Range<usize>)` variant alongside the current `Stage0`/`Stage1` CPU variants.

## Scope

### In scope

1. GPU rANS WGSL kernels (encode + decode).
2. GPU demux kernel (LZ77 match → stream split).
3. Scheduler `GpuBatch` task variant and partitioning heuristics.
4. CPU/GPU output equivalence tests.
5. On-device pipeline chaining (single command buffer submission).

### Out of scope (initial)

1. GPU-only decode path (CPU decode remains the reference).
2. GPU BWT+rANS pipelines (BBW).
3. Adaptive per-block CPU↔GPU migration mid-batch.
4. Multi-GPU dispatch.

## Existing Assets

| Asset | Location | Status |
|-------|----------|--------|
| GPU LZ77 (4 variants + decode) | `kernels/lz77_*.wgsl` | Production |
| GPU Huffman encode | `kernels/huffman_encode.wgsl` | Production |
| GPU FSE encode/decode | `kernels/fse_encode.wgsl`, `fse_decode.wgsl` | Production |
| CPU interleaved rANS | `src/rans.rs` (`encode_interleaved_n`, `decode_interleaved`) | PR #91 |
| Unified scheduler prototype | `src/pipeline/parallel.rs` (`compress_parallel_unified_lz_rans`) | PR #91 |
| GPU streaming pipeline | `src/pipeline/parallel.rs` (`compress_streaming_gpu`) | Production |
| GPU buffer ring | `src/webgpu/lz77.rs` (`BufferRing`) | Production |

**Critical gap:** No GPU rANS kernels exist. This is the prerequisite for everything else.

## Implementation Phases

### Phase 1: GPU rANS Kernels

**Goal:** Byte-identical GPU rANS encode/decode matching CPU interleaved format.

Tasks:

1. Implement `rans_encode.wgsl` — per-lane interleaved rANS encode.
   - Each GPU thread processes one lane (one rANS state).
   - Output format must match CPU `encode_interleaved_n`: `[scale_bits][freq_table][num_states][final_states][word_counts][lane_words...]`.
   - Backward-pass dependency within a lane is sequential per-thread; parallelism comes from N independent lanes.
2. Implement `rans_decode.wgsl` — per-lane interleaved rANS decode.
   - Decode hot path is multiply-add (GPU-friendly, no division via reciprocal table).
   - Word-aligned I/O (16-bit) suits GPU memory access patterns.
3. Add `WebGpuEngine` methods: `rans_encode_interleaved`, `rans_decode_interleaved`.
4. CPU/GPU equivalence tests: encode on GPU → decode on CPU, encode on CPU → decode on GPU, bit-exact both directions.

Acceptance:

1. Round-trip parity with CPU interleaved rANS for all tested inputs.
2. GPU rANS decode throughput >= 2x CPU single-stream decode on 1MB+ data.
3. No new WGSL compilation warnings.

### Phase 2: GPU Demux Kernel

**Goal:** Eliminate CPU roundtrip between LZ77 matching and entropy encoding.

Tasks:

1. Implement `demux_lz77.wgsl` — parallel stream splitting from LZ77 match output (offsets/lengths/literals).
   - Two-pass approach: count pass for output offsets, then scatter-write pass.
2. Wire through `WebGpuEngine::demux_lz77` with buffer management.
3. Chain: GPU LZ77 output buffer → GPU demux → GPU rANS encode (no host readback between stages).
4. Add on-device pipeline integration test (full Lzr encode on GPU).

Acceptance:

1. Demux output matches CPU `stage_demux_compress` byte-for-byte.
2. Full GPU Lzr encode produces output decompressible by existing CPU decode path.
3. Device memory stays within 2x current GPU LZ77 peak allocation.

### Phase 3: Scheduler GpuBatch Task Type

**Goal:** Extend unified scheduler to dispatch full-pipeline GPU batches alongside CPU tasks.

Tasks:

1. Add `UnifiedTask::GpuBatch(Range<usize>)` variant to the task enum in `src/pipeline/parallel.rs`.
2. Worker that claims a `GpuBatch`:
   - Submits full-pipeline batch to `WebGpuEngine` (LZ77 → demux → rANS in one submission).
   - Blocks on device fence.
   - Writes compressed blocks to indexed result slots.
3. Other workers continue pulling CPU `Stage0`/`Stage1` tasks concurrently.
4. Output assembly unchanged (same header + block table + data format via `assemble_multiblock_output`).

Acceptance:

1. Mixed CPU+GPU compression produces byte-identical output to CPU-only for same block inputs.
2. No deadlocks or liveness issues under error conditions (GPU submit failure, device lost).
3. Unified scheduler + GPU path round-trip tests for Lzr, LzssR, Lz78R.

### Phase 4: Partitioning Policy

**Goal:** Automatically decide which blocks go to GPU vs CPU.

Tasks:

1. Add `SchedulerPolicy` enum: `CpuOnly`, `GpuOnly`, `Auto`.
2. Auto policy considers: input size, block count, device memory budget, GPU warm-up cost.
3. Heuristic: small inputs (< crossover threshold) → CPU, large batches → GPU, remainder → CPU.
4. Expose `--scheduler-policy` in CLI (`src/bin/pz.rs`) and profile harness (`examples/profile.rs`) for A/B testing.
5. Benchmark sweep to find crossover points per pipeline.

Acceptance:

1. Auto policy matches or beats best-of(CPU-only, GPU-only) within 5% on standard corpus.
2. No regression on small-input latency (Auto never picks GPU when CPU is faster).
3. Crossover thresholds documented with benchmark data.

### Phase 5: On-Device Pipeline Chaining

**Goal:** Single command buffer submission for full encode pipeline.

Tasks:

1. Eliminate host↔device copies between stages: LZ77 output buffer feeds demux, demux output feeds rANS encode — all in a single command buffer submission.
2. Add buffer lifetime management for chained stages (reuse LZ77 output buffer after demux reads it).
3. Benchmark: single-submission vs multi-submission overhead.

Acceptance:

1. Single command buffer submission for full Lzr GPU encode.
2. Device memory peak <= 1.5x current GPU LZ77 peak (buffer reuse working).
3. >= 2x throughput improvement over current GPU LZ77 + CPU entropy path on 8MB+ inputs.

## Benchmark and Validation Protocol

1. GPU rANS stage: `cargo bench --bench stages --features webgpu -- rans`
2. Full pipeline: `./scripts/bench.sh --webgpu -n 5 -p lzr`
3. Cross-device equivalence: `cargo test --features webgpu` (CPU encode → GPU decode and vice versa)
4. Profile harness: `./scripts/profile.sh --stage rans --size 1048576 --webgpu`
5. Memory budget: `./scripts/gpu-meminfo.sh` before and after each phase

## Guardrails

1. CPU/GPU output must be byte-identical per algorithm path — no "GPU-only" format variants.
2. CPU path performance must not regress (unified scheduler overhead budget: <1% on CPU-only runs).
3. No phase starts without prior phase acceptance criteria met.
4. GPU path must degrade gracefully to CPU when device is unavailable or submit fails.

## Risks

1. **rANS encode backward-pass is hard to parallelize within a single lane.** Mitigation: interleaved format already partitions into N independent lanes; each lane is a separate GPU thread. Parallelism comes from inter-lane independence, not intra-lane parallelism.
2. **Demux kernel memory amplification.** Splitting one match stream into 3-4 output streams requires scatter writes with unpredictable output sizes. Mitigation: two-pass approach (count pass for offsets, then write pass), matching the existing GPU Huffman encode pattern.
3. **Partitioning heuristic is workload-dependent.** Wrong split wastes one device. Mitigation: conservative defaults (CPU-only unless input is large), user override via policy flag, benchmark-driven threshold tuning.
4. **Command buffer chaining complexity.** Buffer lifetime and synchronization across stages in a single submission. Mitigation: Phase 5 is last; validate correctness in Phases 1-4 with separate submissions first.

## Relationship to Other Plans

- **PLAN-interleaved-rans.md** Phase A (PR #91): provides the CPU interleaved rANS implementation and wire format that GPU kernels must match bit-for-bit.
- **PLAN-competitive-roadmap.md** Phase 4 (nvCOMP-style batch track): this roadmap is the enabling work for that throughput target.

## Immediate Next Actions

1. Prototype `rans_encode.wgsl` for single-lane rANS encode; validate against CPU `rans::encode` output.
2. Extend to N-lane interleaved encode matching `encode_interleaved_n` output format.
3. Add `WebGpuEngine::rans_encode` method and CPU↔GPU equivalence test.
4. Benchmark GPU rANS encode throughput vs CPU baseline on 256KB and 1MB inputs.
