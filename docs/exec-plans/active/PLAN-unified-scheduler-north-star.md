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

## Key Insights

**Scheduler compatibility.** If the GPU owns all stages on device (LZ77 → demux → rANS), a "GPU block" is just another synchronous task from the scheduler's perspective: one worker submits a batch, blocks on device fence, gets back compressed output. The existing `Mutex<VecDeque>` + `Condvar` task model survives — it just needs a `GpuBatch(Range<usize>)` variant alongside the current `Stage0`/`Stage1` CPU variants.

**GPU occupancy requires chunk-level parallelism.** Sequential entropy coding (rANS, FSE) on GPU gives one thread per lane. Even with N=32 interleaved lanes × 3 semantic streams × 10 batched blocks = 960 threads — far below the ~8K-16K needed for GPU saturation. The existing GPU FSE kernels (`fse_encode.wgsl`, `fse_decode.wgsl`) use `@workgroup_size(1)` with one thread per lane for the same reason — entropy is inherently sequential per-state. This is tolerable today because GPU value comes from LZ77, not entropy. For rANS to be the primary GPU entropy path at high throughput, streams must be chunked into small independent segments (4-16KB each), each encoded separately, giving chunk × lane × stream × block parallelism:

| Chunk size | Chunks/300KB stream | × 4 lanes | × 3 streams | × 10 blocks | Total threads |
|------------|---------------------|-----------|-------------|-------------|---------------|
| 16KB | 19 | 76 | 228 | 2,280 | marginal |
| 4KB | 75 | 300 | 900 | 9,000 | good |
| 1KB | 300 | 1,200 | 3,600 | 36,000 | saturated |

The tradeoff: smaller chunks = more parallelism but more framing overhead (per-chunk final states, potential per-chunk frequency tables). The wire format must include chunk boundaries from the start — retrofitting chunking into a flat interleaved format is a format break. PLAN-interleaved-rans.md Phase C anticipates this with `entropy_chunk_bytes` as a tunable.

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

### Phase 1: Chunked GPU rANS Kernels

**Goal:** GPU rANS encode/decode with chunk-level parallelism and CPU decode compatibility.

The critical design decision is chunking granularity. A naive one-thread-per-lane kernel (like the existing FSE kernels) will not saturate the GPU. Streams must be split into independent chunks, each encoded with its own rANS state(s), to achieve sufficient thread count.

Tasks:

1. **Design chunked rANS wire format.** Extend the interleaved rANS framing with chunk boundaries:
   ```
   [scale_bits: u8]
   [freq_table: 256 × u16]        ← shared across all chunks (one table per stream)
   [num_chunks: u16]
   [chunk_original_lens: num_chunks × u16]
   per chunk:
     [num_states: u8]
     [final_states: N × u32]
     [word_counts: N × u32]
     [lane_words...]
   ```
   Frequency table is shared (computed from the full stream) to avoid per-chunk ratio loss. Each chunk carries only its own final states and word data.
2. **CPU chunked encode/decode.** Add `rans::encode_chunked` and `rans::decode_chunked` to `src/rans.rs` before writing GPU kernels. This gives a CPU reference implementation for equivalence testing and allows the format to be validated end-to-end without GPU hardware.
3. **Implement `rans_encode.wgsl`** — one workgroup per chunk, one thread per lane within chunk.
   - Backward-pass dependency within a lane is sequential per-thread; parallelism comes from chunk × lane independence.
   - Frequency table loaded once into workgroup shared memory.
   - Dispatch: `num_chunks × num_lanes` threads total (e.g., 75 chunks × 4 lanes = 300 threads per stream).
4. **Implement `rans_decode.wgsl`** — same dispatch structure.
   - Decode hot path is multiply-add (GPU-friendly, no division via reciprocal table).
   - Word-aligned I/O (16-bit) suits GPU memory access patterns.
5. **Add `WebGpuEngine` methods:** `rans_encode_chunked`, `rans_decode_chunked`.
6. **CPU/GPU equivalence tests:** encode on GPU → decode on CPU, encode on CPU → decode on GPU, bit-exact both directions. Include edge cases: single-chunk streams (below chunk threshold), empty chunks, maximum chunk count.
7. **Chunk size tuning sweep.** Benchmark 1KB, 4KB, 8KB, 16KB chunk sizes on 256KB and 1MB inputs. Find the Pareto frontier of GPU occupancy vs framing overhead (ratio loss).

Acceptance:

1. Round-trip parity between CPU chunked rANS and GPU chunked rANS for all tested inputs.
2. GPU rANS decode throughput >= 2x CPU single-stream decode on 1MB+ data.
3. Compression ratio within 0.5% of unchunked interleaved rANS at chosen default chunk size.
4. No new WGSL compilation warnings.

**Design note:** The unchunked interleaved format from PR #91 remains valid for CPU-only paths where GPU occupancy is irrelevant. The chunked format is a superset — a stream with `num_chunks=1` is equivalent to the unchunked format. The `RANS_INTERLEAVED_FLAG` in the per-stream compressed length field can be extended with a second flag bit (`RANS_CHUNKED_FLAG = 1 << 30`) to signal chunked payloads, preserving backward compatibility with existing single-stream and unchunked-interleaved decoders.

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

1. **rANS encode backward-pass is hard to parallelize within a single lane.** Mitigation: chunking splits streams into many independent segments; parallelism comes from chunk × lane count, not from parallelizing the sequential state machine within a single lane.
5. **Chunk size is a ratio/throughput tradeoff.** Smaller chunks = more GPU threads but more framing overhead and potentially worse ratio (less context for frequency estimation, though shared freq table mitigates this). Mitigation: shared frequency table across chunks; benchmark sweep in Phase 1 to find the Pareto frontier; make chunk size a tunable with conservative default.
2. **Demux kernel memory amplification.** Splitting one match stream into 3-4 output streams requires scatter writes with unpredictable output sizes. Mitigation: two-pass approach (count pass for offsets, then write pass), matching the existing GPU Huffman encode pattern.
3. **Partitioning heuristic is workload-dependent.** Wrong split wastes one device. Mitigation: conservative defaults (CPU-only unless input is large), user override via policy flag, benchmark-driven threshold tuning.
4. **Command buffer chaining complexity.** Buffer lifetime and synchronization across stages in a single submission. Mitigation: Phase 5 is last; validate correctness in Phases 1-4 with separate submissions first.

## Relationship to Other Plans

- **PLAN-interleaved-rans.md** Phase A (PR #91): provides the CPU interleaved rANS implementation and wire format that GPU kernels must match bit-for-bit.
- **PLAN-competitive-roadmap.md** Phase 4 (nvCOMP-style batch track): this roadmap is the enabling work for that throughput target.

## Immediate Next Actions

1. Design chunked rANS wire format and get agreement on flag bit allocation (`RANS_CHUNKED_FLAG`).
2. Implement CPU `rans::encode_chunked` / `rans::decode_chunked` in `src/rans.rs` as reference implementation.
3. Add round-trip tests for chunked format at various chunk sizes (1KB, 4KB, 16KB).
4. Prototype `rans_encode.wgsl` for chunk-parallel encode; validate against CPU `rans::decode_chunked`.
5. Benchmark chunk size sweep: ratio impact and GPU thread count at 256KB and 1MB input sizes.
