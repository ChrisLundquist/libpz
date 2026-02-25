# Pareto-Competitiveness Implementation Plan — Phase 5: Heterogeneous Block Scheduling

**Goal:** Enable CPU+GPU cooperative block compression with per-stage backend assignment and zero-overhead CPU fallback.

**Architecture:** Extend `parallel.rs` block loop with backend assignment per stage. CPU does match finding, GPU does entropy for large blocks. Ring-buffered handoff pattern from `webgpu/lz77.rs` generalized to entropy coding.

**Tech Stack:** Rust, wgpu v27

**Scope:** 8 phases from original design (phase 5 of 8)

**Codebase verified:** 2026-02-22

---

## Acceptance Criteria Coverage

### pareto-competitiveness.AC3: Multi-thread and GPU scaling
- **pareto-competitiveness.AC3.3 Success:** Heterogeneous scheduling (CPU match + GPU entropy) achieves higher throughput than pure CPU on 512KB+ inputs with GPU available
- **pareto-competitiveness.AC3.5 Failure:** No GPU available — pure CPU path works with zero overhead (no GPU initialization, no fallback penalty)
- **pareto-competitiveness.AC3.6 Failure:** GPU device lost mid-compression — graceful fallback to CPU, no data corruption

---

## Context and Existing Patterns

Phase 5 builds on two established patterns:

**Unified scheduler** (`compress_parallel_unified_lz_rans` in `src/pipeline/parallel.rs`): A shared `VecDeque<UnifiedTask>` with a `Condvar` drives all workers. `Stage0` tasks (match finding) enqueue `Stage1` tasks (entropy) on completion. Workers pick up any available task. This is the pattern to extend with GPU-dispatched entropy.

**Ring-buffered GPU streaming** (`compress_streaming_gpu` in `src/pipeline/parallel.rs`, `Lz77BufferSlot` + `BufferRing` in `src/webgpu/lz77.rs`): A GPU coordinator thread round-robins through pre-allocated buffer slots. When a slot is re-acquired, the coordinator reads back the previous result and forwards it to CPU entropy workers via `mpsc::sync_channel`. This ring-buffered handoff is the model for GPU entropy dispatch.

**Existing `CompressOptions`** (`src/pipeline/mod.rs`): Already has `backend: Backend` (Cpu or WebGpu) and `unified_scheduler: bool`. Phase 5 adds per-stage backend assignment that sits alongside these fields.

**GPU entropy** (`src/webgpu/rans.rs`, `src/webgpu/lzseq.rs`): The `WebGpuEngine::lzseq_encode_gpu` path already chains GPU match finding with GPU demux on-device. Phase 4 adds GPU entropy encoding for LzSeqR on top of this. Phase 5 schedules that work across multiple blocks concurrently.

**Threshold from design doc**: GPU entropy wins at >= 256KB per block; CPU is faster below 128KB. The 512KB total input threshold for AC3.3 means at least 2 full-size blocks must go through GPU entropy.

---

## Subcomponent A: Backend Assignment Per Stage

### Task 1: Add `BackendAssignment` and per-stage fields to `CompressOptions`

**File:** `src/pipeline/mod.rs`

Add a new enum before `CompressOptions`:

```rust
/// Per-stage compute backend override.
///
/// Controls which backend executes a specific pipeline stage.
/// `Auto` lets the scheduler decide based on block size and GPU availability.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BackendAssignment {
    /// Scheduler chooses: GPU entropy for blocks >= GPU_ENTROPY_THRESHOLD,
    /// CPU for smaller blocks or when no GPU is available.
    #[default]
    Auto,
    /// Force CPU execution for this stage (always available, zero-overhead).
    Cpu,
    /// Force GPU execution for this stage (requires `webgpu` feature and a device).
    #[cfg(feature = "webgpu")]
    Gpu,
}
```

Add two fields to `CompressOptions` after `unified_scheduler`:

```rust
/// Backend assignment for stage 0 (match finding / transform).
/// Match finding is always CPU — LZ77's sequential dependency rules out GPU.
/// For BWT-based pipelines, Auto routes to GPU.
pub stage0_backend: BackendAssignment,
/// Backend assignment for stage 1 (entropy coding).
/// Auto routes to GPU when block size >= GPU_ENTROPY_THRESHOLD and GPU available.
pub stage1_backend: BackendAssignment,
```

Add a threshold constant after the existing block size constants:

```rust
/// Minimum block size for GPU entropy to win over CPU (empirical from Phase 4).
pub(crate) const GPU_ENTROPY_THRESHOLD: usize = 256 * 1024;
```

Update `Default for CompressOptions` to set both new fields to `BackendAssignment::Auto`.

**Tests** (in `mod.rs` test module at bottom of file):
- `backend_assignment_default_is_auto`: construct `CompressOptions::default()`, assert both stage fields are `BackendAssignment::Auto`
- `backend_assignment_cpu_variant_always_available`: construct with `stage1_backend: BackendAssignment::Cpu`, assert no compile errors in non-webgpu builds

**Run:** `cargo test pipeline` — all pass. `cargo clippy --all-targets` — zero warnings.

**Commit:** "pipeline: add BackendAssignment enum and per-stage fields to CompressOptions"

---

### Task 2: Extend `compress_parallel_unified_lz_rans` to route stage 1 by backend assignment

**File:** `src/pipeline/parallel.rs`

The existing `compress_parallel_unified_lz_rans` already handles `Stage0` → `Stage1` task sequencing via a shared `VecDeque`. Extend it to dispatch `Stage1` to GPU when the assignment and block size warrant it.

**New enum variant** — extend `UnifiedTask` (already `Copy`):

```rust
#[derive(Copy, Clone)]
enum UnifiedTask {
    Stage0(usize),
    Stage1(usize),
    Stage1Gpu(usize), // entropy via GPU; block_idx
}
```

**Resolve effective backend** — add a helper at the top of `compress_parallel_unified_lz_rans`:

```rust
fn should_use_gpu_entropy(
    block: &[u8],
    options: &CompressOptions,
) -> bool {
    #[cfg(feature = "webgpu")]
    {
        use super::{BackendAssignment, GPU_ENTROPY_THRESHOLD};
        match options.stage1_backend {
            BackendAssignment::Gpu => options.webgpu_engine.is_some(),
            BackendAssignment::Cpu => false,
            BackendAssignment::Auto => {
                options.webgpu_engine.is_some()
                    && block.len() >= GPU_ENTROPY_THRESHOLD
            }
        }
    }
    #[cfg(not(feature = "webgpu"))]
    { false }
}
```

**In the `Stage0` completion branch** inside the worker closure, replace the unconditional `Stage1` enqueue with:

```rust
Ok(stage0_block) => {
    let use_gpu = should_use_gpu_entropy(blocks[block_idx], &opts);
    *stage0_slots_ref[block_idx].lock()... = Some(stage0_block);
    let mut guard = queue_ref.lock()...;
    if !guard.failed {
        let task = if use_gpu {
            UnifiedTask::Stage1Gpu(block_idx)
        } else {
            UnifiedTask::Stage1(block_idx)
        };
        guard.queue.push_back(task);
        guard.pending_tasks += 1;
        cv_ref.notify_one();
    }
    ...
}
```

**Add `Stage1Gpu` handling** in the worker match arm. At this point (Task 2) the GPU entropy path is a stub that falls through to CPU so the routing wires up correctly before the ring buffer exists:

```rust
UnifiedTask::Stage1Gpu(block_idx) => {
    // Stub: GPU entropy not yet wired. Fall through to CPU path.
    // Replaced in Task 4 when ring-buffered entropy handoff is added.
    let stage0 = stage0_slots_ref[block_idx]
        .lock()...
        .take()
        .expect("stage0 result missing");
    let result = run_compress_stage(pipeline, 1, stage0, &opts).map(|b| b.data);
    let is_err = result.is_err();
    *results_ref[block_idx].lock()... = Some(result);
    if is_err { /* same error handling as Stage1 */ }
}
```

**Note on thread safety**: The GPU entropy path in Task 4 will require the GPU coordinator to run on a dedicated thread (not a worker thread), consistent with the `compress_streaming_gpu` design. The stub approach here avoids restructuring the worker loop before the ring is ready.

**Tests** (in `parallel.rs` test module):
- `stage1_routing_cpu_when_no_gpu`: compress a 512KB block with `stage1_backend: BackendAssignment::Auto` and no `webgpu_engine`. Assert the result equals a plain CPU compress of the same block.
- `stage1_routing_cpu_forced`: same input but `stage1_backend: BackendAssignment::Cpu`. Assert result is identical.
- `stage1_routing_respects_size_threshold`: compress a 128KB block (below `GPU_ENTROPY_THRESHOLD`) with a mock engine present; assert `Stage1` (not `Stage1Gpu`) is enqueued. Verify by observing CPU-path output is correct.

**Run:** `./scripts/test.sh --quick` — all pass.

**Commit:** "pipeline: route stage-1 to GPU task variant based on BackendAssignment"

---

### Task 3: Tests for backend assignment routing

These tests go in `src/pipeline/parallel.rs` `#[cfg(test)] mod tests`, supplementing Task 2's tests with round-trip correctness coverage:

- `heterogeneous_routing_roundtrip_lzseqr_cpu_only`: compress and decompress a 1MB synthetic payload (alternating byte pattern) using `Pipeline::LzSeqR` with `stage1_backend: BackendAssignment::Cpu`, `unified_scheduler: true`. Assert decompressed output equals input.
- `heterogeneous_routing_roundtrip_lzseqr_auto_no_gpu`: same as above with `BackendAssignment::Auto` and no webgpu engine. Assert no panic and output is correct.
- `backend_assignment_threshold_boundary`: compress exactly `GPU_ENTROPY_THRESHOLD` bytes with `Auto` and no GPU. Assert CPU path taken (no panic, correct output). Compress `GPU_ENTROPY_THRESHOLD - 1` bytes. Assert same.

**Run:** `cargo test parallel` — all pass.

**Commit:** "pipeline: add backend assignment routing tests"

---

## Subcomponent B: Ring-Buffered Entropy Handoff

### Task 4: Generalize ring buffer for LzSeq-match to GPU-entropy handoff

This task replaces the `Stage1Gpu` stub from Task 2 with the real GPU entropy dispatch. It follows the `compress_streaming_gpu` threading model: a dedicated GPU coordinator thread owns the ring and forwards results; CPU workers execute fallback entropy when GPU is not used.

**Key structural decision**: Rather than embedding GPU dispatch inside the unified worker loop (which has no wgpu context and runs on arbitrary thread pool threads), add a separate `compress_parallel_heterogeneous_lzseq_rans` function in `parallel.rs`. This mirrors `compress_streaming_gpu` and `compress_parallel_unified_lz_rans` as a named strategy function. `compress_parallel` dispatches to it when `pipeline` is `LzSeqR`, `unified_scheduler` is true, `stage1_backend` is `Auto` or `Gpu`, and a webgpu engine is present.

**New type** in `parallel.rs` (webgpu-gated):

```rust
#[cfg(feature = "webgpu")]
struct EntropyHandoffSlot {
    /// Pre-allocated GPU input buffer for entropy encode (LzSeq streams).
    streams_buf: wgpu::Buffer,
    /// Pre-allocated staging buffer for readback.
    staging_buf: wgpu::Buffer,
    /// Capacity in bytes.
    capacity: usize,
}
```

The slot capacity is set to worst-case LzSeq stream bytes for one block. Slots are created once in `create_entropy_ring(block_size, depth)` on `WebGpuEngine`.

**Function signature:**

```rust
#[cfg(feature = "webgpu")]
fn compress_parallel_heterogeneous_lzseq_rans(
    input: &[u8],
    pipeline: Pipeline,
    options: &CompressOptions,
    num_threads: usize,
    blocks: &[&[u8]],
    engine: &crate::webgpu::WebGpuEngine,
) -> PzResult<Vec<u8>>
```

**Threading model** (mirrors `compress_streaming_gpu`):

```
GPU coordinator (1 thread):
  For each block:
    1. Run CPU Stage0 (match finding + LzSeq demux) — produces 6 streams
    2. If block >= GPU_ENTROPY_THRESHOLD:
         Acquire ring slot
         If slot has previous in-flight: complete readback, send to collector
         Submit entropy dispatch to GPU on this slot
         Record slot_inflight[slot_idx] = block_idx
    3. Else (small block):
         Send Stage0 output to a CPU entropy worker via mpsc

CPU entropy workers (num_threads - 1):
  Receive (block_idx, stage0_block) from coordinator
  Run Stage1 (rANS on CPU)
  Send (block_idx, result) to collector

Collector (calling thread):
  Receive (block_idx, result) from all workers
  Assemble ordered output
```

**GPU coordinator detail** for slot drain at end (mirrors lz77.rs lines 525-551):

```rust
// Drain remaining in-flight slots after all blocks submitted
for (slot_idx, inflight) in slot_inflight.iter_mut().enumerate() {
    if let Some(prev_idx) = inflight.take() {
        engine.poll_wait();
        let result = engine.complete_entropy_from_slot(&ring.slots[slot_idx]);
        match result {
            Ok(data) => { let _ = result_tx.send((prev_idx, Ok(data))); }
            Err(e) => {
                eprintln!("[pz-gpu] entropy slot readback failed: {e}");
                // Fallback: re-run entropy on CPU for this block
                let stage0 = stage0_cache[prev_idx].take()
                    .expect("stage0 cache missing for fallback");
                let cpu_result = run_compress_stage(pipeline, 1, stage0, options)
                    .map(|b| b.data);
                let _ = result_tx.send((prev_idx, cpu_result));
            }
        }
    }
}
```

**Stage0 cache**: The coordinator must keep a clone of stage0 output for each in-flight block so CPU fallback is possible if GPU readback fails (AC3.6). Use `Vec<Option<StageBlock>>` indexed by block_idx, same as `stage0_slots` in the unified scheduler.

**Entry point wiring** — in `compress_parallel`, add dispatch before the existing `is_batchable` webgpu check:

```rust
#[cfg(feature = "webgpu")]
if options.unified_scheduler
    && matches!(pipeline, Pipeline::LzSeqR)
    && options.stage1_backend != BackendAssignment::Cpu
{
    if let super::Backend::WebGpu = options.backend {
        if let Some(ref engine) = options.webgpu_engine {
            return compress_parallel_heterogeneous_lzseq_rans(
                input, pipeline, options, num_threads, &blocks, engine,
            );
        }
    }
}
```

**New `WebGpuEngine` methods** (`src/webgpu/rans.rs` or a new `src/webgpu/entropy_ring.rs`):

```rust
impl WebGpuEngine {
    /// Create a ring of pre-allocated entropy encode buffer slots.
    pub(crate) fn create_entropy_ring(
        &self,
        block_size: usize,
        depth: usize,
    ) -> Option<BufferRing<EntropySlot>>

    /// Submit LzSeq streams for GPU entropy encoding into a ring slot.
    pub(crate) fn submit_entropy_to_slot(
        &self,
        streams: &[Vec<u8>; 6],
        slot: &EntropySlot,
    ) -> PzResult<()>

    /// Complete entropy encoding from a ring slot after poll_wait().
    pub(crate) fn complete_entropy_from_slot(
        &self,
        slot: &EntropySlot,
    ) -> PzResult<Vec<u8>>
}
```

These wrap the Phase 4 GPU rANS encode kernel (the same `encode_gpu_lzseq_streams` path) with slot-based buffer reuse.

**Tests** (in `parallel.rs`):
- `heterogeneous_compress_roundtrip_lzseqr_with_gpu_entropy`: compile-time gated with `#[cfg(feature = "webgpu")]`. Create a `WebGpuEngine`, compress a 1MB block (4 x 256KB) with `stage1_backend: BackendAssignment::Gpu`, decompress with CPU. Assert input == output.
- `heterogeneous_compress_block_count_mix`: compress 5 x 256KB blocks. Assert all 5 produce correct output (tests the ring slot recycling path where depth < block count).

**Run:** `./scripts/test.sh --quick` — all pass. `cargo test --features webgpu heterogeneous` — GPU tests pass (skip if no device).

**Commit:** "pipeline: add ring-buffered GPU entropy handoff for LzSeqR heterogeneous path"

---

### Task 5: Tests for ring-buffered entropy handoff correctness

These tests verify the ring slot recycling, ordering, and cross-path correctness specifically, beyond Task 4's functional tests.

- `entropy_ring_slot_recycling`: compress 16 blocks of 256KB (8 ring slots, depth=4 — forces each slot to be recycled twice). Assert all 16 blocks decompress correctly.
- `entropy_handoff_output_order_preserved`: compress blocks with deliberate varying sizes (some above, some below `GPU_ENTROPY_THRESHOLD`) in one call. Assert decompressed output equals input byte-for-byte regardless of which path each block took.
- `entropy_handoff_cpu_gpu_cross_decode`: compress with GPU entropy, decompress with explicit CPU-only options. Assert round-trip success. Also compress with CPU entropy, decompress with GPU — assert round-trip success. (Verifies decoder is backend-agnostic, per existing container format guarantee.)
- `entropy_ring_empty_blocks_handled`: compress an input where the last block is smaller than `MIN_GPU_INPUT_SIZE`. Assert no panic, correct output. (Edge case: ring slot is acquired but not submitted.)

**Run:** `cargo test --features webgpu entropy_ring` — pass if GPU present, skip if not.

**Commit:** "pipeline: add ring-buffered entropy handoff correctness tests"

---

## Subcomponent C: Load Balancing and Fallback

### Task 6: CPU compresses block N+1 while GPU entropy-encodes block N

This is the key throughput mechanism. The GPU coordinator in `compress_parallel_heterogeneous_lzseq_rans` (from Task 4) already overlaps work via the ring buffer: when it acquires slot `k` and that slot was in-flight, it completes the previous block's GPU entropy before submitting the new block's entropy. The CPU match finding (Stage0) for the new block happens *before* the submit, so there is natural overlap.

The remaining gap is Stage0 latency for block N+1 while GPU is encoding block N. Close this by prefetching Stage0 for the next block:

**Prefetch approach** — in the GPU coordinator loop, use a dedicated Stage0 worker:

```rust
// Stage0 prefetch channel: coordinator pre-submits Stage0 for block N+1
// while it is completing GPU readback for block N.
let (stage0_tx, stage0_rx) = mpsc::sync_channel::<(usize, PzResult<StageBlock>)>(ring_depth);

// Stage0 prefetch worker
scope.spawn(move || {
    for (block_idx, block) in blocks.iter().enumerate() {
        let result = run_compress_stage(pipeline, 0, make_stage_block(block_idx, block), &opts);
        if stage0_tx.send((block_idx, result)).is_err() { break; }
    }
});
```

The coordinator reads from `stage0_rx` in order (blocks are sequential, Stage0 is independent per block). While the coordinator is doing `engine.poll_wait()` + `complete_entropy_from_slot()` for block N, the Stage0 prefetch worker is already running match finding for block N+1.

This is a straightforward extension to the `compress_parallel_heterogeneous_lzseq_rans` function from Task 4. The prefetch worker replaces the inline Stage0 call in the coordinator.

**Implementation note**: Stage0 ordering matters because LzSeq match finding is per-block independent, but the Stage0 prefetch must stay in-order so `stage0_cache[block_idx]` is always populated before the coordinator needs it. `mpsc::sync_channel` with capacity `ring_depth` provides natural backpressure — the Stage0 worker cannot run too far ahead.

**No new files**. This is a modification to `compress_parallel_heterogeneous_lzseq_rans` in `parallel.rs`.

**Tests:**
- `prefetch_stage0_overlaps_with_gpu_entropy`: time a 4MB compression run (8 x 512KB blocks) with and without the Stage0 prefetch worker. Assert the prefetch path is not slower (throughput >= without-prefetch). This is a timing test and should use `#[ignore]` so it is not run in CI by default — invoke manually as `cargo test --release -- --ignored prefetch_stage0`.
- `prefetch_stage0_correct_output`: assert round-trip correctness with the prefetch path enabled.

**Run:** `./scripts/test.sh --quick` — all pass.

**Commit:** "pipeline: prefetch stage-0 match finding to overlap with GPU entropy encoding"

---

### Task 7: Zero-overhead CPU-only path when no GPU

The CPU-only path must have zero overhead — no GPU initialization, no fallback penalty. The current `CompressOptions::default()` already sets `backend: Backend::Cpu` and `webgpu_engine: None`. The routing in `compress_parallel` (Task 4) dispatches to `compress_parallel_heterogeneous_lzseq_rans` only when `options.webgpu_engine.is_some()`. When the engine is absent, the existing `compress_parallel_unified_lz_rans` path is taken unchanged.

This task verifies that guarantee holds and that no new code paths introduce latency when GPU is absent:

**Verify no new feature-gated overhead**: Review all new code added in Tasks 1-6. Confirm:
1. `BackendAssignment` enum and new `CompressOptions` fields are zero-cost when `stage0_backend` and `stage1_backend` are `Auto` and no engine is present (no branch taken inside the unified scheduler that was not already there before Phase 5).
2. The `should_use_gpu_entropy` helper returns `false` immediately when `webgpu_engine.is_none()` — confirmed by the `options.webgpu_engine.is_some()` check.
3. The new dispatch branch in `compress_parallel` is guarded by `#[cfg(feature = "webgpu")]` and `options.webgpu_engine.is_some()`, so it compiles away entirely in non-webgpu builds.

**Micro-benchmark** — add to `benches/stages.rs` (or existing bench file):

```rust
#[bench]
fn bench_lzseqr_cpu_only_no_gpu(b: &mut Bencher) {
    let input = make_benchmark_input(512 * 1024);
    let opts = CompressOptions {
        backend: Backend::Cpu,
        unified_scheduler: true,
        threads: 4,
        ..Default::default()
    };
    b.iter(|| compress(&input, Pipeline::LzSeqR, &opts).unwrap());
}
```

Run before and after Phase 5 changes to confirm no regression.

**Tests:**
- `cpu_only_path_no_webgpu_engine`: compress 1MB with `CompressOptions::default()` (no engine). Assert result decompresses correctly. Assert no GPU-related code was invoked (observable by ensuring no `Stage1Gpu` task is ever enqueued — verified via the `should_use_gpu_entropy` returning false path).
- `cpu_only_path_cpu_backend_explicit`: set `backend: Backend::Cpu`, `stage1_backend: BackendAssignment::Cpu`. Compress 1MB. Assert correct output. Assert no webgpu-gated code branched (tested by running in a non-webgpu feature build: `cargo test --no-default-features cpu_only_path`).

**Run:** `cargo test --no-default-features cpu_only_path` — pass. `./scripts/test.sh --quick` — all pass.

**Commit:** "pipeline: verify and test zero-overhead CPU-only path with no GPU"

---

### Task 8: Graceful GPU device-lost fallback

When the GPU device is lost mid-compression (wgpu returns `wgpu::Error::Lost` or a dispatch returns `Err(PzError::GpuError(...))`), the scheduler must fall back to CPU entropy for all remaining blocks without corrupting output.

The `compress_streaming_gpu` function in `parallel.rs` already demonstrates this pattern (lines 483-494 and 536-548): when `complete_lz77_from_slot` returns `Err`, it falls back to `compress_lazy_to_matches` on the CPU and continues normally.

Apply the same pattern to `compress_parallel_heterogeneous_lzseq_rans`:

**Extend the GPU coordinator**:

```rust
// Track whether GPU has been lost — once lost, all subsequent blocks use CPU entropy
let gpu_lost = std::sync::atomic::AtomicBool::new(false);

// In the completion loop:
match engine.complete_entropy_from_slot(&ring.slots[slot_idx]) {
    Ok(data) => { let _ = result_tx.send((prev_idx, Ok(data))); }
    Err(e) => {
        eprintln!("[pz-gpu] entropy slot readback failed (device lost?): {e}");
        gpu_lost.store(true, Ordering::Release);
        // Fallback: CPU entropy for this block using cached stage0 output
        let stage0 = stage0_cache[prev_idx].take()
            .expect("stage0 cache required for GPU fallback");
        let cpu_result = run_compress_stage(pipeline, 1, stage0, &opts)
            .map(|b| b.data);
        let _ = result_tx.send((prev_idx, cpu_result));
    }
}

// Before submitting to GPU on a new block:
if gpu_lost.load(Ordering::Acquire) {
    // Route directly to CPU entropy worker
    let _ = cpu_tx.send((block_idx, stage0_block));
} else {
    engine.submit_entropy_to_slot(&streams, &ring.slots[slot_idx])?;
}
```

**No data corruption guarantee**: The stage0 cache always holds the `StageBlock` until the entropy result is confirmed (CPU or GPU). GPU dispatch consumes the encoded streams from `StageBlock`, not the block itself — so the `StageBlock` is still valid for CPU fallback after a failed GPU readback.

**Tests:**
- `gpu_device_lost_fallback_no_corruption` — gated `#[cfg(feature = "webgpu")]`: Create a mock or real engine. Compress a 4-block input. Inject a simulated `Err(PzError::GpuError("device lost".into()))` from `complete_entropy_from_slot` on block 1. Assert: (a) no panic, (b) decompressed output equals input, (c) blocks 2 and 3 used CPU entropy path (observable if `gpu_lost` is exposed as a return value or via an atomic counter in test builds).
- `gpu_device_lost_all_blocks_fallback`: inject the error on block 0 (first readback). Assert all 4 blocks fall back to CPU, output is correct.
- `gpu_device_lost_mid_stream_no_partial_write`: inject error mid-ring. Assert the partially GPU-encoded block does not produce a corrupt partial output — the CPU fallback re-encodes from stage0 scratch, not from a partial GPU result.

**Implementation note on mock injection**: The test injection is simplest with a `#[cfg(test)]` hook on `WebGpuEngine::complete_entropy_from_slot` that checks a thread-local override. If that is too invasive, the fallback behavior is also exercisable by running on hardware where the GPU ring slot capacity underflows the block size, causing a real allocation error on submit. Document the chosen approach in the test.

**Run:** `cargo test --features webgpu gpu_device_lost` — pass.

**Commit:** "pipeline: graceful GPU device-lost fallback during entropy handoff"

---

### Task 9: Tests for fallback scenarios (no GPU, GPU lost mid-compression)

Consolidate and expand fallback coverage as a dedicated test suite. These tests verify AC3.5 and AC3.6 from a higher-level perspective.

**File:** `src/pipeline/parallel.rs` `#[cfg(test)] mod tests` — add a `mod fallback_tests` submodule.

Tests:

- `ac3_5_no_gpu_zero_overhead_cpu_path`: Compress a 1MB Canterbury-excerpt with `CompressOptions::default()` (no GPU). Assert output decompresses correctly. Assert wall-clock time is within 10% of the baseline `compress_parallel_unified_lz_rans` path measured in Task 7 bench. Use `std::time::Instant` for timing. Mark `#[ignore]` for CI, run with `-- --ignored` for validation.

- `ac3_5_webgpu_feature_disabled_compiles_and_works`: Run `cargo test --no-default-features -- ac3_5_webgpu_feature_disabled` to confirm the entire Phase 5 code path compiles and produces correct output without the webgpu feature. Verify by including a `#[test]` in a `#[cfg(not(feature = "webgpu"))]` block.

- `ac3_6_gpu_lost_on_first_block`: Inject a GPU loss on the first block. Assert no panic, all blocks decompress correctly, no corruption.

- `ac3_6_gpu_lost_on_last_block`: Inject a GPU loss on the last block of a 4-block input. Assert the first 3 blocks (already GPU-encoded successfully) and the last block (CPU fallback) all decompress correctly when concatenated.

- `ac3_6_gpu_lost_then_recovered_new_job`: After a compression with a simulated GPU loss, run a second compression job on the same engine. Assert the second job uses CPU entropy for all blocks (the device-lost flag persists for the duration of the job). Assert the second job completes correctly.

- `ac3_6_no_data_corruption_on_partial_slot_state`: Verify that a GPU loss during `submit_entropy_to_slot` (not `complete`) also falls back cleanly. The slot's `submit` failing means no in-flight work was started — the CPU fallback should run without waiting for `poll_wait()`. Assert correct output.

**Run:** `cargo test --features webgpu fallback_tests` — all pass (GPU-dependent tests skip gracefully if no device). `cargo test --no-default-features fallback_tests` — CPU-only tests pass.

**Commit:** "pipeline: comprehensive fallback scenario tests for AC3.5 and AC3.6"

---

## Files Modified

| File | Tasks | Change |
|------|-------|--------|
| `src/pipeline/mod.rs` | 1 | Add `BackendAssignment` enum, `stage0_backend`/`stage1_backend` fields, `GPU_ENTROPY_THRESHOLD` constant |
| `src/pipeline/parallel.rs` | 2, 4, 5, 6, 7, 8, 9 | `UnifiedTask::Stage1Gpu` variant, `should_use_gpu_entropy` helper, `compress_parallel_heterogeneous_lzseq_rans` function, Stage0 prefetch worker, device-lost flag, dispatch routing in `compress_parallel` |
| `src/webgpu/rans.rs` (or new `src/webgpu/entropy_ring.rs`) | 4 | `create_entropy_ring`, `submit_entropy_to_slot`, `complete_entropy_from_slot` on `WebGpuEngine` |
| `benches/stages.rs` (or existing bench file) | 7 | `bench_lzseqr_cpu_only_no_gpu` benchmark |

## Verification

After each task:
1. `cargo fmt && cargo clippy --all-targets` — zero warnings
2. `cargo test` — all CPU tests pass
3. `cargo test --features webgpu` — GPU tests pass (skip gracefully if no device)
4. `cargo test --no-default-features` — no GPU code compiled in, CPU tests pass

After all tasks:
- `./scripts/bench.sh` baseline comparison — LzSeqR throughput on 1MB+ input with GPU >= CPU-only throughput
- `cargo test --features webgpu -- --ignored prefetch_stage0` — prefetch overlap timing test
- `./scripts/test.sh --quick` — full suite passes clean

## Done When

- Multi-block LzSeqR compression on a GPU-equipped machine achieves higher throughput than the pure CPU path for inputs >= 512KB (`./scripts/bench.sh` shows measurable improvement; AC3.3).
- `cargo test --no-default-features` passes with zero overhead from any GPU initialization code (AC3.5).
- GPU device-lost injection tests pass: no panic, no data corruption, fallback to CPU completes successfully (AC3.6).
- `./scripts/test.sh --quick` passes clean with zero warnings.
