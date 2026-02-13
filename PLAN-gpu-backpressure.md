# GPU Backpressure: Limit Queue Depth and In-Flight Work

## Problem

When running GPU workloads (especially benchmarks that exercise both OpenCL and WebGPU on the same physical GPU), we see 80-100% GPU utilization in Activity Monitor and degraded/inconsistent performance. The root cause: **there is no backpressure mechanism limiting how much GPU work is enqueued ahead of consumption by subsequent pipeline stages.**

### Symptoms
- `gpu_compare` benchmark shows wildly inconsistent results vs criterion (e.g. OpenCL LZ77 hash at 0.1 MB/s in gpu_compare vs 3-15 MB/s in criterion)
- OpenCL batch LZ77 kernel at 4MB triggers a **GPU hang** (`GPU hang occurred, CoreAnalytics returned false` → SIGABRT)
- 80-100% GPU usage observed in Activity Monitor during benchmarks
- Both backends share the same physical GPU (AMD Radeon Pro 5500M) with no mutual coordination

### Root Cause Analysis

The current batched GPU path (`find_matches_batched()` in `src/webgpu/lz77.rs`) submits up to `MAX_GPU_BATCH_SIZE = 8` blocks before a single `device.poll(Wait)`. Each block is 256KB, so up to 2MB of input data generates ~72MB of GPU working memory (256KB × 36 bytes/position × 8 blocks). This is fire-and-forget: all 8 blocks are submitted, then we wait for ALL of them, then read ALL results back.

The pipeline-parallel path (`src/pipeline/parallel.rs`) uses `sync_channel(2)` between CPU stages for backpressure, but the GPU-batched path bypasses this — it submits ALL blocks to the GPU in one call (`engine.find_matches_batched(&blocks)` at line 130), then does CPU entropy encoding in a second phase. There's no overlap between GPU and CPU work.

**The ideal behavior**: GPU should stay a small fixed number of blocks (e.g. 2-3) ahead of the CPU entropy encoding stage, so:
1. GPU doesn't over-commit memory or saturate the command queue
2. CPU entropy work can start as soon as the first block's LZ77 results are ready
3. Overall latency is hidden by overlapping GPU LZ77 (block N+2) with CPU Huffman (block N)

## Current Architecture (for context)

```
Input → split into 256KB blocks
     ↓
[GPU-batched path: src/pipeline/parallel.rs lines 118-179]
  Phase 1: engine.find_matches_batched(&all_blocks)  ← submits ALL blocks to GPU
    └─ Internally: chunks of 8, submit → poll(Wait) → readback
  Phase 2: thread pool entropy-encodes all blocks in parallel
     ↓
Output
```

```
[Pipeline-parallel path: src/pipeline/parallel.rs lines 324-420]
  Producer → sync_channel(2) → Stage1 → sync_channel(2) → Stage2 → Collector
  (one thread per stage, bounded channels provide backpressure)
```

The GPU-batched path has no pipelining between LZ77 and entropy. The pipeline-parallel path has good backpressure but doesn't use GPU batching.

## Proposed Fix

### Strategy: Bounded producer-consumer with small GPU prefetch window

Replace the two-phase GPU-batched path with a **pipelined producer-consumer** where the GPU stays a fixed number of blocks ahead of CPU entropy encoding.

### Key constant
```rust
/// Maximum number of GPU LZ77 blocks in flight before we wait for
/// the entropy stage to consume one.  Keeps GPU memory bounded and
/// allows CPU work to overlap with GPU work.
const GPU_PREFETCH_DEPTH: usize = 2;  // or 3, tune empirically
```

### Implementation Plan

#### Step 1: Add a `PendingBlock` type to `src/webgpu/lz77.rs`

The existing `submit_find_matches_lazy()` already returns a `PendingLz77` handle that can be polled later. We need a wrapper that pairs this with block metadata:

```rust
pub(crate) struct PendingBlock {
    pub block_index: usize,
    pub block_data: Vec<u8>,       // original input block (needed for dedup reference)
    pub pending: PendingLz77,      // GPU handle
}
```

#### Step 2: Create `find_matches_pipelined()` in `src/webgpu/lz77.rs`

A new method that yields LZ77 results one block at a time with bounded in-flight work:

```rust
/// Submit GPU LZ77 work with bounded prefetch, yielding results as they complete.
/// `callback` is called for each block as its results become available,
/// while subsequent blocks are still being processed on the GPU.
pub fn find_matches_pipelined<F>(
    &self,
    blocks: &[&[u8]],
    callback: F,
) -> PzResult<()>
where
    F: FnMut(usize, Vec<lz77::Match>) -> PzResult<()>,
```

Logic:
1. Submit up to `GPU_PREFETCH_DEPTH` blocks without waiting
2. When the window is full, `device.poll(Wait)` + complete the oldest pending block
3. Call `callback` with the completed block's matches (this does CPU entropy work)
4. Submit the next block to refill the window
5. Drain remaining blocks at the end

This gives a sliding window: GPU is always working on the next 2 blocks while CPU processes the current one.

#### Step 3: Modify `compress_gpu_batched()` in `src/pipeline/parallel.rs`

Replace the current two-phase approach:

**Before:**
```rust
// Phase 1: ALL blocks to GPU
let all_matches = engine.find_matches_batched(&blocks)?;
// Phase 2: ALL blocks entropy-encoded on CPU
thread::scope(|s| { for block in blocks { s.spawn(entropy_encode(block)); } });
```

**After:**
```rust
// Pipelined: GPU stays GPU_PREFETCH_DEPTH blocks ahead of CPU entropy
let (tx, rx) = sync_channel::<(usize, Vec<u8>, Vec<Match>)>(GPU_PREFETCH_DEPTH);

// GPU producer thread
let gpu_handle = scope.spawn(|| {
    engine.find_matches_pipelined(&blocks, |block_idx, matches| {
        tx.send((block_idx, blocks[block_idx].to_vec(), matches))?;
        Ok(())
    })
});

// CPU consumer: entropy-encode blocks as they arrive
for (block_idx, block_data, matches) in rx {
    let lz_bytes = serialize_matches(&matches);
    let encoded = demux_and_huffman_encode(&lz_bytes, &block_data)?;
    results[block_idx] = encoded;
}

gpu_handle.join()?;
```

The `sync_channel(GPU_PREFETCH_DEPTH)` provides natural backpressure: if CPU entropy is slower than GPU LZ77, the channel fills up and the GPU producer blocks on `send()`, preventing unbounded GPU memory growth.

#### Step 4: Add GPU idle fence to `gpu_compare` benchmark

Between OpenCL and WebGPU benchmark groups, add explicit synchronization to prevent cross-backend contention:

```rust
// In gpu_compare.rs, between backend benchmarks:
// Give the GPU time to fully drain between backend switches
std::thread::sleep(std::time::Duration::from_millis(100));
```

This is a quick fix for the benchmark; the real fix (Steps 1-3) prevents the saturation in the first place.

#### Step 5: Tune `GPU_PREFETCH_DEPTH` empirically

Run `./scripts/bench.sh` with different prefetch depths (1, 2, 3, 4) and measure:
- End-to-end throughput (should improve due to GPU/CPU overlap)
- Peak GPU memory usage (should decrease vs current 8-block batch)
- Latency to first block completion (should be ~1 block's GPU time instead of 8 blocks')

### Files to Modify

| File | Change |
|------|--------|
| `src/webgpu/lz77.rs` | Add `PendingBlock`, `find_matches_pipelined()` |
| `src/webgpu/mod.rs` | Add `GPU_PREFETCH_DEPTH` constant (next to existing `MAX_GPU_BATCH_SIZE`) |
| `src/pipeline/parallel.rs` | Rewrite `compress_gpu_batched()` to use pipelined producer-consumer |
| `examples/gpu_compare.rs` | Add sleep/fence between backend benchmark groups |

### What NOT to change
- `deflate_chained()` in huffman.rs — this already processes a single block; backpressure is the pipeline's job
- `find_matches_batched()` — keep it for backward compatibility; `find_matches_pipelined()` is an alternative entry point
- The pipeline-parallel path (`compress_pipeline_parallel`) — already has good backpressure via sync_channel(2)
- Single-block code paths — no batching, no backpressure needed

### Testing
1. `cargo test` — all existing tests must pass (pipelined path must produce identical output)
2. `cargo test --features webgpu` — GPU tests including the new ported test suite
3. `cargo run --example gpu_compare --release --features opencl,webgpu` — verify no GPU hangs, more consistent numbers
4. `./scripts/bench.sh` — verify end-to-end throughput is same or better
5. Manual: Activity Monitor GPU usage should drop from 80-100% sustained to lower, bursty pattern
