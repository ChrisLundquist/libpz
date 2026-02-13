# GPU Backpressure: Limit Queue Depth and In-Flight Work

## Problem

When running GPU workloads (especially benchmarks that exercise both OpenCL and WebGPU on the same physical GPU), we see 80-100% GPU utilization in Activity Monitor and degraded/inconsistent performance. The root cause: **there is no backpressure mechanism limiting how much GPU work is enqueued ahead of consumption by subsequent pipeline stages.**

### Symptoms
- `gpu_compare` benchmark shows wildly inconsistent results vs criterion (e.g. OpenCL LZ77 hash at 0.1 MB/s in gpu_compare vs 3-15 MB/s in criterion)
- OpenCL batch LZ77 kernel at 4MB triggers a **GPU hang** (`GPU hang occurred, CoreAnalytics returned false` → SIGABRT)
- 80-100% GPU usage observed in Activity Monitor during benchmarks
- Both backends share the same physical GPU (AMD Radeon Pro 5500M) with no mutual coordination

### GPU Work Budget for a 4MB Block

A single 4MB input block generates enormous GPU work:

| Kernel | Threads/Pass | Passes | GPU Memory | Notes |
|--------|-------------|--------|-----------|-------|
| LZ77 Hash | 4,194,304 | 2 | **60 MB** | hash_table alone is 8.4 MB (32K buckets × 64 entries × 4B) |
| LZ77 Lazy | 4,194,304 | 3 | **108 MB** | raw_matches + resolved = 2 × 50 MB |
| LZ77 Batch | 131,072 | 1 | **52 MB** | O(n×w) per thread, 32KB window → **GPU hang** |
| BWT | 4,194,304 | ~132 | **128 MB** | 22 doubling steps × ~6 radix passes |
| Huffman | 4,194,304 | 4 | **~22 MB** | histogram + bit lengths + prefix sum + write |

The batched path (`find_matches_batched`) submits 8 blocks at once: **168-192 MB peak GPU memory** with zero backpressure. The batch kernel is worse — `lz77_batch.cl` runs `FindMatchClassic` which is O(n × window) brute force per work-item (32 positions × 32KB window = 1M byte comparisons per thread), with 131K threads all hammering global memory simultaneously.

### Root Cause Analysis

The current batched GPU path (`find_matches_batched()` in `src/webgpu/lz77.rs`) submits up to `MAX_GPU_BATCH_SIZE = 8` blocks before a single `device.poll(Wait)`. This is fire-and-forget: all 8 blocks are submitted, then we wait for ALL of them, then read ALL results back.

The pipeline-parallel path (`src/pipeline/parallel.rs`) uses `sync_channel(2)` between CPU stages for backpressure, but the GPU-batched path bypasses this — it submits ALL blocks to the GPU in one call (`engine.find_matches_batched(&blocks)` at line 130), then does CPU entropy encoding in a second phase. There's no overlap between GPU and CPU work.

**The ideal behavior**: GPU should stay a small fixed number of blocks (2-3) ahead of the CPU entropy encoding stage, so:
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

## Proposed Changes

There are two independent improvements: (A) kernel-level resource hints that help the GPU compiler, and (B) host-side backpressure that limits in-flight work. Both should be implemented.

---

### Part A: Kernel Resource Hints

The GPU compiler can make better occupancy/register tradeoffs if we declare the workgroup size we intend to use. Currently, none of our OpenCL kernels declare `reqd_work_group_size`, so the compiler must pessimistically support arbitrary work-group sizes.

#### A1: Add `__attribute__((reqd_work_group_size))` to OpenCL kernels

This is a hard contract: the compiler optimizes for exactly this size, and dispatch MUST use it. The benefit is the compiler can pick optimal SIMD width and register allocation. If dispatch uses a different local size, the kernel will fail to launch (which is fine — we always use fixed sizes).

**Changes to `kernels/lz77_hash.cl`:**
```opencl
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void BuildHashTable(...)

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void FindMatches(...)
```

**Changes to `kernels/lz77.cl`:**
```opencl
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void Encode(...)
```

**Changes to `kernels/lz77_batch.cl`:**
```opencl
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void Encode(...)
```

**Changes to `kernels/lz77_topk.cl`:**
```opencl
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void EncodeTopK(...)
```

**Changes to `kernels/huffman_encode.cl`:**
```opencl
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void ComputeBitLengths(...)

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void WriteCodes(...)

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void ByteHistogram(...)

// Prefix sum uses 256, not 64:
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void PrefixSumBlock(...)

__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void PrefixSumApply(...)
```

**Changes to `kernels/bwt_rank.cl` and `kernels/bwt_radix.cl`:**
These already get `-DWORKGROUP_SIZE=N` at compile time. Add:
```opencl
__kernel __attribute__((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
void rank_compare(...)
// ... same for all BWT kernels
```

**Note on OpenCL dispatch:** The Rust host code in `src/opencl/lz77.rs` currently uses `.set_global_work_size(n)` without setting local work size for the LZ77 kernels, letting the driver pick. After adding `reqd_work_group_size(64,1,1)`, we MUST also add `.set_local_work_size(64)` to every dispatch of those kernels, or the driver may pick a different size and the kernel will refuse to launch.

#### A2: WGSL kernels already have `@workgroup_size()` — no changes needed

WGSL requires `@workgroup_size()` on every compute entry point, and all our WGSL shaders already declare them. WGSL has no additional resource hint mechanisms — `@workgroup_size()` is the only lever available. The WebGPU spec deliberately keeps the shader language minimal; there are no register count hints, occupancy hints, or shared memory budget preferences.

#### A3: Add `__local` memory declarations where beneficial

Currently `lz77_hash.cl` uses global atomics for `hash_counts`. The histogram pattern (many threads atomically incrementing a small array) is a classic case where `__local` memory helps:

```opencl
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void BuildHashTable(...) {
    // Each workgroup builds a local histogram, then flushes to global
    __local volatile uint local_counts[256]; // per-workgroup hash count cache
    // ...
}
```

This is a separate optimization PR, not part of the backpressure fix. Mention it here for completeness but don't block on it.

---

### Part B: Host-Side Backpressure

#### B1: Key constant

```rust
/// Maximum number of GPU LZ77 blocks in flight before we wait for
/// the entropy stage to consume one.  Keeps GPU memory bounded and
/// allows CPU work to overlap with GPU work.
const GPU_PREFETCH_DEPTH: usize = 2;  // tune empirically
```

At `GPU_PREFETCH_DEPTH = 2` with 256KB blocks, peak GPU memory is ~42-48 MB (2 blocks × 21-24 MB each). Compare with the current 168-192 MB for 8 blocks.

#### B2: Add `find_matches_pipelined()` to `src/webgpu/lz77.rs`

The existing `submit_find_matches_lazy()` already returns a `PendingLz77` handle. Build a sliding-window dispatcher on top:

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
3. Call `callback` with the completed block's matches (entropy work happens here)
4. Submit the next block to refill the window
5. Drain remaining blocks at the end

This gives a sliding window: GPU is always working on the next 2 blocks while CPU processes the current one.

For WebGPU, use `queue.submit()` returning `SubmissionIndex` + `device.poll(WaitForSubmissionIndex(idx))` to wait for specific submissions rather than draining everything. This is a wgpu-native API that gives precise per-submission synchronization.

#### B3: Modify `compress_gpu_batched()` in `src/pipeline/parallel.rs`

Replace the two-phase approach with a pipelined producer-consumer:

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

The `sync_channel(GPU_PREFETCH_DEPTH)` provides natural backpressure: if CPU entropy is slower than GPU LZ77, the channel fills up and the GPU producer blocks on `send()`.

#### B4: Add GPU idle fence to `gpu_compare` benchmark

Between OpenCL and WebGPU benchmark groups, add explicit synchronization:

```rust
// Give the GPU time to fully drain between backend switches
std::thread::sleep(std::time::Duration::from_millis(100));
```

Quick benchmark fix; the real fix (B2-B3) prevents saturation in production.

---

## Files to Modify

| File | Change | Part |
|------|--------|------|
| `kernels/lz77.cl` | Add `reqd_work_group_size(64,1,1)` | A1 |
| `kernels/lz77_batch.cl` | Add `reqd_work_group_size(64,1,1)` | A1 |
| `kernels/lz77_hash.cl` | Add `reqd_work_group_size(64,1,1)` | A1 |
| `kernels/lz77_topk.cl` | Add `reqd_work_group_size(64,1,1)` | A1 |
| `kernels/huffman_encode.cl` | Add `reqd_work_group_size(64/256,1,1)` per kernel | A1 |
| `kernels/bwt_rank.cl` | Add `reqd_work_group_size(WORKGROUP_SIZE,1,1)` | A1 |
| `kernels/bwt_radix.cl` | Add `reqd_work_group_size(WORKGROUP_SIZE,1,1)` | A1 |
| `src/opencl/lz77.rs` | Add `.set_local_work_size(64)` to all LZ77 dispatches | A1 |
| `src/opencl/huffman.rs` | Add `.set_local_work_size(64/256)` to Huffman dispatches | A1 |
| `src/webgpu/lz77.rs` | Add `find_matches_pipelined()` | B2 |
| `src/webgpu/mod.rs` | Add `GPU_PREFETCH_DEPTH` constant | B1 |
| `src/pipeline/parallel.rs` | Rewrite `compress_gpu_batched()` to use pipelined producer-consumer | B3 |
| `examples/gpu_compare.rs` | Add sleep/fence between backend groups | B4 |

## What NOT to Change

- `deflate_chained()` in huffman.rs — processes a single block; backpressure is the pipeline's job
- `find_matches_batched()` — keep for backward compat; `find_matches_pipelined()` is the new entry point
- `compress_pipeline_parallel()` — already has good backpressure via `sync_channel(2)`
- WGSL kernel files — already have `@workgroup_size()`, no additional hints available
- Single-block code paths — no batching, no backpressure needed

## Implementation Order

1. **Part A first** (kernel hints) — low risk, independent of Part B, easy to validate
2. **Part B second** (backpressure) — more invasive, requires careful testing of the producer-consumer handoff
3. Parts A and B can be separate commits/PRs

## Testing

1. `cargo test` — all existing tests must pass
2. `cargo test --features opencl` — verify OpenCL kernels still compile and pass with `reqd_work_group_size`
3. `cargo test --features webgpu` — GPU tests including ported test suite
4. `cargo run --example gpu_compare --release --features opencl,webgpu` — verify no GPU hangs, more consistent numbers
5. `./scripts/bench.sh` — verify end-to-end throughput is same or better
6. `cargo bench --bench stages --features opencl,webgpu -- lz77` — verify no 4MB GPU hang
7. Manual: Activity Monitor GPU usage should show bursty pattern instead of 80-100% sustained
