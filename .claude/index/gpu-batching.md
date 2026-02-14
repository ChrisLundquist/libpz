# GPU Batching & Memory Management

Memory-aware batching strategies and resource management for multi-block GPU compression.

**Last updated:** 2026-02-14

## Overview

libpz uses two complementary batching approaches:

1. **Batched LZ77 multi-block** — Ring-buffered streaming with backpressure
2. **Pipeline-parallel** — Different algorithms on different GPU blocks

## Ring-Buffered Streaming Batching

**Introduced:** Commit `4390d90` (2026-02-13)  
**File:** `/Users/clundquist/code/libpz/src/webgpu/lz77.rs` (128 +/- lines)

### Architecture

Pre-allocated ring buffer slots (double/triple-buffered) for multi-block processing:

```rust
pub struct Lz77BufferSlot {
    pub input_buf: wgpu::Buffer,
    pub params_buf: wgpu::Buffer,
    pub raw_match_buf: wgpu::Buffer,
    pub resolved_buf: wgpu::Buffer,
    pub staging_buf: wgpu::Buffer,
    pub capacity: usize,
}
```

### Buffer Allocation Formula

Per block (256KB default):

```
Input buffer:      256 KB (padded to u32 boundary)
Raw matches:       256 KB * 12 bytes = 3.08 MB (offset/length/next per position)
Resolved matches:  256 KB * 12 bytes = 3.08 MB (after lazy resolution)
Staging buffer:    256 KB * 12 bytes = 3.08 MB (GPU→CPU readback)
─────────────────────────────────────
Per-block total:   ~9.3 MB
```

**Ring overhead (3 slots for triple-buffering):** ~27.9 MB

### Performance Impact

**Before ring-buffering (per-block alloc/map):**
```
16-block batch (4MB): 82 ms
Overhead: 35% of kernel time
```

**After ring-buffering:**
```
16-block batch (4MB): 70 ms (17% faster, 57 MB/s)
Overhead: eliminated (pre-allocated)

Full deflate pipeline:  96 ms → 89 ms (7% faster)
Full lzfi pipeline:     101 ms → 90 ms (11% faster)
```

### Backpressure Strategy

Ring buffer enables overlapped GPU/CPU execution:

```
Loop iteration N:
  1. GPU computes on slot N (async, non-blocking)
  2. CPU reads results from slot N-1 (overlapped)
  3. Submit slot N+1 work

Key benefit: GPU and CPU work in parallel without blocking
```

**Implementation:** Backpressure tracking via `PendingLz77` handle returned by `submit_lz77_to_slot()`

## Cost-Model-Driven Batching

**Introduced:** Commit `246ae37` (2026-02-12)

GPU batch scheduling uses kernel cost annotations to predict GPU time:

```wgsl
// @pz_cost {
//   threads_per_element: 1
//   passes: 2
//   buffers: input=N, raw_matches=N*12, resolved=N*12, staging=N*12
//   local_mem: 1024
// }
```

**Cost model predicts:**
- GPU kernel execution time
- Memory bandwidth usage
- Device memory pressure
- Batching decisions (when to stay on CPU vs GPU)

**Validated by:** Commit `b0e942b` (2026-02-12) — Cross-validation tests

## Batching Heuristics

**Current hardcoded limit:** `MAX_GPU_BATCH_SIZE=8` blocks (CLAUDE.md)

### Block Size Auto-Selection

**Introduced:** Commit `48114cc` (2026-02-13)

Auto-reblocking to 128KB for GPU LZ77 pipelines:

**Rationale:**
- Smaller blocks → more batching parallelism
- Larger blocks → higher per-block throughput
- 128KB balances quality and batching efficiency

**Quality regression tests:** Formalized in commit `23247f9` (2026-02-13)

### Decision Tree

Current heuristic (implementation in `src/pipeline/parallel.rs`):

```
Input size < 256 KB?
  ├─ Yes: Use CPU (GPU overhead not worth it)
  └─ No: Is GPU device available?
      ├─ Yes: Estimate GPU time via cost model
      │   ├─ < 100ms: Use GPU
      │   └─ ≥ 100ms: Check memory constraints
      │       ├─ Enough VRAM: Batch up to MAX_GPU_BATCH_SIZE
      │       └─ Memory tight: Reduce batch size or fall back to CPU
      └─ No: Use CPU
```

## GPU Memory Constraints

### Available VRAM

Typical discrete GPU: 4-12 GB VRAM

**Allocation breakdown:**

```
libpz staging overhead:
  Ring buffer (3 slots @ 256KB): ~9.3 MB
  Frequency tables (Huffman):    ~2 MB
  FSE tables:                     ~0.5 MB
  rANS tables:                    ~0.2 MB
  ─────────────────────────────
  Total overhead:                 ~12 MB

Per-batch memory (8 blocks @ 256KB):
  LZ77 buffers: 8 * 9.3 MB = 74.4 MB
  Entropy coding (Huffman): varies by data
  ─────────────────────────────
  Total batch: ~75-100 MB
```

**Safe upper bound:** Allocate ≤25% of device VRAM

### Memory Pressure Handling

Ring buffer backpressure prevents GPU memory exhaustion:

```rust
fn find_matches_batched(&self, blocks: &[&[u8]]) {
    for (i, block) in blocks.iter().enumerate() {
        if ring_full() {
            // Backpressure: wait for GPU slot to free
            self.poll_wait();
        }
        self.submit_lz77_to_slot(block)?;
    }
}
```

## Pipeline-Parallel Batching

**Status:** Experimental/documented (commit `246ae37`)

Different compression stages on different GPU blocks:

```
Block 0: LZ77 encoding (GPU)
Block 1: Huffman encoding (GPU, consumes LZ77 output from Block 0)
Block 2: LZ77 encoding (GPU, independent)
Block 3: Huffman encoding (GPU, consumes LZ77 output from Block 2)
```

**Benefits:**
- GPU utilization stays high (pipelined stages)
- CPU doesn't block on any single stage
- Overlapped GPU compute

**Challenges:**
- Synchronization complexity
- Correct handling of stage dependencies
- Memory allocation for intermediate results

## Known Limitations & Future Work

### 1. Hardcoded MAX_GPU_BATCH_SIZE

**Current:** `MAX_GPU_BATCH_SIZE=8` (hardcoded)

**Issue:** Optimal batching depends on device VRAM and pipeline composition

**Suggested fix (from friction report):** Extract to runtime config or environment variable

### 2. Non-Deterministic GPU Scheduling

**Issue:** Thread scheduling randomizes match results on GPU atomics

**Status:** Mitigated by using cooperative kernel (no atomics)

**Remaining risk:** Future kernels that use atomics should be aware

### 3. GPU Memory Estimation

**Current approach:** Comments in code (e.g., "buffer size = 36×N + overhead")

**Issue:** Only source of truth is actual `create_buffer` calls; estimates easy to miss

**Suggested fix:** Central ARCHITECTURE.md table showing Input Size → Buffer Overhead → Max Batch Size

## Profiling Memory Usage

**Tools:**
- `./scripts/profile.sh` — CPU profiling with samply
- `./scripts/bench.sh` — End-to-end throughput comparison
- `examples/webgpu_diag.rs` — Match quality vs GPU memory trade-off

**GPU-specific:**
- wgpu-profiler integration (`src/webgpu/mod.rs`) — GPU timestamps
- AMD Vulkan driver timestamps unreliable (first query slot returns zeros; workaround in `008d8ba`)

## Reference Implementation

**File:** `/Users/clundquist/code/libpz/src/webgpu/lz77.rs` (ring buffer implementation)

**Key methods:**
- `submit_lz77_to_slot()` — Submit block to ring buffer slot
- `complete_find_matches_coop()` — Retrieve results from slot
- `poll_wait()` — Backpressure synchronization
- Slot allocation/rotation logic in `find_matches_batched()`

## Related Documentation

- **lz77-gpu.md** — LZ77 kernel details
- **research-log.md** — Historical evolution of batching strategies
- **ARCHITECTURE.md** — GPU bottleneck analysis, cost-model notation
- **CLAUDE.md** — Profiling and benchmarking commands
