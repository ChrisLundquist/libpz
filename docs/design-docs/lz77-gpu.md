# LZ77 GPU Implementation Details

Comprehensive guide to the LZ77 GPU implementation in libpz, including current state, architecture, and known bottlenecks.

**Last updated:** 2026-02-14

## Overview

LZ77 GPU implementation uses a cooperative-stitch kernel strategy that achieves 25-35% speedup over brute-force lazy matching while maintaining ~94% match quality on natural text.

**Status:** Default GPU LZ77 algorithm (all dispatch paths integrated)

## Kernel Variants

### 1. Cooperative-Stitch Kernel (DEFAULT)

**File:** `/Users/clundquist/code/libpz/kernels/lz77_coop.wgsl` (345 lines)

**When it was introduced:** Commit `1fce493` (2026-02-13)  
**When it became default:** Commit `7457360` (2026-02-13)

#### Algorithm

Two-pass cooperative search:

**Pass 1: find_matches_coop**
- Phase A (per-thread search):
  - Each thread searches near window [1, NEAR_RANGE=1024]
  - Plus strided offset band: [t*STRIDE+1, t*STRIDE+WINDOW_SIZE]
  - Retains top-K matches in registers

- Phase B (cooperative stitching, after workgroup barrier):
  - All threads share discoveries via shared memory (1024 bytes)
  - Each thread re-tests all 63*4=252 offsets discovered by other threads
  - Key insight: if offset d produced good match at p, it likely produces good matches at p+1, p+2, ... (same repeated region)

**Pass 2: resolve_lazy**
- Standard lazy resolution: if pos+1 has strictly longer match, demote pos to literal
- Only runs on positions actually found in Pass 1

#### Coverage & Probes

```
Near window [1, 1024]:         1024 probes (all threads)
Far window [1025, 33792]:      64 * (WINDOW_SIZE/STRIDE) = 512 probes per thread
Cooperative stitching:         252 probes (re-testing other threads' offsets)
Total per position:            ~1788 probes
Effective range:               [1, 33792]

Comparison:
  Brute-force scan [1, 32768]: ~4896 probes per position
  Cooperative-stitch:          ~1788 probes per position (64% reduction)
```

#### Performance Profile

**Benchmark (Feb 2026, AMD Radeon Pro 5500M / Metal):**

```
Configuration: NEAR=1024, S=512, W=512

| File          | Ratio | Speed   | Speedup | Quality |
|---------------|-------|---------|---------|---------|
| alice29.txt   | 0.728 | 1.8x    | vs BF   | 94%     |
| asyoulik.txt  | 0.685 | 1.8x    | vs BF   | 94%     |
| kennedy.xls   | 0.913 | 1.8x    | vs BF   | ~100%   |
| text 256KB    | 1.000 | 1.0x    | vs BF   | 100%    |
```

**Larger scale benchmarks:**

```
Batched 16-block (4MB) after ring-buffering:
  Speed: 70 ms total (57 MB/s)
  Per-block equivalent: ~4.4 ms per 256KB block

Full deflate pipeline (GPU LZ77 + GPU Huffman):
  Speed: 89 ms (7% faster than before ring-buffering)
  Throughput: Dependent on entropy coding overhead
```

#### Quality Analysis

**Key finding:** Near-window exhaustiveness matters more than far-window coverage

- 1024-byte near window covers ~45% of match bytes on natural text
- Far-window search provides diminishing returns
- 6% quality loss on text files is acceptable trade-off for 1.8x speedup
- Structured data (excel sheets) maintains near-100% quality

#### Implementation Details

**Host code:** `/Users/clundquist/code/libpz/src/webgpu/lz77.rs`

Key functions:
- `find_matches(&self, input)` — Public API, calls `find_matches_coop_impl()`
- `submit_find_matches_coop(&self, input)` — Submit without blocking
- `complete_find_matches_coop()` — Retrieve results after poll_wait()
- `submit_lz77_to_slot()` — Ring-buffered streaming variant

**Dispatch paths using coop kernel:**
1. `find_matches()` — Single-block download path
2. `find_matches_to_device()` — Single-block on-device path
3. `find_matches_batched()` — Multi-block batched path (ring-buffered)
4. `submit_lz77_to_slot()` — Pre-allocated buffer ring path (streaming)

**GPU Resource Cost (from @pz_cost annotation):**
```
Threads per element: 1
Passes: 2
Buffers: input=N, raw_matches=N*12, resolved=N*12, staging=N*12
Local memory (workgroup): 1024 bytes (shared_topk: 256 u32s for top-K)
Register pressure: Moderate (stores 4 top-K entries per thread)
```

### 2. Brute-Force Lazy Kernel (LEGACY)

**File:** `/Users/clundquist/code/libpz/kernels/lz77_lazy.wgsl` (11663 bytes)

**Status:** Retained for A/B benchmarking (#[allow(dead_code)])

#### Algorithm

Simple backward scan with two-tier window:
- Near window [1, 1024]: full brute-force scan
- Far window [1024, 32768]: subsampled scan (every FAR_STEP=8 positions)

**Optimizations included:**
- Workgroup shared memory tiling for near-window spot-checks (1280 bytes tile)
- Spot-check first bytes before full comparison
- U32-wide comparison in match extension

**Performance:**
- Slower than cooperative kernel (~1.8x)
- Slightly better quality (~6% higher on natural text)
- Kept for reference comparisons

### 3. Hash-Table Kernel (REMOVED)

**Status:** Abandoned in commit `ef067d0` (2026-02-13). Quality collapsed on repetitive data due to non-deterministic GPU atomic insertion order. See `experiments.md` "Hash-Table LZ77 Kernel" for full analysis.

## Ring-Buffered Streaming

**Introduced:** Commit `4390d90` (2026-02-13)

The multi-block batched path uses pre-allocated ring buffers instead of per-block allocation, eliminating 35% overhead and enabling overlapped GPU/CPU execution. See `gpu-batching.md` for the `Lz77BufferSlot` struct, buffer allocation formula, and performance benchmarks.

## Data Flow Through Pipeline

See `pipeline-architecture.md` for the full data flow from `compress_block()` through demux and entropy coding. The key integration points for LZ77 GPU are `submit_lz77_to_slot()` (async dispatch) and `complete_find_matches_coop()` (result retrieval via backpressure loop).

## Known Bottlenecks & Optimization Opportunities

### 1. Far-Window Coverage (Medium priority)

**Current state:**
- Strided scan (FAR_STEP=8) covers [1024, 32768] at 1.8x speedup
- Misses ~55% of far-window match bytes on natural text

**Opportunities:**
- Hash probes for extremely long-distance matches (>32KB offset)
- Tunable FAR_STEP (trade speed for quality)
- Offset distribution analysis available in `examples/webgpu_diag.rs`

### 2. GPU Driver Non-Determinism (Low priority, documented)

**Issue:** AMD Vulkan driver timestamps unreliable; first query slot returns zeros

**Workaround:** Timestamp fix in commit `008d8ba` (2026-02-13)

**Status:** Documented in `.claude/friction/2026-02-14-lz77-gpu-research-friction.md`

### 3. GPU Startup Overhead (Addressed)

**Previous issue:** Lazy kernel compilation overhead on first use

**Solution:** Lazy pipeline compilation in commit `c856239` (2026-02-13) and `fdc5198` (2026-02-12)

### 4. Block Sizing Heuristics (Addressed)

**Previous issue:** GPU LZ77 block sizes not optimized for quality

**Solution:** Auto-reblocking to 128KB in commit `48114cc` (2026-02-13) with quality regression tests

## Configuration Parameters

### Kernel Constants (lz77_coop.wgsl)

```wgsl
const WG_SIZE: u32 = 64u;              // Workgroup size
const TOP_K: u32 = 4u;                 // Top-K matches per thread
const NEAR_RANGE: u32 = 1024u;         // Exhaustive near window
const WINDOW_SIZE: u32 = 512u;         // Far window step size
const STRIDE: u32 = 512u;              // Thread band spacing
const MIN_MATCH: u32 = 3u;             // Minimum match length
const LAZY_SKIP_THRESHOLD: u32 = 32u;  // Skip lazy evaluation
const GOOD_ENOUGH: u32 = 128u;         // Early-exit threshold
```

**Tuning notes (from commit `b06cfb4`):**
- NEAR_RANGE=1024 essential for quality (45% of match bytes)
- STRIDE=512, WINDOW_SIZE=512 balanced for 1.8x speedup
- Increasing STRIDE costs quality without proportional speedup
- Increasing WINDOW_SIZE adds probes but diminishing returns

### Runtime Configuration

**Batching:** `MAX_GPU_BATCH_SIZE=8` (hardcoded, per CLAUDE.md)

**Memory overhead per block:**
- Input: N bytes
- Raw matches: N * 12 bytes (offset u32, length u32, next u32)
- Resolved: N * 12 bytes (filtered/lazy-resolved)
- Staging: N * 12 bytes (GPU→CPU readback)
- Total: ~36N + buffer management overhead

## Testing & Validation

**Quality regression tests:** `src/validation.rs` — GPU↔CPU cross-decompression

**Diagnostic harness:** `examples/coop_test.rs` — A/B benchmarking vs brute-force

**Offset distribution analysis:** `examples/webgpu_diag.rs` — Understand quality/speed trade-offs

## Related Documentation

- **ARCHITECTURE.md:** GPU benchmarks, bottleneck analysis
- **research-log.md:** Historical evolution of kernels
- **gpu-batching.md:** Batching strategies and memory management
- **CLAUDE.md:** Build commands, profiling instructions
