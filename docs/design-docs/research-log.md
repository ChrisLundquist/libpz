# libpz GPU Optimization Journey — Research Log

Comprehensive chronological documentation of major GPU optimization work, kernel iterations, and architectural decisions from the project history.

**Period covered:** Initial GPU experiments → Current state (cooperative-stitch default)
**Last updated:** 2026-02-14

## Executive Summary

The GPU journey progressed through four major phases:

1. **OpenCL Foundation (2024-2025)** — Initial GPU infrastructure with brute-force kernels
2. **Hash Table Experiment (Feb 2025)** — 2x speedup at 1MB+ sizes, but quality collapse on repetitive data
3. **Brute-Force Evolution (Feb 2026)** — Two-tier scanning, shared memory tiling, FAR_STEP tuning
4. **Cooperative-Stitch Era (Feb 2026)** — 25-35% faster than brute-force with 94% quality retention

## Phase 1: OpenCL Foundation

### Initial GPU Infrastructure

| Commit | Date | Description | Impact |
|--------|------|-------------|--------|
| `dee8f17` | 2024-11 | Add OpenCL framework for GPU-accelerated compression | Foundation: GPU context, queue, kernel loading |
| `537f881` | 2024-12 | Trivial LZ77 encoder in OpenCL | First working GPU LZ77 kernel |
| `d948dc8` | 2025-01 | Add OpenCL BWT kernel, benchmark infrastructure, CLI --gpu flag | GPU BWT (radix sort), benchmark harness |
| `7a41169` | 2025-01 | Fix OpenCL data race and slim CI benchmarks | Data race in hash bucket insertion |

**Key findings from Phase 1:**
- Brute-force O(n*w) GPU kernels were functional but slow (10-50x slower than CPU)
- GPU overhead dominated on small inputs (<64KB)
- Hash-table approach (like CPU) seemed like the natural next step

### WebGPU Migration Begins

| Commit | Date | Description | Impact |
|--------|------|-------------|--------|
| `c3092cc` | 2025-02 | GPU backend migration evaluation: OpenCL → wgpu/Vulkan | Evaluation memo; WebGPU promising due to broad platform support |
| `ce32f9d` | 2025-02 | Optimize WebGPU backend: close perf gap with OpenCL | Early WebGPU kernels within 80-90% of OpenCL speed |
| `6be6e15` | 2025-02 | Fix WebGPU dispatch limit panic and WGSL reserved keyword errors | WGSL language fixups |

---

## Phase 2: Hash Table Experiment

### Hash-Table LZ77 Kernel Introduced

**Commit:** `d966133` (2026-02-10)  
**Metric:** 2x faster than CPU at 1MB input size  
**Files changed:** `src/webgpu/lz77.rs`, kernel code, `CLAUDE.md`

```
64KB:  GPU hash 1.6ms (38 MiB/s) vs CPU hashchain 1.3ms (48 MiB/s)
256KB: GPU hash 3.1ms (82 MiB/s) vs CPU hashchain 6.2ms (40 MiB/s) — 2x faster
1MB:   GPU hash 9.6ms (104 MiB/s) vs CPU hashchain 20ms (50 MiB/s) — 2x faster
```

**Design:** Two-pass hash-table approach
- Pass 1: Parallel atomic insertion into fixed-size hash buckets
- Pass 2: Bounded bucket search (MAX_CHAIN=64 per position)
- Matched CPU's hash-chain strategy

**Why it failed:**

**Commit:** `ef067d0` (2026-02-13)  
**Title:** "Replace WebGPU LZ77 hash table with near brute-force scan"

The hash-table kernel had a **fundamental flaw**: parallel atomic insertion filled fixed-size buckets with early positions, causing catastrophic match quality loss on repetitive data:

```
Repetitive data (1MB):
  Hash-table quality: 6.25% match ratio
  CPU reference: 99.61% match ratio
  Ring buffers don't help — GPU thread scheduling randomizes insertion order
```

**Root cause:** GPU threads executed in arbitrary order, filling buckets with whatever positions got there first, not the most-recent positions that CPU hash-chains maintain.

---

## Phase 3: Brute-Force Evolution

### Near-Window Brute-Force Strategy

**Commit:** `ef067d0` (2026-02-13)  
**Title:** "Replace WebGPU LZ77 hash table with near brute-force scan"

Abandoned hash table; replaced with simple backward scan of NEAR_WINDOW (1024 bytes):

```
Match quality fixed:
  - Repetitive data: 6.25% → 99.99% match ratio
  - alice29.txt: maintained high quality
  - Memory overhead: 134MB (hash tables) → 0 (no aux buffers)
  - Kernel passes reduced: 3 → 2

Quality tradeoff:
  - Binary data ~3% lower (1024 vs 32768 window)
  - Addressable later with hash probes for long-range matches
```

**Diagnostic tools added:** `examples/webgpu_diag.rs` — CPU vs GPU match quality and speed comparison

### Two-Tier Window Scanning

**Commit:** `9395204` (2026-02-13)  
**Title:** "Add two-tier far window scanning to WebGPU LZ77 matcher"

Extended near-window scan with subsampled far-window search (1024-32768, every 4th position):

```
alice29.txt results:
  Match ratio: 0.667 (near-only) → 0.796 (two-tier)
  CPU reference: 0.819
  Quality: ~97% of CPU (only 12% larger compressed size)
  
Probe count: ~10K per position
  Near (1-1024): full brute-force
  Far (1024-32768): every 4th position
```

**Impact:** Captured 55% of missed match bytes from far distances while keeping probe count manageable.

### Shared Memory Tiling Optimization

**Commit:** `27bdf12` (2026-02-13)  
**Title:** "Add workgroup shared memory tiling to LZ77 near-window scan"

Cooperative tile loading into var<workgroup> shared memory:

```
Design:
  - 64-thread workgroup cooperatively loads ~1280 bytes (320 u32 words)
  - Tile covers near-window lookback region
  - ~65K scattered global reads → fast shared memory reads

Performance:
  - AMD Radeon Pro 5500M / Metal: neutral (~81ms for 256KB)
  - Likely helps on discrete GPUs with weaker L2 cache
  - Match quality unchanged; all 648 tests pass

Implementation details:
  - try_match_near(): tile-based spot-checks
  - try_match_far(): global reads (working set too large)
  - Fallback to global memory when position exceeds tile
  - workgroupBarrier() placed before early returns for correctness
```

**File:** `/Users/clundquist/code/libpz/kernels/lz77_lazy.wgsl` (lines 43-120)

### FAR_STEP Tuning — 37% Speedup

**Commit:** `a6ae499` (2026-02-13)  
**Title:** "Increase FAR_STEP from 4 to 8 for 37% LZ77 speedup"

Far window scan was 72% of kernel time. Doubling step size:

```
Changes:
  FAR_STEP: 4 → 8
  Result: ~9K probes → ~3872 probes per position
  GPU time: 82ms → 52ms for 256KB
  Speedup: 37% (4.8 MB/s)

Quality tradeoff (alice29.txt):
  Match ratio: 0.796 → 0.772 (3% loss)
  Still 94% of CPU quality (0.819)
  Full pipeline GPU deflate: 2.5 MB/s → 4.0 MB/s

Summary:
  Before: 82ms (3.1 MB/s), ratio 0.796
  After:  52ms (4.8 MB/s), ratio 0.772
  CPU:     5ms (47 MB/s),  ratio 0.819
```

---

## Phase 4: Cooperative-Stitch Era

### Cooperative-Stitch Kernel Introduction

**Commit:** `1fce493` (2026-02-13)  
**Title:** "Add cooperative-stitch LZ77 match finding kernel"

New kernel implementing cooperative search strategy where each thread in a 64-thread workgroup searches a distinct offset band, shares discoveries via shared memory, then all threads re-test discovered offsets:

**Algorithm:**

```
Phase A: Initial search (per-thread)
  - Each thread searches [1, 64] near window
  - Plus strided band [t*64+1, t*64+256]
  - Discovers top-K offsets locally

Phase B: Cooperative stitching (after barrier)
  - All threads share discoveries via shared memory
  - Each thread tries all 63*4=252 offsets discovered by others
  - Total: 572 probes/thread covering [1, 4288] effective lookback
    (vs 4896 probes for brute-force of same range)

Key insight:
  If offset d produces good match at position p,
  nearby positions p+1, p+2, ... are likely inside same repeated region.
  So offset d produces good (slightly shorter) matches there too.
```

**Benchmark on AMD Radeon Pro 5500M / Metal:**

```
| File          | BF ratio | Coop ratio | BF ms | Coop ms | Speedup |
|---------------|----------|------------|-------|---------|---------|
| text 256KB    | 0.9998   | 0.9998     | 207   | 208     | 1.0x    |
| alice29.txt   | 0.772    | 0.532      | 31    | 8.3     | 3.7x    |
| asyoulik.txt  | 0.759    | 0.490      | 24    | 6.6     | 3.7x    |
| kennedy.xls   | 0.913    | 0.909      | 210   | 56      | 3.7x    |
```

**Quality analysis:**
- kennedy.xls: <1% loss (matches mostly within 4K)
- text: identical quality (matches within 256 bytes)
- alice29/asyoulik: ~30% loss (54% of CPU matches use offsets >1K)

**Files:**
- New kernel: `kernels/lz77_coop.wgsl` (345 lines)
- Test harness: `examples/coop_test.rs` (115 lines)
- Host code: `src/webgpu/lz77.rs` (+142 lines)

### Cooperative Kernel Tuning

**Commit:** `b06cfb4` (2026-02-13)  
**Title:** "Tune cooperative kernel: NEAR=1024, S=512, W=512 for balanced quality/speed"

Explored Pareto frontier of probe count vs match quality:

```
| Config                  | alice29 ratio | speedup | probes |
|-------------------------|---------------|---------|--------|
| BF (baseline)           | 0.772         | 1.0x    | 4896   |
| N=64, S=64, W=256       | 0.532         | 3.7x    | 572    |
| N=64, S=512, W=512      | 0.618         | 2.5x    | 828    |
| N=1024, S=512, W=512    | 0.728         | 1.8x    | 1788   |
| N=1024, S=512, W=1024   | 0.741         | 1.4x    | 2300   |
```

**Selected configuration:** N=1024, S=512, W=512

Rationale:
- 1.8x faster than brute-force on natural text
- 94% of brute-force match quality on alice29/asyoulik
- Equal or better quality on structured data (kennedy.xls)
- Exhaustive near window (1024) dominates quality (~45% of CPU match bytes within offset 1K)
- Cooperative far search adds incremental coverage for remaining ~55%

**Key finding:** Near-window quality matters more than far-window coverage on natural text.

**File modified:** `kernels/lz77_coop.wgsl` (28 lines changed)

### Making Cooperative Kernel the Default

**Commit:** `7457360` (2026-02-13)  
**Title:** "Switch default GPU LZ77 match finder to cooperative-stitch kernel"

Integrated cooperative kernel into all four GPU LZ77 dispatch paths:

```
Dispatch paths now using coop kernel:
  1. find_matches() — single-block download path
  2. find_matches_to_device() — single-block on-device path
  3. find_matches_batched() — multi-block batched path
  4. submit_lz77_to_slot() — pre-allocated buffer ring path

Performance summary:
  - 25-35% faster than brute-force lazy kernel
  - Tested on AMD Radeon RX 9070 XT / RDNA 4
  - Comparable compression quality maintained

Lazy kernel code retained (#[allow(dead_code)]) for A/B benchmarking.
```

**Files changed:** `src/webgpu/lz77.rs` (115 +/- lines), `src/webgpu/mod.rs` (6 additions)

---

## Ring-Buffered Streaming Optimization

### Eliminating Buffer Allocation Overhead

**Commit:** `4390d90` (2026-02-13)  
**Title:** "Use ring-buffered streaming for batched LZ77 match finding"

Replaced per-block buffer allocation in `find_matches_batched` with pre-allocated ring slots (double/triple-buffered). Eliminated 35% alloc/map overhead, achieving 17% faster batched compression. See `gpu-batching.md` for architecture details and full benchmark numbers.

**Files changed:** `src/webgpu/bwt.rs`, `src/webgpu/lz77.rs` (128 +/- lines), `src/webgpu/mod.rs` (23 +/- lines)

**Profiling fix:** `008d8ba` (2026-02-13) — Fixed GPU profiling timestamps for LZ77 coop kernel to work around AMD Vulkan driver bug (first query slot returns zeros)

---

## Backend Architecture Evolution

### OpenCL → WebGPU Transition

**Period:** Feb 2025 — Feb 2026

Key commits tracking the transition:

| Commit | Date | Description | Status |
|--------|------|-------------|--------|
| `c3092cc` | 2025-02 | GPU backend migration evaluation | Planning |
| `04ccaf6` | 2025-02 | Add OpenCL GPU profiling and macOS compatibility fixes | Improving OpenCL |
| `9ec7857` | 2025-02 | Remove Vulkan backend, add --list-pipelines CLI | Consolidation |
| `e8af146` | 2026-02 | Add WebGPU DeviceBuf zero-copy abstraction | Parity with OpenCL |
| `e71f1e8` | 2026-02 | Modular GPU pipeline with zero-copy DeviceBuf | Refactoring |
| `22fe18a` | 2026-02-13 | Upgrade wgpu from 24 to 27 | Infrastructure |
| `9e24938` | 2026-02-13 | Integrate wgpu-profiler for GPU timestamp profiling | Profiling capability |
| `6504641` | 2026-02-13 | Remove OpenCL backend entirely in favor of WebGPU | **Final consolidation** |

**Final state:** WebGPU only; OpenCL completely removed (commit `6504641`).

### Entropy Coding on GPU

**Huffman GPU Implementation:**

| Commit | Date | Description | Impact |
|--------|------|-------------|--------|
| `ee8f79d` | 2026-02-12 | Add stage_huffman_encode_gpu() and wire modular GPU Deflate pipeline | On-device Huffman |
| `b4bf45d` | 2026-02-13 | Add GPU prefix sum and fully-on-device Huffman encode to WebGPU | Full Huffman on device |
| `67e449e` | 2026-02-13 | Replace GPU Blelloch prefix sum with CPU prefix sum in Huffman encoder | Optimization |

**FSE GPU Implementation:**

| Commit | Date | Description | Impact |
|--------|------|-------------|--------|
| `7879eb1` | 2026-02-12 | Add GPU FSE encode for Lzfi pipeline (OpenCL + WebGPU) | New pipeline |
| `e66af7c` | 2026-02-12 | Add WebGPU FSE encode host code and pipeline wiring | Host integration |
| `f08561a` | 2026-02-12 | Add GPU FSE encode tests and cost annotations | Testing |
| `e653eaa` | 2026-02-12 | Wire GPU FSE decode into decompression pipeline and CLI | Decompression |
| `57fedb1` | 2026-02-13 | Fix GPU FSE interleaved encode lane count: 32 → 4 | Bug fix |

---

## Experimental Features & Learnings

### Quality Regression Tests

**Commit:** `23247f9` (2026-02-13)  
**Title:** "Add GPU LZ77 match quality regression tests"

Formalized quality metrics across Canterbury corpus and artificial test data.

### Block Sizing Experiments

**Commit:** `48114cc` (2026-02-13)  
**Title:** "Auto-reblock GPU LZ77 pipelines to 128KB and relax quality test bounds"

Auto-adjustment of block sizes for GPU pipelines based on empirical quality measurements.

### Pipeline Stream Switching

**Commit:** `3b162de` (2026-02-13)  
**Title:** "Switch lzfi pipeline from LZ77 3-stream to LZSS 4-stream demux"

Optimization of pipeline-specific demuxing strategy.

---

## Key Metrics & Learnings

### GPU Profiling Infrastructure

**Timeline:**
- `9e24938` (2026-02-13): Added wgpu-profiler integration for GPU timestamps
- `008d8ba` (2026-02-13): Fixed profiling for cooperative kernel (AMD driver workaround)
- `730a2c7` (2026-02-13): Added scripts/test.sh wrapper for testing

### Quality vs Speed Trade-offs

**Summary of cooperative kernel tuning:**
- Exhaustive near-window (NEAR=1024) covers ~45% of match bytes on natural text
- Far-window coverage provides diminishing returns (strided scan adequate)
- 1.8x speedup at cost of 6% quality loss is favorable trade-off
- Quality still matches CPU on structured data

### Memory Efficiency

**Buffer allocation patterns:**
- Hash table approach: 134MB overhead (failed due to quality)
- Brute-force near-only: 0 overhead (too low quality for far matches)
- Ring-buffered streaming: pre-allocated, eliminates per-block overhead (17% gain on 4MB batch)

---

## Current State (as of 2026-02-14)

**Default GPU LZ77 kernel:** Cooperative-stitch (`lz77_coop.wgsl`)
- 25-35% faster than brute-force lazy kernel
- 94% of brute-force match quality on natural text
- All four dispatch paths integrated

**Entropy coding:** Fully GPU-accelerated (Huffman, FSE, rANS on device)

**Backend:** WebGPU only (OpenCL removed)

**Profiling:** wgpu-profiler integrated with AMD driver workarounds

**Outstanding optimization opportunities:**
1. Hash probes for extremely long-distance matches (>32KB offset)
2. SIMD-accelerated rANS decode interleaving
3. Further reduction in GPU startup overhead (lazy compilation enabled)
4. Cross-platform profiling reliability (AMD driver timestamps)

---

## References & Related Documentation

- **ARCHITECTURE.md** — Detailed GPU notes, bottleneck analysis, roadmap
- **CLAUDE.md** — Day-to-day development instructions, build commands
- **.claude/friction/2026-02-14-lz77-gpu-research-friction.md** — Known impediments and workflow issues
- **Kernel files** — `/Users/clundquist/code/libpz/kernels/lz77_*.wgsl`
- **GPU module** — `/Users/clundquist/code/libpz/src/webgpu/`
