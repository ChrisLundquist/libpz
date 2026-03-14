# GpuLz Experiment Findings

**Period:** March 2026
**Commits:** `220b055` (kernel + codec), `c05b534` (multi-block + timing),
`dea0c04` (Huffman decode fix), `8390b97` (hybrid buffers)
**Hardware:** AMD Ryzen 9 9950X3D (16C/32T) + AMD Radeon RX 9070 XT (16GB RDNA4, ~40 TFLOPS)

## Summary

We designed and validated a GPU-friendly LZ compression codec ("GpuLz") that uses
sync-point Huffman encoding for parallel decode. The codec works correctly and the
GPU kernel is fast, but **wgpu host overhead dominates GPU dispatch cost**, making
CPU-parallel decompress 5-6x faster than GPU at realistic block counts.

**Key result:** Sync-point parallel Huffman + thread-per-block CPU decompress hits
**1.7 GiB/s** at 32 blocks (7.4x over single-threaded). This is the validated
architecture for pipeline integration.

---

## Codec Design

### Wire Format

```text
BLOCK HEADER:
  [meta_len: u16 LE]              // LzSeq metadata (num_tokens + num_matches)
  [meta: meta_len bytes]
  [num_streams: u8]               // always 6

PER STREAM (x6):
  [orig_len: u32 LE]              // decompressed stream byte count
  [total_bits: u32 LE]            // Huffman bitstream total bits
  [code_lengths: 256 bytes]       // canonical Huffman code lengths
  [num_sync_points: u16 LE]       // count (includes sentinel)
  [sync_points: (bit_offset: u32, symbol_index: u32) x num_sync_points]
  [huffman_data: ceil(total_bits/8) bytes]
```

### Pipeline

```
Input -> LZ77 match-finding -> LzSeq tokenization (6 streams)
      -> Per-stream canonical Huffman + sync points -> Wire format
```

Decode reverses: parse wire -> Huffman decode (parallel per-segment) -> LzSeq reconstruct.

### Sync-Point Mechanism

Every N symbols (default 1024), a checkpoint records `(bit_offset, symbol_index)`.
Each segment between adjacent sync points can be decoded independently — the decoder
starts at the recorded bit offset and emits symbols to the recorded output offset.

A sentinel sync point at the end marks `(total_bits, input_len)`. Number of decode
segments = `num_sync_points - 1`.

---

## Experiment Results

### Experiment 1: Sync-Point Encode Overhead

Overhead of embedding sync points in the Huffman bitstream:

| Interval | Overhead at 128KB | Overhead at 4MB |
|----------|------------------|-----------------|
| 512      | +2.7%            | +2.6%           |
| 1024     | +1.4%            | +1.3%           |
| 2048     | +0.7%            | +0.7%           |

**Conclusion:** Interval=1024 adds only 1.3% wire overhead. Negligible.

### Experiment 2: CPU Tiled Huffman Decode

CPU tiled decode (one segment per iteration) vs monolithic:

| Size | Monolithic | Tiled (1024) | Speedup |
|------|-----------|--------------|---------|
| 128KB | ~160 MiB/s | ~320 MiB/s | 2.0x |
| 4MB  | ~165 MiB/s | ~322 MiB/s  | 2.0x   |

**Conclusion:** Even single-threaded, tiled decode is 2x faster (cache locality).

### Experiment 3: GpuLz Compression Ratio

| Size | GpuLz | LzSeqH | Lzf |
|------|-------|--------|-----|
| 8KB  | 68.5% | 120.4% | 83.4% |
| 64KB | 48.9% | 50.6%  | 46.0% |
| 128KB | 46.6% | 44.4% | 42.1% |
| 4MB  | 30.2% | 25.8%  | 24.5% |

GpuLz is 2-6pp worse than existing pipelines at 128KB+ because sync-point overhead
and canonical Huffman (vs rANS) slightly reduce efficiency. At small sizes (8KB),
GpuLz wins because LzSeqH's per-block rANS overhead is proportionally large.

### Experiment 4: GPU Huffman Decode Kernel

GPU kernel: one thread per sync-point segment, 12-bit LUT decode, WGSL compute shader.

**Single-stream raw Huffman decode at 4MB:**

| Path | Throughput | Relative |
|------|-----------|----------|
| CPU monolithic | 165 MiB/s | 1.0x |
| CPU tiled (1024) | 322 MiB/s | 2.0x |
| GPU sync-point | 751 MiB/s | 4.6x |

**Full GpuLz decompress (GPU Huffman + CPU LzSeq):**

| Path | 4MB throughput |
|------|---------------|
| CPU (1 thread) | 330 MiB/s |
| GPU batched (6 streams, 1 submit) | 450 MiB/s |

**Timing breakdown at 4MB (single block):**

| Phase | Time (us) | % |
|-------|----------|---|
| Parse | 35 | 0.4% |
| GPU buffer creation | 900 | 10% |
| GPU submit + compute | 800 | 9% |
| GPU readback | 800 | 9% |
| **CPU LzSeq** | **6,100** | **69%** |
| **Total** | **8,900** | 100% |

**Key insight:** CPU LzSeq is 69% of total time. The GPU kernel compute is ~200us
inside the 800us submit. The rest is wgpu overhead.

### Experiment 5: Multi-Block Batched Decompress

32 x 128KB blocks (= 4MB total), realistic Canterbury corpus data (~46% ratio):

| Path | Throughput | vs Serial |
|------|-----------|-----------|
| CPU serial (1 thread) | 229 MiB/s | 1.0x |
| **CPU parallel (thread-per-block)** | **1.7 GiB/s** | **7.4x** |
| GPU batched + parallel LzSeq | 284 MiB/s | 1.2x |

**Why GPU loses at scale:**

At 32 blocks x 6 streams = 192 GPU dispatches, the wgpu overhead scales linearly:
- 192 streams x 5 buffers each = 960 buffer allocations
- Each buffer alloc: ~10-50us (GPU memory allocation + upload)
- Total GPU overhead: ~10.9ms (vs ~2.3ms for threaded LzSeq)

The GPU kernel itself is fast (~500us of actual compute for 192 streams), but the
host overhead is 20x larger.

**Scaling behavior:**

| Blocks | GPU Total | GPU Huffman | Parallel LzSeq | GPU Throughput |
|--------|----------|-------------|----------------|---------------|
| 1 | 2.3ms | 1.8ms | 0.4ms | 55 MiB/s |
| 4 | 3.0ms | 2.3ms | 0.7ms | 168 MiB/s |
| 8 | 4.4ms | 3.2ms | 1.0ms | 230 MiB/s |
| 16 | 7.4ms | 5.5ms | 1.6ms | 272 MiB/s |
| 32 | 13.8ms | 10.9ms | 2.3ms | 291 MiB/s |

GPU Huffman time grows ~linearly with stream count (buffer alloc dominated).
Parallel LzSeq scales well (saturates ~16 cores around 16-32 blocks).

### Experiment 6: Fully Merged Buffers (Failed)

**Hypothesis:** Pack all N streams' data into 5 large buffers (one per binding type)
using `BufferBinding` with offsets. Reduce 960 buffer allocations to 5.

**Results at 32 x 128KB (192 GPU streams):**

| Phase | Batched (5N bufs) | Fully Merged (5 bufs) |
|-------|------------------|-----------------------|
| Buffer creation | 5,375us | **4,068us** |
| Submit | 3,291us | 2,150us |
| **Readback** | **3,643us** | **46,227us** |
| **Total GPU** | **14,212us** | **52,445us** |
| Throughput | 216 MiB/s | **66 MiB/s** |

**Root cause: D3D12 UAV barrier serialization.** When all 192 dispatches write to
sub-ranges of the same `read_write` storage buffer, the D3D12 backend (via wgpu-hal)
inserts Unordered Access View barriers between every dispatch, forcing sequential
execution. With separate output buffers, dispatches run in parallel.

**Evidence:** Sub-phase timing showed `poll_wait()` was the entire bottleneck.
Poll time scaled linearly with stream count (1.7ms at 6 streams, 5ms at 24 streams,
~46ms at 192 streams), confirming serialized dispatch execution. Buffer map, copy,
and split were all negligible (<0.1ms combined).

**Conclusion:** Fully merged buffers are a dead end. The buffer creation savings
(6ms) are dwarfed by the dispatch serialization penalty (44ms). Do not merge
`read_write` storage buffers across dispatches in wgpu/D3D12.

### Experiment 7: Hybrid Buffers (Success)

**Hypothesis:** Merge the 4 read-only buffers (bitstream, LUT, sync_points, params)
via `BufferBinding` with sub-range offsets, but keep separate per-stream output +
staging buffers to avoid UAV barrier serialization.

Buffer count: `2N + 4` (vs `6N` batched, vs `6` fully merged).

**Results (same-session comparison):**

| Config | Batched buf/sub/rb | Hybrid buf/sub/rb | Batched MiB/s | Hybrid MiB/s |
|--------|-------------------|-------------------|---------------|--------------|
| 1x128KB | 156/1166/525 | 117/1041/492 | 40 | **44** |
| 4x128KB | 583/1293/746 | 451/1635/738 | 121 | 113 |
| 8x128KB | 1203/1558/1148 | 773/1249/951 | 178 | **210** |
| 16x128KB | 2275/2025/1707 | 1684/1952/1562 | 228 | **249** |
| 32x128KB | 5375/3291/3643 | 4710/3530/3240 | 216 | **221** |

**Key findings:**
- Buffer creation reduced 30-36% across all block counts (4 bulk uploads vs 4N small ones)
- Readback unchanged (same per-stream staging approach)
- Submit slightly higher at some counts (sub-range binding overhead)
- **Peak throughput: 249 MiB/s** (hybrid, 16 blocks) vs 228 MiB/s (batched) — **+9%**
- Sweet spot is 16 blocks; at 32 blocks per-stream output buffer allocation still dominates

**Conclusion:** Hybrid is a clean win. Read-only buffer merging is safe; output buffer
merging is not. The remaining bottleneck is per-stream output + staging buffer allocation
(~4.7ms at 32 blocks = 384 allocations). Next steps would be buffer pooling or native
D3D12 placed resources.

### Experiment 8: Huffman Decode Correctness (Canterbury Corpus)

Fixed two bugs in canonical Huffman decode that caused silent data corruption:

**Bug 1: Tree-walk fallback used pre-canonical topology.** `canonicalize()` reassigns
codewords but doesn't rebuild the tree node structure. Codes <= 12 bits use the 4096-entry
LUT (correct). Codes 13-15 bits fell back to tree walk, which followed the pre-canonical
tree — producing wrong symbols. Fixed with `decode_long_code()`: brute-force lookup search
over the canonical code table.

**Bug 2: Codes > 15 bits silently dropped.** `canonicalize()` only tracked `bl_count[0..16]`.
Degenerate frequency distributions can produce codes > 15 bits (Canterbury block 15 had
16-bit codes). These were silently ignored, producing a corrupt code table. Fixed with
`limit_code_lengths()`: DEFLATE-style depth limiting that caps codes at MAX_CODE_BITS(15)
and redistributes via Kraft inequality.

**Result:** 20/21 Canterbury blocks now round-trip correctly. Block 16 has a pre-existing
LzSeq encoder panic (match length < MIN_MATCH) — separate bug, not Huffman-related.

---

## Architecture Analysis

### Why wgpu Overhead Dominates

Each GPU Huffman stream dispatch requires 5 wgpu objects:
1. **Bitstream buffer** (STORAGE) — upload + GPU alloc
2. **Decode LUT buffer** (STORAGE, 16KB) — upload + GPU alloc
3. **Sync-point buffer** (STORAGE) — upload + GPU alloc
4. **Output buffer** (STORAGE + COPY_SRC) — zero-init + GPU alloc
5. **Params buffer** (UNIFORM, 16 bytes) — upload + GPU alloc

Plus per-stream: 1 bind group, 1 staging buffer for readback.

Per-stream host cost: ~50-100us (buffer creation + upload + bind group).
At 192 streams: ~10-19ms of pure host overhead.

### Merged Buffer Strategy (Tried — Partial Success)

**Fully merged (5 buffers total):** Dead end. D3D12 UAV barriers serialize dispatches
that share a `read_write` storage buffer, even for non-overlapping sub-ranges. 20x
readback regression. See Experiment 6.

**Hybrid (4 merged read-only + N separate output):** Clean win. Reduces buffer
allocations from 6N to 2N+4. Buffer creation 30-36% faster. Peak throughput +9%.
See Experiment 7.

**Remaining bottleneck:** Per-stream output + staging buffer allocation (~4.7ms
at 192 streams). Options:
- **Buffer pooling**: pre-allocate output/staging buffers, reuse across calls
- **Native D3D12**: placed resources in pre-allocated heaps (~1us vs ~25us per buffer)

### Native D3D12 (Not Yet Tried)

The wgpu abstraction adds overhead at several layers:
- **Buffer creation** goes through HAL + validation, even in release mode
- **Memory allocation** is per-buffer (no suballocation from heaps)
- **Synchronization** uses `poll_wait()` (busy-spin) instead of fence-based
- **UAV barriers** are inserted conservatively (can't mark non-overlapping sub-ranges)

With native D3D12:
- **Placed resources** in pre-allocated heaps: ~1us per buffer (vs ~25us in wgpu)
- **Ring buffer** for upload staging: zero per-frame allocation
- **Fence-based sync**: lower CPU overhead than polling
- **Root signatures** with descriptor tables: faster binding than wgpu bind groups
- **Explicit UAV barriers**: could allow merged output buffer if non-overlapping

Expected improvement: 5-10x reduction in host overhead.

**Trade-offs:**
- Windows-only (no Linux/Mac/WebGPU portability)
- Significant implementation effort (~2-4 weeks)
- Must maintain two backends (wgpu for portability, D3D12 for performance)

**Status:** Hybrid wgpu buffers got us to 249 MiB/s (vs 228 MiB/s batched), still
6.8x behind CPU-parallel 1.7 GiB/s. D3D12 could close the gap further but the
cost/benefit ratio is questionable unless GPU decompress is a hard requirement.

### Why Not GPU LzSeq Decode?

LzSeq reconstruction has serial dependencies:
- Each token's output position depends on all prior tokens' output lengths
- Match copies reference earlier output (forward data dependency)
- Repeat offset tracking requires sequential state updates

While ~85% of tokens could theoretically be parallelized via prefix-sum +
parallel scatter, the remaining ~15% (chained match copies) require wavefront
execution. This complexity, combined with the small per-block working set
(128KB), makes GPU LzSeq unlikely to beat CPU single-threaded decode.

The GpuLz GPU encode path already avoids repeat offsets (all offset codes are
literal), which would simplify a future GPU decode kernel. But the practical
value is low given that CPU-parallel per-block decompress already saturates
memory bandwidth.

---

## Files

| File | Description |
|------|-------------|
| `kernels/huffman_decode.wgsl` | GPU sync-point Huffman decode kernel |
| `src/gpulz.rs` | GpuLz codec (compress, decompress, multi-block GPU) |
| `src/webgpu/huffman.rs` | GPU dispatch (batched + merged + hybrid + timed) |
| `src/webgpu/pipelines.rs` | `HuffmanDecodePipelines` registration |
| `src/webgpu/tests/huffman.rs` | GPU decode + GpuLz round-trip tests |
| `benches/stages_gpulz.rs` | Experiments 1-5 criterion benchmarks |

## Known Issues

1. **Canterbury corpus block 16 LzSeq encoder panic:** `encode_length()` panics with
   "attempt to subtract with overflow" when match length < MIN_MATCH(3). Pre-existing
   bug in the LZ77→LzSeq pipeline, not Huffman-related. 20/21 blocks pass.

2. **Compression ratio gap:** GpuLz is 2-6pp worse than existing pipelines. The
   sync-point overhead is only 1.3% — the gap comes from canonical Huffman (vs
   rANS/FSE) and the 6-stream LzSeq framing overhead.

3. **No kernel fusion:** GPU Huffman results are read back to host, then CPU does
   LzSeq reconstruct. Fusing into a single GPU pipeline would avoid the PCIe
   roundtrip but LzSeq decode is serial per-block — poor GPU fit.

---

## Conclusions

1. **Sync-point parallel Huffman is validated.** The codec design works correctly
   and enables excellent parallel decode. Canonical Huffman bugs fixed (Exp 8).

2. **CPU-parallel decompress (1.7 GiB/s) is the clear winner** for the pipeline.
   Thread-per-block with tiled Huffman decode, no GPU needed.

3. **GPU kernel is fast but wgpu host overhead kills it.** The actual compute is
   ~200us for 4MB; the wgpu buffer management is ~4.7ms at 32 blocks (hybrid).
   Hybrid merged buffers improved throughput by 9% (228→249 MiB/s) but the GPU
   path is still 6.8x behind CPU-parallel.

4. **Fully merged output buffers are a dead end.** D3D12 UAV barriers serialize
   dispatches that share a `read_write` storage buffer, even for non-overlapping
   sub-ranges. Only read-only buffers can be safely merged (Exp 6-7).

5. **Remaining GPU overhead is per-stream output+staging buffer allocation.**
   Next steps would be buffer pooling (reuse across calls) or native D3D12
   (placed resources in pre-allocated heaps).

6. **The codec should be integrated with CPU-parallel decompress** into the
   streaming pipeline as the primary decompression path.
