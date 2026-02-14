# Pipeline Architecture & Data Flow

Detailed guide to multi-stage compression pipelines, data flow patterns, and block processing.

**Last updated:** 2026-02-14

## Overview

libpz uses modular, composable compression pipelines with three major patterns:

1. **Block-parallel** — Multiple independent blocks compressed in parallel
2. **Pipeline-parallel** — Different stages process different blocks (streaming)
3. **GPU-batched** — Multiple blocks submitted to GPU at once with backpressure

## Core Concepts

### V2 Container Format

The multi-block container format (committed in earlier phases, used throughout):

```
[Header: 4 bytes = "Pz" + version + flags]
[Block 0: compressed data + metadata]
[Block 1: compressed data + metadata]
[...]
```

**Per-block format (multistream):**
```
[num_streams: u8]
[pre_entropy_len: u32 LE]
[meta_len: u16 LE]
[meta: entropy codec-specific data]
[stream_0: entropy-encoded bytes...]
[stream_1: entropy-encoded bytes...]
[...]
```

### Stream Demuxing

Multi-stream pipelines (Deflate, Lzf, Lzr) split LZ77 output into independent streams:

| Stream | Contents | Why separate? |
|--------|----------|---------------|
| Offsets | High bytes of match offsets (offset >> 8) | Narrow range; dedicated table helps |
| Lengths | Match lengths (capped to u8) | Highly skewed distribution (short matches dominate) |
| Literals | Literal bytes + low offset bytes | Natural distribution; easy to predict |

**Demuxer interface:** `StreamDemuxer` trait in `src/pipeline/demux.rs`

**Implementation:**
```rust
pub trait StreamDemuxer {
    fn stream_count(&self) -> usize;
    fn demux(&self, matches: &[Match]) -> Vec<Vec<u8>>;
    fn remux(&self, streams: &[Vec<u8>]) -> Vec<u8>;
}
```

## Pipeline Implementations

### Deflate Pipeline (Baseline)

**Stages:** LZ77 → Huffman

**Files:** `src/pipeline/blocks.rs`, `src/webgpu/huffman.rs`

**Multi-stream:** 3 streams (offsets, lengths, literals)

**Dispatch paths:**
- CPU LZ77 + CPU Huffman (baseline)
- GPU LZ77 + GPU Huffman (GPU batched)
- GPU LZ77 + CPU Huffman (mixed mode for small batches)

**Block size:** 256KB (default), auto-reblocked to 128KB for GPU

### Lzf Pipeline (LZ77 + FSE)

**Status:** Experimental, GPU-accelerated

**Stages:** LZ77 → FSE (Finite State Entropy)

**Multi-stream:** 3 streams

**Advantages over Deflate:**
- FSE often better compression than Huffman on structured data
- Faster decoding on CPU

**GPU implementation:** `src/webgpu/fse_encode.wgsl` (5616 bytes)

### Lzr Pipeline (LZ77 + rANS)

**Status:** Experimental

**Stages:** LZ77 → rANS (Range ANS)

**Multi-stream:** 3 streams

**Advantages:**
- rANS approaches Shannon entropy (better than Huffman)
- GPU-parallelizable decode via interleaved state machines

## Data Flow Through a Block

### Compression Path

```
compress_block(input_block) in src/pipeline/blocks.rs
  ├─ Demux dispatch (demuxer_for_pipeline)
  │  ├─ For Deflate: Call Lz77Demuxer
  │  └─ For LzDemuxer: Demux LZ77 matches → 3 streams
  │
  ├─ LZ77 compression (src/pipeline/stages.rs)
  │  ├─ CPU: lz77::compress_lazy(input)
  │  └─ GPU: webgpu::submit_lz77_to_slot(input)
  │
  ├─ Stage block structure population
  │  ├─ StageBlock.data = None (for GPU LZ77)
  │  └─ StageBlock.streams = Some(Vec of match bytes) after demux
  │
  ├─ Entropy encoding (src/pipeline/blocks.rs)
  │  ├─ stage_huffman_encode_gpu() for Deflate
  │  ├─ stage_fse_encode_gpu() for Lzf
  │  └─ stage_rans_encode_cpu() for Lzr
  │
  └─ Write to container format
     └─ Multistream header + compressed streams
```

### Key Data Structure: StageBlock

```rust
pub struct StageBlock {
    pub input: Vec<u8>,           // Original input
    pub data: Option<Vec<u8>>,    // After stage 1 processing
    pub streams: Option<Vec<Vec<u8>>>,  // After demux (entropy coders read this)
    pub output: Vec<u8>,          // Final compressed output
}
```

**Critical handoff:** `streams` field
- `None` before demux
- `Some(Vec<Vec<u8>>)` after demux
- Entropy encoders consume `streams`, not `data`

### Decompression Path

```
decompress_block(compressed_data) in src/pipeline/blocks.rs
  ├─ Read container header (stream count, sizes)
  │
  ├─ Entropy decoding (stage_*_decode functions)
  │  ├─ stage_huffman_decode_gpu() for Deflate
  │  ├─ stage_fse_decode_gpu() for Lzf
  │  └─ stage_rans_decode_cpu() for Lzr
  │  └─ Result: Vec<Vec<u8>> of decompressed streams
  │
  ├─ Remux (stream reassembly)
  │  └─ Demuxer::remux() restores original match format
  │
  ├─ LZ77 decompression (stage_lz77_decompress)
  │  ├─ CPU: lz77::decompress(remuxed_data)
  │  └─ GPU: webgpu::lz77_decompress_block(remuxed_data) for block-parallel decompress
  │
  └─ Output: decompressed block
```

## Block Parallelism Strategies

### 1. Block-Parallel Compression

**File:** `src/pipeline/parallel.rs` - `compress_parallel()`

Multiple independent blocks processed in parallel:

```
Input: Large file or memory region
       ├─ Split into N blocks (256KB each)
       └─ Process in parallel:
           ├─ Block 0 → thread 0 → compress_block()
           ├─ Block 1 → thread 1 → compress_block()
           ├─ Block 2 → thread 2 → compress_block()
           └─ Block N → thread N → compress_block()

Synchronization: Join all threads before writing container
```

**Implementation:** Rayon thread pool, one block per thread

**When used:** Default for >256KB inputs with multiple CPU cores

### 2. GPU-Batched Compression

**File:** `src/pipeline/parallel.rs` - `compress_parallel_gpu_batched()`

Multiple blocks submitted to GPU with ring-buffered streaming:

```
Input: 8 blocks (2MB batch)

Loop:
  Iteration 0:
    GPU: LZ77 on block 0 (async, non-blocking)
    CPU: write metadata for previous block (overlapped)
    
  Iteration 1:
    GPU: LZ77 on block 1 (while block 0 still computing)
    CPU: entropy code block 0, write output
    
  ... ring buffer enables pipelining ...
```

**Ring buffer:** 3 pre-allocated slots, managed via `Lz77BufferSlot`

**Backpressure:** `poll_wait()` called when ring full to prevent GPU queue overflow

**When used:** GPU available AND input ≥ 256KB AND not too many small blocks

### 3. Pipeline-Parallel Compression

**Status:** Documented but not heavily used

**Idea:** Different stages on different GPU blocks

```
Block 0: GPU LZ77 → GPU Huffman (Stage A + Stage B)
Block 1: GPU LZ77 → GPU Huffman (Stage A + Stage B)

GPU timeline (idealized):
  [Block 0 LZ77] [Block 0 Huff] [Block 1 LZ77] [Block 1 Huff] ...
                                [Block 1 LZ77]
```

**Challenges:**
- Synchronization between dependent blocks
- Memory allocation for intermediate results
- Work balancing across stages

## Auto-Selection Strategy

**File:** `src/pipeline/mod.rs` - `select_pipeline()`

Current heuristic (simplistic):

```
if entropy > threshold_high
  ├─ Data is random/incompressible
  └─ Use: Deflate (fast, minimal overhead)
else if match_density > threshold_high
  ├─ Data has many repeating patterns
  └─ Use: Lzf or Lzr (better entropy coding)
else
  └─ Use: Deflate (balanced for mixed data)
```

**Advanced selection:** `select_pipeline_trial()` (mentioned in CLAUDE.md)

**Future work:** ML-based pipeline selection (not implemented)

## GPU Cost Annotations

**File:** `kernels/*.wgsl` - `@pz_cost` blocks

Kernels are annotated with resource usage for cost-model-driven scheduling:

```wgsl
// @pz_cost {
//   threads_per_element: 1          # Parallelism: 1 thread per byte processed
//   passes: 2                        # Number of dispatch passes
//   buffers: input=N, output=N*12   # GPU memory: N bytes input, N*12 output
//   local_mem: 1024                 # Workgroup shared memory
// }
```

**Used by:** Cost model in `src/webgpu/gpu_cost.rs` to predict execution time and decide GPU vs CPU

## Known Pitfalls

### 1. Multi-Stream Format Changes

**Risk:** Changing stream count or byte ordering breaks round-trip compatibility

**Example:** Commit `3b162de` switched lzfi from LZ77 3-stream to LZSS 4-stream
- LZ77: 3 streams (offsets, lengths, literals)
- LZSS: 4 streams (additional level in encoding)
- **Must update demuxer** before changing stream format

**Safe approach:** Increment container version number when changing format

### 2. GPU vs CPU Output Mismatch

**Risk:** GPU and CPU produce different output for same input

**Root cause:** Usually in demux (stream splitting), not entropy coding

**Debug strategy:**
1. Check stream count (most common issue)
2. Check byte ordering (little-endian expectations)
3. Compare CPU output vs GPU output directly
4. Trace through `stage_demux_compress()` step-by-step

### 3. StageBlock.streams Assumptions

**Risk:** Code assumes `streams` is populated but it's `None`

**Example:** Entropy coder tries to read from `streams[0]` but field is `None`

**Cause:** Forgot to call demuxer, or demuxer returned empty results

**Safe approach:** Use `if let Some(streams) = &self.streams` guards

## Performance Debugging

### Profiling Commands

See `CLAUDE.md` "Benchmarking & profiling" for the full command reference (`scripts/profile.sh`, `scripts/bench.sh`, etc.).

### Key Metrics to Monitor

- **LZ77 time:** Should be ~5-50ms for 256KB (CPU ~5ms, GPU 50-100ms)
- **Demux overhead:** Usually <2ms (negligible)
- **Entropy coding time:** Depends on data entropy; Huffman ~50-200ms for 256KB
- **GPU buffer overhead:** 35% of LZ77 time before ring-buffering; eliminated after
- **Ring buffer backpressure:** If > 0 wait cycles, GPU is bottleneck (increase batch size or reduce block size)

## Related Documentation

- **research-log.md** — Historical evolution of pipeline architecture
- **lz77-gpu.md** — LZ77 kernel details
- **gpu-batching.md** — Batching strategies and memory management
- **ARCHITECTURE.md** — High-level design, cost model notation
- **CLAUDE.md** — Build and profiling commands
