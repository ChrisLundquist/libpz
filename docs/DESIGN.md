# Design Principles

**Last Updated:** 2026-02-14
**Owner:** Engineering team

## Purpose

This document consolidates libpz's technical design principles. For agent-first operating principles, see `design-docs/core-beliefs.md`.

## Architectural Principles

### 1. Composition Over Configuration

**Algorithms are standalone, pipelines compose them.**

Each algorithm (`bwt`, `huffman`, `lz77`, `fse`, `rans`, etc.) is:
- Independently testable
- Independently benchmarkable
- Usable standalone or in pipelines
- Not tied to a specific compression format

Pipelines (`deflate`, `bw`, `lzr`, `lzf`) combine algorithms via the demuxer pattern:
```rust
// Pipeline = sequence of stages
LZ77 → demux into streams → Huffman encode → merge streams
```

**Why:** Flexibility. New pipelines reuse existing algorithms. New algorithms enhance all pipelines.

**Example:** The same LZ77 implementation works in Deflate (LZ77+Huffman), Lzf (LZ77+FSE), and Lzr (LZ77+rANS) pipelines.

### 2. GPU-Friendly Design Patterns

**Prefer table-driven, branchless algorithms with data parallelism.**

GPU acceleration is viable when algorithms exhibit:
- **Data parallelism** - Independent work items (e.g., per-byte hash lookups)
- **Predictable memory access** - Sequential or strided patterns, not random
- **Minimal branching** - Table lookups instead of conditionals
- **No data-dependent division** - Use multiply-shift or table lookups

**Example:** Huffman encoding uses two-pass design:
1. Count frequencies (parallel reduction)
2. Encode via table lookup (parallel map)

Avoid: Recursive algorithms, heavy branching, pointer chasing.

**Trade-offs:**
- GPU is slower than CPU for small inputs (<128KB) due to overhead
- GPU requires batch processing to amortize kernel launch costs
- GPU memory is limited (track usage with `scripts/gpu-meminfo.sh`)

### 3. Correctness First, Then Performance

**The validation hierarchy:**
1. **Unit tests** - Algorithm correctness in isolation
2. **Round-trip tests** - encode(decode(data)) == data
3. **Cross-decompression tests** - GPU encode → CPU decode and vice versa
4. **Validation corpus** - Canterbury, enwik8, large corpus
5. **Benchmarks** - Throughput and compression ratio vs gzip

Never skip steps 1-4 to optimize step 5.

**Why:** Compression bugs can be silent (produces valid output with worse compression) or catastrophic (data corruption). Comprehensive validation catches both.

### 4. Zero-Copy Where Possible

**Provide `_to_buf` variants for caller-allocated output buffers.**

Every `encode()` / `decode()` function should have a `_to_buf` variant:
```rust
pub fn encode(input: &[u8]) -> PzResult<Vec<u8>>;
pub fn encode_to_buf(input: &[u8], output: &mut [u8]) -> PzResult<usize>;
```

**Why:** Enables caller-managed allocation, reduces allocations in hot paths, supports pre-allocated pools.

### 5. Graceful GPU Fallback

**GPU features are optional, tests skip if no device available.**

- GPU is enabled by default via `webgpu` feature
- Tests check `is_gpu_available()` and skip gracefully
- `--no-default-features` builds pure CPU version
- No code should assume GPU is present

**Why:** CI runs on machines without GPUs. Users may disable GPU for portability or minimal builds.

### 6. Error Handling Philosophy

**Use `PzError` variants, never panic on bad input.**

Error types:
- `InvalidInput` - Malformed compressed data, invalid parameters
- `BufferTooSmall` - Output buffer insufficient (include required vs provided)
- `Unsupported` - Valid input but unsupported feature (e.g., unknown pipeline ID)
- `InternalError` - Should never happen (indicates library bug)

Never panic on user-provided data. Invalid input should return `Err(PzError::InvalidInput)`.

**Example:**
```rust
if output.len() < max_compressed_size {
    return Err(PzError::BufferTooSmall {
        required: max_compressed_size,
        provided: output.len(),
    });
}
```

## Implementation Patterns

### Public API Naming

- **Compression:** `encode()` (not compress, pack, deflate)
- **Decompression:** `decode()` (not decompress, unpack, inflate)
- **Caller-allocated:** `encode_to_buf()`, `decode_to_buf()`

**Why:** Consistent naming across all algorithms. "Encode/decode" emphasizes transformation, not just compression.

### Module Structure

One file per algorithm:
- `src/bwt.rs` - Burrows-Wheeler Transform
- `src/huffman.rs` - Huffman coding
- `src/lz77.rs` - LZ77 compression
- etc.

Tests in `#[cfg(test)] mod tests` at bottom of file.

**Why:** Single-file modules are easier to understand and navigate. Tests colocated with implementation.

### GPU Module Organization

- `src/webgpu/mod.rs` - Device initialization, shared utilities
- `src/webgpu/lz77.rs` - LZ77 GPU kernels
- `src/webgpu/huffman.rs` - Huffman GPU kernels
- `kernels/*.wgsl` - WGSL shader source

GPU modules mirror CPU module structure.

### Pipeline Composition

Pipeline architecture:
- `src/pipeline/mod.rs` - Container format (V2 header), compress/decompress entry points
- `src/pipeline/blocks.rs` - Single-block compress/decompress via demuxer dispatch
- `src/pipeline/demux.rs` - `StreamDemuxer` trait, `LzDemuxer` enum, `demuxer_for_pipeline()`
- `src/pipeline/stages.rs` - Per-stage functions (`stage_lz77`, `stage_huffman_encode_gpu`, etc.)
- `src/pipeline/parallel.rs` - Block-parallel, pipeline-parallel, GPU-batched multi-block

**Adding a new LZ-based pipeline:**
1. Add enum variant to `Pipeline` in `mod.rs`
2. Add entry to `demuxer_for_pipeline()` in `demux.rs`
3. Add entropy encode/decode dispatch in `blocks.rs`

No new block function needed - demuxer pattern handles it.

## Debugging & Development Patterns

### Tracing Data Flow Through Pipelines

To understand how a pipeline works, trace a single block from `compress_block()` in `pipeline/blocks.rs` through each stage:
1. The `StageBlock.streams` field is the key handoff point
2. It's `None` before demux, `Some(Vec<Vec<u8>>)` after demux
3. Entropy encoders consume `streams`, not `data`

**When GPU and CPU paths diverge:** The bug is almost always in the demux (stream splitting), not the entropy coder. Check:
- Stream count (should be same for GPU/CPU)
- Byte ordering within streams
- Stream length consistency

**Adding a new LZ-based pipeline:** Only requires:
1. Entry in `demuxer_for_pipeline()` in `pipeline/demux.rs`
2. Entropy encode/decode dispatch in `pipeline/blocks.rs`
No new block function needed - demuxer pattern handles it.

### Understanding GPU Resource Usage

**Never trust memory estimates in comments or plans.** The only source of truth is actual `Buffer::create()` / `create_buffer()` calls in `src/webgpu/*.rs`:
- Buffer sizes are computed at runtime from input length
- Account for staging buffers and padding/alignment
- Use `scripts/gpu-meminfo.sh` to analyze actual allocations

**GPU batched vs pipeline-parallel paths:** Both in `pipeline/parallel.rs`:
- `compress_parallel_gpu_batched` - Ring buffer batching for GPU
- `compress_pipeline_parallel` - Pipeline-parallel execution
They solve the same problem differently. Understand both before modifying either.

### Feature Flags and Build Configurations

- **Default features:** WebGPU enabled by default via wgpu (Vulkan/Metal/DX12)
- **`--no-default-features`:** Pure CPU build (disables WebGPU)
- **`--features webgpu`:** Explicitly enable WebGPU (same as default)
- Tests gracefully skip if no GPU device available (never fail due to missing GPU)

**Common mistake:** Using `--no-default-features` unintentionally disables WebGPU. If GPU tests/benchmarks aren't running, check your feature flags.

### Multi-Stream Format Details

LZ-based pipelines demux into independent streams:
- **LZ77 (Deflate/Lzf/Lzr):** 3 streams (offsets high byte, lengths, literals+offsets low)
- **LZSS:** 4 streams
- **LZ78:** 1 stream

Per-block multistream container header: `[num_streams: u8][pre_entropy_len: u32][meta_len: u16][meta]`

**Warning:** Multi-stream format changes are subtle. Don't modify without understanding round-trip implications for all pipelines.

## Performance Considerations

### GPU Break-Even Points

GPU is faster than CPU starting at:
- **LZ77:** ~256KB blocks
- **Huffman:** ~128KB blocks
- **BWT:** ~512KB blocks (radix sort is still slower than CPU SA-IS)

Below these thresholds, kernel launch overhead dominates.

**Implication:** Don't optimize GPU paths for small blocks. Focus on large-block batched workloads.

### Multi-Stream Entropy Coding

LZ-based pipelines (Deflate, Lzf, Lzr) use multi-stream encoding:
- **Stream 0:** Match offsets (high byte)
- **Stream 1:** Match lengths
- **Stream 2:** Literals + offset low bytes

Each stream gets independent entropy coding, exploiting tighter symbol distributions.

**Results:** 16-18% better compression ratio, 2-8% faster decompression.

**Trade-off:** Overhead of 3 stream headers. Auto-fallback to single-stream for small blocks (<256 bytes).

### Memory Management

- **CPU:** Allocate per-block, reclaim immediately
- **GPU:** Ring buffer with backpressure (see `docs/design-docs/gpu-batching.md`)
- **Pipeline-parallel:** 2-3 blocks in flight (one per stage)

Use `scripts/gpu-meminfo.sh` to analyze GPU memory requirements.

## Testing Strategy

### Unit Tests

Each algorithm has:
- Empty input test
- Single-byte input test
- Small input tests (1-100 bytes)
- Large input tests (10KB - 1MB)
- Edge cases (all zeros, incompressible random data)

### Round-Trip Tests

Every encoder has a matching decoder with round-trip test:
```rust
let original = b"test data";
let encoded = encode(original)?;
let decoded = decode(&encoded)?;
assert_eq!(original, &decoded[..]);
```

### Cross-Decompression Tests

In `src/validation.rs`:
- GPU encode → CPU decode
- CPU encode → GPU decode

Ensures implementations produce identical output.

### Validation Corpus

Real-world data in `validation/`:
- Canterbury corpus (11 files, 2.8MB)
- Large corpus (3 files, 10.5MB)
- All pipelines tested on all files

Run via `cargo test validation`.

### Property Tests

Fuzz testing (M5.3, not yet implemented):
- Random inputs 0-1MB
- Verify round-trip property
- Check error handling on malformed compressed data

## Related Documents

- **CLAUDE.md** - Day-to-day development instructions
- **ARCHITECTURE.md** - Technical architecture, benchmarks, roadmap
- **design-docs/core-beliefs.md** - Agent-first operating principles
- **GOLDEN_PRINCIPLES.md** - Mechanically-enforced coding standards
- **design-docs/** - Detailed design docs (GPU memory model, pipeline architecture, etc.)
