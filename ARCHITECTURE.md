# Architecture & GPU Notes

Detailed implementation notes, benchmarks, and roadmap for libpz.
For day-to-day development instructions, see `CLAUDE.md`.

## Completed milestones (12/12)
- **Algorithms:** LZ77 (brute, hashchain, lazy, parallel), LzSeq (code+extra-bits, repeat offsets, 128KB window), Huffman, BWT (SA-IS), MTF, RLE, FSE, rANS
- **Pipelines:** Deflate (LZ77+Huffman), Bw (BWT+MTF+RLE+FSE), Lzr (LZ77+rANS), Lzf (LZ77+FSE), LzSeqR (LzSeq+rANS), LzSeqH (LzSeq+Huffman), SortLz (sort-LZ77+FSE) — Deflate, Lzr, and Lzf use multi-stream entropy coding for ~16-18% better compression; LzSeqR/LzSeqH use zstd-style code+extra-bits encoding with 6-stream demux; SortLz uses sort-based match finding (GPU-accelerated)
- **Auto-selection:** Heuristic (`select_pipeline`) and trial-based (`select_pipeline_trial`) pipeline selection using data analysis (entropy, match density, run ratio, autocorrelation); LzSeqR included in trial candidates
- **Data analysis:** `src/analysis.rs` — statistical profiling (Shannon entropy, autocorrelation, run ratio, match density, distribution shape) with sampling support
- **Optimal parsing:** GPU top-K match table → CPU backward DP (4-6% better compression)
- **Multi-threading:** Block-parallel and pipeline-parallel via V2 container format; within-block parallel LZ77 match finding (`compress_lazy_parallel`)
- **SortLZ:** Sort-based match finder — standalone pipeline (ID 10) and pluggable `MatchFinder::SortLz` for Deflate/Lzr/Lzf/LzSeqR/LzSeqH; GPU radix sort batched (single submit); adaptive `select_match_finder()` heuristic; u64-optimized `extend_match`; 39.6% ratio (beats Deflate 43.4%)
- **GPU kernels:** LZ77 hash-table (fast), LZ77 batch/per-position (legacy), LZ77 top-K, BWT radix sort + parallel rank assignment, SortLZ radix sort + match verification, Huffman encode (two-pass with Blelloch prefix sum), GPU Deflate chaining (LZ77→Huffman on device)
- **Tooling:** CLI (`pz` with `-a`/`--auto` and `--trial` flags), C FFI, Criterion benchmarks, CI (3 OS)
- **Fuzz testing (M5.3):** `cargo-fuzz` infrastructure with 12 targets covering all algorithms and pipelines (roundtrip + crash resistance)

### Partially complete
- **rANS SIMD decode paths** — N-way interleaved rANS decode in `src/simd.rs` (SSE2 4-way, AVX2 8-way). The scalar interleaved encoder/decoder is implemented; SIMD intrinsics for the hot decode loop are not yet wired.
- **rANS reciprocal multiplication** — Replace division in the encode loop with precomputed reciprocal multiply-shift for GPU/SIMD (avoids data-dependent division). Documented as future optimization due to u32 overflow edge cases with small frequencies.

## BWT implementation
- **CPU:** Uses SA-IS (Suffix Array by Induced Sorting) — O(n) linear time via doubled-text-with-sentinel strategy.
- **GPU:** Uses LSB-first 8-bit radix sort with prefix-doubling for suffix array construction. Replaced earlier bitonic sort (PR #21). Features adaptive key width (skip zero-digit radix passes) and event chain batching (one host sync per doubling step). Rank assignment runs on GPU via Blelloch prefix sum + scatter. Still slower than CPU SA-IS at all sizes but dramatically improved from bitonic sort (7-14x faster). The GPU uses circular comparison `(sa[i]+k) % n` vs CPU SA-IS's doubled-text approach — both produce valid BWTs that round-trip correctly.

## Multi-stream Deflate

The Deflate, Lzr, and Lzf pipelines use **multi-stream entropy coding** to improve
compression ratio by separating LZ77 output into independent byte streams with
tighter symbol distributions. Instead of feeding one mixed stream to the entropy
coder, the encoder deinterleaves tokens into three streams:

| Stream | Contents | Why it helps |
|--------|----------|-------------|
| **Offsets** | High bytes of match offsets (offset >> 8) | Offsets cluster in a narrow range; dedicated Huffman/RC table exploits this |
| **Lengths** | Match lengths (capped to u8) | Length distribution is highly skewed (short matches dominate) |
| **Literals** | Literal bytes + low offset bytes + next bytes | Natural-language / binary byte distribution |

Each stream gets its own Huffman tree (Deflate), FSE table (Lzf), or rANS context (Lzr),
yielding lower per-stream entropy than a single combined stream.

### Encoding format

Multi-stream data is stored with a `0x02` stream-format flag in the container
header, followed by three length-prefixed compressed sub-streams:

```
[stream_format: u8 = 0x02]
[offsets_len: u32 LE] [offsets compressed data...]
[lengths_len: u32 LE] [lengths compressed data...]
[literals_len: u32 LE] [literals compressed data...]
```

The decoder reads the flag, decompresses each sub-stream independently, then
reinterleaves them back into the original LZ77 token sequence. Single-stream
format (`0x01`) is used as fallback for small inputs (< 256 bytes) or when
multi-stream produces larger output.

### Benchmark results

Comparison on Canterbury + Large corpus (14 files, 13.3 MB total), averaged over
3 iterations. "Before" = single-stream, "After" = multi-stream:

**Compression (size and throughput):**

| Pipeline | Before (bytes) | After (bytes) | Size delta | Throughput delta |
|----------|---------------|--------------|------------|-----------------|
| Deflate  | 6,319,168     | 5,301,184    | **-16.1%** | +1.6% faster    |
| Lzf      | 6,199,044     | 5,107,601    | **-17.6%** | +2.8% faster    |

**Decompression throughput:**

| Pipeline | Throughput delta |
|----------|-----------------|
| Deflate  | **+8.4%** faster |
| Lzf      | **+2.4%** faster |

Multi-stream is a pure win: better compression **and** faster speed. The largest
gains are on big files (E.coli: -21% size, +11% decode throughput; bible.txt:
-14% size, +16% decode throughput). Small files (< 4 KB) may see slight
expansion due to the overhead of three separate stream headers — the encoder
automatically falls back to single-stream when multi-stream would be larger.

## rANS entropy coder

`src/rans.rs` implements **range ANS (rANS)**, a streaming entropy coder that
uses multiply-shift arithmetic instead of table lookups (FSE/tANS) or bit-level
tree walks (Huffman). rANS approaches Shannon entropy like arithmetic/range
coding but with a simpler, more parallelizable decode hot path.

### Comparison with other entropy coders

| Property          | Huffman | FSE (tANS) | rANS       |
|-------------------|---------|------------|------------|
| Decode operation  | Bit-level tree walk | Table lookup | Multiply + lookup |
| I/O granularity   | Bits    | Bits       | 16-bit words |
| Branch predict    | Poor    | Good       | Good       |
| State independence| N/A     | Awkward    | Interleave N states |
| GPU shared memory | Large trees | Large tables | Small freq tables |

### Variants

- **Single-stream** (`encode` / `decode`): Reference scalar implementation.
  32-bit state in `[2^16, 2^32)`, 16-bit word I/O.
- **Interleaved N-way** (`encode_interleaved` / `decode_interleaved`): N
  independent rANS states with round-robin symbol assignment. Default N=4
  (maps to SSE2 lanes). All N decode chains run in parallel with zero data
  dependencies between them.

### Encoding format

**Single-stream:**
```
[scale_bits: u8] [freq_table: 256 × u16 LE] [final_state: u32 LE]
[num_words: u32 LE] [words: num_words × u16 LE]
```

**Interleaved N-way:**
```
[scale_bits: u8] [freq_table: 256 × u16 LE] [num_states: u8]
[final_states: N × u32 LE] [num_words: N × u32 LE]
[stream_0_words] [stream_1_words] ... [stream_N-1_words]
```

Header overhead is 521 bytes (1 + 512 + 4 + 4) for single-stream, making rANS
most effective for inputs larger than ~1 KB.

### Frequency normalization

Frequencies are normalized to sum to `1 << scale_bits` (default 12 bits = 4096).
Every symbol present in the input gets at least frequency 1. Excess is trimmed
from the most-frequent symbol; deficit is added to it. The normalization code
is shared conceptually with `src/fse.rs` (both operate on power-of-2 tables).

### Pipeline: Lzr (LZ77 + rANS)

`Pipeline::Lzr` (ID 3) reuses the existing multi-stream LZ77 architecture
(offsets, lengths, literals) with rANS as the entropy coder instead of Huffman
(Deflate) or FSE (Lzf). It participates in auto-selection via
`select_pipeline_trial()`.

### Forward TODOs

See `docs/exec-plans/tech-debt-tracker.md` for rANS SIMD decode and reciprocal multiplication work items. Benchmark integration (rANS/Lzr in `benches/throughput.rs` and `benches/stages.rs`) is also pending.

## SIMD acceleration
`src/simd.rs` provides runtime-dispatched SIMD for CPU hot paths:
- **Byte frequency counting** — 4-bank histogramming with AVX2 merge, integrated into `FrequencyTable::count()`
- **LZ77 match comparison** — SSE2 (16 bytes/cycle) or AVX2 (32 bytes/cycle) `compare_bytes`, integrated into `HashChainFinder::find_match()` and `find_top_k()`
- **u32 array summation** — widened u64 accumulator lanes for overflow-safe SIMD sum

| Architecture | Baseline | Extended | Status |
|-------------|----------|----------|--------|
| x86_64      | SSE2     | AVX2     | Implemented + integrated |
| aarch64     | NEON     | SVE      | Stubs (dispatch to scalar) |

Runtime detection via `Dispatcher::new()` caches the best ISA level at first call. All SIMD implementations are verified against scalar reference in tests.

## SortLZ: Sort-Based Match Finding

SortLZ is a deterministic, GPU-friendly LZ77 match finder. It replaces hash-chain
match finding with radix sort of (hash, position) pairs followed by adjacent-pair
match verification. Zero atomics, fully deterministic — ideal for GPU execution.

### Two modes of operation

| Mode | Description | Wire format |
|------|-------------|-------------|
| **`Pipeline::SortLz` (ID 10)** | Standalone pipeline with its own wire format | SortLZ-specific (see below) |
| **`MatchFinder::SortLz`** | Pluggable match finder for other pipelines | Host pipeline's format |

When used as a `MatchFinder`, SortLZ is transparent to the wire format — the
output is 100% compatible with the host pipeline (Deflate, Lzr, Lzf, LzSeqR,
LzSeqH). The consumer and decompressor see no difference.

### Pipeline::SortLz wire format (per block)

```
[num_tokens: u32 LE]       total token count (literals + matches)
[num_literals: u32 LE]     literal count
[flags_len: u32 LE]        ceil(num_tokens / 8)
[flags: flags_len bytes]   bitfield (1 = literal, 0 = match, MSB-first)
[fse_lit_len: u32 LE]      [fse_literals: ...]   FSE-encoded literal bytes
[fse_off_len: u32 LE]      [fse_offsets: ...]    FSE-encoded u16 LE offsets
[fse_len_len: u32 LE]      [fse_lengths: ...]    FSE-encoded u16 LE lengths
```

This is NOT wire-compatible with any other pipeline. It uses FSE entropy coding
on three raw byte streams (literals, offsets as u16 LE, lengths as u16 LE),
with a bitfield flag stream to interleave them during decompression.

### Algorithm

1. **Hash**: Compute 4-byte window hashes (u32 from LE bytes, no collisions for 4-byte matches)
2. **Radix sort**: 4-pass 8-bit LSB radix sort on (hash, position) pairs
3. **Verify**: Adjacent same-hash entries → extend match byte-by-byte (u64 chunk comparison)
4. **Select**: Best match per position (longest wins, max_candidates=8 per sorted entry)
5. **Parse**: Greedy or lazy token emission

### GPU implementation (`src/webgpu/sortlz.rs`)

Uses GPU radix sort (same kernels as BWT) + GPU match verification:
- 4-pass radix sort batched into single command encoder (1 submit, not 16+)
- `encoder.clear_buffer()` for histogram zeroing (no CPU↔GPU sync)
- Separate submit for match verification (needs sort results)
- 10.6x faster than CPU SortLZ at 4MB (89 vs 8.4 MB/s)

### Adaptive match finder selection

`select_match_finder()` in `src/pipeline/mod.rs` chooses SortLz when:
- GPU available and input ≥ MIN_GPU_INPUT_SIZE
- High match density (>0.3) and moderate entropy (<6.5 bits/byte)
- Large input (≥64KB) with match density >0.2 and low entropy (<5.5)

### Performance (AMD RX 9070 XT / RDNA4)

| Size | CPU hashchain | CPU SortLZ | GPU SortLZ | GPU vs CPU SortLZ |
|------|--------------|-----------|-----------|-------------------|
| 8KB  | 244 MB/s     | 85 MB/s   | 4 MB/s    | GPU overhead |
| 64KB | 140 MB/s     | 44 MB/s   | 31 MB/s   | 0.7x |
| 256KB| 131 MB/s     | 31 MB/s   | 53 MB/s   | **1.7x faster** |
| 4MB  | 142 MB/s     | 8 MB/s    | 89 MB/s   | **10.6x faster** |

SortLZ compression ratio: **39.6%** (vs hashchain+Deflate 43.4%, BWT 32.7%).

## GPU stage chaining
The Deflate GPU path chains LZ77 → Huffman on the GPU with minimized transfers:
1. GPU: LZ77 hash-table kernel → download match array → CPU dedupe + serialize
2. GPU: upload LZ77 bytes once → `ByteHistogram` kernel → download only 256×u32 (1KB)
3. CPU: build Huffman tree from histogram, produce code LUT
4. GPU: Huffman encode (reusing LZ77 buffer) with Blelloch prefix sum
5. GPU: download final encoded bitstream

The `ByteHistogram` kernel eliminates the need to scan LZ77 data on CPU for frequency counting — only 1KB of histogram data is transferred instead of the full LZ77 stream.

This is activated automatically when a GPU backend is selected and input ≥ `MIN_GPU_INPUT_SIZE`.

## Parallel LZ77
`compress_lazy_parallel(input, num_threads)` pre-computes matches in parallel (each thread builds its own hash chain), then serializes sequentially with lazy evaluation. Thresholds:
- `MIN_PARALLEL_SIZE = 256KB` — below this, single-threaded is faster
- `MIN_SEGMENT_SIZE = 128KB` — caps thread count to amortize hash chain warmup

## GPU benchmark results (AMD gfx1201 / RDNA3)

### LZ77 GPU

| Size | CPU hashchain | CPU lazy | GPU hash | GPU vs CPU hashchain |
|------|--------------|----------|----------|---------------------|
| 1KB  | 14µs (71 MiB/s) | 6µs (164 MiB/s) | 1.1ms (1 MiB/s) | 65x slower |
| 10KB | 57µs (171 MiB/s) | 42µs (231 MiB/s) | 1.4ms (7 MiB/s) | 20x slower |
| 64KB | 1.3ms (48 MiB/s) | 611µs (102 MiB/s) | 1.7ms (36 MiB/s) | 1.3x slower |
| 256KB | 6.2ms (40 MiB/s) | 2.6ms (97 MiB/s) | 3.4ms (74 MiB/s) | **2x faster** |
| 1MB | 20ms (50 MiB/s) | 16.7ms (60 MiB/s) | 9.0ms (111 MiB/s) | **2x faster** |

### Huffman GPU (encode only)

| Size | CPU | GPU + CPU scan | GPU + GPU scan | Best GPU vs CPU |
|------|-----|----------------|----------------|-----------------|
| 10KB | 23µs (418 MiB/s) | 312µs (31 MiB/s) | 518µs (19 MiB/s) | CPU 13x faster |
| 64KB | 432µs (145 MiB/s) | 926µs (68 MiB/s) | 1.45ms (43 MiB/s) | CPU 2x faster |
| 256KB | 1.85ms (135 MiB/s) | 999µs (250 MiB/s) | **543µs (460 MiB/s)** | **GPU 3.4x faster** |

GPU Huffman with Blelloch prefix sum crosses over ~128KB. At 256KB the GPU scan path is 3.4x faster than CPU.

### Deflate chained (GPU LZ77 → GPU Huffman)

| Size | CPU 1-thread | GPU chained | Speedup |
|------|-------------|-------------|---------|
| 64KB | 1.63ms (38 MiB/s) | 3.01ms (21 MiB/s) | CPU 1.8x faster |
| 256KB | 6.06ms (41 MiB/s) | 4.93ms (51 MiB/s) | **GPU 1.2x faster** |
| 1MB | 23.5ms (43 MiB/s) | 18.3ms (55 MiB/s) | **GPU 1.3x faster** |

### BWT GPU (radix sort)

| Size | GPU radix | Throughput | Old bitonic | Speedup vs bitonic |
|------|-----------|------------|-------------|--------------------|
| 1KB  | 3.4ms | 295 KiB/s | 23ms | **6.8x** |
| 10KB | 5.9ms | 1.6 MiB/s | 42ms | **7.1x** |
| 64KB | 4.1ms | 15.3 MiB/s | 56ms | **13.7x** |
| 256KB | 11.6ms | 21.6 MiB/s | — | — |
| 4MB | 333ms | 12.0 MiB/s | — | — |
| 16MB | 1.73s | 9.2 MiB/s | — | — |

GPU BWT radix sort is 7-14x faster than the old bitonic sort. Still slower than CPU SA-IS at small sizes but becoming competitive at 64KB+ (CPU SA-IS ~1ms at 64KB vs GPU 4.1ms). The gap narrows at larger sizes where GPU parallelism helps more.

## GPU/CPU strategy (settled)

The optimal split for libpz is **GPU for LZ77 match-finding, CPU for entropy coding**,
overlapped via the unified scheduler with ring-buffered streaming.

**Why GPU wins on LZ77:** Match-finding is embarrassingly parallel — each position's
search is independent. The cooperative-stitch kernel does 1,788 probes/position and
is 2x faster than CPU at 256KB+. Ring-buffered batching (`find_matches_batched`)
adds +7-17% throughput by amortizing buffer allocation and overlapping GPU compute
with CPU readback.

**Why CPU wins on entropy:** rANS/FSE/Huffman are serial state machines — each
symbol depends on the previous state. GPU entropy has been tried extensively
(500+ iterations: single-stream, independent blocks, Recoil checkpoints, batched
cross-block) and is 0.77x CPU on encode, 0.54x on decode. The serial dependency
limits GPU to ~300 useful threads when saturation needs ~8K-16K. PCIe transfer
overhead dominates at typical block sizes (128KB-256KB).

**Architecture:** The unified scheduler dispatches LZ77 to GPU and entropy to CPU
workers in parallel. While CPU thread N entropy-encodes block K, the GPU is already
match-finding block K+1. The `GPU_ENTROPY_THRESHOLD` (256KB) is deliberately set
above `DEFAULT_GPU_BLOCK_SIZE` (128KB) to prevent routing entropy to the slower
GPU path.

See `docs/design-docs/gpu-strategy.md` for full analysis and `CLAUDE.md` "Known
dead ends" for the complete list of GPU optimization attempts that failed.

## Remaining GPU bottlenecks

1. **GPU BWT still slower than CPU SA-IS** — Radix sort improved 7-14x over bitonic
   sort, but CPU SA-IS (O(n)) remains faster at small/medium sizes. GPU catches up
   at 64KB+ but prefix-doubling's O(n log n) work is inherently more than SA-IS's O(n).

2. **Hash bucket overflow** — Fixed BUCKET_CAP=64 means highly repetitive data
   may miss good matches. Adaptive bucket sizing could help.

3. **LZ77 match array still downloaded for dedupe** — GPU match dedup is sequential
   and runs on CPU. Keeping serialized LZ77 bytes on GPU for histogram+Huffman
   is already done (ByteHistogram optimization), but the match download is unavoidable.

## Next steps

### Priority 0: Close the gzip compression ratio gap

LzSeqR is our best pipeline at 35.1% vs gzip's 28.6% (6.5pp gap). The gap is
encoding efficiency, not match quality (see `CLAUDE.md` "Known dead ends"). The
format is pre-release so all changes are free. Key opportunities:

- **Zstd-style literal-run sequences** — replace per-token flags with
  `(literal_run_length, offset, match_length)` tuples, eliminating the flags
  stream entirely. Highest ceiling.
- **Larger repeat offset cache** (4→8) — each additional repeat saves all extra
  bits for that match.
- **Entropy-code the extra bits** — `offset_extra` and `length_extra` currently
  bypass rANS; if values are skewed, 5-15% savings.
- **Sparse frequency tables** — 512 bytes per rANS stream → ~61 bytes for
  narrow-alphabet streams. Saves ~1.3KB/block.

### Priority 1: LzSeq-specific optimal parser

The optimal parser currently uses LZ77's `match_cost` approximation. A dedicated
LzSeq optimal parser that tracks repeat offset state through the backward DP would
find more repeat matches, directly improving ratio.

### Priority 2: aarch64 NEON/SVE SIMD implementation
- Replace scalar stubs in `src/simd.rs` with actual NEON intrinsics
- `compare_bytes`: `vceqq_u8` + `vmovn_u16` for 16-byte comparison
- `byte_frequencies`: 4-bank unrolled (NEON lacks efficient gather/scatter)
- `sum_u32`: `vld1q_u32` + `vaddq_u32` + `vaddvq_u32`
- SVE for ARMv8.2+ (variable-length vectors, predicated operations)
- Requires aarch64 hardware for benchmarking
