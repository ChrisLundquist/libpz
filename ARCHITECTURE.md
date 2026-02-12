# Architecture & GPU Notes

Detailed implementation notes, benchmarks, and roadmap for libpz.
For day-to-day development instructions, see `CLAUDE.md`.

## Completed milestones (11/12)
- **Algorithms:** LZ77 (brute, hashchain, lazy, parallel), Huffman, BWT (SA-IS), MTF, RLE, FSE, rANS
- **Pipelines:** Deflate (LZ77+Huffman), Bw (BWT+MTF+RLE+FSE), Lzr (LZ77+rANS), Lzf (LZ77+FSE) — Deflate, Lzr, and Lzf use multi-stream entropy coding for ~16-18% better compression
- **Auto-selection:** Heuristic (`select_pipeline`) and trial-based (`select_pipeline_trial`) pipeline selection using data analysis (entropy, match density, run ratio, autocorrelation)
- **Data analysis:** `src/analysis.rs` — statistical profiling (Shannon entropy, autocorrelation, run ratio, match density, distribution shape) with sampling support
- **Optimal parsing:** GPU top-K match table → CPU backward DP (4-6% better compression)
- **Multi-threading:** Block-parallel and pipeline-parallel via V2 container format; within-block parallel LZ77 match finding (`compress_lazy_parallel`)
- **GPU kernels:** LZ77 hash-table (fast), LZ77 batch/per-position (legacy), LZ77 top-K, BWT radix sort + parallel rank assignment, Huffman encode (two-pass with Blelloch prefix sum), GPU Deflate chaining (LZ77→Huffman on device)
- **Tooling:** CLI (`pz` with `-a`/`--auto` and `--trial` flags), C FFI, Criterion benchmarks, CI (3 OS + OpenCL)

### Not started
- M5.3: Fuzz testing (`cargo-fuzz`)
- M12: Vulkan compute backend

### Partially complete
- **rANS SIMD decode paths** — N-way interleaved rANS decode in `src/simd.rs` (SSE2 4-way, AVX2 8-way). The scalar interleaved encoder/decoder is implemented; SIMD intrinsics for the hot decode loop are not yet wired.
- **rANS OpenCL kernels** — GPU rANS encode/decode kernels (~200 lines). The CPU implementation and pipeline integration are done; GPU kernels are not yet written.
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

1. **SIMD decode paths** (~150 lines in `src/simd.rs`): Wire SSE2 4-way and
   AVX2 8-way intrinsics into the interleaved decode loop. The scalar N-way
   interleave already handles the data layout; SIMD replaces the per-state
   multiply-shift with packed 32×32→64 multiplies (`_mm_mul_epu32` /
   `_mm256_mul_epu32`).

2. **OpenCL kernels** (~200 lines in `kernels/rans.cl`): GPU rANS encode and
   decode. The interleaved format maps directly to GPU work-items (one state
   per thread, 32–256 threads). Frequency tables fit in `__local` memory
   (2 KB for 256×u16 + 256×u16 cumulative).

3. **Reciprocal multiplication trick**: Replace `x / freq` and `x % freq` in
   the encode loop with precomputed `ceil(2^(32+shift) / freq)` reciprocals.
   This eliminates data-dependent divisions (critical for GPU throughput).
   Not yet implemented due to u32 overflow edge cases when `freq` is small;
   needs u64 intermediate arithmetic or per-frequency shift selection.

4. **Benchmark integration**: Add rANS and Lzr pipeline to
   `benches/throughput.rs` and `benches/stages.rs` for head-to-head comparison
   with Huffman and FSE.

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

## GPU stage chaining
The Deflate GPU path chains LZ77 → Huffman on the GPU with minimized transfers:
1. GPU: LZ77 hash-table kernel → download match array → CPU dedupe + serialize
2. GPU: upload LZ77 bytes once → `ByteHistogram` kernel → download only 256×u32 (1KB)
3. CPU: build Huffman tree from histogram, produce code LUT
4. GPU: Huffman encode (reusing LZ77 buffer) with Blelloch prefix sum
5. GPU: download final encoded bitstream

The `ByteHistogram` kernel eliminates the need to scan LZ77 data on CPU for frequency counting — only 1KB of histogram data is transferred instead of the full LZ77 stream.

This is activated automatically when `Backend::OpenCl` is selected and input ≥ `MIN_GPU_INPUT_SIZE`.

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

## Remaining GPU bottlenecks

1. **GPU BWT still slower than CPU SA-IS** — Radix sort improved 7-14x over bitonic
   sort, but CPU SA-IS (O(n)) remains faster at small/medium sizes. GPU catches up
   at 64KB+ but prefix-doubling's O(n log n) work is inherently more than SA-IS's O(n).

2. **No shared memory usage** — LZ77 hash kernel uses only global memory.
   Loading hash buckets into `__local` memory could help at larger sizes.

3. **Hash bucket overflow** — Fixed BUCKET_CAP=64 means highly repetitive data
   may miss good matches. Adaptive bucket sizing could help.

4. **Huffman WriteCodes atomic contention** — Per-bit atomic_or on the output
   buffer limits scaling. Chunk-based packing could reduce contention.

5. **LZ77 match array still downloaded for dedupe** — GPU match dedup is sequential
   and runs on CPU. Keeping serialized LZ77 bytes on GPU for histogram+Huffman
   is already done (ByteHistogram optimization), but the match download is unavoidable.

## Next steps

### Priority 0: rANS SIMD + GPU completion
Three pieces remain to complete the rANS integration (~350 lines total):

1. **SIMD decode paths** (~150 lines in `src/simd.rs`): SSE2 4-way and AVX2 8-way
   intrinsics for the interleaved rANS decode hot loop. The scalar N-way interleave
   is implemented; SIMD replaces per-state multiply-shift with packed `_mm_mul_epu32` /
   `_mm256_mul_epu32`. Expected 3-4x decode throughput improvement.

2. **OpenCL rANS kernels** (~200 lines in `kernels/rans_decode.cl`): GPU decode
   kernel mapping one rANS state per work-item. Frequency + cumulative tables in
   `__local` memory (2 KB). The interleaved format maps directly: each work-item
   independently decodes its symbol stream, then results are scattered to output
   positions. Encode kernel is lower priority (CPU encode + GPU decode is the
   typical asymmetric pattern).

3. **Reciprocal multiplication trick**: Replace `x / freq` and `x % freq` with
   precomputed `ceil(2^(32+shift) / freq)` reciprocals to eliminate data-dependent
   division in encode. Needs u64 intermediate arithmetic or per-frequency adaptive
   shift to handle overflow when `freq` is small. Critical for GPU encode throughput.

### Priority 1: Use local/shared memory in LZ77 hash kernel
- Load hash buckets into `__local` memory for faster repeated access
- Could improve GPU LZ77 performance at mid-range sizes (64KB-256KB)
- May lower the GPU crossover point from 256KB toward 64-128KB

### Priority 3: Chunk-based Huffman bit packing
- Replace per-bit `atomic_or` in WriteCodes with work-group-local packing
- Each work-group packs its chunk into local memory (no atomics within WG)
- Single copy from local → global per chunk
- Could 5-10x GPU Huffman throughput

### Priority 4: Fuzz testing
- Set up `cargo-fuzz` for round-trip correctness on all pipelines
- Target edge cases in LZ77, Huffman, BWT decode paths

### Priority 5: Auto-selection threshold tuning
- Run all 3 pipelines on Canterbury + Silesia corpora
- Measure actual compression ratios vs analysis metrics
- Tune heuristic decision tree thresholds empirically

### Priority 6: aarch64 NEON/SVE SIMD implementation
- Replace scalar stubs in `src/simd.rs` with actual NEON intrinsics
- `compare_bytes`: `vceqq_u8` + `vmovn_u16` for 16-byte comparison
- `byte_frequencies`: 4-bank unrolled (NEON lacks efficient gather/scatter)
- `sum_u32`: `vld1q_u32` + `vaddq_u32` + `vaddvq_u32`
- SVE for ARMv8.2+ (variable-length vectors, predicated operations)
- Requires aarch64 hardware for benchmarking
