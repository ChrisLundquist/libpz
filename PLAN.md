# libpz Project Plan

## Vision

libpz aims to accelerate lossless data compression on modern hardware by
offloading compute-intensive stages to GPUs and multi-core CPUs. Single-thread
CPU performance is plateauing while GPU/multi-core throughput continues to
scale. Compression algorithms have significant parallelism that is largely
untapped by existing tools (gzip, xz, bzip2 are all single-threaded by
default). pigz demonstrated that even naive block-parallel gzip compression
yields near-linear speedups. libpz goes further by accelerating the
inner loops themselves (match finding, BWT sorting, etc.) on GPUs.

## Prior Art

| Tool | Algorithms | Parallelism | Notes |
|------|-----------|-------------|-------|
| **gzip** | LZ77 + Huffman (DEFLATE) | Single-threaded | RFC 1951. 32KB window. The baseline. |
| **pigz** | DEFLATE | pthread block-parallel compress, single-thread decompress | Chunks input into independent blocks, compresses each on a thread. Near-linear speedup. Decompression stays single-threaded because block boundaries are unknown and LZ77 back-refs cross blocks. |
| **zstd** | LZ77-variant + Huffman + FSE (tANS) | Single-threaded (multi-thread via blocks) | Levels 1-19 use greedy/lazy/optimal parsing. Optimal parsing (levels 17+) uses price-based dynamic programming to select match chains minimizing total bit cost. FSE (Finite State Entropy) is faster than Huffman for decode. |
| **bzip2** | BWT + MTF + RLE + Huffman | Single-threaded (pbzip2 exists) | BWT is the expensive step. Suffix array construction is O(n) and parallelizable. CUDA/OpenCL BWT implementations exist in academic papers. |
| **xz/lzma** | LZ77 (large window) + range coder | Single-threaded (xz has --threads) | Optimal parsing via dynamic programming. Very high compression, slow. Threaded mode uses independent blocks. |
| **lz4** | LZ77-variant | Single-threaded | Optimized for decode speed, not ratio. Greedy match selection. Simple format, very fast. |
| **nvcomp** | LZ4, Snappy, DEFLATE, others | CUDA GPU | NVIDIA's GPU compression library. Demonstrates GPU viability for both compress and decompress. Closed source. |
| **GPU LZSS (academic)** | LZSS | CUDA | Papers by Ozsoy & Swany (2011), Shyni & Aruna (2015). GPU finds all matches in parallel, CPU selects optimal chain. Speedups of 2-10x for compression. |

### Key Insight from Prior Art: Optimal Parsing

The central problem libpz faces with GPU LZ77 is **match selection**. The GPU
can find all possible matches for every position in parallel, but we must then
choose which matches to actually use. This is the "optimal parsing" problem.

**Greedy parsing:** At each position, pick the longest match. Fast but
suboptimal -- a slightly shorter match might leave the next position with a
much better match.

**Lazy parsing (gzip levels 4-9):** Find a match, then check if the *next*
position has a longer match. If so, emit a literal and use the longer match.
Simple heuristic, moderate improvement.

**Optimal parsing (zstd level 17+, xz/lzma):** Model the encoding cost of
each possible match/literal at each position, then use dynamic programming
(backward pass) to find the minimum-cost path through the input. This is
expensive on CPU but the GPU has already done the hard part (finding all
matches), so the DP selection pass is relatively cheap.

**Price-based optimal parsing** works as follows:
1. GPU produces a match table: for each position i, a list of (offset, length) candidates
2. CPU runs backward DP: `cost[i] = min over all matches at i of (match_encoding_cost + cost[i + match.length + 1])`
3. The literal cost at each position is estimated from a frequency model (entropy)
4. Forward pass traces the optimal path, emitting the selected matches

This is exactly what libpz should do: **GPU finds matches, CPU selects the
minimum-entropy chain via DP**.

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │           libpz Public API           │
                    │  pz_compress() / pz_decompress()     │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │         Pipeline Orchestrator        │
                    │  Chains stages: LZ77 → Huffman, etc │
                    │  Selects best backend per stage      │
                    └──────────────┬──────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                     ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │  Reference (CPU) │ │   OpenCL GPU    │ │   pthread CPU   │
    │  Single-thread   │ │                 │ │   Multi-thread  │
    │  Always available │ │  If GPU present │ │  If cores > 1  │
    └─────────────────┘ └─────────────────┘ └─────────────────┘
              │                    │                     │
              ▼                    ▼                     ▼
    ┌─────────────────────────────────────────────────────────┐
    │              Algorithm Implementations                   │
    │  LZ77, Huffman, BWT, MTF, RLE, Arithmetic, FSE, etc     │
    └─────────────────────────────────────────────────────────┘
```

### Runtime Backend Selection

At startup, libpz probes available compute resources:
1. Query OpenCL platforms/devices (GPU, CPU accelerators)
2. Query Vulkan compute devices (future)
3. Count CPU cores (for pthread backends)
4. Fall back to reference single-threaded implementation

Each algorithm stage advertises which backends it supports. The orchestrator
picks the best available backend per stage, considering:
- Device availability
- Input size (GPU overhead not worth it for small inputs)
- Algorithm suitability (see parallelism notes below)

---

## Phase 1: Bug-Free Reference Implementations

Fix all bugs documented in `BUGS.md`, then complete the reference
implementations. These serve as the correctness oracle for all other backends.

### 1.1 Fix Existing Bugs
- Fix all 25 bugs in BUGS.md (priority queue, LZ77 bounds, Huffman encode, etc.)
- Ensure round-trip correctness: `decompress(compress(data)) == data` for all algorithms
- All tests pass, fuzz tests run clean under ASan

### 1.2–1.9 Reference Implementations — COMPLETE

All reference implementations completed in Rust:
- LZ77 (brute-force, hash-chain, lazy matching)
- Huffman (canonical codes, bit-packed encode/decode)
- Range Coder (adaptive model)
- BWT (prefix-doubling suffix array, inverse BWT)
- MTF, RLE, Frequency analysis
- Three compression pipelines: Deflate, Bw, Lza

---

## Phase 1B Status: Reference Implementation Complete

The Rust reference implementation of all core algorithms is complete.
This serves as the correctness oracle for future GPU and multi-threaded backends.

### Completed Modules

| Module | File | Description | Tests |
|--------|------|-------------|-------|
| **Priority Queue** | `pqueue.rs` | Min-heap for Huffman tree construction | Unit tests |
| **Frequency** | `frequency.rs` | Byte frequency counting and entropy calculation | Unit tests |
| **Huffman** | `huffman.rs` | Tree construction, canonical codes, bit-packed encode/decode | Unit tests |
| **LZ77** | `lz77.rs` | Brute-force, hash-chain (O(n)), and lazy matching | Unit tests |
| **RLE** | `rle.rs` | bzip2-style run-length encoding (4+count format) | Unit tests |
| **MTF** | `mtf.rs` | Move-to-Front transform | Unit tests |
| **BWT** | `bwt.rs` | Burrows-Wheeler Transform (prefix-doubling suffix array) | Unit tests |
| **Range Coder** | `rangecoder.rs` | Subbotin carryless range coder with adaptive model | Unit tests |
| **Pipeline** | `pipeline.rs` | Orchestrator with self-describing container format | Unit tests |
| **FFI** | `ffi.rs` | C-callable API (`pz_compress` / `pz_decompress`) | Integration tests |

### Compression Pipelines Implemented

| Pipeline | ID | Stages | Similar to |
|----------|----|--------|------------|
| `Deflate` | 0 | LZ77 → Multi-stream Huffman | gzip |
| `Bw` | 1 | BWT → MTF → RLE → RangeCoder | bzip2 |
| `Lza` | 2 | LZ77 → Multi-stream RangeCoder | lzma-like |

### Container Format

Self-describing binary format for compressed data:
```
[2B magic "PZ"] [1B version=1] [1B pipeline_id] [4B original_len LE] [pipeline metadata...] [compressed data...]
```
- Deflate pipeline includes Huffman frequency table in metadata (~1KB)
- Bw pipeline includes BWT primary_index (4 bytes) in metadata
- Decompression auto-detects pipeline from header
- Deflate and Lza use a `stream_format` byte: `0x01` = single-stream (legacy),
  `0x02` = multi-stream (offsets/lengths/literals split into 3 sub-streams)

### Modular Algorithm Composition

Each algorithm module is an independent building block with `encode`/`decode` pairs
that can be chained arbitrarily. This modular design enables:

1. **Mix-and-match experimentation**: Any transform/entropy coder combination can be
   tested (e.g., BWT→MTF→Huffman, LZ77→RangeCoder, BWT→RLE→Huffman)
2. **Algorithm comparison**: Multiple implementations of the same stage (e.g., three
   LZ77 match finders: brute-force, hash-chain, lazy) can be benchmarked head-to-head
3. **Future extensibility**: New algorithms (FSE/tANS, optimal LZ77 parsing, GPU
   backends) slot in as new modules without changing existing code
4. **Backend validation**: When GPU/pthread backends are added, they must produce
   output that round-trips identically through the same decode path

---

## Reference Implementation Validation Strategy

### Validation Test Infrastructure (`validation.rs`)

The validation test module provides comprehensive correctness testing across all
algorithm modules and their compositions. Tests are organized in layers:

#### 1. Per-Algorithm Round-Trip Tests
A `round_trip_test!` macro generates tests for every algorithm against a standard
set of test vectors:
- **all_zeros**: 1000 bytes of zeros (worst case for some, best for others)
- **uniform**: All 256 byte values (uniform distribution)
- **skewed**: Heavily biased toward zeros with rare other values
- **repeating_text**: Natural language with repetition
- **sawtooth**: Repeating 0..63 pattern
- **runs**: Alternating run lengths of different bytes
- **binary_pattern**: Pseudo-random binary data
- **single_byte**: Degenerate single-byte input
- **two_bytes**: Minimal non-trivial input

Algorithms tested: LZ77 (brute-force, hash-chain, lazy), Huffman, RangeCoder,
BWT, MTF, RLE — all must round-trip perfectly on every test vector.

#### 2. Cross-Module Composition Tests
Tests that chain multiple modules to validate they compose correctly:
- BWT → MTF
- BWT → MTF → RLE
- BWT → MTF → RLE → RangeCoder (full Bw chain)
- LZ77 → Huffman (Deflate chain)
- LZ77 → RangeCoder (Lza chain)
- MTF → RLE → Huffman (arbitrary composition)

#### 3. Pipeline Integration Tests
All three pipelines tested against all test vectors via `pipeline::compress`
and `pipeline::decompress`, validating the container format, metadata handling,
and end-to-end round-trip.

#### 4. Algorithmic Property Validation
Tests that verify expected mathematical/algorithmic properties:
- **Entropy bounds**: Huffman/RangeCoder output size ≤ Shannon entropy + overhead
- **BWT clustering**: BWT output has more same-byte adjacencies than input
- **MTF locality**: After BWT+MTF, >30% of output bytes are small (< 4)
- **RLE compression**: RLE compresses data with long runs
- **RangeCoder vs Huffman**: RangeCoder achieves better compression on skewed data
- **LZ77 match finding**: All three strategies find matches and decompress identically
- **Compression effectiveness**: Huffman beats uniform, RangeCoder beats Huffman on skewed

#### 5. Corpus Tests
Real-world file testing using standard compression benchmarks:

**Canterbury Corpus** (`samples/cantrbry/`):
- alice29.txt, asyoulik.txt, cp.html, fields.c, grammar.lsp, xargs.1
- Full round-trip through all three pipelines
- Individual algorithm tests on representative files (alice29.txt)

**Large Corpus** (`samples/large/`):
- bible.txt (4MB), E.coli genome (4.6MB), world192.txt (2.5MB)
- Tested on 64KB slices (full files too slow for BWT in test suite)

#### 6. Edge Cases
Adversarial and boundary-condition inputs:
- Two-byte input, alternating bytes, all 256 byte values
- Maximum RLE runs (259+ bytes), descending sequences
- Repeated short patterns, LZ77 window boundary patterns

### Future Validation Work

#### Sample File Test Infrastructure (Phase 1C)
Create a standalone test harness that:
1. Compresses each corpus file with all pipelines at all settings
2. Records compressed sizes in a results table
3. Verifies round-trip correctness
4. Compares compression ratios against known-good baselines (gzip, bzip2)
5. Generates a benchmark report (`BENCHMARK-REPORT.md`)

**Planned corpus expansion:**
- Silesia corpus (212MB, diverse file types)
- enwik8 (100MB Wikipedia dump)
- Larger Canterbury files (kennedy.xls, lcet10.txt, plrabn12.txt)
  currently skipped due to BWT O(n log² n) cost on large inputs

#### Cross-Implementation Validation (Phase 1C)
When C bug fixes are complete:
1. Compress with C reference, decompress with Rust — must match
2. Compress with Rust, decompress with C reference — must match
3. Byte-for-byte comparison of compressed output where formats match
4. Performance comparison (throughput MB/s, compression ratio)

#### Fuzz Testing (Phase 1C)
Set up continuous fuzz testing:
1. `cargo-fuzz` targets for each encode/decode function
2. Round-trip fuzz: `decode(encode(input)) == input` for all algorithms
3. Malformed input fuzz: decode functions must return `Err`, never panic
4. Pipeline fuzz: random bytes to `decompress()` must not panic
5. Run under ASan for memory safety validation

#### Regression Test Suite
Maintain a set of known-good compressed files:
1. For each corpus file × pipeline, store the compressed output
2. After any code change, verify compressed output is bit-identical
3. If intentional format changes occur, update baselines and document why
4. This catches accidental behavioral changes in the encoder

---

## Phase 2: Optimal LZ77 Match Selection ✅

This is the core research contribution of libpz. The GPU finds all matches;
the CPU selects the minimum-cost chain.

### 2.1 GPU Match Table Generation ✅
**Implemented** in `kernels/lz77_topk.cl` and `src/opencl.rs`:
- New kernel `EncodeTopK`: one work-item per position, K=4 candidates per position
- `lz77_candidate_t { u16 offset, u16 length }` — 4 bytes per candidate
- 32KB sliding window, brute-force scan with spot-check optimization
- `OpenClEngine::find_topk_matches()` returns `MatchTable` for CPU DP

CPU fallback: `build_match_table_cpu()` uses `HashChainFinder::find_top_k()`
to collect diverse candidates (distinct lengths, prefer smaller offsets).

### 2.2 Cost Model ✅
**Implemented** in `src/optimal.rs::CostModel`:

Token-based cost model reflecting the PZ format where every token (literal or
match) serializes to exactly 9 bytes (`offset:u32 + length:u32 + next:u8`):

- **Literal token cost:** `literal_overhead` (8 zero bytes ≈ 8 bits after entropy
  coding) + Shannon cost of `next` byte
- **Match token cost:** `match_overhead` (8 varied offset/length bytes ≈ 32 bits)
  + Shannon cost of `next` byte
- All costs in fixed-point (×256) to avoid float in DP inner loop
- Uses `FrequencyTable` for Shannon entropy estimates of byte values

The key insight is that both literals and matches are 9-byte tokens, but literals
have 8 zero bytes (very cheap after entropy coding) while matches have varied
offset/length bytes (more expensive). Matches amortize their overhead by covering
`length + 1` input bytes per token.

### 2.3 Backward DP Optimal Parse ✅
**Implemented** in `src/optimal.rs::optimal_parse()`:
```
cost[n] = 0
for i = n-1 down to 0:
    cost[i] = literal_cost(input[i]) + cost[i+1]
    for each candidate at position i:
        if i + length < n:
            mcost = match_overhead + literal_cost(next_byte) + cost[i+length+1]
            cost[i] = min(cost[i], mcost)
```
Accounts for the mandatory `next` byte in the Match format (each token
covers `length + 1` bytes). Forward trace recovers the optimal sequence.

Pipeline integration: `ParseStrategy::Optimal` in `CompressOptions`,
`-O`/`--optimal` CLI flag. Works with both Deflate and Lza pipelines,
single-threaded and multi-block parallel modes.

### 2.4 Benchmark Results

Optimal parsing vs greedy on Canterbury corpus (~2.8MB, via `cantrbry.tar.gz`):

| Pipeline | Greedy (bytes) | Optimal (bytes) | Improvement |
|----------|---------------|-----------------|-------------|
| Deflate  | 1,290,991     | 1,234,738       | **-4.4%**   |
| Lza      | 1,207,948     | 1,134,960       | **-6.0%**   |

Speed trade-off: optimal parsing is ~4.5–7× slower than greedy (expected, due to
DP pass and match table construction). On highly repetitive data, improvement is
minimal since greedy already finds long matches.

### 2.5 Entropy Feedback (Future)
After an initial parse, update the frequency model with the actual
match/literal distribution, then re-run the DP with updated costs. Converges
in 2-3 iterations (diminishing returns after that). This is similar to
zstd's "optimal parsing with price updates".

---

## Phase 3: OpenCL Backend — PARTIALLY COMPLETE

### 3.1 Fix Existing OpenCL Bugs — DONE (superseded by Rust rewrite)
The C OpenCL engine was replaced by a Rust implementation using the `opencl3` crate.

### 3.2 LZ77 OpenCL Match Finder — DONE (basic)
- Batch kernel (`lz77-batch.cl`) ported to Rust OpenCL engine
- Produces one match per position (greedy parse)
- Future: emit top-K matches per position for optimal parsing

### 3.3 BWT OpenCL — DONE
- GPU bitonic sort kernel (`kernels/bwt_sort.cl`) for suffix array construction
- Prefix-doubling with GPU sort steps, CPU rank assignment (O(n) per step)
- Produces byte-identical output to CPU BWT
- `MIN_GPU_BWT_SIZE = 32KB` threshold before using GPU path
- Performance: currently slower than CPU at tested sizes (1KB–64KB) due to
  O(log² n) kernel launch overhead per doubling step; gap narrows with size
- Future optimizations: local memory sort, GPU rank assignment, batched launches

### 3.4 Huffman OpenCL — PENDING
- Tree construction is inherently sequential (priority queue)
- But encoding (table lookup) and decoding (bit-parallel) can be GPU-accelerated
- Encoding: trivially parallel per-symbol table lookup
- Decoding: more challenging, but bit-parallel techniques exist

### 3.5 Pipeline Integration — DONE (basic)
- `bwt_encode_with_backend()` and `lz77_compress_with_backend()` dispatch helpers
  in `pipeline.rs` select GPU or CPU based on `CompressOptions`
- CLI tool supports `-g`/`--gpu` flag to enable OpenCL acceleration
- Future: chain GPU stages with minimal host↔device transfers

---

## Phase 4: pthread Backend

### 4.1 Block-Parallel Compression (pigz-style)
- Split input into N blocks (N = number of cores)
- Compress each block independently on a thread
- Concatenate outputs with block index/offset table
- Trade-off: independent blocks can't back-reference across boundaries,
  slightly worse compression ratio

### 4.2 Pipeline-Parallel
- Alternative to block-parallel: run different pipeline stages concurrently
- Thread 1 does LZ77 on block N while thread 2 does Huffman on block N-1
- Better compression ratio (no boundary penalty) but limited to pipeline depth

### 4.3 Parallel Decompression
- Requires block offset table (sidecar metadata or trailer)
- If block boundaries are known, decompress blocks in parallel
- For gzip compatibility: generate offset table on first single-threaded
  decompress, cache it for subsequent decompresses (as noted in compression-notes.md)

---

## Phase 5: Vulkan Compute Backend (Future)

### 5.1 Motivation
- Vulkan compute shaders are more widely available than OpenCL on consumer hardware
- Better driver support on mobile GPUs, game consoles, embedded
- SPIR-V shader format, explicit memory management

### 5.2 Implementation
- Port OpenCL kernels to GLSL compute shaders compiled to SPIR-V
- Vulkan compute pipeline setup (descriptor sets, command buffers)
- Same algorithms, different API surface

---

## Algorithm Parallelism Analysis

Not all algorithms benefit equally from parallelism. This table guides
backend selection:

| Algorithm | Compress Parallel? | Decompress Parallel? | GPU Suitable? | Notes |
|-----------|-------------------|---------------------|---------------|-------|
| **LZ77 match finding** | Excellent | N/A | Yes (current focus) | Each position searches independently. This is the expensive step. |
| **LZ77 match selection** | Poor | N/A | No (DP is sequential) | Backward DP is inherently sequential. Run on CPU after GPU match finding. |
| **LZ77 decompression** | Poor (per block) | Good (across blocks) | Marginal | Back-references create serial dependency within a block. Parallel across independent blocks only. |
| **Huffman encode** | Excellent | N/A | Yes | Per-symbol table lookup, fully independent. |
| **Huffman decode** | Moderate | N/A | Moderate | Bit-aligned reads create dependencies. Techniques exist (split at byte boundaries, speculative decode). |
| **BWT** | Excellent | Excellent | Yes | Suffix array construction is parallel sort. Inverse BWT is also parallelizable. |
| **MTF** | Poor | Poor | No | Sequential by definition (each symbol changes the list state). But very fast on CPU (tiny working set). |
| **RLE** | Moderate | Excellent | Marginal | Encode needs to find run boundaries (parallel scan). Decode is trivially parallel. |
| **Arithmetic coding** | Poor | Poor | No | Inherently sequential (state machine). But very fast per symbol. |
| **FSE/tANS** | Moderate | Excellent | Yes (decode) | Encode is sequential, but decode can be parallelized (zstd-style interleaved streams). |

### Strategy Summary
- **GPU:** LZ77 match finding, BWT suffix array, Huffman encode
- **pthread:** Block-parallel pipelines, parallel decode with block table
- **CPU only:** Optimal parse DP, MTF, arithmetic coding, match selection

---

## File Format

libpz needs its own container format to store:
1. **Magic number** and version
2. **Pipeline descriptor:** which stages were used and in what order
3. **Block table:** offsets and sizes for parallel decompression
4. **Per-block headers:** frequency tables, Huffman trees, or other metadata
5. **Compressed data blocks**
6. **Checksum** (xxhash or crc32)

The format should be extensible (new algorithm IDs) and support streaming
(blocks can be decompressed independently once located via the block table).

Consider compatibility modes that output standard gzip/deflate for
interoperability, at the cost of some features (no parallel decompress
without sidecar metadata).

---

## Build & Test Plan

### Testing Strategy
- **Unit tests:** Each algorithm, each backend, small inputs
- **Round-trip tests:** `decompress(compress(data)) == data` for all pipelines
- **Cross-backend tests:** Reference output must match GPU/pthread output byte-for-byte
- **Corpus tests:** Canterbury, Silesia, enwik8, E.coli genome
- **Fuzz tests:** libFuzzer + ASan for all encode/decode paths
- **Performance benchmarks:** Throughput (MB/s) and compression ratio vs gzip/zstd/bzip2

### Build System
- Autotools (existing) with feature detection for OpenCL, Vulkan, pthreads
- `./configure --with-opencl --with-vulkan --with-pthreads`
- Backends compile conditionally based on available libraries
- Reference backend always builds (no external dependencies beyond libc + libm)

---

## Milestones

| Milestone | Description | Depends On | Status |
|-----------|-------------|------------|--------|
| **M1** | All BUGS.md fixes landed, existing tests pass | - | **Done** (superseded by Rust rewrite) |
| **M2** | LZ77 + Huffman reference round-trip working | M1 | **Done** (Rust) |
| **M3** | Arithmetic coder reference working | M1 | **Done** (Rust) |
| **M4** | BWT + MTF + RLE reference working | M1 | **Done** (Rust) |
| **M5** | Full pipeline round-trips (DEFLATE-like, BW-like, LZA-like) | M2, M3, M4 | **Done** (Rust) |
| **M5.1** | Validation test suite with corpus tests | M5 | **Done** |
| **M5.2** | Cross-implementation validation (Rust vs C) | M1, M5 | N/A (C code removed) |
| **M5.3** | Fuzz testing infrastructure | M5 | Pending |
| **M5.4** | Benchmark suite and regression baselines | M5.1 | **Done** (criterion, stages + throughput) |
| **M6** | OpenCL LZ77 produces top-K match table | M1 | **Done** (lz77_topk.cl, K=4 candidates/position) |
| **M7** | Optimal parse DP selects minimum-cost chain from GPU matches | M6 | **Done** (backward DP in `optimal.rs`, `-O` CLI flag) |
| **M8** | OpenCL backend outperforms reference on large inputs | M7 | Pending |
| **M9** | pthread block-parallel compression | M5 | **Done** (V2 container, `-t` flag, pipeline parallelism) |
| **M10** | File format defined, CLI tool works end-to-end | M5 | **Done** (`pz` CLI with .pz format, `--gpu` flag) |
| **M11** | Benchmark suite vs gzip/zstd/bzip2/lz4 | M8, M9, M10 | **Done** (throughput.rs compares gzip/pigz/zstd) |
| **M12** | Vulkan compute backend | M8 | Pending |
| **M13** | zstd (.zst) format support | M7, FSE | Pending |

---

## zstd (.zst) Support — Gap Analysis

libpz already has most of the algorithmic building blocks needed for zstd:

### What we have

| Component | Module | Notes |
|-----------|--------|-------|
| FSE (tANS) entropy coder | `fse.rs` | GPU-friendly decode, table-driven |
| Huffman coding | `huffman.rs` | Canonical codes, bit-packed encode/decode |
| LZ77 (hash-chain, lazy) | `lz77.rs` | 32KB window, greedy/lazy/hash-chain strategies |
| Optimal parsing (backward DP) | `optimal.rs` | zstd levels 17+ style, top-K candidates |
| CRC32 | `crc32.rs` | Table-driven, used in gzip verification |
| Multi-block parallel compress | `pipeline.rs` | V2 container, block table, thread pool |
| GPU match finding | `opencl.rs` | Top-K match table for optimal parsing |

### What's missing

1. **zstd frame format (RFC 8878)** — Magic number `0xFD2FB528`, frame header
   with descriptor byte, window size, optional content size, optional
   dictionary ID, and frame checksum. Skippable frames.

2. **zstd block format** — Block header (last-block flag, block type, block
   size). Three block types: raw, RLE, compressed. Each compressed block
   contains a literals section and a sequences section.

3. **Sequence encoding** — zstd encodes LZ77 output as (literal_length,
   match_offset, match_length) triples using three interleaved FSE
   bitstreams. This is the core innovation: separate FSE tables for
   literal lengths, match lengths, and offsets, all interleaved into a
   single bitstream read in alternating order.

4. **Literals section** — Four types: raw (uncompressed), RLE (single byte
   repeated), Huffman-compressed (with embedded tree), and treeless
   (reuses previous block's Huffman tree).

5. **Large-window LZ77** — zstd supports windows up to 128MB (current
   libpz: 32KB). Requires extended offset encoding and larger hash
   tables. The `lz77.rs` window size constants and hash table would need
   to be parameterized.

6. **Repeat offset tracking** — zstd maintains 3 most-recent match offsets.
   Offset codes 1-3 reuse these without transmitting the full offset,
   significantly improving compression on structured data.

7. **xxHash64** — zstd uses xxHash64 for frame checksums (libpz has CRC32
   but not xxHash64). Could add as a new module or use an external crate.

8. **Predefined FSE tables** — When a compressed block omits custom
   frequency tables, zstd falls back to predefined default distributions
   for literal lengths, match lengths, and offsets. These are fixed
   tables defined in the spec.

9. **Dictionary support** — Optional pre-trained dictionaries with a
   dictionary ID and CRC. The LZ77 window is pre-filled with dictionary
   content, giving immediate match references for small inputs.

10. **Compression level mapping** — zstd defines 22 compression levels
    mapping to different strategies (greedy, lazy, double-lazy, optimal)
    with tuned parameters. libpz has the strategies but not the level
    mapping or parameter tuning.

### Implementation strategy

The recommended approach is to build zstd support as a new pipeline
(`Pipeline::Zstd`) that reuses existing modules where possible:

- **Reuse:** `fse.rs` for entropy coding, `huffman.rs` for literal
  compression, `lz77.rs` core match-finding logic, `optimal.rs` for
  high-level parsing.
- **New module `zstd.rs`:** Frame/block format parsing and serialization,
  sequence encoding/decoding, repeat offset tracking, predefined tables.
- **Extend `lz77.rs`:** Parameterize window size, add large-window support.
- **New module `xxhash.rs`:** xxHash64 implementation for frame checksums.
- **Extend `pipeline.rs`:** Add `Pipeline::Zstd` variant with zstd-specific
  stage sequence.

The largest effort is item 3 (sequence encoding with interleaved FSE
streams) — this is architecturally distinct from how libpz currently uses
entropy coders (one coder for the entire byte stream). zstd's approach
requires three coordinated FSE state machines operating on different
symbol alphabets within the same bitstream.
