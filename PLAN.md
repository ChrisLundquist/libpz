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

### 1.2 Complete LZ77 Reference
- Fix `FindMatchClassic` bounds bugs (BUG-02, BUG-03)
- Fix `LZ77_Decompress` overflow (BUG-06)
- Add hash-chain match finder (O(n) average vs current O(n*w) brute force)
- Add lazy matching (gzip-style: check if next position has longer match)
- Validate with Canterbury corpus and silesia corpus

### 1.3 Complete Huffman Reference
- Implement `huff_Encode` bit-packing (BUG-08)
- Implement `huff_Decode` tree walk (BUG-09)
- Fix signed char index bug (BUG-05)
- Add canonical Huffman codes (deterministic code assignment for decode without transmitting tree shape)
- Round-trip test with all test corpora

### 1.4 Implement Arithmetic / Range Coder
- Implement basic range coder (encode + decode)
- Support adaptive frequency model (no need for two-pass)
- Advantage over Huffman: fractional bits per symbol (no ceil(entropy) waste)
- Reference: Mark Nelson's arithmetic coder, ryg's rans implementation

### 1.5 Implement BWT (Burrows-Wheeler Transform)
- Implement naive BWT via suffix array construction
- Use SA-IS algorithm (linear time suffix array, Nong et al. 2009)
- Implement inverse BWT
- BWT is a transform, not a compressor -- pair with MTF + RLE + entropy coder

### 1.6 Implement MTF (Move to Front)
- Simple transform: maintain alphabet list, emit index, move symbol to front
- Pairs with BWT output to convert clustered symbols to small integers
- Trivially invertible

### 1.7 Implement RLE (Run Length Encoding)
- Basic RLE with escape mechanism for non-runs
- Useful after BWT+MTF where runs of zeros are common

### 1.8 Implement Frequency Analysis Fixes
- Fix unsigned overflow bug (BUG-15)
- `get_entropy` is already implemented and correct

### 1.9 Define Compression Pipelines
Standard pipelines combining the above stages:

| Pipeline | Stages | Similar to |
|----------|--------|------------|
| `PZ_DEFLATE` | LZ77 → Huffman | gzip |
| `PZ_BW` | BWT → MTF → RLE → Arithmetic | bzip2 |
| `PZ_LZA` | LZ77 → Arithmetic | lzma-like |
| `PZ_LZH` | LZ77 (optimal) → Huffman + FSE | zstd-like |

---

## Phase 2: Optimal LZ77 Match Selection

This is the core research contribution of libpz. The GPU finds all matches;
the CPU selects the minimum-cost chain.

### 2.1 GPU Match Table Generation
Current state: OpenCL kernel produces one match per input position. After
deduplication, this gives the greedy parse.

Target: For each position, produce the **top-K matches** (varying offsets and
lengths). Store as a match table:
```c
typedef struct {
    uint16_t offset;
    uint16_t length;
} lz77_candidate_t;

// match_table[position * K + k] = k-th best candidate at position
```

### 2.2 Cost Model
Define encoding cost for each match/literal:
- **Literal cost:** `-log2(freq[byte] / total)` bits (Shannon entropy of the byte)
- **Match cost:** `offset_bits(offset) + length_bits(length)` (depends on encoding format)
- Shorter matches with common offsets can be cheaper than longer matches with rare offsets

### 2.3 Backward DP Optimal Parse
```
cost[n] = 0  (end of input)
for i = n-1 down to 0:
    cost[i] = literal_cost(input[i]) + cost[i+1]     // literal option
    for each match (offset, length) at position i:
        match_price = match_cost(offset, length) + cost[i + length]
        cost[i] = min(cost[i], match_price)
```
Forward trace recovers the optimal sequence of matches/literals.

This is the approach used by zstd (levels 17+) and xz/lzma. The difference
is that libpz gets the match candidates from the GPU rather than from
hash chains on the CPU.

### 2.4 Entropy Feedback
After an initial parse, update the frequency model with the actual
match/literal distribution, then re-run the DP with updated costs. Converges
in 2-3 iterations (diminishing returns after that). This is similar to
zstd's "optimal parsing with price updates".

---

## Phase 3: OpenCL Backend

### 3.1 Fix Existing OpenCL Bugs
- Fix all OpenCL bugs from BUGS.md (BUG-04, BUG-07, BUG-11, BUG-13, BUG-16-20)
- Fix kernel bounds check order
- Fix null termination of kernel source
- Fix function pointer constness
- Proper error propagation in engine init

### 3.2 LZ77 OpenCL Match Finder
- Current: one match per position, 128KB window (lz77.cl) or 32KB batched (lz77-batch.cl)
- Target: emit top-K matches per position for optimal parsing
- Optimize memory access patterns (coalesced reads, local memory for window)
- Tune work-group sizes for different GPU architectures

### 3.3 BWT OpenCL
- GPU-parallel suffix array construction (prefix-doubling algorithm)
- Each doubling step is a parallel sort -- well suited to GPU
- Academic precedent: Deo & Keely (2013) GPU suffix array construction

### 3.4 Huffman OpenCL
- Tree construction is inherently sequential (priority queue)
- But encoding (table lookup) and decoding (bit-parallel) can be GPU-accelerated
- Encoding: trivially parallel per-symbol table lookup
- Decoding: more challenging, but bit-parallel techniques exist

### 3.5 Pipeline Integration
- Chain GPU stages with minimal host↔device transfers
- Keep data on GPU between stages where possible (LZ77 output → Huffman input)
- Use OpenCL events for pipeline overlap

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

| Milestone | Description | Depends On |
|-----------|-------------|------------|
| **M1** | All BUGS.md fixes landed, existing tests pass | - |
| **M2** | LZ77 + Huffman reference round-trip working | M1 |
| **M3** | Arithmetic coder reference working | M1 |
| **M4** | BWT + MTF + RLE reference working | M1 |
| **M5** | Full pipeline round-trips (DEFLATE-like, BW-like) | M2, M3, M4 |
| **M6** | OpenCL LZ77 produces top-K match table | M1 |
| **M7** | Optimal parse DP selects minimum-cost chain from GPU matches | M6 |
| **M8** | OpenCL backend outperforms reference on large inputs | M7 |
| **M9** | pthread block-parallel compression | M5 |
| **M10** | File format defined, CLI tool works end-to-end | M5 |
| **M11** | Benchmark suite vs gzip/zstd/bzip2/lz4 | M8, M9, M10 |
| **M12** | Vulkan compute backend | M8 |
