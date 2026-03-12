# GPU Compression Strategy

**Last updated:** 2026-03-01

## Executive summary

GPU acceleration in libpz is effective for LZ77 match-finding (parallel probes) but not for entropy coding (serial state machines). The optimal strategy is **GPU for LZ77 + CPU for entropy**, overlapped via the unified scheduler. Full on-device pipelines are not competitive on current hardware.

## What GPU is good at: LZ77 match-finding

The cooperative-stitch kernel (`lz77_coop.wgsl`) finds LZ77 matches using 1,788 probes per position across a 33KB lookback window:

1. **Near search** (1,024 probes): Every thread exhaustively scans offsets 1..1024
2. **Strided far search** (512 probes): 64 threads partition the 1024..33792 range into bands; each thread scans its band
3. **Stitch phase** (~252 re-tests): Threads share their top-4 matches via shared memory; each thread tries those offsets from its own position

This achieves ~94% of brute-force match quality on text data while running fully in parallel — one workgroup per input position.

### Why hash tables failed on GPU

Two hash-table kernels were attempted and abandoned:

- **Global hash table** (`lz77_hash.wgsl`): Atomic `atomicAdd` for insertion makes bucket order nondeterministic. Quality collapsed to 6.25% of CPU on repetitive data because last-writer-wins semantics lose the recent-match bias that makes hash chains effective.
- **Shared-memory hash table** (`lz77_local.wgsl`): Per-workgroup 4KB blocks. Single-candidate-per-slot (LZ4 style). 4KB window limit produced LZ4-class ratio — not competitive.

The probe-based approach avoids atomics entirely. Match quality depends on spatial coverage, not insertion order.

### Compression ratio gap vs gzip

Benchmark totals (Canterbury corpus, 13.9 MB, gzip default level 6):

```
gzip:       28.6%
PZ-LzSeqR:  35.1%   (code+extra-bits encoding, repeat offsets)
PZ-LzSeqH:  36.8%   (code+extra-bits encoding, Huffman entropy)
PZ-LzR:     41.6%   (fixed-width 5-byte LZ77 tokens, rANS entropy)
PZ-Deflate:  43.4%   (fixed-width 5-byte LZ77 tokens, Huffman entropy)
```

**The gap is primarily about match encoding, not match quality.** GPU match-finding is competitive with the CPU optimal DP parser. The problem is how matches are encoded:

| Encoding | Match cost | Example: offset=5, length=4 |
|----------|-----------|---------------------------|
| gzip (Deflate RFC 1951) | Variable: Huffman-coded length + distance symbols with extra bits | ~2-3 bytes |
| LzSeq (code+extra-bits) | Log2-based codes + packed extra bits + repeat offsets | ~2-4 bytes |
| LZ77 fixed-width | 5 bytes always: u16 offset + u16 length + u8 literal | 5 bytes |

A short match at offset 3, length 3 saves 3 literal bytes but costs 5 bytes in the LZ77 format — a **net loss of 2 bytes**. The same match in gzip costs ~2 bytes — a **net savings of 1 byte**. This per-token overhead compounds across thousands of matches.

LzSeq pipelines (35.1%) are much closer to gzip (28.6%) because they use efficient variable-width encoding with repeat offsets. The remaining ~6.5% gap comes from gzip's Deflate-specific Huffman code tables (tuned for LZ output per RFC 1951) and potentially more aggressive lazy matching decisions.

## What GPU is bad at: entropy coding

GPU rANS kernels exist (`rans_encode.wgsl`, `rans_decode.wgsl`) using chunked + lane-interleaved dispatch. They are slower than CPU:

| Metric | GPU | CPU | Ratio |
|--------|-----|-----|-------|
| 1MB encode | 57.9 MB/s | 75.5 MB/s | 0.77x |
| 1MB decode | 103.0 MB/s | 191.7 MB/s | 0.54x |

**Why GPU entropy loses:**

1. **Serial data dependency**: Each rANS state update depends on the previous state. This limits each thread to processing one lane sequentially. Parallelism only comes from chunk count x lane count, which is ~300 threads for a 300KB stream with 4KB chunks and 4 lanes — far below GPU saturation (~8K-16K threads).

2. **PCIe transfer overhead**: Uploading input and downloading compressed output adds fixed latency that dominates at typical block sizes (256KB-1MB).

3. **Diminishing returns from batching**: Ring-buffered overlap across streams is already implemented (`rans_encode_chunked_payload_gpu_batched`). Cross-block mega-batching was tried as independent block splitting (15+ iterations, commit `d2d75fe`) — it regressed to 0.3-0.6x CPU throughput due to per-block metadata overhead and load imbalance.

The GPU rANS performance gate (Slice 4 in PLAN-p0a) remains FAIL. Recommendation from that plan: "hold promotion to P0-B until GPU stage throughput improves materially."

### GPU Huffman and FSE

GPU Huffman encode (`huffman_encode.wgsl`) and FSE encode/decode (`fse_encode.wgsl`, `fse_decode.wgsl`) exist in production. These have the same serial-per-stream limitation as rANS but are used for specific pipelines (Deflate uses Huffman, Lzf uses FSE). Their per-stream performance has not been systematically compared to CPU in the same way as rANS.

## Current architecture: dual-path GPU support

GPU-accelerated compression uses the **streaming path** (`compress_stream` in `streaming.rs`), not the parallel scheduler. The parallel scheduler (`compress_parallel` in `parallel.rs`) is CPU-only by design — its GPU coordinator was removed after proving it bottlenecked at 28 MiB/s due to serializing entropy encoding on a single thread.

When `compress_with_options` is called with `Backend::WebGpu`, it routes through `compress_stream` internally (producing framed-format output, which `decompress` handles natively).

### Streaming GPU coordinator

For LZ-demux pipelines (Lzf, LzSeqR, etc.), `compress_stream_parallel` spawns a GPU coordinator thread:

```
Workers: read blocks → if GPU available and pressure < limit,
                        try_send to GPU coordinator via bounded channel
                      → else, compress on CPU

GPU coordinator thread:
  1. Block on gpu_rx.recv() for first request
  2. Drain additional requests via gpu_rx.try_recv()
  3. Batch blocks → engine.find_matches_batched()
  4. Demux matches → compress_block_from_demux() (CPU entropy)
  5. Send results to output_tx for ordered writing
  6. Decrement backpressure by batch_len (enables worker → GPU flow)
```

**Adaptive backpressure**: An `AtomicUsize` pressure score gates worker → GPU routing. Workers increment on `try_send` Full (+2), decrement on Ok (-1). The coordinator decrements by `batch_len` after completing each batch, preventing the one-way ratchet that would permanently lock out GPU.

**GPU failure recovery**: If `find_matches_batched` fails, all blocks in the batch fall back to CPU via `compress_block`.

### GPU entropy is not used

GPU entropy (rANS/FSE) is 0.54-0.77x CPU throughput due to serial state dependencies. All entropy encoding runs on CPU — the GPU is only used for LZ77 match-finding.

## What would need to change for GPU to win

### Highest leverage: focus on LzSeq pipelines

LzSeq pipelines (35.1%) are already much closer to gzip (28.6%) than LZ77-based pipelines (41-43%) because of their efficient code+extra-bits encoding with repeat offsets. **LzSeq is the right pipeline family for GPU compression.** The GPU already has a fused match+demux path for LzSeq via `lzseq_encode_gpu()`.

Remaining ratio gap (35.1% vs 28.6%) likely comes from:
- Gzip's Deflate Huffman tables are specifically designed for LZ output (RFC 1951)
- Our entropy coding (rANS/Huffman) may be less efficient on the specific distribution of LzSeq code streams
- Match quality differences may still exist in edge cases (lazy evaluation, repeat offset utilization)

### Near-term: better CPU/GPU overlap (low effort, moderate impact)

The current split — GPU LZ77 overlapped with CPU entropy — is already the right strategy. Gains come from reducing idle time:

- Batch-collect already overlaps LZ77 across blocks via `find_matches_batched()` ring buffer
- CPU workers process entropy stages while GPU works on the next batch of LZ77
- The unified scheduler handles this automatically

### Medium-term: improve LzSeq GPU path (medium effort, meaningful impact)

Since LzSeq is the best-performing pipeline family:
- Improve `lzseq_encode_gpu()` throughput — it already does fused match+demux on-device
- Add repeat offset tracking to the GPU coop kernel (currently CPU-only via LzSeq encoder)
- Profile LzSeqR end-to-end to find where time is actually spent

### Lower priority: LZ77 pipeline encoding efficiency

The LZ77-based pipelines (Deflate, Lzr, Lzf) use 5 bytes per token. Improving them requires either:
- Switching to variable-width encoding (essentially becoming LzSeq)
- Or accepting they'll always have worse ratio and focusing on throughput instead

### Long-term: competitive GPU entropy (high effort, uncertain payoff)

GPU entropy would need a fundamentally different approach to be competitive:

- **More parallelism**: Inputs would need to be 10-100x larger to amortize dispatch cost and saturate occupancy. At current block sizes (256KB-1MB), the overhead dominates.
- **On-device chaining**: LZ77 → demux → entropy in a single command buffer submission would eliminate PCIe round-trips between stages. This requires the `lz77_demux.wgsl` kernel (not yet written).
- **Alternative entropy coders**: Huffman is more parallelizable than rANS (no backward pass), but has worse compression ratio. The tradeoff may favor Huffman for GPU-heavy workloads.

## Approaches to try: parallel GPU entropy

The fundamental problem with GPU entropy is that rANS/Huffman are serial per-stream.
Three known approaches exist for parallelizing entropy coding on GPU. None have been
tried in libpz in their full form.

### Approach 1: dietGPU-style warp-per-segment rANS

**Source:** [dietGPU](https://github.com/facebookresearch/dietgpu) (Facebook Research)
**Performance:** 250-410 GB/s on A100

**How it works:**

The input is split into independent 4KB segments at encode time. Each segment is
rANS-encoded independently with its own final state. All segments share a single
probability table. At decode time, each CUDA warp (32 threads) takes one segment
and all 32 threads decode in lockstep, each maintaining its own rANS state.

```
Encode:
  Input: [seg0: 4KB] [seg1: 4KB] [seg2: 4KB] ...
  Each segment: independent rANS encode → compressed blob + final state

Wire format (ANSCoalescedHeader):
  [shared probability table]
  [per-segment final states: one u32 per lane per segment]
  [per-segment compressed word counts]
  [compressed data: read backward]

Decode:
  One warp (32 threads) per segment
  Each thread = one rANS lane, decodes its symbols round-robin
  Threads coordinate compressed word reads via __ballot_sync() + __popc()
  → 32 lanes × thousands of segments = massive parallelism
```

**Key differences from our chunked rANS:**
- dietGPU: 32 lanes per segment, thousands of segments → millions of threads
- libpz: 4 lanes per chunk, ~160 chunks → 640 threads
- dietGPU: warp-level ballot/popc for coordinated reads (CUDA-specific)
- libpz: WebGPU has no warp-level primitives (subgroup ops are limited)

**What we'd need to try:**
1. Increase lanes from 4 to 16 or 32 (our wg64 entry point exists)
2. Decrease chunk size from 256 bytes to match dietGPU's effective per-lane chunk
3. Accumulate segments across multiple streams and blocks into one mega-dispatch
4. WebGPU subgroup operations (`subgroupBallot`, `subgroupAdd`) as ballot/popc
   substitute — available in WebGPU but not universally supported

**Risk:** WebGPU's subgroup support is spotty. Without warp-level coordination,
threads can't efficiently share compressed word reads. May need to fall back to
workgroup shared memory, adding latency.

### Approach 2: Huffman sync-point decode

**Source:** nvCOMP, GDeflate
**Reference:** `docs/exec-plans/active/TODO-huffman-sync-decode.md`

**How it works:**

During Huffman encode, periodically record `(bit_offset, symbol_index)` checkpoints
every N symbols (N=1024). During decode, each GPU thread independently decodes one
segment between checkpoints using a LUT (2^L entries for max code length L).

```
Encode:
  Normal Huffman encode, but every 1024 symbols:
    record SyncPoint { bit_offset: u32, symbol_index: u32 }

Wire format:
  [Huffman tree header]
  [num_sync_points: u16]
  [sync_points: (bit_offset, symbol_index) × N]
  [bitstream]

Decode:
  One thread per segment (between adjacent sync points)
  Each thread: read L bits → LUT lookup → (symbol, code_length) → advance
  Segments are fully independent once sync points are known
```

**Advantage over rANS:** Huffman decode is forward-only (no backward pass) and
uses simple table lookup — no multiply/divide. The sync points add ~8 bytes per
1024 symbols (~0.8% overhead) but enable embarrassingly parallel decode.

**What we'd need to try:**
1. Modify `src/huffman.rs` encode to emit sync points (the GPU encode path
   already computes per-symbol bit offsets via prefix sum — sync points are
   just a sample of those offsets)
2. Write `kernels/huffman_decode.wgsl` — one thread per segment, LUT decode
3. Wire through `WebGpuEngine::huffman_decode_gpu()`
4. Measure: does parallel decode + sync point overhead beat CPU?

**Risk:** Lower compression ratio than rANS. But if decode throughput is 10x
higher, the ratio tradeoff may be worthwhile for GPU-heavy workloads. This is
the approach GDeflate uses in production.

### Approach 3: Recoil-style arbitrary-position rANS decode

**Source:** [Recoil (ICPP 2023)](https://arxiv.org/abs/2306.12141)

**How it works:**

Unlike dietGPU (which requires independent segments at encode time), Recoil
encodes as a single contiguous rANS stream. At encode time, it records the
intermediate rANS state at periodic positions. At decode time, any thread can
start decoding from any recorded position because the state is known.

```
Encode:
  Normal rANS encode (one contiguous stream)
  Every K symbols: store intermediate state in metadata
  States have a smaller upper bound after renormalization → compact storage

Wire format:
  [probability table]
  [num_checkpoints: u16]
  [checkpoints: (state: u32, symbol_index: u32, word_offset: u32) × N]
  [compressed stream]

Decode:
  Split stream heuristically to balance workload
  Each thread starts from a checkpoint with known state
  Decode forward until the next checkpoint's symbol_index
```

**Advantage:** No per-segment encode overhead. The stream is encoded as one
unit (best compression ratio). Parallelism is decided at decode time based
on available decoder resources ("decoder-adaptive scalability").

**What we'd need to try:**
1. Modify rANS encode to record intermediate states every K symbols
2. Store checkpoint metadata in wire format (compact — states have bounded
   range after renormalization)
3. Write decode kernel that starts from arbitrary checkpoints
4. Measure: does checkpoint overhead eat the ratio savings vs dietGPU segments?

**Risk:** More complex implementation. The checkpoint metadata format needs
careful design. The paper claims "reducing unnecessary data transfer by
adaptively scaling parallelism overhead to match decoder capability" but
the practical gain over dietGPU's simpler approach is unclear for our
input sizes.

### Comparison of approaches

| Approach | Encode change | Ratio impact | Parallelism | WebGPU feasible? |
|----------|--------------|-------------|-------------|-----------------|
| dietGPU segments | Split into 4KB independent segments | Small (shared table) | Very high | Maybe (needs subgroup ops) |
| Huffman sync points | Add checkpoints every 1024 symbols | ~0.8% overhead | High | Yes (no warp primitives needed) |
| Recoil checkpoints | Store intermediate rANS states | Minimal | High | Yes |

**Recommendation:** Try Huffman sync-point decode first (Approach 2). It has
the simplest implementation, doesn't require subgroup operations, and a
detailed implementation plan already exists (`TODO-huffman-sync-decode.md`).
If Huffman ratio is unacceptable, try Recoil checkpoints (Approach 3) which
preserves rANS ratio. dietGPU segments (Approach 1) are the proven approach
but may not be feasible without CUDA warp primitives.

### Approach 4: Reduce per-stream framing overhead

**Source:** Internal analysis (2026-03-01)

Orthogonal to the parallel entropy approaches above. Currently each rANS stream
carries a 512-byte frequency table (256 × u16). For narrow-alphabet streams
like `offset_codes` (~20 symbols) and `length_codes` (~15 symbols), most
entries are zero.

**Sparse frequency tables:** Store only nonzero entries:
```
Dense:  [freq: 256 × u16 LE]                    = 512 bytes always
Sparse: [n: u8] [symbols: n × u8] [freqs: n × u16 LE] = 3n + 1 bytes
```

For `offset_codes` with 20 symbols: 61 bytes instead of 512. Saves ~450
bytes per stream, ~1.3-1.6 KB per block across all streams.

**Impact:** Closes ~9% of the gzip gap on the Canterbury corpus (~83 KB
total savings on 13.9 MB). Small but free — the GPU kernels read pre-built
table buffers and never parse the wire format, so zero WGSL changes needed.

Touch points: `serialize_freq_table()` and `deserialize_freq_table()` in
`src/rans.rs`, plus callers in `src/webgpu/rans.rs`. The `HEADER_SIZE`
constant becomes variable.

### Approach 5: Improve match encoding efficiency

**Source:** Internal analysis (2026-03-01)

The largest ratio gap vs gzip comes from per-match encoding cost, not
entropy coding throughput. Structural encoding improvements for LzSeq:

**5a. Zstd-style sequences (eliminate flags stream)**

Replace per-token flags with sequences: `(literal_run_length, offset, length)`.
Kills the flags stream entirely. Reduces from 6 streams to 4-5. The literal
run length also acts as implicit context.

**5b. Entropy-code the extra bits**

`offset_extra` and `length_extra` bypass entropy coding (raw packed bits).
If extra bit values are skewed (small values more common), rANS could
save 5-15%. Tradeoff: needs additional frequency tables per stream.

**5c. Larger repeat offset cache**

LzSeq tracks 3 recent offsets (matching zstd). Expanding to 4-8 could
capture more repeats on structured data. Each additional repeat offset
saves all extra bits for that match (0 extra bits vs 8-16 for a far match).

**5d. Combined literal/length alphabet**

Deflate puts literals (0-255) and length codes (257-285) into one
286-symbol Huffman tree. Common literals get short codes. LzSeq keeps
them separate — they can't share probability mass. A combined alphabet
could save bits but would change the stream structure.

**Impact:** These are format-level changes. 5a (zstd sequences) has the
highest ceiling — it's the structural difference between how zstd and
LzSeq encode. 5c (repeat offsets) is the simplest to implement. The
format is not yet released, so all changes are free.

## Key files

| File | Purpose |
|------|---------|
| `kernels/lz77_coop.wgsl` | Cooperative-stitch LZ77 kernel (production) |
| `kernels/lz77_hash.wgsl` | Hash-table LZ77 kernel (abandoned, kept for reference) |
| `kernels/lz77_local.wgsl` | Per-block 4KB hash kernel (abandoned) |
| `kernels/lz77_topk.wgsl` | Brute-force top-K for optimal parsing |
| `kernels/rans_encode.wgsl` | GPU rANS encode (functional, slower than CPU) |
| `kernels/rans_decode.wgsl` | GPU rANS decode (functional, slower than CPU) |
| `src/pipeline/parallel.rs` | Unified scheduler with GPU coordinator |
| `src/webgpu/lz77.rs` | GPU LZ77 host API, ring buffer, batching |
| `src/webgpu/rans.rs` | GPU rANS host API, batched encode/decode |
| `src/webgpu/lzseq.rs` | Fused GPU LZ77+demux for LzSeq pipelines |

## Related plans

| Plan | Status | Relevance |
|------|--------|-----------|
| `PLAN-unified-scheduler-north-star.md` | Phases 1-3 complete, 4-5 blocked | Scheduler done; GPU rANS perf gate failed |
| `PLAN-p0a-gpu-rans-vertical-slice.md` | Slices 0-3 PASS, Slice 4 FAIL | GPU rANS works but is slower than CPU |
| `imperative-weaving-star.md` (worktree plan) | Phase 1-2a done, 2b not started | Batch-collect + FusedGpu done; lz77_demux deferred |
