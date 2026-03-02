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

## Current architecture: unified scheduler

All GPU work routes through `compress_parallel_unified()` in `src/pipeline/parallel.rs` (PR #101). The architecture:

```
CPU workers: pick up tasks from shared VecDeque<UnifiedTask>
  ├─ Stage(stage_idx, block_idx)     → run on CPU
  ├─ StageGpu(stage_idx, block_idx)  → send to GPU coordinator via bounded channel
  └─ FusedGpu(start, end, block_idx) → send to GPU coordinator

GPU coordinator thread:
  ├─ Batch-collects Stage 0 requests → find_matches_batched() (ring-buffered)
  ├─ Processes Stage N requests → run_compress_stage() individually
  └─ Processes Fused requests → runs stages start..=end sequentially
```

**Deadlock prevention**: Workers use `try_send()` on the bounded channel. If the channel is full, the block falls back to CPU processing.

**GPU failure recovery**: If any GPU operation fails, the block is re-enqueued as `Stage(0, block_idx)` for CPU retry.

### The FusedGpu path problem

`gpu_fused_span()` currently returns `Some((0, 1))` for Lzr and LzSeqR, routing both stages to GPU. But since GPU entropy is 0.77x CPU encode, this actually makes those pipelines *slower* than the decomposed path (GPU LZ77 + CPU entropy). The fused path is architectural preparation for the case where GPU entropy becomes competitive — it is not a performance win today.

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
