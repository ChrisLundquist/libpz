# GPU Recoil Decode: Implementation & Performance Findings

**Date:** 2026-03-05
**Branch:** claude/recoil-parallel-rans-72wQu (PR #105 + GPU decode extension)

## Summary

Implemented GPU Recoil decode by mapping each Recoil split to a GPU "chunk" and dispatching
via the existing `rans_decode_chunk_impl` kernel. No new WGSL kernel was needed — the existing
kernel handles multiple independent chunks natively. The host-side packs split states (with
lane rotation for non-aligned boundaries) and word stream slices into the per-chunk buffer layout.

## Architecture

### How it works

1. Parse the interleaved rANS header (shared freq table, word streams)
2. For each Recoil split → one GPU chunk:
   - **Lane rotation**: kernel lane `l` gets Recoil state for original lane `(sym_start + l) % K`
     (where K = num_lanes, sym_start = split's starting symbol index)
   - **Word streams**: per-split word regions copied right-aligned (matching kernel's expected layout)
   - **Output offset**: set to split's symbol_index so output writes go to correct positions
3. Single dispatch via `rans_decode_chunked_gpu_with_chunk_meta` processes all splits in parallel

### Lane rotation explained

The kernel assumes lane 0 processes positions 0, K, 2K, ... relative to chunk start. But for a
Recoil split starting at symbol S, position S was originally decoded by lane S%K. So we rotate
the states array: `kernel_state[l] = recoil_state[(S + l) % K]`. This preserves correctness
without modifying the kernel.

## Performance Results

### Raw rANS decode (entropy decode only, 4 MB input)

| Path | Throughput | Notes |
|------|-----------|-------|
| CPU interleaved (baseline) | 268 MB/s | Single-threaded, cache-hot |
| CPU Recoil (64 splits, sequential) | 86 MB/s | 3.1x slower than baseline |
| GPU interleaved (1 chunk) | 8.5 MB/s | PCIe transfer dominated |
| GPU Recoil (4 splits) | 21 MB/s | |
| GPU Recoil (16 splits) | 36 MB/s | |
| GPU Recoil (64 splits) | 65 MB/s | **7.5x faster than GPU baseline** |

GPU Recoil scales with split count (more splits = more parallel workgroups = better GPU utilization).
At 64 splits it's 7.5x faster than GPU single-chunk, but still 4x slower than CPU baseline.

### Full pipeline (LZR: LZ77 + rANS, 4 MB input)

| Path | Throughput |
|------|-----------|
| Standard (no recoil) | 5606 MB/s |
| Recoil CPU (16 splits, thread::scope) | 88 MB/s |
| Recoil GPU (16 splits) | 62 MB/s |

The full pipeline is dominated by LZ77 decompression, not rANS. Repetitive test data compresses
to ~0.7% (29 KB payload for 4 MB input), making the rANS decode stage negligible. GPU overhead
(buffer alloc + upload + readback) far exceeds the tiny rANS compute.

### CPU Recoil overhead

CPU Recoil with `std::thread::scope` is 60x slower than standard interleaved decode (88 vs 5606 MB/s).
The thread spawning overhead dominates at all tested sizes. More splits = worse CPU performance.

## Key Findings

1. **GPU parallelism works**: GPU Recoil with 64 splits is 7.5x faster than single-chunk GPU decode
2. **PCIe overhead dominates**: Buffer alloc + upload + readback costs ~90% of GPU wall time
3. **rANS is not the bottleneck**: In LZ-based pipelines, LZ77 decode (inherently sequential) dominates
4. **CPU thread::scope is too expensive**: Thread creation cost exceeds rANS decode work at typical sizes
5. **Test data skews results**: Repetitive patterns compress to <1%, making rANS streams tiny

## Where GPU Recoil would win

- **Integrated GPU** (shared memory, no PCIe): eliminates transfer overhead
- **Fused GPU pipeline**: rANS data stays on device between stages (no round-trip)
- **High-entropy data**: larger rANS payloads relative to output (more compute to parallelize)
- **Batch multi-stream decode**: amortize dispatch overhead across many streams
- **Very large inputs** (>100 MB): GPU scaling would eventually overcome fixed overhead

## Recommendations for future work

1. **CPU Recoil**: Add minimum chunk size threshold; fall back to single-threaded decode when
   `original_len / num_splits < ~64KB`
2. **GPU Recoil**: Integrate into batched shared-table decode path for multi-stream pipelines
3. **Fused decode**: When GPU LZ77 decode exists, chain rANS → LZ77 on-device
4. **Benchmark with high-entropy data**: Canterbury corpus or random data would stress rANS more
5. **Consider removing CPU thread::scope**: Use sequential Recoil decode as default (it's still
   useful for GPU), only parallelize when a thread pool is already available
