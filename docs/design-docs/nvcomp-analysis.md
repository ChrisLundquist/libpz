# nvcomp GPU Compression: Architecture Analysis & Applicable Patterns

Technical analysis of NVIDIA nvcomp's GPU compression architecture, with patterns applicable to libpz's GPU backend.

**Last updated:** 2026-02-22

## Overview

NVIDIA nvcomp is the reference GPU compression library, achieving 90-320 GB/s on A100 hardware. This document analyzes how nvcomp achieves those speeds and which architectural patterns are applicable to libpz's WebGPU backend.

## nvcomp Throughput Numbers

**Hardware: NVIDIA A100 (80GB HBM2e, 2 TB/s memory bandwidth)**

| Algorithm | Compress | Decompress | Notes |
|-----------|----------|------------|-------|
| LZ4 | 90-96 GB/s | 312-320 GB/s | Hash-table matching, block-independent |
| Cascaded (RLE+Delta+Bitpack) | 75-500 GB/s | Similar | Data-type optimized, analytical workloads |
| ANS (dietGPU) | 250-410 GB/s | 250-410 GB/s | Segmented rANS, warp-per-segment |
| GDeflate | ~50 GB/s | ~150 GB/s | GPU-optimized Deflate variant |
| Snappy | ~80 GB/s | ~200 GB/s | Fast, low-ratio |
| Blackwell HW engine | N/A | 600+ GB/s | Fixed-function, fused copy+decompress |

**For comparison, libpz on AMD Radeon Pro 5500M:**

| Algorithm | Compress | Decompress | Bottleneck |
|-----------|----------|------------|------------|
| LZ77 (GPU) | 8.2 MB/s | N/A | PCIe transfer (70% of time) |
| Huffman encode (GPU) | 250-460 MB/s | N/A (CPU only) | Atomic contention |
| Full Deflate chain | 7.2 MB/s | N/A | PCIe + dispatch overhead |
| LzSeqR (CPU) | 22 MB/s | 34 MB/s | LZ match finding, rANS decode |

Sources: [nvcomp benchmarks](https://github.com/NVIDIA/nvcomp/blob/main/doc/Benchmarks.md), [nvcomp algorithms overview](https://github.com/NVIDIA/nvcomp/blob/main/doc/algorithms_overview.md)

## Architectural Patterns

### Pattern 1: Massive Batching

**nvcomp approach:** Process hundreds to thousands of independent blocks simultaneously. The low-level batch API processes multiple chunks in fewer, higher-occupancy kernels. Small workloads (single image, single block) are explicitly documented as CPU-favorable.

> "GPUs only outperform CPUs on large batches due to amortization of fixed overhead costs."
> — [nvcomp flexible interfaces blog](https://developer.nvidia.com/blog/accelerating-lossless-gpu-compression-with-new-flexible-interfaces-in-nvidia-nvcomp/)

**libpz status:** We dispatch 40-160 chunks per stream (chunk_size=256, 40KB input). Wave-packing fills RDNA waves but we're still only launching 3-10 workgroups. On a 13MB corpus, we could batch all files' rANS chunks into a single dispatch (thousands of chunks).

**Applicable pattern:** Batch across streams, not just within one stream. Accumulate multiple compression requests and submit as one GPU dispatch. This is the single highest-impact architectural change.

### Pattern 2: Block-Independent LZ Matching

**nvcomp approach:** Each block gets its own independent hash table. No cross-block dependencies. This sacrifices some compression ratio (can't reference matches across blocks) but enables embarrassingly parallel matching.

> "Per-block independent hash tables, parallel search... smaller windows per block (KiB scale)"
> — [nvcomp algorithms overview](https://github.com/NVIDIA/nvcomp/blob/main/doc/algorithms_overview.md)

**libpz status:** Our GPU LZ77 kernel already uses per-block hash tables (cooperative-stitch kernel). The 128KB sliding window is per-block. This pattern is already implemented.

**Gap:** Our blocks are too small (128-256KB). nvcomp typically uses larger blocks or processes many small blocks simultaneously. The key insight is that block size matters less than total batch size.

### Pattern 3: Segmented Entropy Coding

**nvcomp approach:** Split entropy bitstreams into many independent segments. Each GPU warp (32 threads) handles one 4KB segment. This converts the sequential entropy decode problem into an embarrassingly parallel one.

Facebook's dietGPU achieves 250-410 GB/s on A100 using byte-oriented rANS with warp-per-segment design.

> "Split single rANS bitstream into multiple starting points with known intermediate states."
> — [dietGPU: GPU-based lossless compression](https://github.com/facebookresearch/dietgpu)

**libpz status:** Our chunked rANS already segments the stream (256-byte chunks, 4 lanes each). The wave-packed kernel fills 64-wide waves. But our chunk count (40-160) is far below the thousands of segments nvcomp uses to saturate GPU occupancy.

**Applicable pattern:** Smaller chunks (64-128 bytes?) with many more of them, processed in a single massive dispatch. The per-chunk framing overhead increases, but GPU utilization goes up dramatically. The break-even depends on our framing overhead per chunk.

### Pattern 4: Persistent GPU Memory

**nvcomp approach:** Pre-allocate reusable GPU memory pools. Avoid per-operation allocation/deallocation. Keep compressed data GPU-resident when possible ("FreeLunch" pattern — compressed data stays on GPU, decompressed data flushed after use).

> "Pre-allocate reusable GPU memory pools to avoid repeated allocation/deallocation."
> — [nvcomp documentation](https://docs.nvidia.com/cuda/nvcomp/)

**libpz status:** We use ring-buffered slots for LZ77 batching (3 slots, ~28MB). Huffman encode has a `fully_on_device` variant. rANS chunked encode/decode allocates fresh buffers per call.

**Applicable pattern:** Pool all GPU buffers across the WebGpuEngine lifetime. Pre-size for the maximum expected batch. This eliminates allocation jitter but trades memory for latency predictability.

### Pattern 5: Minimize Host-Device Transfers

**nvcomp approach:** Chain operations on-device. LZ77 output feeds directly into entropy coding without host round-trip. Blackwell goes further with fused copy+decompress (data decompressed in transit over PCIe).

> "Decompression Engine eliminates sequential host-to-device copy + software decompress... compressed data transferred directly across PCIe/C2C and decompressed in transit."
> — [Blackwell decompression engine blog](https://developer.nvidia.com/blog/speeding-up-data-decompression-with-nvcomp-and-the-nvidia-blackwell-decompression-engine/)

**libpz status:** GPU Deflate chaining keeps LZ77→Huffman on-device. But our GPU rANS encode still requires host-side table building and chunk metadata assembly. The on-device demux kernel was added (Phase 6) but the full pipeline isn't chained yet.

**Applicable pattern:** This is the north-star goal (unified scheduler). Each host↔device transfer we eliminate saves 10-25ms. Priority order:
1. Keep rANS frequency tables on-device (small, reusable)
2. Chain demux→rANS encode on-device
3. Eliminate match array download (GPU-side dedup)

### Pattern 6: Hardware-Aware Kernel Design

**nvcomp approach:** Kernels are tuned per GPU generation. Warp size, shared memory budget, register pressure, and occupancy are all tuned for specific SM architectures. The low-level API exposes these controls.

**libpz status:** We target WebGPU (portable) which abstracts hardware details. Workgroup size is our main tuning knob. The wave-packed kernels (workgroup_size=64) are tuned for RDNA but may not be optimal for other architectures.

**Applicable pattern:** Limited by WebGPU abstraction. We can offer multiple workgroup-size entry points (already do: wg4, wg8, wg64, packed) and select at runtime based on adapter properties. Beyond that, WebGPU prevents the fine-grained tuning nvcomp does with CUDA.

## Why nvcomp Is Fast and We're Not (Yet)

The throughput gap is primarily explained by three factors:

### 1. Hardware Gap (100x)

| Resource | A100 | Radeon Pro 5500M | Ratio |
|----------|------|------------------|-------|
| Memory bandwidth | 2,039 GB/s | ~96 GB/s | 21x |
| Compute (FP32) | 19.5 TFLOPS | 4 TFLOPS | 5x |
| PCIe | Gen4 x16 (32 GB/s) | Gen3 x8 (8 GB/s) | 4x |
| VRAM | 80 GB HBM2e | 8 GB GDDR6 | 10x |

Even a perfectly optimized kernel on our hardware would be 20x+ slower than A100 due to memory bandwidth alone.

### 2. Batch Size Gap (10-100x)

nvcomp processes thousands of blocks per dispatch. We process 40-160. GPU fixed costs (kernel launch, synchronization, PCIe latency) are amortized over 10-100x more work items.

### 3. API Abstraction Gap

nvcomp uses CUDA with direct access to warp-level primitives, shared memory control, and hardware-specific optimizations. We use WebGPU which provides portability but prevents warp-shuffle, warp-vote, explicit register allocation, and hardware-specific tuning.

## Actionable Recommendations for libpz

Ordered by expected impact:

### High Impact (would move the needle)

1. **Cross-stream batching** — Accumulate chunks from multiple compression requests into a single GPU dispatch. Target: 1000+ chunks per dispatch instead of 40-160.

2. **On-device pipeline chaining** — Eliminate host round-trips between demux→rANS. This is the north-star unified scheduler goal.

3. **Smaller rANS chunks for GPU** — 64-128 byte chunks give 4-8x more workgroups. Measure framing overhead vs occupancy gain.

### Medium Impact (incremental improvements)

4. **Persistent buffer pools** — Pre-allocate all GPU buffers at engine creation. Eliminates allocation jitter.

5. **GPU Huffman decode** — Sync-point parallel decode (already planned, P2). Enables full Deflate pipeline on GPU.

6. **Chunk-based Huffman atomic reduction** — Replace per-bit atomic_or with workgroup-local packing. Documented as potential 5-10x for Huffman encode.

### Low Impact (nice to have)

7. **Runtime workgroup size selection** — Query adapter for optimal workgroup size instead of fixed 64.

8. **Larger test hardware** — Many GPU optimizations only pay off on discrete GPUs with high memory bandwidth. Testing on Apple M-series or discrete AMD/NVIDIA would reveal different break-even points.

## References

- [NVIDIA nvcomp GitHub](https://github.com/NVIDIA/nvcomp)
- [nvcomp Benchmarks](https://github.com/NVIDIA/nvcomp/blob/main/doc/Benchmarks.md)
- [nvcomp Algorithms Overview](https://github.com/NVIDIA/nvcomp/blob/main/doc/algorithms_overview.md)
- [Accelerating Lossless GPU Compression with Flexible Interfaces](https://developer.nvidia.com/blog/accelerating-lossless-gpu-compression-with-new-flexible-interfaces-in-nvidia-nvcomp/)
- [Blackwell Decompression Engine](https://developer.nvidia.com/blog/speeding-up-data-decompression-with-nvcomp-and-the-nvidia-blackwell-decompression-engine/)
- [dietGPU: GPU ANS Implementation](https://github.com/facebookresearch/dietgpu)
- [Massively Parallel ANS Decoding](https://github.com/weissenberger/multians)
- [GDeflate GPU-Friendly Deflate](https://github.com/elasota/gstd)

## Related libpz Documents

- [GPU Batching Strategy](gpu-batching.md)
- [LZ77 GPU Implementation](lz77-gpu.md)
- [Pipeline Architecture](pipeline-architecture.md)
- [North Star: Unified Scheduler](../exec-plans/active/PLAN-unified-scheduler-north-star.md)
- [GPU vs CPU LzSeqR Benchmark](../exec-plans/BENCHMARK-GPU-vs-CPU-LZSEQR-2026-02-21.md)
