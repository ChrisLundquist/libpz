## GPU wins on LZ77 match-finding, loses on entropy coding

The optimal CPU/GPU split is GPU for LZ77 (parallel probes, no atomics) and CPU for entropy (serial state machines). The unified scheduler already overlaps these.

The FusedGpu path for Lzr/LzSeqR routes entropy to GPU, which is *slower* than CPU. gpu_fused_span() returning Some((0, 1)) is counterproductive until GPU entropy reaches CPU parity.

**For future agents**: Don't add more stages to gpu_fused_span() until GPU entropy is competitive. The current best strategy is StageGpu for stage 0 (LZ77) and Stage for stage 1+ (CPU entropy).
