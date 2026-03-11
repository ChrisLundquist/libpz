# TODO: Benchmark Lzfi vs LzssR — consolidation candidate

## Question

Are both Lzfi and LzssR worth keeping? They use the same demuxer (LZSS, 4
streams) and differ only in entropy coder (interleaved FSE vs rANS).

## Current state

| Property | Lzfi | LzssR |
|----------|------|-------|
| Demuxer | LzssEncoder (4 streams) | LzssEncoder (4 streams) |
| Entropy | Interleaved FSE | rANS |
| Auto-selected | Yes (high entropy + matches) | Never |
| Pipeline ID | 5 | 6 |
| GPU entropy | Yes (interleaved FSE) | Yes (rANS Recoil) |

## Criterion benchmark data (2026-03-10)

### Entropy throughput (CPU-only, Canterbury 64KB)

| Coder | Encode | Decode |
|-------|--------|--------|
| FSE | 238-302 MB/s | 412-533 MB/s |
| rANS (basic) | 326-446 MB/s | 262-316 MB/s |
| rANS (chunked) | 279-486 MB/s | 453 MB/s – 1.06 GB/s |

FSE decode is ~2x faster than rANS basic decode. rANS chunked decode is
competitive but requires the chunked wire format.

### Full pipeline throughput (Canterbury corpus, 25 MB)

| Pipeline | Compress | Decompress |
|----------|----------|------------|
| Lzfi | **543 MB/s** | **1.23 GB/s** |
| LzSeqR | 333 MB/s | 1.22 GB/s |
| Lzf | 295 MB/s | 1.02 GB/s |

Lzfi dominates compress speed (63% faster than LzSeqR). Decompress is
effectively tied across all LZ pipelines.

### Recommendation

LzssR has no measurable advantage over Lzfi:
- Same demuxer (LZSS, 4 streams)
- rANS decode is slower than FSE decode (262 vs 533 MB/s)
- rANS encode is faster but Lzfi pipeline throughput is still higher
- LzssR is never auto-selected
- GPU Recoil decode is interesting but GPU entropy is known to be slower
  than CPU (0.54-0.77x), so the GPU rANS advantage doesn't materialize

**Action:** LzssR is a candidate for removal, similar to the Lzr removal.
Keep only if a concrete use case for GPU Recoil decode emerges.

## Remaining action items

1. ~~Run Criterion benchmarks~~ Done (see above)
2. Run `./scripts/bench.sh` for compression ratio comparison on Canterbury+Silesia
3. Decide: remove LzssR or document its niche use case
4. If removing: retire Pipeline ID 6, update wire-formats.md, add to retired IDs

## Files

- `src/pipeline/stages.rs` — stage dispatch for both pipelines
- `src/pipeline/mod.rs` — `auto_select_pipeline`, `select_pipeline_trial`
- `src/pipeline/blocks.rs` — entropy encode/decode dispatch
