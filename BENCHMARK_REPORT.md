# Compression Benchmark Report

**Date:** 2026-02-11
**Platform:** Linux x86_64, SIMD (SSE2/AVX2), no GPU/OpenCL
**Corpus:** Canterbury (11 files) + Large (3 files) = 14 files, 13.3 MiB total

## Tools Tested

| Tool | Version | Description |
|------|---------|-------------|
| **pz** | 0.1.0 (libpz) | This project — deflate/bw/lza pipelines |
| gzip | 1.12 | Standard LZ77+Huffman |
| pigz | 2.8 | Parallel gzip |
| bzip2 | 1.0.8 | BWT-based |
| zstd | 1.5.5 | Facebook's Zstandard |
| xz | 5.4.5 | LZMA2-based |
| lz4 | 1.9.4 | Ultra-fast LZ compression |

## Aggregate Results — Full Corpus

| Tool | Level | Ratio | Comp MiB/s | Decomp MiB/s |
|------|-------|------:|-----------:|--------------:|
| lz4 | 1 (fast) | 0.503 | 28.7 | 30.1 |
| zstd | 1 (fast) | 0.298 | 23.4 | 26.0 |
| zstd | 3 (default) | 0.276 | 21.9 | 25.9 |
| gzip | 1 (fast) | 0.344 | 21.1 | 26.4 |
| pigz | 6 (default) | 0.286 | 20.9 | 26.1 |
| pigz | 9 (best) | 0.281 | 13.6 | 26.2 |
| **pz** | **deflate-greedy** | **0.520** | **12.8** | **16.4** |
| zstd | 9 (high) | 0.252 | 12.7 | 25.8 |
| bzip2 | 1 (fast) | 0.251 | 9.7 | 15.3 |
| bzip2 | 9 (default) | 0.224 | 9.1 | 11.0 |
| xz | 1 (fast) | 0.264 | 8.9 | 19.4 |
| gzip | 6 (default) | 0.286 | 7.0 | 25.6 |
| **pz** | **lza-greedy** | **0.509** | **6.4** | **5.2** |
| **pz** | **auto** | **0.419** | **6.1** | **10.1** |
| **pz** | **deflate-lazy** | **0.527** | **5.8** | **16.8** |
| **pz** | **bw-lazy** | **0.240** | **5.2** | **7.6** |
| **pz** | **trial** | **0.240** | **4.5** | **7.5** |
| lz4 | 9 (high) | 0.348 | 4.5 | 31.1 |
| gzip | 9 (best) | 0.281 | 2.0 | 26.6 |
| **pz** | **deflate-optimal** | **0.504** | **1.7** | **16.6** |
| **pz** | **lza-optimal** | **0.486** | **1.6** | **4.9** |
| zstd | 19 (ultra) | 0.219 | 1.4 | 24.2 |
| xz | 6 (default) | 0.218 | 1.4 | 20.8 |

## Head-to-Head: pz Pipelines vs Competitors

### pz deflate vs gzip (LZ77 + Huffman)

| Metric | pz deflate-lazy | gzip -6 | Delta |
|--------|----------------:|--------:|------:|
| Ratio | 0.527 | 0.286 | +84% (worse) |
| Compress | 5.8 MiB/s | 7.0 MiB/s | -17% |
| Decompress | 16.8 MiB/s | 25.6 MiB/s | -34% |

pz's deflate pipeline compresses ~1.8x worse than gzip at default settings. The LZ77 match finder is producing significantly less efficient output. This is the weakest pipeline.

### pz bw vs bzip2 (BWT-based)

| Metric | pz bw-lazy | bzip2 -9 | Delta |
|--------|----------:|--------:|------:|
| Ratio | 0.240 | 0.224 | +7% (worse) |
| Compress | 5.2 MiB/s | 9.1 MiB/s | -43% |
| Decompress | 7.6 MiB/s | 11.0 MiB/s | -31% |

The BW pipeline is competitive on ratio (only 7% behind bzip2), but about 2x slower on compression. This is pz's strongest pipeline. On several files (E.coli, kennedy.xls), pz bw matches or beats bzip2's ratio.

### pz lza vs xz (LZ + entropy coder)

| Metric | pz lza-optimal | xz -6 | Delta |
|--------|---------------:|------:|------:|
| Ratio | 0.486 | 0.218 | +123% (worse) |
| Compress | 1.6 MiB/s | 1.4 MiB/s | +19% |
| Decompress | 4.9 MiB/s | 20.8 MiB/s | -76% |

The LZA pipeline is dramatically behind xz on ratio despite similar speed, suggesting the range coder + LZ77 combination isn't achieving the entropy coding efficiency of LZMA2.

### pz trial vs zstd (auto-select vs balanced default)

| Metric | pz trial | zstd -3 | Delta |
|--------|--------:|-------:|------:|
| Ratio | 0.240 | 0.276 | -13% (better) |
| Compress | 4.5 MiB/s | 21.9 MiB/s | -79% |
| Decompress | 7.5 MiB/s | 25.9 MiB/s | -71% |

Trial mode achieves the best ratio of any pz mode by correctly selecting the BW pipeline per-file. It beats zstd default on ratio by 13%, but is ~5x slower on compression and ~3.5x slower on decompression.

## Best Compression Ratio Per File (Top 5)

### Large files (>= 256KB)

**E.coli (4.5 MiB)** — genomic data, low alphabet, repetitive
| Rank | Tool | Ratio | Compress ms |
|------|------|------:|----------:|
| 1 | zstd -19 | 0.248 | 3907 |
| 2 | xz -6 | 0.256 | 4541 |
| 3 | **pz bw** | **0.256** | **715** |
| 4 | bzip2 -9 | 0.270 | 407 |
| 5 | gzip -9 | 0.280 | 4868 |

pz bw matches xz's ratio at 6x the speed on genomic data.

**bible.txt (3.9 MiB)** — English text, large dictionary
| Rank | Tool | Ratio | Compress ms |
|------|------|------:|----------:|
| 1 | bzip2 -9 | 0.209 | 337 |
| 2 | xz -6 | 0.219 | 2629 |
| 3 | zstd -19 | 0.221 | 2316 |
| 4 | **pz bw** | **0.243** | **615** |
| 5 | bzip2 -1 | 0.248 | 310 |

**kennedy.xls (1006 KiB)** — binary spreadsheet, highly structured
| Rank | Tool | Ratio | Compress ms |
|------|------|------:|----------:|
| 1 | xz -6 | 0.048 | 434 |
| 2 | zstd -19 | 0.068 | 555 |
| 3 | zstd -9 | 0.108 | 67 |
| 4 | **pz bw** | **0.110** | **192** |
| 5 | zstd -1 | 0.112 | 39 |

**ptt5 (502 KiB)** — fax image, sparse binary
| Rank | Tool | Ratio | Compress ms |
|------|------|------:|----------:|
| 1 | xz -6 | 0.082 | 157 |
| 2 | zstd -19 | 0.085 | 357 |
| 3 | bzip2 -9 | 0.097 | 43 |
| 4 | zstd -9 | 0.097 | 55 |
| 5 | **pz bw** | **0.107** | **80** |

## Auto-Selection Analysis

The `--auto` (heuristic) mode often picks deflate when bw would be significantly better:

| File | Auto picks | Auto ratio | Best pz ratio | Best pipeline |
|------|-----------|----------:|--------------:|--------------|
| E.coli | bw | 0.256 | 0.256 | bw (correct) |
| bible.txt | deflate | 0.533 | 0.243 | bw (wrong) |
| world192.txt | deflate | 0.514 | 0.246 | bw (wrong) |
| kennedy.xls | deflate | 0.285 | 0.110 | bw (wrong) |
| ptt5 | bw | 0.107 | 0.107 | bw (correct) |
| plrabn12.txt | deflate | 0.758 | 0.336 | bw (wrong) |
| lcet10.txt | deflate | 0.632 | 0.289 | bw (wrong) |
| alice29.txt | deflate | 0.721 | 0.303 | bw (wrong) |

The heuristic auto-selector picks the wrong pipeline on 6/14 files. The `--trial` mode always picks correctly since it tests all 3 pipelines, but is slower due to the trial overhead.

## Pareto Frontier (ratio vs speed)

These are the tools not dominated by any other on both ratio AND speed:

| Tool | Level | Ratio | Comp MiB/s |
|------|-------|------:|-----------:|
| lz4 | 1 (fast) | 0.503 | 28.7 |
| zstd | 1 (fast) | 0.298 | 23.4 |
| zstd | 3 (default) | 0.276 | 21.9 |
| zstd | 9 (high) | 0.252 | 12.7 |
| bzip2 | 1 (fast) | 0.251 | 9.7 |
| bzip2 | 9 (default) | 0.224 | 9.1 |
| zstd | 19 (ultra) | 0.219 | 1.4 |
| xz | 6 (default) | 0.218 | 1.4 |

**No pz configuration reaches the Pareto frontier.** Every pz setting is dominated by at least one external tool that achieves both better ratio and better speed.

## Key Findings

### Strengths

1. **pz bw is competitive on ratio** — Only 7% behind bzip2 overall, and matches xz on genomic data (E.coli). The BWT + MTF + RLE + RangeCoder pipeline is well-tuned.

2. **Trial mode auto-selection works well** — Correctly identifies bw as optimal for most files, achieving the best pz ratio (0.240) which beats zstd default on ratio.

3. **pz bw on structured/genomic data** — On E.coli, pz bw achieves 0.256 ratio matching xz -6 but at 6x the compression speed (715ms vs 4541ms).

### Weaknesses

1. **Deflate pipeline is weak** — pz deflate achieves ~0.52 ratio vs gzip's 0.29 at similar speeds. The LZ77 match finder and/or Huffman coding is significantly underperforming. This is the biggest gap.

2. **LZA pipeline underperforms xz** — Despite the same conceptual approach (LZ77 + entropy coder), pz lza ratios (0.48-0.51) are far behind xz (0.22). The range coder is not achieving LZMA2-level efficiency.

3. **Decompression speed lags** — pz decompression is 2-4x slower than gzip/zstd across all pipelines. This hurts real-world usability.

4. **Auto-selection heuristic is unreliable** — Picks deflate too often when bw would be far better. The heuristic thresholds need retuning.

5. **Small file overhead** — On files <10KB, pz deflate can produce output *larger* than input (ratio > 1.0 on xargs.1 and grammar.lsp), due to container format overhead.

### Recommended Improvements

1. **Fix deflate LZ77 output quality** — The ~80% ratio gap vs gzip suggests the match finder isn't finding long enough matches, or Huffman tree construction is suboptimal. This is the single highest-impact fix.

2. **Retune auto-selection thresholds** — The bw pipeline should be the default for most text/structured data. The current heuristic over-favors deflate.

3. **Optimize decompression paths** — 2-4x gap vs gzip/zstd decompression suggests room for improvement in decode hot loops.

4. **Reduce container overhead** — The .pz header causes ratio > 1.0 on tiny files. Consider minimal headers for small inputs.

5. **Improve range coder efficiency** — The LZA pipeline's range coder achieves ~0.49 ratio vs xz's 0.22 on the same data. Investigate context modeling and match representation.
