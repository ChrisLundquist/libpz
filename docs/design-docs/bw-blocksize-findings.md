# BW block-size sweep — findings (default → 1 MiB)

**Date:** 2026-06-09
**Tool:** `examples/bw_blocksize.rs` (ratio + 1-thread + all-cores decode per size)
**Change:** `DEFAULT_BW_BLOCK_SIZE` 512KB → 1 MiB. 2-4 MiB measured and rejected
for the default.

## Why the sweep

The 512KB BW default predated adaptive FSE `accuracy_log` (#129) and RUNA/RUNB
zero-run coding (#131), and the BWT+CM spike observed that post-BWT entropy is
block-size-monotonic. Question: does a larger block now pay, and what does it
cost on the decode side?

## Results (Silesia blob, 202 MB, M5 Max; `compress_with_options` path)

| block | ratio | enc all-cores | dec 1-thread | dec all-cores* |
|---|---|---|---|---|
| 256K | 29.03% | 136 MB/s | 84.6 MB/s | 1118 MB/s |
| 512K (old default) | 28.28% | 113 | 77.0 | 864 |
| **1M (new default)** | **27.74%** | 99.5 | 69.6 | 444* |
| 2M | 27.38% | 86 | 62.9 | 235 |
| 4M | 27.39% | 73 | 50.1 | 176 |

Text members keep improving past 2M (webster 21.02→20.41% at 4M, dickens
27.98→27.42%) but binary members regress slightly, netting the blob flat.

\* The all-cores column here is `decompress_parallel` (in-memory API), whose
scaling degrades with block size (11.2× → 6.4× → 3.7× over 512K→1M→2M). The
**CLI streaming decoder scales ~16× at 1M blocks** (the documented 1120 MB/s
blob figure) — see "decode scaling gap" below.

## Finding 1: the CLI was already shipping 1 MiB bw blocks

The streaming path reads `options.block_size` directly and never applies the
BW adjustment in `adjusted_options`, so when #125 raised the global
`DEFAULT_BLOCK_SIZE` to 1 MiB, `pz bw` silently inherited it. Proof: the
documented CLI numbers match the 1M sweep rows exactly (blob 27.79≈27.74,
dickens 28.9≈28.89), not the 512K rows (28.28, 30.08). The 512KB constant only
ever governed the library `compress_with_options` path — the two paths
disagreed by 0.54pp. This change **ratifies the accident**: the constant now
matches shipped CLI behavior, and the library path gains −0.54pp.

## Finding 2: 2-4 MiB defaults are Pareto-dominated

At 2M the blob lands on 27.38% — almost exactly zstd-12's 27.39%, which
decodes ~1500 MB/s on a *single* core vs bw's ~235 aggregate here. Buying
≤0.36pp to land on a dominated point while halving aggregate decode again is
a bad default. The aggregate collapse is structural, not scheduling: per-core
decode only drops ~10% per doubling, but 18 concurrent inverse-BWT random
walks outgrow the shared cache (each block needs ~5n of LF/rank working set),
so scaling falls superlinearly. Large-block bw is bandwidth-walled on this
machine.

(The old "FSE degrades beyond ~1MB" rationale for 512KB is obsolete: adaptive
`accuracy_log` (#129) re-tunes the table per block size.)

## Finding 3: decode scaling gap between the two decode paths

At 1M blocks, the framed/streaming decoder achieves ~16× thread scaling
(1120 MB/s) while `decompress_parallel` achieves ~6.4× (444 MB/s) on the same
data. Worth a look if the in-memory API's aggregate decode ever matters; the
CLI is unaffected.

## What this preserves

- CLI behavior and all documented CLI benchmarks: unchanged (it was already 1M).
- The deferred BWT+CM tail wants ≥1M blocks (its ratio gate passed at 1M, not
  512K) — the default now matches.
- Per-file text maximization (webster 20.4% at 4M) remains available to
  library callers via an explicit `block_size`; there is deliberately no CLI
  knob for it today.
