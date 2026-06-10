# Clean-slate parallel codec design ("pz2")

**Date:** 2026-06-09
**Status:** ✅ Design validated and **graduated into the container** — the §5
prototype gate **passed at 3.45×** (target was 2×). `src/pz2.rs` decodes the
Silesia blob at **1405 MB/s single-thread at 32.22%** vs `Lzf`'s 408 MB/s at
32.18% — same parse, byte-identical match decisions, new wire format.
Per-core decode is within ~3% of zstd-3 (1452 MB/s measured same-box), while
every block stays independently decodable. pz2 is now `Pipeline::Pz2`
(`pz -p pz2`, id 13), riding the shipped streaming container with a 2 MiB
block default (window = block, §9): **blob 32.0% at 11.0 GiB/s all-cores CLI
decode** — better ratio than lzf (32.2%) AND 2.3× its decode wall, 1.28×
faster than `pzstd -3 -p18` (the honest competitor) at 0.6pp ratio cost.
See §7-§9.
**Method:** Derived, not invented — every choice below cites the libpz
measurement that forces it. Receipts live in CLAUDE.md "Known dead ends",
`gpu-experiments-wave2-conclusions.md`, `bwt-cm-findings.md`,
`ctx-literals-stage0-findings.md`, `bw-blocksize-findings.md`, and the
2026-06 decode profile.

## 1. The three serial enemies

Four months of experiments killed ideas in exactly three ways:

1. **Per-symbol entropy state.** FSE decode is latency-bound at ~4.7 cyc/byte
   (the `decode_table[state]` load-to-use chain); 4-way interleave buys only
   1.15× on a wide OoO core; SSE2 rANS is 32% *slower* than scalar; GPU
   entropy is 0.54×; the CM mixer caps at ~11 MB/s/core.
2. **Parse-order dependence.** Fully parallel GPU parsing (ParlZ) lost 37.6%
   ratio; GPU hash tables lose recency through atomics (match quality 6% vs
   99.6%).
3. **Adaptive model state.** Kills GPU (serial bit feedback + RMW tables),
   kills SIMD, kills lanes. Its static replacement — per-context tables —
   loses to header economics (256-context literals: −2.2 to −12.4pp *worse*),
   and the information ceiling for order-1 literal context is below 3pp
   anyway.

A clean-slate design evicts all three from the **decoder** and pays for ratio
elsewhere.

## 2. Principles (each with its receipt)

**P1 — Decode is the product; encode buys it.** Asymmetric to the point of
rudeness: the decoder is a straight-line, branch-light, byte-aligned,
static-table consumer; all intelligence is encode-time. Receipt: every decode
"optimization" we tried on the existing entropy stack was futile, while
encode-side levers (window, parse, adaptive accuracy, zrle) shipped pp after
pp at zero decode cost.

**P2 — Independence is a format invariant, at three granularities.**
- *Segments* (~64–256 MB): fully independent; distributed/multi-machine unit.
- *Blocks* (~1 MiB): independent except **read-only** references into a
  shared immutable dictionary region. Receipt: `decompress_block` carries no
  history argument today and that is exactly why pz decode fans out 13–16×;
  the cross-block-dict spike proved a read-only shared past costs parallel
  decode nothing.
- *Lanes* (intra-block): N independent entropy bitstreams with format-defined
  framing. Receipt: retrofitting interleave onto pz was found to be a wire
  format change — so it must be in the format from byte zero. N targets CPU
  ILP (4–8), not GPU occupancy (thousands), per the GPU-entropy dead end.

**P3 — Entropy: shortest critical path wins, not best theoretical ratio.**
Bulk stream (literals) = multi-stream table-lookup Huffman (huff0 class:
one L1-resident flat table, no inter-symbol state, 4 lanes for ILP). Small
skewed streams (sequence codes) = tiny FSE, where per-symbol latency is
amortized ~1:8 against output bytes. Receipt: the decode profile — the wall
is chained loads per symbol; zstd's huff0/FSE split is the existence proof at
~1500 MB/s/core, and our FSE-side micro-opts all failed.

**P4 — No adaptivity, no per-context bulk tables.** All models static per
block. Receipt: ctx-literals Stage-0 (ideal ceiling 1.1–2.7pp, realized
+0.1–0.6pp, fine contexts negative) and BWT+CM (11 MB/s/core). The ratio
budget that adaptive modeling would chase is spent on window/parse/dict
instead, which are decode-free.

**P5 — Transforms over modeling.** A detected, per-block, pluggable transform
stage (stride/byte-plane/delta) in front of the LZ core. Receipt: the largest
per-class wins ever measured here came from O(n) branch-free transforms
(x-ray −29pp, sao −21pp vs LZ), with SIMD-trivial inverses and validated
cheap detection (stride decorrelation, PR #138). Transform tags are per
block, so mixed corpora route per block.

**P6 — Sequential encoder, GPU-assisted candidate generation.** GPU does
deterministic sort-based match candidates, histogramming, transform
detection; the parse itself stays serial on CPU consuming those candidates.
Single-dispatch, persistent buffers. Receipts: hybrid GPU-find + CPU-parse is
the only GPU architecture that survived (+7–17% shipped); ParlZ and
per-round-sync designs died.

**P7 — Tokens shaped like memcpy.** zstd-style sequences
`(literal_run_len, match_len, offset)`: bulk literal-run copies from a
pre-decoded literal buffer, match copies via `extend_from_within` with
exponential-doubling overlap handling. Receipt: match-copy is already
memcpy-bound in pz; per-literal flag streams (pz's current wire) buy a little
ratio but force per-token branching. A decode-first codec takes the runs.
u32 offsets, log2-bucket codes + raw extra bits (proven tight), repeat-offset
cache (cheap, helps structured data).

**P8 — No BWT in the parallel tiers.** The subtlest lesson: inverse BWT is
embarrassingly parallel on paper and anti-parallel in silicon — concurrent
LF-mapping pointer-chases blow shared cache, collapsing aggregate scaling
11.2×→6.4×→3.7× as blocks grow 512K→1M→2M. LZ decode is sequential-write and
bandwidth-shaped; it scales. Buy ratio with windows, parse and dictionaries,
not pointer-chases.

**P9 — Per-block method byte.** Every block self-describes
(LZ / numeric-transform / store), so routing is per block, store-fallback is
per block, and mixed files get the right treatment per region. Receipt: the
Num integration gotcha — pz's single whole-stream pipeline tag forced
whole-file routing and a separate scheduler registration; per-block tags are
strictly more capable at one byte per block.

**P10 — GPU decode is a non-goal, stated with receipts.** All measurements
(0.54× entropy decode, recency-dependent gathers, init-time overheads) say
decode customers are CPU many-core. Caveat recorded honestly: these are
M5-unified-memory results at our batch sizes; a 10K-lane GPU-decode format
(multians-style) is a different machine class we did not disprove — but its
per-lane state flush overhead costs ratio and nothing measured here suggests
it beats CPU many-core on workstation hardware.

## 3. Format sketch

```
segment := segment_header, block_table, blocks...
block_table[i] := { comp_len: u32, orig_len: u32, method: u8, xform: u8 }
block (method=LZ) :=
  [seq_count: u32]
  [literal_total: u32]
  [lit_lane_lens: 4 × u32]        // 4 independent Huffman bitstreams
  [huff_table_desc]                // one shared canonical table, ≤11-bit codes
  [lit_lanes: bytes...]            // byte-aligned, decode into literal buffer
  [ll_code_stream][ml_code_stream][of_code_stream]   // tiny FSE/Huffman
  [extra_bits_lane]                // raw LSB bit lane, byte-aligned
block (method=NUM) := existing numeric wire (stride, per-plane tags, FSE planes)
block (method=STORE) := raw bytes
```

Dictionary tier (phase 2): `dict_id` + window priming exactly as spiked —
one immutable `Arc<[u8]>` shared read-only by all decode workers.

## 4. Honest landing zone

- **Ratio:** ~zstd-2/3 class on Silesia (the parse and window are pz's
  current ones; Huffman literals give back ~0.3–0.5pp vs FSE; sequences give
  back a little vs flag-streams; dict/window growth claws it back).
- **Decode:** target ≥2× pz today per core (~850+ MB/s; huff0-class loops
  reach 1–1.5 GB/s), × the proven 13–16× block fan-out → 10–20 GB/s
  aggregate on this machine, against pzstd's frame parallelism as the honest
  competitor — finer-grained, on by default, transform-aware.
- **Encode:** GPU-assisted into the GB/s range (already shipped machinery).
- **The frontier statement our data supports:** ratio below ~27% on mixed
  data costs decoder serialism that no parallel packaging recovers. The
  parked carry-safe range coder remains the only owned bridge (11 MB/s/core).

## 5. Prototype gate (what `src/pz2.rs` must prove)

The only unproven load-bearing claim is P3+P7 in Rust on this machine:

> Single-thread decode of (4-lane Huffman literals + sequence splicing)
> ≥ **2× pz lzf** (≥ ~850 MB/s) at ratio within ~1pp of lzf on Silesia,
> using the same parse.

Pass → the decode-first format graduates (container, dict, transforms fold in
from shipped pz components). Fail → write the findings doc and accept that
pz's niche is the Num corner plus research substrate.

## 6. Explicit non-goals (the graveyard, do not revisit)

GPU entropy in any direction; GPU hash tables / recency structures; parallel
parse; adaptive decoder state of any kind; per-context bulk-stream tables;
sub-block entropy parallelism beyond fixed lanes (Recoil-style splits need
recomputable state — adaptive context has none; 8-way scalar rANS degrades);
BWT tiers as the parallel story; window-capped suffix sorts.

## 7. Prototype results (2026-06-09, M5 Max, single-thread, round-trip-verified)

`examples/pz2_eval.rs`, 1 MiB blocks, same parse as `Lzf` (tokenizer
fidelity-tested byte-identical). Numbers below are the tuned decoder
(8 lanes + unchecked hot-loop stores + exponential-doubling overlap copies):

| file | pz2 % | pz2 dec MB/s | lzf % | lzf dec MB/s | speedup |
|---|---|---|---|---|---|
| **silesia blob** | **32.223** | **1405** | 32.183 | 408 | **3.45×** |
| dickens | 38.852 | 1017 | 38.728 | 278 | 3.65× |
| webster | 29.339 | 1255 | 29.335 | 361 | 3.48× |
| mozilla | 36.226 | 1236 | 36.030 | 381 | 3.24× |
| nci | 7.909 | 3112 | 8.085 | 1219 | 2.55× |
| xml | 11.570 | 2651 | 11.804 | 911 | 2.91× |
| sao | 80.151 | 1050 | 79.859 | 247 | 4.25× |
| x-ray | 77.024 | 1406 | 76.753 | 241 | 5.84× |

(nci/xml *improve* ratio — sequences beat per-token flag streams on
structured data; the worst regression anywhere is +0.3pp on sao.)

**What the iteration taught (3 measured steps, 293 → 480 → 1361 MB/s on the
blob path):**
1. Safe-Rust v0 with per-token `Vec` operations: only 1.04-1.3× — allocation
   and per-call overhead swamp the entropy win.
2. + wildcopy splice (validated-then-unsafe 16-byte chunks, fully
   initialized buffers): 1.7-1.9× on text.
3. + **fusing the sequence-code streams into the splice loop** as three more
   independent Huffman chains (instead of three upfront FSE passes with
   intermediate `Vec`s): 3.1-3.6×. P3 amended by measurement: even at ~1:8
   amortization, *separate-pass* FSE costs more than fused Huffman lanes;
   what matters is one pass over the sequences with all chains live in
   registers — exactly zstd's shape.

**Post-gate decoder tuning (1361 → 1405 blob; sao +8%, x-ray +9%):**
- **Lane count swept 4/6/8/12/16: 8 wins** (+7% over 4 on text; 12/16
  regress slightly — register spill). Lane count is part of the wire, so it
  was settled now, while the format is a day old. Ratio cost of the 4 extra
  lane headers: +0.002-0.005pp. NUM_LANES = 8 shipped.
- **Unchecked hot-loop stores** (raw write cursors; bound proven by
  `rounds * 5 ≤ min_len`): +2%.
- **Exponential-doubling overlap copies for offsets 2-15** above 32 bytes
  (mirrors the shipped lzf fix): Silesia-neutral, removes the O(ml)
  byte-chain worst case on periodic data.
- **huff0-style X2 dual-symbol table: measured dead end on this core.**
  Decoder-only change, no wire impact, full implementation measured: with 8
  lanes live the literal loop is *execution-throughput*-bound, not
  chain-latency-bound, so even 81% pair coverage (dickens, slot-weighted)
  bought only +1.5%, while low-coverage data paid for the wider entries
  (x-ray, 27-34% coverage: **−8.5%**). Reverted. Same physics as the FSE
  "4-way interleave buys only 1.15×" finding: the M5's OoO window already
  hides table-load latency once enough independent chains exist. Do not
  revisit X2 without first checking lane saturation.

**Honest caveats:** encode is unoptimized (~lzf-parse-bound, fine — P1);
no dict/transform integration yet (shipped pz components). Hardening
status: Miri is **not runnable on this box** (no nightly toolchain, rustup
shims broken) — the compensating control is `examples/pz2_soak.rs`, a
deterministic seed-reportable soak (round-trip fuzz over 8 input families
incl. block-boundary sizes; bit-flip/stomp/truncate/length-field mutations
of valid streams; pure garbage — all decodes under `catch_unwind`, wrong
`orig_len` included). Passed 45 s debug (checked arithmetic: 5.2K round
trips, 209K mutated + 42K garbage decodes) and 180 s release at 1 MiB
blocks (18K round-trips, 722K mutated + 144K garbage decodes), zero
panics. Run Miri on the unit suite when a nightly toolchain exists before
default-pipeline promotion.

## 8. Container integration results (2026-06-09, CLI end-to-end, all cores)

pz2 graduated into the shipped streaming container as `Pipeline::Pz2`
(id 13, `pz -p pz2`): single-stage scheduler entry like Num, decode gets
`orig_len` from the existing block table, zero container-format changes.
Covered by the four cross-pipeline test matrices in `pipeline/tests.rs`.

Methodology: silesia blob (202.1 MiB), hyperfine (2 warmup + 5 runs, warm
page cache), `pz -d -c -q file > /dev/null`, M5 Max (6P+12E). These numbers
are NOT comparable to the CLAUDE.md table (different I/O mode); compare
within this table only.

| codec | comp % | decode wall | decode user CPU | aggregate |
|---|---|---|---|---|
| **pz pz2** | 32.22 | **16.3 ms** | 174 ms | **12.1 GiB/s** |
| pz lzf | 32.18 | 40.5 ms | 569 ms | 4.9 GiB/s |
| pz lzseqr | 32.16 | 48.0 ms | 687 ms | 4.1 GiB/s |
| pzstd -3 -p18 | 31.40 | 22.9 ms | 157 ms | 8.6 GiB/s |
| zstd -3 (1 thread) | 31.40 | 139.2 ms | 137 ms | 1.42 GiB/s |

- **pz2 beats pzstd -3 by 1.40× on parallel decode wall time** at 0.8pp
  ratio cost — and not by burning more cores: user CPU is comparable
  (174 vs 157 ms). The §4 prediction ("pzstd as the honest competitor")
  is settled on this machine.
- The old §7 projection of 17-20 GB/s assumed ideal 13-16× fan-out; at
  16 ms wall, fixed costs (process start, container parse, serial stdout
  writer) dominate — 202 MiB is simply not enough work to saturate. The
  per-core × cores ceiling from user CPU is ~21 GiB/s.
- Compress all-cores: pz2 725 ms (279 MiB/s) vs lzf 1034 ms — **1.43×
  faster encode** at equal ratio (Huffman bit-writer beats FSE encode),
  14.8× thread scaling.
- `--trial` now includes Pz2 (listed before Lzf so exact size ties go to
  the faster decoder; verified picking pz2 on a mozilla sample). The `-a`
  heuristic still answers Lzf for the general-LZ case — switching that
  default to Pz2 is a deliberate follow-up decision, gated on the
  hardening soak + a Miri pass.
- `--greedy` / window / max-match-len flags now reach Pz2 via
  `pz2_seq_config` (same mapping as the LzSeq demux path); greedy takes
  the dickens-slice ratio 38.6% → 35.7%, round-trip verified.
- Remaining integration gaps: no dict tier; Num-style transforms not yet
  routed per block (P5/P9 phase 2).

## 9. Block-size = window sweep → 2 MiB default (2026-06-10)

Pz2's window is block-capped (blocks parse cold), so block size is the
window lever. Sweep (`examples/pz2_block_sweep.rs`, window = block size,
ST, round-trip-verified) — ratio improves monotonically on real files;
the blob's 8 MiB reversal is concatenation-boundary content mixing:

| input | 1 MiB | 2 MiB | 4 MiB | 8 MiB |
|---|---|---|---|---|
| blob | 32.223 | 32.004 | 31.949 | 32.021 |
| webster | 29.339 | 29.007 | 28.779 | 28.572 |
| dickens | 38.852 | 38.497 | 38.291 | 38.177 |
| mozilla | 36.226 | 35.912 | 35.729 | 35.639 |
| samba | 23.910 | 23.653 | 23.151 | 22.978 |

ST decode is size-neutral-to-better at every size (1401→1427 on the blob —
fewer table builds; the bw 2-4 MiB decode collapse was inverse-BWT cache
physics, and LZ's sequential-write decode confirms immunity). The decisive
axis was **concurrent encode cache pressure**, invisible to the ST sweep:

| default | blob ratio | dec wall (all-cores) | enc wall (all-cores) |
|---|---|---|---|
| 1 MiB | 32.22% | 16.3 ms (12.1 GiB/s) | 0.73 s (279 MiB/s) |
| **2 MiB (shipped)** | **32.00%** | **17.9 ms (11.0 GiB/s)** | **1.89 s (107 MiB/s)** |
| 4 MiB | 31.95% | 18.9 ms (10.4 GiB/s) | 4.0 s (50 MiB/s) |

The ST sweep predicted 4 MiB encode at 1.9x slower; e2e measured **5.5x** —
18 workers each pointer-chasing a 16 MiB hash-chain `prev` array thrash the
shared cache (P8's lesson, encode-side). 2 MiB keeps most of the ratio
(−0.22pp blob, −0.26 to −0.36pp files), decodes within 10% of the 1 MiB
wall (a tail-packing artifact of the 202 MiB corpus, not per-byte cost),
and still beats pzstd -3 decode by 1.28x. The decode-wall deltas here are
fan-out granularity, not codec cost. Revisit 4 MiB when encode gets GPU
candidate generation or cache-aware chain layouts; the dict tier (P2
phase 2) is the structural fix that decouples reach from block size
entirely.

Also landed: package-merge (optimal length-limited) Huffman lengths
replacing the halve-and-rebuild heuristic — measured only ~0.004pp (the
heuristic was near-optimal) but it is exact, Kraft-guaranteed by
construction, simpler, and closes the "slightly suboptimal" caveat.

## 10. Auto-greedy parse → 31.0%, Pareto-superior to pzstd-3 (2026-06-10)

The per-file probe (`examples/pz2_parse_probe.rs`, 2 MiB blocks) overturned
the inherited "greedy regresses structured data" rule for the pz2 wire:
**greedy ≤ lazy on 11/12 Silesia files** (dickens −3.2pp, reymont −3.0,
sao −2.3, webster/mr −1.9; worst case mozilla +0.03pp = noise). The lazy
deferral's win evidently belonged to the flag-stream wire and smaller
windows, not to the parse itself at 2 MiB reach.

So Pz2's `Auto` strategy now parses **greedy per block**, with one guard
(`pz2_auto_greedy`): near-random blocks (entropy > 7.5, match density
< 0.1 — same rule as `select_pipeline`) stay lazy since there is no ratio
to buy. Explicit `--lazy` / `--greedy` are respected.

Blob e2e (same methodology as §8):

| codec | ratio | dec wall | enc wall |
|---|---|---|---|
| **pz pz2 (auto-greedy)** | **31.04%** | **16.8 ms (12.0 GiB/s)** | 3.15 s (64 MiB/s) |
| pz pz2 (lazy, prior) | 32.00% | 17.9 ms | 1.89 s |
| pzstd -3 -p18 | 31.40% | 22.9 ms | — |
| zstd -3 (1 thread) | 31.40% | 139.2 ms | — |

Decode got *faster* with greedy (fewer, longer sequences → fewer entropy
symbols and fewer splice iterations; user CPU 181 → 172 ms). Net:
**pz2 is now Pareto-superior to pzstd -3 on (ratio, parallel decode)** —
0.36pp better ratio AND 1.36× faster wall — and dominates lzf on every
axis except encode. Encode at 64 MiB/s all-cores is the P1 trade,
recoverable later via GPU candidate generation (P6).

## 11. Dict-tier spike PASSED — head dict captures the ceiling (2026-06-10)

The roadmap's spike #2 (cross-block dictionary; previously *planned*, never
run — P2's "as spiked" citation was aspirational) is now executed via
`pz2::encode_with_prefix` / `decode_with_prefix` + `examples/pz2_dict_probe.rs`
(2 MiB blocks, greedy, round-trip-verified per block). Two modes measured:
**sliding prefix** (each block references the preceding D bytes — the
ratio *ceiling*, decode-serializing) and **fixed head dict** (every block
references the file's first D bytes — the parallel-friendly production
shape: one immutable region, 2-wave decode).

ratio delta vs no dict (pp):

| file | sliding 4Mi | sliding 16Mi | head 4Mi | head 16Mi | head/ceiling @16Mi |
|---|---|---|---|---|---|
| samba | −1.234 | −1.344 | −1.036 | −1.347 | ~100% |
| webster | −1.039 | −1.111 | −0.901 | −1.075 | 97% |
| nci | −0.568 | −0.673 | −0.490 | −0.606 | 90% |
| xml | −0.287 | −0.287 | −0.287 | −0.287 | 100% |
| mozilla | −0.679 | −0.745 | −0.021 | −0.494 | 66% |

**Finding 1: per-file redundancy is global, not local-recency** — a fixed
head dict captures 66-100% of the sliding ceiling on individual files, so
the dict tier needs NO decode serialization. (mozilla is the outlier
wanting reach proportional to its 49 MB size; xml saturates at 4 MiB
because the file is 5 MB.)

**Finding 2: the dict must be scoped per SEGMENT, not per stream.** On the
concatenated blob, a global head dict captures almost nothing (−0.15pp —
the head is dickens, alien content for the samba/webster/nci blocks later
in the stream). Scoping the dict to each 32 MiB segment's own head
(`--seg` probe mode) recovers it:

| blob (202 MiB) | dict 4Mi | dict 16Mi |
|---|---|---|
| global head | −0.155 | −0.138 |
| per-32MiB-segment head | −0.310 | **−0.569 → 30.48%** |

30.48% is ~0.9pp under pzstd-3 (31.40%) — and this is exactly P2's
segment tier earning its place in the format.

### Production architecture (next build)

- **Format:** new pipeline id (`Pz2d`); the stream is a sequence of
  segments (~32 MiB), each `[dict_len: u32]` + framed blocks. Within a
  segment, blocks whose cumulative offset < dict_len are cold (they ARE
  the dict); later blocks may reference the dict. Segments are fully
  independent (P2's distribution unit). pz2 (id 13) streams unaffected.
- **Decode:** wave 1 decodes the dict blocks in parallel (cold), assembles
  one immutable `Arc<[u8]>`; wave 2 fans out the rest. Two engineering
  paths for priming, decided by measurement: (a) worker-local arenas —
  memcpy the dict once per WORKER (18 × 16 MiB ≈ 290 MB, one-time), splice
  blocks into the arena tail; or (b) two-region splice (match sources with
  offset > in-block position read from the dict slice; boundary-spanning
  copies split). Naive per-block priming is ruled out by arithmetic:
  94 blocks × 16 MiB ≈ 1.5 GB of memcpy ≈ +15-30 ms — would double the
  16.8 ms decode wall.
- **Encode: the frozen shared match-finder is BUILT and measured**
  (`lz77::FrozenDict` + `lzseq::tokenize_with_dict` +
  `pz2::encode_with_frozen_dict`): dict chains built once (immutable,
  `Arc`-shared, dict-relative coordinates), each worker holds a
  `dict‖block` arena (so frozen coordinates match and compares read one
  buffer) and parses starting at `dict_len` — no dict re-parse, no
  token-skipping/straddle handling. `find_best` grows one extra chain
  walk over the frozen tables after the block-local walk, sharing the
  chain budget. Probe `--frozen` (per-segment, blob): **identical ratio
  to the re-parse spike (30.475%) at 3× its encode speed** (70.5 s vs
  208.8 s ST). Remaining encode cost is the dict chain walks themselves
  (5.3× the no-dict baseline at 16 MiB) — a 32-byte weak-local-match
  gate was measured and rejected (−8% time, +0.018pp: most text
  positions have weak local matches, so the walk is inherent). Tuning
  levers for integration: dict-specific chain caps, sampled dict
  insertion, 4-8 MiB dicts (4 MiB: −0.31pp at 33.7 s).
- **Measured payoff (per-segment, 16 MiB dict, frozen finder):** blob
  31.04% → **30.48%**, ~0.9pp under pzstd-3 (31.4%), at unchanged decode
  parallelism.
- **What remains for shipping `Pz2d`:** container integration only —
  segment framing (`[dict_len] +` blocks), 2-wave parallel decode with
  the `Arc`-shared dict + worker arenas, encode-side worker arenas, and
  the encode-cost tuning pass above.

### §11b — Pz2d v1 SHIPPED (`-p pz2d`, id 14) and its honest ledger

Landed as segment-as-container-block: one container block = one 32 MiB
segment; payload = `[num_inner]` + inner frame table + pz2 wires. The
segment codec (`pz2::encode_segment`/`decode_segment`) encodes the dict
region as ONE parse split at 2 MiB boundaries (split match halves keep
their offsets — still valid against earlier region content) — this took
the 16 MiB-dict blob encode from 70.5 s (frozen probe, partial-dict
rebuilds) to **38.6 s ST**. Decode in the container is 2-wave per
segment: the dict region is a prefix chain (sequential), the remaining
blocks fan out across scoped threads against the completed region.

Blob, CLI e2e, measured:

| | ratio | dec wall | enc wall |
|---|---|---|---|
| pz pz2 | 31.04% | 16.8 ms | 3.15 s |
| **pz pz2d v1** | **30.48%** | 42.5 ms | 23.7 s |
| pzstd -3 -p18 | 31.40% | 22.9 ms | — |
| zstd -3 (1T) | 31.40% | 139 ms | — |

Position: best LZ-family ratio in pz (0.92pp under pzstd-3, 0.56pp under
pz2) at 3.3× faster decode than zstd ST — an opt-in max-ratio tier. Two
measured bottlenecks, both predicted by this section's arithmetic:

1. **Decode 42.5 ms is memory-traffic-bound, not compute-bound** (wall
   saturates at 4 threads): wave-2 still uses naive per-block priming —
   each block zeroes an 18 MiB buffer and memcpys the 16 MiB dict
   (~3.3 GB total traffic). The fix is the planned arena decode
   (`decode_into_arena`: dict stays resident per worker, splice appends)
   or the two-region splice; either should land near pz2-class walls
   (~17-20 ms projected from chain physics). Touches the proven unsafe
   splice → own session + fresh soak.
2. **Encode 23.7 s (112.8 s user vs 38.6 s ST probe — 2.9× concurrent
   inflation)**: 7 segments encoding in parallel each walk a 16 MiB dict
   + 64 MB frozen prev array — the 4 MiB-block cache lesson at segment
   scale. Levers: dict chain caps, sampled dict insertion, smaller
   dicts (4 MiB: −0.31pp at much lower walk cost).

### §11c — Pz2d v2: arena decode SHIPPED; the traffic diagnosis was wrong

The arena decode landed (`pz2::decode_into_arena`: the splice core takes
an arena already holding the dict, never writes below it, callers
truncate-and-reuse; `decode_with_prefix` is now a wrapper). Wave 2 uses
a **narrow** fan-out (`available_parallelism/4` workers per segment,
strided blocks) so each worker pays one dict copy amortized over its
share; wave 1 decodes the chain directly into the output vec (zero
prefix copies). One allocation gotcha cost 6 ms before being fixed:
fresh decode buffers must come from `vec![0; n]` (calloc → pre-zeroed
pages, lazy faulting), not `with_capacity` + `resize` (explicit memset,
double page touch) — `decode_into_arena` takes the calloc path when the
arena is fresh.

Result: **dec wall 42.1 → 36.6 ms (−13%) at unchanged 30.48%**, user
CPU 250 → 190 ms, sys 72 → 40 ms. pz2 (non-dict) is byte- and
wall-identical (17.8 ms both, re-anchored same-day). Soaked (release
180 s + 120 s, debug 45 s, zero panics).

**But §11b's projection (~17-20 ms) was wrong, and the reason matters:**
decode is **DRAM-latency-bound on the format's random dict reads, not
traffic-bound**. Evidence:

- ST per-segment decode is 16.8 ms (117.7 ms / 7 segments at `-t 1`) —
  exactly on model. Concurrent segments inflate it 2.4×: outer-thread
  sweep gives 71.3 ms (t=2) → 52.1 (t=4) → **40.6 (t=6) → 41.9 (t=7) →
  46.0 (t=18)** — a hard floor near 40 ms (now 36.6 with the copies
  gone) while aggregate decode sits at ~5 GB/s, far under the M5 Max's
  bandwidth. Classic latency wall, not bandwidth.
- Inner-worker sweep at the floor: W=4 → 40.8 ms, W=2 → ~64 ms (noisy),
  W=1 → 44.4 ms. W=1 is one dict copy per segment — equivalent traffic
  to a shared two-region splice — and it does NOT beat W=4. **The
  two-region splice is therefore predicted to be a no-op on wall and is
  not worth its unsafe complexity.** (Don't build it without new
  evidence.)
- Root cause: every match copy in a dicted/chain block is a random read
  into a ~16-18 MiB region; 7 concurrent segments make the hot set
  ~112-450 MiB ≫ SLC, so those reads are DRAM-latency misses. That is
  the price of the dict reach that buys the ratio.
- Confirmation via dict size: a 4 MiB-dict build (format const flip,
  same code) measures **30.73% / 31.3 ms dec / 20.1 s enc** — smaller
  hot set, shorter chain, faster wall, −0.26pp ratio. A real point on
  the (ratio, decode) curve if a dict-size header field is ever added;
  16 MiB stays shipped because pz2d is the max-ratio tier.

Remaining decode headroom would need format-level changes (smaller/
tiered dict reach, locality-sorted matches) — incremental copy
elimination is exhausted. Encode-side concurrent inflation (§11b #2)
is unchanged and is the next lever.

### §11d — Encode: dict chain cap (DICT_CHAIN_CAP = 16), −44% wall

The §11b encode lever, executed. Frozen-dict walks in `find_best` now
get their own link budget (`lz77::DICT_CHAIN_CAP`), applied on top of
the live walk's leftover `max_chain`. Blob e2e sweep (encode wall /
user / ratio):

| cap | enc wall | user | ratio |
|---|---|---|---|
| 64 (= old shared budget) | 25.0 s | 115 s | 30.475% |
| 32 | 18.6 s | 91 s | 30.500% |
| 24 | 16.4 s | 82 s | 30.515% |
| **16 (shipped)** | **14.1 s** | **69 s** | **30.535%** |
| 8 | 12.65 s | 61 s | 30.570% |

No sharp knee; 16 is the judgment call — **−44% encode wall and −40%
CPU for +0.06pp**, keeping pz2d 0.87pp under pzstd-3. Below 16 the
returns invert (8 buys only −10% more wall for +0.035pp more). The cap
attacks the §11b inflation at its source: the inflation IS the dict
walk's random reads, so walking less is the fix.

Decode is **neutral-to-better** on the capped parse (same-session
hyperfine: 39.0 ms for cap-16 wire vs 40.5 ms cap-64 — fewer far-dict
matches means fewer random dict reads). Same-session re-anchor of
§11c's headline: master v1 44.9 ms vs arena 39.1 ms (1.15×) — the
−13% holds; absolute walls drift ±10% with machine state, so compare
binaries within ONE hyperfine invocation only.

## §12 — tANS entropy probe: per-block tANS PASSES the gate; global tables and order-1 are DEAD (2026-06-10)

The question (asked when the dict tier shipped): *would tANS or another
entropy coder beat the 8-lane Huffman if we had global state tables?*
Answered by pure entropy accounting (`examples/pz2_entropy_probe.rs` +
`pz2::probe_lane_streams` hook) — bit-exact pricing of the shipped coder
(package-merge lengths, real headers, CONST/RAW fallbacks) against three
alternatives on the exact lane streams the encoder feeds its Huffman
lanes (2 MiB blocks, greedy parse, 32 MiB segments for global scenarios):

- **A** shipped per-block Huffman
- **B** per-block tANS (Shannon ideal + A's own header — isolates the
  fractional-bit win, which is all tANS adds over optimal Huffman)
- **C** segment-global tANS tables (cross-entropy vs segment histogram,
  amortized header; C' = per-block min(A,C) + mode byte)
- **D** order-1 (prev symbol) segment-global, seq-code lanes only

Results (Δpp of input vs A; negative = smaller):

| input | A total | B tANS/blk | C global | C' choice | D o1/seg |
|---|---|---|---|---|---|
| **blob** | 20.749pp | **−0.255** | **+0.894** | −0.066 | −0.055 |
| dickens | 13.580pp | −0.548 | −0.546 | −0.546 | −0.508 |
| webster | 11.853pp | −0.307 | −0.310 | −0.310 | −0.299 |
| mozilla | 28.129pp | −0.206 | +0.200 | −0.116 | −0.406 |
| sao | 66.365pp | −0.432 | −0.386 | −0.386 | −0.467 |
| xml | 5.100pp | −0.125 | ~0.000 | −0.083 | −0.098 |

Blob lane split for B: ll −0.114, lit −0.059, ml −0.054, of −0.027 —
the win lives in the small-alphabet sequence-code lanes, exactly the
predicted Huffman-1-bit-floor effect (greedy parses make `lit_run = 0`
dominate the ll lane far past p=0.5; Huffman cannot pay less than 1 bit
for it, tANS can). The 8-lane Huffman literals are near-optimal
(−0.06pp headroom — huff0's classic result, reconfirmed).

**Verdicts:**

1. **Per-block tANS passes the ratio gate: −0.25pp blob, −0.13 to
   −0.55pp per file, no global tables needed.** Converting only the
   three seq-code lanes nets ~−0.20pp; literals can stay Huffman.
2. **Global tables are DEAD — the hypothesis inverted.** On the
   heterogeneous blob, segment-global tables are +0.89pp (WORSE than
   shipped); 2 MiB per-block histograms are already statistically
   saturated, so sharing buys nothing on homogeneous files and actively
   hurts on mixed segments. Same lesson as the global head dict (§11).
3. **Order-1 segment-global seq-code modeling is DEAD at blob scope**
   (−0.055pp < gate). One honest outlier: mozilla D = −0.41pp (real
   order-1 structure in executable-heavy seq streams) — recorded, not
   actionable alone.

**Caveats before anyone builds it:** B prices Shannon ideal with
Huffman-equal headers; a real tANS at 2^11-12 states with normalized-
count headers lands ~−0.20pp on the blob, right at the gate. And the
binding constraint is DECODE: the fused splice would swap three Huffman
chain reads for three interleaved FSE state updates (zstd's exact
design — proof it can be fast), but pz2's 12 GiB/s all-cores wall and
1.4 GB/s ST must hold. A wire change only proceeds if a decode
prototype is speed-neutral; that is its own gated task, not a rider.

## §13 — CODES_FSE shipped: per-block tANS on the seq-code lanes, −0.18/−0.19pp at decode parity (2026-06-10)

The §12 candidate, built and gate-checked. The three sequence-code
streams (ll/of/ml) gained a third wire mode alongside CONST/HUFF:

- **CODES_FSE = 2**: textbook tANS, 32-symbol alphabet, `FSE_LOG = 10`
  (1024 states; 4 KB u32 decode table per lane, 12 KB total next to the
  4 KB literal table — comfortably L1-resident).
- Header: 32 × u16 normalized counts (Σ = 1024) + u16 initial state +
  u32 lane length = 71 B/lane vs Huffman's 21 B. Noise at 2 MiB blocks.
- **Forward-readable payload**: the encoder walks symbols in reverse
  (tANS requirement) but then writes the emitted bit groups re-reversed,
  so the decoder reads the stream FORWARD with the same LSB-first
  whole-byte-refill (`LaneState`) discipline as the Huffman lanes. No
  backward bitstream, no sentinel byte, no second reader implementation.
- **Entry prefetch is what bought decode parity**: the lane caches
  `table[state]` and each `next()` issues the *following* symbol's table
  load at its end, overlapping the load latency with the splice copies.
  Without it the fused loop measured +2.4% ST / +3.5% MT; with it ST is
  neutral-to-faster. (First version measured before/after in one
  session: 138.8 → 137.0 ms ST vs master's 137.9.)
- The encoder builds BOTH candidates per stream and ships the
  byte-smaller, so the wire is bit-exact never-worse: CONST still wins
  all-rep0 offset lanes (2 bytes), HUFF still wins near-uniform and
  tiny streams, FSE wins the skewed bulk. Old streams decode unchanged.

Measured (M5 Max, /tmp/silesia.blob, same-session hyperfine pairs):

| metric | master | FSE branch | Δ |
|---|---|---|---|
| pz2 ratio | 31.044% | **30.863%** | **−0.181pp** |
| pz2d ratio | 30.535% | **30.341%** | **−0.194pp** |
| pz2 decode MT | 17.1–17.4 ms | 17.5–17.7 ms | +2–3% (≈0.4 ms; E-core side) |
| pz2 decode ST | 137.9 ms | 137.0 ms | neutral (−0.7%) |
| pz2d decode MT | 33.1 ms | 33.6 ms | neutral (+1.5%, within σ) |
| pz2 encode | 3.08 s | 3.14 s | +1.9% (dual-candidate pricing) |

The realized −0.18pp matches §12's quantization-adjusted prediction
(~−0.20pp) almost exactly. The residual MT cost is order-independent
across hyperfine runs and absent at ST on a P-core, so it is most
likely the E-cores paying slightly more for the extra μops; at ≈0.4 ms
on a 17 ms wall it is inside the documented ±10% machine-state drift
band and far under the 3–5% kill line. pz2 headline becomes **blob
30.86% at ~11.5–12 GiB/s all-cores**; pz2d becomes **30.34%** — 1.06pp
under pzstd-3, the widest the Pareto edge has been.

Validation: fresh soak after the wire change (release 180 s: 19,176
round-trips + 767k mutated + 153k garbage decodes; debug 45 s; zero
panics), blob round-trips verified for both pipelines, FSE-specific
unit tests (mode selection, all-32-symbol normalization stress,
per-byte corruption + truncation never panicking).

Not pursued: FSE for the 8-lane literals (§12 measured −0.06pp
headroom — not worth touching the proven splice's literal path), and
interleaving the three FSE states into one bitstream (zstd's layout;
separate lanes already decode at parity, so the only win would be
stream-count bookkeeping, not speed).
