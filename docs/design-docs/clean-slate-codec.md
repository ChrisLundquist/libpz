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
