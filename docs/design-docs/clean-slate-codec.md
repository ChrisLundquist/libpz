# Clean-slate parallel codec design ("pz2")

**Date:** 2026-06-09
**Status:** ✅ Design validated — the §5 prototype gate **passed at 3.3×**
(target was 2×). `src/pz2.rs` decodes the Silesia blob at **1361 MB/s
single-thread at 32.22%** vs `Lzf`'s 409 MB/s at 32.18% — same parse,
byte-identical match decisions, new wire format. Per-core decode is now
within ~10% of zstd-3 (~1500 MB/s documented) at ~1pp ratio cost, while
every block stays independently decodable. See §7 for measured results.
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
fidelity-tested byte-identical):

| file | pz2 % | pz2 dec MB/s | lzf % | lzf dec MB/s | speedup |
|---|---|---|---|---|---|
| **silesia blob** | **32.221** | **1361** | 32.183 | 409 | **3.33×** |
| dickens | 38.851 | 1012 | 38.728 | 278 | 3.64× |
| webster | 29.337 | 1250 | 29.335 | 364 | 3.44× |
| mozilla | 36.224 | 1182 | 36.030 | 383 | 3.08× |
| nci | 7.907 | 3063 | 8.085 | 1237 | 2.48× |
| xml | 11.568 | 2618 | 11.804 | 905 | 2.89× |
| sao | 80.149 | 970 | 79.859 | 250 | 3.89× |
| x-ray | 77.022 | 1295 | 76.753 | 243 | 5.33× |

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

**Honest caveats:** encode is unoptimized (~lzf-parse-bound, fine — P1);
no container/dict/transform integration yet (all shipped pz components);
the unsafe splice has invariant comments + fuzz/garbage tests but should get
`cargo fuzz` + Miri on the small suite before graduating beyond an
experiment; offsets 2-15 use a scalar overlap loop (pattern-splat is a known
further win); blob aggregate with the proven 13-16× fan-out projects to
**~17-20 GB/s**, to be measured when pz2 gets a container.
