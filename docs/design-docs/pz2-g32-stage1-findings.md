# pz2-G32 stage-1 findings: 32-lane literal relayout costs nothing in ratio

**Date:** 2026-06-10
**Status:** Stage 1 **PASS** — aggregate Silesia ratio delta **+0.0066pp**
against a 0.5pp kill line (75x under). Round-trip byte-exact on all 12
files. Stage 2 (Metal literal-phase kernel) is unblocked.
**Context:** First gated stage of the pz2-G32 candidate
(`gpu-path-research.md` §2 #1): can pz2's literal wire be re-laid into a
GDeflate-style 32-lane lane-pinned layout — the shape a Metal simdgroup
decodes at 32 symbols/round — without paying ratio? This stage is
deliberately zero-GPU: a transcoder + scalar CPU decoder that measures the
**format cost** deterministically, per the report's instruction to base the
verdict only on byte counts (shared machine, noisy timing).

## 1. What was built

`src/pz2.rs` (branch `claude/spike-pz2-g32`, spike API — not a shipping
wire):

- **`transcode_g32(block)`** — parses a shipped pz2 block, decodes the
  8-lane literal section, and re-emits it in the G32 layout. The canonical
  Huffman table (128B packed lengths) is reused **verbatim** and the entire
  sequence section is copied **byte-identical**, so the size delta is purely
  the cost of the GPU-friendly framing.
- **`decode_g32(data, orig_len)`** — scalar decoder for the new layout; the
  sequence splice loop was factored out of `decode()` and is shared by both
  decoders (one copy of the unsafe wildcopy code).
- **`examples/pz2_g32_probe.rs`** — per-file size delta, byte-exact
  round-trip verification, and ST decode timing over Silesia, using the
  shipped `-p pz2` recipe (2 MiB blocks, window = block, auto-greedy parse).
- Unit tests: round-trip suite incl. lane-round boundary sizes (31/32/33,
  1023/1024/1025), periodic-overlap inputs, splice-of-history fuzz, and
  truncation/bit-flip garbage rejection through `decode_g32`.

### The G32 layout (as implemented)

- Literal `i` is pinned to lane `i % 32`; each decode round produces 32
  **contiguous** output bytes (one simdgroup per tile on GPU; also the
  reason scalar writes stay cache-friendly).
- Each lane's Huffman codes are packed LSB-first into 32-bit words.
- The shared word stream interleaves lane words in **exact decode-read
  order**: the encoder simulates the decoder's deterministic refill schedule
  — before each symbol, a lane holding < 32 live bits fetches one word
  (lane order within the round). Since a literal symbol consumes
  ≤ `MAX_CODE_LEN` = 11 bits, a lane consumes **at most 1 word per round**,
  inside the GDeflate ≤2-words-per-round budget.
- Lane word counts are **implicit** (the decoder replays the same
  schedule). Fetches past a lane's real data are zero padding words emitted
  by the encoder (≤ ~2 per lane beyond `ceil(lane_bits/32)`).
- Literal-section wire: `[128B packed lengths][word_bytes: u32][words]`
  (replacing pz2's 8 × u32 lane lengths + 8 byte-aligned lanes). `LIT_RAW`
  blocks pass through unchanged. Header/seq sections identical.

## 2. PRIMARY result: compressed-size delta (deterministic)

Probe: `cargo run --release --no-default-features --example pz2_g32_probe
-- --reps 7 samples/silesia/*`. Shipped recipe (2 MiB blocks, auto-greedy).

| file | orig | pz2 B | g32 B | pz2 % | g32 % | Δpp |
|---|---|---|---|---|---|---|
| dickens | 10192446 | 3597645 | 3598349 | 35.297 | 35.304 | +0.0069 |
| mozilla | 51220480 | 18409382 | 18412316 | 35.941 | 35.947 | +0.0057 |
| mr | 9970564 | 3618473 | 3619145 | 36.292 | 36.298 | +0.0067 |
| nci | 33553445 | 2562062 | 2564354 | 7.636 | 7.643 | +0.0068 |
| ooffice | 6152192 | 3088253 | 3088636 | 50.198 | 50.204 | +0.0062 |
| osdb | 10085684 | 3475126 | 3475769 | 34.456 | 34.462 | +0.0064 |
| reymont | 6627202 | 1831095 | 1831632 | 27.630 | 27.638 | +0.0081 |
| samba | 21606400 | 4980374 | 4981893 | 23.050 | 23.057 | +0.0070 |
| sao | 7251944 | 5634801 | 5635281 | 77.701 | 77.707 | +0.0066 |
| webster | 41458703 | 11250109 | 11252871 | 27.136 | 27.142 | +0.0067 |
| x-ray | 8474240 | 6454569 | 6455224 | 76.167 | 76.175 | +0.0077 |
| xml | 5345280 | 582245 | 582652 | 10.893 | 10.900 | +0.0076 |
| **aggregate** | **211938580** | **65484134 (30.8977%)** | **65498122 (30.9043%)** | | | **+0.0066** |

**VERDICT: PASS.** +0.0066pp aggregate vs the 0.5pp kill line; worst file
is reymont at +0.0081pp. There are no outliers because the cost is pure
framing arithmetic: +13988 bytes over ~106 blocks ≈ **+132 bytes/block**,
of which −28 B header savings (one u32 stream length vs eight u32 lane
lengths), ~+60 B expected in-word padding (32 lanes × ~16 bits avg vs 8
lanes × ~3.5 bits), and ~+100 B schedule padding words. The cost is a
per-block constant, so it shrinks further at larger blocks and is
independent of content. **The wire-format question is settled: ratio is
not a reason to avoid the G32 layout.**

Round-trip: every block of all 12 files decoded byte-exact through
`decode_g32` (and through shipped `decode` as a control). Full
`--no-default-features` test suite: 595 passed.

## 3. SECONDARY result: scalar ST decode (indicative only — shared box)

Median of 7 reps, full-file decode (entropy + splice), single thread.
Within-run spread was tight (min–max ≤ ~3%) but the machine runs
concurrent agent builds; treat as indicative.

| file | pz2 MB/s | g32 MB/s | g32/pz2 |
|---|---|---|---|
| dickens | 1361 | 1250 | 0.92x |
| mozilla | 1250 | 996 | 0.80x |
| mr | 1494 | 1134 | 0.76x |
| nci | 3581 | 3437 | 0.96x |
| ooffice | 1019 | 706 | 0.69x |
| osdb | 1857 | 1252 | 0.67x |
| reymont | 1513 | 1436 | 0.95x |
| samba | 1940 | 1581 | 0.81x |
| sao | 1125 | 578 | 0.51x |
| x-ray | 1360 | 615 | 0.45x |
| webster | 1500 | 1365 | 0.91x |
| xml | 3106 | 2831 | 0.91x |

The scalar penalty correlates exactly with literal share: structured files
(nci, xml, reymont — splice-dominated) lose ≤ 9%; literal-heavy files
(sao, x-ray — the decode is almost all entropy phase) lose ~2x. Causes,
in expected order: (a) 32 live lane states = 32 × (u64 acc + u32 nbits) ≈
384 B — far past the register file, so per-symbol state spills, where
pz2's 8 lanes stay in registers; (b) a conditional refill branch per
symbol vs pz2's branchless 5-symbols-per-refill rounds; (c) bounds-checked
word fetches from the shared stream. None of this is wire cost — it is
"32-wide format decoded 1-wide" cost.

## 4. What stage 1 teaches stage 2 (Metal literal-phase kernel)

- **Word-order = lane-order-within-round is GPU-ready as emitted.** The
  refill schedule maps to a simdgroup ballot: each lane's fetch index is
  `base + popcount(refill_mask & lanes_below_me)`; `simd_prefix_exclusive_sum`
  / ballot on Metal gives this in two instructions. No per-lane offsets
  table is needed — exactly why the boundaries were made implicit.
- **Only 1 of the ≤2 word budget is used.** A literal symbol is ≤ 11 bits,
  so even decoding **two** symbols per lane per round (64 symbols/round)
  stays ≤ 22 bits < 32 → still ≤ 1 word/round. The second word of budget is
  real headroom for fusing sequence extra-bits or wider symbols into lanes
  later; alternatively a 2-symbols/round kernel halves round count for free.
- **The flat 2048 × u16 decode table is 4 KB** — fits threadgroup memory
  with room to spare; one cooperative load per block.
- **`LIT_RAW` blocks must be handled** (incompressible literal sections
  pass through unchanged) — on GPU that phase is a plain memcpy.
- **Tail rounds are trivial**: the last `< 32` literals follow the same
  schedule with inactive high lanes (encoder and decoder agree because lane
  activity depends only on `lit_total`).
- **The honest stage-2/3 denominator is still open.** The scalar 0.45–0.96x
  confirms the research report's warning: a format designed for 32-wide
  decode is *not* automatically fast 1-wide, so a unified single wire would
  tax the CPU fallback up to ~2x on literal-heavy data unless a NEON
  4×u32-lane decoder recovers it. Before committing the wire (stage 3
  gate), measure a NEON/SIMD CPU decoder of this exact layout — it may
  *beat* the shipped 8-lane decoder and is the correct baseline for any GPU
  number.
- **Padding-word rule must be part of any spec**: fetch-when-below-32
  including past-end zero words. It is what keeps boundaries implicit; an
  implementation that clamps fetches instead will desynchronize.

## 5. Verdict and next step

**PASS.** Stage 1's question — "does the GDeflate-shaped wire cost ratio?"
— is answered: **no** (+0.0066pp, a per-block constant ~132 B). Proceed to
stage 2 per `gpu-path-research.md` §5: a single Metal kernel decoding the
literal Huffman phase of ~100 repacked tiles from persistent buffers, kill
if < ~5 GB/s effective — with the stage-3 cooperative-splice gate (and the
NEON-of-this-wire baseline) still standing between any pass and a wire
commitment.

Artifacts: `pz2::transcode_g32` / `pz2::decode_g32` (+ shared `splice`
refactor) in `src/pz2.rs`; `examples/pz2_g32_probe.rs`; branch
`claude/spike-pz2-g32`.
