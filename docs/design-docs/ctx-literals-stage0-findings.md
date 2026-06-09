# Order-1 context-bucketed literals — Stage-0 spike findings (dead end)

**Date:** 2026-06-09
**Status:** ❌ Gate failed. The Brotli-style context-literal path is dead as a
ratio lever for pz's LZ pipelines, and per the
[future-pipelines roadmap](../design-plans/2026-06-09-future-pipelines-roadmap.md)'s
own decision rule ("only if a cheap order-1-literals-into-existing-rANS
measurement shows >2-3pp on text"), the LZMA-class range-coder family stays
parked.
**Tool:** `examples/ctx_literals_probe.rs` (rerunnable).

## Question

The roadmap's LZMA-class assessment claimed context-modeled literals are "the
real lever" worth ~2-3pp+ on text, and gated the whole family on a cheap
Stage-0 measurement: bucket the **post-LZ literal stream** by the previous
output byte (available to any LZ decoder at literal-emit time) and entropy-code
each bucket with the **existing** FSE — no new coder, no wire change.

## Method

Per 1 MiB block (the shipped block size): lazy-parse, collect each literal
with its previous original byte, then compare per-block:

- **base** — one `fse::encode_best` stream (what `lzf` ships today)
- **C4/C16/C64/C256** — per-(prev>>6 / >>4 / >>2 / prev) bucket FSE, real
  table headers included
- **idealO0/idealO1** — header-free entropy and conditional entropy
  H(lit | prev): the ceiling for ANY prev-byte context scheme

The parse is `lz77::compress_lazy_to_matches` (32 KiB window), which leaves
**more** surviving literals than the shipped 1 MiB-window lzseq parse
(14-19% literal share here) — so every number below is an **optimistic
bound**. A fail here is a fail everywhere.

## Results (pp of input saved vs base; positive = smaller)

| File | lit share | C4 | C16 | C64 | C256 | idealO1 ceiling |
|---|---|---|---|---|---|---|
| dickens | 18.8% | +0.49 | +0.49 | −0.02 | **−2.23** | +1.84 |
| webster | 13.9% | +0.23 | +0.34 | −0.26 | **−2.94** | +1.70 |
| reymont | 15.0% | +0.33 | +0.63 | +0.15 | **−2.78** | +2.68 |
| samba | 15.5% | +0.11 | +0.23 | −0.88 | **−5.67** | +1.53 |
| xml | 6.5% | +0.14 | +0.11 | −0.74 | **−3.91** | +1.12 |
| nci | 4.3% | −0.00 | −0.16 | −0.85 | **−2.39** | +0.33 |
| x-ray | 26.6% | +0.47 | +0.04 | −2.12 | **−12.36** | +1.77 |
| sao | 50.6% | +0.41 | +0.47 | −1.03 | **−9.28** | +3.57 |
| mozilla | 22.2% | +0.13 | −0.05 | −1.89 | **−10.69** | +2.03 |
| osdb | 23.6% | +0.40 | +0.36 | −1.52 | **−10.56** | +2.63 |

Binary files follow the identical pattern (best realized ≤ +0.47pp, C256
loses 9-12pp). sao has the largest ideal ceiling (+3.6pp) — but `pz -a` now
routes sao to `Num` (PR #138, −21pp), so even that ceiling is moot.

## Why it fails (three stacked reasons)

1. **Realized gain is ~0.1-0.6pp, not 2-3pp.** The best static operating
   point (16 coarse contexts) nets less than the gate by 4-10×. This
   *confirms* the repo's older ~0.5pp estimate for order-1 literals and
   refutes the research-side 2-3pp hope.
2. **The information-theoretic ceiling is below the gate.** Perfect
   prev-byte conditioning with *free* tables caps at +1.1-2.7pp on text
   (idealO1). No context map of any granularity can beat that; finer maps
   (2-byte Brotli-style) only chase the same bounded information with even
   worse header economics at 1 MiB block scale.
3. **Header economics invert fine contexts.** C256 *loses* 2.2-5.7pp — the
   per-block, per-bucket FSE table descriptions cost 3-7pp, swamping the
   conditional-entropy win (same effect as Num's 28 plane headers at 64KB
   sample scale). The only way to dodge table costs is adaptive per-bit
   probability modeling — exactly the latency-bound decoder class this repo
   has measured dead (BWT+CM: 11 MB/s/core; see
   [bwt-cm-findings](bwt-cm-findings.md)).

Literals that survive LZ matching are, definitionally, the
hard-to-predict residue; on Silesia text the prev-byte tells you ~0.8-1.4
bits about the next literal in the ideal, and a static, block-scale,
byte-aligned coder can monetize only a fraction of that.

## What remains live

- This spike does NOT touch the **BWT+CM** family (deferred on its own
  decode-wall terms, not on literal-context grounds).
- The FSE/rANS infrastructure is untouched; no wire or code changes shipped
  — this was a measurement-only spike.
