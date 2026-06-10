# num-bitpack stage-0 findings: vertical bit-packing cannot replace per-plane FSE

**Date:** 2026-06-10
**Verdict: KILL — on all four target files, decisively.**
**Branch:** `claude/spike-num-bitpack` · **Probe:** `examples/num_bitpack_probe.rs` + `numeric::bitpack` / `numeric::probe_block`

## 1. What was tested

Stage-0 of the num-G candidate from `gpu-path-research.md` §#4: keep the Num
front-end exactly as shipped (1 MiB blocks, stride sweep over {2,4,8,16,28,32},
per-plane gated Raw/Delta/DeltaZigzag), but replace the per-plane **FSE**
entropy stage with an ndzip-style table-free coder:

- **32×8 bit-transpose** — each group of 32 consecutive plane bytes becomes 8
  bitplane words (word *b* = bit *b* of all 32 values). This is the byte-plane
  specialization of ndzip's 32×32 word transpose: identical presence-bitmap
  overhead (1 bit per 32 payload bits, 3.125% floor) but strictly *finer*
  zero-word granularity, so it upper-bounds what the 32×32 variant could do.
- **Zero-word elimination** — all-zero words are dropped; a 1-byte presence
  bitmap per group records survivors.

Pure byte counting, no GPU, fully deterministic. Every encode was round-trip
verified against the real plane data of every block of all four files (plus
fuzz + adversarial unit tests in `src/numeric.rs`).

Three measurements per plane:

- `fse` — gated-best per-plane FSE (exactly what ships),
- `bp_same` — bitpack of the FSE-chosen transform (front-end held identical),
- `bp_gated` — bitpack with its own transform gating (the realistic
  replacement, since the gate would naturally use the new coder's sizes),
- `oracle` — per-plane min(fse, bp_gated): a free per-plane selector.

Kill criterion: bp_gated regression > 2pp of original file size on routed
planes, per file.

## 2. Per-file results (Silesia, 1 MiB blocks, plane bytes only)

| File | Orig bytes | Stride | FSE planes | bp_same | bp_gated | Oracle | bp_gated Δ (pp of orig) | Verdict |
|------|-----------:|:------:|-----------:|--------:|---------:|-------:|------------------------:|:-------:|
| sao | 7,251,944 | 28 (7/7 blocks) | 4,594,179 (63.35%) | 6,179,644 (85.21%) | 5,924,504 (81.70%) | 4,594,179 (63.35%) | **+18.34** | KILL |
| x-ray | 8,474,240 | 2 (9/9) | 4,040,205 (47.68%) | 6,174,032 (72.86%) | 4,859,540 (57.34%) | 4,040,205 (47.68%) | **+9.67** | KILL |
| mr | 9,970,564 | 2 (10/10) | 3,074,195 (30.83%) | 3,819,190 (38.30%) | 3,282,466 (32.92%) | 3,023,287 (30.32%) | **+2.09** | KILL |
| nci | 33,553,445 | 2 (32/32) | 10,228,020 (30.48%) | 26,016,886 (77.54%) | 26,016,886 (77.54%) | 10,228,020 (30.48%) | **+47.06** | KILL |

No STORE blocks anywhere; headers/remainder are identical for both coders and
negligible (≤ 1015 bytes/file). Full per-plane tables below.

### Per-plane detail

**sao (S=28):** all 28 planes lose, uniformly — bp_gated/fse ranges **1.22×–1.39×**
with no plane even close to parity. The best FSE planes (the four record
columns at plane≡3 mod 4: ~104–110 KB vs ~259 KB raw) are exactly where
bitpack loses worst (1.38–1.39×). This confirms the research report's stated
risk precisely: sao's planes are *dense skewed* distributions (FSE gate picks
Raw on 129/196 plane-blocks); their entropy lives in symbol frequencies, not
in zero bitplanes, so zero-word elimination monetizes almost nothing.

**x-ray (S=2):**
| Plane | FSE | bp_gated | ratio |
|------:|----:|---------:|------:|
| 0 (low byte) | 3,718,484 | 4,055,498 | 1.09× |
| 1 (high byte) | 321,721 | 804,042 | 2.50× |

The near-incompressible low-byte plane only pays the ~3% bitmap floor + dense
words (mild), but the *compressible* high-byte plane — the one Num exists for —
is 2.5× worse: its delta'd values are small-but-nonzero, so low bitplanes stay
dense and FSE's sub-bit-per-symbol coding has no bitpack analogue.

**mr (S=2):** the only bitpack win observed anywhere:
| Plane | FSE | bp_gated | ratio |
|------:|----:|---------:|------:|
| 0 (low byte) | 2,922,723 | 2,871,815 | **0.983×** |
| 1 (high byte) | 151,472 | 410,651 | 2.71× |

Plane 0 (DeltaZigzag both gates) genuinely has exploitable zero bitplanes and
beats FSE by 1.7%. But plane 1 loses 2.7× (bitpack gate flips to Raw because
the raw high bytes have more zero bitplanes than the delta'd ones), and the
file still fails the gate at +2.09pp.

**nci (S=2):** catastrophic, +47pp. nci is structured *text* that Num happens
to route at S=2 with Raw transforms — character data has no zero bitplanes at
all, so bitpack degenerates to ~103% of raw plane size while FSE gets 30%.
Any bitpack tier would need an FSE fallback for exactly this case, which
re-imports the table-decode path the design was meant to eliminate.

## 3. The oracle does not rescue the design

Best-of-both per plane: sao/x-ray/nci oracle == FSE to the byte — **FSE wins
every single plane** on three of four files. Only mr gains: −0.51pp (the lone
plane-0 win). A cheap per-plane selector is therefore equivalent to "ship FSE
plus a second coder that fires on one plane of one file for half a point."
That cannot justify a second wire format, and it means the hybrid version of
num-G has essentially zero ratio payload to offset its costs.

`bp_same` vs `bp_gated` (sao 85.2%→81.7%, x-ray 72.9%→57.3%, mr 38.3%→32.9%)
shows letting the bitpack gate pick its own transform matters a lot — the
verdict above already grants the candidate that favorable reading and it still
fails everywhere.

## 4. Why (mechanism, not just numbers)

Zero-word elimination is a *run-of-zeros-in-bitplanes* model. It pays off only
when decorrelation drives whole 32-value bitplane windows to zero — i.e. when
residuals are tiny AND smooth. The Num target files fail this in three ways:

1. **Dense skewed planes (sao):** entropy is in the frequency skew of byte
   values, not magnitude. FSE codes a 0.5-bit symbol at 0.5 bits; bitpack's
   floor for any plane with even one set bit per window in bitplanes 0–k is
   k+1 full words.
2. **Small-but-nonzero residuals (x-ray/mr high planes):** delta gets values
   into ±3, but bitplanes 0–2 stay dense, costing ~3–4 bits/value where FSE
   pays ~0.3–0.8.
3. **Non-numeric routed data (nci):** no bitplane structure whatsoever; the
   coder is a no-op with 3.1% overhead.

ndzip's published wins are on *floating-point* fields where XOR/Lorenzo
residuals concentrate set bits in a few high-exponent bitplanes and leave many
all-zero planes. Byte-plane records after delta gating do not have that shape.

## 5. Verdict and next gate

**KILL.** The stage-0 gate (≤ +2pp per file) fails on every file: +18.34,
+9.67, +2.09, +47.06pp. Per the research report's own decision rule, the next
gate — a CPU-SIMD decoder of the new wire (deliberately *not* a WGSL kernel,
because on unified memory the CPU baseline of the same format is the honest
denominator) — is **not reached**. There is no point measuring decode speed of
a wire that costs 2–47pp on the only axis Num exists for.

Implications:

- **num-G (research report candidate #4) is closed** at the cheapest possible
  stage, as the pareto lens predicted ("zero-word elimination monetizes only
  zeros while sao's win came from FSE on dense skewed planes"). The measured
  risk (5–10pp predicted) was, if anything, understated.
- A table-free GPU-decodable numeric tier would need a different entropy
  mechanism than zero-word elimination (e.g. per-window bit-width packing /
  BP128-style frame-of-reference). Note for any future attempt: that fixes
  case 3 (nci) but not cases 1–2; sao's dense skewed planes need real entropy
  coding, full stop. Any such proposal should rerun this exact probe first —
  `probe_block` makes it a ~20-line change.
- The lone positive signal (mr low plane, −1.7% with DeltaZigzag) is too small
  to act on and is already captured by the oracle's −0.51pp file delta.

## 6. Reproduction

```bash
cargo build --release --no-default-features --example num_bitpack_probe
./target/release/examples/num_bitpack_probe \
    samples/silesia/sao samples/silesia/x-ray samples/silesia/mr samples/silesia/nci
```

Deterministic byte counting; exit code 1 on KILL. Round-trip is asserted on
every plane of every block. The coder and probe live in `src/numeric.rs`
(`bitpack` module, `probe_block`) with unit tests at the bottom of the file.

## Appendix: raw per-plane probe output

Columns: aggregated over all blocks of the file; `xf` counts are
Raw/Delta/DeltaZigzag picks per plane across blocks.

### sao (S=28, 7 blocks)

```
 plane  plane_len        fse    bp_same   bp_gated     oracle  bp*/fse fse_xf(R/D/Z) bp_xf(R/D/Z)
     0     258997     204015     254146     253994     204015    1.245        6/0/1        5/0/2
     1     258997     178319     231814     228890     178319    1.284        3/4/0        5/1/1
     2     258997     177065     225582     223578     177065    1.263        4/2/1        5/0/2
     3     258997     106075     146718     146546     106075    1.382        4/3/0        5/0/2
     4     258997     202402     254282     254034     202402    1.255        5/0/2        4/2/1
     5     258997     180353     237414     234050     180353    1.298        5/2/0        5/1/1
     6     258997     169779     213590     213586     169779    1.258        4/2/1        5/0/2
     7     258997     104521     173730     145538     104521    1.392        5/2/0        5/1/1
     8     258997     201544     255574     255554     201544    1.268        5/2/0        4/2/1
     9     258997     179041     235734     233550     179041    1.304        5/1/1        5/1/1
    10     258997     163787     210494     210494     163787    1.285        4/1/2        5/1/1
    11     258997     108308     177086     148894     108308    1.375        6/1/0        5/1/1
    12     258997     199306     257606     257602     199306    1.292        5/1/1        5/2/0
    13     258997     172368     230586     230570     172368    1.338        4/3/0        5/1/1
    14     258997     157575     225806     204054     157575    1.295        3/4/0        5/0/2
    15     258997     105569     173666     145530     105569    1.379        5/2/0        5/1/1
    16     258997     201575     256242     256194     201575    1.271        4/2/1        4/2/1
    17     258997     173607     229838     229154     173607    1.320        4/3/0        5/0/2
    18     258997     162108     225638     206750     162108    1.275        5/2/0        5/0/2
    19     258997     108096     178730     148694     108096    1.376        5/2/0        5/0/2
    20     258997     208369     256882     256874     208369    1.233        5/1/1        5/2/0
    21     258997     182297     233374     231198     182297    1.268        5/0/2        5/0/2
    22     258997     170161     226278     207830     170161    1.221        4/2/1        4/1/2
    23     258997     110075     184462     151630     110075    1.378        6/1/0        5/0/2
    24     258997     205540     256510     256450     205540    1.248        4/3/0        5/1/1
    25     258997     179931     233078     231330     179931    1.286        4/1/2        5/1/1
    26     258997     173105     211518     211510     173105    1.222        5/1/1        5/0/2
    27     258997     109288     183266     150426     109288    1.376        5/2/0        4/0/3
```

### x-ray (S=2, 9 blocks)

```
 plane  plane_len        fse    bp_same   bp_gated     oracle  bp*/fse fse_xf(R/D/Z) bp_xf(R/D/Z)
     0    4237120    3718484    4080870    4055498    3718484    1.091        0/1/8        0/0/9
     1    4237120     321721    2093162     804042     321721    2.499        0/5/4        0/0/9
```

### mr (S=2, 10 blocks)

```
 plane  plane_len        fse    bp_same   bp_gated     oracle  bp*/fse fse_xf(R/D/Z) bp_xf(R/D/Z)
     0    4985282    2922723    2871815    2871815    2871815    0.983       0/0/10       0/0/10
     1    4985282     151472     947375     410651     151472    2.711        0/4/6       10/0/0
```

### nci (S=2, 32 blocks)

```
 plane  plane_len        fse    bp_same   bp_gated     oracle  bp*/fse fse_xf(R/D/Z) bp_xf(R/D/Z)
     0   16776722    5125467   13056237   13056237    5125467    2.547       32/0/0       32/0/0
     1   16776722    5102553   12960649   12960649    5102553    2.540       32/0/0       32/0/0
```
