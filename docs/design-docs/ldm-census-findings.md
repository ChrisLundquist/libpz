# Long-range dedup census — findings (CPU spike, 2026-06-10)

**Verdict: PASS.** Verified long-range (>1 MiB offset) duplicate coverage on the
tarball-class corpus is **14.15%** (single-toolchain tar) and **56.08%**
(dual-toolchain tar) at zstd-LDM-equivalent parameters — 4.7x and 18.7x above the
3% kill line. zstd `--long` confirms the gain is real and realizable: it cuts the
single-toolchain tar by 3.8pp (13% smaller output) and the dual-toolchain tar by
16.9pp (**2.30x smaller output**) versus plain zstd -3. pz captures none of this
today; on these corpora pz2's honest gap to "zstd with the right flag" is not
~0.8pp — it is up to 2.27x.

Probe: `examples/ldm_census.rs`. Spawned from
`docs/design-docs/gpu-path-research.md` §"#2 — GPU long-range dedup front-end"
(CPU census leg). Branch: `claude/spike-ldm-census`.

## Method

Exact 8-byte fingerprints (raw bytes, zero false positives) sampled at stride 32
over the whole input; `(fp, pos)` pairs sorted; each sample paired with the
nearest earlier same-fingerprint sample at distance ≥ 1 MiB; every candidate
verified by byte comparison with greedy extension in both directions; coverage
counted through a bitmap so overlapping matches are never double-counted.
Headline numbers use `min_match = 64` (zstd LDM's default minimum match — the
"economical long-range copy" bar). Coverage and compressed sizes are
deterministic; the machine was shared during the run, so the only timing below is
labeled indicative.

## Corpora

| Corpus | Size | Description |
|---|---|---|
| `silesia.blob` | 202.1 MiB | the 12 Silesia files concatenated (dickens…xml, repo standard order) |
| `rustup-one.tar` | 547.8 MiB | `tar cf` of one complete rustup toolchain (`1.96.0-aarch64-apple-darwin`: bin, lib, etc, libexec, share). Realistic single-install tarball; contains genuine internal duplication (e.g. `librustc_driver` 195 MB and `rust-lld` 125 MB both embed LLVM, ~300 MB apart in the tar). |
| `rustup-two.tar` | 1064.9 MiB | `tar cf` of two toolchains that are byte-identical builds (`1.96.0` and `stable`, both rustc 1.96.0 ac68faa20; stable's `share/doc` excluded). Real-world stand-in for VM-image/container-layer-class input — no synthetic perturbation needed, the machine genuinely has two copies. |

## Verified long-range coverage (offset > 1 MiB)

| Corpus | min_match=64 (headline) | min_match=32 | min_match=8 (any verified dup) |
|---|---|---|---|
| silesia.blob | **6.18%** | 14.54% | 27.70% |
| rustup-one.tar | **14.15%** | 19.26% | 30.88% |
| rustup-two.tar | **56.08%** | 58.68% | 64.63% |

Kill criterion was coverage < 3% on the tarball-class corpus → **not killed**, by
a wide margin, under every parameter choice. Sensitivity: stride 16 instead of 32
raises Silesia coverage 6.18% → 8.50% (stride-32 numbers are conservative).

Match-shape facts that matter for design (from the probe's histograms):

- **Tarballs: the duplicate mass is file-level.** rustup-one has ~20k verified
  matches ≥ 2 KiB, with single matches up to 4 MiB. These are whole duplicated
  artifacts (rlibs/dylibs/LLVM copies), not statistical coincidences.
- **Silesia: the mass is short.** 73% of its verified matches are 64–127 bytes;
  almost nothing above 8 KiB. This is why its 6.18% coverage does not convert
  into ratio (below).
- **Distance distribution is bimodal and deep.** On rustup-one, 83k matches
  (most of the covered bytes) sit at 256–512 MiB offsets; on rustup-two the
  cross-toolchain copy sits at ~540 MiB. A 128 MiB window (`--long=27`) misses
  most of it — see the table.

## Compression bar (deterministic sizes, same corpora)

zstd 1.5.7 single-threaded defaults; pz built from this worktree
(`--no-default-features`, all cores); pz2 output round-trip-verified.

| Method | silesia.blob | rustup-one.tar | rustup-two.tar |
|---|---|---|---|
| zstd -3 | 31.24% | 29.04% | 29.84% |
| zstd -3 --long=27 | 30.98% | 28.18% | 28.92% |
| zstd -3 --long=30 | 30.98%¹ | **25.26%** | **12.99%** |
| pz lzf | 32.18% | 29.25% | 30.02% |
| pz pz2 | 31.04% | 28.69% | 29.46% |

¹ Silesia row uses `--long=28` (corpus is 202 MiB, so ≥2^28 window = whole input).

Deltas vs zstd -3: long-range matching is worth **−0.26pp** on Silesia,
**−3.78pp** on the single-toolchain tar, **−16.85pp** (333 MB → 145 MB, 2.30x)
on the dual-toolchain tar. rustup-two compressed with `--long=30` lands within
0.03% of rustup-one's size — the entire second toolchain is deduplicated away.

Two readings of that table:

1. **Coverage converts to ratio almost 1:1 on tarballs.** Predicted gain ≈
   coverage × baseline ratio: rustup-one 14.15% × 29% ≈ 4.1pp (realized 3.78pp);
   rustup-two 56.08% × 29.8% ≈ 16.7pp (realized 16.85pp). The census is an
   accurate predictor when the duplicate mass is in long matches. On Silesia the
   same model predicts 1.9pp but only 0.26pp is realized — 64–256-byte matches
   at multi-MiB distance are mostly already-compressible content where copy
   overhead eats the margin. **Census coverage is only a valid go-signal when
   the length histogram is heavy-tailed.**
2. **Window size dominates.** `--long=27` (zstd's default long window, 128 MiB)
   captures only ~23% of the available win on rustup-one and ~5% on rustup-two.
   The honest comparator for any future pz feature is zstd `--long` *with a
   window covering the input*, and any pz front-end must be whole-input-scoped.
   The sort-based census approach is naturally windowless.

## Cost of the census itself (indicative only)

5 reps on rustup-one.tar (548 MiB), shared machine: total probe time median
0.7 s, spread 0.6–0.7 s (~800–900 MB/s effectively single-threaded; sort
dominates and is trivially parallelizable). A production front-end pass at this
cost is negligible next to pz's 740–1130 MB/s all-cores compress.

## Verdict and recommendation

**PASS — build the CPU front-end; the GPU leg remains gratuitous.** The matching
itself costs <1 s/GB on one core; there is nothing for a GPU to accelerate, which
is what all three review lenses already concluded.

Routing, in light of the measured match shapes:

- **The immutable front-loaded-dictionary form is the right one, and the numbers
  make it easy.** Duplicate sources are first occurrences scattered through the
  whole input, and matches are file-level. So: census pass identifies verified
  long matches; copy instructions live in a thin container above the block
  codec; sources are constrained to regions that are themselves copy-free
  (depth-1 dependency). Decode = decompress all blocks in parallel exactly as
  today, then apply coarse splices (or decode source-containing blocks first —
  one barrier, no rolling history). This keeps pz's parallel-decode axis intact.
  Rolling inter-block history remains rejected — it would serialize decode for a
  gain the immutable form already captures (the dual-toolchain numbers above are
  ~all file-level dups, exactly the case depth-1 handles).
- **Gate the feature on the length histogram, not raw coverage.** Silesia shows
  6% coverage worth only 0.26pp. A cheap router: run the census (sub-second),
  enable the front-end only when covered-bytes-in-matches-≥2KiB clears a
  threshold (say 3% of input). On general corpora the feature is then a no-op
  with negligible encode cost.
- **Set expectations by corpus class.** This wins on tarballs, container
  layers, VM images, backup streams — multi-hundred-MB inputs with file-level
  duplication. It does nothing for Silesia-class mixed corpora, and that is
  fine; it is additive to (not competitive with) the block codec.
- Min-match for the production front-end should be ≥ 64 (zstd's LDM default);
  the census shows dropping to 32 buys ~2–5pp more *coverage* on tarballs but in
  the regime the ratio model says doesn't convert.

## Reproduction

```bash
# corpora
cat samples/silesia/{dickens,mozilla,mr,nci,ooffice,osdb,reymont,samba,sao,webster,x-ray,xml} > /tmp/silesia.blob
tar cf /tmp/rustup-one.tar -C ~/.rustup/toolchains 1.96.0-aarch64-apple-darwin
tar cf /tmp/rustup-two.tar -C ~/.rustup/toolchains --exclude='*/share/doc' \
    1.96.0-aarch64-apple-darwin stable-aarch64-apple-darwin

# census
cargo run --release --no-default-features --example ldm_census -- /tmp/rustup-one.tar
# bar
zstd -3 -k -o /dev/null ... ; zstd -3 --long=30 ... ; pz -c -p pz2 ...
```

## Postscript: reconciliation with PRs #146/#147/#148 (merged later the same day)

After this census ran, #146 (segment-scoped head dict spike), #147 (`lz77::FrozenDict`
shared match-finder), and #148 (shipped `-p pz2d` tier) landed. They implement the
*architecture* this doc recommends — an immutable dictionary, depth-1 dependency, no
rolling inter-block history, 2-wave parallel decode — and prove it pays on general
corpora (blob 31.04% → 30.48%, best LZ-family ratio in pz). But their *scope* is
segment-local: `PZ2D_SEGMENT_SIZE` is 32 MiB and the dict is the first 16 MiB of the
**same segment**, so the maximum match offset any block can express is < 32 MiB and
cross-segment redundancy is untouched by construction. The census's key unique finding
is therefore still open: on the tarball-class corpora the duplicate mass sits at
256–540 MiB offsets (rustup-one's LLVM copies ~300 MiB apart; rustup-two's
cross-toolchain copy at ~540 MiB), which no 32 MiB-scoped dict can reach. The measured
gap stands: pz2 hit 28.69% / 29.46% on rustup-one/-two vs zstd `--long=30`'s 25.26% /
**12.99%** — up to 2.27x — and pz2d's segment dict cannot close the rustup-two gap at
all. What remains is exactly this doc's routing: a whole-input-scoped census pass
emitting coarse copy instructions in a thin container *above* the (now-existing) dict
machinery, gated on the heavy-tail length histogram. FrozenDict + the 2-wave decode are
the right reusable substrate for that front-end; the windowless census matcher is the
missing piece.
