# GPU-First Compression for libpz: Final Report

> Research workflow output, 2026-06-10. Produced by a multi-agent run: 4 repo-history
> readers + 5 web researchers (dietgpu, nvCOMP/GDeflate, academic literature, industry
> GPU decode, scientific/numeric codecs), 2 candidate generators, 3 adversarial
> verification lenses per candidate (dead-end conflict, GPU feasibility on Apple
> silicon incl. Metal-native, Pareto value), 1 synthesis pass. 37 agents total.

## 1. The core diagnosis

libpz's GPU entropy dead end ("0.54x decode, proven across 500+ iterations") was a measurement of a specific stream topology, not of GPU entropy coding. The measured design used 4 rANS lanes per chunk across ~160 chunks — roughly 300–640 concurrent states — on hardware that saturates at 8K–16K threads. dietgpu ([github.com/facebookresearch/dietgpu](https://github.com/facebookresearch/dietgpu)) makes the state count proportional to input size: 4 KiB segments × 32 lane-states = one rANS state per 128 bytes, so a 64 MB buffer carries ~512K independent states, with ballot+popcount compaction making the variable-rate interleave metadata-free (~3% total overhead). That gets 250–410 GB/s on an A100. The repo's own dead-end ledger concedes this: "the massive-parallelism model was never implemented in libpz." The same retrofit-vs-codesign distinction explains every shipping GPU format: GDeflate ([IETF draft](https://www.ietf.org/archive/id/draft-uralsky-gdeflate-00.html)), Brotli-G ([gpuopen.com/brotlig-sdk](https://gpuopen.com/brotlig-sdk/)), and Kraken/PS5 all converge on one recipe — independent ~64 KiB blocks, entropy streams split per token component, lane-pinned stateless Huffman in word-aligned sub-streams, decode = parallel entropy phase + splice phase. Nobody decodes serial-entropy wires fast on shaders; they redesign the wire. pz2 (independent blocks, 8-lane Huffman, sequence streams, fused splice) is already this format family on CPU; the gap to GPU decode is wire layout plus a kernel, not new theory.

The second half of the diagnosis is hardware, and it cuts against libpz. Every headline number above was earned on discrete HBM parts (A100: ~2 TB/s) where the GPU has a bandwidth moat over the host. On M5 Max, the GPU shares ~0.5 TB/s of unified memory with 18 fast CPU cores that already convert that bandwidth into 12.1 GiB/s pz2 decode. Naive bandwidth-scaling of the A100 results frequently lands *below* the existing CPU wall (e.g., bzip2gpu's 2.4 GB/s iBWT scales to ~0.6 GB/s vs the 1.12 GiB/s CPU bw baseline). Unified memory removes the PCIe tax that motivates much of nvcomp's design — a real advantage — but it also removes the GPU's exclusive bandwidth pool. Add the ~260 ms wgpu device init against a 17 ms all-cores decode of the entire Silesia blob, and no one-shot CLI invocation can ever win wall-clock; the only winnable axes are CPU-core offload in a persistent process, and workloads where the entropy phase genuinely bottlenecks.

What this implies: the encode side is settled (GPU hash tables, parallel parsing, GPU entropy encode are all genuinely dead — the literature never improved on the shipped hybrid GPU-match-finding + CPU-parse split), and the decode side has exactly one credible path — a GDeflate-shaped evolution of pz2's wire, where Metal's guaranteed 32-wide simdgroups map the lane-pinned layout 1:1. Nothing surveyed requires CUDA-only features; the binding constraints on this machine are arithmetic (shared bandwidth, CPU competition) and economic (no persistent-process customer yet), not expressibility.

## 2. Ranked candidates

### #1 — pz2-G32: GDeflate-style 32-lane wire for pz2, full GPU block decode

**Verdicts: spike-first / spike-first / spike-first (the only candidate all three lenses endorse).** **Feasibility: either (Metal-first; WGSL works via subgroups extension, or Brotli-G-style without them).**

**Design.** Per 2 MiB pz2 block, re-lay literals as Huffman codes across 32 sub-streams pinned to SIMD lanes (one shared canonical table), packed into 32-bit words with the GDeflate ≤2-word constraint, words interleaved in exact decode-read order. Sequence lanes unchanged. Decode = dispatch 1 (one simdgroup per literal tile, 32 symbols/round) + dispatch 2 (one workgroup per block, cooperative splice — the shipped `lz77_decode.wgsl` pattern). Match window stays full-block, so ratio is preserved; CPU encode is unchanged pz2 parse + a relayout pass; a scalar CPU decoder remains possible.

**Existence proof.** GDeflate ships in DirectStorage 1.1 at DEFLATE ratio parity, decoding via plain compute shader on any D3D12 GPU; vkd3d-proton ships a GLSL fallback with zero vendor intrinsics ([release v2.10](https://github.com/HansKristian-Work/vkd3d-proton/releases/tag/v2.10)); Brotli-G (MIT, [github.com/GPUOpen-LibrariesAndSDKs/brotli_g_sdk](https://github.com/GPUOpen-LibrariesAndSDKs/brotli_g_sdk)) is a second proof using no wave intrinsics at all. pz2 itself proves the format family on CPU: 1405 MB/s/core at 32.2%.

**Why it dodges the dead ends.** "GPU Huffman is dead" was encode-side bit-writing; this is decode-only over format-designed word-aligned lanes, never attempted in libpz. The 0.54x entropy result was serial-state rANS at ~640 threads; this has zero per-symbol state and thousands of simdgroups. Parse stays serial CPU (ParlZ/hash-table entries untouched). `clean-slate-codec.md` P10 explicitly parks — does not kill — this exact format class ("we did not disprove").

**Strongest surviving objection.** On unified memory the GPU shares the bandwidth the CPU already turns into 12.1 GiB/s; GDeflate-class decoders post ~20–30 GB/s on discrete GPUs with 3–4x M5's compute, so the realistic landing zone (5–15 GB/s) *brackets* rather than clears the CPU wall — and the per-block sequence splice (which the literal-phase spike does not measure) is the divergence-prone serial residue that could drag end-to-end below it. Even on success, the winning axis is "zero CPU cores consumed," which only pays in a persistent-process embedding.

**First spike + kill.** (a) Zero-GPU transcoder: repack existing pz2 blocks into the 32-lane layout + scalar CPU decoder; kill if Silesia ratio delta > 0.5pp vs shipped pz2. (b) Single Metal kernel doing only the literal Huffman phase over ~100 repacked tiles from persistent buffers; kill if < ~5 GB/s effective. Per the verifier consensus: extend the spike to gate the cooperative splice phase before any wire commitment — the literal phase passing alone proves little.

### #2 — GPU long-range dedup front-end (sort-based LDM)

**Verdicts: spike-first / spike-first / spike-first — but all three lenses agree the GPU leg is gratuitous.** **Feasibility: WGSL-feasible (sortlz_ops.wgsl primitives already in-tree).**

**Design.** Sample 8-byte fingerprints every K positions over the whole input, radix-sort (fp, pos) pairs (the SortLz trick, zero atomics), adjacent-pair verify, emit coarse cross-block copy instructions in a thin container above the block codec.

**Existence proof.** zstd `--long` gains 5–30% on tarballs/VM images at negligible decode cost; SortLz in-tree proves sort-based matching at 97% of CPU quality.

**Why it dodges dead ends.** Sorting replaces recency, so the hash-table dead end doesn't apply; no prior experiment touched cross-block redundancy; the in-tree roadmap independently rates this family PROMISING.

**Strongest surviving objection.** Two-fold. First, the GPU is a costume: zstd's LDM does the same matching with a tiny rolling-hash table on one CPU core, and sorting 16M pairs is sub-second on 18 cores — if the redundancy exists, CPU captures it cheaper. Second, the proposed "copy from live earlier output" layer reintroduces rolling inter-block history, which the repo's own roadmap vetting flagged as the one variant that would serialize decode and destroy pz's only defensible axis; the vetted immutable front-loaded-dictionary sibling captures most of the win with zero decode risk. The honest comparator is zstd `--long`, not zstd-default.

**First spike + kill.** CPU-only, one afternoon: fingerprint census at stride 32 over Silesia blob + a few-hundred-MB tarball/docker export; count verified duplicate bytes at offsets > 1 MiB; benchmark zstd `--long` on the same corpus as the bar. Kill if extra coverage < 3% on the tarball-class corpus. If it passes, route the result into the immutable-dict CPU form — this exits the GPU-first track.

### #3 — dietpz / pz-ans32: dietgpu-topology interleaved rANS lane

**Verdicts: spike-first / spike-first / reject (pareto, medium confidence).** **Feasibility: Metal-only in practice (ballot/popc/shuffle/umulhi map 1:1 to simdgroup ops at fixed width 32; WGSL path degraded — variable subgroup size, naga bug, no lane masks).**

**Design.** New entropy lane with dietgpu's exact wire: 4 KiB segments × 32 lane-states, ballot-compacted renorm words, shared order-0 table, prefix-sum coalesce. First customer: Num's high-entropy planes; second: a literal lane for GPU-routed pz2 blocks.

**Existence proof.** dietgpu 250–410 GB/s on A100; hipANS ([github.com/PAA-NCIC/hipANS](https://github.com/PAA-NCIC/hipANS)) proves no CUDA-exclusive dependency.

**Why it dodges dead ends.** The ledger's own fine print: the massively-interleaved topology "was never implemented in libpz"; new wire lane, so no PR-#120-style compatibility trap.

**Strongest surviving objection (the pareto reject).** Even a dietgpu-class kernel improves no shipped end-to-end number: Num exists for ratio and the ~3% rANS tax gives back 0.5–1pp of exactly that margin; pz2 already decodes 12.1 GiB/s on the same shared bus at better ratio; gpu-recoil-findings already taught the system-level lesson that "rANS is not the bottleneck" in any full pipeline. A passing 30 GB/s kernel still has no customer, purchased with a new macOS-only native Metal backend. The dead-ends lens adds: a topology-correct kernel landing at 10–15 GB/s passes its own kill line yet is a wall-clock wash against bandwidth-contending CPU workers.

**First spike + kill.** Decoder-only: CPU-encode one Num plane into the segment layout, Metal decode kernel over a 100 MB batch, persistent buffers. Kill if < 10 GB/s or ratio tax > 5% vs CPU FSE — but per the dead-ends lens, replace the absolute threshold with "beats all-cores CPU FSE on the same planes by enough to pay the tax and the backend cost."

### #4 — num-G: GPU-decodable numeric tier (decorrelation + vertical bit-packing, no tables)

**Verdicts: spike-first / pursue / reject (pareto, high confidence).** **Feasibility: WGSL-feasible, no subgroups needed (the only CUDA-flavored piece, cuSZp's decoupled-lookback fusion, is avoided by multi-dispatch).**

**Design.** Numeric-routed blocks keep the Num front-end but replace per-plane FSE with ndzip-style 32×32 bit-transpose + zero-word elimination + scan/compact. Every stage embarrassingly parallel both directions.

**Existence proof.** ndzip-gpu (SC'21), MPC, nvcomp Cascaded (up to 500 GB/s on integer columns), cuSZp2 (332–513 GB/s avg on A100).

**Why it dodges dead ends.** Exp-D bitplane failure was general data with no decorrelation and init-dominated measurement; Num's front-end shipped after it; no entropy coder exists to hit the entropy dead end.

**Strongest surviving objection (the pareto reject).** A deliberately table-free word-aligned format is memory-bound, so on unified memory the CPU decodes it in the same speed class — the GPU tier is dominated by its own mandatory CPU fallback. And zero-word elimination monetizes only zeros while sao's win came from FSE on dense skewed planes (only 6/28 columns take delta), so the ratio risk is 5–10pp on the one axis Num exists for. The feasibility lens's correction stands: any throughput gate must benchmark against a CPU-SIMD decoder of the *new wire*, not against Num-FSE.

**First spike + kill.** ~150 lines, no GPU: scalar vertical bit-pack + zero-word plane coder in `src/numeric.rs`, compare bytes vs FSE planes on sao/x-ray/mr/nci. Kill if regression > 2pp on routed planes. If it passes, the next gate is CPU-SIMD-of-own-format, not WGSL.

### #5 — gpu-ibwt: latency-hidden GPU inverse BWT

**Verdicts: spike-first / spike-first / reject (pareto, high confidence).** **Feasibility: WGSL-feasible (plain dispatches, primitives in-tree).**

**Design.** Keep the bw wire; K sampled LF-cursors per 1 MiB block (~0.1–0.4% overhead), thousands of concurrent pointer-chase chains, GPU scheduler hides the latency that collapses CPU scaling (the P8 pathology is a big-cache OoO artifact, never measured on GPU).

**Existence proof.** bzip2gpu (ICPP 2024, [dl.acm.org/doi/10.1145/3673038.3673067](https://dl.acm.org/doi/10.1145/3673038.3673067)): iBWT+iMTF up to 2.4 GB/s on A100 — the only published attack on this exact bottleneck.

**Why it dodges dead ends.** FWST untouched (full sort kept; only the inversion engine changes); entropy stays CPU; persistent buffers avoid the Repair failure mode.

**Strongest surviving objection (the pareto reject).** GPU iBWT improves the one axis bw isn't losing on. bw is off the frontier because of a hairline ratio margin over zstd-9 (erased by cursor overhead) and a 4x encode deficit — neither touched. pzstd frame-parallel decode works identically at -9/-12, so "parallel decode zstd can't reach" repeats the banned framing. And the arithmetic: 4-byte random loads suffer ~32x cache-line amplification; bandwidth-scaling bzip2gpu predicts ~0.6 GB/s — below the 1.12 GiB/s CPU baseline — unless SLC residency rescues it.

**First spike + kill.** One afternoon, no format change: CPU-precompute LF arrays for 32 real blocks, upload once, sweep K ∈ {64, 256, 1024}, measure chase throughput. Kill if < 2x CPU all-cores at any K. Rider from the feasibility lens: also benchmark a multi-cursor *CPU* iBWT (the in-tree decoder is single-cursor per block), which may cheaply raise the bar the GPU must beat.

### #6 — pz2-GA: gap-array checkpoints (one wire, CPU-to-GPU adaptive)

**Verdicts: spike-first / spike-first / reject (pareto, high confidence).** **Feasibility: WGSL-feasible, baseline (no subgroups).**

**Design.** Append optional per-lane (bit_offset, output_offset) checkpoints every ~512–1024 symbols to the unchanged pz2 wire; CPU ignores them, GPU runs one thread per segment.

**Existence proof.** Yamamoto et al. ICPP 2020 (gap arrays): 0.39–1.48% size overhead, large GPU decode speedups; Recoil (ICPP 2023) proves the one-wire economics.

**Why it dodges dead ends.** Stateless Huffman has nothing to checkpoint but bit positions; the clean-slate §6 ban on Recoil splits applies only to adaptive-context coders.

**Strongest surviving objection (the pareto reject).** pz2's 3.45x came from *fusing* sequence entropy into the splice; GPU offload re-separates the phases pz2's own iteration measured as slower (1.7–1.9x vs 3.1–3.6x). The 6 GB/s kill gate is miscalibrated — 18 cores run the entropy phase at ~24 GiB/s effective, so the spike can pass while the project fails. Best case is ~1.1–2x on the axis pz already oversupplies, paid in ratio on the axis where pz2 trails zstd-3. Also note the closest measured precedent: gpu-recoil checkpoint decode hit 65 MB/s — 4x slower than one CPU core — albeit 90% transfer-overhead on that box. If pz2-G32 proceeds, this candidate is subsumed by it.

**First spike + kill.** Checkpoint emission behind a flag, measure wire overhead at N ∈ {256, 1024, 4096} (kill > 0.5pp at N=1024); baseline-WGSL one-thread-per-segment kernel — with the gate raised to ~20+ GB/s or an explicit core-offload framing, per the feasibility lens.

## 3. Rejected candidates

- **Recoil graduation (GPU decode of shipped rANS streams)** — already built and measured in-tree (`docs/design-docs/gpu-recoil-findings.md`, 2026-03): split-per-workgroup decode scaled with split count exactly as predicted yet landed at 65 MB/s, 4x slower than *one* CPU core; the "topology, not format" hypothesis was tested on this wire and failed, and pz2 strictly dominates the best case.
- **GPU optimal parse graduation (top-K + backward DP)** — the "first experiment" was run on 2026-06-08 (commit c3068c5): `-O` measured worse than the lazy default and gated off, because both the CPU table builder and `lz77_topk.wgsl` are hardwired to a 32 KiB window vs the 1 MiB window the baseline ratio depends on; fixing that on GPU is a 32x probe blowup, and the DP cost model can't price the repeat offsets that drove the recent parse wins. The surviving idea (CPU-only DP with 1 MiB window + rep-offset state) is real but is not a GPU candidate.

## 4. What we did NOT find

- **Any measurement of the M5 GPU on any of these workloads.** Every throughput projection in this report is bandwidth/compute-scaled from A100/H100 numbers. The scaling factor for latency-bound table-lookup and pointer-chase kernels on Apple's lower-clocked, smaller-L2 GPU is the single biggest unknown, and it decides candidates #1, #3, and #5 alike.
- **A CPU-SIMD baseline for any GPU-friendly wire.** The recurring verifier finding: on unified memory, a format designed for GPU decode is also faster to decode on CPU. Nobody has measured NEON decode of a 32-lane pz2 layout or of the bit-packed Num wire — the correct denominator for every GPU gate.
- **A customer for "decode using zero CPU cores."** Every surviving win is a persistent-process/library story (260 ms init vs 17 ms corpus decode kills the CLI case). Whether libpz has or wants such an embedding is a product question this research cannot answer, and it gates the value of the entire track.
- **A public Apple-GPU decompression existence proof.** MTLIO does not document GPU-side decompression; the niche is genuinely open, which cuts both ways — first-mover opportunity, or evidence that people who tried found the unified-memory arithmetic unfavorable.
- **Splice-phase numbers for any GDeflate-class decoder.** Published GDeflate/Brotli-G figures don't break out the LZ-copy residue, which is exactly the phase the verifiers flagged as the likely end-to-end limiter for pz2-G32.
- **WGSL subgroup-size portability in practice** (8–64 variance, naga bug #8180) — untested against a lane-pinned format; Metal sidesteps it but commits to a second native backend.

## 5. Recommended next step

Run the **pz2-G32 spike**, in three gated stages, cheapest first:

1. **Zero-GPU transcoder (1–2 days):** repack shipped pz2 blocks into the 32-lane word-interleaved layout + scalar CPU decoder. Kill if Silesia ratio delta > 0.5pp. *Also record the CPU-SIMD decode speed of the new layout — it is the honest baseline for stage 2.*
2. **Literal-phase Metal kernel (2–3 days):** decode only the literal Huffman phase of ~100 repacked tiles from persistent buffers; measure on-device GB/s. Kill if < ~5 GB/s effective.
3. **Cooperative splice kernel (gate before any wire commitment):** the verifier consensus is unanimous that stage 2 passing proves little — the per-block sequence splice is where end-to-end can land below the 12.1 GiB/s CPU wall. Measure it before touching the format.

This is the only candidate all three adversarial lenses endorsed, it targets the format class the repo's own design doc parked rather than killed (P10), it has two shipping existence proofs with portable reference decoders to crib from (vkd3d-proton GLSL, Brotli-G HLSL), and its failure mode is cheap and informative: if stage 2 or 3 fails on M5 silicon, that result generalizes to candidates #3–#6 and closes the GPU-decode question for this hardware with data instead of extrapolation. In parallel, the long-range-dedup CPU census (one afternoon, no GPU code) is worth running because its payoff — a ratio win on tarball-class corpora — is independent of every GPU unknown above; just route any success into the vetted immutable-dict form.

## Appendix: full verdict matrix

| Candidate | dead-ends | gpu-feasibility | pareto-value |
|-----------|-----------|-----------------|--------------|
| pz2-G32 (GDeflate-style 32-lane wire) | spike-first | spike-first | spike-first |
| GPU long-range dedup front-end | spike-first | spike-first | spike-first |
| dietpz / pz-ans32 (interleaved rANS) | spike-first | spike-first | reject |
| num-G (vertical bit-packing tier) | spike-first | pursue | reject |
| gpu-ibwt (sampled-LF inverse BWT) | spike-first | spike-first | reject |
| pz2-GA (gap-array checkpoints) | spike-first | spike-first | reject |
| GPU optimal parse graduation | reject | spike-first | reject |
| Recoil graduation | reject | spike-first | reject |
