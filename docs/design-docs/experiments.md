# Things We've Tried — Experiments & Learnings

Documentation of optimization attempts, successful iterations, and abandoned approaches.

**Last updated:** 2026-02-14

## Successful Experiments

### 1. Cooperative-Stitch Kernel Strategy

**Period:** Feb 2026  
**Commits:** `1fce493`, `b06cfb4`, `7457360`

**What was tried:**
- Instead of brute-force backward scan, use cooperative search where each thread searches a distinct offset band
- Share discoveries via shared memory (1024 bytes)
- All threads re-test all discovered offsets from their own positions

**Why it worked:**
- Key insight: if offset d produces good match at position p, nearby positions p+1, p+2 are likely in the same repeated region
- Same-offset stitching captures this locality pattern
- Probe count reduced from 4896 to 1788 per position (64% reduction)

**Metrics:**
- Speedup: 1.8x over brute-force lazy kernel
- Quality: 94% of brute-force on natural text (6% loss)
- Trade-off: Excellent value — small quality loss for significant speed gain
- Now default in all four GPU dispatch paths

**Lessons:**
- GPU algorithms benefit from thread cooperation for data reuse
- Shared memory is valuable for cross-thread communication
- Quality trade-offs are data-dependent (text vs binary vs structured)

### 2. Ring-Buffered Streaming

**Period:** Feb 2026  
**Commit:** `4390d90`

**What was tried:**
- Replace per-block GPU buffer allocation with pre-allocated ring slots (double/triple-buffered)
- Enable overlapped GPU/CPU execution

**Why it worked:**
- Buffer alloc/map was 35% of kernel time on multi-block batches
- Pre-allocation eliminates this overhead entirely
- Ring buffer enables async pipelining: GPU computes slot N while CPU reads slot N-1

**Metrics:** 17% faster on 4MB batch, 7-11% faster on full pipelines. See `gpu-batching.md` for detailed numbers.

**Lessons:**
- GPU memory allocation is non-trivial overhead; pre-allocation is worth the code complexity
- Streaming patterns enable overlap that sequential patterns miss
- Backpressure is critical to prevent GPU queue overflow

### 3. Two-Tier Far Window Scanning

**Period:** Feb 2026  
**Commit:** `9395204`

**What was tried:**
- Add subsampled far-window search (1024-32768, every FAR_STEP=4 positions) to near-only brute-force
- Analyze CPU match offset distributions to understand quality loss

**Why it worked:**
- Near-only (1024 bytes) was missing ~55% of CPU match bytes on real files
- Far-window coverage recovers most quality at modest cost

**Metrics (alice29.txt):**
- Match ratio: 0.667 (near-only) → 0.796 (two-tier)
- CPU reference: 0.819
- Quality achieved: 97% of CPU
- Probe count: ~10K per position (vs 1K for near-only)

**Lessons:**
- Offset distribution analysis helps understand algorithm bottlenecks
- Two-tier strategies allow fine-grained trade-off between coverage and speed
- Real-world data has long-distance patterns (55% of matches >1KB offset)

### 4. FAR_STEP Tuning — 37% Speedup

**Period:** Feb 2026  
**Commit:** `a6ae499`

**What was tried:**
- Increase FAR_STEP from 4 to 8 (sample every 8th position instead of 4th)
- Profiling showed far window was 72% of kernel time

**Why it worked:**
- Halving far probes (~7744 → ~3872) significantly reduces GPU time
- Quality loss is modest (3% on alice29)
- Far-window matches are longer but less frequent — ~55% of bytes

**Metrics:**
- GPU LZ77 time: 82 ms → 52 ms for 256KB (37% faster)
- Full pipeline: 2.5 MB/s → 4.0 MB/s
- Quality: 0.796 → 0.772 (3% loss, still 94% of CPU)

**Lessons:**
- Profiling identifies bottlenecks (far window was unexpected hotspot)
- Strided sampling allows speed/quality tuning dials
- Early speedup wins available through profiling + tuning

### 5. Workgroup Shared Memory Tiling

**Period:** Feb 2026  
**Commit:** `27bdf12`

**What was tried:**
- Cooperatively load ~1280 bytes (320 u32 words) of near-window lookback region into shared memory
- Replace ~65K scattered global reads with shared memory reads

**Results:**
- Neutral on AMD Radeon Pro 5500M / Metal (~81ms for 256KB)
- Likely helps on discrete GPUs with weaker L2 cache

**Lessons:**
- Shared memory optimization helps only if L2 cache is weak
- Integrated GPUs have good L2, so bandwidth optimization may not help
- Optimization is portable but not universally beneficial

### 6. WebGPU Profiler Integration

**Period:** Feb 2026  
**Commits:** `22fe18a` (wgpu upgrade), `9e24938` (profiler)

**What was tried:**
- Upgrade wgpu from 24 to 27 to enable wgpu-profiler 0.25
- Add GPU timestamp profiling for kernel hotspot identification

**Why it worked:**
- Enables detailed GPU timeline analysis
- Critical for distinguishing kernel time from overhead
- Profiling identified buffer alloc (35%), helping target ring-buffering optimization

**Metrics:**
- Profiling overhead: minimal (<2% on batched runs)
- Identified correctness issue: AMD Vulkan timestamps unreliable (first query slot returns zeros)
- Workaround in `008d8ba` (2026-02-13)

**Lessons:**
- GPU profiling infrastructure enables data-driven optimization
- Driver bugs (AMD timestamps) require careful interpretation
- Multi-pass profiling necessary to separate kernel vs overhead

---

## Failed Experiments & Abandonments

### 1. Hash-Table LZ77 Kernel

**Period:** Feb 2025  
**Introduced:** Commit `d966133`  
**Abandoned:** Commit `ef067d0` (2026-02-13)

**What was tried:**
- Implement two-pass hash-table kernel matching CPU hash-chain strategy
- Pass 1: Parallel atomic insertion into fixed-size buckets
- Pass 2: Bounded bucket search (MAX_CHAIN=64 per position)

**Why it failed:**
- **Fundamental flaw:** Parallel atomic insertion filled buckets with arbitrary positions
- GPU thread scheduling randomizes insertion order
- Ring buffers don't help — insertion order is determined by thread scheduling, not position recency

**Catastrophic quality loss:**
- Repetitive data (1MB): 99.61% (CPU) → 6.25% (GPU)
- Same data, hash table couldn't find matches

**Root cause analysis:**
- CPU hash chains maintain most-recent positions (insertion order deterministic)
- GPU atomics have no notion of "most recent" — just "first to write"
- Different threads reach atomic update at different times based on hardware scheduling

**Lessons learned:**
- **Never assume GPU atomic ordering matches CPU insertion patterns**
- Quality-critical algorithms requiring deterministic state (like hash chains) don't port to GPU
- Test on repetitive data early to catch catastrophic quality loss
- Profiling alone can't detect quality issues (need compression ratio validation)

### 2. Blelloch Prefix Sum for Huffman

**Period:** Feb 2026  
**Introduced:** Commit `b4bf45d`  
**Abandoned:** Commit `75b8f17` & `67e449e`

**What was tried:**
- Implement GPU Blelloch prefix sum for frequency table computation in Huffman encoding
- Parallel reduction phase should be faster than CPU

**Why it failed (or wasn't worth it):**
- Frequency table computation is small (256 symbols max)
- CPU prefix sum faster due to branch prediction and cache locality
- Complexity overhead not justified by speed gain
- GPU overhead dominated for small frequency tables

**Metrics:**
- GPU Blelloch: ~10-20 ms
- CPU prefix sum: ~1-2 ms
- 10x slower on GPU (not worth it)

**Lessons:**
- **GPU is not always faster than CPU**
- Algorithms that work well at scale may have poor constant factors
- Small working sets often cache better on CPU L3
- Know the break-even point (typically ~64KB for GPU worthiness)

### 3. WHT/Haar Spectral Compression

**Period:** Feb 2026  
**Attempt:** Commit `d3ddeda`  
**Result:** Negative result (abandoned)

**What was tried:**
- Evaluate Walsh-Hadamard Transform (WHT) or Haar wavelets for compression
- Spectral decorrelation before entropy coding

**Why it failed:**
- No significant improvement in compression ratio over existing pipelines
- Added latency without offsetting size gains
- Not suitable for this compression pipeline's use cases

**Lessons:**
- **Not all transforms help compression ratio** (even orthogonal transforms)
- Evaluation before implementation critical (commit `d3ddeda` is a good example)
- Domain-specific nature of compression — transforms work for images, not arbitrary data

---

## Unresolved Questions & Future Opportunities

### 1. Hash Probes for Long-Distance Matches

**Status:** Documented but not implemented

**Idea:**
- Current cooperative kernel covers [1, 33792] effective lookback
- Extreme long-distance matches (>32KB offset) are rare but valuable on some files
- Hash probes could find these with low additional cost

**Open question:**
- Is the quality gain worth the added complexity?
- What's the actual distribution of extremely long-distance matches?

**Benchmark needed:** Run `examples/webgpu_diag.rs` on large file corpus; analyze matches >32KB offset

### 2. Deterministic GPU Matching for Reproducibility

**Status:** Friction-documented (see `.claude/friction/2026-02-14-lz77-gpu-research-friction.md`)

**Idea:**
- GPU thread scheduling is non-deterministic
- Some algorithms (like atomics) produce non-reproducible results
- Could add `--deterministic` flag that disables atomics and uses barrier-based synchronization

**Trade-off:**
- Determinism: reproducible results for debugging
- Performance: likely slower (barriers instead of atomics)

**Open question:**
- Is reproducibility worth the overhead? (probably not for production, valuable for research)

### 3. Adaptive Block Sizing Based on GPU Device

**Status:** Partially implemented (auto-reblocking to 128KB for quality)

**Idea:**
- Different GPU architectures have different compute/memory ratios
- Mobile GPUs (integrated): prefer smaller blocks, higher memory bandwidth
- Discrete GPUs: prefer larger blocks, higher compute throughput

**Current heuristic:** Fixed 128KB blocks

**Future work:**
- Detect GPU architecture and adjust block size
- Example: 64KB for integrated, 256KB for discrete

### 4. GPU Memory Consumption Predictor

**Status:** Documented gap. Use `scripts/gpu-meminfo.sh` for actual buffer analysis. See `design-docs/core-beliefs.md` belief #4 (buffer allocations as source of truth).

---

## Benchmarking Lessons

### What Worked

1. **Real-world data testing** (Canterbury corpus) caught quality issues hash table missed
2. **A/B benchmarking** (cooperative vs lazy kernel) quantified trade-offs
3. **Offset distribution analysis** explained why far window matters
4. **Ring-buffer backpressure testing** validated streaming strategy

### What Didn't

1. **Micro-benchmarks alone** — Hash table looked great at 256KB, failed at 1MB on real data
2. **Assumed GPU is always faster** — Blelloch prefix sum slower than CPU on small data
3. **Skipped quality validation** — Hash table quality collapse would have been caught earlier with regular ratio tests

### Best Practices Learned

- Test on multiple data types (text, binary, structured)
- Test at multiple scales (64KB, 256KB, 1MB+)
- Always validate compression ratio, not just speed
- Use profiling to identify actual bottlenecks, not assumed ones
- Prefer streaming/ring-buffered patterns for multi-block work
- Know GPU break-even points (see `docs/DESIGN.md` "GPU Break-Even Points")

---

## Related Documentation

- **research-log.md** — Chronological evolution of all attempts
- **lz77-gpu.md** — Current kernel details (final state after experiments)
- **gpu-batching.md** — Successful batching strategy details
- **.claude/friction/2026-02-14-lz77-gpu-research-friction.md** — Ongoing impediments
