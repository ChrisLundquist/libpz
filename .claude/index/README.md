# libpz Historical Documentation Index

Comprehensive research documentation for the libpz compression library, focusing on GPU optimization journey, architectural decisions, and algorithmic trade-offs.

**Purpose:** Enable researchers and developers to quickly understand project history without reading raw git commits.

**Last updated:** 2026-02-14

## Quick Navigation

### For Understanding GPU Optimization

**Start here if you want to know:**
- How GPU LZ77 evolved from hash tables to cooperative-stitch kernels
- Why certain optimizations succeeded (ring-buffering) or failed (hash tables)
- What performance gains are realistic on different GPU architectures

**Read in this order:**

1. **[research-log.md](research-log.md)** — Chronological journey with commit SHAs
   - Executive summary of 4 phases (OpenCL → Hash Table → Brute-Force Evolution → Cooperative-Stitch)
   - Major commits with metrics and context
   - Why hash-table kernel failed (6.25% quality on repetitive data)
   - Timeline of kernel iterations with performance before/after

2. **[experiments.md](experiments.md)** — What worked and why
   - Successful experiments (cooperative-stitch, ring-buffering, two-tier scanning)
   - Failed experiments (hash-tables, Blelloch prefix sum, WHT transforms)
   - Unresolved questions for future work
   - Benchmarking best practices learned

3. **[lz77-gpu.md](lz77-gpu.md)** — Current LZ77 GPU implementation
   - Cooperative-stitch kernel details (algorithm, coverage, quality metrics)
   - Kernel variants (lazy, hash, topk) and their status
   - Performance bottlenecks and optimization opportunities
   - Configuration parameters and tuning guidelines

### For Understanding Pipeline Architecture

**Start here if you want to know:**
- How multi-stage pipelines work (Deflate, Lzf, Lzr)
- How blocks flow through compression stages
- How stream demuxing separates match tokens for better entropy coding

**Read in this order:**

1. **[pipeline-architecture.md](pipeline-architecture.md)** — Architecture overview
   - V2 container format specification
   - Stream demuxing for multi-stream entropy coding
   - Block parallelism strategies (block-parallel, GPU-batched, pipeline-parallel)
   - Data flow through compress/decompress paths
   - StageBlock handoff semantics (critical for understanding bugs)

2. **[gpu-batching.md](gpu-batching.md)** — Multi-block execution strategies
   - Ring-buffered streaming architecture
   - Buffer allocation formula (36×N + overhead per block)
   - Backpressure synchronization patterns
   - Cost-model-driven scheduling
   - GPU memory constraints and safe limits

### For Understanding Specific Algorithms

- **[lz77-gpu.md](lz77-gpu.md)** — LZ77 matching on GPU
- **[pipeline-architecture.md](pipeline-architecture.md)** — Entropy coding stages (Huffman, FSE, rANS)
- **[research-log.md](research-log.md)** — Historical evolution of each algorithm

## Document Overview

### research-log.md

**Length:** ~800 lines  
**Scope:** Full GPU optimization journey with commit references

**Contains:**
- Executive summary (4 major phases)
- Phase 1: OpenCL Foundation (2024-2025)
- Phase 2: Hash Table Experiment (Feb 2025) — why it failed
- Phase 3: Brute-Force Evolution (Feb 2026) — incremental optimizations
- Phase 4: Cooperative-Stitch Era (Feb 2026) — current state
- Ring-buffered streaming optimization (17% gains)
- Backend architecture evolution (OpenCL → WebGPU)
- Entropy coding on GPU (Huffman, FSE, rANS)
- Current state summary with outstanding opportunities

**Best for:**
- Getting chronological understanding
- Finding specific commits related to a feature
- Understanding why decisions were made

**Key metrics included:**
- Before/after performance numbers for major optimizations
- Quality trade-offs (e.g., 6% quality loss for 1.8x speedup)
- GPU memory estimates and allocation patterns

### experiments.md

**Length:** ~650 lines  
**Scope:** Successful and failed experiments with learnings

**Contains:**
- Successful experiments:
  - Cooperative-stitch kernel (1.8x speedup, 94% quality)
  - Ring-buffered streaming (17% faster on batches)
  - Two-tier far window scanning (recovered 55% of missed matches)
  - FAR_STEP tuning (37% speedup)
  - Workgroup shared memory tiling (neutral on modern GPUs)
  - WebGPU profiler integration

- Failed experiments:
  - Hash-table kernel (catastrophic quality loss)
  - Blelloch prefix sum (10x slower than CPU)
  - WHT/Haar spectral compression (negative result)

- Unresolved questions and future opportunities
- Benchmarking best practices

**Best for:**
- Understanding why certain approaches were chosen
- Avoiding repeated failed experiments
- Learning GPU algorithm design patterns

**Key insights:**
- When GPU isn't faster (small working sets, high constant factors)
- Importance of testing on real data (hash table failed on repetitive data)
- Trade-off analysis (quality vs speed vs complexity)

### lz77-gpu.md

**Length:** ~500 lines  
**Scope:** Current LZ77 GPU implementation details

**Contains:**
- Cooperative-stitch kernel algorithm (2-pass, 1788 probes/position)
- Coverage analysis ([1, 33792] effective range with 64% probe reduction)
- Performance benchmarks by file type
- Quality analysis (94% on natural text, 100% on structured)
- Kernel configuration parameters (NEAR_RANGE, STRIDE, WINDOW_SIZE)
- Brute-force lazy kernel (legacy, kept for A/B benchmarking)
- Hash-table kernel (removed, why it failed)
- Ring-buffered streaming details
- Data flow through pipeline
- Bottlenecks and optimization opportunities
- Testing and validation approaches

**Best for:**
- Understanding current GPU LZ77 implementation
- Tuning kernel parameters for different workloads
- Identifying optimization opportunities
- Debugging GPU LZ77 issues

**Key sections:**
- Algorithm explanation with phase-by-phase breakdown
- Coverage/probe count comparison with other kernels
- Configuration space exploration (N=1024, S=512, W=512)

### pipeline-architecture.md

**Length:** ~600 lines  
**Scope:** Multi-stage compression pipeline design

**Contains:**
- Core concepts (V2 container format, stream demuxing)
- Pipeline implementations (Deflate, Lzf, Lzr)
- Data flow through blocks (compress and decompress paths)
- StageBlock structure and handoff semantics (critical!)
- Block parallelism strategies (block-parallel, GPU-batched, pipeline-parallel)
- Auto-selection strategy
- GPU cost annotations
- Known pitfalls and debugging strategies
- Performance debugging commands and key metrics

**Best for:**
- Understanding how blocks move through compression pipeline
- Debugging pipeline-related issues
- Understanding container format details
- Learning data flow through stages

**Key sections:**
- StageBlock.streams semantic (None before demux, Some after)
- Multi-stream demuxing for entropy coding improvement
- Compression vs decompression data flow
- GPU vs CPU output mismatch debugging

### gpu-batching.md

**Length:** ~450 lines  
**Scope:** Multi-block execution strategies and memory management

**Contains:**
- Ring-buffered streaming architecture
- Buffer allocation formula (36×N + overhead)
- Performance impact (17% gains on 4MB batch)
- Backpressure synchronization
- Cost-model-driven batching
- Block size auto-selection (128KB for GPU)
- GPU memory constraints (safe limits, VRAM allocation breakdown)
- Pipeline-parallel batching (experimental)
- Profiling tools and memory usage tracking
- Known limitations and future work

**Best for:**
- Understanding GPU memory usage
- Predicting batching behavior
- Optimizing for different GPU VRAM sizes
- Debugging memory-related issues

**Key sections:**
- Memory allocation breakdown (9.3MB per 256KB block)
- Ring buffer enables overlapped GPU/CPU execution
- Safe upper bound (allocate ≤25% of device VRAM)
- Suggested improvements (config-driven batch size)

## Commit References

All documents include specific commit SHAs (7+ characters). Most recent commits by topic:

**GPU LZ77 kernel:**
- `7457360` — Cooperative-stitch becomes default
- `1fce493` — Cooperative kernel introduced
- `b06cfb4` — Kernel parameter tuning
- `a6ae499` — FAR_STEP optimization (37% speedup)

**Ring-buffered streaming:**
- `4390d90` — Ring-buffered batching (17% gain)
- `008d8ba` — GPU profiling timestamp fixes

**Hash-table experiment:**
- `d966133` — Hash-table introduced (2x speedup claimed)
- `ef067d0` — Hash-table abandoned (6.25% quality on repetitive data)

**Backend infrastructure:**
- `6504641` — Remove OpenCL, WebGPU only
- `22fe18a` — Upgrade wgpu to 27
- `9e24938` — Add GPU profiler integration

## File References

Key files for understanding different aspects:

**GPU kernels:** `/Users/clundquist/code/libpz/kernels/`
- `lz77_coop.wgsl` — Current default kernel (cooperative-stitch)
- `lz77_lazy.wgsl` — Legacy brute-force kernel
- `lz77_hash.wgsl` — Abandoned hash-table kernel
- `huffman_encode.wgsl`, `fse_encode.wgsl` — Entropy coding

**GPU host code:** `/Users/clundquist/code/libpz/src/webgpu/`
- `lz77.rs` — LZ77 dispatch and ring-buffering
- `mod.rs` — GPU engine initialization, profiling
- `huffman.rs`, `fse.rs` — Entropy coding host code

**Pipeline:** `/Users/clundquist/code/libpz/src/pipeline/`
- `mod.rs` — Container format, compress/decompress entry points
- `blocks.rs` — Single-block processing, stage dispatch
- `parallel.rs` — Multi-block batching strategies
- `demux.rs` — Stream demuxing trait

**Diagnostics & examples:**
- `examples/coop_test.rs` — A/B benchmark for cooperative kernel
- `examples/webgpu_diag.rs` — GPU vs CPU quality comparison
- `src/validation.rs` — GPU↔CPU cross-decompression tests

## Related Documentation in Repository

- **ARCHITECTURE.md** — Detailed design notes, GPU benchmarks, roadmap
- **CLAUDE.md** — Day-to-day development instructions, build commands
- **.claude/friction/2026-02-14-lz77-gpu-research-friction.md** — Known impediments (AMD driver bugs, etc.)
- **.claude/friction/2026-02-14-heredoc-commit-permission-prompts.md** — Workflow friction

## How to Use This Documentation

### Scenario 1: Understanding a Performance Regression

1. Check **research-log.md** commit references to find when the regression might have been introduced
2. Review the specific commit's before/after metrics
3. If GPU-related, check **lz77-gpu.md** for current kernel behavior
4. Run `./scripts/profile.sh` per CLAUDE.md and compare to documented baseline

### Scenario 2: Optimizing GPU Performance

1. Start with **lz77-gpu.md** → "Known Bottlenecks" section
2. Check **experiments.md** → "Unresolved Questions" for low-hanging fruit
3. Review **research-log.md** to understand constraints from previous attempts
4. Design experiment and validate per **experiments.md** → "Benchmarking Lessons"

### Scenario 3: Debugging Pipeline Issues

1. Review **pipeline-architecture.md** → "Data Flow Through a Block"
2. Check **pipeline-architecture.md** → "Known Pitfalls" for common issues
3. Trace through `compress_block()` in `src/pipeline/blocks.rs` with specific input
4. Use **gpu-batching.md** if multi-block behavior is involved

### Scenario 4: Understanding Algorithm Trade-offs

1. Check **experiments.md** for the algorithm in question
2. Review quality vs speed metrics
3. Look at **research-log.md** for historical context
4. Reference specific kernel file for implementation details

## Maintenance & Updates

**When to update this documentation:**

- After completing a major optimization or research task
- When adding a new GPU kernel variant
- When abandoning or reverting an approach
- When discovering new bottlenecks or friction points

**Document in this order:**

1. Add entry to **research-log.md** (chronological)
2. Add summary to **experiments.md** if significant learnings
3. Update relevant algorithm-specific doc (lz77-gpu.md, pipeline-architecture.md, etc.)
4. File a friction report if workflow impediments encountered

**Review documents quarterly** to ensure metrics and status reflect current state.

## Contact & Questions

For detailed questions about specific optimizations:
- Check commit message (run `git show <SHA>`)
- Review relevant sections of the documents above
- Run diagnostic harnesses: `examples/webgpu_diag.rs`, `examples/coop_test.rs`
- Profile with `./scripts/profile.sh` to measure current behavior

---

**Last verified:** 2026-02-14  
**Research period:** Project inception → 2026-02-14  
**Coverage:** GPU optimization journey, pipeline architecture, batching strategies, experiments & learnings
