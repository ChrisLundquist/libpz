# Technical Debt Tracker

**Last Updated:** 2026-02-14
**Owner:** Engineering team

## Purpose

Catalog of known issues, gaps, and technical debt in libpz. Items are prioritized by impact and assigned to milestones or marked as ongoing maintenance.

## Priority Levels

- **P0** - Critical (blocks release, correctness impact)
- **P1** - High (major functionality gap, security concern)
- **P2** - Medium (performance opportunity, usability improvement)
- **P3** - Low (nice-to-have, research topic)

## Active Issues

### P0: Critical

#### M5.3: Fuzz Testing Not Started
**Status:** Not started
**Impact:** Correctness (cannot verify random input handling)
**Milestone:** M5.3
**Estimated effort:** 2-3 days

**Description:**
No fuzz testing infrastructure. Should use `cargo-fuzz` to test:
- Random inputs 0-1MB
- Verify round-trip property (encode → decode → original)
- Check error handling on malformed compressed data
- Test all pipelines and standalone algorithms

**References:**
- ARCHITECTURE.md mentions M5.3 as only incomplete milestone
- validation.rs has corpus tests but no property tests

**Action items:**
1. Add `cargo-fuzz` to dev dependencies
2. Create fuzz targets for each algorithm
3. Create fuzz targets for each pipeline
4. Run 24h fuzz campaign on CI
5. Document findings in design-docs/

---

### P1: High Priority

#### Optimal Parsing Design Doc Missing
**Status:** Not started
**Impact:** Documentation (hard to understand/maintain optimal.rs)
**Estimated effort:** 1 day

**Description:**
`src/optimal.rs` implements backward DP for optimal LZ77 parsing but lacks design documentation:
- How the cost model works
- Why backward DP (vs forward greedy)
- GPU top-K match table → CPU DP handoff
- Parameter tuning (cost weights)

**References:**
- QUALITY.md rates optimal.rs as D for documentation
- Code is correct (tests pass) but opaque

**Action items:**
1. Create docs/design-docs/optimal-parsing.md
2. Explain backward DP algorithm with diagram
3. Document cost model parameters
4. Link to academic references (if any)

---

### P2: Medium Priority

#### GPU Huffman Decode Not Implemented
**Status:** Planned (see exec-plans/active/TODO-huffman-sync-decode.md)
**Impact:** Performance (decode is CPU-only, 300 MB/s)
**Estimated effort:** 3-5 days

**Description:**
GPU Huffman encoding is implemented (two-pass with Blelloch prefix sum), but decode is CPU-only. GPU decode has potential but low priority because:
- CPU decode is already fast (300+ MB/s)
- GPU decode requires synchronization (hard to parallelize)
- Break-even point likely >1MB

**References:**
- exec-plans/active/TODO-huffman-sync-decode.md has full plan
- Huffman encode GPU is ~400 MB/s, decode could match

**Action items:** See [TODO-huffman-sync-decode.md](active/TODO-huffman-sync-decode.md) for full plan.

---

#### rANS SIMD Decode Not Wired
**Status:** Partial (declarations exist, not connected)
**Impact:** Performance (rANS decode 40% slower than FSE)
**Estimated effort:** 2-3 days

**Description:**
`src/simd.rs` declares SSE2 4-way and AVX2 8-way rANS decode paths, but they're not wired into the main decode loop. Interleaved rANS is naturally SIMD-friendly (4-8 independent states).

**References:**
- ARCHITECTURE.md "Partially complete: rANS SIMD decode paths"
- QUALITY.md rates rans as B due to performance gap

**Action items:**
1. Wire SIMD intrinsics into rans::decode()
2. Benchmark SSE2 vs AVX2 vs scalar
3. Add feature detection for runtime dispatch
4. Update QUALITY.md if performance improves to A

---

#### rANS Reciprocal Multiplication for GPU
**Status:** Documented as future optimization
**Impact:** Performance (GPU rANS encode could avoid division)
**Estimated effort:** 5-7 days

**Description:**
rANS encode loop has data-dependent division. Could replace with precomputed reciprocal multiply-shift, but has edge cases with small frequencies and u32 overflow.

**References:**
- ARCHITECTURE.md "Partially complete: rANS reciprocal multiplication"
- Documented as low priority due to complexity

**Action items:**
1. Research reciprocal multiplication techniques
2. Prototype and validate edge cases
3. Benchmark GPU rANS encode
4. Only implement if >3x speedup

---

### P3: Low Priority

#### BWT GPU Performance
**Status:** Ongoing research
**Impact:** Performance (BWT GPU slower than CPU at all sizes)
**Estimated effort:** Unknown (research topic)

**Description:**
GPU BWT uses radix sort + prefix-doubling, which is 7-14x faster than previous bitonic sort but still slower than CPU SA-IS at all sizes. Options:
1. Port SA-IS to GPU (complex, unclear if beneficial)
2. Improve radix sort (better prefix-doubling strategy)
3. Accept CPU-only BWT (mark GPU path as research only)

**References:**
- ARCHITECTURE.md "GPU BWT Implementation" section
- QUALITY.md rates bwt as B due to GPU performance

**Action items:**
1. Profile GPU BWT to identify bottleneck
2. Research GPU suffix array construction algorithms
3. Decide: improve or remove GPU BWT path
4. Update QUALITY.md based on decision

---

#### C FFI Test Coverage and Documentation
**Status:** Minimal
**Impact:** Usability (FFI users have limited guidance)
**Estimated effort:** 2 days

**Description:**
`src/ffi.rs` has basic bindings but:
- No comprehensive test suite (only basic tests)
- No example C programs
- No FFI-specific documentation

**References:**
- QUALITY.md rates C FFI as B for testing/documentation

**Action items:**
1. Create examples/c/ with sample programs
2. Add FFI tests for all pipelines
3. Document memory management (who owns buffers)
4. Add to CLAUDE.md or separate FFI.md

---

#### GPU Decode for All Algorithms
**Status:** Not started
**Impact:** Performance (decode is mixed CPU/GPU)
**Estimated effort:** 10-15 days (all algorithms)

**Description:**
Most algorithms have GPU encode but CPU decode:
- LZ77: GPU encode, CPU decode
- Huffman: GPU encode, CPU decode
- FSE: CPU-only (table lookups)
- rANS: CPU-only

GPU decode is low priority because:
- Decode is usually fast enough on CPU
- GPU decode requires synchronization
- Break-even points are high (>1MB)

**Action items:**
1. Start with Huffman (see P2 above)
2. Prototype LZ77 GPU decode
3. Benchmark break-even points
4. Implement only if clear win

---

## Resolved Issues

### Documentation Structure (Resolved 2026-02-14)
**Was:** Documentation scattered across root, .claude/index/, .claude/plans/
**Now:** Structured docs/ hierarchy with progressive disclosure

**Resolution:**
- Created docs/ with design-docs/, exec-plans/, references/, generated/
- Migrated content from .claude/index/ and root
- Created DESIGN.md, QUALITY.md, core-beliefs.md

---

## Recurring Maintenance

### Documentation Gardening
**Frequency:** Weekly
**Owner:** doc-gardener agent (when implemented)

**Tasks:**
- Detect stale code examples in docs/
- Verify QUALITY.md grades match test results
- Check tech debt tracker items for resolved status
- Update references/ for dependency version changes

---

### Dependency Updates
**Frequency:** Monthly
**Owner:** Engineering team

**Tasks:**
- ~~Update wgpu~~ — Completed (now at 27.x, see exec-plans/completed/upgrade-wgpu-to-27.md)
- Update criterion, other dev dependencies
- Verify all tests/benchmarks still pass

---

### Benchmark Regression Checks
**Frequency:** Per-commit (CI)
**Owner:** CI pipeline

**Tasks:**
- Run quick benchmarks on Canterbury corpus
- Flag regressions >10% slowdown
- Update QUALITY.md if performance changes

---

## Friction Log Summary

Issues from .claude/friction/ (if any):
- None currently tracked

When friction reports are added, summarize here with links to full reports.

---

## Related Documents

- **QUALITY.md** - Quality grades per module/feature
- **ARCHITECTURE.md** - Roadmap and milestone status
- **exec-plans/active/** - Active execution plans for specific issues
- **.claude/friction/** - Workflow impediment reports

---

**Next review:** Weekly (or after major milestone completion)
