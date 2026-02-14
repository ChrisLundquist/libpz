# Agent Harness Implementation Plan

**Created:** 2026-02-14
**Status:** Active
**Context:** Based on analysis of OpenAI's agent_harness.md article on building software with AI agents

## Executive Summary

This plan implements applicable practices from OpenAI's agent-first development methodology for libpz. Since libpz is an unreleased compression library (not a production service), we focus on:
- Documentation structure and progressive disclosure
- Quality tracking and technical debt management
- Mechanical enforcement of coding standards
- Automated documentation maintenance

**NOT applicable:** UI testing, observability stacks, deployment tooling, service-layer architecture

## Current State

**Implemented:**
- ✅ CLAUDE.md as concise table of contents (~200 lines)
- ✅ ARCHITECTURE.md with technical design and roadmap
- ✅ .claude/plans/ for execution plans
- ✅ .claude/agents/ for custom agents (historian, tooling, benchmarker)
- ✅ .claude/friction/ for workflow impediment tracking
- ✅ Repository knowledge as system of record
- ✅ Pre-commit hooks (fmt, clippy)

**Gaps identified:**
1. Documentation lacks structured hierarchy (design-docs/, exec-plans/, references/)
2. No quality tracking document (module maturity, test coverage, GPU vs CPU status)
3. No design principles document
4. No core beliefs about agent-first development for libpz
5. No golden principles for mechanical enforcement
6. No tech debt tracker
7. No CI validation of documentation structure/freshness
8. No custom linters for libpz-specific invariants
9. No doc-gardening agent to detect stale documentation
10. No cleanup/refactoring agent for pattern enforcement

## Implementation Phases

### Phase 1: Documentation Structure (Week 1)

**Goal:** Create organized docs/ hierarchy for progressive disclosure

#### Tasks:

1. **Create docs/ directory structure:**
```
docs/
├── design-docs/
│   ├── index.md              # Catalog of all design decisions
│   ├── core-beliefs.md       # Agent-first principles for libpz
│   ├── gpu-memory-model.md   # Existing GPU batching design
│   └── pipeline-composition.md
├── exec-plans/
│   ├── active/               # Move current .claude/plans/ here
│   ├── completed/            # Archive completed plans
│   └── tech-debt-tracker.md  # Known technical debt with priority/impact
├── references/
│   ├── wgpu-llms.txt        # wgpu API reference for agents
│   ├── criterion-llms.txt   # Criterion benchmarking reference
│   ├── webgpu-shading-language.txt  # WGSL reference
│   └── compression-algorithms.md    # FSE, Huffman, LZ77 theory
├── generated/
│   └── gpu-memory-formulas.md  # Auto-generated from gpu-meminfo.sh
├── DESIGN.md                # Design principles and patterns
├── QUALITY.md               # Quality grades per module/feature
└── GOLDEN_PRINCIPLES.md     # Mechanically-enforced coding standards
```

2. **Migrate existing content:**
   - Move .claude/index/ content to docs/design-docs/
   - Move active plans to docs/exec-plans/active/
   - Archive old/completed plans to docs/exec-plans/completed/
   - Extract GPU memory analysis to docs/generated/

3. **Update CLAUDE.md** to reference new structure with progressive disclosure pattern

**Acceptance criteria:**
- All documentation findable via CLAUDE.md → docs/ → specific category
- No orphaned .md files in root or .claude/
- Each docs/ subdirectory has index.md explaining its purpose

---

### Phase 2: Quality & Debt Tracking (Week 2)

**Goal:** Create transparency on module maturity and technical debt

#### Tasks:

1. **Create docs/QUALITY.md:**
   - Grade each module (bwt, deflate, fse, huffman, lz77, mtf, optimal, pipeline, rans, rle, webgpu)
   - Dimensions: correctness, test coverage, GPU implementation status, performance, documentation
   - Track gaps and improvement priorities
   - Example format:
     ```markdown
     ## lz77
     **Grade:** A
     - Correctness: ✅ All validation tests pass
     - Test Coverage: ✅ 95%+ lines covered
     - GPU: ✅ Full GPU acceleration, CPU fallback
     - Performance: ✅ Beats gzip on 256KB+ blocks
     - Documentation: ⚠️ Missing optimal parsing explanation
     ```

2. **Create docs/exec-plans/tech-debt-tracker.md:**
   - Catalog known issues from friction reports
   - Track M5.3 (fuzz testing) milestone
   - Priority ranking (P0/P1/P2)
   - Impact assessment (correctness/performance/usability)
   - Example entries:
     - Missing fuzz testing (M5.3) - P1, correctness impact
     - Huffman sync decode TODO - P2, performance impact
     - GPU memory documentation drift risk - P2, maintainability

**Acceptance criteria:**
- Quality grades reflect actual test/benchmark results
- Tech debt tracker covers all known gaps from ARCHITECTURE.md and friction reports
- Both documents have owners and last-reviewed dates

---

### Phase 3: Design Principles & Core Beliefs (Week 3)

**Goal:** Codify libpz design philosophy for agent guidance

#### Tasks:

1. **Create docs/DESIGN.md:**
   - Composition over configuration (algorithms are standalone, pipelines compose them)
   - GPU-friendly design (table-driven, branchless, data-parallel)
   - Correctness first, then performance
   - Zero-copy where possible (`_to_buf` variants)
   - Graceful GPU fallback (skip tests if no device, don't fail)
   - Error handling philosophy (PzError variants, never panic on bad input)

2. **Create docs/design-docs/core-beliefs.md:**
   - Agent-first operating principles specific to libpz
   - Examples:
     - "Always compile-check GPU code with --features webgpu before committing"
     - "GPU is not faster for small inputs - don't optimize GPU paths for <128KB"
     - "Buffer allocations are the source of truth - never trust memory estimates in comments"
     - "Read existing code before proposing changes"
     - "Commit at every logical completion point"
     - "Use dedicated tools (Grep/Glob) over shell pipelines"

3. **Create docs/GOLDEN_PRINCIPLES.md:**
   - Mechanical rules for consistency (to be enforced by linters)
   - Examples:
     - Public API: `encode()` / `decode()` returning `PzResult<T>`
     - Always provide `_to_buf` variants for caller-allocated buffers
     - Tests go in `#[cfg(test)] mod tests` at bottom of file
     - GPU tests must check `is_gpu_available()` and skip gracefully
     - Annotations: `@pz_cost` must match actual buffer allocations
     - File size: modules should not exceed 2000 lines
     - Error messages: always include context (expected vs actual)

**Acceptance criteria:**
- DESIGN.md captures existing patterns from CLAUDE.md and ARCHITECTURE.md
- Core beliefs are actionable and specific to libpz
- Golden principles are verifiable/enforceable mechanically

---

### Phase 4: Reference Documentation (Week 4)

**Goal:** Make third-party API docs accessible to agents in-repo

#### Tasks:

1. **Create docs/references/wgpu-llms.txt:**
   - Extract relevant wgpu API docs (Buffer, Queue, CommandEncoder, ComputePass)
   - Focus on memory management, buffer binding, compute pipeline creation
   - Include common pitfalls (buffer alignment, map_async, staging buffers)

2. **Create docs/references/criterion-llms.txt:**
   - Criterion benchmarking API
   - black_box usage, throughput measurement, baseline comparison
   - How to structure benches/ files

3. **Create docs/references/webgpu-shading-language.txt:**
   - WGSL syntax reference
   - Built-in functions (atomicAdd, workgroupBarrier, etc.)
   - Memory model (storage buffers, workgroup shared memory)
   - Compute shader structure

4. **Create docs/references/compression-algorithms.md:**
   - FSE (Finite State Entropy) theory and implementation notes
   - Huffman canonical codes
   - LZ77 match finding and optimal parsing
   - BWT/MTF for reference (less critical)

**Acceptance criteria:**
- References are concise (agent can fit relevant section in context)
- Focused on APIs actually used by libpz
- No need to fetch external docs during development

---

### Phase 5: Automated Validation (Week 5)

**Goal:** Mechanically enforce documentation structure and freshness

#### Tasks:

1. **Create .github/workflows/doc-validation.yml:**
   - Check all docs/ have owners and last-reviewed dates
   - Verify cross-links are valid (no broken references)
   - Ensure docs/exec-plans/active/ plans have status field
   - Fail if orphaned .md files outside docs/ structure

2. **Update pre-commit hook (.githooks/pre-commit):**
   - Run doc-validation checks locally
   - Ensure QUALITY.md grades match test output (if tests run)
   - Validate @pz_cost annotations match buffer allocations (custom lint)

3. **Create scripts/validate-docs.sh:**
   - Reusable script for CI and local validation
   - Checks:
     - All docs/ indexes are up to date
     - Tech debt tracker references real issues
     - Quality grades have supporting evidence
     - Generated docs match source (gpu-memory-formulas.md vs gpu-meminfo.sh)

**Acceptance criteria:**
- CI fails on PR if documentation structure invalid
- Pre-commit hook catches obvious doc issues locally
- Scripts are fast (<5s) to not slow down commits

---

### Phase 6: Custom Linters (Week 6)

**Goal:** Enforce golden principles mechanically

#### Tasks:

1. **Create scripts/lint-libpz.sh:**
   - Check public API naming (encode/decode, _to_buf variants)
   - Verify tests are in `#[cfg(test)] mod tests` at end of file
   - Check GPU tests call `is_gpu_available()` before running
   - Validate @pz_cost annotations:
     - Parse src/webgpu/*.rs for buffer allocations
     - Compare to @pz_cost comments
     - Fail if mismatch
   - Enforce file size limits (warn >1500 lines, error >2000)

2. **Integrate into CI:**
   - Add lint-libpz.sh to .github/workflows/
   - Run on every PR
   - Zero-tolerance policy (must pass before merge)

3. **Create helpful error messages:**
   - Include remediation instructions in lint output
   - Example: "Error: @pz_cost(4MB) but create_buffer allocates 8MB at lz77.rs:42"
   - Link to GOLDEN_PRINCIPLES.md for explanation

**Acceptance criteria:**
- Linter catches actual violations in current codebase (if any)
- Error messages guide agents to fix (not just report problem)
- Runs in CI and locally via pre-commit hook

---

### Phase 7: Automated Agents (Week 7)

**Goal:** Continuous doc maintenance and pattern enforcement

#### Tasks:

1. **Create .claude/agents/doc-gardener.md:**
   - Agent prompt for detecting stale documentation
   - Scan for:
     - Outdated code examples in docs/
     - Quality grades that don't match recent test results
     - Tech debt tracker items that are resolved but not closed
     - References to removed code
   - Open PR with fixes or flag for human review

2. **Create .claude/agents/refactoring.md:**
   - Agent prompt for pattern enforcement
   - Scan for:
     - Violations of DESIGN.md principles (e.g., panic on bad input)
     - Code duplication (should use shared utilities)
     - Opportunities to apply golden principles
   - Open PR with refactoring suggestions

3. **Create scripts/run-gardening.sh:**
   - Wrapper to invoke doc-gardener agent on schedule
   - Can be run manually or via cron/CI
   - Outputs PR with fixes or report of findings

4. **Document usage in CLAUDE.md:**
   - When to manually invoke gardening agents
   - How to review agent-generated cleanup PRs
   - Criteria for auto-merge vs human review

**Acceptance criteria:**
- Doc-gardener successfully detects stale content
- Refactoring agent finds real pattern violations
- PRs are reviewable in <5 minutes
- Agents don't create busywork (high signal-to-noise)

---

### Phase 8: Documentation Finalization (Week 8)

**Goal:** Polish and validate complete system

#### Tasks:

1. **Generate docs/generated/gpu-memory-formulas.md:**
   - Run gpu-meminfo.sh and capture output
   - Include formulas, buffer sizes, batch recommendations
   - Auto-regenerate on every gpu-meminfo.sh change

2. **Create docs/design-docs/index.md:**
   - Catalog all design decisions with dates and status
   - Link to relevant code locations
   - Mark superseded designs clearly

3. **Create docs/exec-plans/active/index.md:**
   - Overview of active work
   - Dependencies between plans
   - Estimated completion

4. **Final validation:**
   - Run full doc-validation suite
   - Verify all cross-links work
   - Check CLAUDE.md progressive disclosure flow
   - Ensure agents can navigate docs/ without context overflow

**Acceptance criteria:**
- Agent can answer "where should X go?" by reading CLAUDE.md + relevant index
- No documentation duplicates between docs/ and top-level
- All generated docs are up to date
- doc-validation CI passes

---

## Success Metrics

1. **Agent efficiency:** Time to find relevant documentation decreases (measured by agent turn count)
2. **Documentation freshness:** <5% of docs flagged as stale by doc-gardener
3. **Quality transparency:** QUALITY.md reflects actual test/benchmark results
4. **Tech debt visibility:** All known gaps tracked in tech-debt-tracker.md
5. **Enforcement:** Custom linters catch 100% of golden principle violations

## Non-Goals

- UI testing infrastructure (libpz has no UI)
- Observability stack (library, not service)
- Deployment tooling (unreleased library)
- Service-layer architecture constraints (too prescriptive for library)

## Dependencies

- Existing .claude/ structure (agents, friction, plans)
- CI infrastructure (.github/workflows/)
- Pre-commit hooks (.githooks/)
- Validation test suite (cargo test)

## Risks & Mitigations

**Risk:** Documentation maintenance becomes burdensome
**Mitigation:** Automate with doc-gardener agent, keep docs/ structure shallow

**Risk:** Custom linters are too strict, block productive work
**Mitigation:** Start with warnings, promote to errors only for high-impact rules

**Risk:** Agents ignore documentation structure
**Mitigation:** Validate in CI, make CLAUDE.md point explicitly to new structure

**Risk:** Generated docs drift from source
**Mitigation:** CI checks compare generated docs to source output

## Future Enhancements

- Automatic quality grade updates based on test output
- Integration with MEMORY.md (agent persistent memory)
- Benchmark regression tracking in QUALITY.md
- Design decision log with rationale and alternatives considered
