# Agent Harness Analysis for libpz

**Date:** 2026-02-14
**Source:** OpenAI's article "Harness engineering: leveraging Codex in an agent-first world"
**Status:** Analysis complete, implementation plan created

## Summary

Evaluated 20+ practices from OpenAI's agent-first development methodology. Identified 10 applicable gaps for libpz (unreleased compression library). Created 8-week implementation plan in `agent-harness-implementation.md`.

## Practices Evaluated

### ✅ Already Implemented (7)

1. **Short CLAUDE.md as table of contents** - CLAUDE.md is ~200 lines, points to deeper docs
2. **ARCHITECTURE.md with technical design** - Comprehensive roadmap and GPU architecture notes
3. **Plans as first-class artifacts** - .claude/plans/ directory with execution plans
4. **Custom agents** - historian, tooling, benchmarker agents in .claude/agents/
5. **Friction tracking** - .claude/friction/ for documenting workflow impediments
6. **Repository knowledge as system of record** - No external Google Docs, all in repo
7. **Pre-commit enforcement** - .githooks/pre-commit runs fmt, clippy

### ❌ Applicable Gaps (10)

1. **Structured docs/ hierarchy** - No design-docs/, exec-plans/, references/ organization
2. **QUALITY.md** - No quality grades tracking module maturity, test coverage, GPU status
3. **DESIGN.md** - Design principles scattered across CLAUDE.md and ARCHITECTURE.md
4. **Core beliefs document** - No agent-first operating principles specific to libpz
5. **Golden principles** - No mechanically-enforced coding standards document
6. **Tech debt tracker** - Known issues not catalogued in single source of truth
7. **Documentation validation CI** - No automated checks for doc structure/freshness
8. **Custom linters** - No libpz-specific invariants enforced (e.g., @pz_cost vs actual buffers)
9. **Doc-gardening agent** - No automated stale documentation detection
10. **Cleanup/refactoring agent** - No automated pattern enforcement and technical debt reduction

### ⛔ Not Applicable (10+)

1. **Chrome DevTools integration** - libpz is a library, not a UI application
2. **Observability stack (LogQL, PromQL)** - library, not a service with logs/metrics
3. **Per-worktree app bootable instances** - no runnable app to boot
4. **Production dashboards** - unreleased library, no production deployment
5. **Layered domain architecture** - too prescriptive for a library (vs app with business domains)
6. **Service runtime validation** - no service layer
7. **Build/deploy infrastructure** - library only, no deployment pipeline
8. **Feature flags** - not applicable to library development
9. **Video recording of UI validation** - no UI
10. **Agent-driven QA of user journeys** - no user-facing app

## Implementation Priority

Created **8-week phased plan** in `agent-harness-implementation.md`:

**Phase 1 (Week 1):** Documentation structure - Create docs/ hierarchy with progressive disclosure
**Phase 2 (Week 2):** Quality & debt tracking - QUALITY.md and tech-debt-tracker.md
**Phase 3 (Week 3):** Design principles - DESIGN.md, core-beliefs.md, GOLDEN_PRINCIPLES.md
**Phase 4 (Week 4):** References - wgpu, criterion, WGSL, compression algorithm docs
**Phase 5 (Week 5):** Automated validation - CI jobs and pre-commit checks
**Phase 6 (Week 6):** Custom linters - Enforce golden principles mechanically
**Phase 7 (Week 7):** Automated agents - doc-gardener and refactoring agents
**Phase 8 (Week 8):** Finalization - Generated docs, indexes, complete validation

## Key Insights

### What Translates Well to Libraries

- **Progressive disclosure** - Start with table of contents, drill down as needed
- **Quality tracking** - Transparency on module maturity helps prioritization
- **Mechanical enforcement** - Linters catch drift before it compounds
- **Documentation as code** - Versioned, structured, validated like source code
- **Automated gardening** - Continuous maintenance scales better than periodic cleanups

### What Doesn't Translate

- **Service-layer concerns** - Observability, deployment, runtime validation
- **UI testing** - DevTools, screenshots, navigation automation
- **Domain architecture** - Business logic layers irrelevant for algorithm library
- **Production feedback loops** - Library has no production deployment to monitor

### Libpz-Specific Adaptations

1. **GPU memory analysis** - scripts/gpu-meminfo.sh as generated documentation
2. **Algorithm composition** - Design principles around standalone, composable modules
3. **CPU/GPU parity** - Quality tracking includes GPU implementation status
4. **Benchmark-driven quality** - Use criterion results, not service metrics
5. **Research artifacts** - docs/design-docs/ includes compression algorithm theory

## Next Steps

1. **Approve plan:** Review agent-harness-implementation.md
2. **Phase 1 execution:** Start with documentation restructure (highest ROI)
3. **Incremental rollout:** Validate each phase before proceeding
4. **Measure impact:** Track agent efficiency (turns to find docs) before/after

## References

- **Source article:** agent_harness.md (OpenAI blog post)
- **Implementation plan:** .claude/plans/agent-harness-implementation.md
- **Current docs:** CLAUDE.md, ARCHITECTURE.md, .claude/agents/, .claude/friction/
