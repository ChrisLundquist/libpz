# CLAUDE.md — libpz development guide

For detailed documentation, see `docs/DESIGN.md`, `docs/QUALITY.md`, `docs/design-docs/`, `docs/exec-plans/`, and `ARCHITECTURE.md`.

## Build & test

```bash
./scripts/test.sh              # Full suite: fmt, clippy, build, test
./scripts/test.sh --quick      # Skip build step, just lint + test
./scripts/test.sh --fix        # Auto-fix fmt + clippy before checking
./scripts/test.sh --all        # Test all feature combinations
```

The pre-commit hook (auto-configured by `scripts/setup.sh`) runs fmt, clippy, and tests before every commit. Use `--no-default-features` for CPU-only builds/tests. Prefer delegating test runs to the **tester** agent to keep your context clean.

## Benchmarking & profiling

```bash
./scripts/bench.sh             # pz vs gzip comparison (all pipelines, quiet)
./scripts/profile.sh           # samply profiling (see --help for options)
./scripts/gpu-meminfo.sh       # GPU memory cost calculator
./scripts/trace-pipeline.sh    # pipeline flow diagrams (text or mermaid)
```

All scripts support `--help`. Optimization workflow: measure (`bench.sh`) → identify (`profile.sh --stage <stage>`) → change → validate (`cargo test`) → re-measure (`cargo bench -- <stage>`) → confirm (`bench.sh`). Prefer delegating benchmark runs to the **benchmarker** agent.

## Agents

Specialized agents in `.claude/agents/` run on cheaper models and keep verbose output out of your context:
- **tester** — run tests, autofix, diagnose failures (Haiku)
- **benchmarker** — run benchmarks, generate comparison reports (Haiku)
- **historian** — git archaeology, research past attempts (Haiku)
- **tooling** — build scripts and workflow automation (Sonnet; consumes `.claude/friction/`)
- **maintainer** — review feedback backlog, update CLAUDE.md, delegate improvements (Opus; consumes `.claude/feedback/`)

## Project layout

- `src/{algorithm}.rs` — one file per composable algorithm (bwt, deflate, fse, huffman, lz77, mtf, rans, rle)
- `src/pipeline/` — multi-stage compression pipelines, auto-selection, block parallelism
- `src/webgpu/` — WebGPU backend (feature-gated behind `webgpu`)
- `kernels/*.wgsl` — WebGPU kernel source
- `scripts/` — test, bench, profile, setup, and analysis tools
- `docs/` — design docs, quality status, exec plans, references

## Key conventions

See **docs/DESIGN.md** for full design principles and **docs/design-docs/core-beliefs.md** for agent-first operating principles.

- Public API: `encode()` / `decode()` returning `PzResult<T>`, plus `_to_buf` variants
- Tests go in `#[cfg(test)] mod tests` at bottom of each module file
- GPU feature enabled by default, skip gracefully if no device available
- Zero warnings policy: `cargo clippy --all-targets` must pass clean
- Commit at every logical completion point (run `./scripts/test.sh --quick` first)

## Agent feedback loops

**Friction** (something blocked you or wasted time): Write a short report to `.claude/friction/YYYY-MM-DD-short-description.md` describing the problem, then move on. The tooling agent consumes the backlog and builds durable fixes.

**Feedback** (insights worth preserving): Write a short note to `.claude/feedback/YYYY-MM-DD-short-description.md` when you:
- Discover something that should be in CLAUDE.md (a convention, gotcha, or pattern not documented here)
- Find something in CLAUDE.md that was wrong, stale, or unhelpful
- Learn a non-obvious insight about the codebase that would save future agents time

Keep notes brief (a few lines). The **maintainer** agent consumes the backlog: evaluates reports, promotes worthy insights into CLAUDE.md, delegates fixes, and discards noise.
