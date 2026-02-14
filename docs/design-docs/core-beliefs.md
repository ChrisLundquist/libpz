# Core Beliefs: Agent-First Development for libpz

**Last Updated:** 2026-02-14
**Owner:** Engineering team

## Purpose

This document codifies agent-first operating principles specific to libpz compression library development. These are actionable rules that guide AI agents (like Claude Code) to work effectively in this codebase.

## Core Beliefs

### 1. Read Code Before Changing It

**Never propose changes to code you haven't read.** If a user asks about or wants you to modify a file, read it first with the Read tool. Understand existing patterns, error handling, and tests before suggesting modifications.

**Why:** Compression algorithms have subtle correctness requirements. Changing code without understanding context leads to bugs.

### 2. GPU Code Must Always Compile

**Always compile-check GPU code with `--features webgpu` before committing.** Even if you're only changing CPU paths, verify GPU paths still build.

```bash
cargo build --features webgpu
cargo test --features webgpu
```

**Why:** GPU feature is enabled by default. Breaking GPU compilation breaks CI and blocks other developers.

### 3. GPU Is Not Faster for Small Inputs

**Don't optimize GPU paths for inputs <128KB.** GPU acceleration breaks even around:
- LZ77: ~256KB
- Huffman: ~128KB
- BWT: ~512KB

Below these thresholds, CPU is faster due to kernel launch overhead and data transfer costs.

**Why:** GPU optimization effort should focus on large-block batched workloads where it actually provides speedup.

### 4. Buffer Allocations Are the Source of Truth

**Never trust memory estimates in comments or plans.** The only source of truth for GPU memory usage is the actual `Buffer::create()` / `create_buffer()` calls in `src/webgpu/*.rs`.

When analyzing GPU memory:
1. Grep for `create_buffer` in the specific file
2. Parse the size expressions (e.g., `input.len() * 4`)
3. Account for staging buffers and padding
4. Update @pz_cost annotations to match

**Why:** Buffer sizes are computed at runtime from input length. Estimates drift as code evolves. Only the Rust allocation code is accurate.

### 5. Commit at Every Logical Completion Point

**Don't let work accumulate uncommitted.** A logical completion point is any self-contained change:
- Bug fix
- New feature
- Refactor
- Test addition
- Documentation update

If a task has multiple independent parts, commit each separately.

**Why:** Small, frequent commits make review easier, enable bisection, and prevent losing work.

### 6. Use Dedicated Tools, Not Shell Pipelines

**Prefer Grep/Glob tools over shell pipelines.** Instead of:
```bash
# DON'T: Shell pipeline
grep -r "encode" src/ | cut -d: -f1 | sort | uniq
```

Use:
```bash
# DO: Dedicated tool
Grep with pattern="encode" path="src/" output_mode="files_with_matches"
```

**Why:** Dedicated tools are faster, don't need permission approval, produce better-structured output, and work correctly with special characters in filenames.

### 7. Algorithms Must Be Composable

**New algorithms must work both standalone and in pipelines.** Every algorithm should:
- Have `encode()` / `decode()` public API
- Provide `_to_buf` variants for caller-allocated buffers
- Return `PzResult<T>` (never panic on bad input)
- Include round-trip tests in `#[cfg(test)] mod tests`
- Document format and limitations

**Why:** libpz's value comes from flexible composition. Standalone algorithms can be tested, benchmarked, and reused in multiple pipelines.

### 8. Correctness First, Then Performance

**Always verify correctness before optimizing.** The validation priority is:
1. Unit tests pass
2. Round-trip tests pass (encode → decode → original)
3. Cross-decompression tests pass (GPU encode → CPU decode, vice versa)
4. Validation corpus tests pass (Canterbury, enwik8)
5. Benchmarks show improvement

Never skip steps 1-4 to chase benchmark numbers.

**Why:** Compression is easy to get wrong. Bugs can be silent (produce valid output with worse compression) or catastrophic (data corruption).

### 9. GPU and CPU Paths Must Produce Bit-Identical Output

**GPU and CPU implementations of the same algorithm must produce identical compressed output.** Not just "equivalent" (decompresses the same), but byte-for-byte identical.

When they diverge, the bug is almost always in:
- Stream demuxing (byte ordering, stream count)
- Padding/alignment handling
- Tie-breaking in sorting or matching

**Why:** Identical output enables cross-validation. Users should get the same compressed file regardless of GPU availability.

### 10. Pre-Commit Hook Failures Require New Commits

**The pre-commit hook auto-reformats and re-stages files.** If a commit fails on clippy warnings:
1. Fix the warning
2. Make a NEW commit (don't amend)

**Why:** The hook runs `cargo fmt` which modifies files in-place. Amending after hook failure risks including unrelated changes.

### 11. Tests Must Skip Gracefully on Missing GPU

**GPU tests must check `is_gpu_available()` and skip if no device is present.** Never fail tests due to missing GPU hardware.

```rust
#[test]
fn test_lz77_gpu() {
    if !crate::webgpu::is_gpu_available() {
        eprintln!("Skipping GPU test (no device available)");
        return;
    }
    // ... test logic
}
```

**Why:** CI runs on machines without GPUs. Tests should be robust to hardware availability.

### 12. Error Messages Must Include Context

**When returning errors, include expected vs actual values.** Bad:
```rust
Err(PzError::InvalidInput("buffer too small"))
```

Good:
```rust
Err(PzError::BufferTooSmall {
    required: compressed_size,
    provided: output.len(),
})
```

**Why:** Context-rich errors enable faster debugging. Agents and humans can understand what went wrong without adding debug prints.

### 13. In a Worktree, Run Git From the Worktree Directory

**Never `cd` to the main repo to run git commands.** Commits will land on the wrong branch.

**Why:** Git worktrees maintain separate working directories but share the .git database. Running git from the wrong directory can corrupt branch state.

### 14. Don't Chain Git With Non-Git Commands

**Don't use `echo`, `printf`, or `bash -c` with git commands.** These create compound commands that don't match permission allow-lists.

Instead, run git commands as standalone Bash calls:
```bash
# DON'T: Compound command
bash -c "git add . && echo 'Files staged'"

# DO: Separate calls
git add .
echo "Files staged"
```

**Why:** Permission system analyzes git commands individually. Compound commands trigger unexpected prompts.

### 15. Zero Warnings Policy

**`cargo clippy --all-targets` must pass with zero warnings before commit.** No exceptions.

The pre-commit hook enforces this, but verify manually if you bypass the hook:
```bash
cargo clippy --all-targets
```

**Why:** Warnings accumulate quickly and hide real issues. Zero-tolerance prevents warning debt.

## When These Beliefs Don't Apply

These are strong defaults, not absolute rules. Override when:
- **User explicitly requests it** (e.g., "skip tests for now")
- **Experimenting in a spike branch** (document as WIP)
- **External constraints** (e.g., dependency requires specific pattern)

When overriding, document WHY in commit message or comments.

### 16. Document Friction Points When You Encounter Them

**When you hit obstacles, document them in `.claude/friction/`:**

What to document:
- Permission prompts that shouldn't require approval
- Tool limitations or unexpected behavior
- Bugs in dependencies or external tools
- Confusing error messages that needed investigation
- Workarounds required for common tasks
- Missing features that would improve workflow

When to document:
- After completing a task where you hit an obstacle
- When you notice a pattern of repeated friction across sessions
- When a workaround feels hacky or unsatisfying
- When you spend >5 minutes debugging a tool or permission issue

Format: Create `YYYY-MM-DD-short-description.md` with Problem, Steps to Reproduce, Workaround, Suggested Fix sections.

**Why:** Friction reports help identify patterns and prioritize tooling improvements.

## Related Documents

- **../DESIGN.md** - Technical design principles (composition, GPU-friendliness, error handling)
- **../../CLAUDE.md** - Day-to-day development instructions (concise reference)
- **../../ARCHITECTURE.md** - Technical architecture and roadmap
- **../exec-plans/tech-debt-tracker.md** - Known issues and gaps
