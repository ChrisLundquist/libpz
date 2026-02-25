# Pareto-Competitiveness Implementation Plan — Phase 7: Auto-Selection Tuning

**Goal:** Tune auto-selection to pick optimal pipeline within 5% of best ratio for >90% of corpus files, with <10% trial overhead.

**Architecture:** Retune heuristic thresholds empirically on Canterbury+Silesia. Reduce trial from 8 to top-3 candidates with 4-8KB sample. Add GPU-awareness to selection.

**Tech Stack:** Rust

**Scope:** 8 phases from original design (phase 7 of 8)

**Codebase verified:** 2026-02-22

---

## Acceptance Criteria Coverage

### pareto-competitiveness.AC4: Measurement and tooling
- **pareto-competitiveness.AC4.3 Success:** Auto-selection picks pipeline within 5% of best-available ratio in >90% of corpus files

---

## Current State

`select_pipeline()` in `src/pipeline/mod.rs` (line 422) uses five hardcoded thresholds:

- `byte_entropy > 7.5 && match_density < 0.1` → Deflate (near-random)
- `run_ratio > 0.3` → Bw
- `byte_entropy < 3.0 && distribution_shape in {Skewed, Constant}` → Bw
- `match_density > 0.4 && byte_entropy > 6.0` → Lzfi
- `match_density > 0.4` → Deflate
- `match_density > 0.2 && byte_entropy > 5.0` → Lzfi
- Default → Deflate

`select_pipeline_trial()` (line 474) tests all 8 pipelines (`Deflate`, `Bw`, `Lzr`, `Lzfi`, `LzssR`, `Lz78R`, `LzSeqR`, `LzSeqH`) against a 65KB sample. This is expensive: 8 full compressions of 65KB before the actual compression begins.

`analysis.rs` computes `autocorrelation_lag1` (Pearson correlation of consecutive bytes) but this metric is never used in any heuristic decision path. High autocorrelation distinguishes structured binary data (executables, structured formats) from text and is a useful signal for BWT suitability.

No GPU-awareness exists anywhere in the selection logic. `select_pipeline()` returns the same pipeline regardless of whether a GPU device is available.

---

## Subcomponent A: Empirical Threshold Tuning

**Goal:** Replace guessed thresholds in `select_pipeline()` with values derived from running all pipelines on Canterbury+Silesia and recording which wins per file.

### Task 1: Create corpus analysis script

**File:** `scripts/tune-selector.sh`

Write a shell script that:

1. Downloads Canterbury corpus (if not cached in `corpus/canterbury/`) and Silesia corpus (if not cached in `corpus/silesia/`).
2. For each corpus file, runs `cargo run --release --bin pz -- --pipeline <p> --compress` for all 8 pipelines and records compressed size.
3. Records the winner (pipeline with smallest output) and all ratios for each file.
4. Emits a CSV to `docs/generated/YYYY-MM-DD-corpus-winner-map.csv` with columns: `filename`, `input_size`, `winner`, `winner_ratio`, `deflate_ratio`, `bw_ratio`, `lzr_ratio`, `lzfi_ratio`, `lzss_r_ratio`, `lz78_r_ratio`, `lzseq_r_ratio`, `lzseq_h_ratio`.
5. Emits a summary table showing, for each pipeline, how many corpus files it wins and what the average ratio gap is when it does not win.

The script should accept `--corpus-dir <path>` to use a pre-existing corpus directory.

**Implementation notes:**
- Use `pz --pipeline` flag to force a specific pipeline. If this flag does not exist yet, add it to `src/bin/pz.rs` alongside the existing `-a`/`--auto` and `--trial` flags.
- Record wall-clock time per pipeline per file as well, for later use in backend-aware selection (Task 7).
- Script output (per-file progress) goes to stderr; only the CSV summary goes to stdout.

**Verification:** Script runs end-to-end on a 3-file subset (`alice29.txt`, `enwik8` first 1MB, `mozilla` binary) without error. CSV output is well-formed.

---

### Task 2: Update `select_pipeline()` thresholds

**File:** `src/pipeline/mod.rs`

After running Task 1 on the full Canterbury+Silesia corpora, analyze `docs/generated/YYYY-MM-DD-corpus-winner-map.csv` to find threshold values that maximize the fraction of files where the heuristic matches the winner.

Specific changes:

1. **Add `autocorrelation_lag1` to the decision tree.** High autocorrelation (> 0.7) with moderate match density signals structured binary or executable data. BWT handles this class well because the sort clusters recurring byte patterns that LZ77 would miss at the 4-8 character scale.

2. **Tune `run_ratio` threshold.** The current 0.3 cutoff may be too aggressive (passing data with only 30% runs to BWT, where LZ-based pipelines could do better). Revise based on empirical data; the likely range is 0.4-0.6.

3. **Tune `match_density` thresholds.** The current 0.4 / 0.2 split was chosen without data. Revise based on per-file winner map. Add a case for LzSeqR as the preferred LZ pipeline when match_density is high (replacing Lzfi/Deflate defaults): LzSeqR is the primary compressor per the design plan and should be the default recommendation where LZ compression applies.

4. **Reorder branches** to put highest-confidence decisions first (near-random → BWT+RLE → LzSeqR → fallback).

5. **Add LzSeqH recommendation** for inputs where speed matters more than ratio. Currently `select_pipeline()` never returns `LzSeqH`. Add a branch for inputs where `byte_entropy > 6.5 && match_density > 0.3`, signaling high-entropy LZ data where Huffman entropy coding is faster and close enough to FSE.

Annotate each threshold with the empirical basis: e.g., `// Tuned on Canterbury+Silesia 2026-02-22: BW wins in 87% of files with run_ratio > 0.45`.

**Verification:** Run `select_pipeline()` against the profiling test vectors in `src/analysis.rs` tests (constant, uniform, run-heavy, text, repetitive). Confirm the function returns a pipeline that makes sense for each data class. No panics.

---

### Task 3: Tests for tuned thresholds

**File:** `src/pipeline/mod.rs`, in the `#[cfg(test)] mod tests` block at the bottom.

Add tests that:

1. **`test_select_pipeline_near_random`**: Feed `select_pipeline()` a `DataProfile` with `byte_entropy = 7.8`, `match_density = 0.05`. Assert result is `Pipeline::Deflate` (fast, no point in BWT or LZ-heavy pipelines).

2. **`test_select_pipeline_run_heavy`**: Feed a profile with `run_ratio = 0.55`, `byte_entropy = 2.5`. Assert result is `Pipeline::Bw`.

3. **`test_select_pipeline_lz_rich`**: Feed a profile with `match_density = 0.6`, `byte_entropy = 4.5`. Assert result is `Pipeline::LzSeqR` (or `Pipeline::Deflate` if LzSeqR is not preferred at this entropy — assert a specific value after Task 2 determines the threshold).

4. **`test_select_pipeline_structured_binary`**: Feed a profile with `autocorrelation_lag1 = 0.85`, `match_density = 0.35`, `byte_entropy = 5.0`. Assert result is `Pipeline::Bw`.

5. **`test_select_pipeline_does_not_panic_on_edge_cases`**: Feed profiles with all metrics at 0.0, and separately at their max values (entropy = 8.0, run_ratio = 1.0, match_density = 1.0, autocorrelation = 1.0). Assert a valid `Pipeline` is returned in all cases.

Note: These tests construct `DataProfile` directly rather than going through `analysis::analyze()`, so they are fast and not sensitive to analysis internals. `DataProfile` must remain constructible with struct literal syntax (no private fields added in Phase 7).

---

## Subcomponent B: Lightweight Trial Mode

**Goal:** Reduce `select_pipeline_trial()` overhead from 8 pipelines x 65KB to 3 pipelines x 4-8KB.

### Task 4: Pre-filter to top-3 heuristic candidates

**File:** `src/pipeline/mod.rs`

Refactor `select_pipeline_trial()` to use `select_pipeline()` as a ranking oracle rather than testing all 8 pipelines blindly.

Add a helper function:

```rust
/// Return the top N pipeline candidates for trial compression, ordered by
/// heuristic confidence (most likely best pipeline first).
fn heuristic_candidates(profile: &DataProfile, n: usize) -> Vec<Pipeline>
```

This function scores all pipelines and returns the top `n` by confidence. Scoring logic:

- Start with the winner from `select_pipeline()` at confidence 1.0.
- Add the second-most-likely pipeline based on which threshold was narrowly missed. For example, if `run_ratio = 0.38` triggered BWT (threshold 0.35 after tuning), add LzSeqR as the second candidate because the run_ratio was not high enough to be certain.
- Add a "wildcard" candidate: always include LzSeqR (the primary designed pipeline) if not already in the list, since it is the intended default for general data.

The function is deterministic given the same profile. It never returns duplicates.

Update `select_pipeline_trial()` to call `heuristic_candidates(profile, 3)` and iterate only over those three pipelines.

**Verification:** Call `heuristic_candidates()` with the test profiles from Task 3 and assert:
- The list has exactly `n` entries.
- No duplicates.
- The heuristic winner from `select_pipeline()` is always first.

---

### Task 5: Reduce trial sample size from 65KB to 4-8KB

**File:** `src/pipeline/mod.rs`

The `sample_size` parameter in `select_pipeline_trial()` is passed by callers. The current usage in `src/bin/pz.rs` (`--trial` mode) passes 65KB (65536 bytes).

Changes:

1. Change the default constant from 65KB to 6KB (6144 bytes). Document this constant as `TRIAL_SAMPLE_SIZE` at the top of the pipeline module alongside other constants.

2. Update the `--trial` flag handling in `src/bin/pz.rs` to use `TRIAL_SAMPLE_SIZE` instead of a hardcoded literal.

3. Add an optional `--trial-sample <bytes>` CLI flag to allow tuning without recompilation. Minimum value: 1024. Maximum value: 65536. Values outside this range are clamped with a warning to stderr.

**Rationale for 6KB:** At 6KB, the sample is large enough to contain enough LZ77 lookback distance for match density to be representative, but small enough that 3 compressions (top-3 candidates) take < 1ms on modern hardware for all CPU-based pipelines. The goal is < 10% trial overhead for inputs >= 64KB.

**Verification:** `select_pipeline_trial()` called with 6KB sample on the standard text test vector (repeated "Hello, world!...") returns a reasonable LZ-friendly pipeline. Timing: three 6KB compressions complete in < 5ms (assert not strictly in test, but verify manually with `--trial` flag).

---

### Task 6: Tests for trial mode accuracy and overhead

**File:** `src/pipeline/mod.rs`, `#[cfg(test)] mod tests`

Add tests:

1. **`test_trial_matches_heuristic_on_simple_data`**: For constant data (1MB of 0xAA), assert `select_pipeline_trial()` returns `Pipeline::Bw` (same as `select_pipeline()` heuristic). This verifies the trial and heuristic agree on obvious cases.

2. **`test_trial_returns_valid_pipeline`**: For a 1KB input of random bytes (use LCG from `analysis.rs` tests), assert `select_pipeline_trial()` returns a `Pipeline` variant without panicking and that the selected pipeline produces valid compressed output.

3. **`test_trial_top3_is_subset_of_all_pipelines`**: Assert `heuristic_candidates()` output is always a subset of the 8 known pipeline variants. (Guards against off-by-one or uninitialized memory bugs if the scoring is changed.)

4. **`test_trial_sample_size_respected`**: Pass a 100-byte input with `sample_size = 6144`. Assert no panic and the function uses all 100 bytes (not trying to read past the end). This tests the `input.len().min(sample_size)` path.

---

## Subcomponent C: Backend-Aware Selection

**Goal:** `select_pipeline()` and `select_pipeline_trial()` account for GPU availability and input size, preferring GPU-friendly pipelines when beneficial.

### Task 7: Add GPU availability and input size to selection logic

**File:** `src/pipeline/mod.rs`

The current signatures are:

```rust
pub fn select_pipeline(input: &[u8]) -> Pipeline
pub fn select_pipeline_trial(input: &[u8], options: &CompressOptions, sample_size: usize) -> Pipeline
```

Add a new extended variant:

```rust
pub fn select_pipeline_with_context(
    input: &[u8],
    gpu_available: bool,
    input_size: usize,
) -> Pipeline
```

Keep `select_pipeline()` as a stable public API that calls `select_pipeline_with_context(input, false, input.len())` for callers that do not have GPU context. This preserves backward compatibility.

Internally, `select_pipeline_with_context()` runs the same heuristic logic as `select_pipeline()` after Task 2, then applies GPU-aware overrides:

- If `gpu_available && input_size >= 262144` (256KB): eligible for GPU override.
- GPU override conditions (checked in order after the base heuristic):
  - If base heuristic returned `Pipeline::Bw` and `input_size >= 262144`: upgrade to `Pipeline::Bbw` (GPU BWT). Bbw achieves higher throughput than CPU BWT on large inputs per AC3.4.
  - If base heuristic returned `Pipeline::LzSeqR` and `input_size >= 262144`: keep LzSeqR but note in a comment that GPU entropy is applied at the pipeline level, not at selection time (the pipeline itself dispatches entropy to GPU when available).
- If `!gpu_available`: never return `Pipeline::Bbw`. This satisfies AC3.5 (no GPU overhead when GPU is absent).

**Verification:** Unit test `test_select_pipeline_with_context_no_gpu()` asserts that `select_pipeline_with_context(input, false, 1_000_000)` never returns `Pipeline::Bbw`. Unit test `test_select_pipeline_with_context_with_gpu_large()` asserts that BWT-suitable data (high run_ratio) with `gpu_available = true` and `input_size = 512 * 1024` returns `Pipeline::Bbw`.

---

### Task 8: Prefer GPU-friendly pipelines when GPU available and data is BWT-suitable

**File:** `src/pipeline/mod.rs`, `src/bin/pz.rs`

Update the `--auto` path in `src/bin/pz.rs`:

1. Before calling `select_pipeline()`, probe for GPU availability using the existing WebGPU device detection (check `cfg!(feature = "webgpu")` and attempt a device query, or use a cached result from earlier in the pipeline if available).

2. Pass `gpu_available` and `input.len()` to `select_pipeline_with_context()`.

3. Log the selection decision to stderr when `--verbose` is specified: `Selected pipeline: Bbw (GPU BWT, GPU available, 512KB input)`.

**Implementation notes:**
- GPU device probe must be fast (< 1ms). If the probe itself takes > 1ms (e.g., first-time adapter initialization), cache the result in a `OnceLock<bool>` at module scope.
- If the `webgpu` feature is not compiled in, `gpu_available` is always `false` — no conditional compilation needed in the selection logic itself, only in the probe code in `pz.rs`.
- The `--trial` path in `pz.rs` should also pass GPU context to `select_pipeline_trial()`. Add a `gpu_available: bool` parameter to `select_pipeline_trial()` and thread it through to `select_pipeline_with_context()` for the candidate ranking step.

---

### Task 9: Tests for backend-aware selection with and without GPU

**File:** `src/pipeline/mod.rs`, `#[cfg(test)] mod tests`

Add tests:

1. **`test_no_gpu_never_returns_bbw`**: For all data profiles (constant, uniform, run-heavy, text, repetitive), assert `select_pipeline_with_context(input, false, input.len())` never returns `Pipeline::Bbw`. This is a correctness invariant: Bbw must never be selected when no GPU is present.

2. **`test_gpu_available_bwt_suitable_large_input`**: Construct run-heavy input (vec of 0xAA repeated 512KB). Assert `select_pipeline_with_context(&input, true, input.len())` returns `Pipeline::Bbw`.

3. **`test_gpu_available_small_input_no_override`**: Construct run-heavy input of 32KB. Assert `select_pipeline_with_context(&input, true, input.len())` does NOT return `Pipeline::Bbw` (input too small for GPU overhead to pay off; threshold is 256KB).

4. **`test_gpu_available_lz_rich_no_bbw`**: Construct repetitive non-run data (repeating 8-byte pattern, low run_ratio). Assert `select_pipeline_with_context(&input, true, 1_000_000)` does not return `Pipeline::Bbw` (data is LZ-suitable, not BWT-suitable).

5. **`test_select_pipeline_backward_compat`**: Assert `select_pipeline(input)` returns the same pipeline as `select_pipeline_with_context(input, false, input.len())` for all test vectors. This guards the backward-compatible wrapper.

---

## File Inventory

| File | Change |
|---|---|
| `scripts/tune-selector.sh` | New — corpus winner-map script |
| `src/bin/pz.rs` | Add `--pipeline` flag, `--trial-sample` flag, GPU probe, verbose logging |
| `src/pipeline/mod.rs` | Update `select_pipeline()`, add `select_pipeline_with_context()`, `heuristic_candidates()`, update `select_pipeline_trial()`, new constants, new tests |

---

## Dependency Notes

Phase 7 depends on Phases 2-6 being complete because:

- Threshold tuning (Subcomponent A) must compare all pipelines, including any new or improved pipelines introduced in those phases. Tuning against incomplete pipelines would produce incorrect thresholds.
- Backend-aware selection (Subcomponent C, `Pipeline::Bbw`) depends on the GPU BWT pipeline introduced in Phase 6. If Phase 6 is incomplete, skip the Bbw override in Task 7 and add a `// TODO(phase6): enable Bbw when GPU BWT is ready` comment.
- Lightweight trial mode (Subcomponent B) depends on LzSeqR and LzSeqH (Phase 2) being production-quality, since LzSeqR is the intended default recommendation and wildcard candidate.

---

## Definition of Done

Phase 7 is complete when:

1. `select_pipeline()` picks a pipeline within 5% of the best-available ratio in > 90% of Canterbury+Silesia files, verified by running `scripts/tune-selector.sh` and checking the summary table.
2. `select_pipeline_trial()` completes in < 10% of total compression time for inputs >= 64KB, verified by profiling with `scripts/profile.sh`.
3. `select_pipeline_with_context()` never returns `Pipeline::Bbw` when `gpu_available = false`, verified by tests.
4. All new and existing tests in `src/pipeline/mod.rs` pass under `cargo test`.
5. `cargo clippy --all-targets` passes with zero warnings.
