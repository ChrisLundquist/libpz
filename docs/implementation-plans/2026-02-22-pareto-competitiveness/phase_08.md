# Pareto-Competitiveness Implementation Plan — Phase 8: Pareto Validation

**Goal:** Verify and document Pareto-competitiveness against gzip, zlib-ng, lz4, zstd across all quality levels.

**Architecture:** Full benchmark matrix using Phase 1 infrastructure. Pareto curve visualization script. Gap analysis document identifying remaining competitive gaps.

**Tech Stack:** Bash, Python (matplotlib for visualization), Rust

**Scope:** Phase 8 of 8 from the pareto-competitiveness design plan.

**Codebase verified:** 2026-02-22

---

## Acceptance Criteria Coverage

### pareto-competitiveness.AC1: Pareto-competitive with gzip
- **pareto-competitiveness.AC1.5 Success:** For each gzip level 1-9, there exists a libpz configuration (pipeline + quality level) that is faster at the same ratio or achieves better ratio at the same speed
- **pareto-competitiveness.AC1.6 Failure:** Random/incompressible data does not crash or produce output larger than input + 1% overhead

### pareto-competitiveness.AC2: Single-thread CPU baseline
- **pareto-competitiveness.AC2.1 Success:** `pz -t 1` with LzSeqR compresses Canterbury corpus at throughput >= gzip -6 throughput
- **pareto-competitiveness.AC2.2 Success:** `pz -t 1` with LzSeqR achieves ratio within 2% of gzip -6 on Canterbury corpus
- **pareto-competitiveness.AC2.3 Success:** `pz -t 1` decompression throughput >= gzip decompression throughput
- **pareto-competitiveness.AC2.4 Edge:** Single-thread mode works correctly for inputs < 1KB (no block-parallel overhead)

### pareto-competitiveness.AC3: Multi-thread and GPU scaling
- **pareto-competitiveness.AC3.1 Success:** Multi-thread CPU compression achieves near-linear speedup up to 4 threads on 1MB+ inputs

---

<!-- START_SUBCOMPONENT_A -->
## Subcomponent A: Full Pareto benchmark

This subcomponent runs the complete speed-vs-ratio matrix across all libpz configurations and all competitor levels, then verifies AC2.1–AC2.4 against the collected numbers.

<!-- START_TASK_1 -->
### Task 1: Create scripts/pareto-bench.sh

**Files:**
- Create: `scripts/pareto-bench.sh`

**Background:**

`scripts/bench.sh` (Phase 1) benchmarks libpz pipelines against gzip at its default level. For Pareto validation we need to sweep across all competitor quality levels (gzip 1-9, zstd 1-19, lz4 1-12) and all libpz configurations (each pipeline × each parse strategy × thread counts 1 and 4) to produce a complete (ratio, throughput) point cloud. `bench.sh` does not support multi-level competitor sweeps — that logic belongs in a dedicated script.

`pareto-bench.sh` will:
1. Build the release binary once.
2. For each corpus file, run each competitor level and each libpz config, recording (compressed_size, time_ns).
3. Accumulate totals across the corpus.
4. Print a CSV file to `docs/generated/pareto-results.csv` with columns: `corpus,codec,level,threads,ratio_pct,throughput_mbs`.
5. Print a human-readable summary table sorted by ratio ascending.

The CSV is the stable artifact consumed by Task 4 (plot-pareto.py) and Task 9 (gap analysis).

**Step 1: Create `scripts/pareto-bench.sh`**

```bash
#!/usr/bin/env bash
# pareto-bench.sh — Full Pareto benchmark: all libpz configs vs all competitor levels.
#
# Produces docs/generated/pareto-results.csv with columns:
#   corpus,codec,level,threads,ratio_pct,throughput_mbs
#
# Usage:
#   ./scripts/pareto-bench.sh [OPTIONS]
#
# Options:
#   -n, --iters N       Iterations per timing (default: 3)
#   --corpus NAME       Corpus to run: cantrbry, large, or both (default: cantrbry)
#   --no-gpu            Skip WebGPU builds
#   -v, --verbose       Show per-file rows in addition to totals
#   -h, --help          Show this help

set -euo pipefail
export LC_ALL=C
export LANG=C

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PZ="$PROJECT_DIR/target/release/pz"
OUT_CSV="$PROJECT_DIR/docs/generated/pareto-results.csv"

ITERATIONS=3
CORPUS="cantrbry"
VERBOSE=false
GPU=true

usage() {
    cat <<'EOF'
pareto-bench.sh — Full Pareto benchmark across all libpz configs and competitor levels.

Usage:
  ./scripts/pareto-bench.sh [OPTIONS]

Options:
  -n, --iters N       Iterations per timing (default: 3)
  --corpus NAME       cantrbry, large, or both (default: cantrbry)
  --no-gpu            Skip GPU builds
  -v, --verbose       Per-file detail rows
  -h, --help          Show this help

Output:
  docs/generated/pareto-results.csv
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -v|--verbose) VERBOSE=true; shift ;;
        -n|--iters)
            [[ $# -lt 2 ]] && { echo "ERROR: --iters requires an argument" >&2; exit 1; }
            ITERATIONS="$2"; shift 2 ;;
        --corpus)
            [[ $# -lt 2 ]] && { echo "ERROR: --corpus requires an argument" >&2; exit 1; }
            CORPUS="$2"; shift 2 ;;
        --no-gpu) GPU=false; shift ;;
        -*)
            echo "ERROR: unknown option '$1'" >&2; exit 1 ;;
        *) shift ;;
    esac
done

# ---- Build ----
BUILD_OUT=$(mktemp)
if ! cargo build --release --manifest-path "$PROJECT_DIR/Cargo.toml" >"$BUILD_OUT" 2>&1; then
    echo "ERROR: build failed" >&2; cat "$BUILD_OUT" >&2; rm -f "$BUILD_OUT"; exit 1
fi
rm -f "$BUILD_OUT"
[[ -x "$PZ" ]] || { echo "ERROR: $PZ not found after build" >&2; exit 1; }

# GPU build (optional, feature-gated)
PZ_GPU=""
if [[ "$GPU" == true ]]; then
    GPU_BUILD_OUT=$(mktemp)
    GPU_BIN="$PROJECT_DIR/target/release/pz-gpu"
    if cargo build --release --features webgpu --manifest-path "$PROJECT_DIR/Cargo.toml" \
            >"$GPU_BUILD_OUT" 2>&1; then
        cp "$PZ" "$GPU_BIN"
        PZ_GPU="$GPU_BIN"
    fi
    rm -f "$GPU_BUILD_OUT"
fi

# ---- Corpus files ----
"$SCRIPT_DIR/setup.sh" 2>/dev/null || true
FILES=()
if [[ "$CORPUS" == "cantrbry" || "$CORPUS" == "both" ]]; then
    for f in "$PROJECT_DIR/samples/cantrbry/"*; do
        [[ -f "$f" ]] || continue
        case "$f" in *.tar.gz|*.pz|*.gz) continue ;; esac
        FILES+=("$f")
    done
fi
if [[ "$CORPUS" == "large" || "$CORPUS" == "both" ]]; then
    for f in "$PROJECT_DIR/samples/large/"*; do
        [[ -f "$f" ]] || continue
        case "$f" in *.tar.gz|*.pz|*.gz) continue ;; esac
        FILES+=("$f")
    done
fi
[[ ${#FILES[@]} -gt 0 ]] || { echo "ERROR: no corpus files found" >&2; exit 1; }

# ---- Timing helpers (same as bench.sh) ----
time_ns() {
    local start end
    start=$(perl -MTime::HiRes=time -e 'printf "%d", time * 1e9')
    "$@" >/dev/null 2>&1
    end=$(perl -MTime::HiRes=time -e 'printf "%d", time * 1e9')
    echo $(( end - start ))
}
avg_ns() {
    local total=0 ns
    for (( i=0; i<ITERATIONS; i++ )); do
        ns=$(time_ns "$@")
        total=$(( total + ns ))
    done
    echo $(( total / ITERATIONS ))
}
fmt_throughput() {
    local bytes=$1 ns=$2
    [[ "$ns" -le 0 ]] && { echo "0.0"; return; }
    awk "BEGIN { printf \"%.2f\", ($bytes / 1048576.0) / ($ns / 1000000000.0); exit }" < /dev/null
}
fmt_ratio() {
    awk "BEGIN { printf \"%.4f\", ($1 / $2) * 100.0; exit }" < /dev/null
}

# ---- Output setup ----
mkdir -p "$PROJECT_DIR/docs/generated"
echo "corpus,codec,level,threads,ratio_pct,throughput_mbs" > "$OUT_CSV"

TMPDIR_BENCH=$(mktemp -d)
trap 'rm -rf "$TMPDIR_BENCH"' EXIT

# ---- Helper: append one row to CSV ----
# append_row CORPUS CODEC LEVEL THREADS RATIO_PCT THROUGHPUT_MBS
append_row() {
    echo "$1,$2,$3,$4,$5,$6" >> "$OUT_CSV"
}

# ---- Competitor availability ----
HAS_GZIP=true   # always required
HAS_ZLIBBNG=false; HAS_LZ4=false; HAS_ZSTD=false
command -v minigzip-ng &>/dev/null && HAS_ZLIBBNG=true
command -v lz4         &>/dev/null && HAS_LZ4=true
command -v zstd        &>/dev/null && HAS_ZSTD=true

# ---- Accumulator helpers ----
# We run per-file and accumulate for corpus-level totals.
# For each (codec, level, threads) config, maintain total_orig, total_comp, total_ns.

declare -A acc_orig acc_comp acc_ns

record() {
    local key="$1" orig="$2" comp="$3" ns="$4"
    acc_orig[$key]=$(( ${acc_orig[$key]:-0} + orig ))
    acc_comp[$key]=$(( ${acc_comp[$key]:-0} + comp ))
    acc_ns[$key]=$(( ${acc_ns[$key]:-0} + ns ))
}

# ---- Per-file loop ----
for FILE in "${FILES[@]}"; do
    FNAME=$(basename "$FILE")
    ORIG=$(stat -c%s "$FILE" 2>/dev/null || stat -f%z "$FILE" 2>/dev/null)
    CORPUS_NAME=$(basename "$(dirname "$FILE")")

    [[ "$VERBOSE" == true ]] && echo "  $FNAME ($ORIG bytes)"

    # -- gzip levels 1-9 --
    for level in 1 2 3 4 5 6 7 8 9; do
        cp "$FILE" "$TMPDIR_BENCH/$FNAME"
        ns=$(avg_ns gzip -"$level" -k -f "$TMPDIR_BENCH/$FNAME")
        comp=$(stat -c%s "$TMPDIR_BENCH/$FNAME.gz" 2>/dev/null || stat -f%z "$TMPDIR_BENCH/$FNAME.gz" 2>/dev/null)
        rm -f "$TMPDIR_BENCH/$FNAME" "$TMPDIR_BENCH/$FNAME.gz"
        record "$CORPUS_NAME|gzip|$level|1" "$ORIG" "$comp" "$ns"
    done

    # -- zstd levels 1-19 --
    if [[ "$HAS_ZSTD" == true ]]; then
        for level in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19; do
            zst_out="$TMPDIR_BENCH/$FNAME.zst"
            ns=$(avg_ns bash -c "zstd -$level -q --single-thread -c '$FILE' > '$zst_out'")
            comp=$(stat -c%s "$zst_out" 2>/dev/null || stat -f%z "$zst_out" 2>/dev/null)
            rm -f "$zst_out"
            record "$CORPUS_NAME|zstd|$level|1" "$ORIG" "$comp" "$ns"
        done
    fi

    # -- lz4 levels 1-12 --
    if [[ "$HAS_LZ4" == true ]]; then
        for level in 1 2 3 4 5 6 7 8 9 10 11 12; do
            lz4_out="$TMPDIR_BENCH/$FNAME.lz4"
            ns=$(avg_ns bash -c "lz4 -$level -q -c '$FILE' > '$lz4_out'")
            comp=$(stat -c%s "$lz4_out" 2>/dev/null || stat -f%z "$lz4_out" 2>/dev/null)
            rm -f "$lz4_out"
            record "$CORPUS_NAME|lz4|$level|1" "$ORIG" "$comp" "$ns"
        done
    fi

    # -- libpz: all pipelines × parse strategies × thread counts --
    PZ_PIPELINES=(deflate bw bbw lzr lzf lzfi lzseqr lzseqh)
    for pipeline in "${PZ_PIPELINES[@]}"; do
        for threads in 1 4; do
            # greedy (fastest)
            cp "$FILE" "$TMPDIR_BENCH/$FNAME"
            ns=$(avg_ns "$PZ" -k -f -p "$pipeline" -t "$threads" --greedy "$TMPDIR_BENCH/$FNAME")
            comp=$(stat -c%s "$TMPDIR_BENCH/$FNAME.pz" 2>/dev/null || stat -f%z "$TMPDIR_BENCH/$FNAME.pz" 2>/dev/null)
            rm -f "$TMPDIR_BENCH/$FNAME" "$TMPDIR_BENCH/$FNAME.pz"
            record "$CORPUS_NAME|pz-$pipeline-greedy|0|$threads" "$ORIG" "$comp" "$ns"

            # lazy (default)
            cp "$FILE" "$TMPDIR_BENCH/$FNAME"
            ns=$(avg_ns "$PZ" -k -f -p "$pipeline" -t "$threads" --lazy "$TMPDIR_BENCH/$FNAME")
            comp=$(stat -c%s "$TMPDIR_BENCH/$FNAME.pz" 2>/dev/null || stat -f%z "$TMPDIR_BENCH/$FNAME.pz" 2>/dev/null)
            rm -f "$TMPDIR_BENCH/$FNAME" "$TMPDIR_BENCH/$FNAME.pz"
            record "$CORPUS_NAME|pz-$pipeline-lazy|0|$threads" "$ORIG" "$comp" "$ns"

            # optimal (slowest, best ratio) — only BWT pipelines and lzseqr for now
            case "$pipeline" in deflate|lzseqr|bw|bbw)
                cp "$FILE" "$TMPDIR_BENCH/$FNAME"
                ns=$(avg_ns "$PZ" -k -f -p "$pipeline" -t "$threads" -O "$TMPDIR_BENCH/$FNAME")
                comp=$(stat -c%s "$TMPDIR_BENCH/$FNAME.pz" 2>/dev/null || stat -f%z "$TMPDIR_BENCH/$FNAME.pz" 2>/dev/null)
                rm -f "$TMPDIR_BENCH/$FNAME" "$TMPDIR_BENCH/$FNAME.pz"
                record "$CORPUS_NAME|pz-$pipeline-optimal|0|$threads" "$ORIG" "$comp" "$ns"
                ;;
            esac
        done
    done

    # auto-select (threads=1 and threads=4) — runs once per file, not per pipeline
    for threads in 1 4; do
        cp "$FILE" "$TMPDIR_BENCH/$FNAME"
        ns=$(avg_ns "$PZ" -k -f -a -t "$threads" "$TMPDIR_BENCH/$FNAME")
        comp=$(stat -c%s "$TMPDIR_BENCH/$FNAME.pz" 2>/dev/null || stat -f%z "$TMPDIR_BENCH/$FNAME.pz" 2>/dev/null)
        rm -f "$TMPDIR_BENCH/$FNAME" "$TMPDIR_BENCH/$FNAME.pz"
        record "$CORPUS_NAME|pz-auto|0|$threads" "$ORIG" "$comp" "$ns"
    done

    # -- GPU configs (optional) --
    if [[ -n "$PZ_GPU" ]]; then
        for pipeline in lzseqr bw bbw; do
            cp "$FILE" "$TMPDIR_BENCH/$FNAME"
            ns=$(avg_ns "$PZ_GPU" -k -f -p "$pipeline" -t 1 --lazy --gpu "$TMPDIR_BENCH/$FNAME")
            comp=$(stat -c%s "$TMPDIR_BENCH/$FNAME.pz" 2>/dev/null || stat -f%z "$TMPDIR_BENCH/$FNAME.pz" 2>/dev/null)
            rm -f "$TMPDIR_BENCH/$FNAME" "$TMPDIR_BENCH/$FNAME.pz"
            record "$CORPUS_NAME|pz-$pipeline-gpu|0|1" "$ORIG" "$comp" "$ns"
        done
    fi
done

# ---- Write CSV from accumulators ----
for key in "${!acc_orig[@]}"; do
    IFS='|' read -r corpus codec level threads <<< "$key"
    total_orig="${acc_orig[$key]}"
    total_comp="${acc_comp[$key]}"
    total_ns="${acc_ns[$key]}"
    ratio=$(fmt_ratio "$total_comp" "$total_orig")
    tput=$(fmt_throughput "$total_orig" "$total_ns")
    append_row "$corpus" "$codec" "$level" "$threads" "$ratio" "$tput"
done

# ---- Human-readable summary sorted by ratio ----
echo ""
echo "=== PARETO BENCHMARK RESULTS ($CORPUS corpus) ==="
echo ""
printf "  %-30s %6s %10s %12s\n" "Codec (level, threads)" "Ratio" "MB/s" "Total comp"
printf "  %s\n" "──────────────────────────────────────────────────────────────────"

# Print rows sorted ratio ascending (best ratio first) using CSV
tail -n +2 "$OUT_CSV" | sort -t',' -k5,5n | while IFS=',' read -r _corpus codec level threads ratio tput; do
    label="$codec (L$level, t$threads)"
    printf "  %-30s %6s%% %8s MB/s\n" "$label" "$ratio" "$tput"
done

echo ""
echo "CSV written to: $OUT_CSV"
```

Make the script executable:
```bash
chmod +x scripts/pareto-bench.sh
```

**Step 2: Commit**

```bash
git add scripts/pareto-bench.sh
git commit -m "scripts: add pareto-bench.sh for full competitor-level sweep benchmark"
```
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Run benchmark and capture results

**Files:**
- Create: `docs/generated/pareto-results.csv` (generated artifact)

**Background:**

With `pareto-bench.sh` in place, run it against the Canterbury corpus and save the results. These numbers are the raw material for visualization (Task 5), AC verification (Tasks 3, 6-8), and gap analysis (Task 9). Run with `--iters 5` for stable timing on the relatively small Canterbury files.

**Step 1: Install missing competitors if needed**

```bash
brew install zlib-ng lz4 zstd 2>/dev/null || true
```

**Step 2: Run the full benchmark**

```bash
/Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/scripts/pareto-bench.sh \
    --corpus cantrbry \
    --iters 5 \
    2>&1 | tee /tmp/pareto-bench-cantrbry.txt
```

Expected runtime: 5-15 minutes depending on machine (gzip levels 1-9 × 11 files × 5 iters + zstd 19 levels + lz4 12 levels + all pz configs).

**Step 3: Also run on the large corpus**

```bash
/Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/scripts/pareto-bench.sh \
    --corpus large \
    --iters 3 \
    2>&1 | tee /tmp/pareto-bench-large.txt
```

**Step 4: Verify CSV structure**

```bash
head -5 /Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/docs/generated/pareto-results.csv
```

Expected: header row `corpus,codec,level,threads,ratio_pct,throughput_mbs` followed by data rows with valid numeric values for ratio and throughput.

**Step 5: Commit**

```bash
git add docs/generated/pareto-results.csv
git commit -m "docs: capture pareto benchmark results for Canterbury and large corpora"
```
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Verify AC2.1–AC2.4 (single-thread baseline results)

**Files:**
- No file modifications — this is a verification task using the CSV from Task 2.

**Background:**

AC2 requires that `pz -t 1` with LzSeqR matches gzip -6 on Canterbury corpus for both throughput and ratio, that decompression throughput is competitive, and that sub-1KB inputs work correctly. The CSV from Task 2 contains the throughput and ratio numbers needed for AC2.1 and AC2.2. AC2.3 (decompression) and AC2.4 (small inputs) need targeted runs that `pareto-bench.sh` does not cover.

**Step 1: Verify AC2.1 — LzSeqR single-thread compression throughput >= gzip -6**

Extract from the CSV:
```bash
CSV=/Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/docs/generated/pareto-results.csv
echo "gzip level 6 (t=1):"
grep "^cantrbry,gzip,6,1," "$CSV"
echo "pz-lzseqr-lazy (t=1):"
grep "^cantrbry,pz-lzseqr-lazy,0,1," "$CSV"
```

Pass condition: `throughput_mbs` for `pz-lzseqr-lazy,0,1` >= `throughput_mbs` for `gzip,6,1`.

If LzSeqR single-thread throughput is lower than gzip -6, document the gap in the gap analysis (Task 9) as a known delta rather than treating it as a blocker. The Phase 7 auto-selection work should have closed this gap; if not, record the delta.

**Step 2: Verify AC2.2 — LzSeqR ratio within 2% of gzip -6 on Canterbury**

From the CSV:
```bash
grep "^cantrbry,gzip,6,1," "$CSV" | awk -F',' '{print "gzip-6 ratio:", $5"%"}'
grep "^cantrbry,pz-lzseqr-lazy,0,1," "$CSV" | awk -F',' '{print "lzseqr ratio:", $5"%"}'
```

Pass condition: `|ratio(lzseqr) - ratio(gzip-6)| <= 2.0` percentage points.

Known state (2026-02-22): LzSeqR achieves 32.0% vs gzip's 28.6% — a ~3.4 point gap. If this gap persists after Phase 7 work, document it as a known gap in Task 9 rather than treating it as a blocker for Phase 8 completion. Phase 8 validates the current state; closing the remaining gap is follow-on work.

**Step 3: Verify AC2.3 — decompression throughput >= gzip**

Run a targeted decompression comparison on the Canterbury corpus:
```bash
/Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/scripts/bench.sh \
    -t 1 -p lzseqr -n 5 \
    2>&1 | grep -A 10 "Decompression Results"
```

Pass condition: The `pz-lzseqr` decompression MB/s >= `gzip` decompression MB/s in the output table.

**Step 4: Verify AC2.4 — single-thread works for inputs < 1KB**

Run against all small Canterbury files and a synthetic 100-byte file:
```bash
PZ=/Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/target/release/pz
TMPDIR_AC24=$(mktemp -d)
trap 'rm -rf "$TMPDIR_AC24"' EXIT

# Generate a small file
python3 -c "import os; open('$TMPDIR_AC24/tiny.txt', 'wb').write(os.urandom(100))"

for f in /Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/samples/cantrbry/* \
         "$TMPDIR_AC24/tiny.txt"; do
    [[ -f "$f" ]] || continue
    orig=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null)
    [[ "$orig" -gt 1024 ]] && continue  # only small files
    cp "$f" "$TMPDIR_AC24/test_input"
    "$PZ" -k -f -p lzseqr -t 1 "$TMPDIR_AC24/test_input"
    "$PZ" -d -k -f "$TMPDIR_AC24/test_input.pz" -c > "$TMPDIR_AC24/test_roundtrip"
    if cmp -s "$TMPDIR_AC24/test_input" "$TMPDIR_AC24/test_roundtrip"; then
        echo "PASS: $(basename "$f") ($orig bytes) roundtrip OK"
    else
        echo "FAIL: $(basename "$f") ($orig bytes) roundtrip mismatch"
    fi
    rm -f "$TMPDIR_AC24/test_input" "$TMPDIR_AC24/test_input.pz" "$TMPDIR_AC24/test_roundtrip"
done
```

Pass condition: All small files print `PASS`. No crash, no hang.

**Step 5: Document AC2 results**

Record pass/fail for each AC in Task 9's gap analysis document. If any AC2 criterion fails, note the measured delta (e.g., "AC2.2: ratio gap is 3.4pp vs 2.0pp target — 1.4pp remaining after Phase 7").

**If AC2.3 fails:** Document the gap and identify which decode path is the bottleneck. Create a follow-on optimization task targeting the specific decode stage (rANS, Huffman, or LzSeq token decode).
<!-- END_TASK_3 -->
<!-- END_SUBCOMPONENT_A -->

---

<!-- START_SUBCOMPONENT_B -->
## Subcomponent B: Pareto curve visualization

This subcomponent produces speed-vs-ratio scatter plots from the CSV collected in Subcomponent A, making it visually obvious where libpz sits relative to competitors.

<!-- START_TASK_4 -->
### Task 4: Create scripts/plot-pareto.py

**Files:**
- Create: `scripts/plot-pareto.py`

**Background:**

There is no existing visualization script in the repo. The Phase 1 infrastructure (`bench.sh`, `analyze-ratio.sh`) only produces text tables. A Python matplotlib script reading the `pareto-results.csv` from Task 2 will produce standard speed-vs-ratio Pareto scatter plots with one point per (codec, level, thread-count) configuration.

Axes convention: x-axis = compression ratio % (lower = better, so x increases leftward or the plot is read right-to-left for "better"), y-axis = throughput MB/s (higher = better). The standard presentation in compression literature plots ratio on x (lower-is-better) and throughput on y (higher-is-better), so a config dominates another if it is to the left and above.

The script produces one PNG per corpus. Points are color-coded by tool family (gzip=blue, zstd=orange, lz4=green, pz=red), with level annotated as small labels.

**Step 1: Create `scripts/plot-pareto.py`**

```python
#!/usr/bin/env python3
"""plot-pareto.py — Pareto curve visualization for pareto-results.csv.

Reads docs/generated/pareto-results.csv and produces:
  docs/generated/pareto-cantrbry.png
  docs/generated/pareto-large.png  (if data present)

Usage:
  python3 scripts/plot-pareto.py [--csv PATH] [--out-dir DIR] [--corpus NAME]
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("ERROR: matplotlib not installed. Run: pip install matplotlib", file=sys.stderr)
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=None, help="Path to pareto-results.csv")
    p.add_argument("--out-dir", default=None, help="Directory for output PNGs")
    p.add_argument("--corpus", default=None, help="Filter to one corpus (cantrbry or large)")
    return p.parse_args()


def tool_family(codec: str) -> str:
    """Return the tool family for color assignment."""
    if codec.startswith("gzip"):
        return "gzip"
    if codec.startswith("zstd"):
        return "zstd"
    if codec.startswith("lz4"):
        return "lz4"
    if codec.startswith("zlib-ng") or codec.startswith("minigzip"):
        return "zlib-ng"
    if codec.startswith("pz-"):
        return "pz"
    return "other"


FAMILY_COLORS = {
    "gzip":   "#2196F3",   # blue
    "zlib-ng": "#00BCD4",  # cyan
    "zstd":   "#FF9800",   # orange
    "lz4":    "#4CAF50",   # green
    "pz":     "#F44336",   # red
    "other":  "#9E9E9E",   # grey
}

FAMILY_MARKERS = {
    "gzip":   "o",
    "zlib-ng": "s",
    "zstd":   "^",
    "lz4":    "D",
    "pz":     "*",
    "other":  "x",
}


def plot_corpus(rows, corpus_name: str, out_path: str):
    """Produce one Pareto scatter plot for the given corpus rows."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Group by family for legend
    family_plotted = set()

    for row in rows:
        codec = row["codec"]
        level = row["level"]
        threads = int(row["threads"])
        ratio = float(row["ratio_pct"])
        tput = float(row["throughput_mbs"])

        family = tool_family(codec)
        color = FAMILY_COLORS.get(family, "#9E9E9E")
        marker = FAMILY_MARKERS.get(family, "x")

        # Use larger markers for pz points
        size = 120 if family == "pz" else 60

        label_str = f"{codec} L{level}" if int(level) > 0 else codec
        if threads > 1:
            label_str += f" t{threads}"

        sc = ax.scatter(ratio, tput, c=color, marker=marker, s=size,
                        alpha=0.85, zorder=3, label=family if family not in family_plotted else "")
        family_plotted.add(family)

        # Annotate each point with a small label
        ax.annotate(
            label_str,
            (ratio, tput),
            fontsize=5,
            alpha=0.7,
            xytext=(3, 3),
            textcoords="offset points",
        )

    # Draw Pareto frontier for pz points
    pz_rows = [(float(r["ratio_pct"]), float(r["throughput_mbs"]))
               for r in rows if tool_family(r["codec"]) == "pz"]
    if pz_rows:
        pz_rows.sort(key=lambda x: x[0])  # sort by ratio ascending
        frontier = []
        best_tput = -1.0
        for ratio, tput in pz_rows:
            if tput > best_tput:
                frontier.append((ratio, tput))
                best_tput = tput
        if frontier:
            fx = [p[0] for p in frontier]
            fy = [p[1] for p in frontier]
            ax.step(fx, fy, where="post", color="#F44336", linewidth=1.5,
                    linestyle="--", alpha=0.6, label="pz frontier")

    # Legend
    legend_handles = [
        mpatches.Patch(color=FAMILY_COLORS[f], label=f)
        for f in sorted(family_plotted)
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    ax.set_xlabel("Compression ratio % (lower = better compression)", fontsize=11)
    ax.set_ylabel("Throughput MB/s (higher = faster)", fontsize=11)
    ax.set_title(f"Pareto: speed vs ratio — {corpus_name} corpus", fontsize=13)
    ax.grid(True, alpha=0.3)

    # Invert x-axis so best ratio is on the right (matches compression literature convention)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote: {out_path}")


def main():
    args = parse_args()

    # Locate CSV
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    csv_path = args.csv or str(project_dir / "docs" / "generated" / "pareto-results.csv")
    out_dir = args.out_dir or str(project_dir / "docs" / "generated")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        print("Run scripts/pareto-bench.sh first.", file=sys.stderr)
        sys.exit(1)

    # Load CSV
    corpora: dict[str, list] = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.corpus and row["corpus"] != args.corpus:
                continue
            corpora[row["corpus"]].append(row)

    if not corpora:
        print("ERROR: No data rows found in CSV (check --corpus filter)", file=sys.stderr)
        sys.exit(1)

    for corpus_name, rows in corpora.items():
        out_path = os.path.join(out_dir, f"pareto-{corpus_name}.png")
        print(f"Plotting {corpus_name}: {len(rows)} data points")
        plot_corpus(rows, corpus_name, out_path)

    print("Done.")


if __name__ == "__main__":
    main()
```

Make executable:
```bash
chmod +x scripts/plot-pareto.py
```

**Step 2: Commit**

```bash
git add scripts/plot-pareto.py
git commit -m "scripts: add plot-pareto.py matplotlib Pareto curve visualization"
```
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: Generate Pareto plots for Canterbury and large corpora

**Files:**
- Create: `docs/generated/pareto-cantrbry.png` (generated artifact)
- Create: `docs/generated/pareto-large.png` (generated artifact, if large corpus data present)

**Background:**

With the CSV from Task 2 and the plot script from Task 4, generate the actual PNG plots. These become the visual artifacts in the gap analysis document (Task 9) and serve as the reproducible evidence that Phase 8 is complete.

**Step 1: Install matplotlib if needed**

```bash
pip install matplotlib 2>/dev/null || pip3 install matplotlib
```

**Step 2: Generate Canterbury plot**

```bash
python3 /Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/scripts/plot-pareto.py \
    --corpus cantrbry
```

Expected: `docs/generated/pareto-cantrbry.png` created. Output line: `Wrote: .../pareto-cantrbry.png`.

**Step 3: Generate large corpus plot**

```bash
python3 /Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/scripts/plot-pareto.py \
    --corpus large
```

Expected: `docs/generated/pareto-large.png` created (if large corpus data is present in the CSV).

**Step 4: Inspect plots**

Open the PNGs to verify:
- Red `*` markers (pz points) are visible on the plot.
- Blue circles (gzip) form a visible curve from low-ratio/slow (level 9) to high-ratio/fast (level 1).
- The pz Pareto frontier dashed line connects the dominant pz points.
- Axes are labeled and title is correct.

**Step 5: Commit**

```bash
git add docs/generated/pareto-cantrbry.png docs/generated/pareto-large.png
git commit -m "docs: add Pareto curve visualizations for Canterbury and large corpora"
```
<!-- END_TASK_5 -->
<!-- END_SUBCOMPONENT_B -->

---

<!-- START_SUBCOMPONENT_C -->
## Subcomponent C: Validation and gap analysis

This subcomponent formally verifies each remaining AC against the measured data and produces the final gap analysis document.

<!-- START_TASK_6 -->
### Task 6: Verify AC1.5 — Pareto-competitive for each gzip level

**Files:**
- No file modifications — verification task using the CSV from Task 2.

**Background:**

AC1.5 requires that for each gzip level 1-9, at least one libpz configuration exists that is either (a) faster at the same or better ratio, or (b) achieves better ratio at the same or better speed. This is the core Pareto-dominance check.

The test runs an awk script over `pareto-results.csv` that, for each gzip level, finds the best libpz point that dominates it.

**Step 1: Run the Pareto-dominance check**

```bash
python3 - <<'EOF'
import csv
from pathlib import Path

csv_path = Path("/Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/docs/generated/pareto-results.csv")

rows = []
with open(csv_path) as f:
    for row in csv.DictReader(f):
        if row["corpus"] != "cantrbry":
            continue
        rows.append({
            "codec": row["codec"],
            "level": int(row["level"]),
            "threads": int(row["threads"]),
            "ratio": float(row["ratio_pct"]),
            "tput": float(row["throughput_mbs"]),
        })

gzip_points = {r["level"]: r for r in rows if r["codec"] == "gzip" and r["threads"] == 1}
pz_points = [r for r in rows if r["codec"].startswith("pz-")]

print(f"{'gzip level':>12}  {'gzip ratio':>10}  {'gzip tput':>9}  {'best pz config':>30}  {'pz ratio':>9}  {'pz tput':>9}  {'dominates?':>12}")
print("-" * 100)

all_pass = True
for level in sorted(gzip_points.keys()):
    gz = gzip_points[level]
    # Find any pz point that dominates: (ratio <= gz.ratio AND tput >= gz.tput)
    # OR (ratio < gz.ratio AND tput >= gz.tput * 0.95)  # within 5% on throughput
    # OR (ratio <= gz.ratio * 1.005 AND tput > gz.tput)  # within 0.5% on ratio
    dominators = []
    for pz in pz_points:
        faster_same_ratio = (pz["tput"] >= gz["tput"]) and (pz["ratio"] <= gz["ratio"] * 1.001)
        better_ratio_same_speed = (pz["ratio"] < gz["ratio"]) and (pz["tput"] >= gz["tput"] * 0.99)
        if faster_same_ratio or better_ratio_same_speed:
            dominators.append(pz)
    if dominators:
        best = max(dominators, key=lambda r: r["tput"])
        label = f"{best['codec']} t{best['threads']}"
        print(f"{level:>12}  {gz['ratio']:>9.2f}%  {gz['tput']:>8.1f}  {label:>30}  {best['ratio']:>8.2f}%  {best['tput']:>8.1f}  PASS")
    else:
        # Find closest pz point even if not dominating
        closest = min(pz_points, key=lambda r: abs(r["ratio"] - gz["ratio"]))
        label = f"{closest['codec']} t{closest['threads']}"
        print(f"{level:>12}  {gz['ratio']:>9.2f}%  {gz['tput']:>8.1f}  {label:>30}  {closest['ratio']:>8.2f}%  {closest['tput']:>8.1f}  GAP")
        all_pass = False

print()
if all_pass:
    print("AC1.5: PASS — libpz dominates every gzip level on Canterbury corpus")
else:
    print("AC1.5: PARTIAL — some gzip levels not yet dominated (see GAP rows)")
EOF
```

Pass condition: All 9 gzip levels print `PASS`. If any print `GAP`, document the specific levels and delta in Task 9.

**Step 2: Re-run with large corpus data**

```bash
# Repeat the above script with corpus="large" substituted for "cantrbry"
```

Pass condition: Same dominance holds on the large corpus.
<!-- END_TASK_6 -->

<!-- START_TASK_7 -->
### Task 7: Verify AC1.6 — incompressible data safety

**Files:**
- No file modifications — verification task.

**Background:**

AC1.6 is a Failure-mode AC: compressing random (incompressible) data must not crash and must not produce output larger than `original_size * 1.01 + 64` bytes (1% overhead + 64 bytes for headers). The Canterbury corpus includes `random.txt` (random bytes), which is the primary test case. We additionally test with a large random block and a zero-byte file.

**Step 1: Test all pipelines on incompressible data**

```bash
PZ=/Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/target/release/pz
TMPDIR_INCOMP=$(mktemp -d)
trap 'rm -rf "$TMPDIR_INCOMP"' EXIT

# Generate test inputs
python3 -c "import os; open('$TMPDIR_INCOMP/random_1mb.bin', 'wb').write(os.urandom(1048576))"
python3 -c "open('$TMPDIR_INCOMP/empty.bin', 'wb').close()"
cp /Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/samples/cantrbry/random.txt \
   "$TMPDIR_INCOMP/random_corpus.txt" 2>/dev/null || true

OVERHEAD_LIMIT_PCT=101  # 1% overhead = 101% of original
ALL_PASS=true

for input in "$TMPDIR_INCOMP/random_1mb.bin" "$TMPDIR_INCOMP/empty.bin" \
             "$TMPDIR_INCOMP/random_corpus.txt"; do
    [[ -f "$input" ]] || continue
    orig=$(stat -c%s "$input" 2>/dev/null || stat -f%z "$input" 2>/dev/null)
    max_allowed=$(awk "BEGIN { printf \"%d\", $orig * 1.01 + 64; exit }" < /dev/null)

    for pipeline in deflate bw bbw lzr lzf lzfi lzseqr lzseqh; do
        cp "$input" "$TMPDIR_INCOMP/test_input"
        if ! "$PZ" -k -f -p "$pipeline" -t 1 "$TMPDIR_INCOMP/test_input" 2>/dev/null; then
            echo "FAIL: $pipeline CRASHED on $(basename "$input")"
            ALL_PASS=false
            continue
        fi
        comp=$(stat -c%s "$TMPDIR_INCOMP/test_input.pz" 2>/dev/null || stat -f%z "$TMPDIR_INCOMP/test_input.pz" 2>/dev/null)
        if [[ "$comp" -gt "$max_allowed" ]]; then
            echo "FAIL: $pipeline output $comp > $max_allowed on $(basename "$input") ($orig bytes)"
            ALL_PASS=false
        else
            echo "PASS: $pipeline $(basename "$input") orig=$orig comp=$comp max=$max_allowed"
        fi
        rm -f "$TMPDIR_INCOMP/test_input" "$TMPDIR_INCOMP/test_input.pz"
    done
done

[[ "$ALL_PASS" == true ]] && echo "AC1.6: PASS" || echo "AC1.6: FAIL — see above"
```

Pass condition: All pipelines on all inputs print `PASS`. No crashes, no oversized output.
<!-- END_TASK_7 -->

<!-- START_TASK_8 -->
### Task 8: Verify AC3.1 — multi-thread near-linear speedup to 4 threads

**Files:**
- No file modifications — verification task using the CSV from Task 2.

**Background:**

AC3.1 requires near-linear speedup up to 4 threads on 1MB+ inputs. "Near-linear" is defined as speedup >= 3.0× at 4 threads (75% parallel efficiency). The Canterbury corpus includes files large enough (enwik8 via large corpus, or the combined corpus at 3.7MB). The `pareto-results.csv` has both `t=1` and `t=4` rows for each pz config.

**Step 1: Extract t=1 vs t=4 throughput comparison from CSV**

```bash
python3 - <<'EOF'
import csv
from pathlib import Path

csv_path = Path("/Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/docs/generated/pareto-results.csv")

t1 = {}
t4 = {}
with open(csv_path) as f:
    for row in csv.DictReader(f):
        if row["corpus"] not in ("cantrbry", "large"):
            continue
        key = (row["corpus"], row["codec"], row["level"])
        t = int(row["threads"])
        tput = float(row["throughput_mbs"])
        if t == 1:
            t1[key] = tput
        elif t == 4:
            t4[key] = tput

print(f"{'Corpus':>10}  {'Codec':>30}  {'t=1 MB/s':>10}  {'t=4 MB/s':>10}  {'Speedup':>8}  {'AC3.1?':>8}")
print("-" * 90)

all_pass = True
for key in sorted(t1.keys()):
    if key not in t4:
        continue
    corpus, codec, level = key
    if not codec.startswith("pz-"):
        continue
    s1 = t1[key]
    s4 = t4[key]
    speedup = s4 / s1 if s1 > 0 else 0.0
    passing = speedup >= 3.0
    if not passing:
        all_pass = False
    mark = "PASS" if passing else "GAP"
    print(f"{corpus:>10}  {codec:>30}  {s1:>10.1f}  {s4:>10.1f}  {speedup:>8.2f}x  {mark:>8}")

print()
if all_pass:
    print("AC3.1: PASS — all pz pipelines achieve >= 3.0x speedup at 4 threads")
else:
    print("AC3.1: PARTIAL — some pipelines below 3.0x (see GAP rows)")
EOF
```

Pass condition: All pz pipeline rows show speedup >= 3.0×. If any fall short, document the specific pipeline and measured speedup in Task 9.

**Step 2: Verify on a file >= 1MB specifically**

The AC requires "1MB+ inputs". Canterbury files range from 1KB to ~4MB; the large corpus files are 10.7MB total. If the CSV only has combined-corpus totals, run a targeted check on a single large file:

```bash
PZ=/Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/target/release/pz
LARGE_FILE=/Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/samples/large/enwik8

if [[ -f "$LARGE_FILE" ]]; then
    TMPDIR_T=$(mktemp -d)
    trap 'rm -rf "$TMPDIR_T"' EXIT

    for threads in 1 4; do
        cp "$LARGE_FILE" "$TMPDIR_T/enwik8"
        t=$(perl -MTime::HiRes=time -e 'printf "%d", time * 1e9')
        "$PZ" -k -f -p lzseqr -t "$threads" "$TMPDIR_T/enwik8" >/dev/null 2>&1
        t2=$(perl -MTime::HiRes=time -e 'printf "%d", time * 1e9')
        ns=$(( t2 - t ))
        orig=$(stat -c%s "$LARGE_FILE" 2>/dev/null || stat -f%z "$LARGE_FILE" 2>/dev/null)
        tput=$(awk "BEGIN { printf \"%.1f\", ($orig / 1048576.0) / ($ns / 1000000000.0); exit }" < /dev/null)
        echo "lzseqr t=$threads: $tput MB/s"
        rm -f "$TMPDIR_T/enwik8" "$TMPDIR_T/enwik8.pz"
    done
fi
```

Pass condition: t=4 throughput >= 3.0× t=1 throughput on enwik8.
<!-- END_TASK_8 -->

<!-- START_TASK_9 -->
### Task 9: Document gap analysis in docs/generated/pareto-validation-results.md

**Files:**
- Create: `docs/generated/pareto-validation-results.md`

**Background:**

The final deliverable for Phase 8 is a documented, reproducible validation report. It summarizes which ACs pass, which have gaps, the measured delta for each gap, and what follow-on work would close them. This is the artifact proving "Phase 8 done-when": Pareto-competitive results documented with reproducible benchmarks.

**Step 1: Create `docs/generated/pareto-validation-results.md`**

Write the document with the actual measured numbers from Tasks 2-8. The template below shows the structure; fill in real numbers from the CSV and verification outputs before committing.

```markdown
# Pareto Validation Results — 2026-02-22

Generated by: `scripts/pareto-bench.sh --corpus cantrbry --iters 5`
Visualized by: `scripts/plot-pareto.py`
Corpora: Canterbury (11 files, 3.7 MB), large (3 files, 10.7 MB)
Machine: [fill in: CPU model, core count, RAM, GPU if relevant]
libpz commit: [fill in: git rev-parse HEAD]

---

## Summary

| AC | Description | Result | Delta |
|----|-------------|--------|-------|
| AC1.5 | Pareto-competitive with each gzip level 1-9 | [PASS/PARTIAL] | [e.g., levels 7-9 gap: X.Xpp ratio] |
| AC1.6 | Incompressible data safety | [PASS/FAIL] | — |
| AC2.1 | LzSeqR t=1 throughput >= gzip -6 | [PASS/FAIL] | [measured gap if any] |
| AC2.2 | LzSeqR t=1 ratio within 2% of gzip -6 | [PASS/FAIL] | [measured delta] |
| AC2.3 | t=1 decompression >= gzip | [PASS/FAIL] | [measured delta] |
| AC2.4 | Single-thread < 1KB inputs work | PASS | — |
| AC3.1 | 4-thread speedup >= 3.0× on 1MB+ | [PASS/PARTIAL] | [pipeline-level detail] |

---

## AC1.5: Pareto-competitiveness by gzip level

[Fill in output from Task 6 verification script]

### Gaps (if any)

[List any gzip levels where no libpz config dominates, with the closest pz config and measured delta]

---

## AC1.6: Incompressible data safety

All pipelines tested on: `random_1mb.bin` (1 MB random), `empty.bin` (0 bytes), `random.txt` (Canterbury random corpus file).

[Fill in output from Task 7 verification script]

---

## AC2.1–AC2.4: Single-thread CPU baseline

### AC2.1 — Compression throughput vs gzip -6

| Codec | Throughput MB/s | vs gzip -6 |
|-------|----------------|------------|
| gzip -6 | [value] | baseline |
| pz lzseqr t=1 lazy | [value] | [+X% or -X%] |

### AC2.2 — Compression ratio vs gzip -6

| Codec | Ratio % | Delta from gzip -6 |
|-------|---------|-------------------|
| gzip -6 | [value]% | baseline |
| pz lzseqr t=1 lazy | [value]% | [+X.Xpp] |

Known state going in: LzSeqR measured at 32.0% vs gzip -6 at 28.6% — a 3.4pp gap before Phase 7.
Post-Phase-7 measured delta: [fill in].

### AC2.3 — Decompression throughput vs gzip

| Codec | Decomp MB/s | vs gzip |
|-------|-------------|---------|
| gzip | [value] | baseline |
| pz lzseqr t=1 | [value] | [+X% or -X%] |

### AC2.4 — Sub-1KB input correctness

[Fill in from Task 3, Step 4 — list files tested and roundtrip results]

---

## AC3.1: Multi-thread scaling

| Corpus | Codec | t=1 MB/s | t=4 MB/s | Speedup | >= 3.0x? |
|--------|-------|----------|----------|---------|----------|
| cantrbry | pz-lzseqr-lazy | [val] | [val] | [val]x | [Y/N] |
| cantrbry | pz-deflate-lazy | [val] | [val] | [val]x | [Y/N] |
| large | pz-lzseqr-lazy | [val] | [val] | [val]x | [Y/N] |

[Fill in from Task 8 output]

---

## Remaining competitive gaps (future work)

[List any ACs that did not fully pass, with:]
- Measured delta
- Root cause (if known)
- Suggested follow-on work
- Priority (blocking vs. nice-to-have)

Example format:
- **AC2.2 ratio gap**: LzSeqR achieves X.X% ratio vs gzip -6's Y.Y% — a Z.Zpp gap (target: <= 2.0pp). Root cause: [match-finding quality / entropy coding overhead / other]. Follow-on: Phase 2 match-finding improvements (already completed?) / Phase 3 optimal parsing.

---

## Visualization

See `docs/generated/pareto-cantrbry.png` and `docs/generated/pareto-large.png` for speed-vs-ratio scatter plots.

---

## Reproducibility

To regenerate these results:

```bash
# 1. Install competitors
brew install zlib-ng lz4 zstd

# 2. Run benchmark (approx 10-20 min)
./scripts/pareto-bench.sh --corpus cantrbry --iters 5
./scripts/pareto-bench.sh --corpus large --iters 3

# 3. Generate plots
pip install matplotlib
python3 scripts/plot-pareto.py

# 4. View results
cat docs/generated/pareto-validation-results.md
open docs/generated/pareto-cantrbry.png
```
```

**Step 2: Fill in measured values**

After running the verification scripts in Tasks 3, 6, 7, and 8, replace all `[fill in]` and `[value]` placeholders with the actual numbers from this machine.

**Step 3: Commit**

```bash
git add docs/generated/pareto-validation-results.md
git commit -m "docs: Phase 8 Pareto validation results and gap analysis"
```
<!-- END_TASK_9 -->
<!-- END_SUBCOMPONENT_C -->

---

## Phase 8 Completion Checklist

- [ ] Task 1: `scripts/pareto-bench.sh` created and produces `docs/generated/pareto-results.csv`
- [ ] Task 2: CSV populated with Canterbury and large corpus data (all competitors × all levels × all pz configs)
- [ ] Task 3: AC2.1–AC2.4 verified; results and any deltas recorded
- [ ] Task 4: `scripts/plot-pareto.py` created
- [ ] Task 5: `docs/generated/pareto-cantrbry.png` and `docs/generated/pareto-large.png` generated
- [ ] Task 6: AC1.5 verified; each gzip level either dominated or gap documented
- [ ] Task 7: AC1.6 verified; all pipelines pass incompressible data safety check
- [ ] Task 8: AC3.1 verified; 4-thread speedup measured and documented
- [ ] Task 9: `docs/generated/pareto-validation-results.md` written with real measured numbers

**Phase 8 done when:** `docs/generated/pareto-validation-results.md` exists with actual measurements (no `[fill in]` placeholders), `pareto-results.csv` is committed, and the PNG plots are generated. Any ACs that did not fully pass are documented with their measured delta and a follow-on work item.
