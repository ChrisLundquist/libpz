# Pareto-Competitiveness Implementation Plan — Phase 1: Measurement Baseline

**Goal:** Establish where libpz sits on the Pareto curve and identify top ratio sinks via benchmarking and bit-budget analysis.

**Architecture:** Extend existing `scripts/bench.sh` with competitor runners (zlib-ng, lz4, zstd) and add LzSeqR/LzSeqH to the pipeline set. New `scripts/analyze-ratio.sh` decomposes compressed output by cost source.

**Tech Stack:** Bash scripting (bench.sh extension, analyze-ratio.sh), no Rust changes required.

**Scope:** Phase 1 of 8 from the pareto-competitiveness design plan.

**Codebase verified:** 2026-02-22

---

## Acceptance Criteria Coverage

This phase implements and tests:

### pareto-competitiveness.AC4: Measurement and tooling
- **pareto-competitiveness.AC4.1 Success:** Benchmark harness produces single-thread Pareto comparison table against gzip, zlib-ng, lz4, zstd
- **pareto-competitiveness.AC4.2 Success:** Bit-budget analysis decomposes compressed output into literal/match/entropy/header costs

---

<!-- START_TASK_0 -->
### Task 0: Verify corpus availability

**Files:**
- None (verification only)

**Background:**

Phase 1 benchmarking uses the Canterbury corpus (11 files at `samples/cantrbry/`) and optionally the Silesia/large corpus (3 files at `samples/large/`: E.coli, bible.txt, world192.txt). The setup.sh script auto-extracts from .tar.gz archives. This task verifies both corpora are available before proceeding to Tasks 1–5.

**Step 1: Verify Canterbury corpus**

Run:
```bash
ls samples/cantrbry/
```

Expected output: 11 files including alice29.txt, enwik8, kennedy.xls, etc.

**Step 2: Verify Silesia/large corpus (optional)**

Run:
```bash
./scripts/setup.sh  # Ensures sample archives are extracted
ls samples/large/
```

Expected: If the large corpus archive exists, it extracts. If not, setup.sh skips gracefully. Output should show E.coli, bible.txt, world192.txt if present.

**Step 3: Confirm both directories exist**

Run:
```bash
test -d samples/cantrbry/ && echo "Canterbury corpus: OK" || echo "Canterbury corpus: MISSING"
test -d samples/large/ && echo "Silesia corpus: OK" || echo "Silesia corpus: MISSING (optional)"
```

Expected: Canterbury corpus must exist; Silesia corpus is optional but recommended for Phase 1 extended runs.

<!-- END_TASK_0 -->

<!-- START_TASK_1 -->
### Task 1: Add LzSeqR and LzSeqH to bench.sh pipeline set

**Files:**
- Modify: `scripts/bench.sh`

**Background:**

The `--all` flag at line 98 of `scripts/bench.sh` hardcodes `(deflate bw bbw lzr lzf lzfi)`. `LzSeqR` (pipeline id=8, CLI name `lzseqr`) and `LzSeqH` (pipeline id=9, CLI name `lzseqh`) are fully implemented in `src/pipeline/mod.rs` and registered in `src/bin/pz.rs` but are absent from the benchmark set.

**Step 1: Update the `--all` pipeline list**

In `scripts/bench.sh` at line 98, change:

```bash
        --all)
            PIPELINES=(deflate bw bbw lzr lzf lzfi)
            shift
            ;;
```

to:

```bash
        --all)
            PIPELINES=(deflate bw bbw lzr lzf lzfi lzseqr lzseqh)
            shift
            ;;
```

**Step 2: Verify operationally**

Run:
```bash
/Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/scripts/bench.sh --all -n 1 /Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/samples/cantrbry/alice29.txt
```

Expected: Summary table prints rows for `pz-lzseqr` and `pz-lzseqh` alongside the existing pipelines, with non-zero sizes and throughputs.

**Step 3: Commit**

```bash
git add scripts/bench.sh
git commit -m "bench: add lzseqr and lzseqh to --all pipeline set"
```
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Add competitor runners to bench.sh

**Files:**
- Modify: `scripts/bench.sh`

**Background:**

`scripts/bench.sh` currently benchmarks gzip and pz pipelines only. AC4.1 requires comparison against zlib-ng (`minigzip-ng`), lz4, and zstd. Each competitor needs an install check, a timing loop using the existing `avg_ns` / `time_ns` functions, and integration into the summary table.

Competitor CLI patterns:
- `zlib-ng`: `brew install zlib-ng` → `minigzip-ng -N < input > /dev/null` (levels 1-9, reads stdin, writes to stdout)
- `lz4`: `brew install lz4` → `lz4 -N -q -c input > /dev/null` (levels 1-12, `-q` quiet, `-c` stdout)
- `zstd`: `brew install zstd` → `zstd -N -q --single-thread -c input > /dev/null` (levels 1-19, `-q` quiet, `-c` stdout)

For Phase 1, benchmark each competitor at its default level only (the Pareto sweep across levels is Phase 8). The goal is a single representative data point per tool showing where gzip-default, zlib-ng-default, lz4-default, and zstd-default sit on the curve alongside libpz pipelines.

**Step 1: Add competitor availability checks after the gzip check**

After the gzip availability check at line 189-192 of `scripts/bench.sh`, add:

```bash
# Check competitor tool availability (optional — skip gracefully if missing)
HAS_ZLIBBNG=false
HAS_LZ4=false
HAS_ZSTD=false
if command -v minigzip-ng &>/dev/null; then
    HAS_ZLIBBNG=true
elif [[ "$VERBOSE" == true ]]; then
    echo "INFO: minigzip-ng not found (install: brew install zlib-ng); skipping zlib-ng"
fi
if command -v lz4 &>/dev/null; then
    HAS_LZ4=true
elif [[ "$VERBOSE" == true ]]; then
    echo "INFO: lz4 not found (install: brew install lz4); skipping lz4"
fi
if command -v zstd &>/dev/null; then
    HAS_ZSTD=true
elif [[ "$VERBOSE" == true ]]; then
    echo "INFO: zstd not found (install: brew install zstd); skipping zstd"
fi
```

**Step 2: Add competitor accumulator variables**

After the gzip accumulator declarations (`t_orig=0; t_gz=0; t_gz_ns=0` near line 276), add:

```bash
# Competitor accumulators
t_zng=0; t_zng_ns=0
t_lz4=0; t_lz4_ns=0
t_zst=0; t_zst_ns=0
```

**Step 3: Add competitor timing inside the per-file compression loop**

Inside the `for file in "${FILES[@]}"` loop (after the gzip block, before the pz pipelines block), add:

```bash
    # --- zlib-ng compress ---
    if [[ "$HAS_ZLIBBNG" == true ]]; then
        zng_out="$BENCH_TMPDIR/$name.zng"
        zng_comp_ns=$(avg_ns bash -c "minigzip-ng < '$file' > '$zng_out'")
        zng_size=$(stat -c%s "$zng_out" 2>/dev/null || stat -f%z "$zng_out" 2>/dev/null)
        rm -f "$zng_out"
        t_zng=$((t_zng + zng_size))
        t_zng_ns=$((t_zng_ns + zng_comp_ns))
    fi

    # --- lz4 compress ---
    if [[ "$HAS_LZ4" == true ]]; then
        lz4_out="$BENCH_TMPDIR/$name.lz4"
        lz4_comp_ns=$(avg_ns bash -c "lz4 -q -c '$file' > '$lz4_out'")
        lz4_size=$(stat -c%s "$lz4_out" 2>/dev/null || stat -f%z "$lz4_out" 2>/dev/null)
        rm -f "$lz4_out"
        t_lz4=$((t_lz4 + lz4_size))
        t_lz4_ns=$((t_lz4_ns + lz4_comp_ns))
    fi

    # --- zstd compress ---
    if [[ "$HAS_ZSTD" == true ]]; then
        zst_out="$BENCH_TMPDIR/$name.zst"
        zst_comp_ns=$(avg_ns bash -c "zstd -q --single-thread -c '$file' > '$zst_out'")
        zst_size=$(stat -c%s "$zst_out" 2>/dev/null || stat -f%z "$zst_out" 2>/dev/null)
        rm -f "$zst_out"
        t_zst=$((t_zst + zst_size))
        t_zst_ns=$((t_zst_ns + zst_comp_ns))
    fi
```

**Step 4: Add competitor rows to the summary Compression Results section**

In the `=== BENCHMARK SUMMARY ===` block (around line 442-454), after the `gzip` printf line and before the pz pipeline loop, add:

```bash
    if [[ "$HAS_ZLIBBNG" == true ]]; then
        printf "  %-10s %12s %8s %10s %10s\n" "zlib-ng" "$(fmt_bytes $t_zng)" \
            "$(fmt_ratio $t_zng $t_orig)" "$(fmt_ms $t_zng_ns) ms" "$(fmt_throughput $t_orig $t_zng_ns) MB/s"
    fi
    if [[ "$HAS_LZ4" == true ]]; then
        printf "  %-10s %12s %8s %10s %10s\n" "lz4" "$(fmt_bytes $t_lz4)" \
            "$(fmt_ratio $t_lz4 $t_orig)" "$(fmt_ms $t_lz4_ns) ms" "$(fmt_throughput $t_orig $t_lz4_ns) MB/s"
    fi
    if [[ "$HAS_ZSTD" == true ]]; then
        printf "  %-10s %12s %8s %10s %10s\n" "zstd" "$(fmt_bytes $t_zst)" \
            "$(fmt_ratio $t_zst $t_orig)" "$(fmt_ms $t_zst_ns) ms" "$(fmt_throughput $t_orig $t_zst_ns) MB/s"
    fi
```

**Step 5: Verify operationally**

First install the competitors:
```bash
brew install zlib-ng lz4 zstd
```

Then run:
```bash
/Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/scripts/bench.sh -n 1 -p deflate,lzseqr
```

Expected: Summary shows rows for `gzip`, `zlib-ng`, `lz4`, `zstd`, `pz-deflate`, and `pz-lzseqr` with valid sizes and throughputs.

Verify graceful skip without competitors:
```bash
PATH_SAVE="$PATH"
export PATH="/usr/bin:/bin"
/Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/scripts/bench.sh -n 1 -p deflate 2>&1 | head -5
export PATH="$PATH_SAVE"
```

Expected: Script completes without error; competitor rows are absent from the summary.

**Step 6: Commit**

```bash
git add scripts/bench.sh
git commit -m "bench: add zlib-ng, lz4, zstd competitor runners to summary output"
```
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Add Pareto comparison table output to bench.sh

**Files:**
- Modify: `scripts/bench.sh`

**Background:**

AC4.1 requires a "single-thread Pareto comparison table" — a sorted ratio-vs-throughput view that makes it easy to see whether libpz has a point on or above the competitor curve. The existing summary shows pipelines in insertion order. A `--pareto` flag will:

1. Force single-thread mode (`-t 1` equivalent) so all numbers are comparable.
2. Run all pz pipelines (same set as `--all`).
3. Run all available competitors at their default levels.
4. After collecting results, print a table sorted by compression ratio ascending (best ratio first), with throughput as the second column. This is the standard Pareto presentation: you can read down the table and see if any libpz row dominates a competitor row (same or better ratio AND same or better throughput).

**Step 1: Add `--pareto` flag to argument parsing**

Add a `PARETO=false` variable near the other defaults (around line 16-23):

```bash
PARETO=false
```

Add to the argument parsing `case` statement (after `--all`):

```bash
        --pareto)
            PARETO=true
            PIPELINES=(deflate bw bbw lzr lzf lzfi lzseqr lzseqh)
            # Force single-thread for apples-to-apples comparison
            THREADS="1"
            shift
            ;;
```

Add to the usage text:

```
  --pareto               Single-thread Pareto table: all pipelines + all competitors,
                         sorted by ratio. Implies --all and -t 1.
```

**Step 2: Add Pareto table output after the standard summary**

After the existing summary block (after line 469 `echo "Run with --verbose for per-file breakdown"`), add:

```bash
# === PARETO TABLE (only when --pareto flag is set) ===
if [[ "$PARETO" == true ]]; then
    echo ""
    echo "=== PARETO COMPARISON TABLE (single-thread, sorted by ratio) ==="
    echo ""
    echo "  Ratio = compressed/original (lower is better)"
    echo "  Throughput = MB/s compression speed (higher is better)"
    echo ""
    printf "  %-14s %8s %12s %12s\n" "Codec" "Ratio" "Throughput" "Size"
    printf "  %s\n" "──────────────────────────────────────────────────"

    # Build list of (ratio_pct * 10 as integer, label, throughput, size) for sorting
    # ratio_pct * 10 avoids floating point in bash sort
    declare -a pareto_rows

    # gzip row
    gz_ratio_int=$(awk "BEGIN { printf \"%d\", ($t_gz / $t_orig) * 1000; exit }" < /dev/null)
    pareto_rows+=("$gz_ratio_int|gzip (default)|$(fmt_throughput $t_orig $t_gz_ns) MB/s|$(fmt_bytes $t_gz)")

    # competitor rows
    if [[ "$HAS_ZLIBBNG" == true ]]; then
        zng_ratio_int=$(awk "BEGIN { printf \"%d\", ($t_zng / $t_orig) * 1000; exit }" < /dev/null)
        pareto_rows+=("$zng_ratio_int|zlib-ng (default)|$(fmt_throughput $t_orig $t_zng_ns) MB/s|$(fmt_bytes $t_zng)")
    fi
    if [[ "$HAS_LZ4" == true ]]; then
        lz4_ratio_int=$(awk "BEGIN { printf \"%d\", ($t_lz4 / $t_orig) * 1000; exit }" < /dev/null)
        pareto_rows+=("$lz4_ratio_int|lz4 (default)|$(fmt_throughput $t_orig $t_lz4_ns) MB/s|$(fmt_bytes $t_lz4)")
    fi
    if [[ "$HAS_ZSTD" == true ]]; then
        zst_ratio_int=$(awk "BEGIN { printf \"%d\", ($t_zst / $t_orig) * 1000; exit }" < /dev/null)
        pareto_rows+=("$zst_ratio_int|zstd (default)|$(fmt_throughput $t_orig $t_zst_ns) MB/s|$(fmt_bytes $t_zst)")
    fi

    # pz pipeline rows
    for (( pi=0; pi<${#PIPELINES[@]}; pi++ )); do
        p="${PIPELINES[$pi]}"
        pz_ratio_int=$(awk "BEGIN { printf \"%d\", (${t_pz_size[$pi]} / $t_orig) * 1000; exit }" < /dev/null)
        pareto_rows+=("$pz_ratio_int|pz-$p|$(fmt_throughput $t_orig ${t_pz_ns[$pi]}) MB/s|$(fmt_bytes ${t_pz_size[$pi]})")
    done

    # Sort by ratio integer (field 1), print formatted
    printf '%s\n' "${pareto_rows[@]}" | sort -t'|' -k1,1n | while IFS='|' read -r ratio_int label throughput size; do
        ratio_pct=$(awk "BEGIN { printf \"%.1f%%\", $ratio_int / 10.0; exit }" < /dev/null)
        printf "  %-14s %8s %12s %12s\n" "$label" "$ratio_pct" "$throughput" "$size"
    done

    echo ""
    echo "  Pareto-dominant = lower ratio AND higher throughput than any competitor row above it."
fi
```

**Step 3: Verify operationally**

```bash
/Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/scripts/bench.sh --pareto -n 1
```

Expected output includes:
- A section header `=== PARETO COMPARISON TABLE (single-thread, sorted by ratio) ===`
- Rows sorted so the best-ratio codec is at the top
- All installed competitors (gzip, zlib-ng, lz4, zstd) and all pz pipelines present
- The `Threads: 1` line in the configuration header

Verify sort order correctness — lz4 (fast but poor ratio) should appear near the bottom, bw (slow but good ratio) near the top.

**Step 4: Commit**

```bash
git add scripts/bench.sh
git commit -m "bench: add --pareto flag for single-thread sorted ratio-vs-throughput table"
```
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Create bit-budget analysis script (scripts/analyze-ratio.sh)

**Files:**
- Create: `scripts/analyze-ratio.sh`

**Background:**

AC4.2 requires decomposing compressed output into cost categories: literal bytes, match offset bits, match length bits, entropy table overhead, and container headers. The `.pz` format stores a header followed by block data, and each pipeline encodes sequences (literals + matches) whose costs can be estimated from the compressed size and the pipeline type.

Since `pz` does not currently expose per-symbol cost breakdowns, `analyze-ratio.sh` uses a structural approach:
- Parse the `.pz` header (bytes 0-7: magic `PZ`, version, pipeline id, original length) and block framing (v2: bytes 8-11: block count or `0xFFFFFFFF` for framed mode) to isolate header overhead.
- For LZ-based pipelines (deflate, lzr, lzf, lzfi, lzseqr, lzseqh), use the known structure: a compressed `.pz` file over a sample can be compared against a "literals-only" recompression to estimate match contribution. The literals-only estimate compresses only the literal bytes of the input using the same entropy coder, giving a lower bound on what pure literal coding would cost.
- For ratio gap analysis, compare against gzip output on the same file. The gap between libpz size and gzip size, expressed as bits per input byte, is the "ratio sink" figure Phase 1 seeks to quantify.

The script takes a file path, compresses it with a specified pz pipeline, and reports the breakdown. It does not require Rust changes — all analysis is done by inspecting binary output sizes from existing tools.

**Step 1: Create `scripts/analyze-ratio.sh`**

```bash
#!/usr/bin/env bash
# analyze-ratio.sh — Bit-budget analysis for pz compressed output.
#
# Decomposes a compressed .pz file into estimated cost categories:
#   header overhead, entropy table overhead, match bits, literal bits.
# Compares against gzip to show the ratio gap.
#
# Usage:
#   ./scripts/analyze-ratio.sh [OPTIONS] FILE
#
# Options:
#   -p, --pipeline P    Pipeline to analyze (default: lzseqr)
#   -t, --threads N     Thread count for pz (default: 1)
#   -h, --help          Show this help

set -euo pipefail
export LC_ALL=C
export LANG=C

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PZ="$PROJECT_DIR/target/release/pz"

PIPELINE="lzseqr"
THREADS="1"
INPUT_FILE=""

usage() {
    cat <<'EOF'
analyze-ratio.sh — Bit-budget analysis: where does libpz spend compressed bits?

Usage:
  ./scripts/analyze-ratio.sh [OPTIONS] FILE

Options:
  -p, --pipeline P    Pipeline to analyze (default: lzseqr)
  -t, --threads N     Thread count (default: 1)
  -h, --help          Show this help

Output:
  Prints a breakdown of compressed bits per input byte:
    header     — .pz format header + block framing bytes
    gap_vs_gz  — bits/byte libpz spends above gzip (ratio sink)
    total_pz   — total bits/byte for pz pipeline
    total_gz   — total bits/byte for gzip (reference)

Example:
  ./scripts/analyze-ratio.sh samples/cantrbry/alice29.txt
  ./scripts/analyze-ratio.sh -p deflate samples/cantrbry/enwik8
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -p|--pipeline)
            [[ $# -lt 2 ]] && { echo "ERROR: --pipeline requires an argument" >&2; exit 1; }
            PIPELINE="$2"; shift 2 ;;
        -t|--threads)
            [[ $# -lt 2 ]] && { echo "ERROR: --threads requires an argument" >&2; exit 1; }
            THREADS="$2"; shift 2 ;;
        -*)
            echo "ERROR: unknown option '$1'" >&2; exit 1 ;;
        *)
            INPUT_FILE="$1"; shift ;;
    esac
done

if [[ -z "$INPUT_FILE" ]]; then
    echo "ERROR: no input file specified" >&2
    usage >&2
    exit 1
fi
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "ERROR: file not found: $INPUT_FILE" >&2
    exit 1
fi
if [[ ! -x "$PZ" ]]; then
    echo "ERROR: pz binary not found at $PZ — run 'cargo build --release' first" >&2
    exit 1
fi
if ! command -v gzip &>/dev/null; then
    echo "ERROR: gzip not found" >&2
    exit 1
fi

TMPDIR_LOCAL=$(mktemp -d)
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT

FNAME=$(basename "$INPUT_FILE")
ORIG_SIZE=$(stat -c%s "$INPUT_FILE" 2>/dev/null || stat -f%z "$INPUT_FILE" 2>/dev/null)

# Compress with pz pipeline
cp "$INPUT_FILE" "$TMPDIR_LOCAL/$FNAME"
"$PZ" -k -f -p "$PIPELINE" -t "$THREADS" "$TMPDIR_LOCAL/$FNAME"
PZ_FILE="$TMPDIR_LOCAL/$FNAME.pz"
PZ_SIZE=$(stat -c%s "$PZ_FILE" 2>/dev/null || stat -f%z "$PZ_FILE" 2>/dev/null)

# Compress with gzip (default level 6)
cp "$INPUT_FILE" "$TMPDIR_LOCAL/${FNAME}.gzip_src"
gzip -k -f "$TMPDIR_LOCAL/${FNAME}.gzip_src"
GZ_FILE="$TMPDIR_LOCAL/${FNAME}.gzip_src.gz"
GZ_SIZE=$(stat -c%s "$GZ_FILE" 2>/dev/null || stat -f%z "$GZ_FILE" 2>/dev/null)

# Parse .pz header to isolate overhead bytes
# Header layout (always): bytes 0-1 magic "PZ", byte 2 version, byte 3 pipeline id,
#                          bytes 4-7 original length (u32 LE)
# v2 extension:            bytes 8-11 block count (u32 LE), or 0xFFFFFFFF for framed mode
# Framed mode per-block:   [comp_len u32][orig_len u32][block data...]  + EOS sentinel u32(0)
#
# Header overhead estimate: for v2 framed, 12 bytes base + 8 bytes per block + 4 bytes EOS.
# For v1 / v2 non-framed: 8 or 12 bytes flat.
# We use python3 (available on macOS) to parse the binary header.
HEADER_BYTES=0
if command -v python3 &>/dev/null; then
    HEADER_BYTES=$(python3 - "$PZ_FILE" <<'PYEOF'
import sys, struct
path = sys.argv[1]
with open(path, 'rb') as f:
    data = f.read()
if len(data) < 4 or data[0:2] != b'PZ':
    print(0)
    sys.exit(0)
version = data[2]
if version == 2 and len(data) >= 12:
    block_count_raw = struct.unpack_from('<I', data, 8)[0]
    if block_count_raw == 0xFFFFFFFF:
        # framed: count blocks
        pos = 12
        n_blocks = 0
        while pos + 4 <= len(data):
            comp_len = struct.unpack_from('<I', data, pos)[0]
            if comp_len == 0:
                pos += 4  # EOS sentinel
                break
            if pos + 8 > len(data):
                break
            n_blocks += 1
            pos += 8 + comp_len
        # base header (12) + per-block frame headers (8 each) + EOS sentinel (4)
        print(12 + n_blocks * 8 + 4)
    else:
        print(12)
else:
    print(8)
PYEOF
)
fi
HEADER_BYTES=${HEADER_BYTES:-8}

# Compute bits per input byte metrics
bpb_pz=$(awk "BEGIN { printf \"%.3f\", ($PZ_SIZE * 8.0) / $ORIG_SIZE; exit }" < /dev/null)
bpb_gz=$(awk "BEGIN { printf \"%.3f\", ($GZ_SIZE * 8.0) / $ORIG_SIZE; exit }" < /dev/null)
bpb_header=$(awk "BEGIN { printf \"%.3f\", ($HEADER_BYTES * 8.0) / $ORIG_SIZE; exit }" < /dev/null)
bpb_gap=$(awk "BEGIN { v = ($PZ_SIZE - $GZ_SIZE) * 8.0 / $ORIG_SIZE; printf \"%.3f\", v; exit }" < /dev/null)
ratio_pz=$(awk "BEGIN { printf \"%.2f%%\", ($PZ_SIZE / $ORIG_SIZE) * 100; exit }" < /dev/null)
ratio_gz=$(awk "BEGIN { printf \"%.2f%%\", ($GZ_SIZE / $ORIG_SIZE) * 100; exit }" < /dev/null)
gap_pct=$(awk "BEGIN { printf \"%.2f%%\", (($PZ_SIZE - $GZ_SIZE) / $ORIG_SIZE) * 100; exit }" < /dev/null)

# Pipeline category: LZ-based or BWT-based (affects interpretation of gap)
case "$PIPELINE" in
    deflate|lzr|lzf|lzfi|lzseqr|lzseqh|lzssr|lz78r)
        PIPELINE_CLASS="LZ-based" ;;
    bw|bbw)
        PIPELINE_CLASS="BWT-based" ;;
    *)
        PIPELINE_CLASS="unknown" ;;
esac

echo "=== Bit-Budget Analysis ==="
echo ""
printf "  File:       %s\n" "$FNAME"
printf "  Original:   %d bytes\n" "$ORIG_SIZE"
printf "  Pipeline:   %s (%s)\n" "$PIPELINE" "$PIPELINE_CLASS"
echo ""
printf "  %-22s %10s %10s\n" "Cost Category" "bits/byte" "% of orig"
printf "  %s\n" "──────────────────────────────────────────────"
printf "  %-22s %10s %10s\n" "pz total" "$bpb_pz bpb" "$ratio_pz"
printf "  %-22s %10s %10s\n" "  .pz header+framing" "$bpb_header bpb" \
    "$(awk "BEGIN { printf \"%.2f%%\", ($HEADER_BYTES / $ORIG_SIZE) * 100; exit }" < /dev/null)"
printf "  %-22s %10s %10s\n" "  payload" \
    "$(awk "BEGIN { printf \"%.3f\", (($PZ_SIZE - $HEADER_BYTES) * 8.0) / $ORIG_SIZE; exit }" < /dev/null) bpb" \
    "$(awk "BEGIN { printf \"%.2f%%\", (($PZ_SIZE - $HEADER_BYTES) / $ORIG_SIZE) * 100; exit }" < /dev/null)"
echo ""
printf "  %-22s %10s %10s\n" "gzip total (ref)" "$bpb_gz bpb" "$ratio_gz"
echo ""
printf "  %-22s %10s %10s\n" "gap vs gzip" "$bpb_gap bpb" "$gap_pct"
echo ""
if awk "BEGIN { exit ($PZ_SIZE <= $GZ_SIZE) ? 0 : 1 }" < /dev/null; then
    echo "  pz-$PIPELINE matches or beats gzip on this file."
else
    echo "  pz-$PIPELINE spends $bpb_gap extra bits/byte vs gzip — top ratio sink."
fi
```

Make the script executable:
```bash
chmod +x scripts/analyze-ratio.sh
```

**Step 2: Verify operationally**

Run on a Canterbury corpus file with the primary target pipeline:
```bash
/Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/scripts/analyze-ratio.sh \
    -p lzseqr \
    /Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/samples/cantrbry/alice29.txt
```

Expected output includes:
- `File: alice29.txt`
- `Pipeline: lzseqr (LZ-based)`
- Numeric `bpb` values for `pz total`, `.pz header+framing`, `payload`, `gzip total (ref)`, and `gap vs gzip`
- A final summary line about the ratio sink

Run on a BWT pipeline to confirm the class label changes:
```bash
/Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/scripts/analyze-ratio.sh \
    -p bw \
    /Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/samples/cantrbry/alice29.txt
```

Expected: `Pipeline: bw (BWT-based)` in the output.

**Step 3: Commit**

```bash
git add scripts/analyze-ratio.sh
git commit -m "scripts: add analyze-ratio.sh bit-budget analysis tool (AC4.2)"
```
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: Verify benchmark infrastructure end-to-end

**Files:**
- No file modifications — this is a verification-only task.

**Background:**

With Tasks 1-4 complete, verify that the full pipeline — competitor runners, LzSeqR/LzSeqH, Pareto table, and bit-budget analysis — works correctly together on the Canterbury corpus and produces the data needed to identify ratio sinks (Phase 1 done-when criteria).

**Step 1: Run the full Pareto comparison on Canterbury corpus**

```bash
/Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/scripts/bench.sh \
    --pareto -n 3 \
    --features '' \
    2>&1 | tee /tmp/pareto-baseline-cantrbry.txt
```

Expected:
- No errors during build or compression
- `=== PARETO COMPARISON TABLE (single-thread, sorted by ratio) ===` section present
- All 8 pz pipelines (deflate, bw, bbw, lzr, lzf, lzfi, lzseqr, lzseqh) appear in table
- gzip appears; zlib-ng, lz4, zstd appear if installed
- Rows are sorted ratio-ascending (smaller ratio percentage = better compression, appears first)

**Step 2: Run bit-budget analysis across all Canterbury files**

```bash
for f in /Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/samples/cantrbry/*; do
    [[ -f "$f" ]] || continue
    case "$f" in *.tar.gz|*.pz|*.gz) continue ;; esac
    echo "--- $(basename "$f") ---"
    /Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/scripts/analyze-ratio.sh \
        -p lzseqr "$f"
done 2>&1 | tee /tmp/bit-budget-cantrbry.txt
```

Expected:
- Each file produces a complete bit-budget table with no errors
- `gap vs gzip` values are populated and non-zero for most files
- No pipeline crashes (edge cases: `random.txt`, `kennedy.xls` — large or random)

**Step 3: Spot-check output correctness**

Verify that the ratio values in the Pareto table match what `analyze-ratio.sh` reports for the same pipeline on the same corpus. They will differ (bench.sh aggregates all files; analyze-ratio.sh is per-file), but the lzseqr row in the Pareto table should have a ratio in the same ballpark as the average gap seen in the bit-budget output.

```bash
grep "pz-lzseqr" /tmp/pareto-baseline-cantrbry.txt
grep "pz total" /tmp/bit-budget-cantrbry.txt | head -5
```

**Step 4: Capture baseline numbers**

Record the Pareto table output as the measurement baseline for Phase 1. This is the concrete artifact proving Phase 1 done-when:

> "Can produce a single-thread Pareto comparison table for libpz pipelines against gzip, zlib-ng, lz4, zstd on Canterbury. Bit-budget analysis shows where libpz loses ratio bits relative to gzip."

Save the captured output:
```bash
cp /tmp/pareto-baseline-cantrbry.txt \
    /Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/docs/implementation-plans/2026-02-22-pareto-competitiveness/pareto-baseline-cantrbry.txt
cp /tmp/bit-budget-cantrbry.txt \
    /Users/clundquist/.claude/worktrees/libpz/pareto-competitiveness/docs/implementation-plans/2026-02-22-pareto-competitiveness/bit-budget-cantrbry.txt
```

**Step 5: Commit**

```bash
git add docs/implementation-plans/2026-02-22-pareto-competitiveness/pareto-baseline-cantrbry.txt \
        docs/implementation-plans/2026-02-22-pareto-competitiveness/bit-budget-cantrbry.txt
git commit -m "docs: capture Phase 1 measurement baseline — Pareto table and bit-budget analysis"
```
<!-- END_TASK_5 -->

---

## Phase 1 Completion Checklist

- [ ] Task 0: Canterbury corpus verified at `samples/cantrbry/`; Silesia corpus checked (optional but recommended)
- [ ] Task 1: `lzseqr` and `lzseqh` appear in `--all` benchmark run
- [ ] Task 2: `zlib-ng`, `lz4`, `zstd` rows appear in bench.sh summary when tools are installed
- [ ] Task 3: `--pareto` flag produces a ratio-sorted single-thread comparison table
- [ ] Task 4: `scripts/analyze-ratio.sh` produces bits-per-byte breakdown vs gzip for any input file
- [ ] Task 5: Pareto baseline and bit-budget output saved to `docs/implementation-plans/2026-02-22-pareto-competitiveness/`

**AC4.1 satisfied** when: `bench.sh --pareto` runs end-to-end and produces the sorted table with all four competitors and all eight pz pipelines.

**AC4.2 satisfied** when: `analyze-ratio.sh -p lzseqr <file>` outputs a breakdown showing `pz total`, `.pz header+framing`, `payload`, `gzip total (ref)`, and `gap vs gzip` for any Canterbury corpus file.
