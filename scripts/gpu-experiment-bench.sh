#!/usr/bin/env bash
# gpu-experiment-bench.sh — Benchmark CSBWT and SortLZ experiments against baselines.
#
# Runs all pipelines on Canterbury corpus files and outputs a CSV with:
#   file, pipeline, compressed_size, ratio, throughput_mbs, time_ms
#
# Usage:
#   ./scripts/gpu-experiment-bench.sh           # Full benchmark
#   ./scripts/gpu-experiment-bench.sh --quick   # Fewer iterations
#   ./scripts/gpu-experiment-bench.sh --help    # Show help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PZ="${PROJECT_DIR}/target/release/pz"
SAMPLES_DIR="${PROJECT_DIR}/samples/cantrbry"

ITERATIONS=3
QUICK=false

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --quick      Fewer iterations (1 instead of 3)"
    echo "  --help       Show this help"
    echo ""
    echo "Benchmarks CSBWT and SortLZ against all existing pipelines on Canterbury corpus."
    echo "Output: CSV to stdout, diagnostics to stderr."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick) QUICK=true; ITERATIONS=1; shift ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

# Build release binary if needed.
if [[ ! -f "$PZ" ]] || [[ "$PZ" -ot "${PROJECT_DIR}/src/lib.rs" ]]; then
    echo "Building release binary..." >&2
    cd "$PROJECT_DIR"
    cargo build --release --quiet 2>&1 >&2
fi

# Canterbury corpus files.
FILES=(
    alice29.txt
    asyoulik.txt
    cp.html
    fields.c
    grammar.lsp
    kennedy.xls
    lcet10.txt
    plrabn12.txt
    ptt5
    sum
    xargs.1
)

# Pipelines to benchmark.
PIPELINES=(
    deflate
    bw
    bbw
    lzr
    lzf
    lzfi
    lzssr
    lz78r
    lzseqr
    lzseqh
    sortlz
)

# CSV header.
echo "file,pipeline,original_size,compressed_size,ratio,throughput_mbs,time_ms"

for file in "${FILES[@]}"; do
    filepath="${SAMPLES_DIR}/${file}"
    if [[ ! -f "$filepath" ]]; then
        echo "SKIP: ${file} not found" >&2
        continue
    fi

    original_size=$(stat -c%s "$filepath" 2>/dev/null || stat -f%z "$filepath" 2>/dev/null)

    for pipeline in "${PIPELINES[@]}"; do
        best_time_ns=999999999999

        for ((iter=0; iter<ITERATIONS; iter++)); do
            # Time compression: compress to stdout, discard output, measure wall clock.
            start_ns=$(date +%s%N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1e9))')
            compressed=$("$PZ" -p "$pipeline" -kc "$filepath" 2>/dev/null)
            end_ns=$(date +%s%N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1e9))')

            elapsed_ns=$((end_ns - start_ns))
            if (( elapsed_ns < best_time_ns )); then
                best_time_ns=$elapsed_ns
            fi

            # Get compressed size on first iteration.
            if [[ $iter -eq 0 ]]; then
                comp_size=${#compressed}
            fi
        done

        # Compute ratio and throughput.
        time_ms=$(echo "scale=2; $best_time_ns / 1000000" | bc 2>/dev/null || echo "0")
        ratio=$(echo "scale=6; $comp_size / $original_size" | bc 2>/dev/null || echo "0")
        throughput_mbs=$(echo "scale=2; $original_size / ($best_time_ns / 1000000000) / 1048576" | bc 2>/dev/null || echo "0")

        echo "${file},${pipeline},${original_size},${comp_size},${ratio},${throughput_mbs},${time_ms}"
    done
done

echo "" >&2
echo "Done. Results above in CSV format." >&2
