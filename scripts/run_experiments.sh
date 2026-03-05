#!/usr/bin/env bash
# run_experiments.sh — Run all wave-2 experimental pipelines + baselines on Canterbury corpus.
#
# Produces a CSV and summary table comparing bitplane, fwst, parlz, repair
# against all existing pipelines and external tools (gzip, zstd).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PZ="$PROJECT_DIR/target/release/pz"
SAMPLES_DIR="$PROJECT_DIR/samples"

usage() {
    cat <<'EOF'
run_experiments.sh — Wave-2 GPU experiment comparison benchmark.

Usage:
  ./scripts/run_experiments.sh [OPTIONS]

Options:
  -n, --iters N     Iterations per measurement (default: 1)
  --csv FILE        Write CSV output to FILE (default: stdout)
  --fwst-sweep      Sweep FWST window sizes (2,4,6,8,10,12,16,24,32)
  --parlz-gap       Show parallel vs greedy ratio gap diagnostic
  --repair-stats    Show Re-Pair per-round convergence stats
  -h, --help        Show this help
EOF
}

ITERATIONS=1
CSV_FILE=""
FWST_SWEEP=false
PARLZ_GAP=false
REPAIR_STATS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--iters)    ITERATIONS="$2"; shift 2 ;;
        --csv)         CSV_FILE="$2"; shift 2 ;;
        --fwst-sweep)  FWST_SWEEP=true; shift ;;
        --parlz-gap)   PARLZ_GAP=true; shift ;;
        --repair-stats) REPAIR_STATS=true; shift ;;
        -h|--help)     usage; exit 0 ;;
        *)             echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# Build release binary
echo "Building pz (release)..." >&2
cargo build --release --manifest-path "$PROJECT_DIR/Cargo.toml" --quiet 2>/dev/null || \
    cargo build --release --manifest-path "$PROJECT_DIR/Cargo.toml"

# Extract Canterbury corpus if needed
CORPUS_DIR="$SAMPLES_DIR/cantrbry"
if [[ ! -d "$CORPUS_DIR" ]]; then
    echo "Extracting Canterbury corpus..." >&2
    mkdir -p "$CORPUS_DIR"
    tar -xzf "$SAMPLES_DIR/cantrbry.tar.gz" -C "$CORPUS_DIR" 2>/dev/null || true
fi

# Find test files
mapfile -t FILES < <(find "$CORPUS_DIR" -type f -not -name '.*' | sort)

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "Error: No Canterbury corpus files found in $CORPUS_DIR" >&2
    exit 1
fi

# Experimental + baseline pipelines
EXPERIMENT_PIPELINES=(bitplane fwst parlz repair)
BASELINE_PIPELINES=(deflate bw lzr lzf lzseqr lzseqh)

# CSV header
CSV_HEADER="file,size,pipeline,params,compressed_size,ratio,throughput_mbs,time_total_ms,notes"

output_csv() {
    if [[ -n "$CSV_FILE" ]]; then
        echo "$1" >> "$CSV_FILE"
    else
        echo "$1"
    fi
}

# Initialize CSV
if [[ -n "$CSV_FILE" ]]; then
    echo "$CSV_HEADER" > "$CSV_FILE"
else
    echo "$CSV_HEADER"
fi

# Benchmark a single pipeline on a single file
bench_pz() {
    local file="$1" pipeline="$2" params="${3:-}" tmpfile
    local basename size

    basename=$(basename "$file")
    size=$(wc -c < "$file" | tr -d ' ')

    tmpfile=$(mktemp)
    trap "rm -f '$tmpfile' '${tmpfile}.pz'" RETURN 2>/dev/null || true

    local total_ms=0
    local compressed_size=0

    for _ in $(seq 1 "$ITERATIONS"); do
        rm -f "$tmpfile" "${tmpfile}.pz"
        cp "$file" "$tmpfile"

        local start end elapsed_ms
        start=$(date +%s%N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1e9))')
        "$PZ" -p "$pipeline" "$tmpfile" 2>/dev/null || { rm -f "$tmpfile" "${tmpfile}.pz"; return; }
        end=$(date +%s%N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1e9))')

        elapsed_ms=$(( (end - start) / 1000000 ))
        total_ms=$((total_ms + elapsed_ms))

        if [[ -f "${tmpfile}.pz" ]]; then
            compressed_size=$(wc -c < "${tmpfile}.pz" | tr -d ' ')
        fi
    done

    local avg_ms=$((total_ms / ITERATIONS))
    local ratio throughput_mbs
    if [[ $size -gt 0 ]]; then
        ratio=$(python3 -c "print(f'{$compressed_size / $size * 100:.1f}%')")
    else
        ratio="N/A"
    fi
    if [[ $avg_ms -gt 0 ]]; then
        throughput_mbs=$(python3 -c "print(f'{$size / $avg_ms / 1000:.1f}')")
    else
        throughput_mbs="N/A"
    fi

    output_csv "$basename,$size,$pipeline,$params,$compressed_size,$ratio,$throughput_mbs,$avg_ms,"

    rm -f "$tmpfile" "${tmpfile}.pz"
}

# Benchmark gzip
bench_gzip() {
    local file="$1" level="${2:-6}"
    local basename size tmpfile

    basename=$(basename "$file")
    size=$(wc -c < "$file" | tr -d ' ')
    tmpfile=$(mktemp)

    local start end elapsed_ms compressed_size
    start=$(date +%s%N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1e9))')
    gzip -"$level" -c "$file" > "$tmpfile" 2>/dev/null
    end=$(date +%s%N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1e9))')

    elapsed_ms=$(( (end - start) / 1000000 ))
    compressed_size=$(wc -c < "$tmpfile" | tr -d ' ')

    local ratio throughput_mbs
    ratio=$(python3 -c "print(f'{$compressed_size / $size * 100:.1f}%')" 2>/dev/null || echo "N/A")
    if [[ $elapsed_ms -gt 0 ]]; then
        throughput_mbs=$(python3 -c "print(f'{$size / $elapsed_ms / 1000:.1f}')" 2>/dev/null || echo "N/A")
    else
        throughput_mbs="N/A"
    fi

    output_csv "$basename,$size,gzip-$level,,$compressed_size,$ratio,$throughput_mbs,$elapsed_ms,"

    rm -f "$tmpfile"
}

# Benchmark zstd if available
bench_zstd() {
    local file="$1" level="${2:-3}"
    command -v zstd >/dev/null 2>&1 || return 0

    local basename size tmpfile
    basename=$(basename "$file")
    size=$(wc -c < "$file" | tr -d ' ')
    tmpfile=$(mktemp)

    local start end elapsed_ms compressed_size
    start=$(date +%s%N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1e9))')
    zstd -"$level" -c "$file" > "$tmpfile" 2>/dev/null
    end=$(date +%s%N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1e9))')

    elapsed_ms=$(( (end - start) / 1000000 ))
    compressed_size=$(wc -c < "$tmpfile" | tr -d ' ')

    local ratio throughput_mbs
    ratio=$(python3 -c "print(f'{$compressed_size / $size * 100:.1f}%')" 2>/dev/null || echo "N/A")
    if [[ $elapsed_ms -gt 0 ]]; then
        throughput_mbs=$(python3 -c "print(f'{$size / $elapsed_ms / 1000:.1f}')" 2>/dev/null || echo "N/A")
    else
        throughput_mbs="N/A"
    fi

    output_csv "$basename,$size,zstd-$level,,$compressed_size,$ratio,$throughput_mbs,$elapsed_ms,"

    rm -f "$tmpfile"
}

echo "" >&2
echo "=== Wave-2 Experiment Benchmark ===" >&2
echo "Files: ${#FILES[@]} Canterbury corpus files" >&2
echo "Iterations: $ITERATIONS" >&2
echo "" >&2

# Run benchmarks
for file in "${FILES[@]}"; do
    basename=$(basename "$file")
    size=$(wc -c < "$file" | tr -d ' ')
    echo "--- $basename ($size bytes) ---" >&2

    # External tools
    bench_gzip "$file" 6
    bench_zstd "$file" 3

    # Baseline pipelines
    for p in "${BASELINE_PIPELINES[@]}"; do
        bench_pz "$file" "$p"
    done

    # Experimental pipelines
    for p in "${EXPERIMENT_PIPELINES[@]}"; do
        bench_pz "$file" "$p"
    done
done

echo "" >&2
echo "=== Benchmark complete ===" >&2

# FWST window sweep (optional)
if $FWST_SWEEP; then
    echo "" >&2
    echo "=== FWST Window Sweep ===" >&2
    echo "" >&2
    echo "file,size,window,compressed_size,ratio"

    # This requires a custom binary or modifying pz to accept --fwst-window
    # For now, document that the sweep needs per-window compression via library API
    echo "(FWST sweep requires library-level API access — use cargo test or a custom binary)" >&2
fi

echo "" >&2
echo "Done." >&2
