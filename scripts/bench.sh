#!/usr/bin/env bash
# bench.sh — Compare pz pipelines vs gzip: size, ratio, time, throughput.

set -euo pipefail

# Normalize locale for portable tool behavior (notably perl Time::HiRes).
# Some hosts do not have C.UTF-8 installed and emit noisy warnings.
export LC_ALL=C
export LANG=C

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PZ="$PROJECT_DIR/target/release/pz"

# Defaults
ITERATIONS=3
PIPELINES=()
FILES=()
FEATURES=""
GPU_FLAG=""
THREADS=""
VERBOSE=false

usage() {
    cat <<'EOF'
bench.sh — Compare pz pipelines vs gzip: size, ratio, time, throughput.

Usage:
  ./scripts/bench.sh [OPTIONS] [FILE ...]

Options:
  -n, --iters N          Number of iterations per operation (default: 3)
  -p, --pipelines LIST   Comma-separated list of pipelines to benchmark
                         (default: deflate,lzr,lzf)
  -t, --threads N        Pass thread count to pz (-t N; 0=auto, 1=single-threaded)
  --all                  Benchmark all available pipelines
  --webgpu               Build with WebGPU feature and pass --gpu to pz
  --features FEAT        Cargo features to enable (e.g. webgpu)
  -v, --verbose          Show detailed output (default: quiet, summary only)
  -h, --help             Show this help

If no FILEs are given, benchmarks all files in samples/cantrbry and samples/large.

Examples:
  ./scripts/bench.sh                              # all corpus, all pipelines
  ./scripts/bench.sh myfile.bin                   # specific file
  ./scripts/bench.sh -p deflate,lzf               # subset of pipelines
  ./scripts/bench.sh -t 1 -p lzr                  # force single-threaded pz
  ./scripts/bench.sh -n 10                        # more iterations
  ./scripts/bench.sh --webgpu -p bw,bbw           # GPU-accelerated via WebGPU
  ./scripts/bench.sh --all                         # benchmark every pipeline
  ./scripts/bench.sh -n 1 -p deflate,lzf file.txt # combine options
  ./scripts/bench.sh -v                           # verbose output with full tables
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -n|--iters)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --iters requires an argument" >&2
                exit 1
            fi
            ITERATIONS="$2"
            shift 2
            ;;
        -p|--pipelines)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --pipelines requires an argument" >&2
                exit 1
            fi
            IFS=',' read -ra PIPELINES <<< "$2"
            shift 2
            ;;
        -t|--threads)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --threads requires an argument" >&2
                exit 1
            fi
            THREADS="$2"
            if ! [[ "$THREADS" =~ ^[0-9]+$ ]]; then
                echo "ERROR: --threads must be a non-negative integer" >&2
                exit 1
            fi
            shift 2
            ;;
        --all)
            PIPELINES=(deflate bw bbw lzr lzf lzfi)
            shift
            ;;
        --webgpu)
            # Shorthand: enable webgpu feature and pass --gpu to pz
            if [[ -z "$FEATURES" ]]; then
                FEATURES="webgpu"
            else
                FEATURES="$FEATURES,webgpu"
            fi
            GPU_FLAG="--gpu"
            shift
            ;;
        --features)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --features requires an argument" >&2
                exit 1
            fi
            if [[ -z "$FEATURES" ]]; then
                FEATURES="$2"
            else
                FEATURES="$FEATURES,$2"
            fi
            shift 2
            ;;
        -*)
            echo "ERROR: unknown option '$1'" >&2
            echo "Run './scripts/bench.sh --help' for usage." >&2
            exit 1
            ;;
        *)
            FILES+=("$1")
            shift
            ;;
    esac
done

# Default pipelines if none specified
if [[ ${#PIPELINES[@]} -eq 0 ]]; then
    PIPELINES=(deflate lzr lzf)
fi

# Collect input files from corpus if none given on command line
if [[ ${#FILES[@]} -eq 0 ]]; then
    # Auto-extract sample archives if needed
    if ! "$SCRIPT_DIR/setup.sh" 2>&1; then
        echo "ERROR: Failed to extract sample archives" >&2
        exit 1
    fi

    for f in "$PROJECT_DIR"/samples/cantrbry/* "$PROJECT_DIR"/samples/large/*; do
        # Skip archives and compressed leftovers — only benchmark raw sample files
        [[ -f "$f" ]] || continue
        case "$f" in *.tar.gz|*.pz|*.gz) continue ;; esac
        FILES+=("$f")
    done
    if [[ ${#FILES[@]} -eq 0 ]]; then
        echo "ERROR: No sample files found even after extraction" >&2
        exit 1
    fi
fi

# Build release binary
CARGO_FEATURES_ARG=()
if [[ -n "$FEATURES" ]]; then
    CARGO_FEATURES_ARG=(--features "$FEATURES")
fi

# Build quietly unless verbose or build fails
BUILD_OUTPUT=$(mktemp)
if cargo build --release --manifest-path "$PROJECT_DIR/Cargo.toml" ${CARGO_FEATURES_ARG[@]+"${CARGO_FEATURES_ARG[@]}"} >"$BUILD_OUTPUT" 2>&1; then
    if [[ "$VERBOSE" == true ]]; then
        if [[ -n "$FEATURES" ]]; then
            echo "Built pz (release, features: $FEATURES)"
        else
            echo "Built pz (release)"
        fi
    fi
else
    echo "ERROR: Build failed" >&2
    cat "$BUILD_OUTPUT" >&2
    rm -f "$BUILD_OUTPUT"
    exit 1
fi
rm -f "$BUILD_OUTPUT"

if [[ ! -x "$PZ" ]]; then
    echo "ERROR: $PZ not found after build" >&2
    exit 1
fi

if ! command -v gzip &>/dev/null; then
    echo "ERROR: gzip not found" >&2
    exit 1
fi

BENCH_TMPDIR=$(mktemp -d)
trap 'rm -rf "$BENCH_TMPDIR"' EXIT

# time_ns CMD ARGS... → prints wall-clock nanoseconds
# macOS `date` lacks %N, so use perl for portable sub-ms precision
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

fmt_bytes() {
    local bytes=$1
    if (( bytes >= 1073741824 )); then
        awk "BEGIN { printf \"%.2f GB\", $bytes / 1073741824.0; exit }" < /dev/null
    elif (( bytes >= 1048576 )); then
        awk "BEGIN { printf \"%.2f MB\", $bytes / 1048576.0; exit }" < /dev/null
    elif (( bytes >= 1024 )); then
        awk "BEGIN { printf \"%.2f KB\", $bytes / 1024.0; exit }" < /dev/null
    else
        echo "${bytes} B"
    fi
}

fmt_ms() {
    awk "BEGIN { printf \"%.1f\", $1 / 1000000.0; exit }" < /dev/null
}

fmt_throughput() {
    local bytes=$1 ns=$2
    if [[ "$ns" -le 0 ]]; then
        echo "-.--"
    else
        awk "BEGIN { printf \"%.1f\", ($bytes / 1048576.0) / ($ns / 1000000000.0); exit }" < /dev/null
    fi
}

fmt_ratio() {
    awk "BEGIN { printf \"%.1f%%\", ($1/$2)*100; exit }" < /dev/null
}

if [[ "$VERBOSE" == true ]]; then
    echo "Averaging over $ITERATIONS iterations per operation."
    echo "Pipelines: ${PIPELINES[*]}"
    if [[ -n "$THREADS" ]]; then
        echo "Threads: $THREADS"
    fi
    if [[ -n "$GPU_FLAG" ]]; then
        echo "GPU: $GPU_FLAG"
    fi
    echo ""
fi

# Build dynamic column header
hdr_file=$(printf "%-20s %8s" "FILE" "ORIG")
hdr_sep=""
hdr_file+=$(printf " | %8s %6s %7s %8s" "GZIP" "RATIO" "ms" "MB/s")
for p in "${PIPELINES[@]}"; do
    tag=$(echo "$p" | tr '[:lower:]' '[:upper:]')
    hdr_file+=$(printf " | %8s %6s %7s %8s" "PZ-$tag" "RATIO" "ms" "MB/s")
done

# === COMPRESSION ===
if [[ "$VERBOSE" == true ]]; then
    echo "=== COMPRESSION ==="
    echo "$hdr_file"
    col_width=${#hdr_file}
    printf '%*s\n' "$col_width" '' | tr ' ' '-'
fi

# Accumulators: gzip
t_orig=0; t_gz=0; t_gz_ns=0
# Accumulators: per-pipeline (parallel arrays)
declare -a t_pz_size t_pz_ns
for (( pi=0; pi<${#PIPELINES[@]}; pi++ )); do
    t_pz_size[$pi]=0
    t_pz_ns[$pi]=0
done

for file in "${FILES[@]}"; do
    name=$(basename "$file")
    orig_size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)

    # --- gzip compress ---
    cp "$file" "$BENCH_TMPDIR/$name"
    gz_comp_ns=$(avg_ns gzip -k -f "$BENCH_TMPDIR/$name")
    gz_size=$(stat -c%s "$BENCH_TMPDIR/$name.gz" 2>/dev/null || stat -f%z "$BENCH_TMPDIR/$name.gz" 2>/dev/null)
    rm -f "$BENCH_TMPDIR/$name" "$BENCH_TMPDIR/$name.gz"

    row=$(printf "%-20s %8d" "$name" "$orig_size")
    row+=$(printf " | %8d %6s %7s %8s" \
        "$gz_size" "$(fmt_ratio $gz_size $orig_size)" \
        "$(fmt_ms $gz_comp_ns)" "$(fmt_throughput $orig_size $gz_comp_ns)")

    t_orig=$((t_orig + orig_size))
    t_gz=$((t_gz + gz_size))
    t_gz_ns=$((t_gz_ns + gz_comp_ns))

    # --- pz pipelines ---
    for (( pi=0; pi<${#PIPELINES[@]}; pi++ )); do
        p="${PIPELINES[$pi]}"
        cp "$file" "$BENCH_TMPDIR/$name"
        pz_comp_ns=$(avg_ns "$PZ" -k -f -p "$p" ${THREADS:+-t "$THREADS"} $GPU_FLAG "$BENCH_TMPDIR/$name")
        pz_size=$(stat -c%s "$BENCH_TMPDIR/$name.pz" 2>/dev/null || stat -f%z "$BENCH_TMPDIR/$name.pz" 2>/dev/null)
        rm -f "$BENCH_TMPDIR/$name" "$BENCH_TMPDIR/$name.pz"

        row+=$(printf " | %8d %6s %7s %8s" \
            "$pz_size" "$(fmt_ratio $pz_size $orig_size)" \
            "$(fmt_ms $pz_comp_ns)" "$(fmt_throughput $orig_size $pz_comp_ns)")

        t_pz_size[$pi]=$(( ${t_pz_size[$pi]} + pz_size ))
        t_pz_ns[$pi]=$(( ${t_pz_ns[$pi]} + pz_comp_ns ))
    done

    if [[ "$VERBOSE" == true ]]; then
        echo "$row"
    fi
done

# Totals row
if [[ "$VERBOSE" == true ]]; then
    printf '%*s\n' "$col_width" '' | tr ' ' '-'
fi
total_row=$(printf "%-20s %8d" "TOTAL" "$t_orig")
total_row+=$(printf " | %8d %6s %7s %8s" \
    "$t_gz" "$(fmt_ratio $t_gz $t_orig)" \
    "$(fmt_ms $t_gz_ns)" "$(fmt_throughput $t_orig $t_gz_ns)")
for (( pi=0; pi<${#PIPELINES[@]}; pi++ )); do
    total_row+=$(printf " | %8d %6s %7s %8s" \
        "${t_pz_size[$pi]}" "$(fmt_ratio ${t_pz_size[$pi]} $t_orig)" \
        "$(fmt_ms ${t_pz_ns[$pi]})" "$(fmt_throughput $t_orig ${t_pz_ns[$pi]})")
done
if [[ "$VERBOSE" == true ]]; then
    echo "$total_row"
    echo ""
fi

# === DECOMPRESSION ===
if [[ "$VERBOSE" == true ]]; then
    echo "=== DECOMPRESSION ==="
    dhdr=$(printf "%-20s %8s" "FILE" "ORIG")
    dhdr+=$(printf " | %7s %8s" "GZIP-ms" "MB/s")
    for p in "${PIPELINES[@]}"; do
        tag=$(echo "$p" | tr '[:lower:]' '[:upper:]')
        dhdr+=$(printf " | %7s %8s" "PZ-${tag:0:3}" "MB/s")
    done
    echo "$dhdr"
    dcol_width=${#dhdr}
    printf '%*s\n' "$dcol_width" '' | tr ' ' '-'
fi

dt_orig=0; dt_gz_ns=0
declare -a dt_pz_ns
for (( pi=0; pi<${#PIPELINES[@]}; pi++ )); do
    dt_pz_ns[$pi]=0
done

for file in "${FILES[@]}"; do
    name=$(basename "$file")
    orig_size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)

    # Prepare compressed versions
    cp "$file" "$BENCH_TMPDIR/$name.src"

    # gzip
    cp -f "$BENCH_TMPDIR/$name.src" "$BENCH_TMPDIR/$name"
    gzip -f "$BENCH_TMPDIR/$name"

    # pz pipelines
    for (( pi=0; pi<${#PIPELINES[@]}; pi++ )); do
        p="${PIPELINES[$pi]}"
        cp -f "$BENCH_TMPDIR/$name.src" "$BENCH_TMPDIR/$name"
        "$PZ" -k -f -p "$p" ${THREADS:+-t "$THREADS"} $GPU_FLAG "$BENCH_TMPDIR/$name"
        mv "$BENCH_TMPDIR/$name.pz" "$BENCH_TMPDIR/$name.$p.pz"
    done

    # Time decompressions
    gz_dec_ns=$(avg_ns gzip -d -k -f "$BENCH_TMPDIR/$name.gz")

    drow=$(printf "%-20s %8d" "$name" "$orig_size")
    drow+=$(printf " | %7s %8s" \
        "$(fmt_ms $gz_dec_ns)" "$(fmt_throughput $orig_size $gz_dec_ns)")

    dt_orig=$((dt_orig + orig_size))
    dt_gz_ns=$((dt_gz_ns + gz_dec_ns))

    for (( pi=0; pi<${#PIPELINES[@]}; pi++ )); do
        p="${PIPELINES[$pi]}"
        pz_dec_ns=$(avg_ns "$PZ" -d -k -f ${THREADS:+-t "$THREADS"} $GPU_FLAG "$BENCH_TMPDIR/$name.$p.pz")
        drow+=$(printf " | %7s %8s" \
            "$(fmt_ms $pz_dec_ns)" "$(fmt_throughput $orig_size $pz_dec_ns)")
        dt_pz_ns[$pi]=$(( ${dt_pz_ns[$pi]} + pz_dec_ns ))
    done

    if [[ "$VERBOSE" == true ]]; then
        echo "$drow"
    fi

    # Cleanup
    rm -f "$BENCH_TMPDIR/$name.src" "$BENCH_TMPDIR/$name.gz" "$BENCH_TMPDIR/$name"
    for p in "${PIPELINES[@]}"; do
        rm -f "$BENCH_TMPDIR/$name.$p.pz" "$BENCH_TMPDIR/$name.$p"
    done
done

if [[ "$VERBOSE" == true ]]; then
    printf '%*s\n' "$dcol_width" '' | tr ' ' '-'
fi
dtotal=$(printf "%-20s %8d" "TOTAL" "$dt_orig")
dtotal+=$(printf " | %7s %8s" \
    "$(fmt_ms $dt_gz_ns)" "$(fmt_throughput $dt_orig $dt_gz_ns)")
for (( pi=0; pi<${#PIPELINES[@]}; pi++ )); do
    dtotal+=$(printf " | %7s %8s" \
        "$(fmt_ms ${dt_pz_ns[$pi]})" "$(fmt_throughput $dt_orig ${dt_pz_ns[$pi]})")
done
if [[ "$VERBOSE" == true ]]; then
    echo "$dtotal"
    echo ""
fi

# === SUMMARY (always shown) ===
if [[ "$VERBOSE" == false ]]; then
    echo "=== BENCHMARK SUMMARY ==="
    echo ""
    echo "Configuration:"
    echo "  Files:      ${#FILES[@]} files ($(fmt_bytes $t_orig) total)"
    echo "  Iterations: $ITERATIONS per operation"
    echo "  Pipelines:  ${PIPELINES[*]}"
    if [[ -n "$THREADS" ]]; then
        echo "  Threads:    $THREADS"
    fi
    if [[ -n "$GPU_FLAG" ]]; then
        echo "  GPU:        enabled"
    fi
    echo ""

    echo "Compression Results:"
    printf "  %-10s %12s %8s %10s %10s\n" "Pipeline" "Size" "Ratio" "Time" "Throughput"
    printf "  %s\n" "────────────────────────────────────────────────────────────"
    printf "  %-10s %12s %8s %10s %10s\n" "gzip" "$(fmt_bytes $t_gz)" \
        "$(fmt_ratio $t_gz $t_orig)" "$(fmt_ms $t_gz_ns) ms" "$(fmt_throughput $t_orig $t_gz_ns) MB/s"

    for (( pi=0; pi<${#PIPELINES[@]}; pi++ )); do
        p="${PIPELINES[$pi]}"
        tag=$(echo "$p" | tr '[:lower:]' '[:upper:]')
        printf "  %-10s %12s %8s %10s %10s\n" "pz-$p" "$(fmt_bytes ${t_pz_size[$pi]})" \
            "$(fmt_ratio ${t_pz_size[$pi]} $t_orig)" "$(fmt_ms ${t_pz_ns[$pi]}) ms" \
            "$(fmt_throughput $t_orig ${t_pz_ns[$pi]}) MB/s"
    done

    echo ""
    echo "Decompression Results:"
    printf "  %-10s %10s %10s\n" "Pipeline" "Time" "Throughput"
    printf "  %s\n" "────────────────────────────────────"
    printf "  %-10s %10s %10s\n" "gzip" "$(fmt_ms $dt_gz_ns) ms" "$(fmt_throughput $dt_orig $dt_gz_ns) MB/s"

    for (( pi=0; pi<${#PIPELINES[@]}; pi++ )); do
        p="${PIPELINES[$pi]}"
        printf "  %-10s %10s %10s\n" "pz-$p" "$(fmt_ms ${dt_pz_ns[$pi]}) ms" \
            "$(fmt_throughput $dt_orig ${dt_pz_ns[$pi]}) MB/s"
    done

    echo ""
    echo "Run with --verbose for per-file breakdown"
fi
