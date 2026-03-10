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
PARETO=false
INCLUDE_SILESIA=false

usage() {
    cat <<'EOF'
bench.sh — Compare pz pipelines vs gzip: size, ratio, time, throughput.

Usage:
  ./scripts/bench.sh [OPTIONS] [FILE ...]

Options:
  -n, --iters N          Number of iterations per operation (default: 3)
  -p, --pipelines LIST   Comma-separated list of pipelines to benchmark
                         (default: deflate,lzf,lzseqr)
  -t, --threads N        Pass thread count to pz (-t N; 0=auto, 1=single-threaded)
  --all                  Benchmark all available pipelines
  --pareto               Single-thread Pareto table: all pipelines + all competitors,
                         sorted by ratio. Implies --all and -t 1.
  --webgpu               Build with WebGPU feature and pass --gpu to pz
  --features FEAT        Cargo features to enable (e.g. webgpu)
  --silesia              Include Silesia corpus files (samples/silesia/)
  -v, --verbose          Show detailed output (default: quiet, summary only)
  -h, --help             Show this help

If no FILEs are given, benchmarks all files in samples/cantrbry and samples/large.
Use --silesia to also include the larger Silesia corpus (211 MB).

Examples:
  ./scripts/bench.sh                              # all corpus, all pipelines
  ./scripts/bench.sh myfile.bin                   # specific file
  ./scripts/bench.sh -p deflate,lzf               # subset of pipelines
  ./scripts/bench.sh -t 1 -p lzseqr                # force single-threaded pz
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
            PIPELINES=(deflate bw bbw lzf lzfi lzseqr lzseqh sortlz)
            shift
            ;;
        --silesia)
            INCLUDE_SILESIA=true
            shift
            ;;
        --pareto)
            PARETO=true
            PIPELINES=(deflate bw bbw lzf lzfi lzseqr lzseqh sortlz)
            # Force single-thread for apples-to-apples comparison
            THREADS="1"
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
    PIPELINES=(deflate lzf lzseqr)
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
        case "$f" in *.tar.gz|*.pz|*.gz|*/.extracted) continue ;; esac
        FILES+=("$f")
    done
    if [[ "$INCLUDE_SILESIA" == true ]] && [[ -d "$PROJECT_DIR/samples/silesia" ]]; then
        for f in "$PROJECT_DIR"/samples/silesia/*; do
            [[ -f "$f" ]] || continue
            case "$f" in *.pz|*.gz) continue ;; esac
            FILES+=("$f")
        done
    fi
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

# Measure GPU init overhead (per-process cost) when webgpu is enabled.
# This is subtracted from per-file timings for accurate throughput reporting.
GPU_INIT_NS=0
if [[ -n "$GPU_FLAG" ]]; then
    gpu_probe="$BENCH_TMPDIR/_gpu_probe"
    printf 'x%.0s' {1..256} > "$gpu_probe"
    # Warmup (shader compilation caching)
    "$PZ" -k -f -p deflate $GPU_FLAG "$gpu_probe" >/dev/null 2>&1
    rm -f "$gpu_probe.pz"
    # Average 3 GPU runs on tiny data
    gpu_total=0
    for (( gi=0; gi<3; gi++ )); do
        gpu_ns=$(time_ns "$PZ" -k -f -p deflate $GPU_FLAG "$gpu_probe")
        gpu_total=$(( gpu_total + gpu_ns ))
        rm -f "$gpu_probe.pz"
    done
    GPU_INIT_NS=$(( gpu_total / 3 ))
    # CPU-only baseline on same data
    cpu_total=0
    for (( gi=0; gi<3; gi++ )); do
        cpu_ns=$(time_ns "$PZ" -k -f -p deflate "$gpu_probe")
        cpu_total=$(( cpu_total + cpu_ns ))
        rm -f "$gpu_probe.pz"
    done
    CPU_INIT_NS=$(( cpu_total / 3 ))
    GPU_INIT_NS=$(( GPU_INIT_NS - CPU_INIT_NS ))
    if [[ $GPU_INIT_NS -lt 0 ]]; then GPU_INIT_NS=0; fi
    rm -f "$gpu_probe"
fi

if [[ "$VERBOSE" == true ]]; then
    echo "Averaging over $ITERATIONS iterations per operation."
    echo "Pipelines: ${PIPELINES[*]}"
    if [[ -n "$THREADS" ]]; then
        echo "Threads: $THREADS"
    fi
    if [[ -n "$GPU_FLAG" ]]; then
        echo "GPU: $GPU_FLAG"
        if [[ $GPU_INIT_NS -gt 0 ]]; then
            echo "GPU init overhead: $(fmt_ms $GPU_INIT_NS) ms/invocation (subtracted)"
        fi
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
# Competitor accumulators
t_zng=0; t_zng_ns=0
t_lz4=0; t_lz4_ns=0
t_zst=0; t_zst_ns=0
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

    # --- zlib-ng compress ---
    # minigzip-ng reads from stdin; we avoid bash -c by using input redirection
    if [[ "$HAS_ZLIBBNG" == true ]]; then
        zng_out="$BENCH_TMPDIR/$name.zng"
        # Create a wrapper function to safely handle file input
        zng_compress() {
            minigzip-ng < "$file" > "$zng_out"
        }
        zng_comp_ns=$(avg_ns zng_compress)
        zng_size=$(stat -c%s "$zng_out" 2>/dev/null || stat -f%z "$zng_out" 2>/dev/null)
        rm -f "$zng_out"
        t_zng=$((t_zng + zng_size))
        t_zng_ns=$((t_zng_ns + zng_comp_ns))
    fi

    # --- lz4 compress ---
    # lz4 -c reads file directly, safe to pass filename without bash -c
    if [[ "$HAS_LZ4" == true ]]; then
        lz4_out="$BENCH_TMPDIR/$name.lz4"
        lz4_compress() {
            lz4 -q -c "$file" > "$lz4_out"
        }
        lz4_comp_ns=$(avg_ns lz4_compress)
        lz4_size=$(stat -c%s "$lz4_out" 2>/dev/null || stat -f%z "$lz4_out" 2>/dev/null)
        rm -f "$lz4_out"
        t_lz4=$((t_lz4 + lz4_size))
        t_lz4_ns=$((t_lz4_ns + lz4_comp_ns))
    fi

    # --- zstd compress ---
    # zstd -c reads file directly, safe to pass filename without bash -c
    if [[ "$HAS_ZSTD" == true ]]; then
        zst_out="$BENCH_TMPDIR/$name.zst"
        zst_compress() {
            zstd -q --single-thread -c "$file" > "$zst_out"
        }
        zst_comp_ns=$(avg_ns zst_compress)
        zst_size=$(stat -c%s "$zst_out" 2>/dev/null || stat -f%z "$zst_out" 2>/dev/null)
        rm -f "$zst_out"
        t_zst=$((t_zst + zst_size))
        t_zst_ns=$((t_zst_ns + zst_comp_ns))
    fi

    # --- pz pipelines ---
    for (( pi=0; pi<${#PIPELINES[@]}; pi++ )); do
        p="${PIPELINES[$pi]}"
        cp "$file" "$BENCH_TMPDIR/$name"
        pz_comp_ns=$(avg_ns "$PZ" -k -f -p "$p" ${THREADS:+-t "$THREADS"} $GPU_FLAG "$BENCH_TMPDIR/$name")
        # Subtract GPU init overhead for accurate per-file throughput
        pz_comp_ns=$(( pz_comp_ns - GPU_INIT_NS ))
        if [[ $pz_comp_ns -lt 0 ]]; then pz_comp_ns=1000000; fi  # 1ms floor
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
        # Subtract GPU init overhead for accurate per-file throughput
        pz_dec_ns=$(( pz_dec_ns - GPU_INIT_NS ))
        if [[ $pz_dec_ns -lt 0 ]]; then pz_dec_ns=1000000; fi
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
        echo "  GPU:        enabled (init overhead: $(fmt_ms $GPU_INIT_NS) ms subtracted)"
    fi
    echo ""

    echo "Compression Results:"
    printf "  %-10s %12s %8s %10s %10s\n" "Pipeline" "Size" "Ratio" "Time" "Throughput"
    printf "  %s\n" "────────────────────────────────────────────────────────────"
    printf "  %-10s %12s %8s %10s %10s\n" "gzip" "$(fmt_bytes $t_gz)" \
        "$(fmt_ratio $t_gz $t_orig)" "$(fmt_ms $t_gz_ns) ms" "$(fmt_throughput $t_orig $t_gz_ns) MB/s"
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

# === PARETO TABLE (only when --pareto flag is set) ===
if [[ "$PARETO" == true ]]; then
    echo ""
    echo "=== PARETO COMPARISON TABLE (single-thread, sorted by ratio) ==="
    echo ""
    echo "  Ratio = compressed/original (lower is better)"
    echo "  Throughput = MB/s compression speed (higher is better)"
    echo ""
    printf "  %-18s %8s %12s %12s\n" "Codec" "Ratio" "Throughput" "Size"
    printf "  %s\n" "────────────────────────────────────────────────────────"

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
        printf "  %-18s %8s %12s %12s\n" "$label" "$ratio_pct" "$throughput" "$size"
    done

    echo ""
    echo "  Pareto-dominant = lower ratio AND higher throughput than any competitor row above it."
fi
