#!/usr/bin/env bash
# bench.sh — Compare pz pipelines vs gzip: size, ratio, time, throughput.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PZ="$PROJECT_DIR/target/release/pz"

# Defaults
ITERATIONS=3
PIPELINES=()
FILES=()

usage() {
    cat <<'EOF'
bench.sh — Compare pz pipelines vs gzip: size, ratio, time, throughput.

Usage:
  ./scripts/bench.sh [OPTIONS] [FILE ...]

Options:
  -n, --iters N          Number of iterations per operation (default: 3)
  -p, --pipelines LIST   Comma-separated list of pipelines to benchmark
                         (default: deflate,lza,lzr,lzf)
  -h, --help             Show this help

If no FILEs are given, benchmarks all files in samples/cantrbry and samples/large.

Examples:
  ./scripts/bench.sh                              # all corpus, all pipelines
  ./scripts/bench.sh myfile.bin                   # specific file
  ./scripts/bench.sh -p deflate,lza               # subset of pipelines
  ./scripts/bench.sh -n 10                        # more iterations
  ./scripts/bench.sh -n 1 -p deflate,lza file.txt # combine options
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
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
    PIPELINES=(deflate lza lzr lzf)
fi

# Collect input files from corpus if none given on command line
if [[ ${#FILES[@]} -eq 0 ]]; then
    for f in "$PROJECT_DIR"/samples/cantrbry/* "$PROJECT_DIR"/samples/large/*; do
        [[ -f "$f" ]] && FILES+=("$f")
    done
    if [[ ${#FILES[@]} -eq 0 ]]; then
        echo "No sample files found. Run:" >&2
        echo "  cd samples && mkdir -p cantrbry large && tar -xzf cantrbry.tar.gz -C cantrbry && tar -xzf large.tar.gz -C large" >&2
        exit 1
    fi
fi

# Build release binary
echo "Building pz (release)..."
cargo build --release --manifest-path "$PROJECT_DIR/Cargo.toml" --quiet 2>/dev/null || \
    cargo build --release --manifest-path "$PROJECT_DIR/Cargo.toml"

if [[ ! -x "$PZ" ]]; then
    echo "ERROR: $PZ not found after build" >&2
    exit 1
fi

if ! command -v gzip &>/dev/null; then
    echo "ERROR: gzip not found" >&2
    exit 1
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

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

fmt_ms() {
    awk "BEGIN { printf \"%.1f\", $1 / 1000000.0 }"
}

fmt_throughput() {
    local bytes=$1 ns=$2
    if [[ "$ns" -le 0 ]]; then
        echo "-.--"
    else
        awk "BEGIN { printf \"%.1f\", ($bytes / 1048576.0) / ($ns / 1000000000.0) }"
    fi
}

fmt_ratio() {
    awk "BEGIN { printf \"%.1f%%\", ($1/$2)*100 }"
}

echo "Averaging over $ITERATIONS iterations per operation."
echo "Pipelines: ${PIPELINES[*]}"
echo ""

# Build dynamic column header
hdr_file=$(printf "%-20s %8s" "FILE" "ORIG")
hdr_sep=""
hdr_file+=$(printf " | %8s %6s %7s %8s" "GZIP" "RATIO" "ms" "MB/s")
for p in "${PIPELINES[@]}"; do
    tag=$(echo "$p" | tr '[:lower:]' '[:upper:]')
    hdr_file+=$(printf " | %8s %6s %7s %8s" "PZ-$tag" "RATIO" "ms" "MB/s")
done

# === COMPRESSION ===
echo "=== COMPRESSION ==="
echo "$hdr_file"
col_width=${#hdr_file}
printf '%*s\n' "$col_width" '' | tr ' ' '-'

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
    cp "$file" "$TMPDIR/$name"
    gz_comp_ns=$(avg_ns gzip -k -f "$TMPDIR/$name")
    gz_size=$(stat -c%s "$TMPDIR/$name.gz" 2>/dev/null || stat -f%z "$TMPDIR/$name.gz" 2>/dev/null)
    rm -f "$TMPDIR/$name" "$TMPDIR/$name.gz"

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
        cp "$file" "$TMPDIR/$name"
        pz_comp_ns=$(avg_ns "$PZ" -k -f -p "$p" "$TMPDIR/$name")
        pz_size=$(stat -c%s "$TMPDIR/$name.pz" 2>/dev/null || stat -f%z "$TMPDIR/$name.pz" 2>/dev/null)
        rm -f "$TMPDIR/$name" "$TMPDIR/$name.pz"

        row+=$(printf " | %8d %6s %7s %8s" \
            "$pz_size" "$(fmt_ratio $pz_size $orig_size)" \
            "$(fmt_ms $pz_comp_ns)" "$(fmt_throughput $orig_size $pz_comp_ns)")

        t_pz_size[$pi]=$(( ${t_pz_size[$pi]} + pz_size ))
        t_pz_ns[$pi]=$(( ${t_pz_ns[$pi]} + pz_comp_ns ))
    done

    echo "$row"
done

# Totals row
printf '%*s\n' "$col_width" '' | tr ' ' '-'
total_row=$(printf "%-20s %8d" "TOTAL" "$t_orig")
total_row+=$(printf " | %8d %6s %7s %8s" \
    "$t_gz" "$(fmt_ratio $t_gz $t_orig)" \
    "$(fmt_ms $t_gz_ns)" "$(fmt_throughput $t_orig $t_gz_ns)")
for (( pi=0; pi<${#PIPELINES[@]}; pi++ )); do
    total_row+=$(printf " | %8d %6s %7s %8s" \
        "${t_pz_size[$pi]}" "$(fmt_ratio ${t_pz_size[$pi]} $t_orig)" \
        "$(fmt_ms ${t_pz_ns[$pi]})" "$(fmt_throughput $t_orig ${t_pz_ns[$pi]})")
done
echo "$total_row"

echo ""

# === DECOMPRESSION ===
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

dt_orig=0; dt_gz_ns=0
declare -a dt_pz_ns
for (( pi=0; pi<${#PIPELINES[@]}; pi++ )); do
    dt_pz_ns[$pi]=0
done

for file in "${FILES[@]}"; do
    name=$(basename "$file")
    orig_size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)

    # Prepare compressed versions
    cp "$file" "$TMPDIR/$name.src"

    # gzip
    cp "$TMPDIR/$name.src" "$TMPDIR/$name"
    gzip -f "$TMPDIR/$name"

    # pz pipelines
    for (( pi=0; pi<${#PIPELINES[@]}; pi++ )); do
        p="${PIPELINES[$pi]}"
        cp "$TMPDIR/$name.src" "$TMPDIR/$name"
        "$PZ" -f -p "$p" "$TMPDIR/$name"
        mv "$TMPDIR/$name.pz" "$TMPDIR/$name.$p.pz"
    done

    # Time decompressions
    gz_dec_ns=$(avg_ns gzip -d -k -f "$TMPDIR/$name.gz")

    drow=$(printf "%-20s %8d" "$name" "$orig_size")
    drow+=$(printf " | %7s %8s" \
        "$(fmt_ms $gz_dec_ns)" "$(fmt_throughput $orig_size $gz_dec_ns)")

    dt_orig=$((dt_orig + orig_size))
    dt_gz_ns=$((dt_gz_ns + gz_dec_ns))

    for (( pi=0; pi<${#PIPELINES[@]}; pi++ )); do
        p="${PIPELINES[$pi]}"
        pz_dec_ns=$(avg_ns "$PZ" -d -k -f "$TMPDIR/$name.$p.pz")
        drow+=$(printf " | %7s %8s" \
            "$(fmt_ms $pz_dec_ns)" "$(fmt_throughput $orig_size $pz_dec_ns)")
        dt_pz_ns[$pi]=$(( ${dt_pz_ns[$pi]} + pz_dec_ns ))
    done

    echo "$drow"

    # Cleanup
    rm -f "$TMPDIR/$name.src" "$TMPDIR/$name.gz" "$TMPDIR/$name"
    for p in "${PIPELINES[@]}"; do
        rm -f "$TMPDIR/$name.$p.pz" "$TMPDIR/$name.$p"
    done
done

printf '%*s\n' "$dcol_width" '' | tr ' ' '-'
dtotal=$(printf "%-20s %8d" "TOTAL" "$dt_orig")
dtotal+=$(printf " | %7s %8s" \
    "$(fmt_ms $dt_gz_ns)" "$(fmt_throughput $dt_orig $dt_gz_ns)")
for (( pi=0; pi<${#PIPELINES[@]}; pi++ )); do
    dtotal+=$(printf " | %7s %8s" \
        "$(fmt_ms ${dt_pz_ns[$pi]})" "$(fmt_throughput $dt_orig ${dt_pz_ns[$pi]})")
done
echo "$dtotal"
