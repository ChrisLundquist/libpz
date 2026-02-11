#!/usr/bin/env bash
# bench_vs_gzip.sh — Compare pz vs gzip: size, CPU time, and throughput.
#
# Usage: ./bench_vs_gzip.sh [FILE...]
#   If no files given, uses samples/cantrbry/* and samples/large/*.
#
# Each compress/decompress operation is timed with nanosecond precision.
# Throughput = original_size / cpu_time (compress) or original_size / cpu_time (decompress).

set -euo pipefail

PZ="./target/release/pz"
ITERATIONS=${BENCH_ITERS:-3}  # average over N runs; override with BENCH_ITERS=5 ./bench_vs_gzip.sh

if [[ ! -x "$PZ" ]]; then
    echo "ERROR: $PZ not found. Run: cargo build --release" >&2
    exit 1
fi

if ! command -v gzip &>/dev/null; then
    echo "ERROR: gzip not found" >&2
    exit 1
fi

# Collect input files
if [[ $# -gt 0 ]]; then
    FILES=("$@")
else
    FILES=()
    for f in samples/cantrbry/* samples/large/*; do
        [[ -f "$f" ]] && FILES+=("$f")
    done
    if [[ ${#FILES[@]} -eq 0 ]]; then
        echo "No sample files found. Run:" >&2
        echo "  cd samples && mkdir -p cantrbry large && tar -xzf cantrbry.tar.gz -C cantrbry && tar -xzf large.tar.gz -C large" >&2
        exit 1
    fi
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

# time_ns CMD ARGS... → prints wall-clock nanoseconds to stdout
# macOS `date` lacks %N, so use perl for portable sub-ms precision
time_ns() {
    local start end
    start=$(perl -MTime::HiRes=time -e 'printf "%d", time * 1e9')
    "$@" >/dev/null 2>&1
    end=$(perl -MTime::HiRes=time -e 'printf "%d", time * 1e9')
    echo $(( end - start ))
}

# Run a command $ITERATIONS times, return average nanoseconds
avg_ns() {
    local total=0 ns
    for (( i=0; i<ITERATIONS; i++ )); do
        ns=$(time_ns "$@")
        total=$(( total + ns ))
    done
    echo $(( total / ITERATIONS ))
}

# Format nanoseconds as milliseconds string
fmt_ms() {
    awk "BEGIN { printf \"%.1f\", $1 / 1000000.0 }"
}

# Format throughput: orig_bytes / ns → MB/s
fmt_throughput() {
    local bytes=$1 ns=$2
    if [[ "$ns" -le 0 ]]; then
        echo "-.--"
    else
        awk "BEGIN { printf \"%.1f\", ($bytes / 1048576.0) / ($ns / 1000000000.0) }"
    fi
}

echo "Averaging over $ITERATIONS iterations per operation."
echo ""

# === COMPRESSION TABLE ===
echo "=== COMPRESSION ==="
printf "%-16s %8s | %8s %6s %7s %8s | %8s %6s %7s %8s | %8s %6s %7s %8s\n" \
    "FILE" "ORIG" \
    "GZIP" "RATIO" "ms" "MB/s" \
    "PZ-DEFL" "RATIO" "ms" "MB/s" \
    "PZ-LZA" "RATIO" "ms" "MB/s"
printf "%s\n" "$(printf '%.0s-' {1..140})"

# Accumulators for totals
t_orig=0; t_gz=0; t_gz_ns=0
t_pzd=0; t_pzd_ns=0
t_pzl=0; t_pzl_ns=0

for file in "${FILES[@]}"; do
    name=$(basename "$file")
    orig_size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)

    # --- gzip compress ---
    cp "$file" "$TMPDIR/$name"
    gz_comp_ns=$(avg_ns gzip -k -f "$TMPDIR/$name")
    gz_size=$(stat -c%s "$TMPDIR/$name.gz" 2>/dev/null || stat -f%z "$TMPDIR/$name.gz" 2>/dev/null)
    rm -f "$TMPDIR/$name" "$TMPDIR/$name.gz"

    # --- pz deflate compress ---
    cp "$file" "$TMPDIR/$name"
    pzd_comp_ns=$(avg_ns "$PZ" -k -f -p deflate "$TMPDIR/$name")
    pzd_size=$(stat -c%s "$TMPDIR/$name.pz" 2>/dev/null || stat -f%z "$TMPDIR/$name.pz" 2>/dev/null)
    rm -f "$TMPDIR/$name" "$TMPDIR/$name.pz"

    # --- pz lza compress ---
    cp "$file" "$TMPDIR/$name"
    pzl_comp_ns=$(avg_ns "$PZ" -k -f -p lza "$TMPDIR/$name")
    pzl_size=$(stat -c%s "$TMPDIR/$name.pz" 2>/dev/null || stat -f%z "$TMPDIR/$name.pz" 2>/dev/null)
    rm -f "$TMPDIR/$name" "$TMPDIR/$name.pz"

    # Ratios
    gz_ratio=$(awk "BEGIN { printf \"%.1f%%\", ($gz_size/$orig_size)*100 }")
    pzd_ratio=$(awk "BEGIN { printf \"%.1f%%\", ($pzd_size/$orig_size)*100 }")
    pzl_ratio=$(awk "BEGIN { printf \"%.1f%%\", ($pzl_size/$orig_size)*100 }")

    printf "%-16s %8d | %8d %6s %7s %8s | %8d %6s %7s %8s | %8d %6s %7s %8s\n" \
        "$name" "$orig_size" \
        "$gz_size" "$gz_ratio" "$(fmt_ms $gz_comp_ns)" "$(fmt_throughput $orig_size $gz_comp_ns)" \
        "$pzd_size" "$pzd_ratio" "$(fmt_ms $pzd_comp_ns)" "$(fmt_throughput $orig_size $pzd_comp_ns)" \
        "$pzl_size" "$pzl_ratio" "$(fmt_ms $pzl_comp_ns)" "$(fmt_throughput $orig_size $pzl_comp_ns)"

    t_orig=$((t_orig + orig_size))
    t_gz=$((t_gz + gz_size));      t_gz_ns=$((t_gz_ns + gz_comp_ns))
    t_pzd=$((t_pzd + pzd_size));   t_pzd_ns=$((t_pzd_ns + pzd_comp_ns))
    t_pzl=$((t_pzl + pzl_size));   t_pzl_ns=$((t_pzl_ns + pzl_comp_ns))
done

printf "%s\n" "$(printf '%.0s-' {1..140})"
printf "%-16s %8d | %8d %6s %7s %8s | %8d %6s %7s %8s | %8d %6s %7s %8s\n" \
    "TOTAL" "$t_orig" \
    "$t_gz" "$(awk "BEGIN{printf\"%.1f%%\",$t_gz/$t_orig*100}")" "$(fmt_ms $t_gz_ns)" "$(fmt_throughput $t_orig $t_gz_ns)" \
    "$t_pzd" "$(awk "BEGIN{printf\"%.1f%%\",$t_pzd/$t_orig*100}")" "$(fmt_ms $t_pzd_ns)" "$(fmt_throughput $t_orig $t_pzd_ns)" \
    "$t_pzl" "$(awk "BEGIN{printf\"%.1f%%\",$t_pzl/$t_orig*100}")" "$(fmt_ms $t_pzl_ns)" "$(fmt_throughput $t_orig $t_pzl_ns)"

echo ""

# === DECOMPRESSION TABLE ===
echo "=== DECOMPRESSION ==="
printf "%-16s %8s | %7s %8s | %7s %8s | %7s %8s\n" \
    "FILE" "ORIG" \
    "GZIP-ms" "MB/s" \
    "PZ-D-ms" "MB/s" \
    "PZ-L-ms" "MB/s"
printf "%s\n" "$(printf '%.0s-' {1..85})"

dt_gz_ns=0; dt_pzd_ns=0; dt_pzl_ns=0; dt_orig=0

for file in "${FILES[@]}"; do
    name=$(basename "$file")
    orig_size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)

    # Prepare compressed files
    cp "$file" "$TMPDIR/$name.src"

    # gzip compressed
    cp "$TMPDIR/$name.src" "$TMPDIR/$name"
    gzip -f "$TMPDIR/$name"

    # pz deflate compressed
    cp "$TMPDIR/$name.src" "$TMPDIR/$name"
    "$PZ" -f -p deflate "$TMPDIR/$name"
    mv "$TMPDIR/$name.pz" "$TMPDIR/$name.defl.pz"

    # pz lza compressed
    cp "$TMPDIR/$name.src" "$TMPDIR/$name"
    "$PZ" -f -p lza "$TMPDIR/$name"
    mv "$TMPDIR/$name.pz" "$TMPDIR/$name.lza.pz"

    # --- gzip decompress ---
    gz_dec_ns=$(avg_ns gzip -d -k -f "$TMPDIR/$name.gz")

    # --- pz deflate decompress ---
    pzd_dec_ns=$(avg_ns "$PZ" -d -k -f "$TMPDIR/$name.defl.pz")

    # --- pz lza decompress ---
    pzl_dec_ns=$(avg_ns "$PZ" -d -k -f "$TMPDIR/$name.lza.pz")

    printf "%-16s %8d | %7s %8s | %7s %8s | %7s %8s\n" \
        "$name" "$orig_size" \
        "$(fmt_ms $gz_dec_ns)" "$(fmt_throughput $orig_size $gz_dec_ns)" \
        "$(fmt_ms $pzd_dec_ns)" "$(fmt_throughput $orig_size $pzd_dec_ns)" \
        "$(fmt_ms $pzl_dec_ns)" "$(fmt_throughput $orig_size $pzl_dec_ns)"

    dt_orig=$((dt_orig + orig_size))
    dt_gz_ns=$((dt_gz_ns + gz_dec_ns))
    dt_pzd_ns=$((dt_pzd_ns + pzd_dec_ns))
    dt_pzl_ns=$((dt_pzl_ns + pzl_dec_ns))

    rm -f "$TMPDIR/$name.src" "$TMPDIR/$name.gz" "$TMPDIR/$name.defl.pz" \
          "$TMPDIR/$name.lza.pz" "$TMPDIR/$name.defl" "$TMPDIR/$name.lza" \
          "$TMPDIR/$name"
done

printf "%s\n" "$(printf '%.0s-' {1..85})"
printf "%-16s %8d | %7s %8s | %7s %8s | %7s %8s\n" \
    "TOTAL" "$dt_orig" \
    "$(fmt_ms $dt_gz_ns)" "$(fmt_throughput $dt_orig $dt_gz_ns)" \
    "$(fmt_ms $dt_pzd_ns)" "$(fmt_throughput $dt_orig $dt_pzd_ns)" \
    "$(fmt_ms $dt_pzl_ns)" "$(fmt_throughput $dt_orig $dt_pzl_ns)"
