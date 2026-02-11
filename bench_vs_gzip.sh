#!/usr/bin/env bash
# bench_vs_gzip.sh — Compare pz (multi-stream Deflate) vs gzip compression.
#
# Usage: ./bench_vs_gzip.sh [FILE...]
#   If no files given, uses samples/cantrbry/* and samples/large/*.

set -euo pipefail

PZ="./target/release/pz"

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

printf "%-30s %10s | %10s %8s | %10s %8s | %10s %8s | %s\n" \
    "FILE" "ORIG" \
    "GZIP" "RATIO" \
    "PZ-DEFL" "RATIO" \
    "PZ-LZA" "RATIO" \
    "WINNER"
printf "%s\n" "$(printf '%.0s-' {1..120})"

total_orig=0
total_gzip=0
total_pz_deflate=0
total_pz_lza=0

for file in "${FILES[@]}"; do
    name=$(basename "$file")
    orig_size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)

    # gzip (default level 6)
    cp "$file" "$TMPDIR/$name"
    gzip -k -f "$TMPDIR/$name"
    gz_size=$(stat -c%s "$TMPDIR/$name.gz" 2>/dev/null || stat -f%z "$TMPDIR/$name.gz" 2>/dev/null)
    rm -f "$TMPDIR/$name" "$TMPDIR/$name.gz"

    # pz deflate
    cp "$file" "$TMPDIR/$name"
    "$PZ" -k -f -p deflate "$TMPDIR/$name"
    pz_defl_size=$(stat -c%s "$TMPDIR/$name.pz" 2>/dev/null || stat -f%z "$TMPDIR/$name.pz" 2>/dev/null)
    rm -f "$TMPDIR/$name" "$TMPDIR/$name.pz"

    # pz lza
    cp "$file" "$TMPDIR/$name"
    "$PZ" -k -f -p lza "$TMPDIR/$name"
    pz_lza_size=$(stat -c%s "$TMPDIR/$name.pz" 2>/dev/null || stat -f%z "$TMPDIR/$name.pz" 2>/dev/null)
    rm -f "$TMPDIR/$name" "$TMPDIR/$name.pz"

    # Ratios
    gz_ratio=$(awk "BEGIN { printf \"%.1f%%\", ($gz_size/$orig_size)*100 }")
    pz_defl_ratio=$(awk "BEGIN { printf \"%.1f%%\", ($pz_defl_size/$orig_size)*100 }")
    pz_lza_ratio=$(awk "BEGIN { printf \"%.1f%%\", ($pz_lza_size/$orig_size)*100 }")

    # Winner
    best=$gz_size
    winner="gzip"
    if [[ $pz_defl_size -lt $best ]]; then
        best=$pz_defl_size
        winner="pz-deflate"
    fi
    if [[ $pz_lza_size -lt $best ]]; then
        best=$pz_lza_size
        winner="pz-lza"
    fi

    printf "%-30s %10d | %10d %8s | %10d %8s | %10d %8s | %s\n" \
        "$name" "$orig_size" \
        "$gz_size" "$gz_ratio" \
        "$pz_defl_size" "$pz_defl_ratio" \
        "$pz_lza_size" "$pz_lza_ratio" \
        "$winner"

    total_orig=$((total_orig + orig_size))
    total_gzip=$((total_gzip + gz_size))
    total_pz_deflate=$((total_pz_deflate + pz_defl_size))
    total_pz_lza=$((total_pz_lza + pz_lza_size))
done

printf "%s\n" "$(printf '%.0s-' {1..120})"
gz_total_ratio=$(awk "BEGIN { printf \"%.1f%%\", ($total_gzip/$total_orig)*100 }")
pz_defl_total_ratio=$(awk "BEGIN { printf \"%.1f%%\", ($total_pz_deflate/$total_orig)*100 }")
pz_lza_total_ratio=$(awk "BEGIN { printf \"%.1f%%\", ($total_pz_lza/$total_orig)*100 }")

best=$total_gzip
winner="gzip"
if [[ $total_pz_deflate -lt $best ]]; then best=$total_pz_deflate; winner="pz-deflate"; fi
if [[ $total_pz_lza -lt $best ]]; then best=$total_pz_lza; winner="pz-lza"; fi

printf "%-30s %10d | %10d %8s | %10d %8s | %10d %8s | %s\n" \
    "TOTAL" "$total_orig" \
    "$total_gzip" "$gz_total_ratio" \
    "$total_pz_deflate" "$pz_defl_total_ratio" \
    "$total_pz_lza" "$pz_lza_total_ratio" \
    "$winner"
