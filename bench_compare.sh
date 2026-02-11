#!/usr/bin/env bash
# bench_compare.sh — Compare pz against gzip, bzip2, zstd, xz, lz4, pigz
# Measures: compressed size, compression time, decompression time
set -euo pipefail

PZ="$(pwd)/target/release/pz"
CORPUS_DIR="/tmp/corpus"
RESULTS_DIR="/tmp/bench_results"
RUNS=3  # number of runs per measurement (take median)

mkdir -p "$RESULTS_DIR"

# CSV header
CSV="$RESULTS_DIR/results.csv"
echo "file,file_size,tool,level,compressed_size,ratio,compress_ms,decompress_ms" > "$CSV"

# Get median of array (expects sorted)
median() {
    local -a arr=("$@")
    local n=${#arr[@]}
    local mid=$((n / 2))
    echo "${arr[$mid]}"
}

# Measure time in milliseconds using date +%s%N
# Usage: measure_ms <command...>
# Prints elapsed time in ms to stdout
measure_ms() {
    local start end elapsed
    start=$(date +%s%N)
    "$@" > /dev/null 2>&1
    end=$(date +%s%N)
    elapsed=$(( (end - start) / 1000000 ))
    echo "$elapsed"
}

# Run a benchmark for a given tool/level/file
# bench <file> <tool_name> <level_name> <compress_cmd_args...>
# compress_cmd_args should read from stdin and write to stdout
bench_stdio() {
    local file="$1" tool="$2" level="$3"
    shift 3
    local compress_cmd=("$@")

    local fname
    fname=$(basename "$file")
    local fsize
    fsize=$(stat -c%s "$file")

    local compressed="$RESULTS_DIR/${fname}.${tool}.${level}"

    # Compress once to get size
    "${compress_cmd[@]}" < "$file" > "$compressed" 2>/dev/null
    local csize
    csize=$(stat -c%s "$compressed")
    local ratio
    ratio=$(awk "BEGIN { printf \"%.3f\", $csize / $fsize }")

    # Measure compression time (RUNS times, take median)
    local -a ctimes=()
    for ((i=0; i<RUNS; i++)); do
        local t
        t=$(measure_ms bash -c "$(printf '%q ' "${compress_cmd[@]}") < $(printf '%q' "$file") > /dev/null")
        ctimes+=("$t")
    done
    IFS=$'\n' ctimes_sorted=($(sort -n <<<"${ctimes[*]}")); unset IFS
    local ctime
    ctime=$(median "${ctimes_sorted[@]}")

    # Measure decompression time
    local -a dtimes=()
    local decompress_cmd
    case "$tool" in
        gzip|pigz) decompress_cmd="gzip -dc" ;;
        bzip2)     decompress_cmd="bzip2 -dc" ;;
        zstd)      decompress_cmd="zstd -dc" ;;
        xz)        decompress_cmd="xz -dc" ;;
        lz4)       decompress_cmd="lz4 -dc" ;;
        pz)        decompress_cmd="$PZ -dc" ;;
    esac
    for ((i=0; i<RUNS; i++)); do
        local t
        t=$(measure_ms bash -c "$decompress_cmd < $(printf '%q' "$compressed") > /dev/null")
        dtimes+=("$t")
    done
    IFS=$'\n' dtimes_sorted=($(sort -n <<<"${dtimes[*]}")); unset IFS
    local dtime
    dtime=$(median "${dtimes_sorted[@]}")

    echo "$fname,$fsize,$tool,$level,$csize,$ratio,$ctime,$dtime" >> "$CSV"
    printf "  %-10s %-14s %8d -> %8d  (%.3f)  compress: %5dms  decompress: %5dms\n" \
        "$tool" "$level" "$fsize" "$csize" "$ratio" "$ctime" "$dtime"

    rm -f "$compressed"
}

# Collect all corpus files sorted by size
mapfile -t FILES < <(find "$CORPUS_DIR" -type f | sort)

echo "================================================================"
echo "Compression Benchmark: pz vs gzip, bzip2, zstd, xz, lz4, pigz"
echo "================================================================"
echo "Corpus: Canterbury + Large ($(echo "${#FILES[@]}") files)"
echo "Runs per measurement: $RUNS (median)"
echo "================================================================"
echo ""

for file in "${FILES[@]}"; do
    fname=$(basename "$file")
    fsize=$(stat -c%s "$file")
    fsize_h=$(numfmt --to=iec "$fsize")
    echo "--- $fname ($fsize_h) ---"

    # gzip levels
    bench_stdio "$file" gzip "1(fast)"    gzip -1 -c
    bench_stdio "$file" gzip "6(default)" gzip -6 -c
    bench_stdio "$file" gzip "9(best)"    gzip -9 -c

    # pigz (parallel gzip)
    bench_stdio "$file" pigz "6(default)" pigz -6 -c
    bench_stdio "$file" pigz "9(best)"    pigz -9 -c

    # bzip2 levels
    bench_stdio "$file" bzip2 "1(fast)"    bzip2 -1 -c
    bench_stdio "$file" bzip2 "9(default)" bzip2 -9 -c

    # zstd levels
    bench_stdio "$file" zstd "1(fast)"     zstd -1 -c --no-progress
    bench_stdio "$file" zstd "3(default)"  zstd -3 -c --no-progress
    bench_stdio "$file" zstd "9(high)"     zstd -9 -c --no-progress
    bench_stdio "$file" zstd "19(ultra)"   zstd -19 -c --no-progress

    # xz levels
    bench_stdio "$file" xz "1(fast)"    xz -1 -c
    bench_stdio "$file" xz "6(default)" xz -6 -c
    bench_stdio "$file" xz "9(best)"    xz -9 -c

    # lz4 levels
    bench_stdio "$file" lz4 "1(fast)"   lz4 -1 -c
    bench_stdio "$file" lz4 "9(high)"   lz4 -9 -c

    # pz: deflate pipeline (greedy, lazy, optimal)
    bench_stdio "$file" pz "deflate-greedy"  "$PZ" -c -p deflate --greedy
    bench_stdio "$file" pz "deflate-lazy"    "$PZ" -c -p deflate --lazy
    bench_stdio "$file" pz "deflate-optimal" "$PZ" -c -p deflate -O

    # pz: bw pipeline
    bench_stdio "$file" pz "bw-lazy"         "$PZ" -c -p bw --lazy
    bench_stdio "$file" pz "bw-optimal"      "$PZ" -c -p bw -O

    # pz: lza pipeline
    bench_stdio "$file" pz "lza-greedy"      "$PZ" -c -p lza --greedy
    bench_stdio "$file" pz "lza-lazy"        "$PZ" -c -p lza --lazy
    bench_stdio "$file" pz "lza-optimal"     "$PZ" -c -p lza -O

    # pz: auto selection
    bench_stdio "$file" pz "auto"            "$PZ" -c -a
    bench_stdio "$file" pz "trial"           "$PZ" -c --trial

    echo ""
done

echo "================================================================"
echo "Results saved to $CSV"
echo "================================================================"
