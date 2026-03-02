#!/usr/bin/env bash
# perf-gate.sh - Local same-laptop perf gate for unified scheduler work.
#
# Collects median throughput and scheduler telemetry for a fixed matrix, then
# compares to a baseline TSV.

set -euo pipefail

export LC_ALL=C
export LANG=C

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CARGO_PROFILE="profiling"
PROFILE_BIN="$PROJECT_DIR/target/${CARGO_PROFILE}/examples/profile"

SIZE=1048576
ITERATIONS=20
REPEATS=3
THREADS=0
PIPELINES_CSV="deflate,lzr,lzf,lzseqr"
CPU_ONLY=false
UPDATE_BASELINE=false
THROUGHPUT_REGRESSION_PCT=4.0
OVERHEAD_REGRESSION_ABS=0.020
BASELINE="$PROJECT_DIR/docs/generated/perf-gate-baseline.tsv"
RUN_ID="$(date +%F-%H%M%S)"
OUTPUT="$PROJECT_DIR/docs/generated/${RUN_ID}-perf-gate-run.tsv"

usage() {
    cat <<'EOF'
perf-gate.sh - local throughput + scheduler-overhead regression gate

Usage:
  ./scripts/perf-gate.sh [OPTIONS]

Options:
  --size N                    Input size in bytes (default: 1048576)
  --iterations N              profile-example loop iterations per run (default: 20)
  --repeats N                 repeated runs per case; must be odd (default: 3)
  --threads N                 pass thread count to profile (0=auto, default: 0)
  --pipelines LIST            comma-separated pipelines (default: deflate,lzr,lzf,lzseqr)
  --cpu-only                  skip WebGPU matrix
  --cargo-profile NAME        cargo profile for example binary (default: profiling)
  --baseline FILE             baseline TSV path
  --output FILE               run-output TSV path
  --update-baseline           write current run to baseline path
  --throughput-regression-pct Percent throughput drop allowed (default: 4.0)
  --overhead-regression-abs   Absolute scheduler-overhead-pct increase allowed (default: 0.020)
  -h, --help                  show help

Examples:
  ./scripts/perf-gate.sh --update-baseline
  ./scripts/perf-gate.sh --repeats 5 --iterations 30
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --size)
            SIZE="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --repeats)
            REPEATS="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --pipelines)
            PIPELINES_CSV="$2"
            shift 2
            ;;
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        --cargo-profile)
            CARGO_PROFILE="$2"
            PROFILE_BIN="$PROJECT_DIR/target/${CARGO_PROFILE}/examples/profile"
            shift 2
            ;;
        --baseline)
            BASELINE="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --update-baseline)
            UPDATE_BASELINE=true
            shift
            ;;
        --throughput-regression-pct)
            THROUGHPUT_REGRESSION_PCT="$2"
            shift 2
            ;;
        --overhead-regression-abs)
            OVERHEAD_REGRESSION_ABS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option '$1'" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if ! [[ "$REPEATS" =~ ^[0-9]+$ ]] || (( REPEATS <= 0 )); then
    echo "ERROR: --repeats must be a positive integer" >&2
    exit 1
fi
if (( REPEATS % 2 == 0 )); then
    echo "ERROR: --repeats must be odd to compute an exact median" >&2
    exit 1
fi
if ! [[ "$THREADS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --threads must be a non-negative integer" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
mkdir -p "$(dirname "$BASELINE")"

IFS=',' read -r -a PIPELINES <<< "$PIPELINES_CSV"

median() {
    local n="$#"
    printf '%s\n' "$@" | sort -g | awk -v idx=$(( (n + 1) / 2 )) 'NR == idx { print; exit }'
}

extract_stat() {
    local line="$1"
    local key="$2"
    printf '%s\n' "$line" | tr '\t' '\n' | awk -F'=' -v k="$key" '$1 == k { print $2; exit }'
}

echo "Building profile example..."
cargo build --profile "$CARGO_PROFILE" --example profile --manifest-path "$PROJECT_DIR/Cargo.toml" >/dev/null

if [[ ! -x "$PROFILE_BIN" ]]; then
    echo "ERROR: profile binary not found at $PROFILE_BIN" >&2
    exit 1
fi

HAS_WEBGPU=false
if [[ "$CPU_ONLY" == false ]]; then
    probe_out="$(mktemp)"
    set +e
    "$PROFILE_BIN" --pipeline lzf --size 262144 --iterations 1 --threads "$THREADS" --gpu --print-scheduler-stats >"$probe_out" 2>&1
    probe_ec=$?
    set -e
    if (( probe_ec == 0 )); then
        HAS_WEBGPU=true
        echo "WebGPU probe: available"
    elif grep -q "webgpu requested but unavailable" "$probe_out"; then
        echo "WebGPU probe: unavailable (GPU matrix will be skipped)"
    else
        echo "ERROR: WebGPU probe failed unexpectedly:" >&2
        cat "$probe_out" >&2
        rm -f "$probe_out"
        exit 1
    fi
    rm -f "$probe_out"
fi

printf "mode\tpipeline\tsize\titerations\trepeats\tmbps_median\tscheduler_overhead_pct_median\truns_median\ttotal_ns_median\ttracked_thread_time_ns_median\tstage_compute_ns_median\tqueue_wait_ns_median\tqueue_admin_ns_median\tgpu_handoff_ns_median\tgpu_try_send_full_count_median\tgpu_try_send_disconnected_count_median\tthreads\n" >"$OUTPUT"

run_case() {
    local mode="$1"
    local pipeline="$2"

    local mbps_vals=()
    local over_vals=()
    local runs_vals=()
    local total_vals=()
    local tracked_vals=()
    local stage_vals=()
    local qwait_vals=()
    local qadmin_vals=()
    local handoff_vals=()
    local full_vals=()
    local disc_vals=()

    for ((i = 1; i <= REPEATS; i++)); do
        local out
        if [[ "$mode" == "webgpu" ]]; then
            out="$("$PROFILE_BIN" --pipeline "$pipeline" --size "$SIZE" --iterations "$ITERATIONS" --threads "$THREADS" --gpu --print-scheduler-stats 2>&1)"
        else
            out="$("$PROFILE_BIN" --pipeline "$pipeline" --size "$SIZE" --iterations "$ITERATIONS" --threads "$THREADS" --print-scheduler-stats 2>&1)"
        fi

        local profile_line
        profile_line="$(printf '%s\n' "$out" | grep '^PROFILE_STATS' | tail -n1)"
        local mbps
        mbps="$(extract_stat "$profile_line" "mbps")"
        if [[ -z "$mbps" ]]; then
            # Backward-compatible fallback for older profile binaries.
            mbps="$(printf '%s\n' "$out" | sed -n 's/.* \([0-9][0-9.]*\) MB\/s.*/\1/p' | tail -n1)"
            if [[ -z "$mbps" ]]; then
                echo "ERROR: could not parse throughput for ${mode}/${pipeline}" >&2
                printf '%s\n' "$out" >&2
                exit 1
            fi
        fi

        local stats_line
        stats_line="$(printf '%s\n' "$out" | grep '^SCHEDULER_STATS' | tail -n1)"
        if [[ -z "$stats_line" ]]; then
            echo "ERROR: missing SCHEDULER_STATS for ${mode}/${pipeline}" >&2
            printf '%s\n' "$out" >&2
            exit 1
        fi

        mbps_vals+=("$mbps")
        over_vals+=("$(extract_stat "$stats_line" "scheduler_overhead_pct")")
        runs_vals+=("$(extract_stat "$stats_line" "runs")")
        total_vals+=("$(extract_stat "$stats_line" "total_ns")")
        tracked_vals+=("$(extract_stat "$stats_line" "tracked_thread_time_ns")")
        stage_vals+=("$(extract_stat "$stats_line" "stage_compute_ns")")
        qwait_vals+=("$(extract_stat "$stats_line" "queue_wait_ns")")
        qadmin_vals+=("$(extract_stat "$stats_line" "queue_admin_ns")")
        handoff_vals+=("$(extract_stat "$stats_line" "gpu_handoff_ns")")
        full_vals+=("$(extract_stat "$stats_line" "gpu_try_send_full_count")")
        disc_vals+=("$(extract_stat "$stats_line" "gpu_try_send_disconnected_count")")
    done

    local mbps_m over_m runs_m total_m tracked_m stage_m qwait_m qadmin_m handoff_m full_m disc_m
    mbps_m="$(median "${mbps_vals[@]}")"
    over_m="$(median "${over_vals[@]}")"
    runs_m="$(median "${runs_vals[@]}")"
    total_m="$(median "${total_vals[@]}")"
    tracked_m="$(median "${tracked_vals[@]}")"
    stage_m="$(median "${stage_vals[@]}")"
    qwait_m="$(median "${qwait_vals[@]}")"
    qadmin_m="$(median "${qadmin_vals[@]}")"
    handoff_m="$(median "${handoff_vals[@]}")"
    full_m="$(median "${full_vals[@]}")"
    disc_m="$(median "${disc_vals[@]}")"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$mode" "$pipeline" "$SIZE" "$ITERATIONS" "$REPEATS" \
        "$mbps_m" "$over_m" "$runs_m" "$total_m" "$tracked_m" "$stage_m" \
        "$qwait_m" "$qadmin_m" "$handoff_m" "$full_m" "$disc_m" "$THREADS" >>"$OUTPUT"

    echo "case ${mode}/${pipeline}: median ${mbps_m} MB/s, overhead ${over_m}"
}

for pipe in "${PIPELINES[@]}"; do
    run_case "cpu" "$pipe"
done

if [[ "$HAS_WEBGPU" == true ]]; then
    for pipe in "${PIPELINES[@]}"; do
        run_case "webgpu" "$pipe"
    done
fi

echo "Wrote run data: $OUTPUT"

if [[ "$UPDATE_BASELINE" == true ]]; then
    cp "$OUTPUT" "$BASELINE"
    echo "Updated baseline: $BASELINE"
    exit 0
fi

if [[ ! -f "$BASELINE" ]]; then
    echo "ERROR: baseline file not found: $BASELINE" >&2
    echo "Run with --update-baseline to create it from current measurements." >&2
    exit 2
fi

failures=0

while IFS=$'\t' read -r mode pipeline size iterations repeats mbps over _rest; do
    if [[ "$mode" == "mode" ]]; then
        continue
    fi
    baseline_row="$(awk -F'\t' -v m="$mode" -v p="$pipeline" -v s="$size" '
        NR > 1 && $1 == m && $2 == p && $3 == s { print; exit }
    ' "$BASELINE")"
    if [[ -z "$baseline_row" ]]; then
        echo "WARN: missing baseline entry for ${mode}/${pipeline}/${size}; skipping compare"
        continue
    fi

    base_mbps="$(printf '%s\n' "$baseline_row" | cut -f6)"
    base_over="$(printf '%s\n' "$baseline_row" | cut -f7)"

    reg_pct="$(awk -v cur="$mbps" -v base="$base_mbps" 'BEGIN {
        if (base <= 0) { print 0; exit }
        drop = (base - cur) / base * 100.0
        if (drop < 0) drop = 0
        printf "%.6f", drop
    }')"
    over_delta="$(awk -v cur="$over" -v base="$base_over" 'BEGIN {
        d = cur - base
        if (d < 0) d = 0
        printf "%.6f", d
    }')"

    throughput_fail="$(awk -v v="$reg_pct" -v t="$THROUGHPUT_REGRESSION_PCT" 'BEGIN { print (v > t) ? 1 : 0 }')"
    overhead_fail="$(awk -v v="$over_delta" -v t="$OVERHEAD_REGRESSION_ABS" 'BEGIN { print (v > t) ? 1 : 0 }')"

    if [[ "$throughput_fail" == "1" ]]; then
        echo "FAIL throughput ${mode}/${pipeline}: current=${mbps} baseline=${base_mbps} drop=${reg_pct}% > ${THROUGHPUT_REGRESSION_PCT}%"
        failures=$((failures + 1))
    fi
    if [[ "$overhead_fail" == "1" ]]; then
        echo "FAIL overhead ${mode}/${pipeline}: current=${over} baseline=${base_over} delta=${over_delta} > ${OVERHEAD_REGRESSION_ABS}"
        failures=$((failures + 1))
    fi
done <"$OUTPUT"

if (( failures > 0 )); then
    echo "perf-gate: FAILED (${failures} regression checks)"
    exit 1
fi

echo "perf-gate: PASS"
