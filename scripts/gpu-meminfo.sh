#!/usr/bin/env bash
# gpu-meminfo.sh — GPU memory cost calculator for LZ77 batched operations
#
# Parses actual buffer allocations from the codebase and computes
# batch sizes for different GPU memory configurations.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    cat <<'EOF'
gpu-meminfo.sh — GPU memory cost calculator for LZ77 batched operations

Parses actual buffer allocations from src/webgpu/lz77.rs to compute
GPU memory costs per block and recommend batch sizes for different
GPU memory budgets.

USAGE:
    ./scripts/gpu-meminfo.sh [OPTIONS]

OPTIONS:
    -b, --block SIZE        Show detailed memory breakdown for specific block size
                            (e.g., 262144, 1048576, default: all standard sizes)
    --explain               Show formula explanations and source locations
    -h, --help              Show this help

EXAMPLES:
    ./scripts/gpu-meminfo.sh                      # overview table
    ./scripts/gpu-meminfo.sh -b 262144            # 256KB block details
    ./scripts/gpu-meminfo.sh -b 1048576 --explain # 1MB block with formulas
EOF
}

BLOCK_SIZE=""
EXPLAIN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -b|--block)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --block requires an argument" >&2
                exit 1
            fi
            BLOCK_SIZE="$2"
            shift 2
            ;;
        --explain)
            EXPLAIN=true
            shift
            ;;
        *)
            echo "ERROR: unknown option '$1'" >&2
            echo "Run './scripts/gpu-meminfo.sh --help' for usage." >&2
            exit 1
            ;;
    esac
done

# Parse buffer allocations from src/webgpu/lz77.rs
# These are the actual source-of-truth allocations (lines 952-989)
#
# Slot structure (alloc_lz77_slot):
#   - input_buf: padded_size = ((block_size + 3) & !3) + 4
#   - params_buf: 16 bytes (4 x u32)
#   - raw_match_buf: block_size * sizeof(GpuMatch) = block_size * 12
#   - resolved_buf: block_size * sizeof(GpuMatch) = block_size * 12
#   - staging_buf: block_size * sizeof(GpuMatch) = block_size * 12
#
# GpuMatch is 3 x u32 = 12 bytes (lines 104-110)

compute_slot_memory() {
    local block_size=$1
    local padded_size=$(( ((block_size + 3) & ~3) + 4 ))
    local gpu_match_size=12
    local match_buf_size=$(( block_size * gpu_match_size ))

    local input_buf=$padded_size
    local params_buf=16
    local raw_match_buf=$match_buf_size
    local resolved_buf=$match_buf_size
    local staging_buf=$match_buf_size

    local total=$(( input_buf + params_buf + raw_match_buf + resolved_buf + staging_buf ))

    echo "$total"
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

# Standard GPU memory budgets
DISCRETE_GPU_MEM=$((4 * 1024 * 1024 * 1024))  # 4GB
INTEGRATED_GPU_MEM=$((1 * 1024 * 1024 * 1024)) # 1GB
LOW_GPU_MEM=$((512 * 1024 * 1024))             # 512MB

compute_batch_size() {
    local block_size=$1
    local budget=$2

    local slot_mem=$(compute_slot_memory "$block_size")

    # Reserve 25% headroom for non-ring allocations (from create_lz77_ring, line 1003)
    local ring_budget=$(( budget * 3 / 4 ))

    # Ring depth is clamped to [2, 3] (line 1008)
    local max_slots=$(( ring_budget / slot_mem ))
    if (( max_slots < 2 )); then
        echo "0"  # Insufficient memory, falls back to per-block alloc
    else
        local depth=$(( max_slots > 3 ? 3 : max_slots ))
        echo "$depth"
    fi
}

if [[ -n "$BLOCK_SIZE" ]]; then
    # Validate that BLOCK_SIZE is a number
    if ! [[ "$BLOCK_SIZE" =~ ^[0-9]+$ ]]; then
        echo "ERROR: block size must be a positive integer, got: '$BLOCK_SIZE'" >&2
        exit 1
    fi

    # Detailed breakdown for specific block size
    echo "=== GPU Memory Breakdown for Block Size: $(fmt_bytes $BLOCK_SIZE) ==="
    echo ""

    padded_size=$(( ((BLOCK_SIZE + 3) & ~3) + 4 ))
    match_buf_size=$(( BLOCK_SIZE * 12 ))

    echo "Per-block buffer allocations (from src/webgpu/lz77.rs:952-989):"
    echo "  input_buf (padded):  $(fmt_bytes $padded_size)"
    echo "  params_buf:          $(fmt_bytes 16)"
    echo "  raw_match_buf:       $(fmt_bytes $match_buf_size)"
    echo "  resolved_buf:        $(fmt_bytes $match_buf_size)"
    echo "  staging_buf:         $(fmt_bytes $match_buf_size)"
    echo "  ───────────────────────────────────────"

    total=$(compute_slot_memory "$BLOCK_SIZE")
    echo "  TOTAL per slot:      $(fmt_bytes $total)"
    echo ""

    if [[ "$EXPLAIN" == true ]]; then
        echo "Memory calculation formulas:"
        echo "  padded_size = ((block_size + 3) & ~3) + 4"
        echo "  match_buf_size = block_size × sizeof(GpuMatch)"
        echo "  sizeof(GpuMatch) = 12 bytes (3 × u32, see src/webgpu/mod.rs:104-110)"
        echo ""
        echo "Ring buffer allocation (src/webgpu/lz77.rs:997-1013):"
        echo "  ring_budget = gpu_memory_budget × 75%  (25% headroom)"
        echo "  max_slots = ring_budget / slot_memory"
        echo "  ring_depth = clamp(max_slots, 2, 3)"
        echo ""
        echo "  Depth < 2: Falls back to per-block allocation (no ring)"
        echo "  Depth = 2: Double buffering (GPU computes slot N, CPU reads slot N-1)"
        echo "  Depth = 3: Triple buffering (overlaps upload/compute/download)"
        echo ""
    fi

    echo "Recommended batch sizes for different GPU memory budgets:"
    echo ""
    printf "  %-20s %12s %10s %15s\n" "GPU Type" "Budget" "Depth" "Fallback"
    printf "  %s\n" "────────────────────────────────────────────────────────"

    discrete_depth=$(compute_batch_size "$BLOCK_SIZE" "$DISCRETE_GPU_MEM")
    integrated_depth=$(compute_batch_size "$BLOCK_SIZE" "$INTEGRATED_GPU_MEM")
    low_depth=$(compute_batch_size "$BLOCK_SIZE" "$LOW_GPU_MEM")

    printf "  %-20s %12s %10s %15s\n" "Discrete (4GB)" "$(fmt_bytes $DISCRETE_GPU_MEM)" "$discrete_depth" \
        "$([ "$discrete_depth" -eq 0 ] && echo "per-block alloc" || echo "ring buffer")"
    printf "  %-20s %12s %10s %15s\n" "Integrated (1GB)" "$(fmt_bytes $INTEGRATED_GPU_MEM)" "$integrated_depth" \
        "$([ "$integrated_depth" -eq 0 ] && echo "per-block alloc" || echo "ring buffer")"
    printf "  %-20s %12s %10s %15s\n" "Low memory (512MB)" "$(fmt_bytes $LOW_GPU_MEM)" "$low_depth" \
        "$([ "$low_depth" -eq 0 ] && echo "per-block alloc" || echo "ring buffer")"

    if [[ "$EXPLAIN" == true ]]; then
        echo ""
        echo "Implementation notes:"
        echo "  - Ring buffer path: compress_streaming_gpu() in src/pipeline/parallel.rs:226"
        echo "  - Per-block alloc path: compress_parallel_gpu_batched() in src/pipeline/parallel.rs:137"
        echo "  - Ring creation: create_lz77_ring() in src/webgpu/lz77.rs:997"
        echo "  - Slot allocation: alloc_lz77_slot() in src/webgpu/lz77.rs:952"
    fi

else
    # Overview table for standard block sizes
    echo "=== GPU Memory Costs per Block (LZ77 Cooperative Kernel) ==="
    echo ""
    echo "Memory usage per block based on actual buffer allocations in src/webgpu/lz77.rs"
    echo ""

    printf "%-12s %15s %15s %15s %15s\n" "Block Size" "Per Slot" "Discrete (4GB)" "Integrated (1GB)" "Low (512MB)"
    printf "%s\n" "─────────────────────────────────────────────────────────────────────────────────────"

    # Standard block sizes
    for size in 65536 131072 262144 524288 1048576 2097152; do
        slot_mem=$(compute_slot_memory "$size")
        discrete_depth=$(compute_batch_size "$size" "$DISCRETE_GPU_MEM")
        integrated_depth=$(compute_batch_size "$size" "$INTEGRATED_GPU_MEM")
        low_depth=$(compute_batch_size "$size" "$LOW_GPU_MEM")

        size_label=$(fmt_bytes "$size")
        slot_label=$(fmt_bytes "$slot_mem")

        discrete_label="$discrete_depth"
        integrated_label="$integrated_depth"
        low_label="$low_depth"

        [[ "$discrete_depth" -eq 0 ]] && discrete_label="fallback"
        [[ "$integrated_depth" -eq 0 ]] && integrated_label="fallback"
        [[ "$low_depth" -eq 0 ]] && low_label="fallback"

        printf "%-12s %15s %15s %15s %15s\n" \
            "$size_label" "$slot_label" "$discrete_label" "$integrated_label" "$low_label"
    done

    echo ""
    echo "Legend:"
    echo "  Per Slot:        Total GPU memory per ring buffer slot"
    echo "  Discrete (4GB):  Ring depth for 4GB discrete GPU (3 = triple buffering)"
    echo "  Integrated (1GB): Ring depth for 1GB integrated GPU (2-3)"
    echo "  Low (512MB):     Ring depth for low-memory GPU (fallback if < 2)"
    echo "  fallback:        Insufficient memory for ring, uses per-block allocation"
    echo ""
    echo "Run './scripts/gpu-meminfo.sh -b SIZE --explain' for detailed breakdown."
fi
