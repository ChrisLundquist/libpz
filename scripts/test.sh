#!/usr/bin/env bash
# test.sh — Run formatting, linting, compilation, and tests.
#
# Usage:
#   ./scripts/test.sh              # CPU-only (default features)
#   ./scripts/test.sh --quick      # Skip compilation-only checks, just lint+test
#   ./scripts/test.sh --fix        # Auto-fix fmt + clippy before checking
#   ./scripts/test.sh --webgpu     # Include WebGPU backend
#   ./scripts/test.sh --opencl     # Include OpenCL backend
#   ./scripts/test.sh --all        # All feature combinations

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Parse arguments
FEATURES=()
QUICK=false
ALL=false
FIX=false

for arg in "$@"; do
    case "$arg" in
        --webgpu)  FEATURES+=(webgpu) ;;
        --opencl)  FEATURES+=(opencl) ;;
        --all)     ALL=true ;;
        --quick)   QUICK=true ;;
        --fix)     FIX=true ;;
        -h|--help)
            sed -n '2,10p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 1
            ;;
    esac
done

PASSED=0
FAILED=0
SLOW_THRESHOLD=30
STEPS=()

pass() { PASSED=$((PASSED + 1)); STEPS+=("  PASS  $1"); }
fail() { FAILED=$((FAILED + 1)); STEPS+=("  FAIL  $1"); }

run_step() {
    local label="$1"
    shift
    local tmpfile
    tmpfile=$(mktemp)

    local start_time
    start_time=$(date +%s)

    if "$@" >"$tmpfile" 2>&1; then
        local elapsed=$(( $(date +%s) - start_time ))
        if [ "$elapsed" -ge "$SLOW_THRESHOLD" ]; then
            pass "$label (${elapsed}s — slow)"
        else
            pass "$label (${elapsed}s)"
        fi
    else
        local elapsed=$(( $(date +%s) - start_time ))
        fail "$label (${elapsed}s)"
        echo ""
        echo "── FAIL: $label ──"
        # Show warnings, errors, and surrounding context
        grep -C 3 -E '(^error|^warning|FAILED|panicked|cannot find|unresolved|unused|dead_code)' "$tmpfile" \
            | head -60
        echo "..."
        # Show the last few lines for test summary / exit info
        tail -5 "$tmpfile"
        echo ""
    fi
    rm -f "$tmpfile"
}

# ── Auto-fix (if requested) ──
if [ "$FIX" = true ]; then
    run_step "cargo fmt (autofix)" cargo fmt
    run_step "clippy --fix (autofix)" cargo clippy --fix --allow-dirty --allow-staged --all-targets -- -D warnings
fi

# ── Formatting ──
run_step "cargo fmt --check" cargo fmt --check

# ── Lint (default features) ──
run_step "clippy (default)" cargo clippy --all-targets -- -D warnings

# ── Build (default features) ──
if [ "$QUICK" = false ]; then
    run_step "build (default)" cargo build
fi

# ── Tests (default features) ──
run_step "test (default)" cargo test

# ── Feature-gated checks ──
if [ "$ALL" = true ]; then
    FEATURES=(webgpu opencl)
fi

if [ "${#FEATURES[@]}" -gt 0 ]; then
    for feat in "${FEATURES[@]}"; do
        run_step "clippy --features $feat" cargo clippy --all-targets --features "$feat" -- -D warnings
        if [ "$QUICK" = false ]; then
            run_step "build --features $feat" cargo build --features "$feat"
        fi
        run_step "test --features $feat" cargo test --features "$feat"
    done
fi

# ── Summary ──
echo ""
echo "════════════════════════════════════"
echo "Results: $PASSED passed, $FAILED failed"
echo "════════════════════════════════════"
for step in "${STEPS[@]}"; do
    echo "$step"
done

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
