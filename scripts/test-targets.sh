#!/usr/bin/env bash
# test-targets.sh — run multiple cargo test targets sequentially.
#
# Usage:
#   ./scripts/test-targets.sh target_one target_two ...
#   ./scripts/test-targets.sh --features webgpu -- target_one target_two

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  ./scripts/test-targets.sh [CARGO_TEST_OPTIONS...] -- <test_target> [test_target...]
  ./scripts/test-targets.sh <test_target> [test_target...]

Examples:
  ./scripts/test-targets.sh lz77::tests::test_lazy_quality_repeated_pattern pipeline::tests::test_optimal_deflate_round_trip
  ./scripts/test-targets.sh --features webgpu -- webgpu::tests::test_smoke
USAGE
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

cargo_opts=()
targets=()

if [[ " $* " == *" -- "* ]]; then
    while [[ $# -gt 0 ]]; do
        if [[ "$1" == "--" ]]; then
            shift
            break
        fi
        cargo_opts+=("$1")
        shift
    done
    if [[ $# -lt 1 ]]; then
        echo "ERROR: no test targets provided after --" >&2
        usage
        exit 1
    fi
    targets=("$@")
else
    targets=("$@")
fi

for target in "${targets[@]}"; do
    if [[ ${#cargo_opts[@]} -gt 0 ]]; then
        echo "==> cargo test ${cargo_opts[*]} $target"
        cargo test "${cargo_opts[@]}" "$target"
    else
        echo "==> cargo test $target"
        cargo test "$target"
    fi
done
