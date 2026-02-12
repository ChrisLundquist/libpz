#!/usr/bin/env bash
# Profile a pz pipeline or stage with samply.
#
# Prerequisites:
#   cargo install samply
#
# Usage:
#   ./scripts/profile.sh                              # default: lzf compress, 256KB
#   ./scripts/profile.sh --pipeline deflate            # profile deflate compress
#   ./scripts/profile.sh --stage lz77                  # profile lz77 encode only
#   ./scripts/profile.sh --stage fse --decompress      # profile fse decode
#   ./scripts/profile.sh --pipeline lzf --size 1048576 # 1MB input
#
# All arguments are forwarded to the profile example binary.

set -euo pipefail

if ! command -v samply &>/dev/null; then
    echo "samply not found. Install with: cargo install samply"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Building profile binary..."
cargo build --profile profiling --example profile --manifest-path "$PROJECT_DIR/Cargo.toml"

BINARY="$PROJECT_DIR/target/profiling/examples/profile"

echo "Launching samply..."
samply record "$BINARY" "$@"
