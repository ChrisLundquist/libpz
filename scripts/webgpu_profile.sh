#!/usr/bin/env bash
# Profile WebGPU GPU performance: per-stage timing, CPU vs GPU comparisons.
#
# Prerequisites:
#   WebGPU-capable GPU (Metal on macOS, Vulkan on Linux)
#
# Usage:
#   ./scripts/webgpu_profile.sh                # run full WebGPU profile suite
#
# This builds and runs the webgpu_profile example binary, which benchmarks:
#   - LZ77 match finding (GPU vs CPU, single-block and batched)
#   - Huffman encode (GPU vs CPU)
#   - Full pipeline compress/decompress (deflate, lzfi, lzssr)
#   - Buffer allocation and readback overhead
#   - Compression ratios

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Building webgpu_profile binary..."
cargo build --release --example webgpu_profile --manifest-path "$PROJECT_DIR/Cargo.toml"

BINARY="$PROJECT_DIR/target/release/examples/webgpu_profile"

echo "Running WebGPU profile..."
"$BINARY" "$@"
