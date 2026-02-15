#!/usr/bin/env bash
# Profile a pz pipeline or stage with samply.
#
# Prerequisites:
#   cargo install samply
#
# Profiles are saved to profiling/<git_sha>/<description>.json.gz by default.
#
# Usage:
#   ./scripts/profile.sh                              # lzf compress, 256KB
#   ./scripts/profile.sh --pipeline deflate            # profile deflate compress
#   ./scripts/profile.sh --stage lz77                  # profile lz77 encode only
#   ./scripts/profile.sh --stage fse --decompress      # profile fse decode
#   ./scripts/profile.sh --pipeline lzf --size 1048576 # 1MB input
#   ./scripts/profile.sh --web                         # open browser UI after recording
#
# Feature flags:
#   ./scripts/profile.sh --features webgpu             # build with webgpu
#   ./scripts/profile.sh --no-default-features         # disable default features (pure CPU)
#
# Extra samply arguments:
#   ./scripts/profile.sh --samply-args "--rate 4000 --reuse-threads"
#
# Environment:
#   SAMPLY=path/to/samply  Override samply binary location

set -euo pipefail

# Normalize locale for predictable CLI behavior.
export LC_ALL=C
export LANG=C

usage() {
    cat <<'EOF'
profile.sh — Profile pz pipelines and stages with samply

By default, saves a profile to profiling/<sha>/<description>.json.gz (headless).
Use --web to open the browser UI instead.

USAGE:
    ./scripts/profile.sh [OPTIONS] [PROFILE-ARGS...]

PROFILER OPTIONS:
    --web                   Open browser UI after recording (default: save only)
    -o, --output FILE       Override output path (skip auto-naming)
    --samply-args "ARGS"    Additional arguments passed directly to samply

BUILD OPTIONS:
    --features FEATURES     Cargo feature flags (default: uses Cargo.toml defaults)
    --no-default-features   Disable default features (pure CPU build)

PROFILE BINARY OPTIONS (forwarded to the profile example):
    --pipeline P            Pipeline: deflate, bw, bbw, lzr, lzf, lzfi, lzssr (default: lzf)
    --stage S               Profile a single stage: lz77, huffman, bwt, mtf, rle, fse, rans
    --decompress            Profile decompression instead of compression
    --iterations N          Number of iterations (default: 200)
    --size N                Input data size in bytes (default: 262144)

EXAMPLES:
    # Profile lz77 → profiling/a1b2c3d/lz77_encode_256KB.json.gz
    ./scripts/profile.sh --stage lz77

    # Profile pipeline → profiling/a1b2c3d/deflate_decompress_256KB.json.gz
    ./scripts/profile.sh --pipeline deflate --decompress

    # Open browser to inspect results interactively
    ./scripts/profile.sh --web --pipeline lzf

    # View a previously saved profile
    samply load profiling/a1b2c3d/lz77_encode_256KB.json.gz

    # Override output path
    ./scripts/profile.sh -o custom.json.gz --stage lz77

    # Pure CPU build (no GPU features)
    ./scripts/profile.sh --no-default-features --pipeline lzf
EOF
}

SAMPLY="${SAMPLY:-samply}"

if ! command -v "$SAMPLY" &>/dev/null; then
    echo "samply not found. Install with: cargo install samply"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default: headless (save-only). --web opts into browser UI.
SAVE_ONLY=true
OUTPUT_OVERRIDE=""
SAMPLY_ARGS=()
CARGO_FEATURES=()
PROFILE_ARGS=()

# Track profile binary args for auto-naming
PROF_PIPELINE="lzf"
PROF_STAGE=""
PROF_DECOMPRESS=false
PROF_SIZE="262144"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --web)
            SAVE_ONLY=false
            shift
            ;;
        -o|--output)
            OUTPUT_OVERRIDE="$2"
            shift 2
            ;;
        --samply-args)
            # shellcheck disable=SC2206
            SAMPLY_ARGS+=($2)
            shift 2
            ;;
        --features)
            CARGO_FEATURES+=("--features" "$2")
            shift 2
            ;;
        --no-default-features)
            CARGO_FEATURES+=("--no-default-features")
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        # Track profile binary args for auto-naming, but still forward them
        --pipeline|-p)
            PROF_PIPELINE="$2"
            PROFILE_ARGS+=("$1" "$2")
            shift 2
            ;;
        --stage|-s)
            PROF_STAGE="$2"
            PROFILE_ARGS+=("$1" "$2")
            shift 2
            ;;
        --decompress|-d)
            PROF_DECOMPRESS=true
            PROFILE_ARGS+=("$1")
            shift
            ;;
        --size)
            PROF_SIZE="$2"
            PROFILE_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            PROFILE_ARGS+=("$1")
            shift
            ;;
    esac
done

# Build descriptive output path: profiling/<sha>/<what>_<direction>_<size>.json.gz
build_output_path() {
    local sha dirty_suffix dir name what direction size_label

    sha=$(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")
    if ! git -C "$PROJECT_DIR" diff --quiet HEAD 2>/dev/null; then
        dirty_suffix="-dirty"
    else
        dirty_suffix=""
    fi

    if [[ -n "$PROF_STAGE" ]]; then
        what="$PROF_STAGE"
    else
        what="$PROF_PIPELINE"
    fi

    if [[ "$PROF_DECOMPRESS" == true ]]; then
        direction="decode"
    else
        direction="encode"
    fi

    # Human-readable size
    if (( PROF_SIZE >= 1048576 && PROF_SIZE % 1048576 == 0 )); then
        size_label="$((PROF_SIZE / 1048576))MB"
    elif (( PROF_SIZE >= 1024 && PROF_SIZE % 1024 == 0 )); then
        size_label="$((PROF_SIZE / 1024))KB"
    else
        size_label="${PROF_SIZE}B"
    fi

    dir="$PROJECT_DIR/profiling/${sha}${dirty_suffix}"
    name="${what}_${direction}_${size_label}.json.gz"

    echo "${dir}/${name}"
}

if [[ "$SAVE_ONLY" == true ]]; then
    if [[ -n "$OUTPUT_OVERRIDE" ]]; then
        OUTPUT="$OUTPUT_OVERRIDE"
    else
        OUTPUT="$(build_output_path)"
    fi
    mkdir -p "$(dirname "$OUTPUT")"
    SAMPLY_ARGS+=("--save-only" "--output" "$OUTPUT")
fi

echo "Building profile binary..."
cargo build --profile profiling --example profile \
    --manifest-path "$PROJECT_DIR/Cargo.toml" \
    "${CARGO_FEATURES[@]+"${CARGO_FEATURES[@]}"}"

BINARY="$PROJECT_DIR/target/profiling/examples/profile"

echo "Launching samply..."
if ! "$SAMPLY" record "${SAMPLY_ARGS[@]+"${SAMPLY_ARGS[@]}"}" \
    "$BINARY" "${PROFILE_ARGS[@]+"${PROFILE_ARGS[@]}"}"; then
    echo "samply record failed."
    echo "Hint: on macOS this may require running outside sandboxed execution contexts."
    exit 1
fi

if [[ "$SAVE_ONLY" == true ]]; then
    echo "Profile saved to $OUTPUT"
    echo "View with: samply load $OUTPUT"
    echo "Note: save-only JSON may remain unsymbolicated (meta.symbolicated=false)."
    echo "Browser symbolization via 'samply load' does not rewrite the saved JSON file."
    echo "Hotspot helper: ./scripts/samply-top-symbols.sh --profile $OUTPUT --binary $BINARY"
fi
