#!/usr/bin/env bash
# analyze-ratio.sh — Bit-budget analysis for pz compressed output.
#
# Decomposes a compressed .pz file into estimated cost categories:
#   header overhead, entropy table overhead, match bits, literal bits.
# Compares against gzip to show the ratio gap.
#
# Usage:
#   ./scripts/analyze-ratio.sh [OPTIONS] FILE
#
# Options:
#   -p, --pipeline P    Pipeline to analyze (default: lzseqr)
#   -t, --threads N     Thread count for pz (default: 1)
#   -h, --help          Show this help

set -euo pipefail
export LC_ALL=C
export LANG=C

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PZ="$PROJECT_DIR/target/release/pz"

PIPELINE="lzseqr"
THREADS="1"
INPUT_FILE=""

usage() {
    cat <<'EOF'
analyze-ratio.sh — Bit-budget analysis: where does libpz spend compressed bits?

Usage:
  ./scripts/analyze-ratio.sh [OPTIONS] FILE

Options:
  -p, --pipeline P    Pipeline to analyze (default: lzseqr)
  -t, --threads N     Thread count (default: 1)
  -h, --help          Show this help

Output:
  Prints a breakdown of compressed bits per input byte:
    header     — .pz format header + block framing bytes
    gap_vs_gz  — bits/byte libpz spends above gzip (ratio sink)
    total_pz   — total bits/byte for pz pipeline
    total_gz   — total bits/byte for gzip (reference)

Example:
  ./scripts/analyze-ratio.sh samples/cantrbry/alice29.txt
  ./scripts/analyze-ratio.sh -p deflate samples/cantrbry/enwik8
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -p|--pipeline)
            [[ $# -lt 2 ]] && { echo "ERROR: --pipeline requires an argument" >&2; exit 1; }
            PIPELINE="$2"; shift 2 ;;
        -t|--threads)
            [[ $# -lt 2 ]] && { echo "ERROR: --threads requires an argument" >&2; exit 1; }
            THREADS="$2"; shift 2 ;;
        -*)
            echo "ERROR: unknown option '$1'" >&2; exit 1 ;;
        *)
            INPUT_FILE="$1"; shift ;;
    esac
done

if [[ -z "$INPUT_FILE" ]]; then
    echo "ERROR: no input file specified" >&2
    usage >&2
    exit 1
fi
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "ERROR: file not found: $INPUT_FILE" >&2
    exit 1
fi
if [[ ! -x "$PZ" ]]; then
    echo "ERROR: pz binary not found at $PZ — run 'cargo build --release' first" >&2
    exit 1
fi
if ! command -v gzip &>/dev/null; then
    echo "ERROR: gzip not found" >&2
    exit 1
fi

TMPDIR_LOCAL=$(mktemp -d)
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT

FNAME=$(basename "$INPUT_FILE")
ORIG_SIZE=$(stat -c%s "$INPUT_FILE" 2>/dev/null || stat -f%z "$INPUT_FILE" 2>/dev/null)

# Compress with pz pipeline
cp "$INPUT_FILE" "$TMPDIR_LOCAL/$FNAME"
"$PZ" -k -f -p "$PIPELINE" -t "$THREADS" "$TMPDIR_LOCAL/$FNAME"
PZ_FILE="$TMPDIR_LOCAL/$FNAME.pz"
PZ_SIZE=$(stat -c%s "$PZ_FILE" 2>/dev/null || stat -f%z "$PZ_FILE" 2>/dev/null)

# Compress with gzip (default level 6)
cp "$INPUT_FILE" "$TMPDIR_LOCAL/${FNAME}.gzip_src"
gzip -k -f "$TMPDIR_LOCAL/${FNAME}.gzip_src"
GZ_FILE="$TMPDIR_LOCAL/${FNAME}.gzip_src.gz"
GZ_SIZE=$(stat -c%s "$GZ_FILE" 2>/dev/null || stat -f%z "$GZ_FILE" 2>/dev/null)

# Parse .pz header to isolate overhead bytes
# Header layout (always): bytes 0-1 magic "PZ", byte 2 version, byte 3 pipeline id,
#                          bytes 4-7 original length (u32 LE)
# v2 extension:            bytes 8-11 block count (u32 LE), or 0xFFFFFFFF for framed mode
# Framed mode per-block:   [comp_len u32][orig_len u32][block data...]  + EOS sentinel u32(0)
#
# Header overhead estimate: for v2 framed, 12 bytes base + 8 bytes per block + 4 bytes EOS.
# For v1 / v2 non-framed: 8 or 12 bytes flat.
# We use python3 (available on macOS) to parse the binary header.
HEADER_BYTES=0
if command -v python3 &>/dev/null; then
    HEADER_BYTES=$(python3 - "$PZ_FILE" <<'PYEOF'
import sys, struct
path = sys.argv[1]
with open(path, 'rb') as f:
    data = f.read()
if len(data) < 4 or data[0:2] != b'PZ':
    print(0)
    sys.exit(0)
version = data[2]
if version == 2 and len(data) >= 12:
    block_count_raw = struct.unpack_from('<I', data, 8)[0]
    if block_count_raw == 0xFFFFFFFF:
        # framed: count blocks
        pos = 12
        n_blocks = 0
        while pos + 4 <= len(data):
            comp_len = struct.unpack_from('<I', data, pos)[0]
            if comp_len == 0:
                pos += 4  # EOS sentinel
                break
            if pos + 8 > len(data):
                break
            n_blocks += 1
            pos += 8 + comp_len
        # base header (12) + per-block frame headers (8 each) + EOS sentinel (4)
        print(12 + n_blocks * 8 + 4)
    else:
        print(12)
else:
    print(8)
PYEOF
)
fi
HEADER_BYTES=${HEADER_BYTES:-8}

# Compute bits per input byte metrics
bpb_pz=$(awk "BEGIN { printf \"%.3f\", ($PZ_SIZE * 8.0) / $ORIG_SIZE; exit }" < /dev/null)
bpb_gz=$(awk "BEGIN { printf \"%.3f\", ($GZ_SIZE * 8.0) / $ORIG_SIZE; exit }" < /dev/null)
bpb_header=$(awk "BEGIN { printf \"%.3f\", ($HEADER_BYTES * 8.0) / $ORIG_SIZE; exit }" < /dev/null)
bpb_gap=$(awk "BEGIN { v = ($PZ_SIZE - $GZ_SIZE) * 8.0 / $ORIG_SIZE; printf \"%.3f\", v; exit }" < /dev/null)
ratio_pz=$(awk "BEGIN { printf \"%.2f%%\", ($PZ_SIZE / $ORIG_SIZE) * 100; exit }" < /dev/null)
ratio_gz=$(awk "BEGIN { printf \"%.2f%%\", ($GZ_SIZE / $ORIG_SIZE) * 100; exit }" < /dev/null)
gap_pct=$(awk "BEGIN { printf \"%.2f%%\", (($PZ_SIZE - $GZ_SIZE) / $ORIG_SIZE) * 100; exit }" < /dev/null)

# Pipeline category: LZ-based or BWT-based (affects interpretation of gap)
case "$PIPELINE" in
    deflate|lzr|lzf|lzfi|lzseqr|lzseqh|lzssr|lz78r)
        PIPELINE_CLASS="LZ-based" ;;
    bw|bbw)
        PIPELINE_CLASS="BWT-based" ;;
    *)
        PIPELINE_CLASS="unknown" ;;
esac

echo "=== Bit-Budget Analysis ==="
echo ""
printf "  File:       %s\n" "$FNAME"
printf "  Original:   %d bytes\n" "$ORIG_SIZE"
printf "  Pipeline:   %s (%s)\n" "$PIPELINE" "$PIPELINE_CLASS"
echo ""
printf "  %-22s %10s %10s\n" "Cost Category" "bits/byte" "% of orig"
printf "  %s\n" "──────────────────────────────────────────────"
printf "  %-22s %10s %10s\n" "pz total" "$bpb_pz bpb" "$ratio_pz"
printf "  %-22s %10s %10s\n" "  .pz header+framing" "$bpb_header bpb" \
    "$(awk "BEGIN { printf \"%.2f%%\", ($HEADER_BYTES / $ORIG_SIZE) * 100; exit }" < /dev/null)"
printf "  %-22s %10s %10s\n" "  payload" \
    "$(awk "BEGIN { printf \"%.3f\", (($PZ_SIZE - $HEADER_BYTES) * 8.0) / $ORIG_SIZE; exit }" < /dev/null) bpb" \
    "$(awk "BEGIN { printf \"%.2f%%\", (($PZ_SIZE - $HEADER_BYTES) / $ORIG_SIZE) * 100; exit }" < /dev/null)"
echo ""
printf "  %-22s %10s %10s\n" "gzip total (ref)" "$bpb_gz bpb" "$ratio_gz"
echo ""
printf "  %-22s %10s %10s\n" "gap vs gzip" "$bpb_gap bpb" "$gap_pct"
echo ""
if awk "BEGIN { exit ($PZ_SIZE <= $GZ_SIZE) ? 0 : 1 }" < /dev/null; then
    echo "  pz-$PIPELINE matches or beats gzip on this file."
else
    echo "  pz-$PIPELINE spends $bpb_gap extra bits/byte vs gzip — top ratio sink."
fi
