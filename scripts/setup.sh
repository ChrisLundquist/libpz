#!/usr/bin/env bash
# setup.sh — Extract sample data archives (idempotent, skips if already done).
#
# Usage:
#   ./scripts/setup.sh           # extract all sample archives
#
# Called automatically by bench.sh and other scripts that need sample data.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SAMPLES_DIR="$PROJECT_DIR/samples"

# extract_archive ARCHIVE DEST_DIR
#   Extracts ARCHIVE into DEST_DIR if DEST_DIR is empty or missing.
#   Uses the archive's mtime as a stamp — re-extracts if the archive is newer.
extract_archive() {
    local archive="$1" dest="$2"

    if [[ ! -f "$archive" ]]; then
        echo "WARNING: $archive not found, skipping." >&2
        return 1
    fi

    # Skip if destination has files and archive hasn't changed since extraction.
    # We touch a hidden stamp file after successful extraction.
    local stamp="$dest/.extracted"
    if [[ -f "$stamp" ]] && [[ ! "$archive" -nt "$stamp" ]]; then
        return 0
    fi

    echo "Extracting $(basename "$archive") → $(basename "$dest")/"
    mkdir -p "$dest"
    tar -xzf "$archive" -C "$dest"
    touch "$stamp"
}

ok=0
extract_archive "$SAMPLES_DIR/cantrbry.tar.gz" "$SAMPLES_DIR/cantrbry" && ok=$((ok + 1))
extract_archive "$SAMPLES_DIR/large.tar.gz"    "$SAMPLES_DIR/large"    && ok=$((ok + 1))

if [[ $ok -eq 0 ]]; then
    echo "ERROR: No sample archives found in $SAMPLES_DIR" >&2
    exit 1
fi
