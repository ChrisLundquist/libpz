#!/usr/bin/env bash
# Download and extract the Silesia compression corpus.
# https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia
#
# Usage: ./scripts/fetch-silesia.sh [--force]
#
# Downloads to samples/silesia/. Skips if already present unless --force.

set -euo pipefail

DEST="samples/silesia"
URL="https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip"
EXPECTED_FILES=12

if [[ -d "$DEST" ]] && [[ "$(ls "$DEST" | wc -l)" -ge "$EXPECTED_FILES" ]] && [[ "${1:-}" != "--force" ]]; then
    echo "Silesia corpus already present in $DEST ($(ls "$DEST" | wc -l) files)"
    echo "Use --force to re-download."
    exit 0
fi

TMPFILE="$(mktemp /tmp/silesia-XXXXXX.zip)"
trap 'rm -f "$TMPFILE"' EXIT

echo "Downloading Silesia corpus (66 MB)..."
curl -fSL "$URL" -o "$TMPFILE"

mkdir -p "$DEST"
echo "Extracting to $DEST..."
unzip -o "$TMPFILE" -d "$DEST"

# Clean up any .pz artifacts from previous runs
rm -f "$DEST"/*.pz

echo "Done: $(ls "$DEST" | wc -l) files, $(du -sh "$DEST" | cut -f1) total"
