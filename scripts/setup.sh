#!/usr/bin/env bash
# setup.sh — One-time project setup (idempotent, safe to re-run).
#
# Usage:
#   ./scripts/setup.sh           # run all setup steps
#
# Called automatically by test.sh, bench.sh, and other scripts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SAMPLES_DIR="$PROJECT_DIR/samples"

# ── Git hooks (worktree-aware) ──
ensure_hooks() {
    local want=".githooks"
    local git_dir
    git_dir="$(git rev-parse --git-dir 2>/dev/null)" || return 0

    # Detect worktree: git-dir for a linked worktree lives under .git/worktrees/<name>
    if [[ "$git_dir" == *"/worktrees/"* ]]; then
        # Per-worktree config so hooks don't bleed into other worktrees
        git config extensions.worktreeConfig true 2>/dev/null || true
        local current
        current="$(git config --worktree --get core.hooksPath 2>/dev/null)" || true
        if [[ "$current" != "$want" ]]; then
            git config --worktree core.hooksPath "$want"
        fi
    else
        local current
        current="$(git config --get core.hooksPath 2>/dev/null)" || true
        if [[ "$current" != "$want" ]]; then
            git config core.hooksPath "$want"
        fi
    fi
}
ensure_hooks

# ── Sample data archives ──
# extract_archive ARCHIVE DEST_DIR
#   Extracts ARCHIVE into DEST_DIR if DEST_DIR is empty or missing.
#   Uses the archive's mtime as a stamp — re-extracts if the archive is newer.
extract_archive() {
    local archive="$1" dest="$2"

    if [[ ! -f "$archive" ]]; then
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

# Sample archives are optional — only error when invoked directly
if [[ $ok -eq 0 ]] && [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "WARNING: No sample archives found in $SAMPLES_DIR" >&2
fi
