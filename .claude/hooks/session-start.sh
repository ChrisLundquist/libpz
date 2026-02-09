#!/bin/bash
set -euo pipefail

# Only run in Claude Code remote environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

echo '{"async": true, "asyncTimeout": 300000}'

# Build Rust dependencies (cached after first run)
cd "$CLAUDE_PROJECT_DIR/libpz-rs"
cargo build 2>&1
cargo clippy --version >/dev/null 2>&1 || rustup component add clippy

# Extract test sample data if not already present
cd "$CLAUDE_PROJECT_DIR/samples"
if [ ! -d cantrbry ]; then
  mkdir -p cantrbry
  tar -xzf cantrbry.tar.gz -C cantrbry
fi
if [ ! -d large ]; then
  mkdir -p large
  tar -xzf large.tar.gz -C large
fi
