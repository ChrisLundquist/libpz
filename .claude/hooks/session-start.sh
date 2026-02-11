#!/bin/bash
set -euo pipefail

# Only run in Claude Code remote environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

echo '{"async": true, "asyncTimeout": 300000}'

cd "$CLAUDE_PROJECT_DIR"

# Ensure clippy is available (quick no-op if already installed)
cargo clippy --version >/dev/null 2>&1 || rustup component add clippy

# Extract test sample data if not already present (fast, do first)
if [ -d samples ]; then
  cd samples
  if [ ! -d cantrbry ] && [ -f cantrbry.tar.gz ]; then
    mkdir -p cantrbry
    tar -xzf cantrbry.tar.gz -C cantrbry
  fi
  if [ ! -d large ] && [ -f large.tar.gz ]; then
    mkdir -p large
    tar -xzf large.tar.gz -C large
  fi
  cd "$CLAUDE_PROJECT_DIR"
fi

# Build Rust dependencies (cached after first run)
# Use --message-format=short to reduce output noise
cargo build 2>&1
