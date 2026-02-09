#!/bin/bash
set -euo pipefail

# Only run in Claude Code remote environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Build Rust dependencies (cached after first run)
cd "$CLAUDE_PROJECT_DIR/libpz-rs"
cargo build 2>&1
cargo clippy --version >/dev/null 2>&1 || rustup component add clippy
