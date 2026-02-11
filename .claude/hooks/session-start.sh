#!/bin/bash
set -euo pipefail

# Only run in Claude Code remote environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# Ensure clippy is available (quick no-op if already installed)
cargo clippy --version >/dev/null 2>&1 || rustup component add clippy
