# rust-toolchain.toml pins 1.96.0 but only `stable` was installed locally

## Problem
PR #127 added `rust-toolchain.toml` pinning `channel = "1.96.0"` for CI clippy
reproducibility. But the local box only had the `stable` toolchain installed
(which *is* 1.96.0). rustup treats `1.96.0` and `stable` as distinct named
toolchains, so the pin redirected builds to a toolchain dir that didn't exist.

The documented agent build path invokes the toolchain binary **directly**:
`~/.rustup/toolchains/stable-aarch64-apple-darwin/bin/cargo`. When cargo finds
`rust-toolchain.toml` it tries to honor the `1.96.0` pin and fails hard with:

```
error: could not execute process `rustc -vV` (never executed)
  No such file or directory (os error 2)
```

Cost me two failed background builds before I traced it to the missing named
toolchain. The `| tail`/`| grep` pipes also masked the real error initially
(PIPESTATUS) — the known gotcha bit again.

## Fix applied
Two parts, because installing the toolchain alone was necessary but NOT
sufficient for the documented direct-binary build path:
1. `rustup toolchain install 1.96.0 --component clippy rustfmt --profile minimal`
   so the named toolchain the pin requests actually exists.
2. Build via the rustup proxy, not the toolchain binary directly:
   `RUSTUP_HOME=~/.rustup rustup run 1.96.0 cargo build ...`
   The real `cargo` binary in a toolchain dir cannot locate `rustc` on its own
   here (no cargo/rustc proxy on PATH; `which rustc` → not found), so it execs a
   bare `rustc` and dies with "could not execute process `rustc -vV`". Going
   through `rustup run` (or setting `RUSTC=<toolchain>/bin/rustc`) fixes it.

## Durable suggestions
- The hardcoded `~/.rustup/toolchains/stable-.../bin/cargo` path in the agent
  docs is now wrong whenever the pin != `stable`. Prefer
  `rustup run 1.96.0 cargo ...` (proxy resolves rustc + honors the pin), or the
  `1.96.0-aarch64-apple-darwin/bin/cargo` with `RUSTC=` set.
- Document in CLAUDE.md that `rust-toolchain.toml` pins a specific version and
  `rustup toolchain install <pinned>` is a one-time local setup step.
- `scripts/setup.sh` could install the pinned toolchain to prevent this.
