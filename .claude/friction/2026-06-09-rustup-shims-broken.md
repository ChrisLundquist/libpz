# rustup itself is a broken symlink — `rustup run` no longer works either

Update to 2026-06-09-cargo-path-classifier.md: every shim in `~/.cargo/bin/`
(including `rustup` itself) is a symlink to a nonexistent `rustup` binary, so
the previously recommended `rustup run 1.96.0 cargo ...` now fails with
"no such file or directory" too.

What works (used all session, fmt/clippy/test/build all fine):

```sh
export PATH="$HOME/.rustup/toolchains/1.96.0-aarch64-apple-darwin/bin:$PATH"
cargo build/test/fmt/clippy ...
```

The toolchain bin contains cargo, rustc, rustfmt, cargo-fmt, cargo-clippy,
clippy-driver — everything needed, no rustup involvement. Suggestion for the
tooling agent stands: scripts/test.sh should fall back to the toolchain bin
path when `cargo` is missing from PATH (and not rely on rustup existing).
