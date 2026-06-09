# CI clippy was weaker than local test.sh (no --all-targets)

## Insight
`scripts/test.sh` runs `cargo clippy --all-targets -- -D warnings` (lints lib +
tests + benches), but `.github/workflows/ci.yml` ran `cargo clippy -- -D warnings`
(lib only). So **test-code lints never failed CI** — they only showed up locally.

This let a `clippy::doc_lazy_continuation` error sit latent in a `#[cfg(test)]`
doc comment in `src/pipeline/blocks.rs` (a line starting `>65535` after `///`
parsed as a markdown blockquote). PR #129's CI lint passed green while local
`clippy --all-targets` failed on the same commit — confusing until I diffed the
two clippy invocations.

## Fix applied (this PR)
Changed ci.yml lint step to `cargo clippy --all-targets -- -D warnings` so CI
matches test.sh. Also reworded the offending doc comment.

## Worth promoting to CLAUDE.md?
A one-liner under "Key conventions" or the CI notes: *CI clippy and
`scripts/test.sh` should stay in sync; both use `--all-targets`. If a lint fails
locally but CI is green, check whether it's test-only code and whether the
invocations match.* The zero-warnings policy already says `clippy --all-targets`
must pass — CI now actually enforces that.
