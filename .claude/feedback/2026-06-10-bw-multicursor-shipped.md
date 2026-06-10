# bw multi-cursor iBWT shipped — CLAUDE.md bw decode row is stale

PR `claude/bw-multicursor-ibwt` ships K=8 interleaved LF-cursor inverse BWT for
the `bw` pipeline (versioned header, `BW_CURSORS_FLAG` on the primary_index
field, 28 B/1 MiB block wire cost = +0.0027% raw). Measured CLI blob decode:
1.55x single-thread, 2.66x all-cores (0.412→0.155 s). The Silesia table's
"pz bw ... Decomp 1120" row should be re-measured/updated. Ratio unchanged to
2 decimals (27.8%). bbw not covered (per-factor sampling is a follow-up; see
docs/design-docs/ibwt-cursor-findings.md status note).

Gotcha preserved: K=8, not 16 — reproducible aarch64 register-spill cliff at
K=16/32 single-thread (788→220 MB/s chase).
