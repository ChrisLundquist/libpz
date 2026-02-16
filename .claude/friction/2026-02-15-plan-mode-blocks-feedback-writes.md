## Plan mode blocks writing feedback/friction files

When a session ends while plan mode is active, feedback and friction notes can't be written
to `.claude/feedback/` or `.claude/friction/` because plan mode only allows editing the plan file.

Workaround: write feedback/friction files before entering plan mode, or communicate them in
text so the user can copy them manually.

Possible fix: allow writes to `.claude/feedback/` and `.claude/friction/` as an exception in
plan mode, since they don't affect code.
