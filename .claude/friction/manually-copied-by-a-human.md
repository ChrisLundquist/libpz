  Feedback (for .claude/feedback/):
  - PR review → north star roadmap is a productive workflow arc. The review surfaced the GPU occupancy concern naturally, which led to the chunking-first design in the roadmap. Future agents working on GPU rANS should
  read PLAN-unified-scheduler-north-star.md before starting.
  - The existing FSE kernels (@workgroup_size(1), one thread per lane) are a known GPU occupancy debt. If FSE ever becomes the bottleneck stage, it needs the same chunking treatment as rANS.
  Friction (for .claude/friction/):
  - Plan mode blocks writing feedback/friction files since they're outside the plan file path. When a session ends in plan mode, these notes have to be communicated in text rather than written to the proper locations.
  Workaround: write them before entering plan mode, or after exiting.
  I can't write these to disk while plan mode is active, but the exec plan itself is saved and ready for a future agent to pick up. Thanks for the good discussion — the GPU occupancy question meaningfully improved the
  plan.

