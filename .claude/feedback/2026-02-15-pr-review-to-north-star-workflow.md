## PR review → north star roadmap workflow

PR #91 review surfaced the GPU occupancy concern naturally during discussion, which led to the
chunking-first design in the roadmap. This arc (review → architectural question → design doc)
is productive and worth repeating.

- Future agents working on GPU rANS should read `docs/exec-plans/active/PLAN-unified-scheduler-north-star.md` before starting.
- The existing FSE kernels (`fse_encode.wgsl`, `fse_decode.wgsl`) use `@workgroup_size(1)` with one thread per lane — same GPU occupancy debt as naive rANS. If FSE ever becomes the bottleneck stage, it needs the same chunking treatment.
