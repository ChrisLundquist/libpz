## GPU hash tables don't work for LZ77 — use probes instead

Two hash-table GPU kernels were tried and abandoned:
- lz77_hash.wgsl (global hash table): atomic inserts make bucket order nondeterministic. Quality collapsed to 6.25% on repetitive data because last-writer-wins loses the recent-match bias.
- lz77_local.wgsl (shared-memory hash table): per-workgroup 4KB window limit, single candidate per slot. LZ4-class ratio — not competitive.

The winning approach is the cooperative-stitch kernel (lz77_coop.wgsl): 1,788 probes/position using exhaustive near + strided far + shared-memory stitch. No atomics needed. 94% of brute-force quality.

**For future agents**: Don't try to build GPU hash tables for LZ matching. The atomic ordering problem is fundamental to the SIMT model. Probe-based scanning with spatial locality heuristics is the right approach.
