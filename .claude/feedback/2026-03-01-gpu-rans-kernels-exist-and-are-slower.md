## GPU rANS kernels exist and are slower than CPU

The north-star plan said "Critical gap: No GPU rANS kernels exist" — this was stale. Kernels were built in Feb 2026 (rans_encode.wgsl, rans_decode.wgsl). They work correctly (parity tests pass) but are 0.77x CPU on encode, 0.54x on decode.

Independent block splitting was tried extensively (15+ iterations, commit d2d75fe) and regressed to 0.3-0.6x. The rANS serial state dependency fundamentally limits GPU parallelism.

**For future agents**: Don't re-implement GPU rANS. Don't try independent block splitting to increase occupancy. These paths have been explored thoroughly. Check PLAN-p0a-gpu-rans-vertical-slice.md for the full history.
