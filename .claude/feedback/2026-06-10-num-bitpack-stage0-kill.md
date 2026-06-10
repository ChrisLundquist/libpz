# num-bitpack stage-0 spike: KILL (candidate for CLAUDE.md dead-ends list)

Vertical bit-packing + zero-word elimination (ndzip-style, num-G candidate #4
in gpu-path-research.md) cannot replace per-plane FSE in the Num pipeline:
measured +18.3pp (sao), +9.7pp (x-ray), +2.1pp (mr), +47.1pp (nci) of original
file size vs the shipping FSE planes, with the bitpack coder given its own
transform gating. Per-plane oracle (min of both coders) rescues nothing — FSE
wins every plane on 3 of 4 files; mr low-byte plane is the lone bitpack win
(0.983x, file oracle −0.51pp). Mechanism: Num's targets have dense skewed or
small-but-nonzero residual planes, not the zero bitplanes ndzip's float
residuals produce. The next gate (CPU-SIMD decoder of the new wire) is not
reached. See docs/design-docs/num-bitpack-stage0-findings.md; probe stays
reusable as numeric::probe_block + examples/num_bitpack_probe.rs.
