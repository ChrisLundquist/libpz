# North Star roadmap follow-ups

1. Align profiling command docs: North Star uses `./scripts/profile.sh ... --webgpu`, but `scripts/profile.sh` currently requires `--features webgpu`.
2. Clarify parity semantics: objective says "byte-identical to CPU interleaved rANS" while Phase 1 introduces chunked framing and `RANS_CHUNKED_FLAG`; define parity as byte-identical for the selected rANS mode.
3. Add explicit chunked-format limits and fallback rules (`num_chunks` and `chunk_original_lens` are `u16`) so large-block behavior is defined.
4. Add a `CLAUDE.md` North Star execution loop with exact commands, thresholds, and required benchmark artifacts.
5. Add phase DRIs + dated exit criteria in the roadmap to reduce ambiguity during execution.
6. Add a format-compatibility ADR covering flag allocation, versioning, and decode guarantees across single/interleaved/chunked rANS payloads.
7. Add CI smoke checks for North Star invariants: CPU↔GPU parity, graceful GPU fallback, and scheduler liveness under GPU submit failure.
