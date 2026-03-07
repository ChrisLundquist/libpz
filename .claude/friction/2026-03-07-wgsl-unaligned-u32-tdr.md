# WGSL unaligned u32 reads cause GPU TDR on RDNA4

Attempted to optimize `sortlz_verify_matches` kernel with u32-level match
comparison via `read_u32_at()` (shift-merge of two adjacent u32 words for
unaligned byte positions). This caused a Windows TDR timeout (>3s for 8KB
input) on AMD RX 9070 XT (RDNA4) with wgpu 27.0.1.

**Root cause**: Non-coalesced global memory access. When adjacent GPU threads
read from different byte offsets within u32 words, the memory subsystem
serializes accesses, creating massive slowdown (1000x+ regression).

**Workaround**: Keep byte-by-byte `read_byte()` for match extension. The
per-byte approach has naturally coalesced access patterns since each thread
reads from sequential global memory addresses.

**Future**: Consider aligned-only u32 comparison (only when `distance % 4 == 0`)
or shared-memory-based match extension where each workgroup loads a tile of
input into local memory first.
