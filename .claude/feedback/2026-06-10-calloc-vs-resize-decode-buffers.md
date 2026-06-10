# Fresh decode buffers: vec![0; n] (calloc), never with_capacity + resize

Refactoring pz2's decode buffer from `vec![0u8; n]` to
`Vec::with_capacity(n)` + `resize(n, 0)` silently cost +6 ms on the
202 MB blob decode (16.8 → 23 ms): `vec![0; n]` hits `alloc_zeroed`
(pre-zeroed mmap pages, no memset, lazy faulting), while
capacity+resize pays an explicit memset AND touches every page twice.

**Why:** large zeroed allocations are effectively free from the OS;
explicit zeroing of fresh pages doubles the memory traffic on hot
decode paths.

**How to apply:** when a decode/scratch buffer is allocated fresh each
call, keep `vec![0; n]`. Only use resize-on-a-reused-buffer when the
allocation is actually amortized across calls (then the memset only
covers the grown tail). `pz2::decode_into_arena` branches on
`arena.capacity() == 0` for exactly this reason.
