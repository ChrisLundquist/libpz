// Per-workgroup shared-memory hash table LZ77 match finding.
//
// Each workgroup processes an independent 4KB block of input using a
// 4096-slot hash table in workgroup-local memory (16KB). This eliminates
// both the global atomic contention of lz77_hash.wgsl and the 1788-probe
// brute-force of lz77_coop.wgsl.
//
// Algorithm (single kernel, 3 barriers):
//   1. INIT:  Threads cooperatively fill hash table with empty sentinel
//   2. BUILD: Each thread hashes its positions, atomicStore to shared ht
//   3. FIND:  Each thread looks up hash table, compares, writes match
//
// Pass 2 (resolve_lazy from lz77_coop.wgsl) runs as a separate dispatch.
//
// Tradeoff: match window limited to BLOCK_SIZE (4KB) instead of 32KB.
// This gives LZ4-level compression quality at much higher throughput.
//
// @pz_cost {
//   threads_per_element: 0.015625
//   passes: 1
//   buffers: input=N+7, params=16, output=N*12
//   local_mem: 16384
//   note: 1 workgroup per 4096 bytes, resolve_lazy is separate dispatch
// }

struct Lz77Match {
    offset: u32,
    length: u32,
    next: u32,
}

const WG_SIZE: u32 = 64u;
const BLOCK_SIZE: u32 = 4096u;
const POS_PER_THREAD: u32 = 64u;  // BLOCK_SIZE / WG_SIZE
const HASH_BITS: u32 = 12u;
const HASH_SIZE: u32 = 4096u;     // 1 << 12
const HASH_MASK: u32 = 4095u;
const MIN_MATCH: u32 = 3u;
const GOOD_ENOUGH: u32 = 128u;
const EMPTY: u32 = 0xFFFFFFFFu;

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<uniform> params: vec4<u32>;  // x = in_len, w = dispatch_width
@group(0) @binding(2) var<storage, read_write> match_output: array<Lz77Match>;

// Shared-memory hash table: 4096 slots × 4 bytes = 16KB.
// Each slot stores an absolute input position (EMPTY = unused).
var<workgroup> ht: array<atomic<u32>, 4096>;

// --- Byte access helpers ---

fn read_byte(pos: u32) -> u32 {
    let word_idx = pos / 4u;
    let byte_idx = pos % 4u;
    return (input[word_idx] >> (byte_idx * 8u)) & 0xFFu;
}

fn read_u32_at(pos: u32) -> u32 {
    let word_idx = pos / 4u;
    let shift = (pos % 4u) * 8u;
    if (shift == 0u) {
        return input[word_idx];
    }
    let lo = input[word_idx] >> shift;
    let hi = input[word_idx + 1u] << (32u - shift);
    return lo | hi;
}

// --- Hash function (same trigram hash as other LZ77 kernels) ---

fn hash3(pos: u32, len: u32) -> u32 {
    if (pos + 2u >= len) {
        return 0u;
    }
    let h = (read_byte(pos) << 12u) ^ (read_byte(pos + 1u) << 6u) ^ read_byte(pos + 2u);
    return h & HASH_MASK;
}

// --- Main kernel ---

@compute @workgroup_size(64)
fn find_matches_local(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let in_len = params.x;
    let dispatch_w = params.w;

    // Linearize 2D workgroup dispatch to get block ID
    let block_id = wg_id.x + wg_id.y * dispatch_w;
    let block_start = block_id * BLOCK_SIZE;

    // Early exit for entire workgroup if block is beyond input
    if (block_start >= in_len) {
        return;
    }

    let block_end = min(block_start + BLOCK_SIZE, in_len);

    // ── Phase 1: Initialize hash table ──
    // 64 threads × 64 slots each = 4096 slots
    for (var i = 0u; i < POS_PER_THREAD; i = i + 1u) {
        let slot = lid * POS_PER_THREAD + i;
        atomicStore(&ht[slot], EMPTY);
    }

    workgroupBarrier();

    // ── Phase 2: Build hash table ──
    // Each thread hashes its assigned positions into the shared table.
    // atomicStore gives last-writer-wins semantics (LZ4-style).
    for (var i = 0u; i < POS_PER_THREAD; i = i + 1u) {
        let pos = block_start + lid + i * WG_SIZE;
        if (pos + 2u < block_end) {
            let h = hash3(pos, in_len);
            atomicStore(&ht[h], pos);
        }
    }

    workgroupBarrier();

    // ── Phase 3: Find matches ──
    // Each thread looks up the hash table for its positions and compares.
    for (var i = 0u; i < POS_PER_THREAD; i = i + 1u) {
        let pos = block_start + lid + i * WG_SIZE;
        if (pos >= block_end) {
            continue;
        }

        var best: Lz77Match;
        best.offset = 0u;
        best.length = 0u;
        best.next = read_byte(pos);

        let remaining = block_end - pos;

        if (remaining >= MIN_MATCH && pos > block_start) {
            let h = hash3(pos, in_len);
            let candidate = atomicLoad(&ht[h]);

            if (candidate != EMPTY && candidate < pos && candidate >= block_start) {
                let dist = pos - candidate;

                // Compare bytes: u32-at-a-time for speed
                var max_len = remaining;
                if (dist < max_len) {
                    max_len = dist;
                }

                var match_len = 0u;
                let safe_limit = max_len & ~3u;
                loop {
                    if (match_len >= safe_limit) {
                        break;
                    }
                    let a = read_u32_at(candidate + match_len);
                    let b = read_u32_at(pos + match_len);
                    let diff = a ^ b;
                    if (diff != 0u) {
                        match_len = match_len + countTrailingZeros(diff) / 8u;
                        break;
                    }
                    match_len = match_len + 4u;
                }
                // Remaining 0-3 bytes
                loop {
                    if (match_len >= max_len) {
                        break;
                    }
                    if (read_byte(candidate + match_len) != read_byte(pos + match_len)) {
                        break;
                    }
                    match_len = match_len + 1u;
                }

                if (match_len >= MIN_MATCH) {
                    best.offset = dist;
                    best.length = match_len;
                }
            }
        }

        // Ensure room for the literal 'next' byte
        loop {
            if (best.length < remaining || best.length == 0u) {
                break;
            }
            best.length = best.length - 1u;
        }

        if (best.length > 0u && best.length < remaining) {
            best.next = read_byte(pos + best.length);
        }

        match_output[pos] = best;
    }
}
