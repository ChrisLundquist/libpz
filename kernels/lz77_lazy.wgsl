// LZ77 GPU-parallel match finding with lazy matching emulation.
//
// Two-pass approach:
//   Pass 1 (find_matches): Each invocation scans backward using a two-tier
//     strategy:
//       - Near window (1-1024): full brute-force scan of every position,
//         using workgroup shared memory tile for spot-check reads.
//         A 64-thread workgroup cooperatively loads ~1280 bytes covering
//         the shared lookback region once, replacing ~65K scattered global
//         reads with fast shared memory reads.
//       - Far window (1024-32768): subsampled scan every 4th position
//         (uses global reads since the working set is too large for tiling)
//     Uses spot-check optimization and u32-wide comparison for speed.
//   Pass 2 (resolve_lazy): Each invocation reads match[pos] and match[pos+1].
//     If pos+1 has a strictly longer match AND pos's match isn't too long to
//     bother checking, pos becomes a literal. This emulates gzip-style lazy
//     matching on the GPU in a single parallel pass.
//
// The brute-force scan avoids the hash table entirely, eliminating bucket
// overflow issues that caused catastrophic quality loss on repetitive data.
// Spot-check pre-filtering keeps the scan fast: ~99.6% of candidates are
// eliminated by a single byte comparison. Far subsampling captures most
// long-distance matches while keeping probe count manageable (~9K probes
// per position vs 1K for near-only). Achieves ~97% of CPU match quality
// on Canterbury corpus text.

// @pz_cost {
//   threads_per_element: 1
//   passes: 2
//   buffers: input=N, raw_matches=N*12, resolved=N*12, staging=N*12
//   local_mem: 1280
// }

struct Lz77Match {
    offset: u32,
    length: u32,
    next: u32,
}

const NEAR_WINDOW: u32 = 1024u;
const FAR_WINDOW: u32 = 32768u;
const FAR_STEP: u32 = 8u;
const MIN_MATCH: u32 = 3u;
// Matches this long skip lazy evaluation (unlikely to be beaten).
const LAZY_SKIP_THRESHOLD: u32 = 32u;
const WG_SIZE: u32 = 64u;
// Shared memory tile: 320 u32 words = 1280 bytes.
// Covers: NEAR_WINDOW (256 words) + WG_SIZE (16 words) + extension margin (48 words).
const TILE_WORDS: u32 = 320u;

// ========================== Pass 1: find_matches ==============================
// Bindings: input(0), params(1), match_output(2)

@group(0) @binding(0) var<storage, read> input_p1: array<u32>;
@group(0) @binding(1) var<uniform> params_p1: vec4<u32>; // x = in_len, w = dispatch_width
@group(0) @binding(2) var<storage, read_write> match_output: array<Lz77Match>;

// Workgroup shared memory tile for near-window spot-check acceleration.
var<workgroup> tile: array<u32, 320>;

// --- Global memory read functions ---

fn read_byte_p1(pos: u32) -> u32 {
    let word_idx = pos / 4u;
    let byte_idx = pos % 4u;
    return (input_p1[word_idx] >> (byte_idx * 8u)) & 0xFFu;
}

fn read_u32_at_p1(pos: u32) -> u32 {
    let word_idx = pos / 4u;
    let shift = (pos % 4u) * 8u;
    if (shift == 0u) {
        return input_p1[word_idx];
    }
    let lo = input_p1[word_idx] >> shift;
    let hi = input_p1[word_idx + 1u] << (32u - shift);
    return lo | hi;
}

// --- Shared memory (tile) read function ---

// Read a byte from the workgroup tile.
// tw_word_start: the global word index that tile[0] corresponds to.
fn read_byte_tw(pos: u32, tw_word_start: u32) -> u32 {
    let word_idx = pos / 4u - tw_word_start;
    let byte_idx = pos % 4u;
    return (tile[word_idx] >> (byte_idx * 8u)) & 0xFFu;
}

// --- Match functions ---

// Try matching in the near window using tile reads for spot-checks.
// The candidate and pos first-byte positions are guaranteed to be in the tile.
// The second spot-check (at best_len offset) falls back to global reads if
// the positions exceed the tile range. Match extension always uses global reads.
fn try_match_near(
    candidate: u32, pos: u32, remaining: u32, best_len: u32,
    tw_word_start: u32, tw_byte_end: u32,
) -> u32 {
    // Spot-check #1: first byte (always in tile for near-window candidates)
    if (read_byte_tw(candidate, tw_word_start) != read_byte_tw(pos, tw_word_start)) {
        return 0u;
    }

    // Spot-check #2: byte at current best length
    if (best_len >= MIN_MATCH && best_len < remaining) {
        let c_pos = candidate + best_len;
        let p_pos = pos + best_len;
        if (c_pos < tw_byte_end && p_pos < tw_byte_end) {
            // Both positions in tile — use shared memory
            if (read_byte_tw(c_pos, tw_word_start) != read_byte_tw(p_pos, tw_word_start)) {
                return 0u;
            }
        } else {
            // Fall back to global memory
            if (read_byte_p1(c_pos) != read_byte_p1(p_pos)) {
                return 0u;
            }
        }
    }

    // Compare 4 bytes at a time using u32 word loads (global memory, cold path)
    var match_len = 0u;
    let safe_limit = remaining & ~3u;
    loop {
        if (match_len >= safe_limit) {
            break;
        }
        let a = read_u32_at_p1(candidate + match_len);
        let b = read_u32_at_p1(pos + match_len);
        let diff = a ^ b;
        if (diff != 0u) {
            match_len = match_len + countTrailingZeros(diff) / 8u;
            return match_len;
        }
        match_len = match_len + 4u;
    }
    // Handle remaining 0-3 bytes
    loop {
        if (match_len >= remaining) {
            break;
        }
        if (read_byte_p1(candidate + match_len) != read_byte_p1(pos + match_len)) {
            break;
        }
        match_len = match_len + 1u;
    }
    return match_len;
}

// Try matching in the far window (global reads throughout).
fn try_match_far(candidate: u32, pos: u32, remaining: u32, best_len: u32) -> u32 {
    // Spot-check: skip if first byte doesn't match
    if (read_byte_p1(candidate) != read_byte_p1(pos)) {
        return 0u;
    }

    // Spot-check: skip if byte at current best length doesn't match
    if (best_len >= MIN_MATCH && best_len < remaining) {
        if (read_byte_p1(candidate + best_len) != read_byte_p1(pos + best_len)) {
            return 0u;
        }
    }

    // Compare 4 bytes at a time using u32 word loads
    var match_len = 0u;
    let safe_limit = remaining & ~3u;
    loop {
        if (match_len >= safe_limit) {
            break;
        }
        let a = read_u32_at_p1(candidate + match_len);
        let b = read_u32_at_p1(pos + match_len);
        let diff = a ^ b;
        if (diff != 0u) {
            match_len = match_len + countTrailingZeros(diff) / 8u;
            return match_len;
        }
        match_len = match_len + 4u;
    }
    // Handle remaining 0-3 bytes
    loop {
        if (match_len >= remaining) {
            break;
        }
        if (read_byte_p1(candidate + match_len) != read_byte_p1(pos + match_len)) {
            break;
        }
        match_len = match_len + 1u;
    }
    return match_len;
}

@compute @workgroup_size(64)
fn find_matches(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let pos = gid.x + gid.y * params_p1.w;
    let in_len = params_p1.x;
    let in_len_words = (in_len + 3u) / 4u;

    // --- Cooperative tile loading (ALL threads participate, even out-of-bounds) ---
    // The tile covers [tw_byte_start, tw_byte_end) in byte positions, spanning
    // the near-window lookback + workgroup + extension margin.
    let wg_base = pos - lid;
    var tw_byte_start: u32;
    if (wg_base >= NEAR_WINDOW) {
        tw_byte_start = (wg_base - NEAR_WINDOW) & ~3u;
    } else {
        tw_byte_start = 0u;
    }
    let tw_word_start = tw_byte_start / 4u;
    let tw_word_end = min(tw_word_start + TILE_WORDS, in_len_words + 1u);
    let tw_word_count = tw_word_end - tw_word_start;
    let tw_byte_end = tw_word_end * 4u;

    // Each thread loads ceil(tw_word_count / 64) words — coalesced, sequential
    for (var i = lid; i < tw_word_count; i = i + WG_SIZE) {
        tile[i] = input_p1[tw_word_start + i];
    }
    workgroupBarrier();

    // --- Per-thread matching (safe to early-return after barrier) ---
    if (pos >= in_len) {
        return;
    }

    let remaining = in_len - pos;
    var best: Lz77Match;
    best.offset = 0u;
    best.length = 0u;
    best.next = read_byte_p1(pos);

    if (remaining < MIN_MATCH || pos == 0u) {
        match_output[pos] = best;
        return;
    }

    // Tier 1: Full brute-force scan of the near window using tile-accelerated spot-checks
    let near_limit = min(NEAR_WINDOW, pos);
    for (var dist = 1u; dist <= near_limit; dist = dist + 1u) {
        let match_len = try_match_near(pos - dist, pos, remaining, best.length, tw_word_start, tw_byte_end);
        if (match_len > best.length && match_len >= MIN_MATCH) {
            best.offset = dist;
            best.length = match_len;
            if (best.length >= 128u) {
                break;
            }
        }
    }

    // Tier 2: Subsampled scan of the far window (every FAR_STEP positions).
    // Uses global reads since the working set is too large for shared memory.
    if (best.length < 128u && pos > NEAR_WINDOW) {
        let far_limit = min(FAR_WINDOW, pos);
        // Start from NEAR_WINDOW+1, rounded up to next FAR_STEP boundary
        let far_start = NEAR_WINDOW + FAR_STEP - (NEAR_WINDOW % FAR_STEP);
        for (var dist = far_start; dist <= far_limit; dist = dist + FAR_STEP) {
            let match_len = try_match_far(pos - dist, pos, remaining, best.length);
            if (match_len > best.length && match_len >= MIN_MATCH) {
                best.offset = dist;
                best.length = match_len;
                if (best.length >= 128u) {
                    break;
                }
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

    if (best.length < remaining) {
        best.next = read_byte_p1(pos + best.length);
    }

    match_output[pos] = best;
}

// ========================== Pass 2: resolve_lazy ==============================
// Reads raw per-position matches and applies lazy selection.
// Bindings: input(0), params(1), resolved(2), raw_matches(3)

@group(0) @binding(0) var<storage, read> input_p2: array<u32>;
@group(0) @binding(1) var<uniform> params_p2: vec4<u32>;
@group(0) @binding(2) var<storage, read_write> resolved: array<Lz77Match>;
@group(0) @binding(3) var<storage, read> raw_matches: array<Lz77Match>;

fn read_byte_p2(pos: u32) -> u32 {
    let word_idx = pos / 4u;
    let byte_idx = pos % 4u;
    return (input_p2[word_idx] >> (byte_idx * 8u)) & 0xFFu;
}

@compute @workgroup_size(64)
fn resolve_lazy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pos = gid.x + gid.y * params_p2.w;
    let in_len = params_p2.x;
    if (pos >= in_len) {
        return;
    }

    var m = raw_matches[pos];

    // Lazy evaluation: if pos+1 has a strictly longer match and our match
    // isn't already very long, demote this position to a literal.
    // This mirrors the CPU lazy matching heuristic in compress_lazy().
    if (m.length >= MIN_MATCH && m.length < LAZY_SKIP_THRESHOLD && pos + 1u < in_len) {
        let next_m = raw_matches[pos + 1u];
        if (next_m.length > m.length) {
            // Demote to literal — the next position's match is better
            m.offset = 0u;
            m.length = 0u;
            m.next = read_byte_p2(pos);
        }
    }

    // Positions with no match — ensure literal byte is correct
    if (m.length == 0u) {
        m.offset = 0u;
        m.next = read_byte_p2(pos);
    }

    resolved[pos] = m;
}
