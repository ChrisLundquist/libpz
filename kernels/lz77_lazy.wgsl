// LZ77 GPU-parallel match finding with lazy matching emulation.
//
// Two-pass approach:
//   Pass 1 (find_matches): Each invocation scans backward through a near
//     window to find the best match at its position. Uses spot-check
//     optimization and u32-wide comparison for speed.
//   Pass 2 (resolve_lazy): Each invocation reads match[pos] and match[pos+1].
//     If pos+1 has a strictly longer match AND pos's match isn't too long to
//     bother checking, pos becomes a literal. This emulates gzip-style lazy
//     matching on the GPU in a single parallel pass.
//
// The near brute-force scan avoids the hash table entirely, eliminating
// bucket overflow issues that caused catastrophic quality loss on repetitive
// data. Spot-check pre-filtering keeps the scan fast: ~99.6% of candidates
// are eliminated by a single byte comparison.

// @pz_cost {
//   threads_per_element: 1
//   passes: 2
//   buffers: input=N, raw_matches=N*12, resolved=N*12, staging=N*12
//   local_mem: 0
// }

struct Lz77Match {
    offset: u32,
    length: u32,
    next: u32,
}

const NEAR_WINDOW: u32 = 1024u;
const MIN_MATCH: u32 = 3u;
// Matches this long skip lazy evaluation (unlikely to be beaten).
const LAZY_SKIP_THRESHOLD: u32 = 32u;

// ========================== Pass 1: find_matches ==============================
// Bindings: input(0), params(1), match_output(2)

@group(0) @binding(0) var<storage, read> input_p1: array<u32>;
@group(0) @binding(1) var<uniform> params_p1: vec4<u32>; // x = in_len, w = dispatch_width
@group(0) @binding(2) var<storage, read_write> match_output: array<Lz77Match>;

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

@compute @workgroup_size(64)
fn find_matches(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pos = gid.x + gid.y * params_p1.w;
    let in_len = params_p1.x;
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

    let scan_limit = min(NEAR_WINDOW, pos);

    // Scan backward through the near window (most recent positions first)
    for (var dist = 1u; dist <= scan_limit; dist = dist + 1u) {
        let candidate = pos - dist;

        // Spot-check: skip if first byte doesn't match
        if (read_byte_p1(candidate) != read_byte_p1(pos)) {
            continue;
        }

        // Spot-check: skip if byte at current best length doesn't match
        if (best.length >= MIN_MATCH && best.length < remaining) {
            if (read_byte_p1(candidate + best.length) != read_byte_p1(pos + best.length)) {
                continue;
            }
        }

        // Compare 4 bytes at a time using u32 word loads
        var max_len = remaining;
        // Allow overlapping matches (length > offset) for run compression
        // but cap at the safe limit for u32 reads
        var match_len = 0u;
        let safe_limit = max_len & ~3u;
        loop {
            if (match_len >= safe_limit) {
                break;
            }
            let a = read_u32_at_p1(candidate + match_len);
            let b = read_u32_at_p1(pos + match_len);
            let diff = a ^ b;
            if (diff != 0u) {
                match_len = match_len + countTrailingZeros(diff) / 8u;
                break;
            }
            match_len = match_len + 4u;
        }
        // Handle remaining 0-3 bytes
        loop {
            if (match_len >= max_len) {
                break;
            }
            if (read_byte_p1(candidate + match_len) != read_byte_p1(pos + match_len)) {
                break;
            }
            match_len = match_len + 1u;
        }

        if (match_len > best.length && match_len >= MIN_MATCH) {
            best.offset = dist;
            best.length = match_len;
            // Early exit: stop searching if we found a long-enough match
            if (best.length >= 128u) {
                break;
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
