// LZ77 GPU-parallel match finding with lazy matching emulation.
//
// Three-pass approach:
//   Pass 1 (build_hash_table): Each invocation hashes its 3-byte prefix and
//     atomically appends its position to a hash bucket (bounded ring buffer).
//     Identical to lz77_hash.wgsl pass 1.
//   Pass 2 (find_matches): Each invocation finds the best match at its position
//     using the hash table. Supports overlapping matches (length > offset) for
//     efficient run compression. Identical to lz77_hash.wgsl pass 2.
//   Pass 3 (resolve_lazy): Each invocation reads match[pos] and match[pos+1].
//     If pos+1 has a strictly longer match AND pos's match isn't too long to
//     bother checking, pos becomes a literal. This emulates gzip-style lazy
//     matching on the GPU in a single parallel pass.
//
// The lazy resolution produces better compression than pure greedy per-position
// matching, approaching CPU lazy matching quality while retaining GPU parallelism.

struct Lz77Match {
    offset: u32,
    length: u32,
    next: u32,
}

const MAX_WINDOW: u32 = 32768u;
const HASH_SIZE: u32 = 32768u; // 1 << 15
const HASH_MASK: u32 = 32767u; // HASH_SIZE - 1
const MAX_CHAIN: u32 = 64u;
const BUCKET_CAP: u32 = 64u;
const MIN_MATCH: u32 = 3u;
// Matches this long skip lazy evaluation (unlikely to be beaten).
const LAZY_SKIP_THRESHOLD: u32 = 32u;

// ========================== Pass 1: build_hash_table ==========================
// Bindings: input(0), params(1), hash_counts(2), hash_table(3)

@group(0) @binding(0) var<storage, read> input_p1: array<u32>;
@group(0) @binding(1) var<uniform> params_p1: vec4<u32>; // x = in_len, w = dispatch_width
@group(0) @binding(2) var<storage, read_write> hash_counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> hash_table: array<u32>;

fn read_byte_p1(pos: u32) -> u32 {
    let word_idx = pos / 4u;
    let byte_idx = pos % 4u;
    return (input_p1[word_idx] >> (byte_idx * 8u)) & 0xFFu;
}

fn hash3_p1(pos: u32, len: u32) -> u32 {
    if (pos + 2u >= len) {
        return 0u;
    }
    let h = (read_byte_p1(pos) << 10u) ^ (read_byte_p1(pos + 1u) << 5u) ^ read_byte_p1(pos + 2u);
    return h & HASH_MASK;
}

@compute @workgroup_size(64)
fn build_hash_table(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pos = gid.x + gid.y * params_p1.w;
    let in_len = params_p1.x;
    if (pos + 2u >= in_len) {
        return;
    }

    let h = hash3_p1(pos, in_len);
    let slot = atomicAdd(&hash_counts[h], 1u);
    if (slot < BUCKET_CAP) {
        hash_table[h * BUCKET_CAP + slot] = pos;
    }
}

// ========================== Pass 2: find_matches ==============================
// Bindings: input(0), params(1), match_output(2), hash_counts_ro(3), hash_table_ro(4)

@group(0) @binding(0) var<storage, read> input_p2: array<u32>;
@group(0) @binding(1) var<uniform> params_p2: vec4<u32>;
@group(0) @binding(2) var<storage, read_write> match_output: array<Lz77Match>;
@group(0) @binding(3) var<storage, read> hash_counts_ro: array<u32>;
@group(0) @binding(4) var<storage, read> hash_table_ro: array<u32>;

fn read_byte_p2(pos: u32) -> u32 {
    let word_idx = pos / 4u;
    let byte_idx = pos % 4u;
    return (input_p2[word_idx] >> (byte_idx * 8u)) & 0xFFu;
}

fn read_u32_at_p2(pos: u32) -> u32 {
    let word_idx = pos / 4u;
    let shift = (pos % 4u) * 8u;
    if (shift == 0u) {
        return input_p2[word_idx];
    }
    let lo = input_p2[word_idx] >> shift;
    let hi = input_p2[word_idx + 1u] << (32u - shift);
    return lo | hi;
}

fn hash3_p2(pos: u32, len: u32) -> u32 {
    if (pos + 2u >= len) {
        return 0u;
    }
    let h = (read_byte_p2(pos) << 10u) ^ (read_byte_p2(pos + 1u) << 5u) ^ read_byte_p2(pos + 2u);
    return h & HASH_MASK;
}

@compute @workgroup_size(64)
fn find_matches(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pos = gid.x + gid.y * params_p2.w;
    let in_len = params_p2.x;
    if (pos >= in_len) {
        return;
    }

    let remaining = in_len - pos;
    var best: Lz77Match;
    best.offset = 0u;
    best.length = 0u;
    best.next = read_byte_p2(pos);

    if (remaining < MIN_MATCH || pos == 0u) {
        match_output[pos] = best;
        return;
    }

    let h = hash3_p2(pos, in_len);
    var count = hash_counts_ro[h];
    if (count > BUCKET_CAP) {
        count = BUCKET_CAP;
    }

    var window_min = 0u;
    if (pos > MAX_WINDOW) {
        window_min = pos - MAX_WINDOW;
    }
    var checked = 0u;

    // Scan bucket entries in reverse order (most recent first)
    for (var idx = count; idx > 0u && checked < MAX_CHAIN; idx = idx - 1u) {
        let candidate = hash_table_ro[h * BUCKET_CAP + idx - 1u];

        if (candidate >= pos || candidate < window_min) {
            continue;
        }

        checked = checked + 1u;

        // Spot-check: skip candidate if it doesn't match at best_length position
        if (best.length >= MIN_MATCH && best.length < remaining) {
            if (read_byte_p2(candidate + best.length) != read_byte_p2(pos + best.length)) {
                continue;
            }
        }

        // Compare 4 bytes at a time using u32 word loads
        var max_len = remaining;
        var match_len = 0u;
        let safe_limit = max_len & ~3u;
        loop {
            if (match_len >= safe_limit) {
                break;
            }
            let a = read_u32_at_p2(candidate + match_len);
            let b = read_u32_at_p2(pos + match_len);
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
            if (read_byte_p2(candidate + match_len) != read_byte_p2(pos + match_len)) {
                break;
            }
            match_len = match_len + 1u;
        }

        if (match_len > best.length && match_len >= MIN_MATCH) {
            best.offset = pos - candidate;
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
        best.next = read_byte_p2(pos + best.length);
    }

    match_output[pos] = best;
}

// ========================== Pass 3: resolve_lazy ==============================
// Reads raw per-position matches and applies lazy selection.
// Bindings: input(0), params(1), resolved(2), raw_matches(3)

@group(0) @binding(0) var<storage, read> input_p3: array<u32>;
@group(0) @binding(1) var<uniform> params_p3: vec4<u32>;
@group(0) @binding(2) var<storage, read_write> resolved: array<Lz77Match>;
@group(0) @binding(3) var<storage, read> raw_matches: array<Lz77Match>;

fn read_byte_p3(pos: u32) -> u32 {
    let word_idx = pos / 4u;
    let byte_idx = pos % 4u;
    return (input_p3[word_idx] >> (byte_idx * 8u)) & 0xFFu;
}

@compute @workgroup_size(64)
fn resolve_lazy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pos = gid.x + gid.y * params_p3.w;
    let in_len = params_p3.x;
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
            m.next = read_byte_p3(pos);
        }
    }

    // Positions with no match — ensure literal byte is correct
    if (m.length == 0u) {
        m.offset = 0u;
        m.next = read_byte_p3(pos);
    }

    resolved[pos] = m;
}
