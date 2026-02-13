// LZ77 hash-table-based WGSL kernel for GPU-parallel match finding.
//
// Two-pass approach:
//   Pass 1 (build_hash_table): Each invocation hashes its 3-byte prefix and
//     atomically appends its position to a hash bucket (bounded ring buffer).
//   Pass 2 (find_matches): Each invocation looks up its hash bucket and only
//     compares against positions in that bucket (bounded by MAX_CHAIN).

// @pz_cost {
//   threads_per_element: 1
//   passes: 2
//   buffers: input=N, hash_counts=524288, hash_table=134217728, output=N*12
//   local_mem: 0
// }

struct Lz77Match {
    offset: u32,
    length: u32,
    next: u32,
}

const MAX_WINDOW: u32 = 32768u;
const HASH_BITS: u32 = 17u;
const HASH_SIZE: u32 = 131072u; // 1 << 17
const HASH_MASK: u32 = 131071u; // HASH_SIZE - 1
const MAX_CHAIN: u32 = 128u;
const BUCKET_CAP: u32 = 256u;
const MIN_MATCH: u32 = 3u;

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<uniform> params: vec4<u32>; // x = in_len
@group(0) @binding(2) var<storage, read_write> hash_counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> hash_table: array<u32>;

fn read_byte(pos: u32) -> u32 {
    let word_idx = pos / 4u;
    let byte_idx = pos % 4u;
    return (input[word_idx] >> (byte_idx * 8u)) & 0xFFu;
}

// Read a u32 starting at any byte offset (may span two u32 words).
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

fn hash3(pos: u32, len: u32) -> u32 {
    if (pos + 2u >= len) {
        return 0u;
    }
    let h = (read_byte(pos) << 12u) ^ (read_byte(pos + 1u) << 6u) ^ read_byte(pos + 2u);
    return h & HASH_MASK;
}

@compute @workgroup_size(64)
fn build_hash_table(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pos = gid.x + gid.y * params.w;
    let in_len = params.x;
    if (pos + 2u >= in_len) {
        return;
    }

    let h = hash3(pos, in_len);
    let slot = atomicAdd(&hash_counts[h], 1u);
    if (slot < BUCKET_CAP) {
        hash_table[h * BUCKET_CAP + slot] = pos;
    }
}

// Second pass uses different bindings
@group(0) @binding(4) var<storage, read_write> match_output: array<Lz77Match>;
@group(0) @binding(5) var<storage, read> hash_counts_ro: array<u32>;
@group(0) @binding(6) var<storage, read> hash_table_ro: array<u32>;

@compute @workgroup_size(64)
fn find_matches(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pos = gid.x + gid.y * params.w;
    let in_len = params.x;
    if (pos >= in_len) {
        return;
    }

    let remaining = in_len - pos;
    var best: Lz77Match;
    best.offset = 0u;
    best.length = 0u;
    best.next = read_byte(pos);

    if (remaining < MIN_MATCH || pos == 0u) {
        match_output[pos] = best;
        return;
    }

    let h = hash3(pos, in_len);
    var count = hash_counts_ro[h];
    if (count > BUCKET_CAP) {
        count = BUCKET_CAP;
    }

    var window_min = 0u;
    if (pos > MAX_WINDOW) {
        window_min = pos - MAX_WINDOW;
    }
    var checked = 0u;

    // Scan bucket entries in reverse order
    for (var idx = count; idx > 0u && checked < MAX_CHAIN; idx = idx - 1u) {
        let candidate = hash_table_ro[h * BUCKET_CAP + idx - 1u];

        if (candidate >= pos || candidate < window_min) {
            continue;
        }

        checked = checked + 1u;

        // Spot-check: skip candidate if it doesn't match at best_length position
        if (best.length >= MIN_MATCH && best.length < remaining) {
            if (read_byte(candidate + best.length) != read_byte(pos + best.length)) {
                continue;
            }
        }

        // Compare 4 bytes at a time using u32 word loads
        var max_len = remaining;
        let dist = pos - candidate;
        if (dist < max_len) {
            max_len = dist;
        }

        var match_len = 0u;
        // Compare u32 words (4 bytes at a time) while safe
        // We need at least 4 bytes remaining AND 4 bytes of padding in the
        // input buffer to safely read a u32 at any byte offset.
        let safe_limit = max_len & ~3u; // round down to u32 boundary
        loop {
            if (match_len >= safe_limit) {
                break;
            }
            let a = read_u32_at(candidate + match_len);
            let b = read_u32_at(pos + match_len);
            let diff = a ^ b;
            if (diff != 0u) {
                // Find which byte differs (count trailing zero bits / 8)
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
            if (read_byte(candidate + match_len) != read_byte(pos + match_len)) {
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
        best.next = read_byte(pos + best.length);
    }

    match_output[pos] = best;
}
