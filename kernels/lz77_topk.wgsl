// LZ77 top-K match finding kernel for optimal parsing.
//
// Each invocation finds the K best match candidates at one input position,
// searching backward through a 32KB sliding window. The host runs backward
// DP on the resulting match table to select the minimum-cost parse.
//
// Output: K candidates per position (flat array, position-major order).
// Candidates are sorted by length descending. Unused slots have length=0.

struct Lz77Candidate {
    offset: u32, // packed: u16 offset in low 16, u16 length in high 16
}

const MAX_WINDOW: u32 = 32768u;
const K: u32 = 4u;
const MIN_MATCH: u32 = 3u;

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>; // packed candidates
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = in_len

fn read_byte(pos: u32) -> u32 {
    let word_idx = pos / 4u;
    let byte_idx = pos % 4u;
    return (input[word_idx] >> (byte_idx * 8u)) & 0xFFu;
}

@compute @workgroup_size(64)
fn encode_topk(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pos = gid.x + gid.y * params.w;
    let in_len = params.x;
    if (pos >= in_len) {
        return;
    }

    // Top-K array stored as packed u32: (length << 16) | offset
    var top: array<u32, 4>;
    for (var i = 0u; i < K; i = i + 1u) {
        top[i] = 0u; // offset=0, length=0
    }

    var window_start = 0u;
    if (pos > MAX_WINDOW) {
        window_start = pos - MAX_WINDOW;
    }
    let remaining = in_len - pos;

    for (var j = window_start; j < pos; j = j + 1u) {
        // Spot-check optimization: skip if current worst top-K doesn't match
        let worst_packed = top[K - 1u];
        let worst_len = worst_packed >> 16u;
        if (worst_len >= MIN_MATCH) {
            if ((j + worst_len) < pos && worst_len < remaining) {
                if (read_byte(j + worst_len) != read_byte(pos + worst_len)) {
                    continue;
                }
            }
        }

        // Compare bytes to find match length
        var match_len = 0u;
        var max_len = min(remaining, pos - j);
        if (max_len > 65535u) {
            max_len = 65535u;
        }

        loop {
            if (match_len >= max_len) {
                break;
            }
            if (read_byte(j + match_len) != read_byte(pos + match_len)) {
                break;
            }
            match_len = match_len + 1u;
        }

        if (match_len < MIN_MATCH) {
            continue;
        }

        let offset = pos - j;
        let packed = (match_len << 16u) | offset;

        // Insert into top-K (sorted by length descending)
        var insert_at = -1;
        for (var i = 0u; i < K; i = i + 1u) {
            let top_len = top[i] >> 16u;
            let top_off = top[i] & 0xFFFFu;
            if (match_len > top_len || (match_len == top_len && offset < top_off)) {
                insert_at = i32(i);
                break;
            }
        }

        if (insert_at >= 0) {
            // Shift down
            for (var i = i32(K) - 1; i > insert_at; i = i - 1) {
                top[i] = top[i - 1];
            }
            top[insert_at] = packed;
        }
    }

    // Write K candidates to output (each as packed u32)
    let base_idx = pos * K;
    for (var i = 0u; i < K; i = i + 1u) {
        output[base_idx + i] = top[i];
    }
}
