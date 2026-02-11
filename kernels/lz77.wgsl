// LZ77 WGSL kernel for GPU-parallel match finding.
//
// Each invocation finds the best match at one input position,
// searching backward through a sliding window of MAX_WINDOW bytes.
// The host deduplicates overlapping matches after readback.

struct Lz77Match {
    offset: u32,
    length: u32,
    next: u32, // packed: next byte in low 8 bits, padding in upper 24
}

const MAX_WINDOW: u32 = 131072u;

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<Lz77Match>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = count

fn read_byte(pos: u32) -> u32 {
    let word_idx = pos / 4u;
    let byte_idx = pos % 4u;
    return (input[word_idx] >> (byte_idx * 8u)) & 0xFFu;
}

fn find_match_classic(search_start: u32, search_size: u32, tgt: u32, tgt_size: u32) -> Lz77Match {
    var best: Lz77Match;
    best.offset = 0u;
    best.length = 0u;
    best.next = read_byte(tgt);

    for (var i = 0u; i < search_size; i = i + 1u) {
        // Spot-check optimization
        if (best.length > 0u && (i + best.length) < search_size && best.length < tgt_size) {
            if (read_byte(search_start + i + best.length) != read_byte(tgt + best.length)) {
                continue;
            }
        }

        var temp_match_length = 0u;
        var tail = i + temp_match_length;
        loop {
            if (tail >= search_size || temp_match_length >= tgt_size) {
                break;
            }
            if (read_byte(search_start + tail) != read_byte(tgt + temp_match_length)) {
                break;
            }
            temp_match_length = temp_match_length + 1u;
            tail = tail + 1u;
        }
        if (temp_match_length > best.length) {
            best.offset = search_size - i;
            best.length = temp_match_length;
        }
    }
    loop {
        if (best.length < tgt_size) {
            break;
        }
        best.length = best.length - 1u;
    }
    best.next = read_byte(tgt + best.length);
    return best;
}

@compute @workgroup_size(64)
fn encode(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let count = params.x;
    if (i >= count) {
        return;
    }

    var window_start = 0u;
    if (i > MAX_WINDOW) {
        window_start = i - MAX_WINDOW;
    }
    let search_size = min(i, MAX_WINDOW);

    output[i] = find_match_classic(window_start, search_size, i, count - i);
}
