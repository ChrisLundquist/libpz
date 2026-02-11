// LZ77 batched WGSL kernel for GPU-parallel match finding.
//
// Each invocation processes STEP_SIZE consecutive positions, reducing
// dispatch overhead. Uses a smaller window (32KB) for faster
// per-position search. The host deduplicates overlapping matches
// after readback.

struct Lz77Match {
    offset: u32,
    length: u32,
    next: u32,
}

const MAX_WINDOW: u32 = 32768u;
const STEP_SIZE: u32 = 32u;

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<Lz77Match>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = in_len

fn read_byte(pos: u32) -> u32 {
    let word_idx = pos / 4u;
    let byte_idx = pos % 4u;
    return (input[word_idx] >> (byte_idx * 8u)) & 0xFFu;
}

fn find_match_classic(search_start: u32, search_size: u32, target: u32, target_size: u32) -> Lz77Match {
    var best: Lz77Match;
    best.offset = 0u;
    best.length = 0u;
    best.next = read_byte(target);

    for (var i = 0u; i < search_size; i = i + 1u) {
        if (best.length > 0u && (i + best.length) < search_size && best.length < target_size) {
            if (read_byte(search_start + i + best.length) != read_byte(target + best.length)) {
                continue;
            }
        }

        var temp_match_length = 0u;
        var tail = i + temp_match_length;
        loop {
            if (tail >= search_size || temp_match_length >= target_size) {
                break;
            }
            if (read_byte(search_start + tail) != read_byte(target + temp_match_length)) {
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
        if (best.length < target_size) {
            break;
        }
        best.length = best.length - 1u;
    }
    best.next = read_byte(target + best.length);
    return best;
}

@compute @workgroup_size(64)
fn encode(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base = gid.x * STEP_SIZE;
    let in_len = params.x;
    if (base >= in_len) {
        return;
    }

    var last_step = 0u;
    var last_match: Lz77Match;
    last_match.offset = 0u;
    last_match.length = 0u;
    last_match.next = 0u;

    var step = 0u;
    loop {
        if (step >= STEP_SIZE || (base + step) >= in_len) {
            break;
        }
        let i_step = base + step;
        var window_start = 0u;
        if (i_step > MAX_WINDOW) {
            window_start = i_step - MAX_WINDOW;
        }
        let search_size = min(i_step, MAX_WINDOW);

        let m = find_match_classic(window_start, search_size, i_step, in_len - i_step);
        output[i_step] = m;
        last_step = step;
        last_match = m;
        step = step + m.length + 1u;
    }

    // Handle boundary condition: truncate the last match so it doesn't
    // overlap into the next work-item's chunk.
    let end_pos = last_step + last_match.length + 1u;
    if (end_pos > STEP_SIZE) {
        let overlap_bytes = end_pos - STEP_SIZE;
        last_match.length = last_match.length - overlap_bytes;
        last_match.next = read_byte(min(in_len, base + STEP_SIZE) - 1u);
        output[base + last_step] = last_match;
    }
}
