// GPU rANS encode kernel (WGSL).
//
// Each dispatch encodes one chunk with N interleaved lanes.

const RANS_L: u32 = 65536u;
const IO_BITS: u32 = 16u;

// Per-chunk metadata:
// [input_offset, input_len, output_words_offset, chunk_id]
@group(0) @binding(0) var<uniform> chunk_meta: vec4<u32>;

// Input data (byte-packed)
@group(0) @binding(1) var<storage, read> input_data: array<u32>;

// Output words (u16 per lane)
@group(0) @binding(2) var<storage, read_write> output_words: array<u32>;

// Output final states (u32 per lane)
@group(0) @binding(3) var<storage, read_write> output_states: array<u32>;

// rANS tables:
// [0..255]   freq table
// [256..511] cum table
// [512..]    cum2sym (decode-only, unused by this kernel)
@group(0) @binding(4) var<storage, read> tables: array<u32>;

// Params: [num_lanes, scale_bits, table_size, unused]
@group(0) @binding(5) var<uniform> params: vec4<u32>;

// Read a byte from the u32-packed input array at a byte position.
fn read_input_byte(byte_pos: u32) -> u32 {
    let word_idx = byte_pos / 4u;
    let byte_in_word = byte_pos % 4u;
    return (input_data[word_idx] >> (byte_in_word * 8u)) & 0xFFu;
}

@compute @workgroup_size(4) // num_lanes should match this
fn rans_encode_chunk(
    @builtin(local_invocation_index) lid: u32
) {
    let num_lanes = params.x;
    let scale_bits = params.y;

    let lane_id = lid;
    if (lane_id >= num_lanes) {
        return;
    }

    let chunk_input_offset = chunk_meta.x;
    let chunk_input_len = chunk_meta.y;
    let chunk_output_words_offset = chunk_meta.z;
    let chunk_id = chunk_meta.w;

    var state = RANS_L;

    // Worst-case lane word budget (u16 words); +4 keeps a safety margin.
    let symbols_per_lane_max = (chunk_input_len + num_lanes - 1u) / num_lanes;
    let max_words_per_lane = symbols_per_lane_max * 2u + 4u;
    let lane_output_start = chunk_output_words_offset + lane_id * max_words_per_lane;
    var cursor = lane_output_start + max_words_per_lane;
    let base = (RANS_L >> scale_bits) << IO_BITS;

    // Process symbols in reverse for this lane.
    for (var i = chunk_input_len; i > 0u; i = i - 1u) {
        let sym_idx = i - 1u;
        if (sym_idx % num_lanes != lane_id) {
            continue;
        }

        let s = read_input_byte(chunk_input_offset + sym_idx);

        let freq = tables[s];
        let cum = tables[256u + s];

        // Renormalize
        var x_max = 0xFFFFFFFFu;
        if (freq <= (0xFFFFFFFFu / base)) {
            x_max = base * freq;
        }
        while (state >= x_max) {
            cursor = cursor - 1u;
            // pack two u16 words into one u32
            let word_index = cursor / 2u;
            let word_half = cursor % 2u;
            if (word_half == 0u) {
                output_words[word_index] = (output_words[word_index] & 0xFFFF0000u) | (state & 0xFFFFu);
            } else {
                output_words[word_index] = (output_words[word_index] & 0x0000FFFFu) | ((state & 0xFFFFu) << 16u);
            }
            state = state >> IO_BITS;
        }

        // Encode
        let q = state / freq;
        let r = state - q * freq;
        state = (q << scale_bits) + r + cum;
    }

    // Store final state and per-lane word count. Layout:
    // [chunk0 states N][chunk0 word_counts N][chunk1 states N][chunk1 word_counts N]...
    let chunk_base = chunk_id * (num_lanes * 2u);
    let final_state_idx = chunk_base + lane_id;
    output_states[final_state_idx] = state;

    let words_written = (lane_output_start + max_words_per_lane) - cursor;
    let num_words_idx = chunk_base + num_lanes + lane_id;
    output_states[num_words_idx] = words_written;
}
