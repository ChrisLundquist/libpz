// GPU rANS decode kernel (WGSL).
//
// Each dispatch decodes one chunk with N interleaved lanes.

const RANS_L: u32 = 65536u;
const IO_BITS: u32 = 16u;

// Per-chunk metadata:
// [output_offset, output_len, word_stream_offset, chunk_id]
@group(0) @binding(0) var<uniform> chunk_meta: vec4<u32>;

// Input word streams (u16 per lane)
@group(0) @binding(1) var<storage, read> input_words: array<u32>;

// Input initial states (u32 per lane)
@group(0) @binding(2) var<storage, read> initial_states: array<u32>;

// Output buffer (decoded bytes)
@group(0) @binding(3) var<storage, read_write> output_bytes: array<atomic<u32>>;

// rANS tables:
// [0..255]   freq table
// [256..511] cum table
// [512..]    cum2sym lookup table (packed bytes, table_size bytes)
@group(0) @binding(4) var<storage, read> tables: array<u32>;

// Params: [num_lanes, scale_bits, table_size, unused]
@group(0) @binding(5) var<uniform> params: vec4<u32>;

fn read_cum2sym(offset_words: u32, slot: u32) -> u32 {
    let word_idx = offset_words + (slot / 4u);
    let byte_in_word = slot % 4u;
    return (tables[word_idx] >> (byte_in_word * 8u)) & 0xFFu;
}

fn read_freq(sym: u32) -> u32 {
    return tables[sym];
}

fn read_cum(sym: u32) -> u32 {
    return tables[256u + sym];
}

fn read_word(word_index: u32) -> u32 {
    let u32_idx = word_index / 2u;
    let half = word_index % 2u;
    return (input_words[u32_idx] >> (half * 16u)) & 0xFFFFu;
}

fn write_output_byte(pos: u32, value: u32) {
    let word_idx = pos / 4u;
    let byte_in_word = pos % 4u;
    atomicOr(&output_bytes[word_idx], value << (byte_in_word * 8u));
}

@compute @workgroup_size(4) // num_lanes must match this
fn rans_decode_chunk(
    @builtin(local_invocation_index) lid: u32
) {
    let num_lanes = params.x;
    let scale_bits = params.y;
    let table_size = params.z;

    let lane_id = lid;
    if (lane_id >= num_lanes) {
        return;
    }

    let chunk_id = chunk_meta.w;
    let chunk_output_offset = chunk_meta.x;
    let chunk_output_len = chunk_meta.y;
    let chunk_words_offset = chunk_meta.z;

    let scale_mask = table_size - 1u;
    let cum2sym_offset_words = 512u;

    // Read per-lane metadata
    let chunk_base = chunk_id * (num_lanes * 2u);
    let state_idx = chunk_base + lane_id;
    var state = initial_states[state_idx];
    let num_words = initial_states[chunk_base + num_lanes + lane_id];
    let symbols_per_lane_max = (chunk_output_len + num_lanes - 1u) / num_lanes;
    let max_words_per_lane = symbols_per_lane_max * 2u + 4u;
    let lane_word_base = chunk_words_offset + lane_id * max_words_per_lane;
    let word_offset = lane_word_base + (max_words_per_lane - num_words);

    var word_pos = 0u;

    // Number of symbols this lane decodes
    let num_symbols = (chunk_output_len + num_lanes - 1u - lane_id) / num_lanes;

    for (var sym_idx = 0u; sym_idx < num_symbols; sym_idx = sym_idx + 1u) {
        let slot = state & scale_mask;
        let sym = read_cum2sym(cum2sym_offset_words, slot);
        let freq = read_freq(sym);
        let cum = read_cum(sym);

        // Write symbol to round-robin output position
        let out_pos = chunk_output_offset + sym_idx * num_lanes + lane_id;
        if (out_pos < chunk_output_offset + chunk_output_len) {
            write_output_byte(out_pos, sym);
        }

        // Advance state
        state = freq * (state >> scale_bits) + slot - cum;

        // Renormalize
        if (state < RANS_L && word_pos < num_words) {
            state = (state << IO_BITS) | read_word(word_offset + word_pos);
            word_pos = word_pos + 1u;
        }
    }
}
