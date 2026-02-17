// GPU chunked rANS decode kernel (WGSL).
//
// Each workgroup decodes one chunk; lanes are mapped to local invocations.
// Many chunks can run concurrently in one dispatch.

const RANS_L: u32 = 65536u;
const IO_BITS: u32 = 16u;

// Per-chunk metadata:
// [output_offset, output_len, word_stream_offset, state_offset]
@group(0) @binding(0) var<storage, read> chunk_meta: array<vec4<u32>>;

// Input word streams (u16 packed in u32)
@group(0) @binding(1) var<storage, read> input_words: array<u32>;

// Input states + per-lane word counts (u32 each)
@group(0) @binding(2) var<storage, read> input_states: array<u32>;

// Output buffer (decoded bytes, packed into u32 atomics)
@group(0) @binding(3) var<storage, read_write> output_data: array<atomic<u32>>;

// Tables:
// [0..255]   freq
// [256..511] cum
// [512..]    cum2sym packed 4 symbols per u32
@group(0) @binding(4) var<storage, read> tables: array<u32>;

// Params: [num_chunks, num_lanes, scale_bits, chunk_size]
@group(0) @binding(5) var<uniform> params: vec4<u32>;

fn read_cum2sym(slot: u32) -> u32 {
    let word_idx = 512u + (slot / 4u);
    let byte_in_word = slot % 4u;
    return (tables[word_idx] >> (byte_in_word * 8u)) & 0xFFu;
}

fn read_freq(sym: u32) -> u32 {
    return tables[sym];
}

fn read_cum(sym: u32) -> u32 {
    return tables[256u + sym];
}

fn read_word(word_index_u16: u32) -> u32 {
    let u32_idx = word_index_u16 / 2u;
    let half = word_index_u16 % 2u;
    return (input_words[u32_idx] >> (half * 16u)) & 0xFFFFu;
}

fn write_output_byte(byte_pos: u32, value: u32) {
    let word_idx = byte_pos / 4u;
    let byte_in_word = byte_pos % 4u;
    atomicOr(&output_data[word_idx], value << (byte_in_word * 8u));
}

fn rans_decode_chunk_impl(
    chunk_id: u32,
    lane_id: u32,
    num_chunks: u32,
    num_lanes: u32,
    scale_bits: u32
) {
    if (chunk_id >= num_chunks || lane_id >= num_lanes) {
        return;
    }

    let chunk_info = chunk_meta[chunk_id];
    let chunk_output_offset = chunk_info.x;
    let chunk_output_len = chunk_info.y;
    let chunk_words_offset = chunk_info.z;
    let state_offset = chunk_info.w;

    let scale_mask = (1u << scale_bits) - 1u;

    var state = input_states[state_offset + lane_id];
    let num_words = input_states[state_offset + num_lanes + lane_id];

    let symbols_per_lane_max = (chunk_output_len + num_lanes - 1u) / num_lanes;
    let max_words_per_lane = symbols_per_lane_max * 2u + 4u;
    let lane_word_base = chunk_words_offset + lane_id * max_words_per_lane;
    let word_offset = lane_word_base + (max_words_per_lane - num_words);
    var word_pos = 0u;

    let num_symbols = (chunk_output_len + num_lanes - 1u - lane_id) / num_lanes;

    for (var sym_idx = 0u; sym_idx < num_symbols; sym_idx = sym_idx + 1u) {
        let slot = state & scale_mask;
        let sym = read_cum2sym(slot);
        let freq = read_freq(sym);
        let cum = read_cum(sym);

        let out_pos = chunk_output_offset + sym_idx * num_lanes + lane_id;
        if (out_pos < chunk_output_offset + chunk_output_len) {
            write_output_byte(out_pos, sym);
        }

        state = freq * (state >> scale_bits) + slot - cum;

        if (state < RANS_L && word_pos < num_words) {
            state = (state << IO_BITS) | read_word(word_offset + word_pos);
            word_pos = word_pos + 1u;
        }
    }
}

@compute @workgroup_size(4)
fn rans_decode_chunk_wg4(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwork: vec3<u32>
) {
    let chunk_id = wid.x + wid.y * nwork.x;
    rans_decode_chunk_impl(chunk_id, lid.x, params.x, params.y, params.z);
}

@compute @workgroup_size(8)
fn rans_decode_chunk_wg8(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwork: vec3<u32>
) {
    let chunk_id = wid.x + wid.y * nwork.x;
    rans_decode_chunk_impl(chunk_id, lid.x, params.x, params.y, params.z);
}

// Wide fallback for high lane-count experiments.
@compute @workgroup_size(64)
fn rans_decode_chunk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwork: vec3<u32>
) {
    let chunk_id = wid.x + wid.y * nwork.x;
    rans_decode_chunk_impl(chunk_id, lid.x, params.x, params.y, params.z);
}
