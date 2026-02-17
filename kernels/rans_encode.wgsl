// GPU chunked rANS encode kernel (WGSL).
//
// Each workgroup encodes one chunk; lanes are mapped to local invocations.
// Many chunks can run concurrently in one dispatch.

const RANS_L: u32 = 65536u;
const IO_BITS: u32 = 16u;

// Chunk metadata array (one item per chunk):
// [input_offset_bytes, input_len_bytes, output_word_offset_u16, state_offset_u32]
@group(0) @binding(0) var<storage, read> chunk_meta: array<vec4<u32>>;

// Input symbols, byte-packed in u32 words.
@group(0) @binding(1) var<storage, read> input_data: array<u32>;

// Output lane word streams (u16 packed into u32).
@group(0) @binding(2) var<storage, read_write> output_words: array<u32>;

// Output per-lane final states and metadata.
@group(0) @binding(3) var<storage, read_write> output_states: array<u32>;

// Shared tables:
// [0..255]   freq
// [256..511] cum
// [512..]    cum2sym lookup (decode path)
@group(0) @binding(4) var<storage, read> tables: array<u32>;

// Params: [num_chunks, num_lanes, scale_bits, chunk_size_bytes]
@group(0) @binding(5) var<uniform> params: vec4<u32>;

fn read_input_byte(byte_pos: u32) -> u32 {
    let word_idx = byte_pos / 4u;
    let byte_in_word = byte_pos % 4u;
    return (input_data[word_idx] >> (byte_in_word * 8u)) & 0xFFu;
}

fn write_word(word_index_u16: u32, value_u16: u32) {
    let word_idx = word_index_u16 / 2u;
    let half = word_index_u16 % 2u;
    if (half == 0u) {
        output_words[word_idx] = (output_words[word_idx] & 0xFFFF0000u) | (value_u16 & 0xFFFFu);
    } else {
        output_words[word_idx] =
            (output_words[word_idx] & 0x0000FFFFu) | ((value_u16 & 0xFFFFu) << 16u);
    }
}

fn rans_encode_chunk_impl(
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
    let chunk_input_offset = chunk_info.x;
    let chunk_input_len = chunk_info.y;
    let chunk_words_offset = chunk_info.z;
    let state_offset = chunk_info.w;

    let symbols_per_lane_max = (chunk_input_len + num_lanes - 1u) / num_lanes;
    let max_words_per_lane = symbols_per_lane_max * 2u + 4u;
    let lane_word_base = chunk_words_offset + lane_id * max_words_per_lane;
    var cursor = lane_word_base + max_words_per_lane;

    var state = RANS_L;
    let base = (RANS_L >> scale_bits) << IO_BITS;

    for (var i = chunk_input_len; i > 0u; i = i - 1u) {
        let sym_idx = i - 1u;
        if (sym_idx % num_lanes != lane_id) {
            continue;
        }

        let s = read_input_byte(chunk_input_offset + sym_idx);
        let freq = tables[s];
        let cum = tables[256u + s];

        if (freq <= (0xFFFFFFFFu / base)) {
            let x_max = base * freq;
            while (state >= x_max) {
                cursor = cursor - 1u;
                write_word(cursor, state & 0xFFFFu);
                state = state >> IO_BITS;
            }
        }

        var q = 0u;
        var r = 0u;
        if (freq == 1u) {
            q = state;
        } else {
            q = state / freq;
            r = state - q * freq;
        }
        state = (q << scale_bits) + r + cum;
    }

    output_states[state_offset + lane_id] = state;
    output_states[state_offset + num_lanes + lane_id] =
        (lane_word_base + max_words_per_lane) - cursor;
}

@compute @workgroup_size(4)
fn rans_encode_chunk_wg4(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwork: vec3<u32>
) {
    let chunk_id = wid.x + wid.y * nwork.x;
    rans_encode_chunk_impl(chunk_id, lid.x, params.x, params.y, params.z);
}

@compute @workgroup_size(8)
fn rans_encode_chunk_wg8(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwork: vec3<u32>
) {
    let chunk_id = wid.x + wid.y * nwork.x;
    rans_encode_chunk_impl(chunk_id, lid.x, params.x, params.y, params.z);
}

// Wide fallback for high lane-count experiments.
@compute @workgroup_size(64)
fn rans_encode_chunk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwork: vec3<u32>
) {
    let chunk_id = wid.x + wid.y * nwork.x;
    rans_encode_chunk_impl(chunk_id, lid.x, params.x, params.y, params.z);
}
