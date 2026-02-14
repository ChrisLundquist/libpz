// GPU rANS decode kernel (WGSL).
//
// Each workgroup (size 1) decodes one lane of N-way interleaved rANS output.
// Uses wrapping u32 arithmetic for state transitions (no u64 needed —
// the low 32 bits of freq * (state >> scale_bits) are identical whether
// computed in u32 or u64, and the final state is always valid).
//
// The cum2sym lookup table is packed 4 bytes per u32 to fit within
// WGSL array size limits. freq and cum tables are packed as u32
// (low 16 = freq, high 16 = cum) to reduce buffer count.
//
// Each lane's symbols are written to round-robin positions:
//   output[sym_idx * num_lanes + lane_id]
//
// Per-lane metadata layout: [initial_state, num_words, word_offset]

// @pz_cost {
//   threads_per_element: 0.0625
//   passes: 1
//   buffers: tables=N, word_data=N, lane_meta=192, output=N
//   local_mem: 0
//   note: threads = num_lanes (typ. K=16), each decodes N/K symbols.
// }

const RANS_L: u32 = 65536u;
const IO_BITS: u32 = 16u;

// Combined tables buffer layout:
//   [0 .. table_size/4): cum2sym packed 4 bytes per u32
//   [table_size/4 .. table_size/4 + 256): freq_cum packed u32 (low16=freq, high16=cum)
@group(0) @binding(0) var<storage, read> tables: array<u32>;

// Word data: all lanes' u16 word streams concatenated, packed as u32 (2 words per u32)
@group(0) @binding(1) var<storage, read> word_data: array<u32>;

// Combined metadata: lane_meta entries (3 u32 per lane)
@group(0) @binding(2) var<storage, read> lane_meta: array<u32>;

// Output buffer: decoded bytes packed as u32
@group(0) @binding(3) var<storage, read_write> output: array<atomic<u32>>;

// Params: x=num_lanes, y=total_output_len, z=scale_bits, w=table_size
@group(0) @binding(4) var<uniform> params: vec4<u32>;

fn read_cum2sym(tables_offset: u32, slot: u32) -> u32 {
    let word_idx = tables_offset + slot / 4u;
    let byte_in_word = slot % 4u;
    return (tables[word_idx] >> (byte_in_word * 8u)) & 0xFFu;
}

fn read_freq(fc_offset: u32, sym: u32) -> u32 {
    return tables[fc_offset + sym] & 0xFFFFu;
}

fn read_cum(fc_offset: u32, sym: u32) -> u32 {
    return tables[fc_offset + sym] >> 16u;
}

fn read_word(word_index: u32) -> u32 {
    let u32_idx = word_index / 2u;
    let half = word_index % 2u;
    return (word_data[u32_idx] >> (half * 16u)) & 0xFFFFu;
}

fn write_output_byte(pos: u32, value: u32) {
    let word_idx = pos / 4u;
    let byte_in_word = pos % 4u;
    atomicOr(&output[word_idx], value << (byte_in_word * 8u));
}

@compute @workgroup_size(1)
fn rans_decode(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lane_id = gid.x;
    let num_lanes = params.x;
    let total_output_len = params.y;
    let scale_bits = params.z;
    let table_size = params.w;

    if (lane_id >= num_lanes) {
        return;
    }

    let scale_mask = table_size - 1u;
    let cum2sym_offset = 0u;
    let fc_offset = table_size / 4u; // freq_cum starts after cum2sym

    // Read per-lane metadata
    let meta_base = lane_id * 3u;
    var state = lane_meta[meta_base];
    let num_words = lane_meta[meta_base + 1u];
    let word_offset = lane_meta[meta_base + 2u];

    var word_pos = 0u;

    // Number of symbols this lane decodes (round-robin)
    let num_symbols = (total_output_len + num_lanes - 1u - lane_id) / num_lanes;

    for (var sym_idx = 0u; sym_idx < num_symbols; sym_idx = sym_idx + 1u) {
        let slot = state & scale_mask;
        let sym = read_cum2sym(cum2sym_offset, slot);
        let freq = read_freq(fc_offset, sym);
        let cum = read_cum(fc_offset, sym);

        // Write symbol to round-robin output position
        let out_pos = sym_idx * num_lanes + lane_id;
        if (out_pos < total_output_len) {
            write_output_byte(out_pos, sym);
        }

        // Advance state: wrapping u32 multiplication
        // state = freq * (state >> scale_bits) + slot - cum
        state = freq * (state >> scale_bits) + slot - cum;

        // Renormalize: read one 16-bit word if state dropped below RANS_L
        if (state < RANS_L && word_pos < num_words) {
            state = (state << IO_BITS) | read_word(word_offset + word_pos);
            word_pos = word_pos + 1u;
        }
    }
}
