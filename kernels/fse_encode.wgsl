// GPU FSE (tANS) interleaved encode kernel (WGSL).
//
// Each workgroup (size 1) encodes one lane of N-way interleaved FSE.
// All operations are 32-bit — no u64 required, making this WebGPU-compatible.
//
// Two-phase algorithm per lane:
//   1. Reverse pass: scan assigned symbols backward, look up encode table,
//      compute (value, bits) chunks, transition state.
//   2. Forward pass: iterate chunks forward, pack bits LSB-first into
//      the per-lane output bitstream.
//
// Encode table entries packed as u32 (indexed by symbol * table_size + state):
//   bits  0..11  = compressed_state (12 bits)
//   bits 12..15  = bits_to_output   (4 bits)
//   bits 16..31  = base             (16 bits)

// @pz_cost {
//   threads_per_element: 0.015625
//   passes: 1
//   buffers: symbols=N, encode_table=131072, output_data=N*2, lane_results=192
//   local_mem: 0
//   note: threads = num_lanes (typ. 4-32). encode_table = 256*table_size*4 (typ 128KB). output worst case ~1.5 bytes/sym.
// }

// Encode table: indexed by symbol * table_size + state.
@group(0) @binding(0) var<storage, read> encode_table: array<u32>;

// Input symbols (all lanes interleaved).
@group(0) @binding(1) var<storage, read> symbols: array<u32>;

// Output: per-lane bitstream data, packed as u32 words.
// Each lane writes to its own section starting at lane_id * max_output_words.
@group(0) @binding(2) var<storage, read_write> output_data: array<u32>;

// Per-lane results: [initial_state, total_bits, byte_len] per lane (3 u32 each).
@group(0) @binding(3) var<storage, read_write> lane_results: array<u32>;

// Params: [num_lanes, total_input_len, table_size, max_output_words_per_lane]
@group(0) @binding(4) var<uniform> params: vec4<u32>;

// Read a byte from the u32-packed symbols array at a byte position.
fn read_symbol(byte_pos: u32) -> u32 {
    let word_idx = byte_pos / 4u;
    let byte_in_word = byte_pos % 4u;
    return (symbols[word_idx] >> (byte_in_word * 8u)) & 0xFFu;
}

@compute @workgroup_size(1)
fn fse_encode(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lane_id = gid.x;
    let num_lanes = params.x;
    let total_input_len = params.y;
    let table_size = params.z;
    let max_output_words = params.w;

    if (lane_id >= num_lanes) {
        return;
    }

    // Count symbols for this lane.
    let num_symbols = (total_input_len - lane_id + num_lanes - 1u) / num_lanes;

    // Phase 1: Reverse pass — compute (value, bits) chunks.
    // Store chunks temporarily in our output section (will be overwritten
    // by phase 2 with the actual bitstream).
    let chunk_base = lane_id * max_output_words;
    var state: u32 = 0u;
    var cursor: u32 = num_symbols;

    // Process symbols in reverse within this lane.
    for (var sym_idx: i32 = i32(num_symbols) - 1; sym_idx >= 0; sym_idx = sym_idx - 1) {
        let global_idx: u32 = u32(sym_idx) * num_lanes + lane_id;
        if (global_idx >= total_input_len) {
            continue;
        }

        let s = read_symbol(global_idx);
        let entry = encode_table[s * table_size + state];

        let compressed_state = entry & 0xFFFu;
        let bits = (entry >> 12u) & 0xFu;
        let base = entry >> 16u;

        let value = state - base;

        cursor = cursor - 1u;
        output_data[chunk_base + cursor] = (value & 0xFFFFu) | (bits << 16u);

        state = compressed_state;
    }

    // Phase 2: Forward pass — pack bits LSB-first into output bytes.
    // We write bytes packed into u32 words in the same output section.
    var container: u32 = 0u;
    var bit_pos: u32 = 0u;
    var byte_count: u32 = 0u;

    for (var i: u32 = cursor; i < num_symbols; i = i + 1u) {
        let chunk = output_data[chunk_base + i];
        let value = chunk & 0xFFFFu;
        let nb_bits = chunk >> 16u;

        if (nb_bits > 0u) {
            container = container | (value << bit_pos);
            bit_pos = bit_pos + nb_bits;

            // Flush complete bytes — accumulate into u32 words.
            for (var f: u32 = 0u; f < 4u; f = f + 1u) {
                if (bit_pos < 8u) {
                    break;
                }
                let byte_val = container & 0xFFu;
                let word_idx = byte_count / 4u;
                let byte_in_word = byte_count % 4u;
                // Write byte into the output word.
                // Since only this lane writes to this section, no atomic needed.
                if (byte_in_word == 0u) {
                    output_data[chunk_base + word_idx] = byte_val;
                } else {
                    output_data[chunk_base + word_idx] = output_data[chunk_base + word_idx] | (byte_val << (byte_in_word * 8u));
                }
                container = container >> 8u;
                bit_pos = bit_pos - 8u;
                byte_count = byte_count + 1u;
            }
        }
    }

    // Flush remaining partial byte.
    let total_bits = byte_count * 8u + bit_pos;
    if (bit_pos > 0u) {
        let byte_val = container & 0xFFu;
        let word_idx = byte_count / 4u;
        let byte_in_word = byte_count % 4u;
        if (byte_in_word == 0u) {
            output_data[chunk_base + word_idx] = byte_val;
        } else {
            output_data[chunk_base + word_idx] = output_data[chunk_base + word_idx] | (byte_val << (byte_in_word * 8u));
        }
        byte_count = byte_count + 1u;
    }

    // Write per-lane results.
    lane_results[lane_id * 3u]      = state;       // initial_state for decoder
    lane_results[lane_id * 3u + 1u] = total_bits;
    lane_results[lane_id * 3u + 2u] = byte_count;  // bitstream byte length
}
