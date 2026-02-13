// GPU FSE (tANS) interleaved encode kernel (OpenCL).
//
// Each work-item encodes one lane of N-way interleaved FSE.
// The encode is two-phase per lane:
//   1. Reverse pass: scan assigned symbols backward, look up the encode
//      table to get (compressed_state, bits, base), compute output value,
//      store (value, bits) chunk, transition state.
//   2. Forward pass: iterate chunks forward, pack bits LSB-first into
//      the per-lane output bitstream section.
//
// Encode table entries are packed as uint (indexed by symbol * table_size + state):
//   bits  0..11  = compressed_state (12 bits)
//   bits 12..15  = bits_to_output   (4 bits)
//   bits 16..31  = base             (16 bits)
//
// Output per lane:
//   - Bitstream bytes in a pre-allocated section of output_data
//   - lane_results[lane*3 + 0] = initial_state (final encoding state)
//   - lane_results[lane*3 + 1] = total_bits written
//   - lane_results[lane*3 + 2] = bitstream byte length

// @pz_cost {
//   threads_per_element: 0.015625
//   passes: 1
//   buffers: symbols=N, encode_table=131072, output_data=N*2, lane_results=192
//   local_mem: 0
//   note: threads = num_lanes (typ. 4-32). encode_table = 256*table_size*4 (typ 128KB). output worst case ~1.5 bytes/sym.
// }

// Maximum encode table entries that fit in __local memory.
// 256 * 128 = 32K entries × 4 bytes = 128KB — too large for most GPUs' local
// memory (typically 32-64KB). So we read from __global. For small table_size
// (32 entries, accuracy_log=5), the table is only 32KB and could fit, but
// the kernel is simple enough that global memory access is fine.

__kernel void FseEncode(
    __global const unsigned char *symbols,        // input symbols (all lanes interleaved)
    __global const unsigned int *encode_table,    // packed encode table [256 * table_size]
    __global unsigned char *output_data,           // per-lane bitstream output sections
    __global unsigned int *lane_results,           // [initial_state, total_bits, byte_len] per lane
    const unsigned int num_lanes,
    const unsigned int total_input_len,
    const unsigned int table_size,                 // 1 << accuracy_log
    const unsigned int max_output_bytes_per_lane)  // worst-case output section size
{
    unsigned int lane_id = get_global_id(0);
    if (lane_id >= num_lanes) return;

    // Count how many symbols this lane processes.
    // Lane L processes symbols at positions L, L+num_lanes, L+2*num_lanes, ...
    unsigned int num_symbols = (total_input_len - lane_id + num_lanes - 1u) / num_lanes;

    // Phase 1: Reverse pass — compute (value, bits) chunks.
    // We process symbols in reverse order within this lane, storing chunks
    // such that chunks[0] corresponds to the first symbol for this lane
    // (forward order). This matches the CPU fse_encode_interleaved().
    //
    // State starts at 0 (same as CPU). After processing all symbols in
    // reverse, the final state becomes the decoder's initial_state.
    unsigned int state = 0u;

    // We need to store chunks temporarily. Since we can't dynamically
    // allocate in a kernel, we write the bitstream in a second pass.
    // For the reverse pass, we accumulate chunks in output_data as
    // packed (value:16, bits:16) pairs, then overwrite with the actual
    // bitstream in the forward pass.
    //
    // Each chunk is at most (accuracy_log) bits of value + 4 bits of count.
    // Pack as: (value & 0xFFFF) | (bits << 16)
    __global unsigned int *chunks = (__global unsigned int *)(output_data + lane_id * max_output_bytes_per_lane);
    // Ensure we have enough room: num_symbols * 4 bytes for chunks,
    // which is always <= max_output_bytes_per_lane (since worst case
    // bitstream is ~1.5 bytes/symbol, and chunks use 4 bytes/symbol,
    // but max_output_bytes_per_lane accounts for this).

    unsigned int cursor = num_symbols; // fill from end

    for (int sym_idx = (int)num_symbols - 1; sym_idx >= 0; sym_idx--) {
        unsigned int global_idx = (unsigned int)sym_idx * num_lanes + lane_id;
        if (global_idx >= total_input_len) {
            continue;
        }

        unsigned int s = (unsigned int)symbols[global_idx];
        unsigned int entry = encode_table[s * table_size + state];

        unsigned int compressed_state = entry & 0xFFFu;
        unsigned int bits = (entry >> 12u) & 0xFu;
        unsigned int base = entry >> 16u;

        unsigned int value = state - base;

        cursor--;
        chunks[cursor] = (value & 0xFFFFu) | (bits << 16u);

        state = compressed_state;
    }

    // Phase 2: Forward pass — pack bits LSB-first into bitstream.
    // Iterate chunks[cursor..num_symbols] in forward order.
    __global unsigned char *out_bytes = output_data + lane_id * max_output_bytes_per_lane;

    unsigned int container = 0u;
    unsigned int bit_pos = 0u;
    unsigned int byte_count = 0u;

    for (unsigned int i = cursor; i < num_symbols; i++) {
        unsigned int chunk = chunks[i];
        unsigned int value = chunk & 0xFFFFu;
        unsigned int nb_bits = chunk >> 16u;

        if (nb_bits > 0u) {
            container |= (value << bit_pos);
            bit_pos += nb_bits;

            // Flush complete bytes
            while (bit_pos >= 8u) {
                out_bytes[byte_count] = (unsigned char)(container & 0xFFu);
                container >>= 8u;
                bit_pos -= 8u;
                byte_count++;
            }
        }
    }

    // Flush remaining partial byte
    unsigned int total_bits = byte_count * 8u + bit_pos;
    if (bit_pos > 0u) {
        out_bytes[byte_count] = (unsigned char)(container & 0xFFu);
        byte_count++;
    }

    // Write results
    lane_results[lane_id * 3u]      = state;       // initial_state for decoder
    lane_results[lane_id * 3u + 1u] = total_bits;
    lane_results[lane_id * 3u + 2u] = byte_count;  // bitstream byte length
}
