// GPU rANS decode kernel (OpenCL).
//
// Each work-item decodes one lane of N-way interleaved rANS output.
// Uses 64-bit intermediate for the state transition to avoid u32 overflow.
//
// The cum2sym lookup table is cached in __local (shared) memory for fast
// access. At scale_bits=12 this is 4KB, fitting easily in local memory
// and leaving room for multiple warps/SM to be active.
//
// Decode table:
//   cum2sym[slot] = symbol for cumulative frequency slot (size: 1 << scale_bits)
//   freq[256]     = per-symbol frequency
//   cum[256]      = per-symbol cumulative frequency
//
// Each lane's symbols are written to round-robin positions:
//   output[sym_idx * num_lanes + lane_id]
//
// Per-lane metadata layout: [initial_state, num_words, word_offset]
// Lane i starts at lane_meta[i * 3].

#define RANS_L 65536u
#define IO_BITS 16u

// Maximum lookup table size for __local caching.
// scale_bits=14 → 16384 bytes.
#ifndef MAX_LOCAL_TABLE
#define MAX_LOCAL_TABLE 16384u
#endif

__kernel void RansDecode(
    __global const unsigned char *cum2sym,     // symbol lookup table [1 << scale_bits]
    __global const unsigned short *freq_table, // per-symbol frequency [256]
    __global const unsigned short *cum_table,  // per-symbol cumulative freq [256]
    __global const unsigned short *word_data,  // all lanes' word streams concatenated
    __global const unsigned int *lane_meta,    // [initial_state, num_words, word_offset] per lane
    __global unsigned char *output,            // output buffer
    const unsigned int num_lanes,
    const unsigned int total_output_len,
    const unsigned int scale_bits,
    __local unsigned char *local_cum2sym)      // __local cache for cum2sym
{
    unsigned int lane_id = get_global_id(0);
    unsigned int lid = get_local_id(0);
    unsigned int wg_size = get_local_size(0);

    unsigned int table_size = 1u << scale_bits;
    unsigned int scale_mask = table_size - 1u;

    // Cooperatively load cum2sym into __local memory
    unsigned int load_size = min(table_size, MAX_LOCAL_TABLE);
    for (unsigned int i = lid; i < load_size; i += wg_size) {
        local_cum2sym[i] = cum2sym[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lane_id >= num_lanes) return;

    int use_local = (table_size <= MAX_LOCAL_TABLE) ? 1 : 0;

    // Read per-lane metadata
    unsigned int meta_base = lane_id * 3u;
    unsigned int state = lane_meta[meta_base];
    unsigned int num_words = lane_meta[meta_base + 1u];
    unsigned int word_offset = lane_meta[meta_base + 2u];

    unsigned int word_pos = 0u;

    // Compute how many symbols this lane decodes
    // Lane i decodes symbols at positions i, i+num_lanes, i+2*num_lanes, ...
    unsigned int num_symbols = (total_output_len + num_lanes - 1u - lane_id) / num_lanes;

    for (unsigned int sym_idx = 0u; sym_idx < num_symbols; sym_idx++) {
        // Decode: table lookup from __local or __global
        unsigned int slot = state & scale_mask;
        unsigned char sym;
        if (use_local) {
            sym = local_cum2sym[slot];
        } else {
            sym = cum2sym[slot];
        }
        unsigned int freq = (unsigned int)freq_table[sym];
        unsigned int cum = (unsigned int)cum_table[sym];

        // Write symbol to round-robin output position
        unsigned int out_pos = sym_idx * num_lanes + lane_id;
        if (out_pos < total_output_len) {
            output[out_pos] = sym;
        }

        // Advance state using 64-bit intermediate to avoid overflow
        // state = freq * (state >> scale_bits) + slot - cum
        ulong wide = (ulong)freq * (ulong)(state >> scale_bits);
        state = (unsigned int)(wide) + slot - cum;

        // Renormalize: read one 16-bit word if state dropped below RANS_L
        if (state < RANS_L && word_pos < num_words) {
            state = (state << IO_BITS) | (unsigned int)word_data[word_offset + word_pos];
            word_pos++;
        }
    }
}
