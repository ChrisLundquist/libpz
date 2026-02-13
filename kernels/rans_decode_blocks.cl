// GPU rANS multi-block decode kernel (OpenCL).
//
// Decodes multiple independently-encoded rANS blocks in a single kernel
// launch. Each block has K interleaved lanes, and each lane gets one
// work-item. Total work-items = num_blocks * lanes_per_block.
//
// This achieves massive parallelism: instead of just K work-items for a
// single stream, we launch thousands (e.g., 256 blocks × 32 lanes = 8192).
//
// The lookup table (cum2sym) is cached in __local memory for fast access.
// Maximum supported scale_bits is 14 (16KB local memory for the table).
//
// Per-block metadata layout (5 entries per block):
//   [output_offset, output_len, lanes_per_block, word_data_offset, lane_meta_offset]
//
// Per-lane metadata layout (3 entries per lane):
//   [initial_state, num_words, word_offset]

#define RANS_L 65536u
#define IO_BITS 16u

// Maximum lookup table size we can cache in __local memory.
// scale_bits=14 → 16384 bytes. Most devices have 32-64KB local memory.
#ifndef MAX_LOCAL_TABLE
#define MAX_LOCAL_TABLE 16384u
#endif

__kernel void RansDecodeBlocks(
    __global const unsigned char *cum2sym,     // symbol lookup table [1 << scale_bits]
    __global const unsigned short *freq_table, // per-symbol frequency [256]
    __global const unsigned short *cum_table,  // per-symbol cumulative freq [256]
    __global const unsigned short *word_data,  // all blocks' word streams concatenated
    __global const unsigned int *block_meta,   // per-block metadata [5 entries each]
    __global const unsigned int *lane_meta,    // per-lane metadata [3 entries each]
    __global unsigned char *output,            // output buffer
    const unsigned int num_blocks,
    const unsigned int scale_bits,
    __local unsigned char *local_cum2sym)      // __local cache for cum2sym
{
    unsigned int lid = get_local_id(0);
    unsigned int wg_size = get_local_size(0);

    // Determine which block and lane this work-item belongs to.
    // We use get_group_id to map workgroups to blocks.
    unsigned int block_id = get_group_id(0);
    if (block_id >= num_blocks) return;

    // Read block metadata
    unsigned int bm_base = block_id * 5u;
    unsigned int out_offset = block_meta[bm_base];
    unsigned int out_len = block_meta[bm_base + 1u];
    unsigned int lanes_per_block = block_meta[bm_base + 2u];
    // block_meta[bm_base + 3u] = word_data_offset (used in lane_meta)
    unsigned int lm_base_offset = block_meta[bm_base + 4u];

    unsigned int lane_id = lid;
    if (lane_id >= lanes_per_block) return;

    unsigned int scale_mask = (1u << scale_bits) - 1u;
    unsigned int table_size = 1u << scale_bits;

    // Cooperatively load cum2sym into __local memory
    for (unsigned int i = lid; i < table_size; i += wg_size) {
        local_cum2sym[i] = cum2sym[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Read per-lane metadata
    unsigned int lm_idx = (lm_base_offset + lane_id) * 3u;
    unsigned int state = lane_meta[lm_idx];
    unsigned int num_words = lane_meta[lm_idx + 1u];
    unsigned int word_offset = lane_meta[lm_idx + 2u];

    unsigned int word_pos = 0u;

    // Compute how many symbols this lane decodes (round-robin within block)
    unsigned int num_symbols = (out_len + lanes_per_block - 1u - lane_id) / lanes_per_block;

    for (unsigned int sym_idx = 0u; sym_idx < num_symbols; sym_idx++) {
        // Decode: table lookup from __local memory
        unsigned int slot = state & scale_mask;
        unsigned char sym = local_cum2sym[slot];
        unsigned int freq = (unsigned int)freq_table[sym];
        unsigned int cum = (unsigned int)cum_table[sym];

        // Write symbol to round-robin output position within this block
        unsigned int out_pos = out_offset + sym_idx * lanes_per_block + lane_id;
        output[out_pos] = sym;

        // Advance state using 64-bit intermediate to avoid overflow
        ulong wide = (ulong)freq * (ulong)(state >> scale_bits);
        state = (unsigned int)(wide) + slot - cum;

        // Renormalize
        if (state < RANS_L && word_pos < num_words) {
            state = (state << IO_BITS) | (unsigned int)word_data[word_offset + word_pos];
            word_pos++;
        }
    }
}
