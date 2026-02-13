// GPU FSE (tANS) multi-block decode kernel (OpenCL).
//
// Decodes multiple independently-encoded FSE blocks in a single kernel
// launch. Each workgroup handles one block. Within each workgroup, work-items
// decode the N interleaved streams (one stream per work-item).
//
// Total work-items = num_blocks * streams_per_block.
//
// Decode table entries are packed as uint:
//   bits 0..7   = symbol (8 bits)
//   bits 8..15  = bits_to_read (8 bits)
//   bits 16..31 = next_state_base (16 bits)
//
// The decode table is cached in __local (shared) memory per workgroup.
// All blocks share the same decode table (encoded with a shared freq table).

// @pz_cost {
//   threads_per_element: 0.015625
//   passes: 1
//   buffers: decode_table=4096, bitstream_data=N, block_meta=N*0.001, stream_meta=N*0.05, output=N
//   local_mem: 16384
//   note: one workgroup per block, N streams per block (typ. 4). Batched: single launch for all blocks.
// }

#ifndef MAX_LOCAL_TABLE_ENTRIES
#define MAX_LOCAL_TABLE_ENTRIES 4096u
#endif

// Read a byte from the bitstream data given a byte offset.
static inline unsigned int read_bs_byte(
    __global const unsigned char *bitstream_data,
    unsigned int byte_offset)
{
    return (unsigned int)bitstream_data[byte_offset];
}

// Per-block metadata layout (3 entries per block):
//   [output_offset, output_len, stream_meta_offset]
//
// Per-stream metadata layout (4 entries per stream):
//   [initial_state, total_bits, bitstream_byte_offset, num_symbols]

__kernel void FseDecodeBlocks(
    __global const unsigned int *decode_table,        // packed DecodeEntry[table_size]
    __global const unsigned char *bitstream_data,     // all blocks' bitstreams concatenated
    __global const unsigned int *block_meta,          // per-block metadata [3 entries each]
    __global const unsigned int *stream_meta,         // per-stream metadata [4 entries each]
    __global unsigned char *output,                   // output buffer (byte-addressed)
    const unsigned int num_blocks,
    const unsigned int max_streams_per_block,
    const unsigned int table_size,                    // 1 << accuracy_log
    __local unsigned int *local_decode_table)          // __local cache
{
    unsigned int lid = get_local_id(0);
    unsigned int wg_size = get_local_size(0);
    unsigned int block_id = get_group_id(0);

    if (block_id >= num_blocks) return;

    // Cooperatively load decode table into __local memory
    unsigned int load_size = min(table_size, MAX_LOCAL_TABLE_ENTRIES);
    for (unsigned int i = lid; i < load_size; i += wg_size) {
        local_decode_table[i] = decode_table[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Read block metadata
    unsigned int bm_base = block_id * 3u;
    unsigned int out_offset = block_meta[bm_base];
    unsigned int out_len = block_meta[bm_base + 1u];
    unsigned int sm_offset = block_meta[bm_base + 2u];

    unsigned int stream_id = lid;
    if (stream_id >= max_streams_per_block) return;

    // Choose table source
    int use_local = (table_size <= MAX_LOCAL_TABLE_ENTRIES) ? 1 : 0;

    // Read per-stream metadata
    unsigned int sm_base = (sm_offset + stream_id) * 4u;
    unsigned int state = stream_meta[sm_base];
    unsigned int total_bits = stream_meta[sm_base + 1u];
    unsigned int bs_byte_offset = stream_meta[sm_base + 2u];
    unsigned int num_symbols = stream_meta[sm_base + 3u];

    // Bit reader state: 32-bit container, LSB-first
    unsigned int container = 0u;
    unsigned int bits_available = 0u;
    unsigned int byte_pos = 0u;
    unsigned int total_bytes = (total_bits + 7u) / 8u;

    // Initial refill (up to 4 bytes)
    for (unsigned int r = 0; r < 4u; r++) {
        if (byte_pos < total_bytes) {
            container |= read_bs_byte(bitstream_data, bs_byte_offset + byte_pos) << bits_available;
            byte_pos++;
            bits_available += 8u;
        }
    }

    // Decode loop: emit one symbol per iteration
    for (unsigned int sym_idx = 0; sym_idx < num_symbols; sym_idx++) {
        // Table lookup
        unsigned int entry;
        if (use_local) {
            entry = local_decode_table[state];
        } else {
            entry = decode_table[state];
        }
        unsigned int symbol = entry & 0xFFu;
        unsigned int bits_to_read = (entry >> 8u) & 0xFFu;
        unsigned int next_state_base = entry >> 16u;

        // Write symbol to round-robin position within this block's output
        unsigned int out_pos = out_offset + sym_idx * max_streams_per_block + stream_id;
        if (out_pos < out_offset + out_len) {
            output[out_pos] = (unsigned char)symbol;
        }

        // Refill if needed
        if (bits_available < bits_to_read) {
            for (unsigned int r = 0; r < 4u; r++) {
                if (bits_available <= 24u && byte_pos < total_bytes) {
                    container |= read_bs_byte(bitstream_data, bs_byte_offset + byte_pos) << bits_available;
                    byte_pos++;
                    bits_available += 8u;
                }
            }
        }

        // Read bits from container (LSB-first)
        unsigned int value = 0u;
        if (bits_to_read > 0u) {
            unsigned int mask = (1u << bits_to_read) - 1u;
            value = container & mask;
            container >>= bits_to_read;
            bits_available -= bits_to_read;
        }

        // Compute next state
        state = next_state_base + value;
    }
}
