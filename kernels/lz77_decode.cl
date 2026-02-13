// GPU LZ77 block-parallel decompression kernel (OpenCL).
//
// Each workgroup decompresses one independent block of serialized LZ77 matches.
// Thread 0 (leader) performs sequential match parsing; all threads cooperate
// on copying back-referenced bytes and literals.
//
// Match format (5 bytes each, little-endian):
//   bytes 0-1: offset (u16) — distance back from current position
//   bytes 2-3: length (u16) — number of back-referenced bytes to copy
//   byte 4:    next   (u8)  — literal byte following the match
//
// Block metadata layout (per block, 3 x uint):
//   [0] match_data_offset  — byte offset into the match data buffer
//   [1] num_matches        — number of 5-byte match records in this block
//   [2] output_offset      — byte offset into the output buffer

// Number of threads per workgroup (set via -D at compile time).
#ifndef WG_SIZE
#define WG_SIZE 64u
#endif

#define MATCH_SIZE 5u

__kernel void Lz77DecodeBlock(
    __global const unsigned char *match_data,   // all blocks' serialized matches
    __global const unsigned int *block_meta,     // [offset, num_matches, out_offset] per block
    __global unsigned char *output,              // decompressed output
    const unsigned int num_blocks,
    const unsigned int total_output_len)
{
    unsigned int block_id = get_group_id(0);
    if (block_id >= num_blocks) return;

    unsigned int lid = get_local_id(0);

    // Read block metadata
    unsigned int meta_base = block_id * 3u;
    unsigned int data_off = block_meta[meta_base];
    unsigned int num_matches = block_meta[meta_base + 1u];
    unsigned int out_base = block_meta[meta_base + 2u];

    // Shared memory for the leader to communicate match info to all threads.
    // [0] = copy_src_start, [1] = copy_len, [2] = write_pos (before this match)
    __local unsigned int match_info[3];

    unsigned int write_pos = 0u;

    for (unsigned int m = 0u; m < num_matches; m++) {
        // Leader reads the match
        if (lid == 0u) {
            unsigned int base = data_off + m * MATCH_SIZE;
            unsigned int offset = (unsigned int)match_data[base]
                                | ((unsigned int)match_data[base + 1u] << 8u);
            unsigned int length = (unsigned int)match_data[base + 2u]
                                | ((unsigned int)match_data[base + 3u] << 8u);

            // Compute source position for back-reference copy
            unsigned int src_start = 0u;
            if (offset > 0u && length > 0u && offset <= write_pos) {
                src_start = write_pos - offset;
            } else {
                length = 0u; // no valid back-reference
            }

            match_info[0] = src_start;
            match_info[1] = length;
            match_info[2] = write_pos;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        unsigned int src_start = match_info[0];
        unsigned int copy_len = match_info[1];
        unsigned int wp = match_info[2];

        // Cooperative copy of back-referenced bytes.
        // Note: LZ77 back-references can overlap (offset < length), so we
        // must copy byte-by-byte in order. Each thread handles a stride.
        // For overlapping matches (offset < length), the pattern repeats,
        // so we compute: src = src_start + (i % offset) when offset > 0.
        if (copy_len > 0u) {
            unsigned int offset_val = wp - src_start; // = original offset
            for (unsigned int i = lid; i < copy_len; i += WG_SIZE) {
                unsigned int src_idx;
                if (offset_val >= copy_len) {
                    // Non-overlapping: direct copy
                    src_idx = src_start + i;
                } else {
                    // Overlapping: use modular indexing for repeating pattern
                    src_idx = src_start + (i % offset_val);
                }
                unsigned int dst = out_base + wp + i;
                if (dst < total_output_len) {
                    output[dst] = output[out_base + src_idx];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Leader appends the literal byte and advances write_pos
        if (lid == 0u) {
            write_pos = wp + copy_len;
            unsigned int dst = out_base + write_pos;
            if (dst < total_output_len) {
                unsigned int base = data_off + m * MATCH_SIZE;
                output[dst] = match_data[base + 4u];
            }
            write_pos++;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Sync write_pos for next iteration
        if (lid == 0u) {
            match_info[2] = write_pos;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        write_pos = match_info[2];
    }
}
