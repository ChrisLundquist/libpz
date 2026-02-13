// GPU FSE (tANS) decode kernel (OpenCL).
//
// Port of fse_decode.wgsl. Each work-item decodes one independent stream
// from N-way interleaved FSE output. Parallelism comes from decoding N
// streams simultaneously.
//
// Decode table entries are packed as uint:
//   bits 0..7   = symbol (8 bits)
//   bits 8..15  = bits_to_read (8 bits)
//   bits 16..31 = next_state_base (16 bits)
//
// Bitstream is LSB-first (matching the CPU FSE BitWriter).
//
// The decode table is cached in __local (shared) memory when it fits
// (up to 4096 entries = 16KB). This avoids repeated global memory reads
// in the hot decode loop.

// @pz_cost {
//   threads_per_element: 0.015625
//   passes: 1
//   buffers: decode_table=4096, bitstream_data=N, stream_meta=64, output=N
//   local_mem: 16384
//   note: threads = num_streams (typ. 4-8), each decodes N/K symbols. Output is u32-packed (atomic_or).
// }

// Maximum decode table entries that fit in __local memory.
// 4096 entries × 4 bytes = 16KB. Most GPUs have 32-64KB local memory,
// so this leaves room for ≥2 warps/SM to be active concurrently.
#ifndef MAX_LOCAL_TABLE_ENTRIES
#define MAX_LOCAL_TABLE_ENTRIES 4096u
#endif

// Read a byte from the bitstream data given a byte offset.
static inline unsigned int read_bitstream_byte(
    __global const unsigned char *bitstream_data,
    unsigned int byte_offset)
{
    return (unsigned int)bitstream_data[byte_offset];
}

// Write a single decoded byte to the output buffer at a byte position.
// Uses atomic_or since multiple streams may write to different bytes
// in the same uint word.
static inline void write_output_byte(
    __global volatile unsigned int *output,
    unsigned int pos,
    unsigned int value)
{
    unsigned int word_idx = pos / 4u;
    unsigned int byte_in_word = pos % 4u;
    unsigned int shift = byte_in_word * 8u;
    atomic_or(&output[word_idx], value << shift);
}

// Per-stream metadata layout: [initial_state, total_bits, bitstream_byte_offset, num_symbols]
// Stream i starts at stream_meta[i * 4].

__kernel void FseDecode(
    __global const unsigned int *decode_table,        // packed DecodeEntry[table_size]
    __global const unsigned char *bitstream_data,     // all streams concatenated
    __global const unsigned int *stream_meta,          // per-stream metadata
    __global volatile unsigned int *output,            // output buffer (u32-packed bytes)
    const unsigned int num_streams,
    const unsigned int total_output_len,
    const unsigned int table_size,                     // 1 << accuracy_log
    __local unsigned int *local_decode_table)          // __local cache for decode table
{
    unsigned int stream_id = get_global_id(0);
    unsigned int lid = get_local_id(0);
    unsigned int wg_size = get_local_size(0);

    // Cooperatively load decode table into __local memory
    unsigned int load_size = min(table_size, MAX_LOCAL_TABLE_ENTRIES);
    for (unsigned int i = lid; i < load_size; i += wg_size) {
        local_decode_table[i] = decode_table[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (stream_id >= num_streams) return;

    // Choose table source: __local if it fits, __global otherwise
    int use_local = (table_size <= MAX_LOCAL_TABLE_ENTRIES) ? 1 : 0;

    // Read per-stream metadata
    unsigned int meta_base = stream_id * 4u;
    unsigned int state = stream_meta[meta_base];
    unsigned int total_bits = stream_meta[meta_base + 1u];
    unsigned int bs_byte_offset = stream_meta[meta_base + 2u];
    unsigned int num_symbols = stream_meta[meta_base + 3u];

    // Bit reader state: 32-bit container, LSB-first
    unsigned int container = 0u;
    unsigned int bits_available = 0u;
    unsigned int byte_pos = 0u;

    // Total bytes in this stream's bitstream
    unsigned int total_bytes = (total_bits + 7u) / 8u;

    // Initial refill (up to 4 bytes)
    for (unsigned int r = 0; r < 4u; r++) {
        if (byte_pos < total_bytes) {
            container |= read_bitstream_byte(bitstream_data, bs_byte_offset + byte_pos) << bits_available;
            byte_pos++;
            bits_available += 8u;
        }
    }

    // Decode loop: emit one symbol per iteration
    for (unsigned int sym_idx = 0; sym_idx < num_symbols; sym_idx++) {
        // Table lookup: use __local or __global depending on table size
        unsigned int entry;
        if (use_local) {
            entry = local_decode_table[state];
        } else {
            entry = decode_table[state];
        }
        unsigned int symbol = entry & 0xFFu;
        unsigned int bits_to_read = (entry >> 8u) & 0xFFu;
        unsigned int next_state_base = entry >> 16u;

        // Write symbol to round-robin output position
        unsigned int out_pos = sym_idx * num_streams + stream_id;
        if (out_pos < total_output_len) {
            write_output_byte(output, out_pos, symbol);
        }

        // Refill if needed
        if (bits_available < bits_to_read) {
            for (unsigned int r = 0; r < 4u; r++) {
                if (bits_available <= 24u && byte_pos < total_bytes) {
                    container |= read_bitstream_byte(bitstream_data, bs_byte_offset + byte_pos) << bits_available;
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
