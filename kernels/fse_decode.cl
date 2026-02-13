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
    const unsigned int total_output_len)
{
    unsigned int stream_id = get_global_id(0);
    if (stream_id >= num_streams) return;

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
        // Table lookup: decode_table[state] -> packed entry
        unsigned int entry = decode_table[state];
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
