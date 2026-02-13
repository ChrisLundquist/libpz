// Static-dictionary LZW decode kernel (OpenCL).
//
// Each work-item decodes one code from the compressed bitstream:
//   1. Read code_bits bits at position (gid * code_bits) from the bitstream
//   2. Look up the dictionary entry length for that code
//   3. (After prefix sum) Copy the dictionary entry to the output
//
// This is embarrassingly parallel: the dictionary is frozen (static),
// so there are no inter-code dependencies.
//
// Three-pass approach:
//   Pass 1 (DecodeLengths): read codes, write entry lengths
//   Pass 2 (prefix sum on lengths -> output_offsets, reuses existing kernels)
//   Pass 3 (WriteOutput): copy dictionary entries to output

// Read `code_bits` bits from a packed bitstream at bit position `bit_offset`.
// The bitstream is packed LSB-first.
static inline unsigned int read_code(
    __global const unsigned char *bitstream,
    unsigned int bit_offset,
    unsigned int code_bits)
{
    unsigned int byte_offset = bit_offset / 8u;
    unsigned int bit_in_byte = bit_offset % 8u;
    // Read up to 4 bytes to cover any code_bits <= 16
    unsigned int raw = 0;
    raw |= (unsigned int)bitstream[byte_offset];
    if (byte_offset + 1u < 0xFFFFFFFFu)
        raw |= (unsigned int)bitstream[byte_offset + 1u] << 8u;
    if (code_bits + bit_in_byte > 16u && byte_offset + 2u < 0xFFFFFFFFu)
        raw |= (unsigned int)bitstream[byte_offset + 2u] << 16u;
    return (raw >> bit_in_byte) & ((1u << code_bits) - 1u);
}

// Pass 1: Decode each code and write its expanded length.
//
// Args:
//   bitstream     - packed code bitstream (LSB-first)
//   dict_lengths  - length of each dictionary entry (u16 per entry)
//   output_lengths - output: expanded length per code (u32 per code)
//   num_codes     - total number of codes
//   code_bits     - bits per code (fixed-width)
__kernel void DecodeLengths(
    __global const unsigned char *bitstream,
    __global const unsigned short *dict_lengths,
    __global unsigned int *output_lengths,
    const unsigned int num_codes,
    const unsigned int code_bits)
{
    unsigned int gid = get_global_id(0);
    if (gid >= num_codes) return;

    unsigned int bit_offset = gid * code_bits;
    unsigned int code = read_code(bitstream, bit_offset, code_bits);
    output_lengths[gid] = (unsigned int)dict_lengths[code];
}

// Pass 3: Copy dictionary entries to output at prefix-summed offsets.
//
// Args:
//   bitstream      - packed code bitstream
//   dict_entries   - flat dictionary: entry i at [i * stride .. i * stride + len]
//   dict_lengths   - length of each dictionary entry
//   output_offsets - exclusive prefix sum of expanded lengths
//   output         - output buffer
//   num_codes      - total number of codes
//   code_bits      - bits per code
//   stride         - max_entry_len (flat dictionary stride)
__kernel void WriteOutput(
    __global const unsigned char *bitstream,
    __global const unsigned char *dict_entries,
    __global const unsigned short *dict_lengths,
    __global const unsigned int *output_offsets,
    __global unsigned char *output,
    const unsigned int num_codes,
    const unsigned int code_bits,
    const unsigned int stride)
{
    unsigned int gid = get_global_id(0);
    if (gid >= num_codes) return;

    unsigned int bit_offset = gid * code_bits;
    unsigned int code = read_code(bitstream, bit_offset, code_bits);
    unsigned int len = (unsigned int)dict_lengths[code];
    unsigned int out_offset = output_offsets[gid];

    __global const unsigned char *src = &dict_entries[code * stride];
    for (unsigned int i = 0; i < len; i++) {
        output[out_offset + i] = src[i];
    }
}
