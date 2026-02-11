// GPU Huffman encoding kernel.
//
// Two-pass approach:
//   Pass 1 (ComputeBitLengths): Each work-item looks up its symbol's code
//     length from the lookup table and writes it to the bit_lengths array.
//   Pass 2 (WriteCodes): After a prefix sum on bit_lengths produces bit
//     offsets, each work-item writes its codeword at the computed offset.
//
// The prefix sum between passes is performed on the host using the GPU
// buffer, or could use a GPU scan kernel for very large inputs.
//
// Bit packing uses MSB-first order matching the CPU Huffman encoder.

// Code lookup table entry: codeword in low 24 bits, bit count in high 8 bits.
// Packed as: (bits << 24) | codeword
// This lets us store the full table in 256 uints.

// Pass 1: Compute the bit length for each symbol position.
// Stores the number of bits each symbol will occupy in the output.
__kernel void ComputeBitLengths(
    __global const unsigned char *symbols,   // input symbols
    __global const unsigned int *code_lut,   // 256-entry: (bits << 24) | codeword
    __global unsigned int *bit_lengths,      // output: bits for each position
    const unsigned int num_symbols)
{
    unsigned int gid = get_global_id(0);
    if (gid >= num_symbols) return;

    unsigned int entry = code_lut[symbols[gid]];
    unsigned int bits = entry >> 24;
    bit_lengths[gid] = bits;
}

// Pass 2: Write codewords at pre-computed bit offsets.
// bit_offsets[i] = sum of bit_lengths[0..i) (exclusive prefix sum).
// Each work-item writes its codeword bits into the output buffer.
//
// Output uses MSB-first bit packing: bit 0 of output byte 0 is the
// most significant bit (bit 7), matching the CPU encoder.
__kernel void WriteCodes(
    __global const unsigned char *symbols,
    __global const unsigned int *code_lut,
    __global const unsigned int *bit_offsets,
    __global volatile unsigned int *output,  // output as uint array for atomic ops
    const unsigned int num_symbols)
{
    unsigned int gid = get_global_id(0);
    if (gid >= num_symbols) return;

    unsigned int entry = code_lut[symbols[gid]];
    unsigned int bits = entry >> 24;
    unsigned int codeword = entry & 0x00FFFFFFu;
    unsigned int start_bit = bit_offsets[gid];

    // Write each bit of the codeword, MSB first
    for (unsigned int i = 0; i < bits; i++) {
        unsigned int bit_idx = (bits - 1) - i;  // MSB first
        unsigned int bit_val = (codeword >> bit_idx) & 1u;
        if (bit_val) {
            unsigned int global_bit = start_bit + i;
            // MSB-first within each byte: bit 0 of byte = position 7
            // But we pack into uint (4 bytes), so:
            unsigned int uint_idx = global_bit / 32;
            unsigned int bit_in_uint = 31 - (global_bit % 32);
            atomic_or(&output[uint_idx], 1u << bit_in_uint);
        }
    }
}

// Histogram kernel: count byte frequencies in the input.
// Uses atomic increments on a 256-entry histogram buffer.
// This avoids downloading data to CPU just to count frequencies.
__kernel void ByteHistogram(
    __global const unsigned char *data,
    __global volatile unsigned int *histogram,  // 256 entries, pre-zeroed
    const unsigned int data_len)
{
    unsigned int gid = get_global_id(0);
    if (gid >= data_len) return;

    atomic_inc(&histogram[data[gid]]);
}
