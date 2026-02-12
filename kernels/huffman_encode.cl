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

// Block-level exclusive prefix sum using work-group local memory.
// Processes one block of up to `block_size` elements per work-group.
// Uses Blelloch scan (up-sweep + down-sweep) within each work-group.
//
// For inputs larger than one work-group, the host must:
// 1. Run PrefixSumBlock on each block, collecting block totals
// 2. Recursively scan block totals
// 3. Run PrefixSumApply to add block offsets
__kernel void PrefixSumBlock(
    __global unsigned int *data,       // in/out: values to scan
    __global unsigned int *block_sums, // out: total for each block (may be NULL for single-block)
    const unsigned int n,              // total number of elements
    __local unsigned int *temp)        // local memory: 2 * local_size uints
{
    unsigned int lid = get_local_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int local_size = get_local_size(0);
    unsigned int block_size = local_size * 2;

    // Load two elements per work-item into local memory
    unsigned int ai = lid;
    unsigned int bi = lid + local_size;
    unsigned int ga = group_id * block_size + ai;
    unsigned int gb = group_id * block_size + bi;

    temp[ai] = (ga < n) ? data[ga] : 0;
    temp[bi] = (gb < n) ? data[gb] : 0;

    // Up-sweep (reduce) phase
    unsigned int offset = 1;
    for (unsigned int d = block_size >> 1; d > 0; d >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < d) {
            unsigned int ai2 = offset * (2 * lid + 1) - 1;
            unsigned int bi2 = offset * (2 * lid + 2) - 1;
            temp[bi2] += temp[ai2];
        }
        offset <<= 1;
    }

    // Save block total and set last element to 0 for exclusive scan
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        if (block_sums != 0) {
            block_sums[group_id] = temp[block_size - 1];
        }
        temp[block_size - 1] = 0;
    }

    // Down-sweep phase
    for (unsigned int d = 1; d < block_size; d <<= 1) {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < d) {
            unsigned int ai2 = offset * (2 * lid + 1) - 1;
            unsigned int bi2 = offset * (2 * lid + 2) - 1;
            unsigned int t = temp[ai2];
            temp[ai2] = temp[bi2];
            temp[bi2] += t;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Write results back to global memory
    if (ga < n) data[ga] = temp[ai];
    if (gb < n) data[gb] = temp[bi];
}

// Apply block offsets to produce the final prefix sum.
// After scanning block totals, add each block's offset to its elements.
// block_size is passed explicitly so local_work_size can be any valid value.
__kernel void PrefixSumApply(
    __global unsigned int *data,             // in/out: block-level prefix sums
    __global const unsigned int *block_sums, // scanned block totals
    const unsigned int n,
    const unsigned int block_size)
{
    unsigned int gid = get_global_id(0);
    if (gid >= n) return;

    unsigned int block_id = gid / block_size;
    if (block_id > 0) {
        data[gid] += block_sums[block_id];
    }
}
