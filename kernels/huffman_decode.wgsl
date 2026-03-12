// GPU Huffman sync-point parallel decode kernel (WGSL).
//
// Each thread independently decodes one segment between adjacent sync points.
// Sync points record (bit_offset, symbol_index) checkpoints placed every N
// symbols during encoding, enabling embarrassingly parallel decode.
//
// The decode table is a 4096-entry 12-bit LUT: entry = (symbol << 8) | code_bits.
// Bitstream is MSB-first packed, uploaded as big-endian u32 words.
//
// Output uses atomic u32 with atomicOr for sub-word byte writes, since
// adjacent segments may write bytes into the same u32 word.

// @pz_cost {
//   threads_per_element: 0.001
//   passes: 1
//   buffers: bitstream=N*0.5, decode_table=16384, sync_points=N*0.016, output=N
//   local_mem: 0
//   note: one thread per sync-point segment. Independent decode, no shared memory.
// }

const WG_SIZE: u32 = 64u;
const DECODE_BITS: u32 = 12u;
const DECODE_MASK: u32 = 0xFFFu;

// Huffman-encoded bitstream, stored as big-endian u32 words.
// Bit 31 of word 0 is the first bit of the stream.
@group(0) @binding(0) var<storage, read> bitstream: array<u32>;

// 4096-entry decode LUT: entry = (symbol << 8) | code_bits
@group(0) @binding(1) var<storage, read> decode_table: array<u32>;

// Sync points as flat u32 pairs: [bit_offset_0, symbol_index_0, bit_offset_1, symbol_index_1, ...]
// Includes sentinel at the end.
@group(0) @binding(2) var<storage, read> sync_points: array<u32>;

// Output buffer (atomic for safe sub-word byte writes from adjacent segments)
@group(0) @binding(3) var<storage, read_write> output: array<atomic<u32>>;

// Params: x=num_segments, y=total_output_symbols, z=0, w=dispatch_width
@group(0) @binding(4) var<uniform> params: vec4<u32>;

/// Peek 12 bits from the MSB-first bitstream at an arbitrary bit position.
///
/// The bitstream is stored as big-endian u32 words: bit 0 of the stream
/// is in bit 31 of word 0. This function loads two adjacent words and
/// combines them to extract 12 bits at any alignment.
fn peek_12_msb(bit_pos: u32) -> u32 {
    let word_idx = bit_pos >> 5u;       // bit_pos / 32
    let bit_in_word = bit_pos & 31u;    // bit_pos % 32

    let w0 = bitstream[word_idx];
    let w1 = bitstream[word_idx + 1u];  // safe: buffer is padded by 4 bytes

    // Shift w0 left by bit_in_word to align the start bit to position 31,
    // then fill the low bits from w1. When bit_in_word == 0, shifting w1
    // right by 32 is undefined, so we use select() to avoid it.
    let combined = select(
        (w0 << bit_in_word) | (w1 >> (32u - bit_in_word)),
        w0,
        bit_in_word == 0u
    );

    // Extract the top 12 bits.
    return (combined >> 20u) & DECODE_MASK;
}

/// Write a single byte to the packed output buffer using atomic OR.
fn write_output_byte(byte_pos: u32, value: u32) {
    let word_idx = byte_pos / 4u;
    let byte_in_word = byte_pos % 4u;
    // Little-endian byte layout within each u32 word.
    atomicOr(&output[word_idx], value << (byte_in_word * 8u));
}

@compute @workgroup_size(64)
fn huffman_sync_decode(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let num_segments = params.x;
    let dispatch_width = params.w;

    // Linearize 2D dispatch to get segment ID.
    let seg_id = gid.x + gid.y * dispatch_width;

    if (seg_id >= num_segments) {
        return;
    }

    // Read this segment's sync point and the next one (sentinel for last segment).
    let sp_base = seg_id * 2u;
    let start_bit = sync_points[sp_base];
    let start_sym = sync_points[sp_base + 1u];
    let end_sym = sync_points[sp_base + 3u];  // next segment's symbol_index

    let num_symbols = end_sym - start_sym;
    if (num_symbols == 0u) {
        return;
    }

    var bit_pos = start_bit;

    for (var i = 0u; i < num_symbols; i = i + 1u) {
        // Peek 12 bits and look up in the decode table.
        let peek = peek_12_msb(bit_pos);
        let entry = decode_table[peek];
        let symbol = entry >> 8u;
        let code_bits = entry & 0xFFu;

        // Write decoded symbol to output.
        write_output_byte(start_sym + i, symbol);

        // Advance bit position by the code length.
        bit_pos = bit_pos + code_bits;
    }
}
