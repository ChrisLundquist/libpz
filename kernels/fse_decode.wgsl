// GPU FSE (tANS) decode kernel (WGSL).
//
// Each workgroup decodes one independent stream from N-way interleaved
// FSE output. Workgroup size is 1 (one thread per stream) because FSE
// decode is inherently sequential per-stream — each state depends on
// the previous state.
//
// Parallelism comes from decoding N streams simultaneously across
// N workgroups.
//
// Decode table entries are packed as u32:
//   bits 0..7   = symbol (8 bits)
//   bits 8..15  = bits_to_read (8 bits)
//   bits 16..31 = next_state_base (16 bits)
//
// Bitstream is LSB-first (matching the CPU FSE BitWriter).

// Decode table: indexed by state, each entry is a packed u32.
@group(0) @binding(0) var<storage, read> decode_table: array<u32>;

// Bitstream data: all streams concatenated, packed as u32 words.
// Each stream's data starts at the byte offset given in stream_meta.
@group(0) @binding(1) var<storage, read> bitstream_data: array<u32>;

// Per-stream metadata: [initial_state(u32), total_bits(u32), bitstream_byte_offset(u32), num_symbols(u32)]
// Stream i starts at stream_meta[i * 4].
@group(0) @binding(2) var<storage, read> stream_meta: array<u32>;

// Output buffer: decoded bytes packed as u32 words.
// Symbols are written in round-robin order: output[sym_idx * num_streams + stream_id].
@group(0) @binding(3) var<storage, read_write> output: array<atomic<u32>>;

// Params: [num_streams(u32), table_size(u32), total_output_len(u32), 0]
@group(0) @binding(4) var<uniform> params: vec4<u32>;

// Read a byte from the bitstream data given a byte offset.
fn read_bitstream_byte(byte_offset: u32) -> u32 {
    let word_idx = byte_offset / 4u;
    let byte_in_word = byte_offset % 4u;
    return (bitstream_data[word_idx] >> (byte_in_word * 8u)) & 0xFFu;
}

// Write a single decoded byte to the output buffer at a byte position.
fn write_output_byte(pos: u32, value: u32) {
    let word_idx = pos / 4u;
    let byte_in_word = pos % 4u;
    let shift = byte_in_word * 8u;
    // Use atomicOr since multiple streams may write to different bytes
    // in the same u32 word.
    atomicOr(&output[word_idx], value << shift);
}

@compute @workgroup_size(1)
fn fse_decode(@builtin(global_invocation_id) gid: vec3<u32>) {
    let stream_id = gid.x;
    let num_streams = params.x;
    let total_output_len = params.z;

    if (stream_id >= num_streams) {
        return;
    }

    // Read per-stream metadata.
    let meta_base = stream_id * 4u;
    var state = stream_meta[meta_base];
    let total_bits = stream_meta[meta_base + 1u];
    let bs_byte_offset = stream_meta[meta_base + 2u];
    let num_symbols = stream_meta[meta_base + 3u];

    // Bit reader state: 32-bit container, LSB-first.
    var container: u32 = 0u;
    var bits_available: u32 = 0u;
    var byte_pos: u32 = 0u;

    // Total bytes in this stream's bitstream.
    let total_bytes = (total_bits + 7u) / 8u;

    // Initial refill.
    for (var r = 0u; r < 4u; r = r + 1u) {
        if (byte_pos < total_bytes) {
            container = container | (read_bitstream_byte(bs_byte_offset + byte_pos) << bits_available);
            byte_pos = byte_pos + 1u;
            bits_available = bits_available + 8u;
        }
    }

    // Decode loop: emit one symbol per iteration.
    for (var sym_idx = 0u; sym_idx < num_symbols; sym_idx = sym_idx + 1u) {
        // Table lookup: decode_table[state] → packed entry.
        let entry = decode_table[state];
        let symbol = entry & 0xFFu;
        let bits_to_read = (entry >> 8u) & 0xFFu;
        let next_state_base = entry >> 16u;

        // Write symbol to round-robin output position.
        let out_pos = sym_idx * num_streams + stream_id;
        if (out_pos < total_output_len) {
            write_output_byte(out_pos, symbol);
        }

        // Read bits from container (LSB-first).
        // Refill first if needed.
        if (bits_available < bits_to_read) {
            for (var r = 0u; r < 4u; r = r + 1u) {
                if (bits_available <= 24u && byte_pos < total_bytes) {
                    container = container | (read_bitstream_byte(bs_byte_offset + byte_pos) << bits_available);
                    byte_pos = byte_pos + 1u;
                    bits_available = bits_available + 8u;
                }
            }
        }

        var value: u32 = 0u;
        if (bits_to_read > 0u) {
            let mask = (1u << bits_to_read) - 1u;
            value = container & mask;
            container = container >> bits_to_read;
            bits_available = bits_available - bits_to_read;
        }

        // Compute next state.
        state = next_state_base + value;
    }
}
