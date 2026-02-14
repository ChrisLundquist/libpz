// GPU FSE (tANS) multi-block decode kernel (WGSL).
//
// Decodes multiple independently-encoded FSE blocks in a single kernel
// launch. Each workgroup handles one block. Within each workgroup, work-items
// decode the N interleaved streams (one stream per work-item).
//
// Total invocations = num_blocks * max_streams_per_block.
//
// Decode table entries are packed as u32:
//   bits 0..7   = symbol (8 bits)
//   bits 8..15  = bits_to_read (8 bits)
//   bits 16..31 = next_state_base (16 bits)
//
// The decode table is cached in workgroup memory per workgroup.
// All blocks share the same decode table (encoded with a shared freq table).

// @pz_cost {
//   threads_per_element: 0.015625
//   passes: 1
//   buffers: decode_table=4096, bitstream_data=N, block_meta=N*0.001, stream_meta=N*0.05, output=N
//   local_mem: 16384
//   note: one workgroup per block, N streams per block (typ. 4). Batched: single launch for all blocks.
// }

@group(0) @binding(0) var<storage, read> decode_table: array<u32>;
@group(0) @binding(1) var<storage, read> bitstream_data: array<u32>;
// Combined metadata buffer: block_meta entries first, then stream_meta entries.
// Block meta: 3 u32 per block [output_offset, output_len, stream_meta_offset]
// Stream meta: 4 u32 per stream [initial_state, total_bits, byte_offset, num_symbols]
// stream_meta_offset is relative to the start of the stream meta region.
@group(0) @binding(2) var<storage, read> metadata: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<atomic<u32>>;
// Params: x=num_blocks, y=max_streams, z=table_size, w=block_meta_count (number of u32 in block_meta region)
@group(0) @binding(4) var<uniform> params: vec4<u32>;

// Workgroup-local decode table cache (max 4096 entries for accuracy_log ≤ 12).
var<workgroup> local_table: array<u32, 4096>;

fn read_bs_byte(byte_offset: u32) -> u32 {
    let word_idx = byte_offset / 4u;
    let byte_in_word = byte_offset % 4u;
    return (bitstream_data[word_idx] >> (byte_in_word * 8u)) & 0xFFu;
}

fn write_byte(pos: u32, value: u32) {
    let word_idx = pos / 4u;
    let byte_in_word = pos % 4u;
    let shift = byte_in_word * 8u;
    // Use atomicOr since multiple streams write to different bytes in the same word
    atomicOr(&output[word_idx], value << shift);
}

@compute @workgroup_size(64)
fn fse_decode_blocks(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let block_id = wgid.x;
    let num_blocks = params.x;
    let max_streams = params.y;
    let table_size = params.z;
    let bm_count = params.w; // number of u32 in block_meta region

    if (block_id >= num_blocks) {
        return;
    }

    // Cooperatively load decode table into workgroup memory
    let load_size = min(table_size, 4096u);
    for (var i = lid.x; i < load_size; i = i + 64u) {
        local_table[i] = decode_table[i];
    }
    workgroupBarrier();

    let stream_id = lid.x;
    if (stream_id >= max_streams) {
        return;
    }

    // Read block metadata from combined buffer: [output_offset, output_len, stream_meta_offset]
    let bm_base = block_id * 3u;
    let out_offset = metadata[bm_base];
    let out_len = metadata[bm_base + 1u];
    let sm_offset = metadata[bm_base + 2u];

    let use_local = table_size <= 4096u;

    // Read per-stream metadata from combined buffer (after block_meta region)
    let sm_base = bm_count + (sm_offset + stream_id) * 4u;
    var state = metadata[sm_base];
    let total_bits = metadata[sm_base + 1u];
    let bs_byte_offset = metadata[sm_base + 2u];
    let num_symbols = metadata[sm_base + 3u];

    // Bit reader state
    var container: u32 = 0u;
    var bits_available: u32 = 0u;
    var byte_pos: u32 = 0u;
    let total_bytes = (total_bits + 7u) / 8u;

    // Initial refill
    for (var r = 0u; r < 4u; r = r + 1u) {
        if (byte_pos < total_bytes) {
            container = container | (read_bs_byte(bs_byte_offset + byte_pos) << bits_available);
            byte_pos = byte_pos + 1u;
            bits_available = bits_available + 8u;
        }
    }

    // Decode loop
    for (var sym_idx = 0u; sym_idx < num_symbols; sym_idx = sym_idx + 1u) {
        var entry: u32;
        if (use_local) {
            entry = local_table[state];
        } else {
            entry = decode_table[state];
        }
        let symbol = entry & 0xFFu;
        let bits_to_read = (entry >> 8u) & 0xFFu;
        let next_state_base = entry >> 16u;

        // Write symbol to round-robin position within this block's output
        let out_pos = out_offset + sym_idx * max_streams + stream_id;
        if (out_pos < out_offset + out_len) {
            write_byte(out_pos, symbol);
        }

        // Refill if needed
        if (bits_available < bits_to_read) {
            for (var r = 0u; r < 4u; r = r + 1u) {
                if (bits_available <= 24u && byte_pos < total_bytes) {
                    container = container | (read_bs_byte(bs_byte_offset + byte_pos) << bits_available);
                    byte_pos = byte_pos + 1u;
                    bits_available = bits_available + 8u;
                }
            }
        }

        // Read bits from container (LSB-first)
        var value: u32 = 0u;
        if (bits_to_read > 0u) {
            let mask = (1u << bits_to_read) - 1u;
            value = container & mask;
            container = container >> bits_to_read;
            bits_available = bits_available - bits_to_read;
        }

        state = next_state_base + value;
    }
}
