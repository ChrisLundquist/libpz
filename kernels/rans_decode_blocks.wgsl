// GPU rANS multi-block decode kernel (WGSL).
//
// Decodes multiple independently-encoded rANS blocks in a single kernel
// launch. Each workgroup handles one block with K interleaved lanes.
// Total invocations = num_blocks * max_lanes_per_block.
//
// The cum2sym lookup table is cached in workgroup memory for fast access.
// Packed 4 bytes per u32 to stay within WGSL workgroup memory limits
// (max 16KB → 4096 u32 → 16384 bytes of cum2sym at scale_bits=14).
//
// Uses wrapping u32 multiplication (no u64 needed).
//
// Per-block metadata layout (5 entries per block):
//   [output_offset, output_len, lanes_per_block, unused, lane_meta_offset]
//
// Per-lane metadata layout (3 entries per lane):
//   [initial_state, num_words, word_offset]

// @pz_cost {
//   threads_per_element: 0.0625
//   passes: 1
//   buffers: tables=N, word_data=N, metadata=N*0.05, output=N
//   local_mem: 16384
//   note: one workgroup per block. K lanes per block (typ. 16). Batched.
// }

const RANS_L: u32 = 65536u;
const IO_BITS: u32 = 16u;

// Combined tables buffer (same layout as rans_decode.wgsl):
//   [0 .. table_size/4): cum2sym packed 4 bytes per u32
//   [table_size/4 .. table_size/4 + 256): freq_cum packed u32 (low16=freq, high16=cum)
@group(0) @binding(0) var<storage, read> tables: array<u32>;

// Word data: all blocks' u16 word streams concatenated, packed as u32
@group(0) @binding(1) var<storage, read> word_data: array<u32>;

// Combined metadata: block_meta entries first, then lane_meta entries.
// Block meta: 5 u32 per block
// Lane meta: 3 u32 per lane
@group(0) @binding(2) var<storage, read> metadata: array<u32>;

// Output buffer
@group(0) @binding(3) var<storage, read_write> output: array<atomic<u32>>;

// Params: x=num_blocks, y=scale_bits, z=table_size, w=block_meta_count
@group(0) @binding(4) var<uniform> params: vec4<u32>;

// Workgroup-local cum2sym cache: 4096 u32 = 16384 bytes of cum2sym.
var<workgroup> local_cum2sym: array<u32, 4096>;

fn read_cum2sym_global(tables_offset: u32, slot: u32) -> u32 {
    let word_idx = tables_offset + slot / 4u;
    let byte_in_word = slot % 4u;
    return (tables[word_idx] >> (byte_in_word * 8u)) & 0xFFu;
}

fn read_cum2sym_local(slot: u32) -> u32 {
    let word_idx = slot / 4u;
    let byte_in_word = slot % 4u;
    return (local_cum2sym[word_idx] >> (byte_in_word * 8u)) & 0xFFu;
}

fn read_freq(fc_offset: u32, sym: u32) -> u32 {
    return tables[fc_offset + sym] & 0xFFFFu;
}

fn read_cum(fc_offset: u32, sym: u32) -> u32 {
    return tables[fc_offset + sym] >> 16u;
}

fn read_word(word_index: u32) -> u32 {
    let u32_idx = word_index / 2u;
    let half = word_index % 2u;
    return (word_data[u32_idx] >> (half * 16u)) & 0xFFFFu;
}

fn write_output_byte(pos: u32, value: u32) {
    let word_idx = pos / 4u;
    let byte_in_word = pos % 4u;
    atomicOr(&output[word_idx], value << (byte_in_word * 8u));
}

@compute @workgroup_size(64)
fn rans_decode_blocks(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let block_id = wgid.x;
    let num_blocks = params.x;
    let scale_bits = params.y;
    let table_size = params.z;
    let bm_count = params.w;

    if (block_id >= num_blocks) {
        return;
    }

    let scale_mask = table_size - 1u;
    let cum2sym_u32_count = table_size / 4u;
    let fc_offset = cum2sym_u32_count;

    // Cooperatively load cum2sym into workgroup memory
    let load_count = min(cum2sym_u32_count, 4096u);
    for (var i = lid.x; i < load_count; i = i + 64u) {
        local_cum2sym[i] = tables[i];
    }
    workgroupBarrier();

    // Read block metadata
    let bm_base = block_id * 5u;
    let out_offset = metadata[bm_base];
    let out_len = metadata[bm_base + 1u];
    let lanes_per_block = metadata[bm_base + 2u];
    let lm_base_offset = metadata[bm_base + 4u];

    let lane_id = lid.x;
    if (lane_id >= lanes_per_block) {
        return;
    }

    // Read per-lane metadata (after block_meta region)
    let lm_idx = bm_count + (lm_base_offset + lane_id) * 3u;
    var state = metadata[lm_idx];
    let num_words = metadata[lm_idx + 1u];
    let word_offset = metadata[lm_idx + 2u];

    var word_pos = 0u;

    // Symbols this lane decodes (round-robin within block)
    let num_symbols = (out_len + lanes_per_block - 1u - lane_id) / lanes_per_block;

    for (var sym_idx = 0u; sym_idx < num_symbols; sym_idx = sym_idx + 1u) {
        let slot = state & scale_mask;
        let sym = read_cum2sym_local(slot);
        let freq = read_freq(fc_offset, sym);
        let cum = read_cum(fc_offset, sym);

        // Write symbol to round-robin output position within this block
        let out_pos = out_offset + sym_idx * lanes_per_block + lane_id;
        write_output_byte(out_pos, sym);

        // Advance state (wrapping u32 mul)
        state = freq * (state >> scale_bits) + slot - cum;

        // Renormalize
        if (state < RANS_L && word_pos < num_words) {
            state = (state << IO_BITS) | read_word(word_offset + word_pos);
            word_pos = word_pos + 1u;
        }
    }
}
