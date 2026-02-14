// GPU LZ77 block-parallel decompression kernel (WGSL).
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
// Block metadata layout (per block, 3 x u32):
//   [0] match_data_offset  — byte offset into the match data buffer
//   [1] num_matches        — number of 5-byte match records in this block
//   [2] output_offset      — byte offset into the output buffer
//
// Output uses atomic u32 with atomicOr for sub-word byte writes and
// atomicLoad for reads, since multiple threads in the same workgroup may
// write/read adjacent bytes packed in the same u32 word.

// @pz_cost {
//   threads_per_element: 0.004
//   passes: 1
//   buffers: match_data=N*5, block_meta=N*0.001, output=N
//   local_mem: 12
//   note: one workgroup (64 threads) per block. Sequential match parsing with cooperative back-ref copy.
// }

const MATCH_SIZE: u32 = 5u;
const WG_SIZE: u32 = 64u;

// Match data: all blocks' serialized matches concatenated, packed as u32
@group(0) @binding(0) var<storage, read> match_data: array<u32>;

// Block metadata: 3 u32 per block [data_offset, num_matches, output_offset]
@group(0) @binding(1) var<storage, read> block_meta: array<u32>;

// Output buffer (atomic for safe sub-word byte writes from multiple threads)
@group(0) @binding(2) var<storage, read_write> output: array<atomic<u32>>;

// Params: x=num_blocks, y=total_output_len
@group(0) @binding(3) var<uniform> params: vec4<u32>;

// Shared memory for the leader to communicate match info to all threads.
// [0] = copy_src_start, [1] = copy_len, [2] = write_pos (before this match)
var<workgroup> match_info: array<u32, 3>;

fn read_match_byte(byte_offset: u32) -> u32 {
    let word_idx = byte_offset / 4u;
    let byte_in_word = byte_offset % 4u;
    return (match_data[word_idx] >> (byte_in_word * 8u)) & 0xFFu;
}

fn read_output_byte(byte_pos: u32) -> u32 {
    let word_idx = byte_pos / 4u;
    let byte_in_word = byte_pos % 4u;
    return (atomicLoad(&output[word_idx]) >> (byte_in_word * 8u)) & 0xFFu;
}

fn write_output_byte(byte_pos: u32, value: u32) {
    let word_idx = byte_pos / 4u;
    let byte_in_word = byte_pos % 4u;
    atomicOr(&output[word_idx], value << (byte_in_word * 8u));
}

@compute @workgroup_size(64)
fn lz77_decode(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let block_id = wgid.x;
    let num_blocks = params.x;
    let total_output_len = params.y;

    if (block_id >= num_blocks) {
        return;
    }

    let thread_id = lid.x;

    // Read block metadata
    let meta_base = block_id * 3u;
    let data_off = block_meta[meta_base];
    let num_matches = block_meta[meta_base + 1u];
    let out_base = block_meta[meta_base + 2u];

    var write_pos = 0u;

    for (var m = 0u; m < num_matches; m = m + 1u) {
        // Leader reads the match
        if (thread_id == 0u) {
            let base = data_off + m * MATCH_SIZE;
            let offset = read_match_byte(base) | (read_match_byte(base + 1u) << 8u);
            var length = read_match_byte(base + 2u) | (read_match_byte(base + 3u) << 8u);

            // Compute source position for back-reference copy
            var src_start = 0u;
            if (offset > 0u && length > 0u && offset <= write_pos) {
                src_start = write_pos - offset;
            } else {
                length = 0u; // no valid back-reference
            }

            match_info[0] = src_start;
            match_info[1] = length;
            match_info[2] = write_pos;
        }
        workgroupBarrier();

        let src_start = match_info[0];
        let copy_len = match_info[1];
        let wp = match_info[2];

        // Cooperative copy of back-referenced bytes.
        // LZ77 back-references can overlap (offset < length), so we
        // must handle that case with modular indexing.
        if (copy_len > 0u) {
            let offset_val = wp - src_start; // = original offset
            for (var i = thread_id; i < copy_len; i = i + WG_SIZE) {
                var src_idx: u32;
                if (offset_val >= copy_len) {
                    // Non-overlapping: direct copy
                    src_idx = src_start + i;
                } else {
                    // Overlapping: use modular indexing for repeating pattern
                    src_idx = src_start + (i % offset_val);
                }
                let dst = out_base + wp + i;
                if (dst < total_output_len) {
                    let byte_val = read_output_byte(out_base + src_idx);
                    write_output_byte(dst, byte_val);
                }
            }
        }
        workgroupBarrier();

        // Leader appends the literal byte and advances write_pos
        if (thread_id == 0u) {
            write_pos = wp + copy_len;
            let dst = out_base + write_pos;
            if (dst < total_output_len) {
                let base = data_off + m * MATCH_SIZE;
                let literal = read_match_byte(base + 4u);
                write_output_byte(dst, literal);
            }
            write_pos = write_pos + 1u;
        }
        workgroupBarrier();

        // Sync write_pos for next iteration
        if (thread_id == 0u) {
            match_info[2] = write_pos;
        }
        workgroupBarrier();
        write_pos = match_info[2];
    }
}
