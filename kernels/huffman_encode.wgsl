// GPU Huffman encoding kernel (WGSL).
//
// Two-pass approach:
//   Pass 1 (compute_bit_lengths): Each invocation looks up its symbol's code
//     length from the lookup table.
//   Pass 2 (write_codes): After prefix sum produces bit offsets, each invocation
//     writes its codeword at the computed offset.

// @pz_cost {
//   threads_per_element: 1
//   passes: 2
//   buffers: input=N, lut=1024, bit_lengths=N*4, offsets=N*4, output=N
//   local_mem: 0
// }

@group(0) @binding(0) var<storage, read> symbols: array<u32>; // packed bytes
@group(0) @binding(1) var<storage, read> code_lut: array<u32>; // 256 entries
@group(0) @binding(2) var<storage, read_write> bit_lengths: array<u32>;
@group(0) @binding(3) var<uniform> huff_params: vec4<u32>; // x=num_symbols

fn read_symbol(pos: u32) -> u32 {
    let word_idx = pos / 4u;
    let byte_idx = pos % 4u;
    return (symbols[word_idx] >> (byte_idx * 8u)) & 0xFFu;
}

@compute @workgroup_size(64)
fn compute_bit_lengths(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x + gid.y * huff_params.w;
    let num_symbols = huff_params.x;
    if (g >= num_symbols) {
        return;
    }

    let sym = read_symbol(g);
    let entry = code_lut[sym];
    let bits = entry >> 24u;
    bit_lengths[g] = bits;
}

// Pass 2: write codewords at pre-computed bit offsets
@group(0) @binding(0) var<storage, read> wc_symbols: array<u32>;
@group(0) @binding(1) var<storage, read> wc_code_lut: array<u32>;
@group(0) @binding(2) var<storage, read> bit_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> wc_output: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> wc_params: vec4<u32>; // x=num_symbols

fn read_wc_symbol(pos: u32) -> u32 {
    let word_idx = pos / 4u;
    let byte_idx = pos % 4u;
    return (wc_symbols[word_idx] >> (byte_idx * 8u)) & 0xFFu;
}

// Chunk-based packing: each thread owns exclusive output u32 words.
// CHUNK_WORDS: each thread covers this many output u32 words exclusively.
// Set to 1 so each thread owns exactly one output word, eliminating intra-word contention.
const CHUNK_WORDS: u32 = 1u;

@compute @workgroup_size(64)
fn write_codes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x + gid.y * wc_params.w;
    let num_output_words = (wc_params.y + 31u) / 32u; // total output u32 words
    let num_symbols = wc_params.x;

    if (g >= num_output_words) {
        return;
    }

    // Bit range this thread owns: [g*32, (g+1)*32)
    let my_bit_start = g * 32u;
    let my_bit_end = my_bit_start + 32u;

    var local_word: u32 = 0u;
    var has_boundary_low = false;  // straddles into next word

    // Scan all symbols to find those whose bits intersect [my_bit_start, my_bit_end).
    // This is O(N) per thread, giving O(N*W) total work where W = num_output_words.
    // For blocks where N/W is large (dense codewords, typical), this is efficient.
    // Optimization: use bit_offsets[] to binary-search for the first symbol
    // in range, then scan forward while bit_offsets[sym] < my_bit_end.
    // For the initial implementation, use a linear scan bounded by num_symbols.

    var sym_idx: u32 = 0u;
    loop {
        if (sym_idx >= num_symbols) { break; }

        let start_bit = bit_offsets[sym_idx];
        if (start_bit >= my_bit_end) {
            // All remaining symbols start at or after our range; done.
            break;
        }

        let sym = read_wc_symbol(sym_idx);
        let entry = wc_code_lut[sym];
        let bits = entry >> 24u;
        let codeword = entry & 0x00FFFFFFu;

        if (bits > 0u) {
            let end_bit = start_bit + bits - 1u;

            if (end_bit >= my_bit_start && start_bit < my_bit_end) {
                // Symbol intersects our range.
                let first_word = start_bit / 32u;
                let last_word = end_bit / 32u;

                if (first_word == last_word) {
                    // Entire codeword fits within one word — it must be ours.
                    let first_shift = 31u - (start_bit % 32u);
                    let shifted = codeword << (first_shift - (bits - 1u));
                    local_word = local_word | shifted;
                } else {
                    // Boundary symbol.
                    if (first_word == g) {
                        // We own the high part (bits that land in our word).
                        let bits_in_first = 32u - (start_bit % 32u);
                        let high_part = codeword >> (bits - bits_in_first);
                        // Place high_part at the MSB side of our word
                        let shift = 31u - (start_bit % 32u);
                        local_word = local_word | (high_part << (shift - (bits_in_first - 1u)));
                    }
                    if (last_word == g) {
                        // We own the low part (bits that overhang into our word).
                        let remaining = bits - (32u - (start_bit % 32u));
                        let low_part = (codeword << (32u - remaining)) & 0xFFFFFFFFu;
                        atomicOr(&wc_output[g], low_part);
                        // Mark that we issued an atomic — we must not stomp it later.
                        has_boundary_low = true;
                    }
                }
            }
        }

        sym_idx = sym_idx + 1u;
    }

    // Write the accumulated local word.
    // If we only have non-boundary bits, a plain store suffices and avoids
    // all atomic overhead for the common case.
    if (!has_boundary_low) {
        wc_output[g] = local_word;
    } else {
        // Merge local_word with the already-atomically-written boundary bits.
        atomicOr(&wc_output[g], local_word);
    }
}

// Histogram kernel: count byte frequencies
@group(0) @binding(0) var<storage, read> hist_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> histogram: array<atomic<u32>>; // 256 entries
@group(0) @binding(2) var<uniform> hist_params: vec4<u32>; // x=data_len

fn read_hist_byte(pos: u32) -> u32 {
    let word_idx = pos / 4u;
    let byte_idx = pos % 4u;
    return (hist_data[word_idx] >> (byte_idx * 8u)) & 0xFFu;
}

@compute @workgroup_size(64)
fn byte_histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x + gid.y * hist_params.w;
    let data_len = hist_params.x;
    if (g >= data_len) {
        return;
    }

    let byte_val = read_hist_byte(g);
    atomicAdd(&histogram[byte_val], 1u);
}

// Block-level exclusive prefix sum (Blelloch scan)
@group(0) @binding(0) var<storage, read_write> ps_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> ps_block_sums: array<u32>;
@group(0) @binding(2) var<uniform> ps_params: vec4<u32>; // x=n

var<workgroup> ps_temp: array<u32, 512>; // 256 * 2

@compute @workgroup_size(256)
fn prefix_sum_block(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let local_size = 256u;
    let block_size = local_size * 2u;
    let l = lid.x;
    let group_id = wgid.x;
    let n = ps_params.x;

    let ai = l;
    let bi = l + local_size;
    let ga = group_id * block_size + ai;
    let gb = group_id * block_size + bi;

    var va = 0u;
    var vb = 0u;
    if (ga < n) {
        va = ps_data[ga];
    }
    if (gb < n) {
        vb = ps_data[gb];
    }
    ps_temp[ai] = va;
    ps_temp[bi] = vb;

    // Up-sweep
    var offset = 1u;
    var d = block_size >> 1u;
    loop {
        if (d == 0u) {
            break;
        }
        workgroupBarrier();
        if (l < d) {
            let ai2 = offset * (2u * l + 1u) - 1u;
            let bi2 = offset * (2u * l + 2u) - 1u;
            ps_temp[bi2] = ps_temp[bi2] + ps_temp[ai2];
        }
        offset = offset << 1u;
        d = d >> 1u;
    }

    workgroupBarrier();
    if (l == 0u) {
        ps_block_sums[group_id] = ps_temp[block_size - 1u];
        ps_temp[block_size - 1u] = 0u;
    }

    // Down-sweep
    var d2 = 1u;
    loop {
        if (d2 >= block_size) {
            break;
        }
        offset = offset >> 1u;
        workgroupBarrier();
        if (l < d2) {
            let ai2 = offset * (2u * l + 1u) - 1u;
            let bi2 = offset * (2u * l + 2u) - 1u;
            let t = ps_temp[ai2];
            ps_temp[ai2] = ps_temp[bi2];
            ps_temp[bi2] = ps_temp[bi2] + t;
        }
        d2 = d2 << 1u;
    }

    workgroupBarrier();
    if (ga < n) {
        ps_data[ga] = ps_temp[ai];
    }
    if (gb < n) {
        ps_data[gb] = ps_temp[bi];
    }
}

// Apply block offsets
@group(0) @binding(0) var<storage, read_write> psa_data: array<u32>;
@group(0) @binding(1) var<storage, read> psa_block_sums: array<u32>;
@group(0) @binding(2) var<uniform> psa_params: vec4<u32>; // x=n, y=block_size

@compute @workgroup_size(256)
fn prefix_sum_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    let n = psa_params.x;
    let block_size = psa_params.y;
    if (g >= n) {
        return;
    }

    let block_id = g / block_size;
    if (block_id > 0u) {
        psa_data[g] = psa_data[g] + psa_block_sums[block_id];
    }
}
