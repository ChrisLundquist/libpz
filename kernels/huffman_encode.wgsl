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

@compute @workgroup_size(64)
fn write_codes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x + gid.y * wc_params.w;
    let num_symbols = wc_params.x;
    if (g >= num_symbols) {
        return;
    }

    let sym = read_wc_symbol(g);
    let entry = wc_code_lut[sym];
    let bits = entry >> 24u;
    let codeword = entry & 0x00FFFFFFu;
    let start_bit = bit_offsets[g];

    if (bits == 0u) {
        return;
    }

    // Write all bits of the codeword using at most 2 atomicOr ops.
    // Bits are stored MSB-first: start_bit is the position of the MSB.
    let end_bit = start_bit + bits - 1u;
    let first_word = start_bit / 32u;
    let last_word = end_bit / 32u;

    // Position of MSB within the first u32 word (bit 31 = leftmost)
    let first_shift = 31u - (start_bit % 32u);
    // Reverse the codeword so MSB is at position `first_shift`
    // codeword has `bits` valid bits in the low end, MSB at bit (bits-1)
    let reversed = codeword << (first_shift - (bits - 1u));

    if (first_word == last_word) {
        // All bits fit in a single u32 word
        atomicOr(&wc_output[first_word], reversed);
    } else {
        // Bits span two u32 words
        let bits_in_first = first_shift + 1u;
        // High part: top `bits_in_first` bits go into first_word
        let high_mask = codeword >> (bits - bits_in_first);
        atomicOr(&wc_output[first_word], high_mask);
        // Low part: remaining bits go into last_word, aligned to MSB
        let remaining = bits - bits_in_first;
        let low_mask = (codeword << (32u - remaining)) & 0xFFFFFFFFu;
        atomicOr(&wc_output[last_word], low_mask);
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
