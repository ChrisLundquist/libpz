// Bit-plane transpose kernel for Experiment D (GPU throughput ceiling).
//
// Transposes N bytes into 8 bit-plane streams. Each workgroup processes
// a 256-byte tile: each thread reads one input byte, extracts 8 bits,
// and writes each bit into the corresponding plane's output buffer.
//
// This kernel has zero serial dependencies and zero data-dependent branching,
// making it a pure throughput benchmark.

// Input bytes (u32-packed).
@group(0) @binding(0) var<storage, read> input_data: array<u32>;

// Output: 8 bit-plane streams packed contiguously as u32s.
// Layout: plane 0 occupies u32s [0..plane_u32s), plane 1 at [plane_u32s..2*plane_u32s), etc.
@group(0) @binding(1) var<storage, read_write> output_planes: array<atomic<u32>>;

// Parameters: [input_len, plane_bytes, 0, 0]
@group(0) @binding(2) var<uniform> params: vec4<u32>;

@compute @workgroup_size(256)
fn bitplane_transpose(@builtin(global_invocation_id) gid: vec3<u32>,
                      @builtin(workgroup_id) wgid: vec3<u32>) {
    let n = params.x;
    let plane_bytes = params.y;
    let plane_u32s = (plane_bytes + 3u) / 4u;

    let global_pos = gid.x + gid.y * 65535u * 256u;

    if global_pos >= n {
        return;
    }

    // Read one input byte.
    let word_idx = global_pos / 4u;
    let byte_off = global_pos % 4u;
    let byte_val = (input_data[word_idx] >> (byte_off * 8u)) & 0xFFu;

    // Position in plane output: which byte and which bit within that byte.
    let out_byte_idx = global_pos / 8u;
    let out_bit_pos = 7u - (global_pos % 8u);  // MSB-first

    // Which u32 in the plane output and which bit within it.
    let out_word_idx = out_byte_idx / 4u;
    let out_byte_in_word = out_byte_idx % 4u;
    let bit_in_word = out_bit_pos + out_byte_in_word * 8u;

    // For each of 8 bit planes, if the corresponding bit is set in byte_val,
    // atomically OR the bit into the output plane.
    for (var bit: u32 = 0u; bit < 8u; bit = bit + 1u) {
        if (byte_val & (1u << (7u - bit))) != 0u {
            let plane_offset = bit * plane_u32s + out_word_idx;
            atomicOr(&output_planes[plane_offset], 1u << bit_in_word);
        }
    }
}
