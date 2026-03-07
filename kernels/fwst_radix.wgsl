// GPU key extraction for FWST radix sort.
//
// Like BWT radix sort, but extracts keys directly from input bytes
// instead of rank arrays. For SA entry sa[i], key = input[(sa[i] + pass) % n].

const RADIX: u32 = 256u;

@group(0) @binding(0) var<storage, read> fk_sa: array<u32>;
@group(0) @binding(1) var<storage, read> fk_input: array<u32>; // packed u8s as u32s
@group(0) @binding(2) var<storage, read_write> fk_keys: array<u32>;
@group(0) @binding(3) var<uniform> fk_params: vec4<u32>; // x=n, y=padded_n, z=pass_idx, w=unused

@compute @workgroup_size(256)
fn fwst_compute_keys(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = fk_params.x;
    let padded_n = fk_params.y;
    let pass_idx = fk_params.z;

    if (i >= padded_n) {
        return;
    }

    let sa_i = fk_sa[i];
    if (sa_i >= n) {
        fk_keys[i] = 0xFFu;
        return;
    }

    // Key = input[(sa_i + pass_idx) % n], reading from packed u32 array.
    let byte_idx = (sa_i + pass_idx) % n;
    let word_idx = byte_idx / 4u;
    let byte_offset = byte_idx % 4u;
    let word = fk_input[word_idx];
    fk_keys[i] = (word >> (byte_offset * 8u)) & 0xFFu;
}
