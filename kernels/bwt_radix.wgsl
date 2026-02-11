// GPU-accelerated radix sort for BWT prefix-doubling.
//
// LSB-first 8-bit radix sort. Each pass sorts by one byte of the
// 64-bit composite key: key = (rank[sa[i]] << 32) | rank[(sa[i]+k) % n]

const RADIX: u32 = 256u;

@group(0) @binding(0) var<storage, read> rk_sa: array<u32>;
@group(0) @binding(1) var<storage, read> rk_rank: array<u32>;
@group(0) @binding(2) var<storage, read_write> rk_keys: array<u32>;
@group(0) @binding(3) var<uniform> rk_params: vec4<u32>; // x=n, y=padded_n, z=k, w=pass

@compute @workgroup_size(256)
fn radix_compute_keys(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = rk_params.x;
    let padded_n = rk_params.y;
    let k = rk_params.z;
    let pass = rk_params.w;
    if (i >= padded_n) {
        return;
    }

    let sa_i = rk_sa[i];
    if (sa_i >= n) {
        rk_keys[i] = 0xFFu;
        return;
    }

    let r1 = rk_rank[sa_i];
    let r2 = rk_rank[(sa_i + k) % n];

    var word = r2;
    if (pass >= 4u) {
        word = r1;
    }
    let shift = (pass & 3u) * 8u;
    rk_keys[i] = (word >> shift) & 0xFFu;
}

// Per-workgroup histogram of radix digits
@group(0) @binding(0) var<storage, read> hist_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> histograms: array<u32>;
@group(0) @binding(2) var<uniform> hist_params: vec4<u32>; // x=padded_n, y=num_groups

var<workgroup> local_hist: array<atomic<u32>, 256>;

@compute @workgroup_size(256)
fn radix_histogram(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let tid = lid.x;
    let group_id = wgid.x;
    let group_size = 256u; // workgroup_size
    let padded_n = hist_params.x;
    let num_groups = hist_params.y;

    // Clear local histogram
    if (tid < RADIX) {
        atomicStore(&local_hist[tid], 0u);
    }
    workgroupBarrier();

    // Count digits in this workgroup's tile
    let idx = gid.x;
    if (idx < padded_n) {
        let digit = hist_keys[idx];
        atomicAdd(&local_hist[digit], 1u);
    }
    workgroupBarrier();

    // Write local histogram to global memory (column-major)
    if (tid < RADIX) {
        histograms[tid * num_groups + group_id] = atomicLoad(&local_hist[tid]);
    }
}

// Convert inclusive prefix sum to exclusive
@group(0) @binding(0) var<storage, read> inclusive: array<u32>;
@group(0) @binding(1) var<storage, read_write> exclusive: array<u32>;
@group(0) @binding(2) var<uniform> ite_params: vec4<u32>; // x=count

@compute @workgroup_size(256)
fn inclusive_to_exclusive(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let count = ite_params.x;
    if (i >= count) {
        return;
    }
    if (i == 0u) {
        exclusive[i] = 0u;
    } else {
        exclusive[i] = inclusive[i - 1u];
    }
}

// Stable scatter: thread 0 of each workgroup sequentially scatters
@group(0) @binding(0) var<storage, read> scat_sa_in: array<u32>;
@group(0) @binding(1) var<storage, read> scat_keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> global_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> scat_sa_out: array<u32>;
@group(0) @binding(4) var<uniform> scat_params: vec4<u32>; // x=padded_n, y=num_groups

@compute @workgroup_size(256)
fn radix_scatter(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let tid = lid.x;
    if (tid != 0u) {
        return;
    }

    let group_id = wgid.x;
    let group_size = 256u;
    let padded_n = scat_params.x;
    let num_groups = scat_params.y;
    let base = group_id * group_size;
    var end = base + group_size;
    if (end > padded_n) {
        end = padded_n;
    }

    for (var i = base; i < end; i = i + 1u) {
        let digit = scat_keys[i];
        let pos = global_offsets[digit * num_groups + group_id];
        scat_sa_out[pos] = scat_sa_in[i];
        global_offsets[digit * num_groups + group_id] = pos + 1u;
    }
}
