// GPU-accelerated rank assignment for BWT prefix-doubling.
//
// Three phases:
//   1. Compare: diff[i] = (key(sa[i]) != key(sa[i-1])) ? 1 : 0
//   2. Prefix sum: inclusive scan of diff[], giving each suffix its rank
//   3. Scatter: new_rank[sa[i]] = prefix[i]

// WORKGROUP_SIZE is set via pipeline constant override
override WORKGROUP_SIZE: u32 = 256u;
const BLOCK_ELEMS_MULT: u32 = 2u; // BLOCK_ELEMS = WORKGROUP_SIZE * 2

@group(0) @binding(0) var<storage, read> sa: array<u32>;
@group(0) @binding(1) var<storage, read> rank: array<u32>;
@group(0) @binding(2) var<storage, read_write> diff: array<u32>;
@group(0) @binding(3) var<uniform> rank_params: vec4<u32>; // x=n, y=padded_n, z=k

@compute @workgroup_size(256)
fn rank_compare(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = rank_params.x;
    let padded_n = rank_params.y;
    let k = rank_params.z;
    if (i >= padded_n) {
        return;
    }

    if (i == 0u) {
        diff[0] = 0u;
        return;
    }

    let curr_sa = sa[i];
    let prev_sa = sa[i - 1u];

    if (curr_sa >= n) {
        diff[i] = 0u;
        return;
    }

    if (prev_sa >= n) {
        diff[i] = 1u;
        return;
    }

    let r1_curr = rank[curr_sa];
    let r2_curr = rank[(curr_sa + k) % n];
    let r1_prev = rank[prev_sa];
    let r2_prev = rank[(prev_sa + k) % n];

    if (r1_curr != r1_prev || r2_curr != r2_prev) {
        diff[i] = 1u;
    } else {
        diff[i] = 0u;
    }
}

// Prefix sum: per-workgroup inclusive Blelloch scan
@group(0) @binding(0) var<storage, read> ps_input: array<u32>;
@group(0) @binding(1) var<storage, read_write> ps_output: array<u32>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(3) var<uniform> ps_params: vec4<u32>; // x=count

var<workgroup> temp: array<u32, 512>; // WORKGROUP_SIZE * 2

@compute @workgroup_size(256)
fn prefix_sum_local(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let tid = lid.x;
    let block_id = wgid.x;
    let block_elems = WORKGROUP_SIZE * BLOCK_ELEMS_MULT;
    let base = block_id * block_elems;
    let count = ps_params.x;

    let ai = tid;
    let bi = tid + WORKGROUP_SIZE;

    var val_a = 0u;
    var val_b = 0u;
    if (base + ai < count) {
        val_a = ps_input[base + ai];
    }
    if (base + bi < count) {
        val_b = ps_input[base + bi];
    }
    temp[ai] = val_a;
    temp[bi] = val_b;

    // Up-sweep (reduce)
    var offset = 1u;
    var d = block_elems >> 1u;
    loop {
        if (d == 0u) {
            break;
        }
        workgroupBarrier();
        if (tid < d) {
            let ai_idx = offset * (2u * tid + 1u) - 1u;
            let bi_idx = offset * (2u * tid + 2u) - 1u;
            temp[bi_idx] = temp[bi_idx] + temp[ai_idx];
        }
        offset = offset << 1u;
        d = d >> 1u;
    }

    // Save block total and clear last
    if (tid == 0u) {
        block_sums[block_id] = temp[block_elems - 1u];
        temp[block_elems - 1u] = 0u;
    }

    // Down-sweep
    var d2 = 1u;
    loop {
        if (d2 >= block_elems) {
            break;
        }
        offset = offset >> 1u;
        workgroupBarrier();
        if (tid < d2) {
            let ai_idx = offset * (2u * tid + 1u) - 1u;
            let bi_idx = offset * (2u * tid + 2u) - 1u;
            let t = temp[ai_idx];
            temp[ai_idx] = temp[bi_idx];
            temp[bi_idx] = temp[bi_idx] + t;
        }
        d2 = d2 << 1u;
    }
    workgroupBarrier();

    // Convert exclusive to inclusive by adding original values
    if (base + ai < count) {
        ps_output[base + ai] = temp[ai] + val_a;
    }
    if (base + bi < count) {
        ps_output[base + bi] = temp[bi] + val_b;
    }
}

// Propagate block offsets
@group(0) @binding(0) var<storage, read_write> prop_data: array<u32>;
@group(0) @binding(1) var<storage, read> block_offsets: array<u32>;
@group(0) @binding(2) var<uniform> prop_params: vec4<u32>; // x=count

@compute @workgroup_size(256)
fn prefix_sum_propagate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    let count = prop_params.x;
    if (g >= count) {
        return;
    }

    let block_elems = WORKGROUP_SIZE * BLOCK_ELEMS_MULT;
    let block_id = g / block_elems;
    if (block_id > 0u) {
        prop_data[g] = prop_data[g] + block_offsets[block_id - 1u];
    }
}

// Scatter ranks
@group(0) @binding(0) var<storage, read> scatter_sa: array<u32>;
@group(0) @binding(1) var<storage, read> scatter_prefix: array<u32>;
@group(0) @binding(2) var<storage, read_write> new_rank: array<u32>;
@group(0) @binding(3) var<uniform> scatter_params: vec4<u32>; // x=n, y=padded_n

@compute @workgroup_size(256)
fn rank_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = scatter_params.x;
    let padded_n = scatter_params.y;
    if (i >= padded_n) {
        return;
    }

    let sa_i = scatter_sa[i];
    if (sa_i < n) {
        new_rank[sa_i] = scatter_prefix[i];
    } else {
        new_rank[sa_i] = 0xFFFFFFFFu;
    }
}
