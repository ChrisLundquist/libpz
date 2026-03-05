// Parallel-parse LZ conflict resolution kernel for Experiment E.
//
// Takes per-position match data (from existing coop match finder) and resolves
// overlapping matches using a forward max-propagation scan (prefix-max).
//
// Three entry points:
//   1. init_coverage  — build coverage array from match lengths
//   2. prefix_max_local — per-workgroup prefix-max scan (Blelloch-style)
//   3. prefix_max_propagate — propagate block maxima across workgroups
//   4. classify — classify each position as match-start, literal, or covered

// Match data: packed as u32 per position. High 16 bits = offset, low 16 bits = length.
// Length == 0 means no match at this position.
@group(0) @binding(0) var<storage, read> matches: array<u32>;

// Coverage array: coverage[p] = p + length (if match) or p (if no match).
// After prefix-max scan: coverage[p] = max coverage of any position <= p.
@group(0) @binding(1) var<storage, read_write> coverage: array<u32>;

// Block sums for multi-level prefix-max scan.
@group(0) @binding(2) var<storage, read_write> block_maxima: array<u32>;

// Output: classification bits. 1 = match start, 0 = literal or covered.
// Packed as u32 flags (one bit per position).
@group(0) @binding(3) var<storage, read_write> flags: array<atomic<u32>>;

// Parameters: [n, workgroup_size, 0, 0]
@group(0) @binding(4) var<uniform> params: vec4<u32>;

var<workgroup> shared_data: array<u32, 512>;

// Initialize coverage array from match lengths.
@compute @workgroup_size(256)
fn init_coverage(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.x;
    let pos = gid.x + gid.y * 65535u * 256u;
    if pos >= n {
        return;
    }

    let match_val = matches[pos];
    let length = match_val & 0xFFFFu;

    if length > 0u {
        coverage[pos] = pos + length;
    } else {
        coverage[pos] = pos;
    }
}

// Per-workgroup inclusive prefix-max scan (Blelloch up-sweep + down-sweep with max).
@compute @workgroup_size(256)
fn prefix_max_local(@builtin(local_invocation_id) lid: vec3<u32>,
                    @builtin(workgroup_id) wgid: vec3<u32>) {
    let n = params.x;
    let wg_size = 256u;
    let block_start = (wgid.x + wgid.y * 65535u) * wg_size;
    let tid = lid.x;
    let global_idx = block_start + tid;

    // Load into shared memory.
    if global_idx < n {
        shared_data[tid] = coverage[global_idx];
    } else {
        shared_data[tid] = 0u;
    }
    workgroupBarrier();

    // Up-sweep (reduce) phase.
    for (var stride: u32 = 1u; stride < wg_size; stride = stride * 2u) {
        let idx = (tid + 1u) * stride * 2u - 1u;
        if idx < wg_size {
            shared_data[idx] = max(shared_data[idx], shared_data[idx - stride]);
        }
        workgroupBarrier();
    }

    // Down-sweep phase for inclusive prefix-max.
    for (var stride: u32 = wg_size / 4u; stride >= 1u; stride = stride / 2u) {
        let idx = (tid + 1u) * stride * 2u - 1u + stride;
        if idx < wg_size {
            shared_data[idx] = max(shared_data[idx], shared_data[idx - stride]);
        }
        workgroupBarrier();
    }

    // Write back.
    if global_idx < n {
        coverage[global_idx] = shared_data[tid];
    }

    // Last thread writes block maximum for propagation.
    if tid == wg_size - 1u {
        let block_idx = wgid.x + wgid.y * 65535u;
        let last_valid = min(block_start + wg_size - 1u, n - 1u);
        block_maxima[block_idx] = coverage[last_valid];
    }
}

// Propagate block maxima across workgroups.
// Each workgroup adds the prefix-max of all preceding blocks to its elements.
@compute @workgroup_size(256)
fn prefix_max_propagate(@builtin(global_invocation_id) gid: vec3<u32>,
                        @builtin(workgroup_id) wgid: vec3<u32>) {
    let n = params.x;
    let wg_size = 256u;
    let block_idx = wgid.x + wgid.y * 65535u;

    // First block doesn't need propagation.
    if block_idx == 0u {
        return;
    }

    let global_idx = gid.x + gid.y * 65535u * 256u;
    if global_idx >= n {
        return;
    }

    // Accumulate max from all preceding blocks.
    // For small block counts, a serial scan over block_maxima is fine.
    var prev_max: u32 = 0u;
    for (var b: u32 = 0u; b < block_idx; b = b + 1u) {
        prev_max = max(prev_max, block_maxima[b]);
    }

    coverage[global_idx] = max(coverage[global_idx], prev_max);
}

// Classify each position based on coverage array.
// Sets flag bit to 1 for match-start positions, 0 otherwise.
@compute @workgroup_size(256)
fn classify(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.x;
    let pos = gid.x + gid.y * 65535u * 256u;
    if pos >= n {
        return;
    }

    let match_val = matches[pos];
    let length = match_val & 0xFFFFu;

    var is_match_start: bool = false;
    if length > 0u {
        if pos == 0u {
            is_match_start = true;
        } else {
            // Not covered by an earlier match if coverage[pos-1] <= pos.
            is_match_start = coverage[pos - 1u] <= pos;
        }
    }

    // Pack into u32 flags (one bit per position).
    if is_match_start {
        let word_idx = pos / 32u;
        let bit_idx = pos % 32u;
        atomicOr(&flags[word_idx], 1u << bit_idx);
    }
}
