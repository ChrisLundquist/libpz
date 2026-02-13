/// GPU-accelerated rank assignment for BWT prefix-doubling.
///
/// Replaces the CPU sequential scan with three parallel phases:
///   1. Compare: diff[i] = (key(sa[i]) != key(sa[i-1])) ? 1 : 0
///   2. Prefix sum: inclusive scan of diff[], giving each suffix its rank
///   3. Scatter: new_rank[sa[i]] = prefix[i]  (position-indexed output)
///
/// Compiled with -DWORKGROUP_SIZE=N (power of 2, typically 256).

// @pz_cost {
//   threads_per_element: 1
//   passes: 4
//   buffers: sa=N*4, rank=N*4, diff=N*4, prefix=N*4
//   local_mem: 2048
//   note: called per doubling step (log2(N) steps)
// }

/// Phase 1: Parallel comparison of consecutive composite keys.
///
/// For each sorted position i, compare the composite key of sa[i] against
/// sa[i-1]. Output diff[i] = 1 if they differ (new rank), 0 if same.
/// The inclusive prefix sum of diff[] gives the rank of each suffix.
///
/// Sentinel entries (sa >= n) produce diff=0 and are ignored by scatter.
__kernel void rank_compare(
    __global const uint *sa,
    __global const uint *rank,
    __global uint *diff,
    const uint n,
    const uint padded_n,
    const uint k
) {
    uint i = get_global_id(0);
    if (i >= padded_n) return;

    if (i == 0) {
        diff[0] = 0;
        return;
    }

    uint curr_sa = sa[i];
    uint prev_sa = sa[i - 1];

    // Current is sentinel: diff irrelevant, scatter will skip it
    if (curr_sa >= n) {
        diff[i] = 0;
        return;
    }

    // Previous was sentinel, current is real: always a new rank group
    if (prev_sa >= n) {
        diff[i] = 1;
        return;
    }

    // Both real: compare composite keys (rank[sa], rank[(sa+k) % n])
    uint r1_curr = rank[curr_sa];
    uint r2_curr = rank[(curr_sa + k) % n];
    uint r1_prev = rank[prev_sa];
    uint r2_prev = rank[(prev_sa + k) % n];

    diff[i] = (r1_curr != r1_prev || r2_curr != r2_prev) ? 1u : 0u;
}

/// Phase 2a: Per-workgroup inclusive prefix sum (Blelloch scan).
///
/// Each workgroup processes BLOCK_ELEMS = WORKGROUP_SIZE * 2 elements.
/// Reads from `input`, writes inclusive prefix sums to `output`, and
/// saves the per-workgroup total to `block_sums[group_id]`.
///
/// Uses __local memory for the cooperative scan within a workgroup.
#define BLOCK_ELEMS (WORKGROUP_SIZE * 2)

__kernel void prefix_sum_local(
    __global const uint *input,
    __global uint *output,
    __global uint *block_sums,
    const uint count
) {
    __local uint temp[BLOCK_ELEMS];

    uint tid = get_local_id(0);
    uint block_id = get_group_id(0);
    uint base = block_id * BLOCK_ELEMS;

    // Load two elements per work-item into local memory
    uint ai = tid;
    uint bi = tid + WORKGROUP_SIZE;
    uint val_a = (base + ai < count) ? input[base + ai] : 0;
    uint val_b = (base + bi < count) ? input[base + bi] : 0;
    temp[ai] = val_a;
    temp[bi] = val_b;

    // Up-sweep (reduce) phase
    uint offset = 1;
    for (uint d = BLOCK_ELEMS >> 1; d > 0; d >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < d) {
            uint ai_idx = offset * (2 * tid + 1) - 1;
            uint bi_idx = offset * (2 * tid + 2) - 1;
            temp[bi_idx] += temp[ai_idx];
        }
        offset <<= 1;
    }

    // Save block total and clear last element for down-sweep
    if (tid == 0) {
        block_sums[block_id] = temp[BLOCK_ELEMS - 1];
        temp[BLOCK_ELEMS - 1] = 0;
    }

    // Down-sweep phase (exclusive scan)
    for (uint d = 1; d < BLOCK_ELEMS; d <<= 1) {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < d) {
            uint ai_idx = offset * (2 * tid + 1) - 1;
            uint bi_idx = offset * (2 * tid + 2) - 1;
            uint t = temp[ai_idx];
            temp[ai_idx] = temp[bi_idx];
            temp[bi_idx] += t;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Convert exclusive scan to inclusive by adding original values
    if (base + ai < count) output[base + ai] = temp[ai] + val_a;
    if (base + bi < count) output[base + bi] = temp[bi] + val_b;
}

/// Phase 2b: Propagate block offsets to convert per-block sums into
/// global prefix sums.
///
/// For each element in blocks 1..N, add the cumulative sum of all
/// preceding blocks.
__kernel void prefix_sum_propagate(
    __global uint *data,
    __global const uint *block_offsets,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    uint block_id = gid / BLOCK_ELEMS;
    // Block 0 needs no offset; blocks 1..N add the cumulative sum
    // of all preceding blocks. block_offsets is an inclusive prefix sum
    // of block totals, so block_offsets[block_id - 1] gives the sum of
    // all blocks before block_id.
    if (block_id > 0) {
        data[gid] += block_offsets[block_id - 1];
    }
}

/// Phase 3: Scatter ranks from sorted order to position-indexed order.
///
/// For each sorted position i: write new_rank[sa[i]] = prefix[i] for
/// real entries, or UINT_MAX for sentinel entries.
__kernel void rank_scatter(
    __global const uint *sa,
    __global const uint *prefix,
    __global uint *new_rank,
    const uint n,
    const uint padded_n
) {
    uint i = get_global_id(0);
    if (i >= padded_n) return;

    uint sa_i = sa[i];
    if (sa_i < n) {
        new_rank[sa_i] = prefix[i];
    } else {
        // Sentinel: keep MAX rank so it sorts to the end
        new_rank[sa_i] = 0xFFFFFFFFu;
    }
}
