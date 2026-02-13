/// GPU-accelerated radix sort for BWT prefix-doubling.
///
/// Replaces bitonic sort with LSB-first 8-bit radix sort. Each pass
/// sorts by one byte of the 64-bit composite key:
///   key = (rank[sa[i]] << 32) | rank[(sa[i]+k) % n]
///
/// Eight passes (bytes 0→7) fully sort the array. LSB-first radix sort
/// is naturally stable, so equal-key tiebreaking is handled by
/// initializing sa[] in descending order on the host.
///
/// Compiled with -DWORKGROUP_SIZE=N (power of 2, typically 256).

// @pz_cost {
//   threads_per_element: 1
//   passes: 3
//   buffers: keys=N*4, histogram=N*4, sa_in=N*4, sa_out=N*4
//   local_mem: 1024
//   note: 4 radix passes per doubling step
// }

/// Number of radix buckets (8-bit digit → 256 values).
#define RADIX 256

/// Phase 1: Compute the 8-bit radix digit for each element.
///
/// Extracts one byte from the 64-bit composite key
///   key = (rank[sa[i]] << 32) | rank[(sa[i]+k) % n]
/// The byte position is selected by `pass` (0 = LSB, 7 = MSB).
///
/// Sentinel entries (sa[i] >= n) get digit 0xFF so they sort to the end.
__kernel void radix_compute_keys(
    __global const uint *sa,
    __global const uint *rank,
    __global uint *keys,
    const uint n,
    const uint padded_n,
    const uint k,
    const uint pass
) {
    uint i = get_global_id(0);
    if (i >= padded_n) return;

    uint sa_i = sa[i];

    if (sa_i >= n) {
        keys[i] = 0xFFu;
        return;
    }

    // Build 64-bit composite key
    uint r1 = rank[sa_i];
    uint r2 = rank[(sa_i + k) % n];

    // Extract the byte for this pass (0=LSB of r2, ..., 3=MSB of r2,
    // 4=LSB of r1, ..., 7=MSB of r1)
    uint word = (pass < 4) ? r2 : r1;
    uint shift = (pass & 3u) * 8u;
    keys[i] = (word >> shift) & 0xFFu;
}

/// Phase 2: Per-workgroup histogram of radix digits.
///
/// Each workgroup processes a tile of elements, counting occurrences of
/// each 8-bit digit value. Results are written to a flat array:
///   histograms[digit * num_groups + group_id] = count
///
/// This column-major layout means a prefix sum over the full array
/// produces correct global scatter offsets directly.
__kernel void radix_histogram(
    __global const uint *keys,
    __global uint *histograms,
    const uint padded_n,
    const uint num_groups
) {
    __local uint local_hist[RADIX];

    uint tid = get_local_id(0);
    uint group_id = get_group_id(0);
    uint group_size = get_local_size(0);
    uint base = group_id * group_size;

    // Clear local histogram (each thread clears multiple bins)
    for (uint b = tid; b < RADIX; b += group_size) {
        local_hist[b] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Count digits in this workgroup's tile
    uint idx = base + tid;
    if (idx < padded_n) {
        uint digit = keys[idx];
        atomic_inc(&local_hist[digit]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Write local histogram to global memory (column-major layout)
    for (uint b = tid; b < RADIX; b += group_size) {
        histograms[b * num_groups + group_id] = local_hist[b];
    }
}

/// Convert inclusive prefix sum to exclusive prefix sum.
///
/// exclusive[i] = (i == 0) ? 0 : inclusive[i - 1]
/// This gives the starting offset for each histogram bucket.
__kernel void inclusive_to_exclusive(
    __global const uint *inclusive,
    __global uint *exclusive,
    const uint count
) {
    uint i = get_global_id(0);
    if (i >= count) return;
    exclusive[i] = (i == 0) ? 0u : inclusive[i - 1];
}

/// Phase 3: Stable scatter — thread 0 of each workgroup sequentially
/// scatters its tile's elements to preserve their original order.
///
/// For each element in the tile (in order), look up its digit, compute
/// the output position from global_offsets, write it, and bump the offset.
///
/// Sequential per workgroup but correct and simple. With workgroup_size=256,
/// each thread 0 processes at most 256 elements — fast enough for scatter.
__kernel void radix_scatter(
    __global const uint *sa_in,
    __global const uint *keys,
    __global uint *global_offsets,
    __global uint *sa_out,
    const uint padded_n,
    const uint num_groups
) {
    uint tid = get_local_id(0);
    if (tid != 0) return;

    uint group_id = get_group_id(0);
    uint group_size = get_local_size(0);
    uint base = group_id * group_size;
    uint end = base + group_size;
    if (end > padded_n) end = padded_n;

    for (uint i = base; i < end; i++) {
        uint digit = keys[i];
        uint pos = global_offsets[digit * num_groups + group_id];
        sa_out[pos] = sa_in[i];
        global_offsets[digit * num_groups + group_id] = pos + 1;
    }
}
