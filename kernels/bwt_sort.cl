/// GPU-accelerated BWT suffix array construction via bitonic sort.
///
/// The BWT uses prefix-doubling: at each step k=1,2,4,..., we sort the
/// suffix array sa[] by the composite key (rank[sa[i]], rank[(sa[i]+k) % n]).
/// This kernel implements one compare-and-swap step of a bitonic sort.
///
/// The host calls bitonic_sort_step O(log^2 padded_n) times per doubling step.
/// Rank assignment after sorting is done on the CPU (cheap O(n) scan).
///
/// The sa[] and rank[] arrays are padded to a power-of-2 size. Padding
/// entries have rank = UINT_MAX so they sort to the end.

/// One compare-and-swap step of bitonic sort.
///
/// Each work-item handles one pair (i, i^j) and swaps if the
/// composite key ordering violates the bitonic sort invariant.
///
/// Parameters:
///   sa       - suffix array (read/write), length padded_n (power of 2)
///   rank     - current rank array (read-only), length padded_n
///   n        - actual input length (for modular indexing in rank lookup)
///   padded_n - padded array length (power of 2, for bounds checking)
///   k        - prefix-doubling offset (1, 2, 4, ...)
///   j        - bitonic step parameter (power of 2)
///   k_sort   - bitonic block size parameter (power of 2)
__kernel void bitonic_sort_step(
    __global uint *sa,
    __global const uint *rank,
    const uint n,
    const uint padded_n,
    const uint k,
    const uint j,
    const uint k_sort
) {
    uint i = get_global_id(0);
    if (i >= padded_n) return;

    uint ixj = i ^ j;
    // Only the lower-index thread in each pair does the comparison
    if (ixj <= i) return;
    if (ixj >= padded_n) return;

    uint sa_i = sa[i];
    uint sa_ixj = sa[ixj];

    // Build composite keys.
    // For real entries (sa < n): key = (rank[sa], rank[(sa+k) % n])
    // For sentinel entries (sa >= n): key = (UINT_MAX, UINT_MAX) so they
    // sort to the end.
    ulong key_i, key_ixj;

    if (sa_i < n) {
        key_i = ((ulong)rank[sa_i] << 32) | (ulong)rank[(sa_i + k) % n];
    } else {
        key_i = (ulong)0xFFFFFFFF << 32 | (ulong)0xFFFFFFFF;
    }

    if (sa_ixj < n) {
        key_ixj = ((ulong)rank[sa_ixj] << 32) | (ulong)rank[(sa_ixj + k) % n];
    } else {
        key_ixj = (ulong)0xFFFFFFFF << 32 | (ulong)0xFFFFFFFF;
    }

    // Determine sort direction for this block
    bool ascending = ((i & k_sort) == 0);

    bool should_swap = ascending ? (key_i > key_ixj) : (key_i < key_ixj);

    if (should_swap) {
        sa[i]   = sa_ixj;
        sa[ixj] = sa_i;
    }
}
