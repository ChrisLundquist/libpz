// LZ77 short-chain hash kernel (Variant B) for GPU-parallel match finding.
//
// Variant B of the hash-table match finder: uses short buckets (2-4 entries)
// instead of the default 64-entry buckets. This trades match quality for
// throughput — fewer candidates to check means less divergence per work-item.
//
// The short bucket approach is designed to maximize GPU occupancy:
// - Small bucket cap (2-4) means less warp divergence in the scan loop
// - Lower memory footprint: 4 × 32768 × 4 = 512KB vs 64 × 32768 × 4 = 8MB
// - Better cache utilization: each bucket fits in one cache line
//
// Compile with -D defines to set parameters:
//   -DHASH_BITS=15 -DBUCKET_CAP=4 -DMAX_CHAIN=4

typedef struct {
    unsigned int offset;
    unsigned int length;
    unsigned char next;
    unsigned char _pad[3];
} lz77_match_t;

#ifndef MAX_WINDOW
#define MAX_WINDOW 32768u
#endif

#ifndef HASH_BITS
#define HASH_BITS 15u
#endif

#define HASH_SIZE  (1u << HASH_BITS)
#define HASH_MASK  (HASH_SIZE - 1u)

#ifndef MAX_CHAIN
#define MAX_CHAIN 4u
#endif

#ifndef BUCKET_CAP
#define BUCKET_CAP 4u
#endif

#define MIN_MATCH 3u

// Same hash function as CPU and default GPU kernel
static inline unsigned int hash3(__global const unsigned char *data, unsigned int pos, unsigned int len) {
    if (pos + 2 >= len) return 0;
    unsigned int h = ((unsigned int)data[pos] << 10)
                   ^ ((unsigned int)data[pos + 1] << 5)
                   ^ ((unsigned int)data[pos + 2]);
    return h & HASH_MASK;
}

// Pass 1: Build hash table with short buckets.
// When a bucket overflows, the oldest entry is implicitly discarded
// (we only keep the most recent BUCKET_CAP entries via modular indexing).
__kernel void BuildHashTableShort(__global const unsigned char *in,
                                  const unsigned int in_len,
                                  __global volatile unsigned int *hash_counts,
                                  __global unsigned int *hash_table) {
    unsigned int pos = get_global_id(0);
    if (pos + 2 >= in_len) return;

    unsigned int h = hash3(in, pos, in_len);
    unsigned int slot = atomic_inc(&hash_counts[h]);
    // With short buckets, use modular indexing to keep most recent entries
    unsigned int actual_slot = slot % BUCKET_CAP;
    hash_table[h * BUCKET_CAP + actual_slot] = pos;
}

// Pass 2: Find best match using short bucket scan.
// With BUCKET_CAP=2-4, the entire scan fits in a few iterations,
// minimizing warp divergence.
__kernel void FindMatchesShort(__global const unsigned char *in,
                               const unsigned int in_len,
                               __global const unsigned int *hash_counts,
                               __global const unsigned int *hash_table,
                               __global lz77_match_t *out) {
    unsigned int pos = get_global_id(0);
    if (pos >= in_len) return;

    unsigned int remaining = in_len - pos;
    lz77_match_t best;
    best.offset = 0;
    best.length = 0;
    best.next = in[pos];

    if (remaining < MIN_MATCH || pos == 0) {
        out[pos] = best;
        return;
    }

    unsigned int h = hash3(in, pos, in_len);
    unsigned int count = hash_counts[h];
    // Effective entries: min(count, BUCKET_CAP)
    unsigned int effective = (count < BUCKET_CAP) ? count : BUCKET_CAP;

    unsigned int window_min = (pos > MAX_WINDOW) ? (pos - MAX_WINDOW) : 0u;
    unsigned int checked = 0;

    // Scan all bucket entries (short scan — only 2-4 entries)
    for (unsigned int idx = 0; idx < effective && checked < MAX_CHAIN; idx++) {
        // Read entries in order; with modular insertion,
        // start from the most recently inserted position
        unsigned int read_slot;
        if (count <= BUCKET_CAP) {
            // Bucket not full: entries are in order 0..count-1, scan reverse
            read_slot = effective - 1u - idx;
        } else {
            // Bucket wrapped: most recent is at (count-1) % BUCKET_CAP
            // Scan backwards from most recent
            unsigned int most_recent = (count - 1u) % BUCKET_CAP;
            read_slot = (most_recent + BUCKET_CAP - idx) % BUCKET_CAP;
        }
        unsigned int candidate = hash_table[h * BUCKET_CAP + read_slot];

        if (candidate >= pos || candidate < window_min)
            continue;

        checked++;

        // Spot-check at best_length position
        if (best.length >= MIN_MATCH && best.length < remaining) {
            if (in[candidate + best.length] != in[pos + best.length])
                continue;
        }

        // Compare bytes
        unsigned int max_len = remaining;
        unsigned int dist = pos - candidate;
        if (dist < max_len) max_len = dist;

        unsigned int match_len = 0;
        while (match_len < max_len &&
               in[candidate + match_len] == in[pos + match_len]) {
            match_len++;
        }

        if (match_len > best.length && match_len >= MIN_MATCH) {
            best.offset = pos - candidate;
            best.length = match_len;
        }
    }

    // Ensure room for the literal 'next' byte
    while (best.length >= remaining && best.length > 0)
        best.length--;

    if (best.length < remaining)
        best.next = in[pos + best.length];

    out[pos] = best;
}
