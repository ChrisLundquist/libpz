// LZ77 hash-table-based OpenCL kernel for GPU-parallel match finding.
//
// Two-pass approach:
//   Pass 1 (BuildHashTable): Each work-item hashes its 3-byte prefix and
//     atomically appends its position to a hash bucket (bounded ring buffer).
//   Pass 2 (FindMatches): Each work-item looks up its hash bucket and only
//     compares against positions in that bucket (bounded by MAX_CHAIN).
//
// This replaces the brute-force O(n*w) search with O(n*MAX_CHAIN) search,
// matching the CPU hash-chain strategy.

typedef struct {
    unsigned int offset;
    unsigned int length;
    unsigned char next;
    unsigned char _pad[3];
} lz77_match_t;

#define MAX_WINDOW 32768u
#define HASH_BITS  15u
#define HASH_SIZE  (1u << HASH_BITS)
#define HASH_MASK  (HASH_SIZE - 1u)
#define MAX_CHAIN  64u
#define BUCKET_CAP 64u
#define MIN_MATCH  3u

// Same hash function as CPU: hash3
inline unsigned int hash3(__global const unsigned char *data, unsigned int pos, unsigned int len) {
    if (pos + 2 >= len) return 0;
    unsigned int h = ((unsigned int)data[pos] << 10)
                   ^ ((unsigned int)data[pos + 1] << 5)
                   ^ ((unsigned int)data[pos + 2]);
    return h & HASH_MASK;
}

// Pass 1: Build hash table.
// Each work-item computes the hash for its position and atomically appends
// the position to the corresponding bucket.
//
// hash_counts[h] = number of entries in bucket h (atomically incremented)
// hash_table[h * BUCKET_CAP + slot] = position stored in that slot
__kernel void BuildHashTable(__global const unsigned char *in,
                             const unsigned int in_len,
                             __global volatile unsigned int *hash_counts,
                             __global unsigned int *hash_table) {
    unsigned int pos = get_global_id(0);
    if (pos + 2 >= in_len) return;

    unsigned int h = hash3(in, pos, in_len);
    unsigned int slot = atomic_inc(&hash_counts[h]);
    // Bounded bucket: only store if there's room
    if (slot < BUCKET_CAP) {
        hash_table[h * BUCKET_CAP + slot] = pos;
    }
}

// Pass 2: Find best match at each position using the hash table.
// Each work-item looks up its bucket and scans candidates within the
// sliding window, up to MAX_CHAIN comparisons.
__kernel void FindMatches(__global const unsigned char *in,
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
    if (count > BUCKET_CAP) count = BUCKET_CAP;

    unsigned int window_min = (pos > MAX_WINDOW) ? (pos - MAX_WINDOW) : 0u;
    unsigned int checked = 0;

    // Scan bucket entries in reverse order (most recent first)
    for (unsigned int idx = count; idx > 0 && checked < MAX_CHAIN; idx--) {
        unsigned int candidate = hash_table[h * BUCKET_CAP + idx - 1];

        // Must be within window and before current position
        if (candidate >= pos || candidate < window_min)
            continue;

        checked++;

        // Spot-check: if we already have a best match, check that the
        // candidate matches at best_length position (prune quickly)
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
