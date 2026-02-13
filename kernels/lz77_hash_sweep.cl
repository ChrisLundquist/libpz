// Parameterized LZ77 hash-table kernel for GPU parameter sweep experiments.
//
// Same algorithm as lz77_hash.cl but with configurable constants via
// preprocessor defines. Compile with:
//   -DSWEEP_HASH_BITS=15 -DSWEEP_MAX_CHAIN=64 -DSWEEP_BUCKET_CAP=64
//
// Defaults match the production kernel if defines are not provided.

typedef struct {
    unsigned int offset;
    unsigned int length;
    unsigned char next;
    unsigned char _pad[3];
} lz77_match_t;

#ifndef SWEEP_HASH_BITS
#define SWEEP_HASH_BITS 15u
#endif

#ifndef SWEEP_MAX_CHAIN
#define SWEEP_MAX_CHAIN 64u
#endif

#ifndef SWEEP_BUCKET_CAP
#define SWEEP_BUCKET_CAP 64u
#endif

#define SWEEP_MAX_WINDOW 32768u
#define SWEEP_HASH_SIZE  (1u << SWEEP_HASH_BITS)
#define SWEEP_HASH_MASK  (SWEEP_HASH_SIZE - 1u)
#define SWEEP_MIN_MATCH  3u

static inline unsigned int sweep_hash3(
    __global const unsigned char *data,
    unsigned int pos,
    unsigned int len)
{
    if (pos + 2 >= len) return 0;
    unsigned int h = ((unsigned int)data[pos] << 10)
                   ^ ((unsigned int)data[pos + 1] << 5)
                   ^ ((unsigned int)data[pos + 2]);
    return h & SWEEP_HASH_MASK;
}

__kernel void BuildHashTable(
    __global const unsigned char *in,
    const unsigned int in_len,
    __global volatile unsigned int *hash_counts,
    __global unsigned int *hash_table)
{
    unsigned int pos = get_global_id(0);
    if (pos + 2 >= in_len) return;

    unsigned int h = sweep_hash3(in, pos, in_len);
    unsigned int slot = atomic_inc(&hash_counts[h]);
    if (slot < SWEEP_BUCKET_CAP) {
        hash_table[h * SWEEP_BUCKET_CAP + slot] = pos;
    }
}

__kernel void FindMatches(
    __global const unsigned char *in,
    const unsigned int in_len,
    __global const unsigned int *hash_counts,
    __global const unsigned int *hash_table,
    __global lz77_match_t *out)
{
    unsigned int pos = get_global_id(0);
    if (pos >= in_len) return;

    unsigned int remaining = in_len - pos;
    lz77_match_t best;
    best.offset = 0;
    best.length = 0;
    best.next = in[pos];

    if (remaining < SWEEP_MIN_MATCH || pos == 0) {
        out[pos] = best;
        return;
    }

    unsigned int h = sweep_hash3(in, pos, in_len);
    unsigned int count = hash_counts[h];
    if (count > SWEEP_BUCKET_CAP) count = SWEEP_BUCKET_CAP;

    unsigned int window_min = (pos > SWEEP_MAX_WINDOW) ? (pos - SWEEP_MAX_WINDOW) : 0u;
    unsigned int checked = 0;

    for (unsigned int idx = count; idx > 0 && checked < SWEEP_MAX_CHAIN; idx--) {
        unsigned int candidate = hash_table[h * SWEEP_BUCKET_CAP + idx - 1];

        if (candidate >= pos || candidate < window_min)
            continue;

        checked++;

        if (best.length >= SWEEP_MIN_MATCH && best.length < remaining) {
            if (in[candidate + best.length] != in[pos + best.length])
                continue;
        }

        unsigned int max_len = remaining;
        unsigned int dist = pos - candidate;
        if (dist < max_len) max_len = dist;

        unsigned int match_len = 0;
        while (match_len < max_len &&
               in[candidate + match_len] == in[pos + match_len]) {
            match_len++;
        }

        if (match_len > best.length && match_len >= SWEEP_MIN_MATCH) {
            best.offset = pos - candidate;
            best.length = match_len;
        }
    }

    while (best.length >= remaining && best.length > 0)
        best.length--;

    if (best.length < remaining)
        best.next = in[pos + best.length];

    out[pos] = best;
}
