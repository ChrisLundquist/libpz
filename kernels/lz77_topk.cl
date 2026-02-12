// LZ77 top-K match finding kernel for optimal parsing.
//
// Each work-item finds the K best match candidates at one input position,
// searching backward through a 32KB sliding window. The host runs backward
// DP on the resulting match table to select the minimum-cost parse.
//
// Output: K candidates per position (flat array, position-major order).
// Candidates are sorted by length descending. Unused slots have length=0.

typedef struct {
    unsigned short offset;
    unsigned short length;
} lz77_candidate_t;

#define MAX_WINDOW (1u << 15)  // 32KB
#define K 4
#define MIN_MATCH 3u

// Insert a new candidate into the top-K array (sorted by length desc).
// For equal lengths, prefer the smaller offset.
static void insert_candidate(lz77_candidate_t top[K],
                      unsigned short offset,
                      unsigned short length) {
    // Find insertion point
    int insert_at = -1;
    for (int i = 0; i < K; i++) {
        if (length > top[i].length ||
            (length == top[i].length && offset < top[i].offset)) {
            insert_at = i;
            break;
        }
    }

    if (insert_at < 0)
        return;

    // Shift down to make room (drop the last element)
    for (int i = K - 1; i > insert_at; i--) {
        top[i] = top[i - 1];
    }
    top[insert_at].offset = offset;
    top[insert_at].length = length;
}

// Each work-item finds top-K matches at one input position.
// in      - The input data array.
// out     - Output array: K candidates per position (total: in_len * K).
// in_len  - Total length of input buffer in bytes.
__kernel void EncodeTopK(__global const char *in,
                         __global lz77_candidate_t *out,
                         const unsigned in_len) {
    unsigned pos = get_global_id(0);
    if (pos >= in_len)
        return;

    lz77_candidate_t top[K];
    for (int i = 0; i < K; i++) {
        top[i].offset = 0;
        top[i].length = 0;
    }

    unsigned window_start = pos > MAX_WINDOW ? pos - MAX_WINDOW : 0;
    unsigned remaining = in_len - pos;

    for (unsigned j = window_start; j < pos; j++) {
        // Spot-check optimization: skip if current best length doesn't match
        if (top[K - 1].length >= MIN_MATCH) {
            unsigned short best_len = top[K - 1].length;
            if ((j + best_len) < pos && best_len < remaining) {
                if (in[j + best_len] != in[pos + best_len])
                    continue;
            }
        }

        // Compare bytes to find match length
        unsigned match_len = 0;
        unsigned max_len = min(remaining, pos - j);
        if (max_len > 65535u) max_len = 65535u;

        while (match_len < max_len &&
               in[j + match_len] == in[pos + match_len]) {
            match_len++;
        }

        if (match_len < MIN_MATCH)
            continue;

        unsigned short offset = (unsigned short)(pos - j);
        unsigned short length = (unsigned short)match_len;

        insert_candidate(top, offset, length);
    }

    // Write K candidates to output
    unsigned base = pos * K;
    for (int i = 0; i < K; i++) {
        out[base + i] = top[i];
    }
}
