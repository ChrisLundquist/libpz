// LZ77 batched OpenCL kernel for GPU-parallel match finding.
//
// Each work-item processes STEP_SIZE consecutive positions, reducing
// kernel launch overhead. Uses a smaller window (32KB) for faster
// per-position search. The host deduplicates overlapping matches
// after readback.

typedef struct {
    unsigned int offset;
    unsigned int length;
    unsigned char next;
    unsigned char _pad[3];
} lz77_match_t;

#define MAX_WINDOW (1u << 15)
#define STEP_SIZE 32u

lz77_match_t FindMatchClassic(__global char* search,
                              const unsigned search_size,
                              __global char* target,
                              const unsigned target_size) {

  lz77_match_t best = {.offset = 0, .length = 0, .next = *target};

  for (unsigned i = 0; i < search_size; ++i) {
    // BUG-04 fix: bounds check before spot-check access
    if (best.length > 0 && (i + best.length) < search_size
        && best.length < target_size) {
      if (search[i + best.length] != target[best.length])
        continue;
    }

    unsigned temp_match_length = 0;
    unsigned tail = i + temp_match_length;
    while (tail < search_size && temp_match_length < target_size
           && search[tail] == target[temp_match_length])
    {
      ++temp_match_length;
      ++tail;
    }
    if (temp_match_length > best.length) {
      best.offset = search_size - i;
      best.length = temp_match_length;
    }
  }
  while (best.length >= target_size)
    --best.length;
  best.next = target[best.length];
  return best;
}

// Each work-item processes STEP_SIZE input positions.
// in     - The start of the input array.
// out    - Match output array (one match per input position).
// in_len - Total length of input buffer in bytes.
__kernel void Encode(__global char *in,
                     __global lz77_match_t *out,
                              const unsigned in_len) {
  unsigned i = get_global_id(0) * STEP_SIZE;
  if(i >= in_len)
    return;

  __local unsigned last_step;
  lz77_match_t last_match;

  for(unsigned step = 0; step < STEP_SIZE && (i + step < in_len); ) {
      const unsigned i_step = i + step;
      __global char *window_start = i_step > MAX_WINDOW ? in + i_step - MAX_WINDOW : in;
      const unsigned search_size = min(i_step, MAX_WINDOW);

      lz77_match_t match = FindMatchClassic(window_start, search_size, in + i_step, in_len - i_step);
      out[i_step] = match;
      last_step = step;
      last_match = match;
      step += match.length + 1;
  }

  // Handle boundary condition: truncate the last match so it doesn't
  // overlap into the next work-item's chunk.
  const int overlap_bytes = last_step + last_match.length + 1 - STEP_SIZE;
  if(overlap_bytes > 0) {
      last_match.length -= overlap_bytes;
      last_match.next = in[min(in_len, i + STEP_SIZE) - 1];
      out[i + last_step] = last_match;
  }
}
