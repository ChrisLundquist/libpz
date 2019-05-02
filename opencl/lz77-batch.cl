#include "../include/lz77.h"

#define MAX_WINDOW (1u << 15)
#define STEP_SIZE 32u
lz77_match_t FindMatchClassic(__global char* search,
                              const unsigned search_size,
                              __global char* target,
                              const unsigned target_size) {

  lz77_match_t best = {.offset = 0, .length = 0, .next = *target};

  for (unsigned i = 0; i < search_size; ++i) {
    if (search[i + best.length] != target[best.length])
      continue;

    unsigned temp_match_length = 0;
    unsigned tail = i + temp_match_length;
    while (search[tail] == target[temp_match_length] && (tail < target_size))
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

/* in - The start of the array we want to encode.
 * out - The place we put our matches. XXX The caller must ensure this is correctly sized.
 * in_len - length of input buffer in bytes.
 * TODO add out_len?
 */
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

      /*
       * const unsigned next_step = step + match.length + 1;
      while( ++step < next_step && step < STEP_SIZE) {
        lz77_match_t bad = { .valid = 0 };
        out[i + step] = bad;
      }
      */
  }
  /* There is an issue with boundary conditions here.
   * We don't know where the last thread intends to jump into on this block.
   * To work around this, we shorten the very last match we get to enter at
   * the start of the next chunk.
  */
  const int overlap_bytes = last_step + last_match.length + 1 - STEP_SIZE;
  //const int overlap_bytes = (i + last_step + last_match.length) % STEP_SIZE;
  if(overlap_bytes > 0) {
      last_match.length -= overlap_bytes;
      last_match.next = in[min(in_len, i + STEP_SIZE) - 1];
      out[i + last_step] = last_match;
  }
}
