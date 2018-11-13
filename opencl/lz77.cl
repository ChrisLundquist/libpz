#include "lz77.h"
#define MAX_WINDOW 32

lz77_match_t FindMatchClassic(__global char* search,
                              unsigned search_size,
                              __global char* target,
                              unsigned target_size) {
  lz77_match_t best = {.offset = 0, .length = 0, .next = *target};
  if (target_size == 1)
    return best;

  for (unsigned i = 0; i < search_size; ++i) {
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


__kernel void Encode(__global char *in,
                     __global lz77_match_t *out,
                              const unsigned count) {
  int i = get_global_id(0);
  if(i > count - 1)
    return;

  __global char *window_start = i > MAX_WINDOW ? in + i - MAX_WINDOW : in;
  unsigned search_size = i > MAX_WINDOW ? MAX_WINDOW : i;

  out[i] = FindMatchClassic(window_start, search_size, in + i, count - i);
}
