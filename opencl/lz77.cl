#include "lz77.h"
__kernel void Encode(__global const char *in,
                     __global lz77_match_t *out,
                              const unsigned count) {
  int i = get_global_id(0);
  if(i > count)
    return;

  lz77_match_t best = { .offset = 0, .length = 0, .next = in[i]};
  out[i] = best;
}
