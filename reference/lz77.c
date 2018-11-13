#include "lz77.h"
#include <stdio.h>
#include <string.h>

#define MAX_WINDOW 4096
inline void PrintMatch(const lz77_match_t* match) {
  fprintf(stderr, "{offset: %d, length: %d, next: %02x}\n", match->offset,
          match->length, match->next & 0xff);
}

static lz77_match_t FindMatchClassic(const char* search,
                                     unsigned search_size,
                                     const char* target,
                                     unsigned target_size) {
  /* assert search and target? */
  register lz77_match_t best = {.offset = 0, .length = 0, .next = *target};

  for (register unsigned i = 0; i < search_size; ++i) {
    /* Spot check if this could be at least as long as our current best */
    if (search[i + best.length] != target[best.length])
      continue;

    register unsigned temp_match_length = 0;
    register unsigned tail = i + temp_match_length;

    while (search[tail] == target[temp_match_length] && (tail < target_size)) {
      ++temp_match_length;
      ++tail;
    }
    if (temp_match_length > best.length) {
      best.offset = search_size - i;
      best.length = temp_match_length;

      /* if we matched this much we can skip ahead */
      // i += best.length - 1;
    }
  }
  /* Ensure we don't point to garbage data */
  /* in the "abcabc" case we can only match ab next: c */
  /* We have to truncate our match so that next points to valid bytes */
  while (best.length >= target_size)
    --best.length;
  best.next = target[best.length];
  return best;
}

/*
static lz77_match_t FindMatchTrivial(const char* search,
                              unsigned search_size,
                              const char* target,
                              unsigned target_size) {
  register lz77_match_t best = {.offset = 0, .length = 0, .next = *target};
  return best;
}

static lz77_match_t FindMatchMemcmp(const char* search,
                             unsigned search_size,
                             const char* target,
                             unsigned target_size) {
  register lz77_match_t best = {.offset = 0, .length = 0, .next = *target};
  if (target_size == 1)
    return best;

  for (register unsigned i = 0; i < target_size; ++i) {
    register unsigned cur_len = 1;
    while ((i + cur_len) < target_size && (i + cur_len) < search_size &&
           memcmp(search + i, target, cur_len) == 0) {
      ++cur_len;
    }
    if (cur_len - 1 > best.length) {
      best.length = cur_len - 1;
      best.offset = search_size - i;
      best.next = target[best.length];
    }
  }

  return best;
}
*/
inline static void WriteMatch(const lz77_match_t* match, char* out) {
  memcpy(out, match, sizeof(lz77_match_t));
  return;
}

// lz77_match_t (*FindMatch)(const char*, unsigned, const char*, unsigned) =
// FindMatchClassic;
lz77_match_t (*FindMatch)(const char*,
                          unsigned,
                          const char*,
                          unsigned) = FindMatchClassic;

int LZ77_Compress(const char* in,
                  unsigned int insize,
                  char* out,
                  unsigned int outsize) {
  if (insize == 0 || outsize == 0) {
    fprintf(stderr, "insize or outsize is 0\n");
    return 0;
  }
  const char* pos = in;
  char* out_pos = out;
  const char* end = in + insize;

  while (pos < end) {
    const int i = (pos - in);
    // lz77_match_t match = FindMatch(in, window, pos, insize - window);
    const char* window_start = i > MAX_WINDOW ? in + i - MAX_WINDOW : in;
    const unsigned search_size = i > MAX_WINDOW ? MAX_WINDOW : i;
    lz77_match_t match = FindMatch(window_start, search_size, pos, insize - i);

    if ((out_pos - out + sizeof(lz77_match_t)) > outsize) {
      fprintf(stderr, "not enough room in output buffer\n");
      break;
    }
    // PrintMatch(&match);
    WriteMatch(&match, out_pos);
    out_pos += sizeof(lz77_match_t);
    pos += match.length + 1;
  }
  /*
  for (Match match; pos < end; pos += match.length + 1) {
    const unsigned window = (pos - in);
    match = FindMatch(in, window, pos, insize - window);
    if ((out_pos - out + sizeof(Match)) > outsize) {
      fprintf(stderr, "not enough room in output buffer\n");
      break;
    }
    out_pos += sizeof(Match);
  }*/

  return out_pos - out;
}

int LZ77_Decompress(const char* in,
                    unsigned int insize,
                    char* out,
                    unsigned int outsize) {
  lz77_match_t* matches = (lz77_match_t*)in;
  int match_size = insize / sizeof(lz77_match_t);
  register char* outpos = out;

  for (int i = 0; i < match_size; i++) {
    register lz77_match_t m = matches[i];
    if (((outpos - out) + m.length) > (outsize)) {
      fprintf(stderr, "not enough room in output buffer\n");
      break;
    }
    register char* seek = outpos - m.offset;
    for (register unsigned j = 0; j < m.length; j++) {
      *outpos = *seek;
      outpos++;
      seek++;
    }
    *outpos = m.next;
    outpos++;
  }

  return outpos - out;
}
