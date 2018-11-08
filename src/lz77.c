#include <stdio.h>
#include <string.h>

typedef struct Match {
  unsigned offset;
  unsigned length;
  char next;
} Match;

static Match FindMatch(const char* search,
                       unsigned int search_size,
                       const char* target,
                       unsigned int target_size) {
  /* assert search and target? */
  register Match best = {.offset = 0, .length = 0, .next = *target};
  if (target_size == 1)
    return best;

  // unsigned window = search_size > 4096 ? (search + search_size - 4096) :
  // search_size;
  for (unsigned i = 0; i < search_size; ++i) {
    register unsigned temp_match_length = 0;
    register unsigned tail = i + temp_match_length;

    while (search[tail] == target[temp_match_length] && (tail < target_size))
    //&& (tail < search_size)) { /* removing this allows us to match into the
    // future */
    {
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
  while (best.length >= target_size - 1)
    --best.length;
  best.next = target[best.length];
  //printf("got match offset: %d, length: %d, next: %02x\n", best.offset,
  //       best.length, best.next & 0xff);
  return best;
}

inline static void WriteMatch(const Match* match, char* out) {
  memcpy(out, match, sizeof(Match));
  return;
}

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
    int window = (pos - in);
    Match match = FindMatch(in, window, pos, insize - window);

    if ((out_pos - out + sizeof(Match)) > outsize) {
      fprintf(stderr, "not enough room in output buffer\n");
      break;
    }
    WriteMatch(&match, out_pos);
    out_pos += sizeof(Match);
    pos += match.length + 1;
  }

  return out_pos - out;
}

int LZ77_Decompress(const char* in,
                    unsigned int insize,
                    char* out,
                    unsigned int outsize) {
  Match* matches = (Match*)in;
  int match_size = insize / sizeof(Match);
  char* outpos = out;

  for (int i = 0; i < match_size; i++) {
    Match m = matches[i];
    if (((outpos - out) + m.length) > (outsize)) {
      fprintf(stderr, "not enough room in output buffer\n");
      break;
    }
    char* seek = outpos - m.offset;
    for (unsigned j = 0; j < m.length; j++) {
      /* TODO the "Do it again until it fits case" */
      *outpos = *seek;
      outpos++;
      seek++;
    }
    *outpos = m.next;
    outpos++;
  }

  return outpos - out;
}
