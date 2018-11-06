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

  for (unsigned i = 0; i < search_size;) {
    register unsigned temp_match_length = 0;
    register unsigned tail = i + temp_match_length;
    while (search[tail] == target[temp_match_length] && (tail < target_size) &&
           (tail < search_size)) {
      ++temp_match_length;
      ++tail;
    }
    if (temp_match_length > best.length) {
      best.offset = search_size - i;
      best.length = temp_match_length;

      /* if we matched this much we can skip ahead */
      i += best.length;
      continue;
    }
    ++i;
  }
  /* Ensure we don't point to garbage data */
  if (best.length < target_size)
    best.next = target[best.length];
  else
    best.next = 0;
  //  printf("got match offset: %d, length: %d, next: %c\n", best.offset,
  //         best.length, best.next);
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
    return -1;
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
    char* seek = outpos - m.offset;
    if (((outpos - out) + m.length) > (outsize)) {
      fprintf(stderr, "not enough room in output buffer\n");
      break;
    }
    for (unsigned j = 0; j < m.length; j++) {
      /* TODO the "Do it again until it fits case" */
      *outpos = *seek;
      outpos++;
      seek++;
    }
    /* make sure we don't spew an extra garbage byte */
    if (m.next) {
      *outpos = m.next;
      outpos++;
    }
  }

  return outpos - out;
}
