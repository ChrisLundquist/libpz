#include <stdio.h>
#include <string.h>

typedef struct Match {
  int offset;
  int length;
  char next;
} Match;

static Match FindMatch(const char* search,
                       unsigned int search_size,
                       const char* target,
                       unsigned int target_size) {
  /* assert search and target? */

  Match match = {.offset = 0, .length = 0, .next = *target};
  for (unsigned i = 0; i < search_size; ++i) {
    int temp_match_length = 0;
    while ((i + temp_match_length < target_size) &&
           search[i + temp_match_length] == target[i + temp_match_length]) {
      temp_match_length++;
    }
    if (temp_match_length >= match.length) {
      match.offset = search_size - i;
      match.length = temp_match_length;
    }
  }
  match.next = target[match.length];
  return match;
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
    printf("got match offset: %d, length: %d, next: %c\n", match.offset,
           match.length, match.next);

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
    for (int j = 0; j < m.length; j++) {
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
