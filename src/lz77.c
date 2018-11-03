#include <stdio.h>
int LZ77_Compress(const unsigned char* in,
                  unsigned int insize,
                  unsigned char* out,
                  unsigned int outsize) {
  if (insize == 0 || outsize == 0) {
    fprintf(stderr, "insize or outsize is 0\n");
    return -1;
  }
  return 0;
}

int LZ77_Decompress(const unsigned char* in,
                    unsigned int insize,
                    unsigned char* out,
                    unsigned int outsize) {
  return 0;
}
