#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "lz77.h"
int LLVMFuzzerTestOneInput(const char *Data, int Size) {
  int outsize = 2 * Size;
  char * compressed = calloc(outsize, 1);
  char * plain = calloc(outsize, 1);
  fprintf(stderr, "Plain size: %d\n", Size);
  int bytes = LZ77_Compress(Data, Size, compressed, outsize);
  fprintf(stderr, "Compressed size: %d\n", bytes);
  bytes = LZ77_Decompress(compressed, bytes, plain, outsize);
  fprintf(stderr, "Decompressed size: %d\n", bytes);
  if(memcmp(Data, plain, bytes) != 0) {
      fprintf(stderr, "* Error input roundtrip is not equal\n");
      //fprintf(stderr, "Input: '%s', Round Trip: '%s',\n", Data, plain);
  }
  free(compressed);
  free(plain);
  return 0;  // Non-zero return values are reserved for future use.
}
