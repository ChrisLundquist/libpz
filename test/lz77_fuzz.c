#include <stdlib.h>
#include "lz77.h"
int LLVMFuzzerTestOneInput(const char *Data, int Size) {
  int outsize = 2 * Size;
  char * compressed = malloc(outsize);
  char * plain = malloc(outsize);
  int compressed_size = LZ77_Compress(Data, Size, compressed, outsize);
  LZ77_Decompress(compressed, compressed_size, plain, outsize);
  free(compressed);
  free(plain);
  return 0;  // Non-zero return values are reserved for future use.
}
