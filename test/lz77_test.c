#include "lz77.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

int test_empty() {
  fprintf(stderr, "Test Empty\n");
  LZ77_Compress(NULL, 0, NULL, 0);
  LZ77_Decompress(NULL, 0, NULL, 0);
  return 0;
}

int test_simple() {
  fprintf(stderr, "Test Simple\n");
  const char text[] =
      "This is the story of a girl, who destroyed the whole world";
  char* plain = malloc(2048);
  char* compressed = malloc(2048);
  memcpy(plain, text, sizeof(text));

  int bytes = LZ77_Compress(plain, sizeof(text), compressed, 2048);
  fprintf(stderr, "text size = %ld, wrote %d bytes\n", sizeof(text), bytes);

  memset(plain, 0, 2048);
  bytes = LZ77_Decompress(compressed, bytes, plain, 2048);
  fprintf(stderr, "text strlen = %ld, decompressed %d bytes\n", strlen(text) + 1 /*null*/,
          bytes);
  fprintf(stderr, "%s\n", plain);
  free(plain);
  free(compressed);
  return 0;
}

static int test_file(const char* filepath) {
  struct stat sb;
  if (stat(filepath, &sb) != 0) {
    fprintf(stderr, "Failed opening %s. Skipping\n", filepath);
    return -1;
  }

  sb.st_size = 40960; /* XXX */
  char* original = malloc(sb.st_size);
  char* plain = malloc(sb.st_size);
  char* compressed = malloc(4 * sb.st_size);

  FILE* file = fopen(filepath, "r");
  fread(original, 1, sb.st_size, file);
  fclose(file);
  memcpy(plain, original, sb.st_size);

  int bytes = LZ77_Compress(plain, sb.st_size, compressed, 4 * sb.st_size);
  fprintf(stderr, "text strlen = %ld, wrote %d compressed bytes\n", sb.st_size,
          bytes);
  bytes = LZ77_Decompress(compressed, bytes, plain, sb.st_size);
  fprintf(stderr, "text strlen = %ld, wrote %d plain bytes\n", sb.st_size,
          bytes);

  if (memcmp(original, plain, sb.st_size) != 0) {
    fprintf(stderr, "* FAILURE: original differs from round trip!\n");
    return -1;
  }

  free(original);
  free(plain);
  free(compressed);
}

#define ECOLI "../samples/E.coli"
#define PTT5 "../samples/ptt5"
int test_ecoli() {
  fprintf(stderr, "Test ecoli\n");
  test_file(ECOLI);
  return 0;
}

int test_ptt5() {
  fprintf(stderr, "Test ptt5\n");
  test_file(PTT5);
  return 0;
}

int main() {
  test_empty();
  test_simple();
  test_ecoli();
  test_ptt5();
  return 0;
}
