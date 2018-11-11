#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "engine.h"
static int test_file(opencl_codec_t *codec, const char* filepath) {
  struct stat sb;
  if (stat(filepath, &sb) != 0) {
    fprintf(stderr, "Failed opening %s. Skipping\n", filepath);
    return -1;
  }

  if(sb.st_size > 40960 * 4)
      sb.st_size = 40960 * 4; /* XXX */
  char* original = malloc(sb.st_size);
  char* plain = malloc(sb.st_size);
  char* compressed = malloc(4 * sb.st_size);

  FILE* file = fopen(filepath, "r");
  fread(original, 1, sb.st_size, file);
  fclose(file);
  memcpy(plain, original, sb.st_size);

  int bytes = codec->Encode(codec, plain, sb.st_size, compressed, 4 * sb.st_size);
  fprintf(stderr, "text strlen = %ld, wrote %d compressed bytes\n", sb.st_size,
          bytes);
  bytes = codec->Decode(codec, compressed, bytes, plain, sb.st_size);
  fprintf(stderr, "text strlen = %ld, wrote %d plain bytes\n", sb.st_size,
          bytes);

  if (memcmp(original, plain, sb.st_size) != 0) {
    fprintf(stderr, "!!! FAILURE: original differs from round trip!\n");
    free(original);
    free(plain);
    free(compressed);
    return -1;
  }
  else{
      fprintf(stderr, "* SUCCESS: original and round trip match\n");
  }

  free(original);
  free(plain);
  free(compressed);
  return 0;
}

int test_simple(opencl_codec_t *codec) {
  fprintf(stderr, "Test Simple\n");
  const char text[] =
      "This is the story of a girl, who destroyed the whole world";
  char* plain = malloc(2048);
  char* compressed = malloc(2048);
  memcpy(plain, text, sizeof(text));

  int bytes = codec->Encode(codec, plain, sizeof(text), compressed, 2048);
  fprintf(stderr, "text size = %ld, wrote %d bytes\n", sizeof(text), bytes);

  memset(plain, 0, 2048);
  bytes = codec->Decode(codec, compressed, bytes, plain, 2048);
  fprintf(stderr, "text strlen = %ld, decompressed %d bytes\n", strlen(text) + 1 /*null*/,
          bytes);
  fprintf(stderr, "%s\n", plain);
  free(plain);
  free(compressed);
  return 0;
}

#define ECOLI "../samples/E.coli"
#define PTT5 "../samples/ptt5"

int test_ecoli(opencl_codec_t *codec) {
  fprintf(stderr, "Test ecoli\n");
  test_file(codec, ECOLI);
  return 0;
}

int test_ptt5(opencl_codec_t *codec) {
  fprintf(stderr, "Test ptt5\n");
  test_file(codec, PTT5);
  return 0;
}
int main() {
    opencl_engine_t engine = CreateEngine();
    opencl_codec_t lz77 = GetCodec(&engine, LZ77);
    if(lz77.state != READY) {
        fprintf(stderr, "Codec not ready\n");
        return -1;
    }


    //lz77.Encode();
    test_simple(&lz77);
    //test_ecoli();
    //test_ptt5();

    //DestroyCodec(&lz77);
    //DestroyEngine(&engine);
    return 0;
}
