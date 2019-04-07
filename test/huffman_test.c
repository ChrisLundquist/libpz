#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include "huffman.h"

int test_simple() {
  const char in[] = "aaaabbbccd";
  char* plain = malloc(2048);
  char* compressed = malloc(2048);
  huffman_tree_t *tree = huff_new_8(&in, sizeof(in));
  unsigned bytes = huff_Encode(tree, &in, sizeof(in), compressed, 2048);
  bytes = huff_Decode(tree, compressed, bytes, plain, 2048);

  if (memcmp(in, plain, bytes) != 0) {
    fprintf(stderr, "!!! FAILURE: original differs from round trip!\n");
  }

  huff_print(tree, 1);
  huff_free(tree);
  free(plain);
  free(compressed);
  return 0;
}

static huffman_tree_t* test_file(const char* filepath) {
    struct stat sb;
    if (stat(filepath, &sb) != 0) {
        fprintf(stderr, "Failed opening %s. Skipping\n", filepath);
        return NULL;
    }

    unsigned char* original = malloc(sb.st_size);

    FILE* file = fopen(filepath, "r");
    fread(original, 1, sb.st_size, file);
    fclose(file);
    huffman_tree_t *tree = huff_new_8(original, sb.st_size);
    huff_print(tree, 1);
    free(original);
    return tree;
}

#define ECOLI "../samples/E.coli"
#define PTT5 "../samples/ptt5"
#define BIBLE "../samples/bible.txt"
int test_ecoli() {
    fprintf(stderr, "Test ecoli\n");
    huffman_tree_t* tree = test_file(ECOLI);
    huff_free(tree);
    return 0;
}

int test_ptt5() { // 513216
    fprintf(stderr, "Test ptt5\n");
    huffman_tree_t* tree = test_file(PTT5);
    huff_free(tree);
    return 0;
}

int test_bible() { // 4047392
    fprintf(stderr, "Test bible\n");
    huffman_tree_t* tree = test_file(BIBLE);
    huff_free(tree);
    return 0;
}

int main(void) {
    test_simple();
    test_ecoli();
    test_bible();
    return 0;
}
