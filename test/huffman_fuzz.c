#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "huffman.h"
int LLVMFuzzerTestOneInput(const char *Data, int Size) {
    if(Size == 0)
        return 0;
    huffman_tree_t *tree = huff_new_8(Data, Size);
    huff_free(tree);
  return 0;  // Non-zero return values are reserved for future use.
}
