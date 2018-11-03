#include <stdlib.h>
#include "lz77.h"

int test_empty() {
   LZ77_Compress(NULL, 0, NULL, 0);
   LZ77_Decompress(NULL, 0, NULL, 0);
   return 0;
}

int main() {
    test_empty();
    return 0;
}
