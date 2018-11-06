#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "lz77.h"

int test_empty() {
   LZ77_Compress(NULL, 0, NULL, 0);
   LZ77_Decompress(NULL, 0, NULL, 0);
   return 0;
}

int test_simple() {
    const char text[] = "This is the story of a girl, who destroyed the whole world";
    char* plain = malloc(2048);
    char* compressed = malloc(2048);
    memcpy(plain, text, sizeof(text));

    int bytes = LZ77_Compress(plain, sizeof(text), compressed, 2048);
    fprintf(stderr, "text strlen = %ld, wrote %d bytes\n", strlen(plain), bytes);

    memset(plain, 0, 2048);
    bytes = LZ77_Decompress(compressed, bytes, plain, 2048);
    fprintf(stderr,"text strlen = %ld, decompressed %d bytes\n", strlen(text), bytes);
    fprintf(stderr, "%s\n", plain);
    free(plain);
    free(compressed);
    return 0;
}

#define ECOLI "../samples/E.coli"
int test_ecoli() {
 struct stat sb;
 if(stat(ECOLI, &sb) != 0) {
     fprintf(stderr, "Failed opening E.Coli file. Skipping\n");
     return -1;
 }
 sb.st_size = 204800; /* XXX */
 char * original= malloc(sb.st_size);
 char * plain = malloc(sb.st_size);
 char * compressed = malloc(4 * sb.st_size);

 FILE * file = fopen(ECOLI, "r");
 fread(original, 1, sb.st_size, file);
 fclose(file);
 memcpy(plain, original, sb.st_size);

 int bytes = LZ77_Compress(plain, sb.st_size, compressed, 4 *sb.st_size);
 bytes = LZ77_Decompress(compressed, bytes, plain, sb.st_size);
 fprintf(stderr,"text strlen = %ld, wrote %d bytes\n", sb.st_size, bytes);

 if(memcmp(original,plain, sb.st_size) != 0) {
     fprintf(stderr, "original differs from round trip!\n");
     return -1;
 }

 free(original);
 free(plain);
 free(compressed);
 return 0;
}

int main() {
    fprintf(stderr, "Test Empty\n");
    test_empty();
    fprintf(stderr, "Test Simple\n");
    test_simple();
    fprintf(stderr, "Test ecoli\n");
    //test_ecoli();
    return 0;
}
