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
    const char text[1024] = "This is the story of a girl, who destroyed the whole world";
    char* out = malloc(2048);
    int bytes = LZ77_Compress(text, strlen(text), out, 2048);
    printf("text strlen = %d, wrote %d bytes\n", strlen(text), bytes);

    bytes = LZ77_Decompress(out, bytes, &text, strlen(text));
    printf("text strlen = %d, decompressed %d bytes\n", strlen(text), bytes);
    free(out);
    return 0;
}

#define ECOLI "../samples/E.coli"
int test_ecoli() {
 struct stat sb;
 if(stat(ECOLI, &sb) != 0) {
     fprintf(stderr, "Failed opening E.Coli file. Skipping\n");
     return -1;
 }
 sb.st_size = 4096; /* XXX */
 char * in = malloc(sb.st_size);
 if(in == NULL) {
     fprintf(stderr, "Failed malloc. Skipping\n");
     return -1;
 }
 char * out = malloc(4 * sb.st_size);
 if(out == NULL) {
     free(in);
     fprintf(stderr, "Failed malloc. Skipping\n");
     return -1;
 }
 FILE * file = fopen(ECOLI, "r");
 fread(in, 1, sb.st_size, file);
 fclose(file);

 int bytes = LZ77_Compress(in, strlen(in), out, 4 *sb.st_size);
 printf("text strlen = %d, wrote %d bytes\n", strlen(in), bytes);

 free(in);
 free(out);

}

int main() {
    test_empty();
    test_simple();
    //test_ecoli();
    return 0;
}
