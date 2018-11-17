#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "frequency.h"

static frequency_t test_file(const char* filepath) {
    struct stat sb;
    frequency_t table = new_table();
    if (stat(filepath, &sb) != 0) {
        fprintf(stderr, "Failed opening %s. Skipping\n", filepath);
        return table;
    }

    unsigned char* original = malloc(sb.st_size);

    FILE* file = fopen(filepath, "r");
    fread(original, 1, sb.st_size, file);
    fclose(file);
    get_frequency(&table, original, sb.st_size);
    float entropy = get_entropy(&table);
    printf("Entropy: %g\n", entropy);
    printf("Actual filesize: %ld\n", table.total);
    printf("Ideal filesize: %g\n", table.total * (entropy / 8.0f));
    printf("Delta filesize: %g\n", table.total - table.total * (entropy / 8.0f));
    printf("\n");
    //print_table(&table);

    free(original);
    return table;
}
#define LARGE "../samples/large.tar.gz"
#define ECOLI "../samples/E.coli"
#define PTT5 "../samples/ptt5"
#define BIBLE "../samples/bible.txt"
int test_ecoli() { // 4638690
    fprintf(stderr, "Test ecoli\n");
    frequency_t table = test_file(ECOLI);
    if(table.total != 4638690) {
        fprintf(stderr, "Incorrect total\n");
        return -1;
    }
    return 0;
}

int test_ptt5() { // 513216
    fprintf(stderr, "Test ptt5\n");
    frequency_t table = test_file(PTT5);
    if(table.total != 513216) {
        fprintf(stderr, "Incorrect total\n");
        return -1;
    }
    return 0;
}

int test_bible() { // 4047392
    fprintf(stderr, "Test bible\n");
    frequency_t table = test_file(BIBLE);
    if(table.total != 4047392) {
        fprintf(stderr, "Incorrect total\n");
        return -1;
    }
    float entropy = get_entropy(&table);
    return 0;
}

int test_large() { // 4047392
    fprintf(stderr, "Test large.tar.gz\n");
    frequency_t table = test_file(LARGE);
    if(table.total != 3258648) {
        fprintf(stderr, "Incorrect total\n");
        return -1;
    }
    return 0;
}

int test_simple() {
    frequency_t table = new_table();
    unsigned char test[256];
    for(int i = 0; i < 256; i++)
        test[i] = i;
    return 0;
}

int main() {
    test_simple();
    test_ecoli();
    test_bible();
    test_ptt5();
    test_large();
    return 0;
}
