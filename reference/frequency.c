#include <stdio.h>
#include <string.h>
#include <math.h>

#include "frequency.h"
long get_frequency(frequency_t *table, const unsigned char* in, long in_len) {
    unsigned i = 0;
    long remaining = in_len;
    while(remaining > 32) {
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        table->byte[in[i++]]++;
        remaining -= 32;
    }

    for(; i < in_len; ++i) {
        table->byte[in[i]]++;
    }

    for(i = 0; i < 256; ++i) {
        table->total += table->byte[i];
        if(table->byte[i] > 0)
            table->used++;
    }
    return in_len;
}

int print_table(const frequency_t *table) {
    printf("byte, count\n");
    for(int i = 0; i < 256; ++i) {
        printf("%03d\t%u\n", i, table->byte[i]);
    }
    printf("total: %lu\n", table->total);
    return 0;
}

inline frequency_t new_table() {
    frequency_t table;
    memset(&table, 0, sizeof(table));
    return table;
}

inline float get_entropy(const frequency_t *table) {
    float total = 0.0f;
    for(unsigned i = 0; i < 256; ++i) {
        float prob = (float) table->byte[i] / (float)table->total;
        if(prob > 0.0f)
            total += prob * log2f(prob);
    }
    return -total;
}
