typedef struct {
    /* the index is the byte pattern, the value is the count */
    /* E.G. byte['A'] would return the count of 'A's or 65s */
    unsigned int byte[256];
    /* the sum of all the values in byte */
    unsigned long total;
    /* the number of indexes in byte that are nonzero */
    unsigned int used;
} frequency_t;

/* returns: number of bytes read on succes, < 0 on error */
/* table is an OUT param to store data into */
/* in is the buffer to read and to collect stats */
/* in_len is the length of the input buffer */
long get_frequency(frequency_t *table, const unsigned char* in, long in_len);

/* prints a table to stdout for debug */
int print_table(const frequency_t *table);

/* print the shannon entropy of the table */
float get_entropy(const frequency_t *table);

frequency_t new_table();
