typedef struct {
    unsigned int byte[256];
    unsigned long total;
    unsigned int used;
} frequency_t;

/* returns: number of bytes read on succes, < 0 on error */
/* table is an OUT param to store data into */
/* in is the buffer to read and to collect stats */
/* in_len is the length of the input buffer */
long get_frequency(frequency_t *table, const unsigned char* in, long in_len);

/* prints a table to stdout for debug */
int print_table(const frequency_t *table);

float get_entropy(const frequency_t *table);

frequency_t new_table();
