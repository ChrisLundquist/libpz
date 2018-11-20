typedef struct {
    int priority;
    const void *data;
} node_t;

typedef struct {
    node_t *nodes;
    int len;
    int size;
} heap_t;

void pq_push (heap_t *h, int priority, const void *data);
void *pq_pop (heap_t *h);
void pq_free(heap_t *h);
