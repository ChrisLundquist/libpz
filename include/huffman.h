/* union symbol {
    unsigned char c;
    unsigned short s;
    unsigned int i;
};
*/

typedef struct huffman_node {
    unsigned weight;
    unsigned value;
    unsigned int codeword;
    char code_bits;
    const struct huffman_node *left;
    const struct huffman_node *right;
    const struct huffman_node *parent;
} huffman_node_t;

typedef struct huffman_tree {
    huffman_node_t* root;
    huffman_node_t* leaves;
    unsigned leave_count;
} huffman_tree_t;

/* returns the root of the constructed tree */
huffman_tree_t* huff_new_8(const unsigned char* in, unsigned in_len);
huffman_tree_t* huff_new_16(const unsigned short* in, unsigned in_len);
huffman_tree_t* huff_new_32(const unsigned int* in, unsigned in_len);

void huff_free(huffman_tree_t* tree);
void huff_print(const huffman_tree_t* tree, int leaves_only);

/* returns the number of bits decoded? bytes remaining? */
unsigned huff_Decode(const huffman_tree_t* tree, const char* in, unsigned in_len, char *out, unsigned out_len);
unsigned huff_Encode(const huffman_tree_t* tree, const char* in, unsigned in_len, char *out, unsigned out_len);
