#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "frequency.h"
#include "huffman.h"
#include "pqueue.h"

/* Prints a node using weights as foreign keys */
static void node_print(const huffman_node_t* node) {
  if (node == NULL)
    return;

  unsigned parent = 0;
  unsigned left = 0;
  unsigned right = 0;
  if (node->parent)
    parent = node->parent->weight;
  if (node->left)
    left = node->left->weight;
  if (node->right)
    right = node->right->weight;

  fprintf(stderr,
          "weight: %u, value: %u, codeword: %u, code_bits: %u, parent: %u, "
          "left: %u, right: %u\n",
          node->weight, node->value, node->codeword, node->code_bits, parent,
          left, right);

  if (node->left)
    node_print(node->left);
  if (node->right)
    node_print(node->right);
}

void huff_print(const huffman_tree_t* tree, int leaves_only) {
  if (tree == NULL)
    return;

  if (leaves_only) {
    for (unsigned i = 0; i < tree->leave_count; ++i) {
      node_print(tree->leaves + i);
    }
  } else
    node_print(tree->root);
}

/* makes a new internal node for our huffman tree */
inline static huffman_node_t* merge_nodes(huffman_node_t* left,
                                          huffman_node_t* right) {
  if (left == NULL || right == NULL) {
    fprintf(stderr, "%s: left and right nodes are required\n", __func__);
    return NULL;
  }
  if (left && !right)
    return left;
  if (right && !left)
    return right;

  huffman_node_t* new = calloc(1, sizeof(huffman_node_t));
  if (new == NULL) {
    fprintf(stderr, "%s: malloc failed\n", __func__);
    return NULL;
  }
  unsigned weight = 0;
  if (left) {
    weight += left->weight;
  }
  if (right) {
    weight += right->weight;
  }

  *new = (huffman_node_t){
      .weight = weight,
      .value = 0,
      .parent = NULL,
      .left = left,
      .right = right,
  };
  if (left)
    left->parent = new;
  if (right)
    right->parent = new;
  return new;
}

inline static huffman_node_t* generate_leaves(frequency_t* table) {
  huffman_node_t* nodes = calloc(table->used, sizeof(huffman_node_t));
  if (nodes == NULL) {
    fprintf(stderr, "%s: malloc failed\n", __func__);
    return NULL;
  }

  /* here the total alphabet possibilities are 256,
   * however not ever character could be used.
   * we only want to generate leaves for the used alphabet characters */
  unsigned node_count = 0;
  for (unsigned i = 0; i < 256; ++i) {
    if (table->byte[i] == 0)  // unused, skip it
      continue;

    nodes[node_count] = (huffman_node_t){
        .value = i,
        .weight = table->byte[i],
    };
    node_count++;
  }
  return nodes;
}

static inline huffman_node_t* build_tree(frequency_t* table,
                                         huffman_node_t* leaves) {
  heap_t queue = {.nodes = NULL, .len = 0, .size = 0};
  for (unsigned i = 0; i < table->used; i++) {
    pq_push(&queue, leaves[i].weight, leaves + i);
  }

  huffman_node_t *left, *right, *root;
  root = NULL;
  while (queue.len > 1) {
    left = (huffman_node_t*)pq_pop(&queue);
    right = (huffman_node_t*)pq_pop(&queue);
    root = merge_nodes(left, right);
    pq_push(&queue, root->weight, root);
  }
  pq_free(&queue);
  return root;
}

static void generate_lookup_table(huffman_node_t* node,
                                  unsigned prefix,
                                  unsigned depth) {
  if (node == NULL)
    return;
  node->code_bits = depth;
  node->codeword = prefix;
  if (node->left) {
    const unsigned left_prefix = prefix << 1;
    generate_lookup_table(node->left, left_prefix, depth + 1);
  }

  if (node->right) {
    const unsigned right_prefix = (prefix << 1) + 1;
    generate_lookup_table(node->right, right_prefix, depth + 1);
  }
}

/* make a new huffman tree where the alphabet is 8 bits */
huffman_tree_t* huff_new_8(const unsigned char* in, unsigned in_len) {
  frequency_t table = new_table();
  get_frequency(&table, in, in_len);

  huffman_node_t* leaves = generate_leaves(&table);
  huffman_node_t* root = build_tree(&table, leaves);
  huffman_tree_t* tree = calloc(1, sizeof(huffman_tree_t));
  if (tree == NULL) {
    free(root);
    fprintf(stderr, "%s failed allocating tree\n", __func__);
    return NULL;
  }
  tree->root = root;
  tree->leaves = leaves;
  tree->leave_count = table.used;
  generate_lookup_table(root, 0, 0);

  return tree;
}

static void free_node(huffman_node_t* node) {
  if (node == NULL)
    return;

  if (node->right)
    free_node((huffman_node_t*)node->right);
  if (node->left)
    free_node((huffman_node_t*)node->left);  // discard const
  if (node->left || node->right)             // internal node
    free(node);
}

void huff_free(huffman_tree_t* tree) {
  if (tree->leaves) {
    free(tree->leaves);
    tree->leaves = NULL;
  }
  free_node(tree->root);
  tree->root = NULL;
  free(tree);
}

/* Returns the number of bits the encoded message took */
unsigned huff_Encode(const huffman_tree_t* tree,
                     const char* in,
                     unsigned in_len,
                     char* out,
                     unsigned out_len) {
  return 0;
}

/* returns the number of bits decoded? bytes remaining? */
unsigned huff_Decode(const huffman_tree_t* tree,
                     const char* in,
                     unsigned in_len,
                     char* out,
                     unsigned out_len) {
  return 0;
}
