#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "frequency.h"
#include "huffman.h"
#include "pqueue.h"

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

  fprintf(stderr, "weight: %u, value: %u, parent: %u, left: %u, right: %u\n",
          node->weight, node->value, parent, left, right);

  if (node->left)
    node_print(node->left);
  if (node->right)
    node_print(node->right);
}

void huff_print(const huffman_tree_t* tree) {
  if (tree == NULL)
    return;
  node_print(tree->root);
}

inline static huffman_node_t* merge_nodes(huffman_node_t* left,
                                          huffman_node_t* right) {
  if (left == NULL || right == NULL) {
    fprintf(stderr, "%s: left and right nodes are required\n", __func__);
    return NULL;
  }

  huffman_node_t* new = calloc(1, sizeof(huffman_node_t));
  if (new == NULL) {
    fprintf(stderr, "%s: malloc failed\n", __func__);
    return NULL;
  }
  unsigned weight = 0;
  if (left)
    weight += left->weight;
  if (right)
    weight += right->weight;

  *new = (huffman_node_t){
      .weight = weight,
      .value = 0,
      .parent = NULL,
      .left = left,
      .right = right,
  };
  left->parent = new;
  right->parent = new;
  return new;
}

inline static huffman_node_t* generate_leaves(frequency_t* table) {
  huffman_node_t* nodes = calloc(table->used, sizeof(huffman_node_t));
  if (nodes == NULL) {
    fprintf(stderr, "%s: malloc failed\n", __func__);
    return NULL;
  }

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

huffman_tree_t* huff_new_8(const unsigned char* in, unsigned in_len) {
  frequency_t table = new_table();
  get_frequency(&table, in, in_len);

  huffman_node_t* leaves = generate_leaves(&table);
  heap_t queue = {.nodes = NULL, .len = 0, .size = 0};
  for (unsigned i = 0; i < table.used; i++) {
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
  huffman_tree_t* tree = calloc(1, sizeof(huffman_tree_t));
  tree->root = root;
  tree->leaves = leaves;

  return tree;
}

static void free_node(huffman_node_t* node) {
  if (node == NULL)
    return;

  if (node->right)
    free_node((huffman_node_t*)node->right);
  if (node->left)
    free_node((huffman_node_t*)node->left);  // discard const
  if (node->left || node->right)  // internal node
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
