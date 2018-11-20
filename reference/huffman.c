#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "frequency.h"
#include "huffman.h"
#include "pqueue.h"

inline static huffman_node_t* merge_nodes(huffman_node_t* left,
                                          huffman_node_t* right) {
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

  for (unsigned i = 0; i < table->used; ++i) {
    if (table->byte[i] == 0)  // unused, skip it
      continue;

    nodes[i] = (huffman_node_t){
        .value = i,
        .weight = table->byte[i],
    };
  }
  return nodes;
}

inline static huffman_node_t* choose_next(huffman_node_t* left,
                                          unsigned* left_size,
                                          huffman_node_t* right,
                                          unsigned* right_size) {
  if (left_size == 0 && right_size == 0) {
    return NULL;
  } else if (right_size == 0) {
    --*left_size;
    return left;
  } else if (left_size == 0) {
    --*right_size;
    return right;
  }

  if (left->weight < right->weight) {
    --*left_size;
    return left;
  }
  --*right_size;
  return right;
}

huffman_tree_t* huff_new_8(const unsigned char* in, unsigned in_len) {
  frequency_t table = new_table();
  get_frequency(&table, in, in_len);

  huffman_node_t* leaves = generate_leaves(&table);
  heap_t queue = { .nodes = NULL, .len = 0, .size = 0};
  for(unsigned i = 0; i < table.used; i++) {
      pq_push(&queue, leaves[i].weight, leaves + i);
  }

  huffman_node_t *left, *right, *root;
  root = NULL;
  while(queue.len > 1) {
      left = (huffman_node_t*) pq_pop(&queue);
      right = (huffman_node_t*) pq_pop(&queue);
      root = merge_nodes(left, right);
      pq_push(&queue, root->weight, root);
  }
  pq_free(&queue);
  huffman_tree_t *tree = calloc(1, sizeof(huffman_tree_t));
  tree->root = root;
  tree->leaves = leaves;

  return tree;
}

static void free_node(huffman_node_t *node) {
    if(node == NULL)
        return;

    if(node->left)
        free_node((huffman_node_t*)node->left); //discard const
    if(node->right)
        free_node((huffman_node_t*)node->right);
    if(node->left || node->right) // internal node
        free(node);
}

void huff_free(huffman_tree_t* tree) {
    if(tree->leaves) {
        free(tree->leaves);
        tree->leaves = NULL;
    }
    free_node(tree->root);
    tree->root = NULL;
    free(tree);
}
