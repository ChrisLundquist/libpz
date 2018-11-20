typedef struct lz77_match {
  unsigned offset;
  unsigned length;
  char next;
} lz77_match_t;

void PrintMatch(const lz77_match_t* match);
