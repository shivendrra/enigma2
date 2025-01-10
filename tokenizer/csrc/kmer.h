#ifndef __KMER__H__
#define __KMER__H__

#include <stddef.h>

#define  MAX_TOKEN_SIZE  100
#define  MAX_VOCAB_SIZE  10000

typedef struct {
  size_t kmers;
  size_t vocab_size;
  char** id_to_token;
  int* token_to_id;
} KMer;

extern "C" {
  KMer* create_tokenizer(size_t kmers);
  void tokenize_sequence(KMer* tokenizer, const char* data, char*** seq, size_t* n_kmers);
  void build_vocab(KMer* tokenizer, const char* seq, size_t n_seq);
  int* encode(KMer* tokenizer, const char* seq, size_t* encoded_size);
  char* decode(KMer* tokenizer, const int* encoded_seq, size_t* encoded_size);
  void save(KMer* tokenizer, const char* path);
  void load(KMer* tokenizer, const char* path);
  void free_tokenizer(KMer* tokenizer);
}


#endif  //!__KMER__H__