#ifndef __KMER_H__
#define __KMER_H__

#define MAX_VOCAB_SIZE 10000
#define MAX_LINE_LENGTH 2048
#define MAX_SPECIAL_TOKENS 100

typedef struct {
  int idx;
  char* value;
} VocabEntry;

typedef struct {
  int idx1;
  int idx2;
} Pair;

typedef struct {
  Pair pair;
  int idx;
} MergeEntry;

typedef struct {
  VocabEntry vocab[MAX_VOCAB_SIZE];
  MergeEntry merges[MAX_VOCAB_SIZE];
  int vocab_size;
  int merge_count;
  int special_token_count;
  char special_tokens[MAX_SPECIAL_TOKENS][MAX_LINE_LENGTH];
} KMerTokenizer;

void init_tokenizer(KMerTokenizer* tokenizer, int k_mers);
void tokenize_sequence(const char* sequence, int k_mers, char*** tokens, int* token_count);
void build_vocab(KMerTokenizer* tokenizer, const char** sequences, int sequence_count);
void encode_sequence(KMerTokenizer* tokenizer, const char* sequence, int** encoded, int* encoded_size);
void decode_sequence(KMerTokenizer* tokenizer, const int* encoded, int encoded_size, char** decoded);
void save_model(KMerTokenizer* tokenizer, const char* model_path);
void load_model(KMerTokenizer* tokenizer, const char* model_path);
void free_tokenizer(KMerTokenizer* tokenizer);

#endif