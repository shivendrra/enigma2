#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kmer.h"

KMer* create_tokenizer(int kmers) {
  KMer* self = (KMer*)malloc(sizeof(KMer));
  self->kmers = kmers;
  self->vocab_size = 0;
  self->id_to_token = (char**)malloc(MAX_VOCAB_SIZE * sizeof(char*));
  self->token_to_id = (int*)malloc(MAX_VOCAB_SIZE * sizeof(int));
  return self;
}

void tokenize_sequence(KMer* tokenizer, const char* data, char*** kmers, int* n_kmers) {
  int len = strlen(data);
  int k = tokenizer->kmers;
  *n_kmers = len/k;
  *kmers = (char**)malloc(*n_kmers * sizeof(char*));
  for (int i = 0; i < *n_kmers; i++) {
    (*kmers)[i] = (char*)malloc((k + 1) * sizeof(char));
    strncpy((*kmers)[i], data + i * k, k);
    (*kmers)[i][k] = '\0';
  }
}

void build_vocab(KMer* tokenizer, const char* seq, int n_seq) {
  char** all_kmers;
  int total_kmers = 0;

  for (int i = 0; i < n_seq; i++) {
    char** kmers;
    int n_kmers;
    tokenize_sequence(tokenizer, seq, &kmers, &n_kmers);
    all_kmers = (char**)realloc(all_kmers, (n_kmers + total_kmers) * sizeof(char*));
    for (int j = 0; j < n_kmers; j++) {
      all_kmers[total_kmers++] = kmers[j];
    }
    free(kmers);
  }
  tokenizer->vocab_size = total_kmers < MAX_VOCAB_SIZE ? total_kmers : MAX_VOCAB_SIZE;
  for (int i = 0; i < tokenizer->vocab_size; i++) {
    tokenizer->id_to_token[i] = all_kmers[i];
    tokenizer->token_to_id[i] = i;
  }
  free(all_kmers);
}

int* encode(KMer* tokenizer, const char* seq, int* encoded_size) {
  char** kmers;
  int n_kmers;
  tokenize_sequence(tokenizer, seq, &kmers, &n_kmers);
  *encoded_size = n_kmers;
  int* encoded_seq = (int*)malloc(n_kmers * sizeof(int));
  for (int i = 0; i < n_kmers; i++) {
    for (int j = 0; j < tokenizer->vocab_size; j++) {
      if (strcmp(kmers[i], tokenizer->id_to_token[j]) == 0) {
        encoded_seq[i] = j;
        break;
      }
    }
    free(kmers[i]);
  }
  free(kmers);
  return encoded_seq;
}

char *decode_sequence(KMer* tokenizer, const int* encoded_sequence, int encoded_size) {
  int k = tokenizer->kmers;
  char* decoded_sequence = (char *)malloc((encoded_size * k + 1) * sizeof(char));
  decoded_sequence[0] = '\0';
  for (int i = 0; i < encoded_size; i++) {
    strcat(decoded_sequence, tokenizer->id_to_token[encoded_sequence[i]]);
  }
  return decoded_sequence;
}

void save_model(KMer* tokenizer, const char* path) {
  FILE* file = fopen(path, "w");
  if (!file) {
    printf("Error opening file for saving model.\n");
    return;
  }
  fprintf(file, "{\"k_mers\": %d, \"vocab\": [", tokenizer->kmers);
  for (int i = 0; i < tokenizer->vocab_size; i++) {
    fprintf(file, "\"%s\"%s", tokenizer->id_to_token[i], (i == tokenizer->vocab_size - 1) ? "" : ", ");
  }
  fprintf(file, "]}\n");
  fclose(file);
  printf("Model saved to %s\n", path);
}

void load_model(KMer* tokenizer, const char* path) {
  FILE* file = fopen(path, "r");
  if (!file) {
    printf("Error opening file for loading model.\n");
    return;
  }
  fscanf(file, "{\"kmers\": %d, \"vocab\": [", &tokenizer->kmers);
  tokenizer->vocab_size = 0;
  while (fscanf(file, "\"%[^\"]\"", tokenizer->id_to_token[tokenizer->vocab_size]) == 1) {
    tokenizer->token_to_id[tokenizer->vocab_size] = tokenizer->vocab_size;
    tokenizer->vocab_size++;
    if (fgetc(file) == ']') break;
  }
  fclose(file);
  printf("Model loaded from %s\n", path);
}

void free_tokenizer(KMer* tokenizer) {
  for (int i = 0; i < tokenizer->vocab_size; i++) {
    free(tokenizer->id_to_token[i]);
  }
  free(tokenizer->id_to_token);
  free(tokenizer->token_to_id);
  free(tokenizer);
}