#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "kmer.h"
#include "inc/tqdm.h"

KMer* create_tokenizer(int kmers) {
  KMer* self = (KMer*)malloc(sizeof(KMer));
  // {a, t, g, c, \n} -> base pairs
  strcpy(self->chars, "\nATGC");  // base characters
  
  // m -> mask token; p -> padding token; b -> begin; s -> separate; e -> end
  // not included the classification token, still tryna understand why tf is it used
  strcpy(self->special_tokens, "MPBSE")
  self->kmers = kmers;

  // vocab_size is basically ``summation from i=0 to n=chars_size len(self->chars)^kmers``, since we're trying to create each
  // possible token -> idx pair till the declared KMer size
  // so if kmer = 4:
  //        vocab_size = 5 + 25 + 125 + 625 = 780
  int vocab_size = 0;
  for (int i = 0; i <= strlen(self->chars); i++) {
    vocab_size += pow(i, kmers);
  }
  self->vocab_size = vocab_size;
  self->id_to_token = (char**)malloc((vocab_size + 1) * sizeof(char*));
  self->token_to_id = (int*)malloc((vocab_size + 1) * sizeof(int));
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

void build_vocab(KMer* tokenizer) {
  int* ids = (int*)malloc(tokenizer->vocab_size * sizeof(int));
  int str_len = strlen(tokenizer->chars) + 1;
  char* combination = (char*)malloc(str_len * sizeof(char));
  combination[str_len] = '\0'; // setting the last index as null
  for (int i = 0; i < str_len; i++) {
    ids[i] = 0;
  }
  int index = 1;
  while(1) {
    for (int i = 0; i < str_len; i++) {
      combination[i] = tokenizer->chars[ids[i]];
    }
    for (int i = 0; i < str_len; i++) {
      if (combination[i] == '\n') {
        combination[i] = 'n';  // replacing the new_line char with 'n'
      }
    }
  }
}

int* encode_sequence(KMer* tokenizer, const char* seq, int* encoded_size) {
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
  tokenizer->vocab_size = 0;
  char buffer[100];
  int index;
  while (fscanf(file, "%s %d", buffer, &index) == 2) {
    tokenizer->id_to_token[index] = strdup(buffer);
    tokenizer->token_to_id[index] = index;
    tokenizer->vocab_size++;
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