#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "kmer.h"
#include "inc/tqdm.h"

KMer* create_tokenizer(int kmers) {
  KMer* self = (KMer*)malloc(sizeof(KMer));
  memset(self->chars, 0, sizeof(self->chars));
  memset(self->special_tokens, 0, sizeof(self->special_tokens));

  // {a, t, g, c, \n} -> base pairs
  strcpy(self->chars, "ATGC\n");  // base characters
  
  // m -> mask token; p -> padding token; b -> begin; s -> separate; e -> end
  // not included the classification token, still tryna understand why tf is it used
  strcpy(self->special_tokens, "MPBSE");
  if (kmers > 6) {
    fprintf(stderr, "KMer size till 6 is supported for now due memory allocation issues\n");
    exit(1);
  }
  self->kmers = kmers;

  // vocab_size is basically ``summation from i=0 to n=chars_size len(self->chars)^kmers``, since we're trying to create each
  // possible token -> idx pair till the declared KMer size
  // so if kmer = 4:
  //        vocab_size = 5 + 25 + 125 + 625 = 780
  int vocab_size = 0;
  for (int i = 1; i <= kmers; i++) {
    vocab_size += pow(strlen(self->chars), i);
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
  const char *chars = tokenizer->chars;
  int num_chars = strlen(chars), max_k = tokenizer->kmers;
  int index = 0;
  tqdm bar;
  init_tqdm(&bar, "Building the vocab: ", false, "pairs", true, tokenizer->vocab_size, 1);

  for (int k = 1; k <= max_k; k++) {
    int *indices = (int *)malloc(k * sizeof(int));
    if (!indices) {
      fprintf(stderr, "Memory allocation failed\n");
      return;
    }
    char *combination = (char *)malloc((k + 1) * sizeof(char));
    combination[k] = '\0';

    for (int i = 0; i < k; i++) {
      indices[i] = 0;
    }

    while (1) {
      for (int i = 0; i < k; i++) {
        combination[i] = chars[indices[i]];
      }
      tokenizer->id_to_token[index] = strdup(combination);
      tokenizer->token_to_id[index] = index;
      index++;
      update_tqdm(&bar, 1, index == tokenizer->vocab_size);
      fflush(stdout);

      int i;
      for (i = k - 1; i >= 0; i--) {
        if (indices[i] < num_chars - 1) {
          indices[i]++;
          break;
        }
        indices[i] = 0;
      }
      if (i < 0) {
        break;
      }
    }
    free(indices);
    free(combination);
  }
  close_tqdm(&bar);
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

void save(KMer* tokenizer, const char* path) {
  FILE* file = fopen(path, "w");
  if (!file) {
    printf("Error opening file for saving model.\n");
    return;
  }

  char temp[MAX_TOKEN_SIZE];
  for (int i = 0; i < tokenizer->vocab_size; i++) {
    strncpy(temp, tokenizer->id_to_token[i], MAX_TOKEN_SIZE - 1);
    temp[MAX_TOKEN_SIZE - 1] = '\0';

    for (int j = 0; temp[j] != '\0'; j++) {
      if (temp[j] == '\n') {
        temp[j] = 'n';
      }
    }
    fprintf(file, "\"%s\" %d\n", temp, i + 1);
  }
  fclose(file);
  printf("Model saved to %s\n", path);
}

void load(KMer* tokenizer, const char* path) {
  FILE* file = fopen(path, "r");
  if (!file) {
    fprintf(stderr, "Error opening file for loading model.\n");
    exit(1);
  }
  char buffer[100];
  int index;

  while (fscanf(file, "\"%[^\"]\" %d", buffer, &index) == 2) {
    // converting 'n' back to '\n' in the loaded token
    for (int i = 0; buffer[i] != '\0'; i++) {
      if (buffer[i] == 'n') {
        buffer[i] = '\n';
      }
    }
    // storing the reconstructed token
    tokenizer->id_to_token[index - 1] = strdup(buffer);
    tokenizer->token_to_id[index - 1] = index - 1;
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