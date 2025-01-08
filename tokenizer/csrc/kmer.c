#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "kmer.h"

void init_tokenizer(KMerTokenizer* tokenizer, int k_mers) {
  tokenizer->vocab_size = 0;
  tokenizer->merge_count = 0;
  tokenizer->special_token_count = 0;
  for (int i = 0; i < MAX_VOCAB_SIZE; ++i) {
    tokenizer->vocab[i].value = NULL;
  }
}

void tokenize_sequence(const char* sequence, int k_mers, char*** tokens, int* token_count) {
  int len = strlen(sequence);
  *token_count = len / k_mers + (len % k_mers != 0);
  *tokens = (char**)malloc(*token_count * sizeof(char*));

  for (int i = 0; i < *token_count; ++i) {
    (*tokens)[i] = (char*)malloc((k_mers + 1) * sizeof(char));
    strncpy((*tokens)[i], &sequence[i * k_mers], k_mers);
    (*tokens)[i][k_mers] = '\0';
  }
}

void build_vocab(KMerTokenizer* tokenizer, const char** sequences, int sequence_count) {
  int token_count;
  char** tokens;

  for (int i = 0; i < sequence_count; ++i) {
    tokenize_sequence(sequences[i], tokenizer->vocab_size, &tokens, &token_count);
    for (int j = 0; j < token_count; ++j) {
      int found = 0;
      for (int k = 0; k < tokenizer->vocab_size; ++k) {
        if (strcmp(tokens[j], tokenizer->vocab[k].value) == 0) {
          found = 1;
          break;
        }
      }
      if (!found) {
        tokenizer->vocab[tokenizer->vocab_size].idx = tokenizer->vocab_size;
        tokenizer->vocab[tokenizer->vocab_size].value = strdup(tokens[j]);
        tokenizer->vocab_size++;
      }
    }
    for (int j = 0; j < token_count; ++j) {
      free(tokens[j]);
    }
    free(tokens);
  }
}

void encode_sequence(KMerTokenizer* tokenizer, const char* sequence, int** encoded, int* encoded_size) {
  char** tokens;
  int token_count;
  tokenize_sequence(sequence, tokenizer->vocab_size, &tokens, &token_count);

  *encoded = (int*)malloc(token_count * sizeof(int));
  *encoded_size = token_count;

  for (int i = 0; i < token_count; ++i) {
    (*encoded)[i] = -1;
    for (int j = 0; j < tokenizer->vocab_size; ++j) {
      if (strcmp(tokens[i], tokenizer->vocab[j].value) == 0) {
        (*encoded)[i] = tokenizer->vocab[j].idx;
        break;
      }
    }
    free(tokens[i]);
  }
  free(tokens);
}

void decode_sequence(KMerTokenizer* tokenizer, const int* encoded, int encoded_size, char** decoded) {
  *decoded = (char*)malloc(encoded_size * tokenizer->vocab_size + 1);
  (*decoded)[0] = '\0';

  for (int i = 0; i < encoded_size; ++i) {
    if (encoded[i] >= 0 && encoded[i] < tokenizer->vocab_size) {
      strcat(*decoded, tokenizer->vocab[encoded[i]].value);
    }
  }
}

void save_model(KMerTokenizer* tokenizer, const char* model_path) {
  FILE* fp = fopen(model_path, "w");
  if (!fp) {
    perror("Failed to save model");
    return;
  }
  fprintf(fp, "%d\n", tokenizer->vocab_size);
  for (int i = 0; i < tokenizer->vocab_size; ++i) {
    fprintf(fp, "%s\n", tokenizer->vocab[i].value);
  }
  fclose(fp);
}

void load_model(KMerTokenizer* tokenizer, const char* model_path) {
  FILE* fp = fopen(model_path, "r");
  if (!fp) {
    perror("Failed to load model");
    return;
  }
  fscanf(fp, "%d\n", &tokenizer->vocab_size);
  for (int i = 0; i < tokenizer->vocab_size; ++i) {
    char buffer[MAX_LINE_LENGTH];
    fgets(buffer, MAX_LINE_LENGTH, fp);
    buffer[strcspn(buffer, "\n")] = '\0';
    tokenizer->vocab[i].value = strdup(buffer);
    tokenizer->vocab[i].idx = i;
  }
  fclose(fp);
}

void free_tokenizer(KMerTokenizer* tokenizer) {
  for (int i = 0; i < tokenizer->vocab_size; ++i) {
    free(tokenizer->vocab[i].value);
  }
}