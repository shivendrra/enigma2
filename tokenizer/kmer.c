#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <json-c/json.h>

#define MAX_VOCAB_SIZE 100000
#define MAX_SEQ_LEN 10000

typedef struct {
  int k_mers;
  int vocab_size;
  char **id_to_token;
  int *token_to_id;
} KMerTokenizer;

void init_tokenizer(KMerTokenizer *tokenizer, int k_mers) {
  tokenizer->k_mers = k_mers;
  tokenizer->vocab_size = 0;
  tokenizer->id_to_token = (char **)malloc(MAX_VOCAB_SIZE * sizeof(char *));
  tokenizer->token_to_id = (int *)malloc(MAX_SEQ_LEN * sizeof(int));
  for (int i = 0; i < MAX_SEQ_LEN; i++) {
    tokenizer->token_to_id[i] = -1;
  }
}

void free_tokenizer(KMerTokenizer *tokenizer) {
  for (int i = 0; i < tokenizer->vocab_size; i++) {
    free(tokenizer->id_to_token[i]);
  }
  free(tokenizer->id_to_token);
  free(tokenizer->token_to_id);
}

char** tokenize_sequence(const char *sequence, int k_mers, int *num_kmers) {
  int len = strlen(sequence);
  *num_kmers = (len + k_mers - 1) / k_mers; // ceil(len / k_mers)
  char **kmers = (char **)malloc(*num_kmers * sizeof(char *));
  for (int i = 0; i < *num_kmers; i++) {
    kmers[i] = (char *)malloc((k_mers + 1) * sizeof(char));
    strncpy(kmers[i], sequence + i * k_mers, k_mers);
    kmers[i][k_mers] = '\0';
  }
  return kmers;
}

void build_vocab(KMerTokenizer *tokenizer, char **sequences, int num_sequences) {
  char **all_kmers = (char **)malloc(MAX_SEQ_LEN * MAX_VOCAB_SIZE * sizeof(char *));
  int total_kmers = 0;
  for (int i = 0; i < num_sequences; i++) {
    int num_kmers;
    char **kmers = tokenize_sequence(sequences[i], tokenizer->k_mers, &num_kmers);
    for (int j = 0; j < num_kmers; j++) {
      all_kmers[total_kmers++] = kmers[j];
    }
    free(kmers);
  }

  int *token_count = (int *)calloc(MAX_VOCAB_SIZE, sizeof(int));
  for (int i = 0; i < total_kmers; i++) {
    bool found = false;
    for (int j = 0; j < tokenizer->vocab_size; j++) {
      if (strcmp(all_kmers[i], tokenizer->id_to_token[j]) == 0) {
        token_count[j]++;
        found = true;
        break;
      }
    }
    if (!found && tokenizer->vocab_size < MAX_VOCAB_SIZE) {
      tokenizer->id_to_token[tokenizer->vocab_size] = all_kmers[i];
      tokenizer->token_to_id[all_kmers[i][0]] = tokenizer->vocab_size;
      token_count[tokenizer->vocab_size]++;
      tokenizer->vocab_size++;
    }
  }
  free(token_count);
  free(all_kmers);
}

int* encode(KMerTokenizer *tokenizer, const char *sequence, int *encoded_length) {
  int num_kmers;
  char **kmers = tokenize_sequence(sequence, tokenizer->k_mers, &num_kmers);
  int *encoded_sequence = (int *)malloc(num_kmers * sizeof(int));
  for (int i = 0; i < num_kmers; i++) {
    if (tokenizer->token_to_id[kmers[i][0]] != -1) {
      encoded_sequence[i] = tokenizer->token_to_id[kmers[i][0]];
    } else {
      encoded_sequence[i] = -1; // unknown token
    }
    free(kmers[i]);
  }
  free(kmers);
  *encoded_length = num_kmers;
  return encoded_sequence;
}

char* decode(KMerTokenizer *tokenizer, int *encoded_sequence, int length) {
  char *decoded_tokens = (char *)malloc((length * tokenizer->k_mers + 1) * sizeof(char));
  int pos = 0;
  for (int i = 0; i < length; i++) {
    if (encoded_sequence[i] >= 0 && encoded_sequence[i] < tokenizer->vocab_size) {
      strcpy(decoded_tokens + pos, tokenizer->id_to_token[encoded_sequence[i]]);
      pos += tokenizer->k_mers;
    } else {
      for (int j = 0; j < tokenizer->k_mers; j++) {
        decoded_tokens[pos++] = '?'; // unknown token
      }
    }
  }
  decoded_tokens[pos] = '\0';
  return decoded_tokens;
}

void save_model(KMerTokenizer *tokenizer, const char *model_path) {
  FILE *file = fopen(model_path, "w");
  if (!file) {
    perror("Could not open file for writing");
    return;
  }

  json_object *jobj = json_object_new_object();
  for (int i = 0; i < tokenizer->vocab_size; i++) {
    json_object_object_add(jobj, tokenizer->id_to_token[i], json_object_new_int(i));
  }

  fprintf(file, "%s", json_object_to_json_string(jobj));
  json_object_put(jobj);
  fclose(file);
  printf("Saved the vocab!\n");
}

void load_model(KMerTokenizer *tokenizer, const char *path) {
  FILE *file = fopen(path, "r");
  if (!file) {
    perror("Could not open file for reading");
    return;
  }

  fseek(file, 0, SEEK_END);
  long file_size = ftell(file);
  fseek(file, 0, SEEK_SET);

  char *file_contents = (char *)malloc(file_size + 1);
  fread(file_contents, 1, file_size, file);
  file_contents[file_size] = '\0';
  fclose(file);

  json_object *jobj = json_tokener_parse(file_contents);
  tokenizer->vocab_size = json_object_object_length(jobj);
  json_object_object_foreach(jobj, key, val) {
    tokenizer->id_to_token[json_object_get_int(val)] = strdup(key);
    tokenizer->token_to_id[key[0]] = json_object_get_int(val);
  }

  json_object_put(jobj);
  free(file_contents);
  printf("Loaded the vocab!\n");
}

int main() {
  KMerTokenizer tokenizer;
  init_tokenizer(&tokenizer, 4);

  const char *sequences[] = {"ATGCGTAC", "GTCAGTAC"};
  // const char *string = "AACATGTCCTGCATGGCATTAGTTTGTTGGGGCAGTGCCCGGATAGCATCAACGCTGCGCTGATTTGCCGTGGCGAGAAA";
  build_vocab(&tokenizer, (char **)sequences, 2);

  const char *sequence = "ATGCGTAC";
  int encoded_length;
  int *encoded = encode(&tokenizer, sequence, &encoded_length);

  printf("Encoded: ");
  for (int i = 0; i < encoded_length; i++) {
    printf("%d ", encoded[i]);
  }
  printf("\n");

  char *decoded = decode(&tokenizer, encoded, encoded_length);
  printf("Decoded: %s\n", decoded);

  save_model(&tokenizer, "vocab.json");
  load_model(&tokenizer, "vocab.json");

  free(encoded);
  free(decoded);
  free_tokenizer(&tokenizer);

  return 0;
}