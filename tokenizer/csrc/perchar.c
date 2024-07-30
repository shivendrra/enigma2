#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB_SIZE 9

typedef struct {
  char chars[VOCAB_SIZE];
  int vocab_size;
  int string_to_index[256];
  char index_to_string[VOCAB_SIZE + 256];
  int special_index;
} PerCharTokenizer;

void init_tokenizer(PerCharTokenizer *tokenizer) {
  char base_chars[VOCAB_SIZE] = {'\n', 'A', 'T', 'G', 'C', 'P', 'M', 'U', ' '};
  memcpy(tokenizer->chars, base_chars, VOCAB_SIZE);
  tokenizer->vocab_size = VOCAB_SIZE;
  tokenizer->special_index = VOCAB_SIZE;

  for (int i = 0; i < 256; i++) {
    tokenizer->string_to_index[i] = -1;
  }

  for (int i = 0; i < VOCAB_SIZE; i++) {
    tokenizer->string_to_index[(unsigned char)base_chars[i]] = i;
    tokenizer->index_to_string[i] = base_chars[i];
  }
}

int* encode(PerCharTokenizer *tokenizer, const char *string, int *encoded_length) {
  int length = strlen(string);
  int *encoded = (int *)malloc(length * sizeof(int));
  if (!encoded) {
    fprintf(stderr, "Memory allocation failed\n");
    return NULL;
  }

  for (int i = 0; i < length; i++) {
    unsigned char ch = (unsigned char)string[i];
    if (tokenizer->string_to_index[ch] != -1) {
      encoded[i] = tokenizer->string_to_index[ch];
    } else {
      tokenizer->string_to_index[ch] = tokenizer->special_index;
      tokenizer->index_to_string[tokenizer->special_index] = ch;
      encoded[i] = tokenizer->special_index;
      tokenizer->special_index++;
    }
  }
  *encoded_length = length;
  return encoded;
}

char* decode(PerCharTokenizer *tokenizer, int *encoded, int length) {
  char *decoded = (char *)malloc((length + 1) * sizeof(char));
  if (!decoded) {
    fprintf(stderr, "Memory allocation failed\n");
    return NULL;
  }

  for (int i = 0; i < length; i++) {
    if (encoded[i] < tokenizer->special_index) {
      decoded[i] = tokenizer->index_to_string[encoded[i]];
    } else {
      decoded[i] = '?';
    }
  }
  decoded[length] = '\0';
  return decoded;
}

int main() {
  PerCharTokenizer tokenizer;
  init_tokenizer(&tokenizer);

  const char *string = "AACATGTCCTGCATGGCATTAGTTTGTTGGGGCAGTGCCCGGATAGCATCAACGCTGCGCTGATTTGCCGTGGCGAGAAA";
  int encoded_length;
  int *encoded = encode(&tokenizer, string, &encoded_length);

  printf("Encoded: ");
  for (int i = 0; i < encoded_length; i++) {
    printf("%d ", encoded[i]);
  }
  printf("\n");

  char *decoded = decode(&tokenizer, encoded, encoded_length);
  printf("Decoded: %s\n", decoded);

  free(encoded);
  free(decoded);
  return 0;
}