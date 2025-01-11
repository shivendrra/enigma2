#include <stdio.h>
#include <stdlib.h>
#include "kmer.h"

int main() {
  KMer* tokenizer = create_tokenizer(4);

  // Test sequence
  const char* sequence = "AACATGTCCTGCATGGCATTAGTTTGTTGGGGCAGTGCCCGGATAGCATCAACGCTGCGCTGATTTGCCGTGGCGAGAAA";

  char** kmers;
  int num_kmers;
  tokenize_sequence(tokenizer, sequence, &kmers, &num_kmers);

  printf("Tokenized k-mers:\n");
  for (int i = 0; i < num_kmers; i++) {
    printf("%s\n", kmers[i]);
    free(kmers[i]);
  }
  free(kmers);

  // Build vocabulary
  const char* sequences[] = {sequence};
  build_vocab(tokenizer, sequences, 1);

  printf("\nVocabulary:\n");
  for (int i = 0; i < tokenizer->vocab_size; i++) {
    printf("%s -> %d\n", tokenizer->id_to_token[i], tokenizer->token_to_id[i]);
  }

  // Encode the sequence
  int* encoded;
  int encoded_size;
  encoded = encode_sequence(tokenizer, sequence, &encoded_size);

  printf("\nEncoded sequence:\n");
  for (int i = 0; i < encoded_size; i++) {
    printf("%d ", encoded[i]);
  }
  printf("\n");

  // Decode the encoded sequence
  char* decoded = decode_sequence(tokenizer, encoded, encoded_size);
  printf("\nDecoded sequence:\n%s\n", decoded);

  free(encoded);
  free(decoded);
  free_tokenizer(tokenizer);

  return 0;
}