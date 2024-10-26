#ifndef KMER_H
#define KMER_H

#include <vector>
#include <string>
#include <unordered_map>

// class in C++ scope, not inside extern "C".
class KMerTokenizer {
public:
  KMerTokenizer(int k_mers);
  std::vector<std::string> tokenize_sequence(const std::string &sequence);
  std::vector<int> encode(const std::string &sequence);
  std::string decode(const std::vector<int> &encoded_sequence);

  void set_vocab(const std::unordered_map<std::string, int> &vocab);
  void set_vocab_from_file(const std::string &filename);
  std::unordered_map<std::string, int> get_vocab();

private:
  int k_mers;
  std::unordered_map<std::string, int> token_to_id;
  std::vector<std::string> id_to_token;
  int vocab_size;
};

// c-compatible api for the shared library
extern "C" {
  KMerTokenizer* KMerTokenizer_new(int k_mers);
  void KMerTokenizer_delete(KMerTokenizer* obj);

  char** KMerTokenizer_tokenize_sequence(KMerTokenizer* obj, const char* sequence);
  void KMerTokenizer_free_tokens(char** tokens);

  int* KMerTokenizer_encode(KMerTokenizer* obj, const char* sequence);
  void KMerTokenizer_free_int_array(int* array);

  char* KMerTokenizer_decode(KMerTokenizer* obj, const int* encoded_sequence, int length);
  void KMerTokenizer_set_vocab(KMerTokenizer* obj, const char* filename);
  char* KMerTokenizer_get_vocab(KMerTokenizer* obj);
}

#endif