#ifndef KMER_H
#define KMER_H

#include <vector>
#include <string>
#include <unordered_map>

class KMerTokenizer {
public:
  KMerTokenizer(int k_mers);
  std::vector<std::string> tokenize_sequence(const std::string &sequence);
  std::vector<int> encode(const std::string &sequence);
  std::string decode(const std::vector<int> &encoded_sequence);

  void set_vocab(const std::unordered_map<std::string, int> &vocab);
  std::unordered_map<std::string, int> get_vocab();

private:
  int k_mers;
  std::unordered_map<std::string, int> token_to_id;
  std::vector<std::string> id_to_token;
};

#endif