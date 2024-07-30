#include "kmer.h"
#include <iostream>
#include <cmath>
#include <algorithm>

KMerTokenizer::KMerTokenizer(int k_mers) : k_mers(k_mers) {}

std::vector<std::string> KMerTokenizer::tokenize_sequence(const std::string &sequence) {
  std::vector<std::string> kmers;
  for (size_t i = 0; i < sequence.size(); i += k_mers) {
    kmers.push_back(sequence.substr(i, k_mers));
  }
  return kmers;
}

std::vector<int> KMerTokenizer::encode(const std::string &sequence) {
  std::vector<int> encoded_sequence;
  std::vector<std::string> kmers = tokenize_sequence(sequence);
  for (const auto &kmer : kmers) {
    if (token_to_id.find(kmer) != token_to_id.end()) {
      encoded_sequence.push_back(token_to_id[kmer]);
    } else {
      encoded_sequence.push_back(token_to_id.size() + 1);
    }
  }
  return encoded_sequence;
}

std::string KMerTokenizer::decode(const std::vector<int> &encoded_sequence) {
  std::string decoded_tokens;
  for (const auto &token_id : encoded_sequence) {
    if (token_id < id_to_token.size()) {
      decoded_tokens += id_to_token[token_id];
    } else {
      decoded_tokens += "";
    }
  }
  return decoded_tokens;
}

void KMerTokenizer::set_vocab(const std::unordered_map<std::string, int> &vocab) {
  token_to_id = vocab;
  id_to_token.resize(vocab.size());
  for (const auto &pair : vocab) {
    id_to_token[pair.second] = pair.first;
  }
}

std::unordered_map<std::string, int> KMerTokenizer::get_vocab() {
  return token_to_id;
}