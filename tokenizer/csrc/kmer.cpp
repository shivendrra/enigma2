#include "kmer.h"
#include <cstring>  // for strdup
#include <iostream> // for debugging
#include <fstream>  // for file I/O
#include <sstream>  // for string stream

KMerTokenizer::KMerTokenizer(int k_mers) : k_mers(k_mers), vocab_size(0) {}

// Tokenizes a sequence into k-mers
std::vector<std::string> KMerTokenizer::tokenize_sequence(const std::string &sequence) {
  std::vector<std::string> tokens;
  for (size_t i = 0; i <= sequence.size() - k_mers; ++i) {
    tokens.push_back(sequence.substr(i, k_mers));
  }
  return tokens;
}

// Encodes a sequence into integers based on the vocabulary
std::vector<int> KMerTokenizer::encode(const std::string &sequence) {
  std::vector<int> encoded;
  for (const auto &token : tokenize_sequence(sequence)) {
    if (token_to_id.find(token) != token_to_id.end()) {
      encoded.push_back(token_to_id[token]);
    } else {
      std::cerr << "Warning: Token '" << token << "' not found in vocab.\n";
      encoded.push_back(-1);  // use -1 to indicate unknown token
    }
  }
  return encoded;
}

// Decodes a sequence of integers back into tokens
std::string KMerTokenizer::decode(const std::vector<int> &encoded_sequence) {
  std::string decoded;
  for (int id : encoded_sequence) {
    if (id >= 0 && id < static_cast<int>(id_to_token.size())) {
      decoded += id_to_token[id];
    } else {
      std::cerr << "Warning: Invalid token ID " << id << ".\n";
    }
  }
  return decoded;
}

// Sets the vocabulary from a key-value format
void KMerTokenizer::set_vocab(const std::unordered_map<std::string, int> &vocab) {
  if (!this) {
    std::cerr << "KMerTokenizer object is invalid." << std::endl;
    return;
  }
  token_to_id = vocab;
  id_to_token.resize(vocab.size());
  for (const auto &[token, id] : vocab) {
    id_to_token[id] = token;
  }
  vocab_size = vocab.size();
}

// Loads vocabulary from a text file
void KMerTokenizer::set_vocab_from_file(const std::string &filename) {
  std::cerr << "Attempting to open vocab file: " << filename << std::endl;

  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open vocab file: " << filename << std::endl;
    return;
  }

  std::string line;
  std::string token;
  int id;

  while (std::getline(file, line)) {
    std::size_t first_quote = line.find('"');
    std::size_t last_quote = line.rfind('"');

    if (first_quote == std::string::npos || last_quote == std::string::npos || first_quote == last_quote) {
      std::cerr << "Malformed line: " << line << std::endl;
      continue;
    }

    token = line.substr(first_quote + 1, last_quote - first_quote - 1);
    size_t pos;
    while ((pos = token.find("\\n")) != std::string::npos) {
      token.replace(pos, 2, "\n");
    }

    std::istringstream stream(line.substr(last_quote + 1));
    if (stream >> id) {
      token_to_id[token] = id;
      id_to_token.resize(std::max(id + 1, static_cast<int>(id_to_token.size())));
      id_to_token[id] = token;
    } else {
      std::cerr << "Error reading line: " << line << std::endl;
    }
  }

  vocab_size = token_to_id.size();
  std::cout << "Vocab loaded with " << vocab_size << " tokens!" << std::endl;
}

std::unordered_map<std::string, int> KMerTokenizer::get_vocab() {
  return token_to_id;
}

// c-compatible functions for shared library
extern "C" {
  KMerTokenizer* KMerTokenizer_new(int k_mers) {
    return new KMerTokenizer(k_mers);
  }

  void KMerTokenizer_delete(KMerTokenizer* obj) {
    delete obj;
  }

  char** KMerTokenizer_tokenize_sequence(KMerTokenizer* obj, const char* sequence) {
    std::vector<std::string> tokens = obj->tokenize_sequence(sequence);
    char** result = new char*[tokens.size() + 1];
    for (size_t i = 0; i < tokens.size(); ++i) {
      result[i] = strdup(tokens[i].c_str());
    }
    result[tokens.size()] = nullptr;  // null-terminated array
    return result;
  }

  void KMerTokenizer_free_tokens(char** tokens) {
    for (int i = 0; tokens[i] != nullptr; ++i) {
      free(tokens[i]);
    }
    delete[] tokens;
  }

  int* KMerTokenizer_encode(KMerTokenizer* obj, const char* sequence) {
    std::vector<int> encoded = obj->encode(sequence);
    int* result = new int[encoded.size() + 1];
    for (size_t i = 0; i < encoded.size(); ++i) {
      result[i] = encoded[i];
    }
    result[encoded.size()] = -1;  // sentinel value
    return result;
  }

  void KMerTokenizer_free_int_array(int* array) {
    delete[] array;
  }

  char* KMerTokenizer_decode(KMerTokenizer* obj, const int* encoded_sequence, int length) {
    std::vector<int> encoded(encoded_sequence, encoded_sequence + length);
    std::string decoded = obj->decode(encoded);
    char* result = strdup(decoded.c_str());
    return result;
  }

  // Sets the vocabulary from a file
  void KMerTokenizer_set_vocab(KMerTokenizer* obj, const char* filename) {
    obj->set_vocab_from_file(filename);
  }

  // Get vocabulary as a simple string representation
  char* KMerTokenizer_get_vocab(KMerTokenizer* obj) {
    std::ostringstream oss;
    for (const auto &pair : obj->get_vocab()) {
      oss << pair.first << " " << pair.second << "\n";
    }
    std::string vocab_str = oss.str();
    char* result = new char[vocab_str.size() + 1];
    std::strcpy(result, vocab_str.c_str());
    return result;
  }
}