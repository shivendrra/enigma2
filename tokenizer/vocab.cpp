#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace std;

void generate_combinations(const vector<char>& data_str, int n_str, unordered_map<string, int>& vocab) {
  int data_len = data_str.size();
  vector<int> indices(n_str, 0);
  string combination(n_str, '\0');
    
  int index = 0;
  while (true) {
    for (int i = 0; i < n_str; ++i) {
      combination[i] = data_str[indices[i]];
    }

    vocab[combination] = index++;

    int i;
    for (i = n_str - 1; i >= 0; --i) {
      if (indices[i] < data_len - 1) {
        indices[i]++;
        break;
      }
      indices[i] = 0;
    }

    if (i < 0) {
      break;
    }
  }
}

int main() {
  vector<char> data_str = {'a', 't', 'c', 'g', '\n'};
  int n_str = 5;
  unordered_map<string, int> vocab;

  generate_combinations(data_str, n_str, vocab);

  json j(vocab);
  ofstream file("vocab.json");

  file << "{\n";
  bool first = true;
  for (const auto& [key, value] : vocab) {
    if (!first) {
      file << ",\n";
    }
    first = false;

    file << "  \"";
    for (char c : key) {
      if (c == '\n') {
        file << "\\n";
      } else {
        file << c;
      }
    }
    file << "\": " << value;
  }
  file << "\n}\n";

  file.close();
  cout << "Written to the file!" << endl;
  return 0;
}