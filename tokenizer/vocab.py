import os
current_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_dir)

from itertools import product
import json

def main_func(data_str, n_str):
  string = sorted(data_str)
  combinations = product(string, repeat=n_str)
  vocab = {}

  for idx, combination in enumerate(combinations):
    out = ''.join(combination)
    vocab[out] = idx
        
  return vocab

data_str = ['a', 't', 'c', 'g', '\n']
n_str = 4
vocab = main_func(data_str, n_str)
with open(f'vocabs/base_{n_str}k.json', 'w') as f:
  json.dump(vocab, f)
  print("written in the file!")
  f.close()