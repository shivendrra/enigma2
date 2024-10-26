"""
  - @tokenizer/vocab.py
  - brief: script to generate custom k-mer vocabs for tokenizing the dna
  - comment: uses basic permutation logic to generate k-mer pairs of given `data_str` &
      progress untill the last permutation of given k-mer pair is reached
"""

import os
from itertools import product

def main_func(data_str, n_str):
  string = sorted(data_str)
  combinations = []
  for i in range(1, n_str + 1):
    combinations.extend(list(product(string, repeat=i)))
  return combinations

def write_to_file(combinations, filename):
  with open(filename, 'w') as f:
    for idx, combination in enumerate(combinations, start=1):
      out = ''.join(combination)
      out = out.replace('\n', '\\n')
      f.write(f'"{out}" {idx}\n')

def main():
  current_dir = os.path.dirname(os.path.realpath(__file__))
  os.chdir(current_dir)

  data_str = ['a', 't', 'c', 'g', '\n']
  n_str = 8
  combinations = main_func(data_str, n_str)
  
  output_file = f'vocabs/base_{n_str}k.model'
  write_to_file(combinations, output_file)
  
  print("Written to the file!")

if __name__ == "__main__":
  main()