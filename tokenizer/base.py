import kmer_tokenizer
import json
from tqdm import tqdm

class KMerTokenizer:
  def __init__(self, k_mers: int = 4):
    self.k_mers = k_mers
    self.tokenizer = kmer_tokenizer.KMerTokenizer(k_mers)
    self.vocab = {}

  def tokenize_sequence(self, sequence):
    return self.tokenizer.tokenize_sequence(sequence)

  def encode(self, sequence):
    return self.tokenizer.encode(sequence)

  def decode(self, encoded_sequence):
    return self.tokenizer.decode(encoded_sequence)
  
  def save_model(self, model_path):
    vocab_file = f"{model_path}/base_{self.k_mers}k.json"
    with open(vocab_file, 'w') as f:
      json.dump(self.vocab, f)
    print("saved the vocab!")

  def load_model(self, path):
    assert path.endswith('.json')
    with open(path, 'r') as f:
      vocab = json.load(f)
    print("loaded the vocab!")
    
    self.vocab = vocab
    self.tokenizer.set_vocab(vocab)
    self.tokenizer.vocab_size = len(vocab)

    self.id_to_token = [None] * self.vocab_size
    for token, idx in self.vocab.items():
      self.id_to_token[idx] = token

if __name__ == "__main__":
  tokenizer = KMerTokenizer(k_mers=4)
  sequences = ["ATGCGTAC", "GTCAGTAC"]
  for sequence in sequences:
    print(tokenizer.tokenize_sequence(sequence))
    encoded = tokenizer.encode(sequence)
    print(encoded)
    decoded = tokenizer.decode(encoded)
    print(decoded)
  tokenizer.save_model("model")
  tokenizer.load_model("model/base_4k.json")