import tiktoken

pre_encodings = 'p50k_base'
pre_model = 'text-davinci-003'

class Tokenizer:
  def __init__(self, encoding=None, model=None):
    self.encodings = encoding if encoding is not None else pre_encodings
    self.model = model if model is not None else pre_model
    self.tokenizer = tiktoken.get_encoding(self.encodings)
    self.tokenizer = tiktoken.encoding_for_model(self.model)
  def encode(self, data): return self.tokenizer.encode(data)
  def decode(self, tokens): return self.tokenizer.decode(tokens)
  def get_vocab(self): return self.tokenizer.n_vocab

with open('training files/file1.txt', 'r', encoding="utf-8") as f:
  test = f.read()

tokenizer = Tokenizer()
encoded_tokens = tokenizer.encode(test)
print(encoded_tokens)
decoded_tokens = tokenizer.decode(encoded_tokens)
print(decoded_tokens)
print(f"seq length: {len(test)} \ntokens length: {len(decoded_tokens)}")
print(test == decoded_tokens)
print(f"file length: {len(test)} \ntokens: {len(encoded_tokens)}")
print(f"compression ration: {(len(test) / len(encoded_tokens)):.2f}x")