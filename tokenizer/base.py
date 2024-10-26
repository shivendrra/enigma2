import json, ctypes, os
from ctypes import c_int, c_char_p, POINTER
current_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_dir)

class KMerTokenizer:
  def __init__(self, k_mers=4):
    self.lib = ctypes.CDLL('./libkmer.so')
    self.lib.KMerTokenizer_delete.argtypes = [ctypes.c_void_p]
    self.obj = self.lib.KMerTokenizer_new(k_mers)

    if not self.obj:
      print("Failed to create KMerTokenizer object.")

  def tokenize_sequence(self, sequence):
    self.lib.KMerTokenizer_tokenize_sequence.restype = POINTER(c_char_p)
    result = self.lib.KMerTokenizer_tokenize_sequence(self.obj, sequence.encode('utf-8'))
    tokens = []

    i = 0
    while result[i]:
      tokens.append(result[i].decode('utf-8'))
      i += 1

    self.lib.KMerTokenizer_free_tokens(result)
    return tokens

  def encode(self, sequence):
    self.lib.KMerTokenizer_encode.restype = POINTER(c_int)
    result = self.lib.KMerTokenizer_encode(self.obj, sequence.encode('utf-8'))
    encoded = []

    i = 0
    while result[i] != -1:
      encoded.append(result[i])
      i += 1

    self.lib.KMerTokenizer_free_int_array(result)
    return encoded

  def decode(self, encoded_sequence):
    array_type = c_int * len(encoded_sequence)
    encoded_array = array_type(*encoded_sequence)

    self.lib.KMerTokenizer_decode.restype = c_char_p
    decoded = self.lib.KMerTokenizer_decode(self.obj, encoded_array, len(encoded_sequence))
    result = decoded.decode('utf-8')
    ctypes.free(decoded)
    return result

  def set_vocab(self, path):
    self.lib.KMerTokenizer_set_vocab.argtypes = [ctypes.c_void_p, c_char_p]
    self.lib.KMerTokenizer_set_vocab(self.obj, path.encode("utf-8"))

  def get_vocab(self):
    self.lib.KMerTokenizer_get_vocab.argtypes = [ctypes.c_void_p]
    self.lib.KMerTokenizer_get_vocab.restype = c_char_p

    vocab_str = self.lib.KMerTokenizer_get_vocab(self.obj)
    return json.loads(vocab_str.decode('utf-8'))

  def save_model(self, model_path):
    vocab_file = f"{model_path}/base_{self.k_mers}k.json"
    with open(vocab_file, 'w') as f:
      json.dump(self.vocab, f)
    print("Saved the vocab!")

  def load_model(self, path):
    assert self.obj is not None, "KMerTokenizer object is not initialized"
    assert path.endswith('.model'), "vocab must be stored as model file"
    self.set_vocab(path)

  def __del__(self):
    if self.obj is not None:
      self.lib.KMerTokenizer_delete(self.obj)
      self.obj = None