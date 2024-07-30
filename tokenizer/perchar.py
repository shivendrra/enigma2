import ctypes
import os
import json

class PerChar:
  def __init__(self):
    self.lib_path = os.path.join(os.path.dirname(__file__), "lib", "DNATokenizer.dll")
    self.tokenizer_lib = ctypes.CDLL(self.lib_path)

    class PerCharTokenizer(ctypes.Structure):
      _fields_ = [("chars", ctypes.c_char * 9),
                  ("vocab_size", ctypes.c_int),
                  ("string_to_index", ctypes.c_int * 256),
                  ("index_to_string", ctypes.c_char * (9 + 256)),
                  ("special_index", ctypes.c_int)]
      
    self.PerCharTokenizer = PerCharTokenizer
    self.tokenizer = PerCharTokenizer()

    self.tokenizer_lib.init_tokenizer.argtypes = [ctypes.POINTER(PerCharTokenizer)]
    self.tokenizer_lib.init_tokenizer(ctypes.byref(self.tokenizer))

    self.tokenizer_lib.encode.argtypes = [ctypes.POINTER(PerCharTokenizer), ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
    self.tokenizer_lib.encode.restype = ctypes.POINTER(ctypes.c_int)
    self.tokenizer_lib.decode.argtypes = [ctypes.POINTER(PerCharTokenizer), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    self.tokenizer_lib.decode.restype = ctypes.c_char_p
    self.tokenizer_lib.free_memory.argtypes = [ctypes.POINTER(ctypes.c_int)]

  def encode(self, sequence):
    encoded_length = ctypes.c_int()
    encoded = self.tokenizer_lib.encode(ctypes.byref(self.tokenizer), sequence.encode('utf-8'), ctypes.byref(encoded_length))
    result = [encoded[i] for i in range(encoded_length.value)]
    self.tokenizer_lib.free_memory(encoded)
    return result

  def decode(self, encoded_sequence):
    length = len(encoded_sequence)
    encoded_array = (ctypes.c_int * length)(*encoded_sequence)
    decoded = self.tokenizer_lib.decode(ctypes.byref(self.tokenizer), encoded_array, length)
    return decoded.decode('utf-8')