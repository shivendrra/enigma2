import os, ctypes

lib_path = os.path.join(os.path.dirname(__file__), "./../build/libkmer.so")
lib = ctypes.CDLL(lib_path)

MAX_TOKEN_SIZE = 100
MAX_VOCAB_SIZE = 10000

class CKMer(ctypes.Structure):
  _fields_ = [
    ("kmers", ctypes.c_int),
    ("vocab_size", ctypes.c_int),
    ("id_to_token", ctypes.c_char_p),
    ("token_to_id", ctypes.c_int)
  ]

lib.create_tokenizer.argtypes = [ctypes.c_int]
lib.create_tokenizer.restype = ctypes.POINTER(CKMer)
lib.tokenizer_sequence.argtypes = [ctypes.POINTER(CKMer), ctypes.c_char_p, ctypes.c_char, ctypes.c_int]
lib.tokenizer_sequence.restype = None
lib.build_vocab.argtypes = [ctypes.POINTER(CKMer), ctypes.c_char_p, ctypes.c_int]
lib.build_vocab.restype = None
lib.encode.argtypes = [ctypes.POINTER(CKMer), ctypes.c_char_p, ctypes.c_int]
lib.encode.restype = ctypes.c_int
lib.decode.argtypes = [ctypes.POINTER(CKMer), ctypes.c_int, ctypes.c_int]
lib.decode.restype = ctypes.c_char_p
lib.save.argtypes = [ctypes.POINTER(CKMer), ctypes.c_char_p]
lib.save.restype = None
lib.load.argtypes = [ctypes.POINTER(CKMer), ctypes.c_char_p]
lib.load.restype = None
lib.free_tokenizer.argtypes = [ctypes.POINTER(CKMer)]
lib.free_tokenizer.restype = None

class KMer(CKMer):
  def __init__(self, kmer:int= 4):
    assert isinstance(kmer, int), "KMer value must be a positive integer"
    self._tokenizer = CKMer()
    lib.create_tokenizer(ctypes.c_int(kmer))

  def tokenize_sequence(self, seq: str):
    lib.tokenize_sequence(ctypes.byref(self._tokenizer), )

  def encode(self, data: str):
    out = lib.decode(ctypes.c_char_p(data.encode("utf-8")))
    return out