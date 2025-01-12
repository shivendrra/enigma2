import ctypes
from .cbase import CKMer, libkmer

class KMer(CKMer):
  def __init__(self, kmer:int= 4):
    assert isinstance(kmer, int), "KMer value must be a positive integer"
    self._tokenizer = CKMer()
    libkmer.create_tokenizer(ctypes.c_int(kmer))

  def tokenize_sequence(self, seq: str):
    libkmer.tokenize_sequence(ctypes.byref(self._tokenizer), )

  def encode(self, data: str):
    out = libkmer.encode_sequence(ctypes.byref(self._tokenizer), data.encode("utf-8"), ctypes.c_int(0))
    return out
  
  def decode(self, ids: int):
    new_ids = (ctypes.c_int * len(ids))(*ids)
    out = libkmer.decode_sequence(ctypes.byref(self._tokenizer), new_ids, len(ids))
    return out.decode("utf-8")

  def save(self, path: str):
    libkmer.save(ctypes.byref(self._tokenizer), path.encode("utf=8"))
  
  def __del__(self):
    libkmer.free_tokenizer(ctypes.byref(self._tokenizer))