import ctypes
import os

class PerCharTokenizer(ctypes.Structure):
  _fields_ = [("chars", ctypes.c_char * 9),
              ("vocab_size", ctypes.c_int),
              ("string_to_index", ctypes.c_int * 256),
              ("index_to_string", ctypes.c_char * (9 + 256)),
              ("special_index", ctypes.c_int)]

lib_path = os.path.join(os.path.dirname(__file__), "lib", "DNATokenizer.dll")
tokenizer_lib = ctypes.CDLL(lib_path)
tokenizer_lib.init_tokenizer.argtypes = [ctypes.POINTER(PerCharTokenizer)]
tokenizer_lib.encode.argtypes = [ctypes.POINTER(PerCharTokenizer), ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
tokenizer_lib.encode.restype = ctypes.POINTER(ctypes.c_int)
tokenizer_lib.decode.argtypes = [ctypes.POINTER(PerCharTokenizer), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
tokenizer_lib.decode.restype = ctypes.c_char_p

tokenizer = PerCharTokenizer()
tokenizer_lib.init_tokenizer(ctypes.byref(tokenizer))

string = b"AACATGTCCTGCATGGCATTAGTTTGTTGGGGCAGTGCCCGGATAGCATCAACGCTGCGCTGATTTGCCGTGGCGAGAAA"
encoded_length = ctypes.c_int()
encoded = tokenizer_lib.encode(ctypes.byref(tokenizer), string, ctypes.byref(encoded_length))

print("Encoded:", [encoded[i] for i in range(encoded_length.value)])

decoded = tokenizer_lib.decode(ctypes.byref(tokenizer), encoded, encoded_length.value)
print("Decoded:", decoded.decode('utf-8'))

tokenizer_lib.free_memory(encoded)