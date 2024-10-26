# import os
# current_dir = os.path.dirname(os.path.realpath(__file__))
# os.chdir(current_dir)
# print(current_dir)

with open('./training files/file1.txt', 'r', encoding='utf-8') as f:
  test_data = f.read().lower()
  print("file opened!")
f.close()

from tokenizer import KMerTokenizer

tokenizer = KMerTokenizer(k_mers=4)
tokenizer.load_model('d:/machine learning/enigma2/tokenizer/vocabs/base_4k.model')

encoded_tokens = tokenizer.encode(test_data)
print(encoded_tokens)
decoded_tokens = tokenizer.decode(encoded_tokens)
print(decoded_tokens)
print(f"seq length: {len(test_data)} \ntokens length: {len(decoded_tokens)}")
print(test_data == decoded_tokens)
print(f"file length: {len(test_data)} \ntokens: {len(encoded_tokens)}")
print(f"compression ration: {(len(test_data) / len(encoded_tokens)):.2f}x")

# import os
# current_dir = os.path.dirname(os.path.realpath(__file__))
# os.chdir(current_dir)

# from tokenizer import PerChar
# tokenizer = PerChar()

# with open('training files/file1.txt', 'r', encoding='utf-8') as f:
#   test_data = f.read()
#   print("file opened!")
# f.close()

# encoded_tokens = tokenizer.encode(test_data)
# print(encoded_tokens)
# decoded_tokens = tokenizer.decode(encoded_tokens)
# print(decoded_tokens)

# print(f"seq length: {len(test_data)} \ntokens length: {len(decoded_tokens)}")
# print(test_data == decoded_tokens)
# print(f"file length: {len(test_data)} \ntokens: {len(encoded_tokens)}")
# print(f"compression ration: {(len(test_data) / len(encoded_tokens)):.2f}x")