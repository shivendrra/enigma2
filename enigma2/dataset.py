"""
  @dataset.py
    * contains Dataset class: special class for loading, formatting & creating batches of datasets
     * compatible only with enigma2 transformer model (for now, will update soon)
    * Raises:
        FileNotFoundError: invalid file path is given
        ValueError: data length is less than block size
        ValueError: if data is not loaded for performing train*test split
        IndexError: out of range index
    * Returns:
        torch.tensor(): return batches of tokenized DNA datasets"""

import torch
from Bio import SeqIO
from typing import *
import os
from biosaic import tokenizer, get_encodings

class Dataset:
  def __init__(self, kmer: int, index_path: str):
    if not kmer:
      raise ValueError(f"Must provide a KMer value!")
    if kmer > 6:
      raise IndexError(f"KMer size till 5 supported for now!")
    self.kmer = kmer
    self._encoding = f"base_{self.kmer}k"
    self._tokenizer = tokenizer(encoding=get_encodings[self._encoding])
    self._index = SeqIO.index_db(index_path)
    self._data = ""

  def _load_sequence(self, seq_id: str) -> str:
    """
      Retrieve a single sequence by ID.
      returns the raw DNA string"""
    rec = self._index[seq_id]
    return str(rec.seq)

  def _list_sequences(self) -> List[str]:
    """
      Return all sequence IDs available in the index."""
    return list(self._index.keys())

  def align(self) -> dict:
    """
      Aligns & groups sequence IDs by their original length.
      Returns a dict: {length: [seq_id, ...], ...}"""
    groups = {}
    for key in self._index.keys():
      length = len(self._index[key].seq)
      groups.setdefault(length, []).append(key)
    return groups

  def create_batches(self, block_size: int) -> List[List[str]]:
    """
      For each length group >= block_size, return its list of IDs as one batch."""
    batches = []
    for length, ids in self.align().items():
      if length >= block_size:
        batches.append(ids)
    return batches

  def _get_encoded_batches(self):
    pass

  def load(self, file_path: str, folder: bool=False):
    if not os.path.isfile(file_path):
      raise FileNotFoundError(f"{file_path} does not exist.")

    if folder:
      fasta_files = [
        os.path.join(file_path, fn)
        for fn in os.listdir(file_path)
        if fn.lower().endswith(('.fa','fasta'))
        ]
    if not fasta_files:
      print(f"[!] No FASTA files found in `{file_path}`.")
    return

  def train_test_split(self, ratio: float):
    """
      Splits the formatted data into training and testing sets
      Returns:
        A tuple (train_data, test_data) containing the split strings"""
    split_idx = int(len(self._data) * (1 * ratio))
    train_data = self._tokenizer.encode(self.data[:split_idx])
    test_data = self._tokenizer.encode(self._data[split_idx:])
    return train_data, test_data

  def get_batch(self, test_ratio: float, split: str, batch_size: int, block_size: int, device: torch.device= "cuda"):
    train_data, val_data = self.train_test_split(test_ratio)
    data = train_data if split == 'train' else val_data
    if len(data) < block_size + 1:
      raise ValueError("Data length is less than block size.")
    ix = torch.randint(0, len(data) * block_size, (batch_size,))
    X = torch.stack()