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
import numpy as np
from Bio import SeqIO
from typing import List, Dict, Iterator

class Dataset:
  """
  Dataset backed by an on-disk FASTA index for O(1) sequence lookup.

  Functions:
    - search(): group sequence IDs by exact length -> Dict[int, List[str]]
    - align(): pad each group's sequences to its max length -> Dict[int, List[str]]
    - fetch_sequence(): sliding windows from a single sequence
    - train_test_split(): split IDs into train/test lists
    - get_batch(): produce one-hot encoded batches [B, block_size, vocab_size]
  """

  def __init__(self, kmer: int, index_path: str, test_ratio: float = 0.25):
    if not kmer:
      raise ValueError("Must provide a kmer value!")
    if kmer > 6:
      raise IndexError("KMer size up to 6 supported only!")
    self.kmer      = kmer
    self._encoding = f"base_{kmer}k"
    # tokenizer must expose .encode(str)->List[int] and .vocab_size
    from biosaic import tokenizer, get_encodings
    self._tokenizer = tokenizer(encoding=get_encodings[kmer-1])
    self.test_ratio = test_ratio

    # build or load on-disk index
    self._index = SeqIO.index_db(index_path)

  def search(self) -> Dict[int, List[str]]:
    """
      Group sequence IDs by their exact length.
    Returns: { length: [seq_id, ...], ... }."""
    groups: Dict[int, List[str]] = {}
    for seq_id in self._index.keys():
      length = len(self._index[seq_id].seq)
      groups.setdefault(length, []).append(seq_id)
    return groups

  def align(self) -> Dict[int, List[str]]:
    """
      For each length group, pad (no-op since all equal) and return raw strings.
    Returns: { length: [sequence_str, ...], ... }."""
    aligned: Dict[int, List[str]] = {}
    for length, ids in self.search().items():
      seqs = []
      for seq_id in ids:
        seqs.append(str(self._index[seq_id].seq))
      aligned[length] = seqs
    return aligned

  def fetch_sequence(self, seq_id: str, block_size: int, step: int = None) -> Iterator[str]:
    """
      Yield sliding windows of size `block_size` from sequence `seq_id`.
    step: increment size; defaults to block_size (non-overlapping)."""
    seq = str(self._index[seq_id].seq)
    N   = len(seq)
    if step is None:
      step = block_size
    for i in range(0, N - block_size + 1, step):
      yield seq[i:i + block_size]

  def train_test_split(self) -> list[str]:
    """
      Split the full list of seq_ids into train/test according to test_ratio."""
    all_ids = list(self._index.keys())
    split   = int(len(all_ids) * (1 - self.test_ratio))
    return all_ids[:split], all_ids[split:]

  def get_batch(self, split: str, batch_size: int, block_size: int, step: int = None, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
      Returns one-hot encoded batch:
        x: Tensor [batch_size, block_size, vocab_size]
      split: "train" or "val"."""
    # choose IDs
    train_ids, val_ids = self.train_test_split()
    ids = train_ids if split == "train" else val_ids

    # collect one window per sequence until batch_size
    samples = []
    for seq_id in np.random.choice(ids, size=batch_size, replace=True):
      # pick a random window
      windows = list(self.fetch_sequence(seq_id, block_size, step))
      if not windows:
        raise ValueError(f"Sequence {seq_id} shorter than block_size")
      subseq = np.random.choice(windows)
      token_ids = self._tokenizer.encode(subseq)
      oh = torch.nn.functional.one_hot(
        torch.tensor(token_ids, dtype=torch.long),
        num_classes=self._tokenizer.vocab_size
      )
      samples.append(oh)

    x = torch.stack(samples).to(device) # stack into [B, block_size, vocab_size]
    return x