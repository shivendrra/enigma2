# 🧬 ``Dataset`` Class — Enigma2 Batch Loader

This `Dataset` class is a DNA-sequence batching utility built for **training machine learning models** (e.g., Transformers) on genomic data. It is designed to work with **FASTA-formatted DNA datasets** indexed using Biopython's `SeqIO.index_db` for fast O(1) access.

## Features

* **Efficient O(1) indexed lookups** for FASTA datasets via `.idx` files
* **Batch generation** with one-hot encoded sequences: `[B, block_size, vocab_size]`
* **K-mer tokenization** via customizable encoding models
* **Train/Validation split** with support for randomized sampling
* **Sliding window fetching** from indexed sequences
* **Grouped alignment** of sequences with equal length

## Class Signature

```python
Dataset(kmer: int, index_path: str, test_ratio: float = 0.25)
```

### Parameters

| Argument     | Type    | Description                                       |
| ---  | --- | --- |
| `kmer`       | `int`   | K-mer length (up to 6 supported)                  |
| `index_path` | `str`   | Path to Biopython `.idx` file for the FASTA data  |
| `test_ratio` | `float` | Proportion of sequences to reserve for validation |

## Dependencies

* [Biopython](https://biopython.org/)
* [NumPy](https://numpy.org/)
* [PyTorch](https://pytorch.org/)
* [Biosaic](https://pypi.org/project/biosaic/)

## Core Methods

### `search() → Dict[int, List[str]]`

Groups sequence IDs by their exact length.

```python
>>> ds.search()
{8: ['seq1', 'seq3'], 6: ['seq2']}
```

### `align() → Dict[int, List[str]]`

Returns raw sequences grouped by identical lengths, useful for downstream pairing.

### `fetch_sequence(seq_id, block_size, step=None) → Iterator[str]`

Yields overlapping (or non-overlapping) subsequences of length `block_size` from a given sequence ID.

```python
>>> for chunk in ds.fetch_sequence("seq1", block_size=4): print(chunk)
ATGC
TGCG
GCGT
...
```

### `train_test_split() → Tuple[List[str], List[str]]`

Returns sequence ID lists split by ratio:

```python
train_ids, val_ids = ds.train_test_split()
```

### `get_batch(split, batch_size, block_size, step=None, device='cpu') → torch.Tensor`

Generates a one-hot encoded tensor batch from randomly sampled subsequences.

```python
batch = ds.get_batch("train", batch_size=16, block_size=128)
# batch shape: [16, 128, vocab_size]
```

## Example Usage

```python
from dataset import Dataset

ds = Dataset(kmer=3, index_path="data/all_sequences.idx")

# List grouped sequences
print(ds.search())

# Get aligned raw strings
aligned = ds.align()

# Fetch windows from one ID
for w in ds.fetch_sequence("seq1", 128):
  print(w)

# Get training batch
train_tensor = ds.get_batch("train", batch_size=32, block_size=256)
print(train_tensor.shape)  # [32, 256, vocab_size]
```

## Notes

* This class is part of the **Enigma2** DNA modeling pipeline.
* `kmer` must match the tokenizer configuration (up to 6 supported).
* `.idx` file must be built using `SeqIO.index_db()` beforehand.

To build one:

```python
from Bio import SeqIO
SeqIO.index_db("my_index.idx", ["file1.fasta", "file2.fasta"], "fasta")
```

## Related

* `tokenizer` — custom tokenizer that supports `encode()` and `vocab_size`
* `SeqIO.index_db()` — Biopython function for FASTA indexing
* Enigma2 GitHub: [shivendrra/enigma2](https://github.com/shivendrra/enigma2)
