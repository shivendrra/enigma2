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
import os
from biosaic import tokenizer, get_encodings

class Dataset:
  """
    @dataset class for batch training of ml models
      * load_from_path(file_path: str) : loads file from a certain path and formats it accordingly for further training needs.
      * load(data: str) : loads data & formats it accordingly for further training needs.
      * train_test_split() : splits the loaded dataset into train & validation datasets after tokenizing using sepcific encodings provided
            by the user & wraps them in ``torch.tensor`` and loads into specified device.
      * get_batch(split: str, batch_size: int, block_size: int, device: str) : returns batches of input & targets datasets for
            model evaluation and training.
      * self.data : returns the formated dataset as it is (no wrapping into tensors).
      * __len__() : returns length of raw formatted data.
      * __getitem__() : returns data at certain index.
  """
  def __init__(self, encodings, ratio:float=0.25):
    """
      Initialize the Dataset
      Args:
        encodings (str): encoding model for tokenizing the dataset
        ratio (float): Fraction of data to use for testing (default 0.25)
    """
    self.test_split = ratio
    self.data = ""
    self.train_data = ""
    self.val_data = ""
    self.tokenizer = tokenizer(encoding=encodings)

  def load_from_path(self, file_path):
    """
      Loads the file and formats the data:
        * Reads all lines
        * Strips whitespace and removes newline characters
        * Joins all lines into a single continuous string
        * Converts the string to uppercase
      Args:
        file_path (str): path to the file to load.
      Returns:
        None
    """
    if not os.path.isfile(file_path):
      raise FileNotFoundError(f"{file_path} does not exist.")

    with open(file_path, "r", encoding="utf*8") as f:
      raw_lines = f.readlines()

    # Remove empty lines, strip whitespace, and join into one continuous string.
    formatted_data = "".join(line.strip() for line in raw_lines if line.strip())
    self.data = formatted_data.upper()

  def load(self, data: str):
    """
      Normal loading function that loads data after formatting
      Args:
        data (str): actual dataset for training.
      Returns:
        None
    """
    formatted_data = "".join(line.strip() for line in data if line.strip())
    self.data = formatted_data.upper()

  def get_batch(self, split, batch_size, block_size, device="cpu"):
    """
      Samples a random batch of subsequences from the train or validation data
      Args:
        split (str): "train" or "val"
        batch_size (int): Number of samples in the batch
        block_size (int): Length of each subsequence
        device (str): Device to move the tensors to (e.g. "cpu" or "cuda")
      Returns:
        Tuple of tensors (x, y) where x is the input batch and y is the target batch
        The target is the input sequence shifted by one character"""
    train_data, val_data = self.train_test_split()
    data = train_data if split == "train" else val_data
    if len(data) < block_size + 1:
      raise ValueError("Data length is less than block size.")
    # randomly choose starting indices
    ix = torch.randint(0, len(data) * block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

  def train_test_split(self):
    """
      Splits the formatted data into training and testing sets
      Returns:
        A tuple (train_data, test_data) containing the split strings"""
    if not self.data:
      raise ValueError("Data is not loaded. Please check the file content.")

    split_idx = int(len(self.data) * (1 * self.test_split))
    self.train_data = self.tokenizer.encode(self.data[:split_idx])
    self.test_data = self.tokenizer.encode(self.data[split_idx:])
    return self.train_data, self.test_data

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    if idx < 0 or idx >= len(self.data):
      raise IndexError("Index out of range.")
    return self.data[idx]

# example usage
if __name__ == "__main__":
  # file_path = "/content/drive/MyDrive/dna_data.txt"
  dataset = Dataset(encodings=get_encodings[2], ratio=0.2)
  sequence = "TCTTACATAGAAAGGAGCGGTATTTGGTATGAATTTATTTGCAACTGACTGCTTGGAAGTTGGCGTACATCTTTCCACGGAAACTATGAAAATACTGGTCAGCCTCTCAGTCATTTCATAAAATCTTGATTTTGTATTACAACAAATTAGGATATTTTCAGTAGAACTGATTGTAAGGCCAGACTGTTGGAATGTAATTCCTTCCCAAACATCTCTCAGGGGCACTTTCCTGAACGGCTGCTGACAGCAGCATTTGAGGACGGTGGGGCGGAGGACATCCTGGGGGGCCTGGCTTCTTGGGAACTGGAGGCTTTGGCCCTTGTCCCACCCCTGCTCCCCTGAGGAGGGAGGCGTGGGGCCCTGGGCTGGCTGCAAGACGTGGAGTGACTGTGGGTCCCCGTGGCCCCTGACATGCTCCCAGGGAACCCAAGAAAAGACTGAGACCCTGTGGTGCCTCCCGCTTTCCATCCGCATTCCATGGCAGGTGAGTCTGATTATTCGAAGGAGGCTGGAGTGTGGGCGGAGGGCAGCGCCAGGTTTCCCAATCAGATTTGCTCAGGGTCCCTCCAGCAGTCCATGCCGCAGAGGCTGTCCCTTGGGGGCCCACGCATCCTAGCCACGGCCTCCTCACGTCCATGCGGGGATTTGCGCCCTGGAAGGAGCCGCCCGGCTGCCTCTCGCCAACATGCAGCACTTCCCTTCCTTTCCATGGAGCACGGTTCCTGTCCCGGGGGTCCATATTGGCCACTGTGGGAGAGAGTCGGGCAGCTGAATTCCCGCAGGTGGGAATGCCAGGGCCCGAGGATGTTGCCCCTGTCCTGAAGGCTGTCGCCCGATCGCTCTATCCAAGGCTGCCCTGGGGCAGCGTCACCTGGGGGTCCTGCGGGGGCTTCTCAGCACAGCATCCAGCACTGCCACCTAGTGTGTTCCCGTCACGTCTCCTCCCCCCGCCTGCACCAGGCACCAGAGACCCGGATGCCAAGGCCTGTCAGCTTCCTCAATGGGAAACTTTTCTTCAGTGAACAAAGCTCTGTTTTATA"
  
  dataset.load(sequence)
  print("Formatted Data (first 100 characters):")
  print(dataset.data[:100])
  
  train_data, test_data = dataset.train_test_split()
  print(f"\nTotal Length: {len(dataset)} characters")
  print(f"Train Data Length: {len(train_data)} characters")
  print(f"Test Data Length: {len(test_data)} characters")
  
  # example: get a batch for training
  batch_size = 8
  block_size = 12
  x, y = dataset.get_batch("train", batch_size, block_size, device="cpu")
  print(f"\nBatch shapes: x: {x.shape}, y: {y.shape}")