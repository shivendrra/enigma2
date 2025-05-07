# Model Training Guide

This document explains how to **train** the VQ-Transformer model on DNA sequence data using **`run.py`**. It covers setup, configuration, data handling, model architecture, training logic, and evaluation.

## Table of Contents

* [Prerequisites](#prerequisites)
* [Dataset Preparation](#dataset-preparation)
* [Model Architecture](#model-architecture)
* [Training Configuration](#training-configuration)
* [Training & Evaluation Loop](#training--evaluation-loop)
* [Learning-Rate Scheduling](#learning-rate-scheduling)
* [Logging & Monitoring](#logging--monitoring)
* [Usage](#usage)
* [Extending & Customization](#extending--customization)
* [References](#references)

## Prerequisites

* Python 3.13+
* PyTorch 1.12+
* `biosaic` package (provides `tokenizer`, `get_encodings`)
* Custom modules: `Dataset`, `Transformer`, and `ModelConfig` in your project

Install dependencies via:

```bash
pip install torch pandas pyarrow biosaic
```

## Dataset Preparation

1. **Input File**: A plain-text file containing concatenated DNA sequences (e.g., `dna_data.txt`).
2. **Loading**: In `run.py`, the file is read into a string:

   ```python
   file_path = "/content/drive/MyDrive/dna_data.txt"
   with open(file_path, 'r', encoding='utf-8') as f:
     sequence = f.read()
   ```

3. **Dataset Class**: Instantiate with tokenization settings and train/val split ratio:

   ```python
    dataset = Dataset(file\_path, get\_encodings\[3], ratio=0.2)
    dataset.load(sequence)
   ```

The `Dataset` handles sequence chunking (`block_size`) and generates `(X, Y)` batches for training and validation.

## Model Architecture

The core model is a **Vector-Quantized Transformer** (`Transformer` class) configured via `ModelConfig`:

```python
_model = Transformer(ModelConfig, vocab_size, TrainConfig.block_size)
```

-**Embedding**: Input token embedding + positional encoding (block size as max length).  
-**Transformer Blocks**: Stacked multi-head self-attention and feed-forward layers.  
-**Vector Quantization**: At the model’s bottleneck, continuous representations are quantized to discrete codes, enabling a compressed latent space.  
-**Output Projection**: Reconstructs token logits over the 4-base DNA vocabulary.

Total parameter count is printed at startup:

```python
  n_param = sum(p.numel() for p in _model.parameters())/1e6
  print(f"{n_param:.2f} million")
```

## Training Configuration

All hyperparameters live in the `TrainConfig` class:

```python
class TrainConfig:
  device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  learning_rate = 1e-4
  weight_decay  = 1e-4
  amsgrad       = True
  warmup_epochs = 50
  epochs        = 2000
  eval_interval = 100
  eval_iters    = 30
  batch_size    = 6
  block_size    = 256
```

-**`device`**: GPU if available.  
-**`learning_rate`**: Base LR used by Adam optimizer.  
-**`weight_decay`**, **`amsgrad`**: Regularization and optimizer variants.  
-**`warmup_epochs`**: Epochs to linearly ramp up LR.  
-**`epochs`**: Total training epochs.  
-**`eval_interval`**: Run validation every N epochs.  
-**`eval_iters`**: Number of validation batches to average loss.  
-**`batch_size`** & **`block_size`**: Controls memory and sequence length per sample.

## Training & Evaluation Loop

The main loop in `run.py` follows:

```python
for epoch in range(TrainConfig.epochs):
  # 1) Get next batch
  xb, yb = dataset.get_batch("train", batch_size=..., block_size=..., device=...)

  # 2) Forward pass
  x_recon, vq_loss, _ = _model(xb)

  # 3) Compute losses
  recon_ce  = F.cross_entropy(x_recon.view(-1,4), yb.view(-1,4))
  recon_mse = F.mse_loss(torch.softmax(x_recon, dim=-1), yb)
  recon_loss = recon_ce + 0.5*recon_mse

  # 4) Backpropagation
  optimizer.zero_grad()
  recon_loss.backward()
  torch.nn.utils.clip_grad_norm_(_model.parameters(), max_norm=1.0)
  optimizer.step()

  # 5) Scheduler step (see below)
  if epoch < TrainConfig.warmup_epochs:
    warmup_scheduler.step()
  else:
    cosine_scheduler.step()

  # 6) Validation & logging
  if (epoch+1) % TrainConfig.eval_interval == 0:
    losses = estimate_loss()
    print(f"Epoch {epoch+1} | train {losses['train']:.4f}  val {losses['val']:.4f}")
    loss_history.append((epoch+1, losses['train'], losses['val']))
```

The auxiliary `estimate_loss()` runs the model on `eval_iters` validation batches (in `no_grad` mode), averaging reconstruction + VQ losses for both train and val splits.

## Learning-Rate Scheduling

Two-stage scheduling combines linear warm-up and cosine decay:

1. **Warm-Up Stage** (0 -> `learning_rate`):

   ```python
   warmup_scheduler = LambdaLR(
     optimizer,
     lr_lambda=lambda epoch: min((epoch+1)/TrainConfig.warmup_epochs, 1.0)
   )
   ```

2. **Cosine Decay** (`learning_rate` -> `eta_min`):

   ```python
   cosine_scheduler = CosineAnnealingLR(
     optimizer,
     T_max=TrainConfig.epochs - TrainConfig.warmup_epochs,
     eta_min=1e-6
   )
   ```

The code step logic ensures seamless transition from warm-up to decay.

## Logging & Monitoring

-**Console Output**: Prints parameter count at startup, file load confirmation, and epoch-level train/val losses.  

-**`loss_history`**: A Python list capturing `(epoch, train_loss, val_loss)` for offline analysis.

-**Time Tracking**: Total training time is measured and displayed:

  ```python
  import timeit
  start_time = timeit.default_timer()
  ...
  end_time = timeit.default_timer()
  print(f"Total time taken: { (end_time - start_time)/3600 :.2f} hrs")
  ```

## Usage

Simply run:

  ```bash
  python run.py
  ````

Ensure `file_path` and hyperparameters in `run.py` point to your data and desired settings. Adjust `TrainConfig` values directly in the file.

## Extending & Customization

* **Data**: Modify `Dataset` for different tokenization or splitting logic.
* **Model**: Swap `Transformer` with alternative architectures or adjust `ModelConfig`.
* **Losses**: Add auxiliary metrics or alternative reconstruction criteria.
* **Monitoring**: Integrate TensorBoard or Weights & Biases for richer analytics.

## References

* Implementation based on Biopython and PyTorch best practices.
* Vector quantization inspired by VQ-VAE literature.
* DNA language modeling concepts from recent computational genomics research.
