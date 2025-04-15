import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

from biosaic import tokenizer, get_encodings
from .dataset import Dataset
from .model import Transformer, ModelConfig

tokenizer = tokenizer(encoding=get_encodings[3])
vocab_size = tokenizer.vocab_size

class TrainConfig:
  device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  learning_rate = 1e-4         # bumped from 1e-5
  weight_decay  = 1e-4
  amsgrad       = True
  warmup_epochs = 50           # linear warm‑up
  epochs        = 2000
  eval_interval = 100
  eval_iters    = 30
  batch_size    = 6
  block_size    = 256
loss_history  = []

# setup
_model = Transformer(ModelConfig, vocab_size, TrainConfig.block_size).to(TrainConfig.device)
n_param = sum(p.numel() for p in _model.parameters())/1e6
print(f"{n_param:.2f} million")
optimizer = torch.optim.Adam(_model.parameters(), lr=TrainConfig.learning_rate, amsgrad=True, weight_decay=1e-5, betas=(0.9, 0.95))

# --- Learning‑rate Schedulers ---
# 1) Warm‑up: linearly ramp LR from 0 → lr over warmup_epochs
warmup_scheduler = LambdaLR(
  optimizer,
  lr_lambda=lambda epoch: min((epoch+1)/TrainConfig.warmup_epochs, 1.0)
)
# 2) After warm‑up, cosine decay from lr → 0 over remaining epochs
cosine_scheduler = CosineAnnealingLR(
  optimizer,
  T_max=TrainConfig.epochs - TrainConfig.warmup_epochs,
  eta_min=1e-6
)

# train-test split
file_path = "/content/drive/MyDrive/dna_data.txt"
data = Dataset(file_path, get_encodings[3], ratio=0.2)
train_data, val_data = data.train_test_split()
train_data, val_data = tokenizer.encode(train_data), tokenizer.encode(val_data)

@torch.no_grad()
def estimate_loss():
  out = {}
  _model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(TrainConfig.eval_iters)
    for k in range(TrainConfig.eval_iters):
      X, Y = data.get_batch(split, batch_size=TrainConfig.batch_size, block_size=TrainConfig.block_size, device=TrainConfig.device)
      x_recon, vq_loss, _ = _model(X)
      recon_loss = F.cross_entropy(x_recon.view(-1, 4), Y.view(-1, 4))
      losses[k] = (recon_loss + vq_loss).item()
    out[split] = losses.mean()
  _model.train()
  return out

import timeit

start_time = timeit.default_timer()
for epoch in range(TrainConfig.epochs):
  xb, yb = data.get_batch("train", batch_size=TrainConfig.batch_size, block_size=TrainConfig.block_size, device=TrainConfig.device)

  x_recon, vq_loss, _ = _model(xb)
  recon_ce  = F.cross_entropy(x_recon.view(-1,4), yb.view(-1,4))
  recon_mse = F.mse_loss(torch.softmax(x_recon, dim=-1), yb)
  recon_loss = recon_ce + 0.5*recon_mse

  optimizer.zero_grad()
  recon_loss.backward()
  # - Gradient clipping -
  torch.nn.utils.clip_grad_norm_(_model.parameters(), max_norm=1.0)

  optimizer.step()

  # - Scheduler step -
  if epoch < TrainConfig.warmup_epochs:
    warmup_scheduler.step()
  else:
    cosine_scheduler.step()

  # - Logging & eval -
  if (epoch+1) % TrainConfig.eval_interval == 0:
    losses = estimate_loss()
    print(f"Epoch {epoch+1:4d} | train {losses['train']:.4f}  val {losses['val']:.4f}")
    loss_history.append((epoch+1, losses['train'], losses['val']))

end_time = timeit.default_timer()
print(f"Total time taken: {(end_time - start_time) / 3600} hrs")