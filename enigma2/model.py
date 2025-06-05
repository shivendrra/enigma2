import torch
import torch.nn as nn
from torch.nn import functional as F

class ModelConfig:
  block_size: int = 256
  n_head: int = 12
  n_layers: int = 12
  d_model: int = 512
  n_latent: int = 64
  bias: bool = False
  ffn_multiplier: int = 4
  n_ff: int = ffn_multiplier * d_model
  dropout: float = 0.2
  norm_eps: float = 1e-5
  num_bins: int = 4
  kan_max: float = 1.0
  kan_min: float = -1.0
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RMSNorm(nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def _norm(self, x: torch.Tensor):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x: torch.Tensor):
    output = self._norm(x.float()).type_as(x)
    return output * self.weight

class SplineEdge(nn.Module):
  """
  A simple piecewise-linear spline activation function on one edge
  """
  def __init__(self, num_bins=4, x_min=-1.0, x_max=1.0):
    super().__init__()
    self.num_bins = num_bins
    self.x_min, self.x_max = x_min, x_max
    # Initialize learnable knot values uniformly
    self.knots = nn.Parameter(torch.linspace(x_min, x_max, num_bins + 1))
    # Heights at each knot
    self.heights = nn.Parameter(torch.zeros(num_bins + 1))

  def forward(self, x: torch.Tensor):
    # x: (...)
    x_clamped = x.clamp(self.x_min, self.x_max)
    # Compute bin indices
    bins = torch.bucketize(x_clamped, self.knots[:-1], right=True)
    bins = bins.clamp(0, self.num_bins - 1)
    # Left and right knot positions
    k0, k1 = self.knots[bins], self.knots[bins + 1]
    h0, h1 = self.heights[bins], self.heights[bins + 1]

    # Linear interpolation factor
    t = (x_clamped - k0) / (k1 - k0 + 1e-6)
    return h0 * (1 - t) + h1 * t

class KANLayer(nn.Module):
  """
  A KAN-based projection layer replacing nn.Linear(d_model, d_model).
  Each edge from input_dim to output_dim is a SplineEdge
  """
  def __init__(self, dim, num_bins=4, x_min=-1.0, x_max=1.0):
    super().__init__()
    self.dim = dim
    self.num_bins = num_bins
    self.edges = nn.ModuleList([
      SplineEdge(num_bins, x_min, x_max)
      for _ in range(dim * dim)
    ])

  def forward(self, x: torch.Tensor):
    B, T, D = x.shape  # x: [B, T, dim]
    assert D == self.dim
    out, idx = x.new_zeros(B, T, D), 0

    # Sum over all input dims for each output dim
    for j in range(D):
      acc = 0
      for i in range(D):
        # apply spline edge to feature i
        acc = acc + self.edges[idx](x[:, :, i])
        idx += 1
      out[:, :, j] = acc
    return out

class RoPE(nn.Module):
  def __init__(self, head_size, block_size, device):
    super().__init__()
    self.head_size = head_size
    self.block_size = block_size
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_size, 2).float() / head_size))
    position = torch.arange(0, self.block_size, dtype=torch.float, device=device)  # (block_size,)
    sinusoidal = torch.einsum("i,j->ij", position, inv_freq)  # Shape: (block_size, head_size // 2)
    self.register_buffer("cos_emb", sinusoidal.cos(), persistent=False)  # (block_size, head_size // 2)
    self.register_buffer("sin_emb", sinusoidal.sin(), persistent=False)  # (block_size, head_size // 2)

  def forward(self, q: torch.Tensor, k: torch.Tensor):
    # splitting tensors into even and odd components
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    assert q.size(-1) == self.head_size, f"Query size mismatch: {q.size(-1)} != {self.head_size}"
    assert k.size(-1) == self.head_size, f"Key size mismatch: {k.size(-1)} != {self.head_size}"
    # retrieving embeddings for current sequence length
    cos = self.cos_emb[:q.shape[1], :].unsqueeze(0).to(q.device)
    sin = self.sin_emb[:q.shape[1], :].unsqueeze(0).to(q.device)

    # applying rotations
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q_rot, k_rot

class LatentHead(nn.Module):
  def __init__(self, head_size, d_model, block_size, latent_dim, dropout, device, bias=False):
    super().__init__()
    if latent_dim is None:
      latent_dim = max(1, head_size // 2)
    self.head_size = head_size
    self.device = device
    self.w_query = nn.Linear(d_model, head_size, bias=bias)
    
    # for keys: decomposing the proj into two parts: wa_key, wb_key
    self.wa_key = nn.Linear(d_model, latent_dim, bias=bias)
    self.wb_key = nn.Linear(latent_dim, head_size, bias=bias)

    # for values: similar decomposing the proj into two parts: wa_value, wb_value
    self.wa_value = nn.Linear(d_model, latent_dim, bias=bias)
    self.wb_value = nn.Linear(latent_dim, head_size, bias=bias)

    self.dropout = nn.Dropout(dropout)
    self.pos_embd = RoPE(head_size, block_size, device)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x: torch.Tensor):
    B, T, C = x.shape
    
    w_query = self.w_query(x)  # (B, T, head_size)
    latent_key, latent_value = self.wa_key(x), self.wa_value(x)  # (B, T, latent_dim)
    # expanding the latent representations to full key & values
    key, value = self.wb_key(latent_key), self.wb_value(latent_value)  # (B, T, head_size)

    w_query, key = self.pos_embd(w_query, key)  # applying Rotary PosEmbedds
    scores = torch.matmul(w_query, key.transpose(-2, -1)) / (self.head_size ** 0.5)  # Fixed scaling

    # Dynamic causal mask handling
    if T > self.tril.size(0):
      new_tril = torch.tril(torch.ones(T, T, device=self.device))
      self.register_buffer('tril', new_tril)
    
    scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    scores = F.softmax(scores, dim=-1)
    scores = self.dropout(scores)  # applying dropout after softmax
    out = torch.matmul(scores, value)
    return out

class MultiHeadLatentAttention(nn.Module):
  def __init__(self, d_model, dropout, n_head, block_size, device, bias=False, latent_dim=None, projection_bins=4):
    head_size = d_model // n_head
    super().__init__()
    self.heads = nn.ModuleList([
      LatentHead(head_size, d_model, block_size, latent_dim, dropout, device, bias) 
      for _ in range(n_head)
    ])
    self.proj = KANLayer(d_model, projection_bins)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor):
    # x: [B, T, d_model]
    head_outs = [h(x) for h in self.heads]  # each [B, T, head_size]
    concat = torch.cat(head_outs, dim=-1)  # [B, T, d_model]
    projected = self.proj(concat)  # KAN-based projection
    return self.dropout(projected)

class FeedForward(nn.Module):
  def __init__(self, d_model, dropout, num_bins=6, hidden_dim=None, ffn_multiplier=None):
    super().__init__()
    # Fixed the logic for hidden_dim calculation
    if hidden_dim is not None:
      self.hidden_dim = int(2 * hidden_dim / 3)
    elif ffn_multiplier is not None:
      self.hidden_dim = int(ffn_multiplier * d_model)
    else:
      self.hidden_dim = 4 * d_model  # Default fallback
      
    self.net = nn.Sequential(
      nn.Linear(d_model, self.hidden_dim),
      nn.GELU(),
      nn.Linear(self.hidden_dim, d_model),
      nn.Dropout(dropout),
    )
    self.projection = KANLayer(d_model, num_bins=num_bins)  # additional KAN layer projection

  def forward(self, x: torch.Tensor):
    return self.projection(self.net(x))

class DecoderBlock(nn.Module):
  def __init__(self, d_model: int, block_size: int, num_bins: int, n_head: int, norm_eps: float, dropout: float, latent_dim: int, projection_bins: int, bias: bool, device: str, n_ff: int = None, ffn_multiplier: int = None):
    super().__init__()
    self.self_att = MultiHeadLatentAttention(
      n_head=n_head, 
      d_model=d_model, 
      dropout=dropout, 
      block_size=block_size, 
      device=device, 
      latent_dim=latent_dim, 
      bias=bias, 
      projection_bins=projection_bins
    )
    self.ffwd = FeedForward(d_model, dropout, num_bins, n_ff, ffn_multiplier)
    self.dropout = nn.Dropout(dropout)
    self.norm1 = RMSNorm(d_model, eps=norm_eps)  # Added separate norm layers
    self.norm2 = RMSNorm(d_model, eps=norm_eps)

  def forward(self, x: torch.Tensor):
    # Pre-norm architecture
    x_out = self.self_att(self.norm1(x))
    x = x + self.dropout(x_out)  # Residual connection

    x_ff = self.ffwd(self.norm2(x))
    x = x + self.dropout(x_ff)  # Residual connection
    return x

class Transformer(nn.Module):
  def __init__(self, params: ModelConfig, vocab_size: int, block_size: int = None):
    super().__init__()
    if block_size is None:
      block_size = params.block_size
    self.block_size = block_size
    self.vocab_size = vocab_size
    self.token_projection = nn.Linear(vocab_size, params.d_model, bias=False)
    self.decoder = nn.Sequential(*[
      DecoderBlock(
        d_model=params.d_model,
        block_size=block_size,
        num_bins=params.num_bins,
        n_head=params.n_head,
        norm_eps=params.norm_eps,
        dropout=params.dropout,
        latent_dim=params.n_latent,
        projection_bins=params.num_bins,
        bias=params.bias,
        device=params.device,
        n_ff=params.n_ff,
        ffn_multiplier=params.ffn_multiplier
      ) for _ in range(params.n_layers)
    ])
    self.norm_final = RMSNorm(params.d_model, eps=params.norm_eps)
    self.linear_final = nn.Linear(params.d_model, vocab_size, bias=False)
    self.dropout = nn.Dropout(params.dropout)
    
    # Initialize weights
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
    B, T = idx.shape
    assert T <= self.block_size, f"Sequence length {T} exceeds block size {self.block_size}"
    
    x = self.token_projection(idx)  # idx shape: [batch, seq_len, vocab_size] with one-hot vectors
    x = self.decoder(x)
    x = self.norm_final(x)
    logits = self.linear_final(x)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits_flat = logits.view(B * T, C)
      targets_flat = targets.view(B * T)
      loss = F.cross_entropy(logits_flat, targets_flat)

    return logits, loss