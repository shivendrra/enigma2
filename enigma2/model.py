import torch
import torch.nn as nn
from torch.nn import functional as F

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

def rotate_half(x):
  x1, x2 = x.chunk(2, dim=-1)
  return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
  q = (q * cos) + (rotate_half(q) * sin)
  k = (k * cos) + (rotate_half(k) * sin)
  return q, k

def apply_rope_x(x, cos, sin):
  return (x * cos) + (rotate_half(x) * sin)

class MLA(torch.nn.Module):
  def __init__(self, d_model, n_heads, max_len=1024, rope_theta=10000.0):
    super().__init__()
    self.d_model, self.n_heads = d_model, n_heads
    self.dh = d_model // n_heads
    self.q_proj_dim, self.kv_proj_dim = d_model // 2, (2*d_model) // 3
    self.qk_nope_dim, self.qk_rope_dim = self.dh // 2, self.dh // 2

    ## Q projections
    # Lora
    self.W_dq = torch.nn.Parameter(0.01*torch.randn((d_model, self.q_proj_dim)))
    self.W_uq = torch.nn.Parameter(0.01*torch.randn((self.q_proj_dim, self.d_model)))
    self.q_layernorm = torch.nn.LayerNorm(self.q_proj_dim)
        
    ## KV projections
    # Lora
    self.W_dkv = torch.nn.Parameter(0.01*torch.randn((d_model, self.kv_proj_dim + self.qk_rope_dim)))
    self.W_ukv = torch.nn.Parameter(0.01*torch.randn((self.kv_proj_dim, self.d_model + (self.n_heads * self.qk_nope_dim))))
    self.kv_layernorm = torch.nn.LayerNorm(self.kv_proj_dim)

    # output projection
    self.W_o = torch.nn.Parameter(0.01*torch.randn((d_model, d_model)))

    # RoPE
    self.max_seq_len = max_len
    self.rope_theta = rope_theta

    # https://github.com/lucidrains/rotary-embedding-torch/tree/main
    # visualize emb later to make sure it looks ok
    # we do self.dh here instead of self.qk_rope_dim because its better
    freqs = 1.0 / (rope_theta ** (torch.arange(0, self.dh, 2).float() / self.dh))
    emb = torch.outer(torch.arange(self.max_seq_len).float(), freqs)
    cos_cached = emb.cos()[None, None, :, :]
    sin_cached = emb.sin()[None, None, :, :]

    # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer
    # This is like a parameter but its a constant so we can use register_buffer
    self.register_buffer("cos_cached", cos_cached)
    self.register_buffer("sin_cached", sin_cached)

  def forward(self, x, kv_cache=None, past_length=0):
    B, S, D = x.size()

    # Q Projections
    compressed_q = x @ self.W_dq
    compressed_q = self.q_layernorm(compressed_q)
    Q = compressed_q @ self.W_uq
    Q = Q.view(B, -1, self.n_heads, self.dh).transpose(1,2)
    Q, Q_for_rope = torch.split(Q, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)

    # Q Decoupled RoPE
    cos_q = self.cos_cached[:, :, past_length:past_length+S, :self.qk_rope_dim//2].repeat(1, 1, 1, 2)
    sin_q = self.sin_cached[:, :, past_length:past_length+S, :self.qk_rope_dim//2].repeat(1, 1, 1, 2)
    Q_for_rope = apply_rope_x(Q_for_rope, cos_q, sin_q)

    # KV Projections
    if kv_cache is None:
      compressed_kv = x @ self.W_dkv
      KV_for_lora, K_for_rope = torch.split(compressed_kv, [self.kv_proj_dim, self.qk_rope_dim], dim=-1)
      KV_for_lora = self.kv_layernorm(KV_for_lora)
    else:
      new_kv = x @ self.W_dkv
      compressed_kv = torch.cat([kv_cache, new_kv], dim=1)
      new_kv, new_K_for_rope = torch.split(new_kv, [self.kv_proj_dim, self.qk_rope_dim], dim=-1)
      old_kv, old_K_for_rope = torch.split(kv_cache, [self.kv_proj_dim, self.qk_rope_dim], dim=-1)
      new_kv = self.kv_layernorm(new_kv)
      old_kv = self.kv_layernorm(old_kv)
      KV_for_lora = torch.cat([old_kv, new_kv], dim=1)
      K_for_rope = torch.cat([old_K_for_rope, new_K_for_rope], dim=1)
            

      KV = KV_for_lora @ self.W_ukv
      KV = KV.view(B, -1, self.n_heads, self.dh+self.qk_nope_dim).transpose(1,2)
      K, V = torch.split(KV, [self.qk_nope_dim, self.dh], dim=-1)
      S_full = K.size(2)        

      # K Rope
      K_for_rope = K_for_rope.view(B, -1, 1, self.qk_rope_dim).transpose(1,2)
      cos_k = self.cos_cached[:, :, :S_full, :self.qk_rope_dim//2].repeat(1, 1, 1, 2)
      sin_k = self.sin_cached[:, :, :S_full, :self.qk_rope_dim//2].repeat(1, 1, 1, 2)
      K_for_rope = apply_rope_x(K_for_rope, cos_k, sin_k)

      # apply position encoding to each head
      K_for_rope = K_for_rope.repeat(1, self.n_heads, 1, 1)

      # split into multiple heads
      q_heads = torch.cat([Q, Q_for_rope], dim=-1)
      k_heads = torch.cat([K, K_for_rope], dim=-1)
      v_heads = V # already reshaped before the split

      # make attention mask
      mask = torch.ones((S,S_full), device=x.device)
      mask = torch.tril(mask, diagonal=past_length)
      mask = mask[None, None, :, :]

      sq_mask = mask == 1
      # attention
      x = torch.nn.functional.scaled_dot_product_attention(q_heads, k_heads, v_heads, attn_mask=sq_mask)
      x = x.transpose(1, 2).reshape(B, S, D)

      # apply projection
      x = x @ self.W_o.T
      return x, compressed_kv

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
  def __init__(self, d_model: int, block_size: int, num_bins: int, n_heads: int, norm_eps: float, dropout: float, latent_dim: int, projection_bins: int, bias: bool, device: str, n_ff: int = None, ffn_multiplier: int = None):
    super().__init__()
    self.self_att = MLA(d_model, n_heads, block_size)
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
  def __init__(self, params, vocab_size: int, block_size: int = None):
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
        n_heads=params.n_heads,
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
      torch.nn.init.kaiming_uniform_(module.weight, mean=0.0, std=0.05)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.sparse_(module.weight, mean=0.0, std=0.05)

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