import torch
import torch.nn as nn
from torch.nn import functional as F
import math

d_model, n_head, n_layers, block_size = 512, 8, 6, 1024
dropout, norm_eps = 0.1, 1e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RoPE(nn.Module):
  def __init__(self, dim, max_seq_len=10000):
    super().__init__()
    self.dim, self.max_seq_len = dim, max_seq_len
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer('inv_freq', inv_freq)

  def forward(self, x, seq_len):
    t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
    freqs = torch.outer(t, self.inv_freq)
    cos, sin = freqs.cos(), freqs.sin()
    return cos, sin

def apply_rope(x, cos, sin):
  x1, x2 = x[..., ::2], x[..., 1::2]
  return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class GroupedQueryAttention(nn.Module):
  def __init__(self, d_model, n_head, n_kv_head, dropout, block_size):
    super().__init__()
    self.d_model, self.n_head, self.n_kv_head = d_model, n_head, n_kv_head
    self.head_dim = d_model // n_head
    self.n_rep = n_head // n_kv_head

    self.q_proj = nn.Linear(d_model, n_head * self.head_dim, bias=False)
    self.k_proj = nn.Linear(d_model, n_kv_head * self.head_dim, bias=False)
    self.v_proj = nn.Linear(d_model, n_kv_head * self.head_dim, bias=False)
    self.o_proj = nn.Linear(n_head * self.head_dim, d_model, bias=False)

    self.dropout = nn.Dropout(dropout)
    self.rope = RoPE(self.head_dim)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
  def forward(self, x, mask=False):
    B, T, C = x.shape
    q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
    k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
    v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
    cos, sin = self.rope(x, T)
    cos, sin = cos[None, None, :, :], sin[None, None, :, :]
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    k = k.repeat_interleave(self.n_rep, dim=1)
    v = v.repeat_interleave(self.n_rep, dim=1)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    if mask: scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    weights = self.dropout(weights)

    out = torch.matmul(weights, v)
    out = out.transpose(1, 2).contiguous().view(B, T, C)
    return self.o_proj(out)

class FeedForward(nn.Module):
  def __init__(self, d_model, dropout):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(d_model, 4 * d_model),
      nn.GELU(),
      nn.Linear(4 * d_model, d_model),
      nn.Dropout(dropout)
    )
  def forward(self, x):
    return self.net(x)

class TransformerBlock(nn.Module):
  def __init__(self, d_model, n_head, n_kv_head, norm_eps, dropout, block_size):
    super().__init__()
    self.attn = GroupedQueryAttention(d_model, n_head, n_kv_head, dropout, block_size)
    self.ffn = FeedForward(d_model, dropout)
    self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
    self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)

  def forward(self, x, mask=False):
    x = x + self.attn(self.norm1(x), mask)
    x = x + self.ffn(self.norm2(x))
    return x

class Transformer(nn.Module):
  def __init__(self, vocab_size, n_kv_head=2):
    super().__init__()
    self.block_size = block_size
    self.token_emb = nn.Embedding(vocab_size, d_model)
    self.blocks = nn.ModuleList([TransformerBlock(d_model, n_head, n_kv_head, norm_eps, dropout, block_size) for _ in range(n_layers)])
    self.norm_final = nn.LayerNorm(d_model, eps=norm_eps)
    self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    self.dropout = nn.Dropout(dropout)
    self.apply(self._init_weights)
    
  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      
  def forward(self, idx, targets=None):
    B, T = idx.shape
    x = self.dropout(self.token_emb(idx))
    for block in self.blocks: x = block(x, mask=True)
    x = self.norm_final(x)
    logits = self.lm_head(x)
    if targets is None: loss = None
    else:
      B, T, C = logits.shape
      logits_flat, targets_flat = logits.view(B*T, C), targets.view(B*T)
      loss = F.cross_entropy(logits_flat, targets_flat)
    return logits, loss