import json
with open('enigma2/config.json', 'r', encoding='utf-8') as file:
  params = json.load(file)

# required parameters
block_size = params['block_size']
d_model = params['d_model']
n_head = params['n_heads']
n_layers = params['n_layers']
learning_rate = params['learning_rate']
dropout = params['dropout']
norm_eps = params['norm_eps']

import torch
import torch.nn as nn
from torch.nn import functional as F

class RMSNorm(nn.Module):
  def __init__(self, dim: int, eps: float=1e-6):
    super().__init__()
    self.eps = eps
    self.wei = nn.Parameter(torch.ones(dim))
  
  def forward(self, x):
    out = self._norm(x.float()).type_as(x)
    return out * self.wei

class Head(nn.Module):
  def __init__(self, head_size, d_model, block_size, dropout):
    super().__init__()
    self.w_k = nn.Linear(d_model, head_size, bias=False)
    self.w_q = nn.Linear(d_model, head_size, bias=False)
    self.w_v = nn.Linear(d_model, head_size, bias=False)
    self.dropout = nn.Dropout(dropout)
    self.pos_embd = nn.Parameter(torch.randn(block_size, block_size, head_size))
    self.register_buffer('trill', torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x: torch.tensor, mask: bool= False):
    B, T, C = x.shape
    w_k, w_q, w_v = self.w_k(x), self.w_q(x), self.w_v(x)
    scores = torch.matmul(w_q, w_k.transpose(-2, -1)) / (w_k.shape[-1]**-0.5)

    if mask is True:
      scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    
    pos_embd = torch.einsum('btc,tvc->btv', w_q, self.pos_embd[:T, :T])
    scores = scores + pos_embd

    scores = F.softmax(scores, dim=-1)
    scores = self.dropout(scores)
    out = torch.matmul(scores, w_v)
    del x, w_k, w_q, w_v, scores, pos_embd
    return out

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, block_size, n_head, dropout):
    head_size = d_model // n_head
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size, d_model, block_size, dropout) for _ in range(n_head)])
    self.proj = nn.Linear(head_size * n_head, d_model)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x: torch.tensor, mask:bool):
    out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out

class FeedForward(nn.Module):
  def __init__(self, d_model, dropout):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(d_model, 5 * d_model),
      nn.GELU(),
      nn.Linear(5 * d_model, d_model),
      nn.Dropout(dropout),
      )

  def forward(self, x: torch.Tensor):
    return self.net(x)

class DecoderBlock(nn.Module):
  def __init__(self, d_model: int, block_size: int, n_head: int, norm_eps: float, dropout: float):
    super().__init__()
    self.self_att = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout, block_size=block_size)
    self.ffwd = FeedForward(d_model, dropout)
    self.dropout = nn.Dropout(dropout)
    self.norm = RMSNorm(d_model, eps=norm_eps)
  
  def forward(self, x: torch.tensor):
    x_out = self.self_att(self.norm(x), mask=True)
    x_out = x + self.dropout(x_out)

    x = self.ffwd(self.norm(x_out))
    x = x_out + self.dropout(x)

    x_out = self.self_att(self.norm(x), mask=False)
    x_out = x + self.dropout(x_out)
    del x

    return x_out

class Transformer(nn.Module):
  def __init__(self, vocab_size: int):
    super().__init__()
    self.block_size = block_size
    self.token_embeddings = nn.Embedding(vocab_size, d_model)
    self.decoder = nn.Sequential(*[DecoderBlock(n_head=n_head, d_model=d_model, dropout=dropout, norm_eps=norm_eps, block_size=block_size) for _ in range(n_layers)])
    self.norm_final = RMSNorm(d_model, eps=norm_eps)
    self.linear_final = nn.Linear(d_model, vocab_size)
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, idx, targets=None):
    B, T = idx.shape
    x = self.token_embeddings(idx)
    x = self.decoder(x)
    logits = self.linear_final(self.norm_final(x))

    if targets is None:
      loss = None

    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  @torch.no_grad()
  def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    self.eval()
    for _ in range(max_new_tokens):
    
      idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
      logits, _ = self(idx_cond)
      logits = logits[:, -1, :] / temperature
    
      if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
      
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)

      return idx