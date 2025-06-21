import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class ModelArgs:
  d_model: int = 768  # BERT-base hidden size
  n_layers: int = 12  # BERT-base layers
  n_heads: int = 12   # BERT-base attention heads
  ffn_multiplier: int = 4
  n_ff: int = ffn_multiplier * d_model
  dropout: float = 0.1  # BERT dropout
  norm_eps: float = 1e-12  # BERT layer norm epsilon
  max_position_embeddings: int = 512  # BERT max sequence length
  n_experts: int = 4
  top_k: int = 2
  capacity_factor: int = 2
  vocab_size: int = 30522  # BERT vocab size
  type_vocab_size: int = 2  # BERT token type embeddings
  device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LayerNorm(nn.Module):
  def __init__(self, hidden_size, eps=1e-12):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.bias = nn.Parameter(torch.zeros(hidden_size))
    self.variance_epsilon = eps

  def forward(self, hidden_states):
    u = hidden_states.mean(-1, keepdim=True)
    s = (hidden_states - u).pow(2).mean(-1, keepdim=True)
    hidden_states = (hidden_states - u) / torch.sqrt(s + self.variance_epsilon)
    return self.weight * hidden_states + self.bias

class GELU(nn.Module):
  def forward(self, x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def linear_attention_kernel(q, k, v, eps=1e-6):
  # Apply feature map (ELU + 1 for positive features)
  q_feat = F.elu(q) + 1
  k_feat = F.elu(k) + 1

  # Compute KV matrix: [B, H, D, D]
  kv = torch.einsum('bhsd,bhse->bhde', k_feat, v)
  k_sum = k_feat.sum(dim=2)  # Compute normalizer: [B, H, D]

  # Compute output: [B, H, S, D]
  numerator = torch.einsum('bhsd,bhde->bhse', q_feat, kv)
  denominator = torch.einsum('bhsd,bhd->bhs', q_feat, k_sum)

  # Avoid division by zero
  denominator = torch.clamp(denominator, min=eps)
  output = numerator / denominator.unsqueeze(-1)
  return output

class BertLinearSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    if config.d_model % config.n_heads != 0:
      raise ValueError(f"Hidden size ({config.d_model}) must be divisible by number of heads ({config.n_heads})")

    self.num_attention_heads = config.n_heads
    self.attention_head_size = int(config.d_model / config.n_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    self.query = nn.Linear(config.d_model, self.all_head_size)
    self.key = nn.Linear(config.d_model, self.all_head_size)
    self.value = nn.Linear(config.d_model, self.all_head_size)

    self.dropout = nn.Dropout(config.dropout)

  def transpose_for_scores(self, x):
    new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)

  def forward(self, hidden_states, attention_mask=None):
    # Generate Q, K, V
    query_layer = self.transpose_for_scores(self.query(hidden_states))
    key_layer = self.transpose_for_scores(self.key(hidden_states))
    value_layer = self.transpose_for_scores(self.value(hidden_states))

    # Apply linear attention
    attention_output = linear_attention_kernel(query_layer, key_layer, value_layer)

    # Reshape back to [B, S, D]
    attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = attention_output.size()[:-2] + (self.all_head_size,)
    attention_output = attention_output.view(*new_context_layer_shape)

    return attention_output

class BertSelfOutput(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dense = nn.Linear(config.d_model, config.d_model)
    self.LayerNorm = LayerNorm(config.d_model, eps=config.norm_eps)
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    return hidden_states

class BertAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.self = BertLinearSelfAttention(config)
    self.output = BertSelfOutput(config)

  def forward(self, hidden_states, attention_mask=None):
    self_outputs = self.self(hidden_states, attention_mask)
    attention_output = self.output(self_outputs, hidden_states)
    return attention_output

class Expert(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dense = nn.Linear(config.d_model, config.n_ff)
    self.intermediate_act_fn = GELU()
    self.output_dense = nn.Linear(config.n_ff, config.d_model)
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, hidden_states):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.intermediate_act_fn(hidden_states)
    hidden_states = self.output_dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    return hidden_states

class NoisyTopkRouter(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.top_k = min(config.top_k, config.n_experts)
    self.n_experts = config.n_experts
    self.topkroute_linear = nn.Linear(config.d_model, config.n_experts, bias=False)
    self.noise_linear = nn.Linear(config.d_model, config.n_experts, bias=False)

  def forward(self, hidden_states):
    logits = self.topkroute_linear(hidden_states)
    
    if self.training and self.n_experts > 1:
      noise_logits = self.noise_linear(hidden_states)
      noise = torch.randn_like(logits) * F.softplus(noise_logits.clamp(max=10.0))
      noisy_logits = logits + noise
    else:
      noisy_logits = logits
    
    if self.n_experts == 1:
      top_k_logits = noisy_logits
      indices = torch.zeros_like(noisy_logits, dtype=torch.long)
    else:
      top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
    
    zeros = torch.full_like(noisy_logits, float('-inf'))
    sparse_logits = zeros.scatter(-1, indices, top_k_logits)
    router_output = F.softmax(sparse_logits, dim=-1)
    
    return router_output, indices

class SparseMoE(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.n_experts = max(1, config.n_experts)
    self.top_k = min(config.top_k, self.n_experts)
    self.capacity_factor = config.capacity_factor
    
    self.router = NoisyTopkRouter(config)
    self.experts = nn.ModuleList([Expert(config) for _ in range(self.n_experts)])

  def forward(self, hidden_states):
    batch_size, seq_len, d_model = hidden_states.shape
    original_shape = hidden_states.shape
    
    if self.n_experts == 1:
      return self.experts[0](hidden_states)
    
    flat_hidden = hidden_states.view(-1, d_model)
    gating_output, indices = self.router(hidden_states)
    flat_gating_output = gating_output.view(-1, self.n_experts)
    flat_indices = indices.view(-1, self.top_k)

    final_output = torch.zeros_like(flat_hidden)
    
    total_tokens = flat_hidden.size(0)
    tokens_per_expert = max(1, (total_tokens * self.top_k) // self.n_experts)
    expert_capacity = max(8, int(tokens_per_expert * self.capacity_factor))

    for expert_idx in range(self.n_experts):
      expert_mask = (flat_indices == expert_idx).any(dim=-1)
      expert_tokens = torch.nonzero(expert_mask, as_tuple=False).squeeze(-1)

      if expert_tokens.numel() == 0:
        continue
        
      if expert_tokens.numel() > expert_capacity:
        perm = torch.randperm(expert_tokens.numel(), device=expert_tokens.device)
        expert_tokens = expert_tokens[perm[:expert_capacity]]
      
      expert_tokens = expert_tokens[expert_tokens < flat_hidden.size(0)]
      if expert_tokens.numel() == 0:
        continue
      
      try:
        expert_input = flat_hidden[expert_tokens]
        expert_output = self.experts[expert_idx](expert_input)
        
        expert_gating = flat_gating_output[expert_tokens]
        gating_weights = expert_gating[:, expert_idx:expert_idx+1]
        weighted_output = expert_output * gating_weights
        
        final_output.scatter_add_(0, expert_tokens.unsqueeze(-1).expand_as(weighted_output), weighted_output)
      except Exception:
        continue

    return final_output.view(original_shape)

class BertIntermediate(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.moe = SparseMoE(config)

  def forward(self, hidden_states):
    return self.moe(hidden_states)

class BertOutput(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.LayerNorm = LayerNorm(config.d_model, eps=config.norm_eps)
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, hidden_states, input_tensor):
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    return hidden_states

class BertLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.attention = BertAttention(config)
    self.intermediate = BertIntermediate(config)
    self.output = BertOutput(config)

  def forward(self, hidden_states, attention_mask=None):
    attention_output = self.attention(hidden_states, attention_mask)
    intermediate_output = self.intermediate(attention_output)
    layer_output = self.output(intermediate_output, attention_output)
    return layer_output

class BertEncoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.n_layers)])

  def forward(self, hidden_states, attention_mask=None):
    for layer_module in self.layer:
      hidden_states = layer_module(hidden_states, attention_mask)
    return hidden_states

class BertEmbeddings(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.word_embeddings = nn.Embedding(config.vocab_size, config.d_model, padding_idx=0)
    self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)
    self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.d_model)

    self.LayerNorm = LayerNorm(config.d_model, eps=config.norm_eps)
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, input_ids, token_type_ids=None, position_ids=None):
    seq_length = input_ids.size(1)
    
    if position_ids is None:
      position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
      position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    
    if token_type_ids is None:
      token_type_ids = torch.zeros_like(input_ids)

    words_embeddings = self.word_embeddings(input_ids)
    position_embeddings = self.position_embeddings(position_ids)
    token_type_embeddings = self.token_type_embeddings(token_type_ids)

    embeddings = words_embeddings + position_embeddings + token_type_embeddings
    embeddings = self.LayerNorm(embeddings)
    embeddings = self.dropout(embeddings)
    return embeddings

class BertPooler(nn.Module):
  # BERT pooler for [CLS] token
  def __init__(self, config):
    super().__init__()
    self.dense = nn.Linear(config.d_model, config.d_model)
    self.activation = nn.Tanh()

  def forward(self, hidden_states):
    # Pool the [CLS] token (first token)
    first_token_tensor = hidden_states[:, 0]
    pooled_output = self.dense(first_token_tensor)
    pooled_output = self.activation(pooled_output)
    return pooled_output

class BertModel(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    
    self.embeddings = BertEmbeddings(config)
    self.encoder = BertEncoder(config)
    self.pooler = BertPooler(config)
    
    self.init_weights()

  def init_weights(self):
    def _init_weights(module):
      if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.padding_idx is not None:
          module.weight.data[module.padding_idx].zero_()
      elif isinstance(module, LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
    
    self.apply(_init_weights)

  def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
    if attention_mask is None:
      attention_mask = torch.ones_like(input_ids)

    # Convert attention mask to proper format
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)
    encoder_outputs = self.encoder(embedding_output, extended_attention_mask)
    pooled_output = self.pooler(encoder_outputs)

    return {
      'last_hidden_state': encoder_outputs,
      'pooler_output': pooled_output,
      'hidden_states': encoder_outputs,
    }

class BertForMaskedLM(nn.Module):
  # BERT for Masked Language Modeling
  def __init__(self, config):
    super().__init__()
    self.bert = BertModel(config)
    self.cls = nn.Linear(config.d_model, config.vocab_size, bias=False)

    # Tie weights between input embeddings and output layer
    self.cls.weight = self.bert.embeddings.word_embeddings.weight

  def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
    outputs = self.bert(input_ids, attention_mask, token_type_ids)
    sequence_output = outputs['last_hidden_state']
    prediction_scores = self.cls(sequence_output)

    loss = None
    if labels is not None:
      loss_fn = nn.CrossEntropyLoss()
      loss = loss_fn(prediction_scores.view(-1, self.bert.config.vocab_size), labels.view(-1))

    return {
      'loss': loss,
      'logits': prediction_scores,
      'hidden_states': outputs['hidden_states'],
      'pooler_output': outputs['pooler_output']
    }

class BertForSequenceClassification(nn.Module):
  # BERT for sequence classification
  def __init__(self, config, num_labels):
    super().__init__()
    self.num_labels = num_labels
    self.bert = BertModel(config)
    self.dropout = nn.Dropout(config.dropout)
    self.classifier = nn.Linear(config.d_model, num_labels)

  def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
    outputs = self.bert(input_ids, attention_mask, token_type_ids)
    pooled_output = outputs['pooler_output']
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)

    loss = None
    if labels is not None:
      if self.num_labels == 1:
        loss_fn = nn.MSELoss()
        loss = loss_fn(logits.squeeze(), labels)
      else:
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

    return {
      'loss': loss,
      'logits': logits,
      'hidden_states': outputs['hidden_states'],
      'pooler_output': outputs['pooler_output']
    }