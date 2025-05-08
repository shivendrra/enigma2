# Model Architecture Documentation

This document explains the **VQ‑Transformer** architecture implemented in **`model.py`**. It is intended for users with a basic understanding of deep learning but unfamiliar with this specific model. We dive into each component, its role, and design choices.

## Table of Contents

* [1. Overview](#1-overview)
* [2. Configuration](#2-configuration)
* [3. Normalization: RMSNorm vs LayerNorm](#3-normalization-rmsnorm-vs-layernorm)
* [4. SplineEdge Activation](#4-splineedge-activation)
* [5. KANLayer Projections](#5-kanlayer-projections)
* [6. Rotary Positional Encoding (RoPE)](#6-rotary-positional-encoding-rope)
* [7. Latent-Attention Head](#7-latent-attention-head)
* [8. Multi-Head Latent Attention](#8-multi-head-latent-attention)
* [9. Feed-Forward & Projections](#9-feed-forward--projections)
* [10. Decoder Blocks & Transformer Stack](#10-decoder-blocks--transformer-stack)
* [11. Final Output Layer](#11-final-output-layer)
* [12. Design Rationale](#12-design-rationale)

## 1. Overview

At a high level, the model is a **decoder‑only Transformer** tailored for DNA sequences:

* **Input**: A sequence of integer‑encoded DNA tokens (`A,C,G,T`).
* **Core**: Stacked **`DecoderBlock`** modules, each containing:

  1. **RMSNorm** normalization
  2. **Multi‑Head Latent Attention** (with rotary embeddings)
  3. **Feed‑Forward** network with KAN projections
* **Output**: A linear layer mapping to logits over the vocabulary.

This architecture supports discrete latent representations via decomposition in attention and flexible non‑linear projections (KAN). It is optimized for sequence modeling in genomics.

## 2. Configuration

All hyperparameters live in `ModelConfig` or `config.json` (these aren't final):

| Parameter        | Description                              |     Default |
| - | - | -: |
| `block_size`     | Maximum sequence length                  |         256 |
| `n_head`         | Number of attention heads                |          12 |
| `n_layers`       | Number of decoder layers                 |          12 |
| `d_model`        | Hidden dimension size                    |         512 |
| `n_latent`       | Latent bottleneck dimension in attention |          64 |
| `ffn_multiplier` | FFN hidden size = multiplier × `d_model` |           4 |
| `dropout`        | Dropout rate                             |         0.2 |
| `norm_eps`       | Epsilon for RMSNorm                      |        1e-5 |
| `num_bins`       | Number of spline bins in KAN and FFN     |           4 |
| `kan_min/max`    | Input range for spline edges             | –1.0 to 1.0 |

These settings balance capacity and computational cost for genomic sequences.

## 3. Normalization: RMSNorm vs LayerNorm

**LayerNorm** normalizes each feature by subtracting the mean and dividing by standard deviation across the feature dimension:

${LN(x) = (x - μ) / √(σ² + ε)}$

**RMSNorm** (Root Mean Square Norm) simplifies this by omitting mean centering:

${RMSNorm(x) = x / √(mean(x²) + ε)}$

### Advantages of RMSNorm

* **Compute Efficiency**: Only second‑moment needed (no mean), slightly cheaper.
* **Stability**: Empirically robust in deep Transformer stacks.
* **Simpler Implementation**: Fewer operations.

RMSNorm ensures stable activation scales with minimal overhead.

## 4. SplineEdge Activation

`SplineEdge` implements a **learnable piecewise-linear** activation function:

* **Knots**: Uniformly spaced x‑coordinates (`num_bins + 1`).
* **Heights**: Learnable y‑values at each knot.
* **Interpolation**: Linear interpolation between adjacent knots.

Use Cases:

* Replace fixed activations (e.g., GELU) with a flexible, data-driven shape.
* Capture non‑linearities specific to genomic patterns.

This underpins the KANLayer and optional FFN projections.

## 5. KANLayer Projections

`KANLayer` (Kolmogorov‑Arnold Network) replaces dense linear layers:

* **Architecture**: For each of the D×D weight entries, a separate `SplineEdge` transforms one input feature to one output.
* **Computation**: Sum over contributions from each input channel via its edge.

Advantages:

* **Universal Approximation**: Piecewise splines can approximate arbitrary continuous functions (Kolmogorov–Arnold theorem).
* **Parameter Efficiency**: Instead of a full D² weight matrix, models complex functions via shared spline parameters.

Use in both attention projection and FFN output layers for richer learned mappings.

## 6. Rotary Positional Encoding (RoPE)

RoPE encodes positions by rotating query/key vectors in the complex plane:

1. Precompute sin/cos embeddings for each position and head dimension.
2. Split each vector into even/odd components and apply rotation:

      ${q' = [q_even * cos – q_odd * sin, q_even * sin + q_odd * cos]}$

Benefits:

* **Relative Awareness**: Maintains token distances implicitly in dot‑product space.
* **Unbounded Extrapolation**: Handles longer sequences than trained on.

RoPE integrates seamlessly into attention without extra parameters.

## 7. Latent-Attention Head

Each `LatentHead` computes queries, keys, and values with a small bottleneck:

1. **Query**: Linear projection to `head_size`.
2. **Key/Value Decomposition**:

   * First project to `latent_dim` (smaller than `head_size`).
   * Then project back up to `head_size`.
3. **RoPE**: Apply rotary embeddings to Q and K.
4. **Attention**:

   * Scaled dot-product: `QKᵀ / √d`.
   * Causal mask for autoregressive decoding.
   * Softmax + dropout.
   * Weighted sum with V.

Why decompose K/V?

* **Compression**: Reduces parameters and enforces a bottleneck.
* **Regularization**: Encourages extraction of key features.

This discrete latent step is inspired by vector‑quantized bottlenecks.

## 8. Multi-Head Latent Attention

`MultiHeadLatentAttention` runs `n_head` parallel `LatentHead`s:

1. Concatenate their outputs → shape `[B, T, d_model]`.
2. **KANLayer** projects this concatenation back to `d_model`:

   * Enables non‑linear mixing across heads.
3. Dropout for regularization.

This provides both multi‑perspective attention and flexible mixing.

## 9. Feed-Forward & Projections

Standard Transformer FFN is two linear layers with activation:

${FFN(x) = W₂·GELU(W₁·x)}$

Here, we extend it by:

1. **Hidden Size**: `ffn_multiplier × d_model`.
2. **KANLayer Projection**: After the 2‑layer net, we apply another spline‑based projection.

This yields a richer, learnable mapping beyond simple dense layers.

## 10. Decoder Blocks & Transformer Stack

Each `DecoderBlock` follows a **pre‑norm** pattern:

1. **Norm → Attention → Add**
2. **Norm → FFN → Add**

Sequence:

```python
x1 = x + dropout(self_att(RMSNorm(x)))
x2 = x1 + dropout(ffn(RMSNorm(x1)))
```

A stack of `n_layers` repeats this, producing deep contextualized representations.

## 11. Final Output Layer

After the decoder stack:

1. **RMSNorm** to stabilize outputs.
2. **Linear** to map `d_model` to `vocab_size` (4 for DNA bases).
3. **Cross-Entropy Loss** for next-token prediction during training.

For inference, softmax over logits yields base probabilities.

## 12. Design Rationale

* **RMSNorm** for efficiency and stability in deep stacks.
* **KAN/Spline** for universal approximation of complex genomic patterns.
* **RoPE** to encode relative positions naturally.
* **Latent Bottleneck** to compress key/value and reduce overfitting.
* **Pre‑norm Architecture** to improve gradient flow in deep networks.

This combination tailors the Transformer to DNA sequence modeling, balancing expressivity and computational efficiency.
