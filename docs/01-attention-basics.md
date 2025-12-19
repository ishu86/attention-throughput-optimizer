# Attention Mechanism Basics

This tutorial covers the mathematical foundations of self-attention, the core mechanism behind transformer models.

## What is Attention?

Attention allows a model to focus on relevant parts of the input when producing an output. In self-attention, every position in a sequence can attend to every other position, enabling the model to capture long-range dependencies.

## The Three Vectors: Query, Key, Value

For each token in a sequence, we compute three vectors:

- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information do I provide?"

These are computed by linear projections of the input:

```
Q = X @ W_Q    # (seq_len, d_model) @ (d_model, d_k) = (seq_len, d_k)
K = X @ W_K    # (seq_len, d_model) @ (d_model, d_k) = (seq_len, d_k)
V = X @ W_V    # (seq_len, d_model) @ (d_model, d_v) = (seq_len, d_v)
```

## Scaled Dot-Product Attention

The attention mechanism computes:

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

Breaking this down step by step:

### Step 1: Compute Attention Scores

```
scores = Q @ K^T    # (seq_len, d_k) @ (d_k, seq_len) = (seq_len, seq_len)
```

This creates an **attention matrix** where entry (i, j) represents how much position i should attend to position j.

### Step 2: Scale

```
scaled_scores = scores / sqrt(d_k)
```

**Why scale?** Without scaling, when `d_k` is large, the dot products grow large in magnitude, pushing the softmax into regions with extremely small gradients. Scaling by `sqrt(d_k)` keeps the variance stable.

### Step 3: Apply Softmax

```
attention_weights = softmax(scaled_scores, dim=-1)
```

Softmax normalizes each row to sum to 1, creating a probability distribution over which positions to attend to.

### Step 4: Compute Weighted Sum

```
output = attention_weights @ V    # (seq_len, seq_len) @ (seq_len, d_v) = (seq_len, d_v)
```

Each output position is a weighted combination of all value vectors.

## Multi-Head Attention

Instead of a single attention function, we use multiple "heads" that attend to different aspects:

```python
def multi_head_attention(x, num_heads):
    d_model = x.shape[-1]
    d_k = d_model // num_heads

    # Split into heads
    Q = x @ W_Q  # Then reshape to (batch, num_heads, seq_len, d_k)
    K = x @ W_K
    V = x @ W_V

    # Apply attention per head
    head_outputs = []
    for h in range(num_heads):
        head_out = attention(Q[:, h], K[:, h], V[:, h])
        head_outputs.append(head_out)

    # Concatenate and project
    concat = torch.cat(head_outputs, dim=-1)
    output = concat @ W_O

    return output
```

## Causal (Autoregressive) Attention

For language modeling, position i should only attend to positions j <= i. This is enforced by masking:

```python
def causal_attention(Q, K, V):
    scores = Q @ K.T / sqrt(d_k)

    # Create causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores.masked_fill_(mask, float('-inf'))

    weights = softmax(scores, dim=-1)
    return weights @ V
```

The `-inf` values become 0 after softmax, preventing information leakage from future positions.

## PyTorch Implementation

Here's how standard attention looks in PyTorch:

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, causal=False):
    """
    Args:
        query: (batch, heads, seq_len, head_dim)
        key: (batch, heads, seq_len, head_dim)
        value: (batch, heads, seq_len, head_dim)
        causal: Whether to apply causal masking

    Returns:
        output: (batch, heads, seq_len, head_dim)
    """
    d_k = query.size(-1)

    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)

    # Apply causal mask if needed
    if causal:
        seq_len = query.size(-2)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device), diagonal=1)
        scores = scores.masked_fill(mask.bool(), float('-inf'))

    # Softmax and weighted sum
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)

    return output
```

## Key Takeaways

1. **Attention computes weighted sums** - Each output is a combination of all inputs, weighted by relevance
2. **QKV projections enable learning** - The model learns what to look for (Q), what to advertise (K), and what to provide (V)
3. **Scaling prevents gradient issues** - Division by `sqrt(d_k)` keeps softmax in a good operating range
4. **Multi-head = multiple perspectives** - Different heads can capture different types of relationships

## What's Next?

Now that you understand the mechanics, let's explore [why this approach has problems at long sequences](02-quadratic-problem.md).

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Jay Alammar
