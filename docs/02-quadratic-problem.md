# The Quadratic Complexity Problem

This tutorial explains why standard attention becomes prohibitively expensive at long sequence lengths.

## The Core Issue

Recall the attention computation:

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

The attention matrix `Q @ K^T` has shape `(seq_len, seq_len)`. This means:

- **Time Complexity**: O(n²) where n = sequence length
- **Memory Complexity**: O(n²) for storing the attention matrix

## Concrete Numbers

Let's compute actual memory requirements for the attention matrix alone:

| Sequence Length | Attention Matrix Size | Memory (fp16) | Memory (fp32) |
|-----------------|----------------------|---------------|---------------|
| 512             | 262K                 | 0.5 MB        | 1 MB          |
| 1,024           | 1M                   | 2 MB          | 4 MB          |
| 2,048           | 4M                   | 8 MB          | 16 MB         |
| 4,096           | 16M                  | 32 MB         | 64 MB         |
| 8,192           | 67M                  | 128 MB        | 256 MB        |
| 16,384          | 268M                 | 512 MB        | 1 GB          |
| 32,768          | 1B                   | 2 GB          | 4 GB          |
| 65,536          | 4B                   | 8 GB          | 16 GB         |
| 131,072         | 17B                  | 32 GB         | 64 GB         |

**This is per attention layer, per batch element, per head!**

For a typical model with:
- Batch size: 8
- Num heads: 32
- Num layers: 32

At 64K sequence length:
- Per-layer attention memory: 8 × 32 × 8 GB = **2 TB** (impossible!)

## Why It Matters

### 1. GPU Memory Limits

Even the largest GPUs have limited memory:
- A100: 40GB or 80GB
- H100: 80GB
- Consumer GPUs: 8-24GB

A single attention layer at 16K sequence length with batch size 8 needs more memory than most GPUs have.

### 2. Compute Time Scaling

The compute time grows quadratically. If 1K tokens takes 1ms:

| Sequence Length | Relative Time | Actual Time (approx) |
|-----------------|---------------|---------------------|
| 1K              | 1x            | 1 ms                |
| 2K              | 4x            | 4 ms                |
| 4K              | 16x           | 16 ms               |
| 8K              | 64x           | 64 ms               |
| 16K             | 256x          | 256 ms              |
| 32K             | 1024x         | ~1 second           |
| 64K             | 4096x         | ~4 seconds          |

This makes training on long documents, codebases, or conversations impractical.

## Real-World Implications

### Context Window Limitations

Early transformers had small context windows:
- Original Transformer (2017): 512 tokens
- GPT-2 (2019): 1024 tokens
- GPT-3 (2020): 2048 tokens

These weren't arbitrary choices—they were memory constraints!

### The Long Document Problem

Many real applications need long contexts:
- Legal documents: 10K-100K+ tokens
- Codebases: 50K-500K+ tokens
- Books: 100K-1M+ tokens
- Conversations: Grows unbounded

Standard attention simply cannot handle these lengths.

## Visualizing the Problem

```python
import matplotlib.pyplot as plt
import numpy as np

seq_lengths = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
memory_gb = (seq_lengths ** 2) * 2 / (1024**3)  # fp16 bytes to GB

plt.figure(figsize=(10, 6))
plt.plot(seq_lengths, memory_gb, 'b-o', linewidth=2)
plt.axhline(y=40, color='r', linestyle='--', label='A100 40GB')
plt.axhline(y=80, color='g', linestyle='--', label='A100 80GB')
plt.xlabel('Sequence Length')
plt.ylabel('Attention Matrix Memory (GB)')
plt.title('Attention Memory Scaling (Single Head, Single Batch)')
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.grid(True)
plt.show()
```

## Solutions Overview

Several approaches address the quadratic complexity:

### 1. Sparse Attention
Only compute attention for a subset of positions:
- Local attention (sliding window)
- Strided patterns
- Learned sparsity (e.g., Longformer, BigBird)

Complexity: O(n × k) where k << n

### 2. Low-Rank Approximation
Approximate the attention matrix with lower-rank factors:
- Linformer projects K, V to lower dimension
- Performer uses random features

Complexity: O(n × k²) where k is projection dimension

### 3. Linear Attention
Restructure the computation to avoid materializing the full attention matrix:
- Use kernel trick: φ(Q) @ (φ(K)^T @ V)
- Compute in O(n) time and memory

This is what we implement in this project! See [Linear Attention Theory](03-linear-attention-theory.md).

### 4. Memory-Efficient Attention (FlashAttention)
Keep O(n²) complexity but avoid storing the full attention matrix:
- Tile-based computation
- Recompute attention during backward pass
- Uses SRAM instead of HBM

See [GPU Memory Hierarchy](04-gpu-memory-hierarchy.md).

## The Trade-off Space

| Approach | Time Complexity | Memory Complexity | Exact Attention? |
|----------|-----------------|-------------------|------------------|
| Standard | O(n²) | O(n²) | Yes |
| Sparse | O(n × k) | O(n × k) | No |
| Linear | O(n) | O(n) | No |
| FlashAttention | O(n²) | O(n) | Yes |

**Key insight**: There's no free lunch. Linear attention changes the computation semantics. FlashAttention maintains exact attention but still has O(n²) time complexity.

## What's Next?

Let's dive into [how linear attention achieves O(n) complexity](03-linear-attention-theory.md) through mathematical reformulation.

## References

- [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732) - Tay et al., 2020
- [Long Range Arena](https://arxiv.org/abs/2011.04006) - Benchmark for long-range models
