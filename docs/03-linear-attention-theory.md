# Linear Attention Theory

This tutorial explains the mathematical reformulation that enables O(n) attention computation.

## The Key Insight: Associativity

Standard attention computes:

```
output = softmax(Q @ K^T) @ V
```

The bottleneck is the `(n × n)` attention matrix from `Q @ K^T`.

**Linear attention** removes the softmax and changes the order of operations:

```
output = φ(Q) @ (φ(K)^T @ V)
```

Where `φ` is a feature map. Let's see why this works.

## The Kernel Trick

The softmax attention can be viewed as:

```
Attention_i = Σ_j [sim(q_i, k_j) / Σ_l sim(q_i, k_l)] × v_j
```

Where `sim(q, k) = exp(q · k / sqrt(d))` is the softmax similarity.

The key observation: if we replace the exponential similarity with a **kernel function** that can be decomposed as a dot product of feature maps:

```
sim(q, k) = φ(q)^T × φ(k)
```

Then we can rewrite attention as:

```
Attention_i = Σ_j [φ(q_i)^T × φ(k_j) / Σ_l φ(q_i)^T × φ(k_l)] × v_j
            = φ(q_i)^T × [Σ_j φ(k_j) × v_j^T] / [φ(q_i)^T × Σ_l φ(k_l)]
```

## The Associativity Trick

For bidirectional (non-causal) attention, we can compute:

```
KV = Σ_j φ(k_j) × v_j^T    # Shape: (d, d) - compute once!
Z = Σ_j φ(k_j)             # Shape: (d,) - normalizer

output_i = (φ(q_i)^T @ KV) / (φ(q_i)^T @ Z)
```

**This is O(n)!** We:
1. Compute KV and Z in O(n × d²) - linear in n
2. Query each position in O(d²) - constant per position
3. Total: O(n × d²) instead of O(n² × d)

Since typically `d << n` for long sequences, this is a massive win.

## Visual Comparison

### Standard Attention
```
Q (n×d) @ K^T (d×n) = Attention (n×n)  ← BOTTLENECK
Attention (n×n) @ V (n×d) = Output (n×d)

Memory: O(n²)
Time: O(n² × d)
```

### Linear Attention (Bidirectional)
```
K^T (d×n) @ V (n×d) = KV (d×d)  ← Compute once
Q (n×d) @ KV (d×d) = Output (n×d)

Memory: O(n × d + d²)
Time: O(n × d²)
```

## Feature Maps

The choice of feature map `φ` is crucial. Common options:

### 1. ELU + 1 (Used in this project)

```python
def elu_plus_one(x):
    return F.elu(x) + 1
```

- Ensures non-negativity (like softmax)
- Simple and fast
- Works well in practice

### 2. Softmax Approximation

```python
def softmax_feature(x):
    return F.softmax(x, dim=-1)
```

- Closer to original attention semantics
- More expensive to compute

### 3. Random Fourier Features (RFF)

```python
def rff(x, omega, d_features):
    proj = x @ omega  # Random projection
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) / sqrt(d_features)
```

- Approximates Gaussian kernel
- Used in Performer

### 4. ReLU

```python
def relu_feature(x):
    return F.relu(x)
```

- Simplest option
- Can have numerical issues (outputs can be zero)

## Causal (Autoregressive) Attention

For causal attention, position i should only use keys/values from j ≤ i. This changes the formula:

```
KV_i = Σ_{j≤i} φ(k_j) × v_j^T    # Cumulative sum
Z_i = Σ_{j≤i} φ(k_j)

output_i = (φ(q_i)^T @ KV_i) / (φ(q_i)^T @ Z_i)
```

This can be computed as a **recurrent update**:

```python
def causal_linear_attention(Q, K, V):
    # φ already applied to Q, K
    B, H, N, D = Q.shape

    S = torch.zeros(B, H, D, D)  # Running state
    Z = torch.zeros(B, H, D)     # Running normalizer
    outputs = []

    for i in range(N):
        k_i = K[:, :, i]  # (B, H, D)
        v_i = V[:, :, i]  # (B, H, D)
        q_i = Q[:, :, i]  # (B, H, D)

        # Update state
        S = S + k_i.unsqueeze(-1) @ v_i.unsqueeze(-2)  # (B, H, D, D)
        Z = Z + k_i  # (B, H, D)

        # Query state
        num = (q_i.unsqueeze(-2) @ S).squeeze(-2)  # (B, H, D)
        denom = (q_i * Z).sum(-1, keepdim=True)    # (B, H, 1)
        out_i = num / (denom + eps)

        outputs.append(out_i)

    return torch.stack(outputs, dim=2)  # (B, H, N, D)
```

### The Problem: Sequential Computation

The causal version requires **sequential updates** - we can't parallelize over the sequence dimension. This limits the speedup for causal attention.

### Solution: Parallel Scan

The cumulative sum can be computed in O(log n) parallel steps using a **parallel prefix sum** (scan) algorithm:

```
Step 1: Compute pairwise sums
Step 2: Combine pairs of pairs
...
Step log(n): Complete
```

This is implemented in our Triton kernels. See `src/ato/kernels/triton/linear_attn.py`.

## Complexity Comparison

| Variant | Time | Memory | Parallelizable? |
|---------|------|--------|-----------------|
| Standard Attention | O(n²d) | O(n²) | Yes |
| Linear Bidirectional | O(nd²) | O(d²) | Yes |
| Linear Causal (naive) | O(nd²) | O(d²) | No |
| Linear Causal (scan) | O(nd² log n) | O(d²) | Yes |

## Trade-offs vs Standard Attention

### Advantages
1. **O(n) complexity** - Scales to very long sequences
2. **Constant memory per head** - Only O(d²) regardless of sequence length
3. **RNN-like inference** - Can process tokens one at a time

### Disadvantages
1. **Different semantics** - Not equivalent to softmax attention
2. **Lower expressivity** - Linear kernel is less powerful than softmax
3. **Numerical stability** - Cumulative operations can have precision issues
4. **Causal is slower** - Sequential nature limits parallelism

## When to Use Linear Attention

**Good for:**
- Very long sequences (8K+)
- Bidirectional tasks (encoding)
- Streaming/online inference

**Not ideal for:**
- Short sequences (standard attention is faster)
- Tasks requiring precise attention patterns
- When exact softmax behavior matters

## Code Reference

See our implementations:
- `src/ato/attention/efficient/linear.py` - Python implementation
- `src/ato/kernels/triton/linear_attn.py` - Optimized Triton kernels

## What's Next?

Understanding [GPU memory hierarchy](04-gpu-memory-hierarchy.md) is essential for writing efficient kernels.

## References

- [Transformers are RNNs](https://arxiv.org/abs/2006.16236) - Katharopoulos et al., 2020
- [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794) - Choromanski et al., 2020
- [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174) - Schlag et al., 2021
