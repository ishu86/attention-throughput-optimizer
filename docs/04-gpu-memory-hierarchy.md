# GPU Memory Hierarchy

This tutorial explains GPU memory architecture and why it matters for attention performance.

## The Memory Wall

Modern GPUs have massive compute power but relatively limited memory bandwidth. This creates the "memory wall" - computation is often bottlenecked by how fast we can move data.

| GPU | Compute (TF32) | Memory Bandwidth | Compute:Memory Ratio |
|-----|----------------|------------------|---------------------|
| A100 | 156 TFLOPS | 2 TB/s | 78:1 |
| H100 | 989 TFLOPS | 3.35 TB/s | 295:1 |
| RTX 4090 | 83 TFLOPS | 1 TB/s | 83:1 |

This means for every byte we read/write, we can do ~100-300 operations. If we can't reuse data, computation sits idle waiting for memory.

## GPU Memory Types

### HBM (High Bandwidth Memory)

- **What**: Main GPU memory (the "40GB" or "80GB" you see in specs)
- **Size**: 16GB - 80GB
- **Bandwidth**: 1-3 TB/s
- **Latency**: ~100s of cycles

This is where tensors live. It's large but relatively slow.

### SRAM (Shared Memory / L1 Cache)

- **What**: On-chip memory close to compute units
- **Size**: 20KB - 228KB per SM
- **Bandwidth**: ~19 TB/s (A100)
- **Latency**: ~20 cycles

Much faster but very limited in size.

### Registers

- **What**: Fastest memory, directly accessible by threads
- **Size**: 256KB per SM
- **Bandwidth**: Essentially unlimited
- **Latency**: 1 cycle

Used for intermediate computations.

## Memory Hierarchy Visualization

```
┌─────────────────────────────────────────────────┐
│                    HBM                          │
│            40-80 GB @ 1-3 TB/s                  │
│                (Global Memory)                   │
└─────────────────────────────────────────────────┘
                      ↑↓
┌─────────────────────────────────────────────────┐
│                 L2 Cache                        │
│              40MB @ ~4 TB/s                     │
└─────────────────────────────────────────────────┘
                      ↑↓
┌───────────┬───────────┬───────────┬─────────────┐
│    SM 0   │    SM 1   │    SM 2   │    ...      │
├───────────┼───────────┼───────────┼─────────────┤
│  SRAM     │  SRAM     │  SRAM     │  SRAM       │
│  192KB    │  192KB    │  192KB    │  192KB      │
│  @19TB/s  │  @19TB/s  │  @19TB/s  │  @19TB/s    │
├───────────┼───────────┼───────────┼─────────────┤
│ Registers │ Registers │ Registers │ Registers   │
│  256KB    │  256KB    │  256KB    │  256KB      │
└───────────┴───────────┴───────────┴─────────────┘
```

## Why Attention is Memory-Bound

Standard attention has low **arithmetic intensity** (operations per byte):

```python
# Q, K, V: (batch, heads, seq_len, head_dim)
# Assume float16: 2 bytes per element

# Read Q, K: 2 * n * d * 2 bytes = 4nd bytes
# Compute Q @ K^T: 2 * n * n * d FLOPs
# Write attention scores: n * n * 2 bytes
# Read attention scores + V: n * n * 2 + n * d * 2 bytes
# Compute scores @ V: 2 * n * n * d FLOPs
# Write output: n * d * 2 bytes

# Total FLOPs: 4 * n² * d
# Total bytes: ~4nd + 2n² + 2nd ≈ 2n² (for large n)
# Arithmetic intensity: 4n²d / 2n² = 2d FLOPs/byte
```

For head_dim = 64: **128 FLOPs/byte**

A100 compute:memory ratio is **78:1 (TF32) or 312:1 (FP16 tensor cores)**

This means attention is **memory-bound** - we're waiting for data, not compute!

## The FlashAttention Insight

FlashAttention asks: what if we never store the full (n × n) attention matrix?

### Tiled Computation

Instead of computing the full attention matrix:

1. **Load tiles** of Q, K, V that fit in SRAM
2. **Compute partial attention** for those tiles
3. **Accumulate results** using online softmax
4. **Never write** the full attention matrix to HBM

```
Traditional:                    FlashAttention:

Q (HBM) ─┬─→ Scores (HBM)      Q (HBM) ─┐
K (HBM) ─┘       ↓              K (HBM) ─┼─→ SRAM tiles ─→ Output (HBM)
          ┌─ Softmax (HBM)      V (HBM) ─┘      ↑
V (HBM) ──┴─→ Output (HBM)              [Loop in SRAM]

HBM reads/writes: 3n²d + 2n²   HBM reads/writes: 3nd + nd = 4nd
```

### Complexity Breakdown

| Operation | Standard | FlashAttention |
|-----------|----------|----------------|
| HBM reads | O(n²) | O(n) |
| HBM writes | O(n²) | O(n) |
| Compute | O(n²d) | O(n²d) |
| Memory bound? | Yes | No |

FlashAttention keeps O(n²) compute but reduces memory access from O(n²) to O(n).

## SRAM Tiling Strategy

The key is choosing tile sizes that:
1. Fit in SRAM
2. Maximize data reuse
3. Keep compute units busy

For A100 with 192KB SRAM:
- Each tile can hold: 192KB / 2 bytes = 96K elements
- For head_dim=64: Can fit ~1500 vectors
- Typical block size: 64-128 tokens

```python
# Pseudocode for tiled attention
BLOCK_M, BLOCK_N = 64, 64  # Tile sizes

for q_block in range(0, seq_len, BLOCK_M):
    # Load Q tile to SRAM
    Q_tile = Q[q_block:q_block+BLOCK_M]  # (BLOCK_M, d)

    acc = zeros(BLOCK_M, d)
    normalizer = zeros(BLOCK_M)

    for k_block in range(0, seq_len, BLOCK_N):
        # Load K, V tiles to SRAM
        K_tile = K[k_block:k_block+BLOCK_N]  # (BLOCK_N, d)
        V_tile = V[k_block:k_block+BLOCK_N]  # (BLOCK_N, d)

        # Compute attention for this tile (in SRAM)
        scores = Q_tile @ K_tile.T  # (BLOCK_M, BLOCK_N)
        scores = scores / sqrt(d)

        # Online softmax update
        max_new = max(max_old, scores.max())
        scores = exp(scores - max_new)
        scale = exp(max_old - max_new)

        acc = acc * scale + scores @ V_tile
        normalizer = normalizer * scale + scores.sum()

    # Write final output to HBM
    Output[q_block:q_block+BLOCK_M] = acc / normalizer
```

## Memory Coalescing

GPUs read memory in **coalesced transactions** (128 bytes at a time). Uncoalesced access can waste 90%+ of bandwidth.

### Good (Coalesced)
```python
# Threads 0-31 access consecutive memory
data[thread_id]  # All 32 threads → 1 transaction
```

### Bad (Strided)
```python
# Threads access memory with large strides
data[thread_id * stride]  # 32 transactions instead of 1!
```

For attention, this means:
- Store tensors in contiguous layouts
- Access along the innermost dimension
- Transpose when needed for better access patterns

## Bank Conflicts

SRAM is organized into **banks** (32 on modern GPUs). If multiple threads access the same bank, accesses are serialized.

```python
# Bank conflict (all threads access bank 0):
shared_mem[thread_id * 32]  # Serialized!

# No conflict:
shared_mem[thread_id]  # Parallel access
```

Good kernel implementations pad shared memory to avoid conflicts.

## Practical Implications for Linear Attention

Linear attention has different memory patterns:

```
Standard Attention:
- Read: Q, K, V (3 × n × d)
- Write: n² attention scores (HUGE)
- Read again: attention scores
- Write: output (n × d)

Linear Attention (Bidirectional):
- Read: K, V (2 × n × d)
- Write: KV state (d × d) - SMALL
- Read: Q, KV (n × d + d × d)
- Write: output (n × d)
```

Linear attention is naturally more memory-efficient because it never creates the n² matrix!

## What's Next?

Now that you understand GPU memory, let's learn [how to write kernels in Triton](05-triton-introduction.md).

## References

- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) - Dao et al., 2022
- [Dissecting the NVIDIA Volta GPU Architecture](https://arxiv.org/abs/1804.06826) - Jia et al., 2018
- [CUDA C++ Programming Guide - Memory Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy)
