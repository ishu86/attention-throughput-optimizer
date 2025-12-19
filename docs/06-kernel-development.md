# Kernel Development Step-by-Step

This tutorial walks through building the linear attention kernels in this project.

## Overview

We'll develop kernels for:
1. Feature map (ELU + 1)
2. Bidirectional linear attention
3. Causal linear attention

Code reference: `src/ato/kernels/triton/linear_attn.py`

## Step 1: Feature Map Kernel

The feature map transforms Q and K before attention:

```python
@triton.jit
def elu_plus_one_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """ELU(x) + 1 feature map for linear attention.

    Why ELU + 1?
    - Ensures non-negativity (like softmax probabilities)
    - ELU provides smooth gradient at 0
    - Adding 1 prevents zeros (numerical stability)

    Math: ELU(x) + 1 = max(x, 0) + exp(min(x, 0))
    """
    # Get program ID - which block am I?
    pid = tl.program_id(0)

    # Calculate which elements this program handles
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)

    # Compute ELU(x) + 1 = max(x, 0) + exp(min(x, 0))
    pos = tl.maximum(x, 0.0)     # ReLU part
    neg = tl.minimum(x, 0.0)     # Negative part
    result = pos + tl.exp(neg)   # exp(negative) < 1

    # Store output
    tl.store(out_ptr + offsets, result, mask=mask)
```

**Python wrapper:**

```python
def apply_feature_map(x: Tensor) -> Tensor:
    """Apply ELU+1 feature map using Triton kernel."""
    out = torch.empty_like(x)
    n_elements = x.numel()

    BLOCK_SIZE = 1024  # Elements per program
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)  # Number of programs

    elu_plus_one_kernel[grid](
        x.view(-1), out.view(-1), n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
```

## Step 2: Bidirectional Linear Attention

Bidirectional attention has two phases:
1. Compute `KV = K^T @ V` (once)
2. Compute `Output = (Q @ KV) / (Q @ K_sum)` (per position)

### Kernel 1: K^T @ V

```python
@triton.jit
def linear_attention_kv_kernel(
    K, V, KV,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_kvb, stride_kvh, stride_kvd1, stride_kvd2,
    B, H, N, D,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute KV = K^T @ V for each batch and head.

    This is the key optimization: instead of O(n²) attention matrix,
    we compute a O(d²) state matrix.

    Input shapes:
        K: (B, H, N, D) - N tokens, D features
        V: (B, H, N, D)
    Output shape:
        KV: (B, H, D, D) - d² regardless of sequence length!

    The kernel computes: KV[b,h,d1,d2] = sum_n K[b,h,n,d1] * V[b,h,n,d2]
    """
    # 3D grid: (batch*heads, d1_blocks, d2_blocks)
    pid_bh = tl.program_id(0)
    pid_d1 = tl.program_id(1)
    pid_d2 = tl.program_id(2)

    b = pid_bh // H
    h = pid_bh % H

    # Which output block are we computing?
    d1_start = pid_d1 * BLOCK_D
    d2_start = pid_d2 * BLOCK_D

    d1_offs = d1_start + tl.arange(0, BLOCK_D)
    d2_offs = d2_start + tl.arange(0, BLOCK_D)

    d1_mask = d1_offs < D
    d2_mask = d2_offs < D

    # Accumulator: will hold sum over all N tokens
    acc = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)

    # Loop over sequence dimension - this is the sum over n
    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        # Load K block: shape (BLOCK_N, BLOCK_D)
        k_ptrs = K + b * stride_kb + h * stride_kh + \
                 n_offs[:, None] * stride_kn + d1_offs[None, :] * stride_kd
        k_block = tl.load(k_ptrs, mask=n_mask[:, None] & d1_mask[None, :], other=0.0)

        # Load V block: shape (BLOCK_N, BLOCK_D)
        v_ptrs = V + b * stride_vb + h * stride_vh + \
                 n_offs[:, None] * stride_vn + d2_offs[None, :] * stride_vd
        v_block = tl.load(v_ptrs, mask=n_mask[:, None] & d2_mask[None, :], other=0.0)

        # Accumulate K^T @ V: (BLOCK_D, BLOCK_N) @ (BLOCK_N, BLOCK_D)
        # tl.trans transposes the first argument
        acc += tl.dot(tl.trans(k_block), v_block)

    # Store the computed KV block
    kv_ptrs = KV + b * stride_kvb + h * stride_kvh + \
              d1_offs[:, None] * stride_kvd1 + d2_offs[None, :] * stride_kvd2
    tl.store(kv_ptrs, acc.to(KV.dtype.element_ty), mask=d1_mask[:, None] & d2_mask[None, :])
```

### Kernel 2: Q @ KV with Normalization

```python
@triton.jit
def linear_attention_qkv_kernel(
    Q, KV, K_sum, Out,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kvb, stride_kvh, stride_kvd1, stride_kvd2,
    stride_ksb, stride_ksh, stride_ksd,
    stride_ob, stride_oh, stride_on, stride_od,
    B, H, N, D,
    eps,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute Output = (Q @ KV) / (Q @ K_sum).

    This is the query phase - for each query, we:
    1. Multiply by the precomputed KV state
    2. Normalize by the sum of keys

    The normalization ensures outputs are well-scaled,
    analogous to softmax normalization in standard attention.
    """
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    # Which tokens is this program handling?
    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # Load Q block: (BLOCK_N, BLOCK_D)
    q_ptrs = Q + b * stride_qb + h * stride_qh + \
             n_offs[:, None] * stride_qn + d_offs[None, :] * stride_qd
    q_block = tl.load(q_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

    # Load K_sum: (BLOCK_D,) - sum of all keys for normalization
    ks_ptrs = K_sum + b * stride_ksb + h * stride_ksh + d_offs * stride_ksd
    k_sum = tl.load(ks_ptrs, mask=d_mask, other=0.0)

    # Compute normalizer: Q @ K_sum -> (BLOCK_N,)
    # This is sum_j phi(q_i) * phi(k_j) for all j
    normalizer = tl.sum(q_block * k_sum[None, :], axis=1)
    normalizer = tl.maximum(normalizer, eps)  # Prevent division by zero

    # Compute Q @ KV
    # Need to handle case where D > BLOCK_D
    acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    for d1_start in range(0, D, BLOCK_D):
        d1_offs = d1_start + tl.arange(0, BLOCK_D)
        d1_mask = d1_offs < D

        # Load Q slice for inner dimension
        q1_ptrs = Q + b * stride_qb + h * stride_qh + \
                  n_offs[:, None] * stride_qn + d1_offs[None, :] * stride_qd
        q1_block = tl.load(q1_ptrs, mask=n_mask[:, None] & d1_mask[None, :], other=0.0)

        # Load KV slice
        kv_ptrs = KV + b * stride_kvb + h * stride_kvh + \
                  d1_offs[:, None] * stride_kvd1 + d_offs[None, :] * stride_kvd2
        kv_block = tl.load(kv_ptrs, mask=d1_mask[:, None] & d_mask[None, :], other=0.0)

        # Accumulate: (BLOCK_N, BLOCK_D) @ (BLOCK_D, BLOCK_D)
        acc += tl.dot(q1_block, kv_block)

    # Normalize and store
    out_block = acc / normalizer[:, None]

    out_ptrs = Out + b * stride_ob + h * stride_oh + \
               n_offs[:, None] * stride_on + d_offs[None, :] * stride_od
    tl.store(out_ptrs, out_block.to(Out.dtype.element_ty), mask=n_mask[:, None] & d_mask[None, :])
```

## Step 3: Putting It Together

```python
def triton_linear_attention_bidirectional(q, k, v, eps=1e-6):
    """Bidirectional linear attention using Triton.

    Overall flow:
    1. Compute KV = K^T @ V  (d×d matrix, computed once)
    2. Compute K_sum = sum(K)  (d vector, for normalization)
    3. For each query: output = (Q @ KV) / (Q @ K_sum)

    Complexity: O(n × d²) instead of O(n² × d)
    """
    B, H, N, D = q.shape

    # Allocate intermediate and output tensors
    kv = torch.empty(B, H, D, D, device=q.device, dtype=q.dtype)
    k_sum = k.sum(dim=2)  # (B, H, D)
    out = torch.empty_like(q)

    # Configure block sizes
    BLOCK_N = min(64, N)
    BLOCK_D = min(64, D)

    # Launch KV kernel: one program per (batch, head, d1_block, d2_block)
    grid_kv = (B * H, triton.cdiv(D, BLOCK_D), triton.cdiv(D, BLOCK_D))
    linear_attention_kv_kernel[grid_kv](
        k, v, kv,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        kv.stride(0), kv.stride(1), kv.stride(2), kv.stride(3),
        B, H, N, D,
        BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )

    # Launch output kernel: one program per (batch, head, n_block)
    grid_qkv = (B * H, triton.cdiv(N, BLOCK_N))
    linear_attention_qkv_kernel[grid_qkv](
        q, kv, k_sum, out,
        # ... strides and parameters
    )

    return out
```

## Step 4: Causal Attention

Causal attention is trickier because we can't precompute a single KV state - it changes for each position.

### Naive Approach: Sequential

```python
def causal_linear_attention_naive(q, k, v, eps=1e-6):
    """Sequential causal attention - correct but slow."""
    B, H, N, D = q.shape

    S = torch.zeros(B, H, D, D, device=q.device)  # Running state
    Z = torch.zeros(B, H, D, device=q.device)     # Running normalizer
    outputs = []

    for i in range(N):
        # Update state with current key-value
        S = S + torch.einsum('bhd,bhe->bhde', k[:,:,i], v[:,:,i])
        Z = Z + k[:,:,i]

        # Query the state
        num = torch.einsum('bhd,bhde->bhe', q[:,:,i], S)
        denom = torch.einsum('bhd,bhd->bh', q[:,:,i], Z).unsqueeze(-1)
        outputs.append(num / (denom + eps))

    return torch.stack(outputs, dim=2)
```

**Problem**: This has O(N) sequential steps - can't parallelize!

### Better: Cumulative Sum

```python
def causal_linear_attention_cumsum(q, k, v, eps=1e-6):
    """Use cumsum for implicit loop - still has memory overhead."""
    B, H, N, D = q.shape

    # Compute all KV pairs: (B, H, N, D, D)
    kv = torch.einsum('bhnd,bhnv->bhndv', k, v)

    # Cumulative sum gives S[i] = sum_{j<=i} kv[j]
    S_cumsum = torch.cumsum(kv, dim=2)  # (B, H, N, D, D)
    Z_cumsum = torch.cumsum(k, dim=2)   # (B, H, N, D)

    # Query each position
    out = torch.einsum('bhnd,bhndv->bhnv', q, S_cumsum)
    norm = torch.einsum('bhnd,bhnd->bhn', q, Z_cumsum).unsqueeze(-1)

    return out / (norm + eps)
```

**Trade-off**: Parallelizable but needs O(N × D²) memory for S_cumsum.

### Kernel: Chunked Processing

Our Triton kernel uses chunking for a balance:

```python
@triton.jit
def causal_linear_attention_chunk_kernel(
    Q, K, V,
    S_in, Z_in,    # State from previous chunks
    Out,
    S_out, Z_out,  # State for next chunks
    # ... parameters
    chunk_size: tl.constexpr,
):
    """Process one chunk of causal linear attention.

    Strategy:
    - Divide sequence into chunks
    - Process each chunk sequentially within the kernel
    - Pass state between chunks

    This balances parallelism (across chunks) with sequential
    processing (within chunks).
    """
    # Load initial state from previous chunk
    s_acc = tl.load(S_in + ...)
    z_acc = tl.load(Z_in + ...)

    # Process each position in the chunk sequentially
    for t in range(chunk_size):
        pos = chunk_start + t

        # Load K, V, Q for this position
        k_t = tl.load(K + pos * stride...)
        v_t = tl.load(V + pos * stride...)
        q_t = tl.load(Q + pos * stride...)

        # Update state: S += k @ v^T
        s_acc += k_t[:, None] * v_t[None, :]
        z_acc += k_t

        # Compute output: out = (q @ S) / (q @ z)
        out_t = tl.sum(q_t[:, None] * s_acc, axis=0)
        norm_t = tl.sum(q_t * z_acc)
        out_t = out_t / tl.maximum(norm_t, eps)

        # Store output
        tl.store(Out + pos * stride..., out_t)

    # Store final state for next chunk
    tl.store(S_out + ..., s_acc)
    tl.store(Z_out + ..., z_acc)
```

## Debugging Tips

### 1. Start Simple
Test with small tensors where you can verify by hand:
```python
x = torch.tensor([[1.0, 2.0, 3.0]], device='cuda')
```

### 2. Compare with PyTorch
Always verify against a PyTorch reference:
```python
def test_kernel():
    x = torch.randn(8, 32, 1024, 64, device='cuda')
    out_triton = triton_linear_attention(x, x, x)
    out_pytorch = pytorch_linear_attention(x, x, x)
    assert torch.allclose(out_triton, out_pytorch, atol=1e-3)
```

### 3. Print Intermediate Values
Use `tl.device_print` for debugging (slow, use sparingly):
```python
if pid == 0:
    tl.device_print("acc", acc)
```

### 4. Check Shapes and Strides
Most bugs come from wrong indexing:
```python
print(f"Q: {Q.shape}, strides: {Q.stride()}")
```

## Performance Tips

1. **Power-of-2 block sizes**: Required for `tl.dot`
2. **Match hardware**: BLOCK_SIZE should be multiple of warp size (32)
3. **Minimize HBM access**: Compute as much as possible before writing
4. **Use autotuning**: Let Triton find optimal parameters

## What's Next?

See [the optimization journey](07-optimization-journey.md) for lessons learned while developing these kernels.
