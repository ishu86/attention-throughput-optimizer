# Introduction to Triton

This tutorial introduces Triton, a language for writing GPU kernels in Python.

## What is Triton?

Triton is a language and compiler for writing highly efficient GPU code. It provides:

- **Python-like syntax**: Much easier than CUDA C++
- **Block-level programming**: Work with tiles of data, not individual threads
- **Automatic optimizations**: Handles memory coalescing, shared memory, etc.
- **Performance close to CUDA**: Often 80-100% of hand-tuned CUDA

## Why Triton?

### CUDA is Hard

Writing efficient CUDA requires managing:
- Thread blocks and warps
- Shared memory allocation
- Memory coalescing
- Bank conflicts
- Register pressure
- Occupancy tuning

```cuda
// CUDA vector add - already complex!
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

### Triton is (Relatively) Easy

```python
# Triton vector add - cleaner!
@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)
```

## Core Concepts

### Programs and Blocks

In Triton, you write **programs** that operate on **blocks** of data:

- **Program**: One instance of your kernel, identified by `tl.program_id()`
- **Block**: A tile of data that one program processes

```python
@triton.jit
def my_kernel(x_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Each program handles BLOCK_SIZE elements
    pid = tl.program_id(0)  # Which program am I?

    # Calculate which elements this program handles
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
```

### The Grid

When launching a kernel, you specify a **grid** - how many programs to run:

```python
# Launch 1024 programs
grid = (1024,)
my_kernel[grid](x_ptr, n, BLOCK_SIZE=256)

# 2D grid: 64 x 32 = 2048 programs
grid = (64, 32)
my_2d_kernel[grid](...)
```

### Compile-Time Constants

Parameters marked with `tl.constexpr` are known at compile time:

```python
@triton.jit
def kernel(x_ptr, BLOCK_SIZE: tl.constexpr):  # Compile-time
    # BLOCK_SIZE is baked into the compiled kernel
    # Allows optimizations like loop unrolling
    offsets = tl.arange(0, BLOCK_SIZE)
```

## Basic Operations

### Loading Data

```python
# Load a block of data
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n  # Bounds checking
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
```

### Storing Data

```python
# Store a block of data
tl.store(out_ptr + offsets, result, mask=mask)
```

### Arithmetic

```python
# Element-wise operations work on entire blocks
y = x + 1.0
y = tl.exp(x)
y = tl.maximum(x, 0.0)
y = tl.where(condition, x, y)
```

### Reductions

```python
# Sum all elements in a block
total = tl.sum(x, axis=0)

# Max across a dimension
max_val = tl.max(x, axis=0)
```

### Matrix Multiply

```python
# Block matrix multiply (requires specific shapes)
c = tl.dot(a, b)  # a: (M, K), b: (K, N) -> c: (M, N)
```

## Complete Example: Softmax

Let's implement softmax, a key attention operation:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    input_stride,
    output_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute softmax over the last dimension."""
    # Which row is this program handling?
    row_idx = tl.program_id(0)

    # Pointers to this row
    row_start_in = input_ptr + row_idx * input_stride
    row_start_out = output_ptr + row_idx * output_stride

    # Load the row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row = tl.load(row_start_in + col_offsets, mask=mask, other=float('-inf'))

    # Compute max for numerical stability
    row_max = tl.max(row, axis=0)

    # Subtract max and exponentiate
    row = row - row_max
    numerator = tl.exp(row)

    # Sum for normalization
    denominator = tl.sum(numerator, axis=0)

    # Normalize
    softmax_output = numerator / denominator

    # Store result
    tl.store(row_start_out + col_offsets, softmax_output, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Wrapper function to call the Triton kernel."""
    n_rows, n_cols = x.shape

    # Allocate output
    output = torch.empty_like(x)

    # Determine block size (must be power of 2, fit in SRAM)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Launch kernel - one program per row
    grid = (n_rows,)
    softmax_kernel[grid](
        x, output, n_cols,
        x.stride(0), output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


# Test it
x = torch.randn(1024, 512, device='cuda')
y_triton = triton_softmax(x)
y_torch = torch.softmax(x, dim=-1)
print(f"Max error: {(y_triton - y_torch).abs().max():.2e}")
```

## 2D Blocks for Matrix Operations

For operations like attention, we work with 2D tiles:

```python
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Matrix multiply: C = A @ B"""
    # Program position in 2D grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute output block indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        # Load tiles of A and B
        a_ptrs = A + offs_m[:, None] * stride_am + (k_start + offs_k[None, :]) * stride_ak
        b_ptrs = B + (k_start + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn

        a_tile = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k_start + offs_k[None, :]) < K))
        b_tile = tl.load(b_ptrs, mask=((k_start + offs_k[:, None]) < K) & (offs_n[None, :] < N))

        # Accumulate
        acc += tl.dot(a_tile, b_tile)

    # Store result
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

## Triton vs CUDA: Key Differences

| Aspect | CUDA | Triton |
|--------|------|--------|
| Abstraction | Thread-level | Block-level |
| Language | C++ | Python |
| Memory management | Manual | Automatic |
| Coalescing | Manual | Automatic |
| Learning curve | Steep | Moderate |
| Performance ceiling | Highest | Near-optimal |

## Auto-Tuning

Triton supports automatic tuning of parameters:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(A, B, C, M, N, K, ...):
    ...
```

Triton will benchmark each configuration and select the fastest.

## What's Next?

Now let's see [how to apply these concepts to build attention kernels](06-kernel-development.md).

## References

- [Triton Documentation](https://triton-lang.org/)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [Triton GitHub](https://github.com/openai/triton)
