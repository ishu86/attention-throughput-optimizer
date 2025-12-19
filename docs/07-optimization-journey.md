# Optimization Journey

This document shares the lessons learned while developing and optimizing the linear attention kernels.

## What Worked

### 1. Bidirectional Attention is the Sweet Spot

The biggest wins came from bidirectional (non-causal) linear attention at long sequences:

| Sequence Length | PyTorch SDPA | Triton Linear | Speedup |
|-----------------|--------------|---------------|---------|
| 4K              | 0.42ms       | 0.45ms        | ~1x     |
| 8K              | 1.60ms       | 0.47ms        | 3.4x    |
| 16K             | 5.01ms       | 0.47ms        | 10.6x   |
| 32K             | 18.03ms      | 0.74ms        | 24.3x   |
| 64K             | 71.4ms       | 1.43ms        | **50x** |

**Why it works**: The O(n) vs O(n²) complexity advantage is real and measurable. At 64K tokens, we do ~4000x fewer operations.

### 2. Fusing Feature Maps

Initial version called separate kernels for feature map and attention:

```python
# Slow: 3 kernel launches, 3 memory round-trips
q = feature_map_kernel(q)
k = feature_map_kernel(k)
output = attention_kernel(q, k, v)
```

Fused version applies feature map inline:

```python
# Fast: 1 kernel launch
@triton.jit
def linear_attention_kernel(...):
    q = apply_feature_map(q)  # In-register
    k = apply_feature_map(k)  # In-register
    # ... continue with attention
```

**Speedup**: ~1.5x from avoiding HBM round-trips.

### 3. Block Size Tuning

The right block sizes made a huge difference:

| BLOCK_N | BLOCK_D | Throughput |
|---------|---------|------------|
| 32      | 32      | 45 GB/s    |
| 64      | 64      | 78 GB/s    |
| 128     | 64      | 92 GB/s    |
| 64      | 128     | 85 GB/s    |

**Sweet spot**: BLOCK_N=128, BLOCK_D=64 for most configurations.

Triton's autotuning helps find optimal parameters automatically:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_D': 64}),
        triton.Config({'BLOCK_N': 128, 'BLOCK_D': 64}),
        triton.Config({'BLOCK_N': 64, 'BLOCK_D': 128}),
    ],
    key=['N', 'D'],
)
```

### 4. Float32 Accumulation

Using float32 for intermediate accumulation prevented precision issues:

```python
# Accumulate in fp32, store in fp16
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
# ... computation ...
tl.store(out_ptr, acc.to(tl.float16), mask=mask)
```

## What Didn't Work (Or Was Harder Than Expected)

### 1. Causal Attention Speed

Causal linear attention was disappointing compared to FlashAttention:

| Sequence Length | FlashAttention-2 | Triton Linear Causal |
|-----------------|------------------|---------------------|
| 4K              | 0.35ms           | 0.52ms              |
| 8K              | 1.20ms           | 1.85ms              |
| 16K             | 4.50ms           | 7.2ms               |

**Why**: FlashAttention's tiled approach can parallelize across the sequence dimension, while causal linear attention has inherent sequential dependencies.

The parallel scan approach helps but doesn't fully close the gap:
- Scan has O(log n) depth, not O(1)
- State updates (d×d matrices) are larger than FlashAttention's scalars
- More memory traffic per step

**Lesson**: For causal attention at moderate lengths (< 8K), FlashAttention is still the better choice.

### 2. Numerical Stability

Early versions had NaN issues at long sequences due to cumulative operations:

```python
# Problematic: values grow unbounded
S = cumsum(k @ v.T)  # Can overflow

# Better: periodic renormalization
if step % renorm_interval == 0:
    S = S / S.norm()
```

The ELU+1 feature map helped by keeping values in a reasonable range, but edge cases still required careful handling.

### 3. Memory Layout Assumptions

Initial kernels assumed contiguous memory:

```python
# Wrong: assumed stride[3] == 1
x_ptr + n * D + d
```

Had to support arbitrary strides for PyTorch tensor views:

```python
# Correct: use actual strides
x_ptr + n * stride_n + d * stride_d
```

This added complexity but was necessary for real-world usage.

### 4. Python Loop Overhead

The first causal implementation used Python loops:

```python
for i in range(seq_len):  # Terrible!
    output[i] = query_state(q[i], state)
    state = update_state(state, k[i], v[i])
```

Even with Triton kernels for the inner operations, the loop overhead was massive:
- 32K iterations × ~10μs overhead = 320ms just for loops!

Moving the loop inside a single kernel eliminated this.

## Key Insights

### 1. Memory is the Bottleneck

Attention is memory-bound, not compute-bound. Optimizations that reduce memory access (tiling, fusion) beat those that reduce compute.

```
Roofline analysis:
- A100 compute: 156 TFLOPS (TF32)
- A100 memory: 2 TB/s
- Attention arithmetic intensity: ~128 FLOPs/byte
- Attainable performance: 256 TFLOPS (limited by memory!)
```

### 2. Linear Attention Trades Accuracy for Speed

Linear attention is not a drop-in replacement for softmax attention. The kernel approximation changes the attention distribution:

```python
# Softmax attention: sharp, peaked distributions
# Linear attention: smoother, more diffuse attention

# Test on real tasks to verify quality is acceptable!
```

In practice, this matters more for some tasks than others. Language modeling seems more sensitive than encoding.

### 3. Hardware Utilization Matters

Triton makes it easy to write kernels but doesn't guarantee efficiency. Profiling revealed issues:

```
Initial kernel:
- Achieved bandwidth: 800 GB/s (40% of peak)
- SM occupancy: 45%

After optimization:
- Achieved bandwidth: 1.6 TB/s (80% of peak)
- SM occupancy: 75%
```

Key fixes:
- Increased block sizes to better fill SMs
- Reduced register pressure by splitting large blocks
- Improved memory access patterns

### 4. Test Against Production Code

Comparing against fla-org's implementations was humbling but educational:

```
fla-org/flash-linear-attention vs our implementation:
- Their forward: 0.8ms
- Our forward: 1.1ms (38% slower)

Their advantages:
- Years of optimization
- Hardware-specific tuning
- Backward pass optimization
```

**Lesson**: For production, use established libraries. For learning, implement yourself.

## Recommendations

### For Learning

1. **Start with bidirectional attention** - Easier to implement and debug
2. **Profile everything** - Use `torch.cuda.Event` and NSight
3. **Compare with baselines** - PyTorch, FlashAttention, fla-org
4. **Read the papers** - Understanding the theory helps with implementation

### For Production

1. **Use flash-linear-attention** - Battle-tested and optimized
2. **Benchmark on your workload** - Synthetic benchmarks can be misleading
3. **Consider hybrid approaches** - Standard attention for short, linear for long
4. **Validate quality** - Linear attention can degrade model quality

## What I Would Do Differently

1. **Start with profiling** - Understand bottlenecks before optimizing
2. **Focus on one variant** - Bidirectional attention has clearer wins
3. **Use autotuning earlier** - Manual tuning is tedious and hardware-specific
4. **Write more tests** - Caught several bugs late that tests would have found

## Future Directions

Areas for continued exploration:

1. **Gated linear attention** - Adds expressivity with minimal overhead
2. **Chunk-wise parallelization** - Better causal attention parallelism
3. **Mixed precision** - FP8 for even faster computation
4. **Multi-GPU scaling** - Tensor parallelism for linear attention

## References

- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) - Production implementation
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) - Memory-efficient attention
- [RWKV](https://arxiv.org/abs/2305.13048) - Linear attention in practice
- [Mamba](https://arxiv.org/abs/2312.00752) - State space models (related ideas)
