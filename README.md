# Learning GPU Optimization: Implementing Linear Attention in Triton

> **Educational Project**: This repository documents my journey learning GPU kernel optimization
> by implementing linear attention mechanisms from scratch in Triton. The goal is education,
> not novelty.

## Learning Objectives

- Understand attention mechanism complexity (O(n^2) vs O(n))
- Learn GPU memory hierarchy (HBM, SRAM, caching)
- Write optimized kernels in Triton
- Profile and benchmark GPU code
- Validate theoretical complexity in practice

## Key Result

Achieved 50x speedup over PyTorch SDPA at 64K sequence length for bidirectional attention,
validating the O(n) vs O(n^2) complexity advantage of linear attention.

| Sequence Length | PyTorch SDPA | Triton Linear | Speedup |
|-----------------|--------------|---------------|---------|
| 4K              | 0.42ms       | 0.45ms        | ~1x     |
| 8K              | 1.60ms       | 0.47ms        | 3.4x    |
| 16K             | 5.01ms       | 0.47ms        | 10.6x   |
| 32K             | 18.03ms      | 0.74ms        | 24.3x   |
| 64K             | 71.4ms       | 1.43ms        | **50x** |

## Prior Work & Acknowledgments

This implementation was built for learning purposes. Production-ready implementations exist:

- **[flash-linear-attention](https://github.com/fla-org/flash-linear-attention)** -
  Comprehensive FLA library (use this for production!)
- **[flash-bidirectional-linear-attention](https://github.com/fla-org/flash-bidirectional-linear-attention)** -
  Bidirectional implementations
- **[FlashAttention](https://github.com/Dao-AILab/flash-attention)** -
  Original memory-efficient attention

This project reproduces known results to learn GPU optimization techniques.

## Documentation

### Tutorial Series
1. [Attention Mechanism Basics](docs/01-attention-basics.md)
2. [The Quadratic Complexity Problem](docs/02-quadratic-problem.md)
3. [Linear Attention Theory](docs/03-linear-attention-theory.md)
4. [GPU Memory Hierarchy](docs/04-gpu-memory-hierarchy.md)
5. [Introduction to Triton](docs/05-triton-introduction.md)
6. [Kernel Development Step-by-Step](docs/06-kernel-development.md)
7. [Optimization Journey](docs/07-optimization-journey.md)

### Interactive Notebooks
- [Attention Visualization](notebooks/01_attention_visualization.ipynb)
- [Complexity Analysis](notebooks/02_complexity_analysis.ipynb)
- [Memory Profiling](notebooks/03_memory_profiling.ipynb)
- [Benchmark Comparison](notebooks/04_benchmark_comparison.ipynb)

## Quick Start

```bash
# Install dependencies
pip install torch triton matplotlib pandas

# Install package
pip install -e .

# Run benchmarks
python examples/02_benchmark_reproduction.py

# Compare with existing implementations
python examples/03_compare_implementations.py
```

## Project Structure

```
attention-throughput-optimizer/
├── docs/                    # Tutorial documentation
├── notebooks/               # Interactive Jupyter notebooks
├── src/ato/
│   ├── attention/           # Attention implementations
│   ├── kernels/             # Triton kernels
│   ├── benchmark/           # Benchmarking framework
│   └── profiling/           # Memory & latency profiling
├── examples/                # Usage examples
├── tests/                   # Test suite
└── results/                 # Benchmark results & plots
```

## What I Learned

### Key Insights
1. **Memory is the bottleneck**: Attention is memory-bound, not compute-bound
2. **Tiling matters**: Small tiles that fit in SRAM = huge speedups
3. **Linear attention trade-offs**: Faster at long sequences, but different semantics
4. **Triton is powerful**: High-level Python-like syntax, performance close to CUDA

### Challenges Faced
- Causal attention requires parallel scan (still slower than FlashAttention)
- Feature map choice significantly impacts accuracy
- Numerical stability in cumulative operations
- Memory alignment and coalescing for performance

### Performance Learnings
- PyTorch 2.2+ has FlashAttention built-in (hard to beat!)
- Bidirectional linear attention wins at 8K+ sequences
- Python loops kill GPU performance (use Triton kernels)
- Chunked processing: memory vs speed trade-off

## Reproducibility

All benchmarks use:
- Hardware: NVIDIA A100 40GB
- CUDA: 12.1
- PyTorch: 2.2.0
- Triton: 2.3.0
- Batch size: 8, Head dim: 64, Num heads: 8

See `results/` for raw data and plots.

## Contributing

This is a learning project, but improvements welcome:
- Better explanations in docs
- Additional visualization notebooks
- Corrections to technical content
- Optimization suggestions

## Further Reading

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer
- [FlashAttention](https://arxiv.org/abs/2205.14135) - IO-aware attention
- [Transformers are RNNs](https://arxiv.org/abs/2006.16236) - Linear attention
- [Linear Attention for Bidirectional Sequence Modeling](https://arxiv.org/abs/2502.16249) - LION

### Tutorials
- [Triton Documentation](https://triton-lang.org/)
- [Understanding Flash Attention](https://alexdremov.me/understanding-flash-attention-writing-the-algorithm-from-scratch-in-triton/)
- [Efficient Attention Survey](https://attention-survey.github.io/)

## License

Apache-2.0 - feel free to learn from and adapt this code.

## Star History

If this helped you learn GPU optimization, consider starring the repo!
