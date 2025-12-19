#!/usr/bin/env python3
"""
Basic Usage - Introduction to ATO Framework

This example demonstrates the core concepts of the ATO (Attention Throughput Optimizer)
framework. You'll learn:

1. How to use the AttentionRegistry to discover and create attention implementations
2. How to configure attention modules with AttentionConfig
3. How to run forward passes and interpret outputs
4. How to profile memory usage

This is a great starting point for understanding the framework before diving into
benchmarking and optimization.

Usage:
    python examples/01_basic_usage.py
"""

import torch

# Import ATO components
from ato.attention import AttentionRegistry, AttentionConfig
from ato.profiling import MemoryProfiler


def main():
    # =========================================================================
    # Step 1: Environment Setup
    # =========================================================================
    print("=" * 60)
    print("Step 1: Environment Setup")
    print("=" * 60)

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16  # Use FP16 for efficiency
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = "cpu"
        dtype = torch.float32  # CPU doesn't benefit from FP16
        print("Running on CPU (limited functionality)")

    # =========================================================================
    # Step 2: Exploring Available Attention Mechanisms
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 2: Available Attention Mechanisms")
    print("=" * 60)

    # The AttentionRegistry provides a plugin architecture for attention implementations
    # This makes it easy to add new implementations without changing existing code

    print("\nAll registered mechanisms:")
    for name in AttentionRegistry.list_registered():
        print(f"  - {name}")

    print("\nMechanisms available on your hardware:")
    for info in AttentionRegistry.list_available():
        status = "available" if info["available"] else "unavailable"
        reason = f" ({info.get('reason', '')})" if not info["available"] else ""
        print(f"  - {info['name']}: {status}{reason}")

    # =========================================================================
    # Step 3: Creating Attention Instances
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 3: Creating Attention Instances")
    print("=" * 60)

    # AttentionConfig holds all configuration parameters
    # embed_dim: Total embedding dimension (must be divisible by num_heads)
    # num_heads: Number of attention heads
    # causal: Whether to use causal masking (for autoregressive models)

    config = AttentionConfig(
        embed_dim=512,      # 512-dimensional embeddings
        num_heads=8,        # 8 attention heads -> head_dim = 64
        device=device,
        dtype=dtype,
        causal=True,        # Autoregressive attention
    )

    # Create different attention types using the registry
    # The registry handles instantiation and configuration

    # Standard scaled dot-product attention (PyTorch native)
    standard_attn = AttentionRegistry.create("standard", config)
    print(f"\nCreated: {standard_attn.backend_name}")
    print(f"  Type: {standard_attn.attention_type}")
    print(f"  Head dim: {config.head_dim}")

    # Multi-head attention with learnable projections
    mha_config = config  # Same config works
    mha = AttentionRegistry.create("multi_head", mha_config).to(device=device, dtype=dtype)
    print(f"\nCreated: {mha.backend_name}")
    print(f"  Has projections: Q, K, V, Output")

    # =========================================================================
    # Step 4: Forward Pass
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 4: Running Forward Pass")
    print("=" * 60)

    # Create sample input tensors
    # Shape: (batch_size, sequence_length, embed_dim)
    batch_size = 4
    seq_len = 256

    q = torch.randn(batch_size, seq_len, config.embed_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, config.embed_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, config.embed_dim, device=device, dtype=dtype)

    print(f"\nInput shapes:")
    print(f"  Query: {q.shape}")
    print(f"  Key:   {k.shape}")
    print(f"  Value: {v.shape}")

    # Run forward pass
    with torch.no_grad():
        output = standard_attn(q, k, v)

    # AttentionOutput contains:
    # - output: The attention output tensor
    # - attention_weights: Optional attention weights (if computed)
    # - metadata: Information about the computation

    print(f"\nOutput:")
    print(f"  Shape: {output.output.shape}")
    print(f"  Metadata: {output.metadata}")

    # =========================================================================
    # Step 5: Memory Profiling
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 5: Memory Profiling")
    print("=" * 60)

    if device == "cuda":
        # MemoryProfiler tracks GPU memory allocation
        profiler = MemoryProfiler()

        # Profile different attention implementations
        with profiler.profile("standard_attention"):
            with torch.no_grad():
                _ = standard_attn(q, k, v)

        with profiler.profile("multi_head_attention"):
            with torch.no_grad():
                _ = mha(q, k, v)

        # Print memory report
        print("\nMemory Usage:")
        profiler.print_report()

        # You can also get theoretical estimates
        estimates = standard_attn.estimate_memory(batch_size, seq_len, dtype)
        print(f"\nTheoretical memory estimate for standard attention:")
        for key, value in estimates.items():
            if isinstance(value, int):
                print(f"  {key}: {value / 1e6:.2f} MB")
            else:
                print(f"  {key}: {value}")
    else:
        print("Memory profiling requires CUDA.")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print("""
Key concepts covered:
1. AttentionRegistry - Plugin system for attention implementations
2. AttentionConfig - Configuration dataclass for attention parameters
3. AttentionOutput - Standardized output with metadata
4. MemoryProfiler - GPU memory tracking utility

Next steps:
- Run 02_benchmark_reproduction.py to see performance comparisons
- Check the notebooks/ directory for interactive exploration
- Read docs/01-attention-basics.md for the math behind attention
""")


if __name__ == "__main__":
    main()
