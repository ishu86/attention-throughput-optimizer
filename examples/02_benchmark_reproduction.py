#!/usr/bin/env python3
"""
Benchmark Reproduction - Reproduce the 50x Speedup Result

This example reproduces the key benchmark showing linear attention achieving
50x speedup over standard attention at 64K sequence length.

You'll learn:
1. How to set up comprehensive benchmarks
2. How to compare standard vs linear attention at various sequence lengths
3. How to identify the crossover point where linear attention becomes faster

Usage:
    python examples/02_benchmark_reproduction.py

Expected output (on A100 40GB):
    Sequence Length | PyTorch SDPA | Linear Attention | Speedup
    ----------------|--------------|------------------|--------
    4K              | ~0.4ms       | ~0.5ms           | ~1x
    8K              | ~1.6ms       | ~0.5ms           | ~3x
    16K             | ~5.0ms       | ~0.5ms           | ~10x
    32K             | ~18ms        | ~0.7ms           | ~25x
    64K             | ~71ms        | ~1.4ms           | ~50x
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path


def pytorch_sdpa_bidir(q, k, v):
    """PyTorch scaled dot-product attention (bidirectional)."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=False)


def linear_attention_bidir(q, k, v, eps=1e-6):
    """
    Linear attention with ELU+1 feature map (bidirectional).

    This achieves O(n) complexity by computing:
        KV = K^T @ V  (d×d matrix, computed once)
        Output = (Q @ KV) / (Q @ sum(K))

    Instead of the O(n²) standard attention:
        Output = softmax(Q @ K^T) @ V
    """
    # Apply feature map: ELU + 1 ensures non-negativity
    q = F.elu(q) + 1
    k = F.elu(k) + 1

    # Key insight: compute K^T @ V first (O(n × d²))
    # This gives a (d × d) matrix regardless of sequence length!
    kv = torch.einsum('bhnd,bhnv->bhdv', k, v)  # (B, H, D, D)

    # Sum of keys for normalization
    k_sum = k.sum(dim=2)  # (B, H, D)

    # Query the KV state: Q @ KV
    out = torch.einsum('bhnd,bhdv->bhnv', q, kv)  # (B, H, N, D)

    # Normalize: Q @ K_sum
    norm = torch.einsum('bhnd,bhd->bhn', q, k_sum).unsqueeze(-1)

    return out / (norm + eps)


def benchmark(fn, q, k, v, warmup=5, iters=20):
    """Benchmark a function with CUDA timing."""
    # Warmup
    for _ in range(warmup):
        _ = fn(q, k, v)
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        _ = fn(q, k, v)

        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return np.mean(times), np.std(times)


def main():
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: This benchmark requires a CUDA GPU.")
        return

    print("=" * 70)
    print("Benchmark: Linear Attention vs PyTorch SDPA")
    print("=" * 70)
    print(f"\nGPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    # Configuration - matches paper benchmarks
    BATCH_SIZE = 8
    NUM_HEADS = 8
    HEAD_DIM = 64

    # Sequence lengths to test
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    if gpu_memory > 70e9:  # 80GB GPU
        seq_lengths = [4096, 8192, 16384, 32768, 65536]
    elif gpu_memory > 30e9:  # 40GB GPU
        seq_lengths = [4096, 8192, 16384, 32768, 65536]
    else:  # Smaller GPU
        seq_lengths = [2048, 4096, 8192, 16384]

    print(f"\nConfiguration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Num heads: {NUM_HEADS}")
    print(f"  Head dim: {HEAD_DIM}")
    print(f"  Sequence lengths: {seq_lengths}")

    # Run benchmarks
    print("\n" + "-" * 70)
    print(f"{'Seq Len':>10} | {'PyTorch SDPA':>15} | {'Linear Attn':>15} | {'Speedup':>10}")
    print("-" * 70)

    results = []

    for seq_len in seq_lengths:
        # Create test tensors
        q = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM,
                        device='cuda', dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Benchmark PyTorch SDPA
        try:
            sdpa_mean, sdpa_std = benchmark(pytorch_sdpa_bidir, q, k, v)
            sdpa_str = f"{sdpa_mean:.2f}ms"
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                sdpa_mean = float('inf')
                sdpa_str = "OOM"
            else:
                raise

        # Benchmark Linear Attention
        try:
            linear_mean, linear_std = benchmark(linear_attention_bidir, q, k, v)
            linear_str = f"{linear_mean:.2f}ms"
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                linear_mean = float('inf')
                linear_str = "OOM"
            else:
                raise

        # Compute speedup
        if sdpa_mean != float('inf') and linear_mean != float('inf'):
            speedup = sdpa_mean / linear_mean
            speedup_str = f"{speedup:.1f}x"
        else:
            speedup = float('nan')
            speedup_str = "N/A"

        # Format sequence length
        if seq_len >= 1024:
            seq_str = f"{seq_len // 1024}K"
        else:
            seq_str = str(seq_len)

        print(f"{seq_str:>10} | {sdpa_str:>15} | {linear_str:>15} | {speedup_str:>10}")

        results.append({
            'seq_len': seq_len,
            'sdpa_ms': sdpa_mean,
            'linear_ms': linear_mean,
            'speedup': speedup
        })

        # Cleanup
        del q, k, v
        torch.cuda.empty_cache()

    print("-" * 70)

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    valid_results = [r for r in results if not np.isnan(r['speedup'])]
    if valid_results:
        max_speedup = max(r['speedup'] for r in valid_results)
        max_seq = max(r['seq_len'] for r in valid_results if r['speedup'] == max_speedup)
        crossover = next((r['seq_len'] for r in valid_results if r['speedup'] > 1), None)

        print(f"\nMax speedup: {max_speedup:.1f}x at sequence length {max_seq // 1024}K")
        if crossover:
            print(f"Crossover point: Linear attention becomes faster at ~{crossover // 1024}K tokens")

    print("""
Key observations:
- Linear attention achieves O(n) complexity vs O(n²) for standard attention
- The speedup grows with sequence length (as expected from complexity analysis)
- At very long sequences (64K+), linear attention is dramatically faster
- At short sequences, standard attention may be faster due to better optimization

Note: Actual results depend on your GPU. The 50x speedup was measured on A100 40GB.
""")

    # Save results
    results_dir = Path(__file__).parent.parent / "results" / "benchmarks"
    results_dir.mkdir(parents=True, exist_ok=True)

    import csv
    with open(results_dir / "speedup_benchmark.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['seq_len', 'sdpa_ms', 'linear_ms', 'speedup'])
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to: {results_dir / 'speedup_benchmark.csv'}")


if __name__ == "__main__":
    main()
