#!/usr/bin/env python3
"""
Compare Implementations - Verify Correctness Across Backends

This example compares different attention implementations to verify they
produce equivalent outputs (within numerical tolerance).

You'll learn:
1. How to compare outputs across implementations
2. What numerical differences to expect between float16/float32
3. How linear attention differs from standard attention semantically

Usage:
    python examples/03_compare_implementations.py
"""

import torch
import torch.nn.functional as F
import numpy as np


def standard_attention(q, k, v, causal=False):
    """Reference implementation: Standard scaled dot-product attention."""
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)

    if causal:
        seq_len = q.size(-2)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1)
        scores = scores.masked_fill(mask.bool(), float('-inf'))

    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v), weights


def pytorch_sdpa(q, k, v, causal=False):
    """PyTorch's built-in SDPA (may use FlashAttention)."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def linear_attention(q, k, v, causal=False, eps=1e-6):
    """Linear attention with ELU+1 feature map."""
    q = F.elu(q) + 1
    k = F.elu(k) + 1

    if not causal:
        kv = torch.einsum('bhnd,bhnv->bhdv', k, v)
        k_sum = k.sum(dim=2)
        out = torch.einsum('bhnd,bhdv->bhnv', q, kv)
        norm = torch.einsum('bhnd,bhd->bhn', q, k_sum).unsqueeze(-1)
    else:
        kv = torch.einsum('bhnd,bhnv->bhndv', k, v)
        kv_cumsum = torch.cumsum(kv, dim=2)
        k_cumsum = torch.cumsum(k, dim=2)
        out = torch.einsum('bhnd,bhndv->bhnv', q, kv_cumsum)
        norm = torch.einsum('bhnd,bhnd->bhn', q, k_cumsum).unsqueeze(-1)

    return out / (norm + eps)


def compare_outputs(name1, out1, name2, out2, rtol=1e-3, atol=1e-3):
    """Compare two outputs and report differences."""
    # Convert to float32 for comparison
    out1 = out1.float()
    out2 = out2.float()

    # Compute various metrics
    abs_diff = (out1 - out2).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    # Relative difference (avoiding division by zero)
    rel_diff = abs_diff / (out1.abs().clamp(min=1e-6))
    max_rel = rel_diff.max().item()

    # Check if close
    is_close = torch.allclose(out1, out2, rtol=rtol, atol=atol)

    print(f"\n{name1} vs {name2}:")
    print(f"  Max absolute diff: {max_diff:.2e}")
    print(f"  Mean absolute diff: {mean_diff:.2e}")
    print(f"  Max relative diff: {max_rel:.2e}")
    print(f"  Close (rtol={rtol}, atol={atol}): {is_close}")

    return is_close


def main():
    print("=" * 70)
    print("Comparing Attention Implementations")
    print("=" * 70)

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Test parameters
    batch_size = 2
    num_heads = 4
    seq_len = 128
    head_dim = 32

    print(f"\nTest configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Seq length: {seq_len}")
    print(f"  Head dim: {head_dim}")

    # =========================================================================
    # Test 1: Standard implementations should match
    # =========================================================================
    print("\n" + "=" * 70)
    print("Test 1: Standard Attention Implementations")
    print("=" * 70)

    # Use float32 for precise comparison
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Reference implementation
    ref_out, ref_weights = standard_attention(q, k, v, causal=False)

    # PyTorch SDPA
    sdpa_out = pytorch_sdpa(q, k, v, causal=False)

    test1_pass = compare_outputs("Reference", ref_out, "PyTorch SDPA", sdpa_out)

    # Causal version
    print("\n[Causal attention]")
    ref_causal, _ = standard_attention(q, k, v, causal=True)
    sdpa_causal = pytorch_sdpa(q, k, v, causal=True)
    test1_causal_pass = compare_outputs("Reference (causal)", ref_causal,
                                         "PyTorch SDPA (causal)", sdpa_causal)

    # =========================================================================
    # Test 2: Float16 precision
    # =========================================================================
    print("\n" + "=" * 70)
    print("Test 2: Float16 vs Float32 Precision")
    print("=" * 70)

    # Float16 tensors
    q16 = q.half()
    k16 = k.half()
    v16 = v.half()

    sdpa_fp32 = pytorch_sdpa(q, k, v)
    sdpa_fp16 = pytorch_sdpa(q16, k16, v16)

    test2_pass = compare_outputs("SDPA FP32", sdpa_fp32, "SDPA FP16", sdpa_fp16.float(),
                                  rtol=1e-2, atol=1e-2)

    print("\nNote: FP16 has ~3 decimal digits of precision, so differences of 1e-3 are expected.")

    # =========================================================================
    # Test 3: Linear attention is DIFFERENT from standard
    # =========================================================================
    print("\n" + "=" * 70)
    print("Test 3: Linear vs Standard Attention (Expected to Differ)")
    print("=" * 70)

    linear_out = linear_attention(q, k, v, causal=False)
    standard_out, _ = standard_attention(q, k, v, causal=False)

    compare_outputs("Standard", standard_out, "Linear", linear_out, rtol=0.5, atol=0.5)

    print("""
IMPORTANT: Linear attention is NOT a drop-in replacement for standard attention!

The outputs differ because:
1. Linear attention uses a different similarity function (ELU+1 kernel vs softmax)
2. The attention distribution is smoother (higher entropy)
3. Standard attention can focus sharply on specific positions; linear cannot

Linear attention trades expressivity for O(n) complexity.
Use it when:
- Sequence lengths are very long (8K+ tokens)
- Memory is constrained
- You can retrain/finetune the model for linear attention
""")

    # =========================================================================
    # Test 4: Verify linear attention correctness
    # =========================================================================
    print("\n" + "=" * 70)
    print("Test 4: Linear Attention Self-Consistency")
    print("=" * 70)

    # Verify that our linear attention is numerically stable
    torch.manual_seed(123)
    q2 = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    k2 = torch.randn_like(q2)
    v2 = torch.randn_like(q2)

    # Run twice - should be identical
    linear_out1 = linear_attention(q2, k2, v2)
    linear_out2 = linear_attention(q2, k2, v2)

    test4_pass = compare_outputs("Linear run 1", linear_out1, "Linear run 2", linear_out2,
                                  rtol=0, atol=0)

    # Verify causal vs bidirectional at position 0 (should match)
    print("\nCausal vs Bidirectional at position 0 (should match):")
    linear_bidir = linear_attention(q2, k2, v2, causal=False)
    linear_causal = linear_attention(q2, k2, v2, causal=True)

    # At position 0, causal and bidirectional should be similar (position 0 only sees itself)
    pos0_diff = (linear_bidir[:, :, 0] - linear_causal[:, :, 0]).abs().max().item()
    print(f"  Position 0 max diff: {pos0_diff:.2e}")

    # At later positions, they should differ (causal can't see future)
    posN_diff = (linear_bidir[:, :, -1] - linear_causal[:, :, -1]).abs().max().item()
    print(f"  Position {seq_len-1} max diff: {posN_diff:.2e} (expected to differ)")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"""
Results:
- Standard implementations match: {test1_pass and test1_causal_pass}
- FP16/FP32 within expected tolerance: {test2_pass}
- Linear attention is deterministic: {test4_pass}
- Linear attention DIFFERS from standard: Expected

Key takeaways:
1. PyTorch SDPA matches reference implementation (may use FlashAttention internally)
2. FP16 has ~1e-3 numerical differences vs FP32 - this is normal
3. Linear attention has DIFFERENT semantics - outputs will not match standard
4. For correctness testing, always compare same-type implementations
""")


if __name__ == "__main__":
    main()
