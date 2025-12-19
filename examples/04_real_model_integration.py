#!/usr/bin/env python3
"""
Real Model Integration - Using ATO Attention in a Transformer

This example shows how to integrate ATO attention implementations into
a real transformer model. This is the pattern you'd use to experiment
with different attention mechanisms in your own models.

You'll learn:
1. How to create a transformer block using ATO attention
2. How to swap attention implementations without changing model code
3. How to benchmark attention in a realistic setting

Usage:
    python examples/04_real_model_integration.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import time


class TransformerBlock(nn.Module):
    """
    Standard transformer block using ATO attention.

    This demonstrates how to integrate ATO attention into a real model.
    The attention implementation can be swapped by changing attention_type.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        attention_type: str = "standard",
        causal: bool = True,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention_type = attention_type
        self.causal = causal
        self.device = device
        self.dtype = dtype

        # Layer norms
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Attention - we'll use a simple implementation for this example
        # In a real scenario, you'd use ATO's AttentionRegistry
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _reshape_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, N, D) -> (B, H, N, head_dim)"""
        B, N, D = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _reshape_from_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, H, N, head_dim) -> (B, N, D)"""
        B, H, N, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, self.embed_dim)

    def standard_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Standard scaled dot-product attention."""
        return F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)

    def linear_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Linear attention with ELU+1 feature map."""
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        if not self.causal:
            kv = torch.einsum('bhnd,bhnv->bhdv', k, v)
            k_sum = k.sum(dim=2)
            out = torch.einsum('bhnd,bhdv->bhnv', q, kv)
            norm = torch.einsum('bhnd,bhd->bhn', q, k_sum).unsqueeze(-1)
            return out / (norm + eps)
        else:
            # Causal version with cumsum
            kv = torch.einsum('bhnd,bhnv->bhndv', k, v)
            kv_cumsum = torch.cumsum(kv, dim=2)
            k_cumsum = torch.cumsum(k, dim=2)
            out = torch.einsum('bhnd,bhndv->bhnv', q, kv_cumsum)
            norm = torch.einsum('bhnd,bhnd->bhn', q, k_cumsum).unsqueeze(-1)
            return out / (norm + eps)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention based on attention_type."""
        # Project to Q, K, V
        q = self._reshape_for_attention(self.q_proj(x))
        k = self._reshape_for_attention(self.k_proj(x))
        v = self._reshape_for_attention(self.v_proj(x))

        # Apply attention
        if self.attention_type == "standard":
            attn_out = self.standard_attention(q, k, v)
        elif self.attention_type == "linear":
            attn_out = self.linear_attention(q, k, v)
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")

        # Reshape and project
        out = self._reshape_from_attention(attn_out)
        return self.out_proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.dropout(self.attention(self.ln1(x)))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


class SimpleTransformer(nn.Module):
    """A simple transformer model for demonstration."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        max_seq_len: int,
        attention_type: str = "standard",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.attention_type = attention_type

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                attention_type=attention_type,
                causal=True,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        # Output head
        self.ln_out = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, N = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(N, device=device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Output
        x = self.ln_out(x)
        logits = self.lm_head(x)

        return logits


def benchmark_model(model: nn.Module, input_ids: torch.Tensor, warmup: int = 5, iters: int = 20) -> float:
    """Benchmark model forward pass."""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids)
            torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(iters):
            _ = model(input_ids)
            torch.cuda.synchronize()

    elapsed = (time.perf_counter() - start) / iters * 1000  # ms
    return elapsed


def main():
    # Check CUDA
    if not torch.cuda.is_available():
        print("This example requires CUDA.")
        return

    print("=" * 70)
    print("Transformer Model with Swappable Attention")
    print("=" * 70)
    print(f"\nGPU: {torch.cuda.get_device_name()}")

    # Model configuration
    VOCAB_SIZE = 32000
    EMBED_DIM = 512
    NUM_HEADS = 8
    NUM_LAYERS = 6
    FF_DIM = 2048
    MAX_SEQ_LEN = 8192

    device = 'cuda'
    dtype = torch.float16

    print(f"\nModel configuration:")
    print(f"  Vocab size: {VOCAB_SIZE}")
    print(f"  Embed dim: {EMBED_DIM}")
    print(f"  Num heads: {NUM_HEADS}")
    print(f"  Num layers: {NUM_LAYERS}")
    print(f"  FF dim: {FF_DIM}")
    print(f"  Max seq len: {MAX_SEQ_LEN}")

    # Create models with different attention types
    print("\n" + "-" * 70)
    print("Creating models...")

    model_standard = SimpleTransformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        max_seq_len=MAX_SEQ_LEN,
        attention_type="standard",
        device=device,
        dtype=dtype,
    ).to(device=device, dtype=dtype)

    model_linear = SimpleTransformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        max_seq_len=MAX_SEQ_LEN,
        attention_type="linear",
        device=device,
        dtype=dtype,
    ).to(device=device, dtype=dtype)

    # Count parameters
    num_params = sum(p.numel() for p in model_standard.parameters())
    print(f"Parameters per model: {num_params / 1e6:.1f}M")

    # Benchmark at different sequence lengths
    print("\n" + "-" * 70)
    print("Benchmarking forward pass...")

    seq_lengths = [512, 1024, 2048, 4096]
    batch_size = 4

    print(f"\n{'Seq Len':>10} | {'Standard':>15} | {'Linear':>15} | {'Speedup':>10}")
    print("-" * 60)

    for seq_len in seq_lengths:
        # Create random input
        input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device=device)

        try:
            time_standard = benchmark_model(model_standard, input_ids)
            std_str = f"{time_standard:.2f}ms"
        except RuntimeError:
            time_standard = float('inf')
            std_str = "OOM"

        try:
            time_linear = benchmark_model(model_linear, input_ids)
            lin_str = f"{time_linear:.2f}ms"
        except RuntimeError:
            time_linear = float('inf')
            lin_str = "OOM"

        if time_standard != float('inf') and time_linear != float('inf'):
            speedup = time_standard / time_linear
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"

        print(f"{seq_len:>10} | {std_str:>15} | {lin_str:>15} | {speedup_str:>10}")

        # Cleanup
        del input_ids
        torch.cuda.empty_cache()

    print("-" * 60)

    # Summary
    print("\n" + "=" * 70)
    print("Integration Patterns")
    print("=" * 70)

    print("""
Key patterns demonstrated:

1. **Swappable Attention**
   The TransformerBlock takes an `attention_type` parameter, making it easy
   to experiment with different attention mechanisms without changing code.

2. **Pre-norm Architecture**
   We use pre-layer-normalization (norm before attention/FF) which is more
   stable for training with different attention types.

3. **Projection Layers**
   Q, K, V projections are kept separate from the attention implementation,
   making it easy to swap just the attention computation.

4. **Memory Considerations**
   - Standard attention: O(n²) memory for attention scores
   - Linear attention: O(d²) memory for KV state

For production use:
- Use ATO's AttentionRegistry for more implementations
- Consider using torch.compile() for additional speedup
- Profile memory usage with torch.cuda.memory_stats()
- For training, implement custom backward pass for linear attention
""")


if __name__ == "__main__":
    main()
