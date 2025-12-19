"""Tensor manipulation utilities for attention operations."""

from typing import Optional

import torch
from torch import Tensor


def reshape_for_attention(
    x: Tensor,
    num_heads: int,
    head_dim: Optional[int] = None,
) -> Tensor:
    """Reshape tensor from (batch, seq, embed) to (batch, heads, seq, head_dim).

    Args:
        x: Input tensor of shape (batch, seq_len, embed_dim).
        num_heads: Number of attention heads.
        head_dim: Dimension per head. Computed from embed_dim if not provided.

    Returns:
        Reshaped tensor of shape (batch, num_heads, seq_len, head_dim).
    """
    batch_size, seq_len, embed_dim = x.shape

    if head_dim is None:
        head_dim = embed_dim // num_heads

    x = x.view(batch_size, seq_len, num_heads, head_dim)
    return x.transpose(1, 2)


def reshape_from_attention(
    x: Tensor,
    embed_dim: Optional[int] = None,
) -> Tensor:
    """Reshape tensor from (batch, heads, seq, head_dim) to (batch, seq, embed).

    Args:
        x: Input tensor of shape (batch, num_heads, seq_len, head_dim).
        embed_dim: Output embedding dimension. Computed from shape if not provided.

    Returns:
        Reshaped tensor of shape (batch, seq_len, embed_dim).
    """
    batch_size, num_heads, seq_len, head_dim = x.shape

    if embed_dim is None:
        embed_dim = num_heads * head_dim

    x = x.transpose(1, 2).contiguous()
    return x.view(batch_size, seq_len, embed_dim)


def create_causal_mask(
    seq_len: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Create a causal (lower triangular) attention mask.

    Args:
        seq_len: Sequence length.
        device: Target device.
        dtype: Output dtype. Defaults to float32.

    Returns:
        Causal mask of shape (seq_len, seq_len) with -inf for masked positions.
    """
    dtype = dtype or torch.float32

    # Create upper triangular mask
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=dtype),
        diagonal=1,
    )

    # Convert to attention mask format
    mask = mask.masked_fill(mask == 1, float("-inf"))

    return mask


def create_causal_mask_bool(
    seq_len: int,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Create a boolean causal mask.

    Args:
        seq_len: Sequence length.
        device: Target device.

    Returns:
        Boolean mask of shape (seq_len, seq_len). True = attend, False = mask.
    """
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))


def create_padding_mask(
    lengths: Tensor,
    max_len: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Create a padding mask from sequence lengths.

    Args:
        lengths: Tensor of sequence lengths (batch_size,).
        max_len: Maximum sequence length. Inferred from lengths if not provided.
        device: Target device.

    Returns:
        Boolean mask of shape (batch_size, max_len). True = valid, False = padding.
    """
    if max_len is None:
        max_len = int(lengths.max().item())

    batch_size = lengths.shape[0]
    device = device or lengths.device

    # Create position indices
    positions = torch.arange(max_len, device=device).unsqueeze(0)

    # Create mask where positions < length
    mask = positions < lengths.unsqueeze(1)

    return mask


def create_attention_mask(
    padding_mask: Optional[Tensor] = None,
    causal: bool = False,
    seq_len: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Optional[Tensor]:
    """Create combined attention mask from padding and/or causal mask.

    Args:
        padding_mask: Boolean padding mask (batch, seq_len).
        causal: Whether to apply causal masking.
        seq_len: Sequence length (required if causal=True and no padding_mask).
        device: Target device.
        dtype: Output dtype.

    Returns:
        Combined attention mask, or None if no masking needed.
    """
    dtype = dtype or torch.float32

    if padding_mask is None and not causal:
        return None

    if seq_len is None:
        if padding_mask is not None:
            seq_len = padding_mask.shape[1]
        else:
            raise ValueError("seq_len required when causal=True and no padding_mask")

    device = device or (padding_mask.device if padding_mask is not None else None)

    # Start with zeros
    batch_size = padding_mask.shape[0] if padding_mask is not None else 1
    mask = torch.zeros(batch_size, 1, seq_len, seq_len, device=device, dtype=dtype)

    # Add causal mask
    if causal:
        causal_mask = create_causal_mask(seq_len, device=device, dtype=dtype)
        mask = mask + causal_mask.unsqueeze(0).unsqueeze(0)

    # Add padding mask
    if padding_mask is not None:
        # Expand padding mask: (batch, seq) -> (batch, 1, 1, seq)
        padding_attn = padding_mask.float()
        padding_attn = padding_attn.masked_fill(padding_attn == 0, float("-inf"))
        padding_attn = padding_attn.masked_fill(padding_attn == 1, 0)
        mask = mask + padding_attn.unsqueeze(1).unsqueeze(2)

    return mask


def repeat_kv(
    hidden_states: Tensor,
    n_rep: int,
) -> Tensor:
    """Repeat key/value heads to match the number of query heads.

    Used in grouped-query attention (GQA) to expand KV heads.

    Args:
        hidden_states: Tensor of shape (batch, num_kv_heads, seq_len, head_dim).
        n_rep: Number of times to repeat each KV head.

    Returns:
        Tensor of shape (batch, num_kv_heads * n_rep, seq_len, head_dim).
    """
    if n_rep == 1:
        return hidden_states

    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape

    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )

    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def split_heads(
    x: Tensor,
    num_heads: int,
) -> Tensor:
    """Split tensor into multiple heads.

    Args:
        x: Tensor of shape (batch, seq_len, num_heads * head_dim).
        num_heads: Number of heads to split into.

    Returns:
        Tensor of shape (batch, num_heads, seq_len, head_dim).
    """
    batch_size, seq_len, dim = x.shape
    head_dim = dim // num_heads

    x = x.view(batch_size, seq_len, num_heads, head_dim)
    return x.transpose(1, 2)


def merge_heads(x: Tensor) -> Tensor:
    """Merge attention heads back together.

    Args:
        x: Tensor of shape (batch, num_heads, seq_len, head_dim).

    Returns:
        Tensor of shape (batch, seq_len, num_heads * head_dim).
    """
    batch_size, num_heads, seq_len, head_dim = x.shape

    x = x.transpose(1, 2).contiguous()
    return x.view(batch_size, seq_len, num_heads * head_dim)


def generate_random_attention_inputs(
    batch_size: int,
    seq_len: int,
    embed_dim: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate random Q, K, V tensors for testing.

    Args:
        batch_size: Batch size.
        seq_len: Sequence length.
        embed_dim: Embedding dimension.
        device: Target device.
        dtype: Data type.

    Returns:
        Tuple of (query, key, value) tensors.
    """
    dtype = dtype or torch.float32
    shape = (batch_size, seq_len, embed_dim)

    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)

    return q, k, v
