"""Triton Flash Attention implementation.

A custom Flash Attention kernel written in Triton, demonstrating
block-wise attention computation for memory efficiency.

This implementation follows the FlashAttention algorithm:
1. Divide Q, K, V into blocks
2. Compute attention block by block
3. Use online softmax normalization to avoid materializing full attention matrix

Reference:
    "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
    (Dao et al., 2022)
"""

from typing import Any, Optional

import torch
import triton
import triton.language as tl
from torch import Tensor

from ato.attention.base import AttentionBase, AttentionConfig, AttentionOutput, AttentionType
from ato.attention.registry import AttentionRegistry


@triton.jit
def _flash_attention_forward_kernel(
    Q, K, V, O,
    L,  # logsumexp for backward pass
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Flash Attention forward kernel.

    Computes attention in blocks to minimize memory access.

    Args:
        Q, K, V: Input tensors
        O: Output tensor
        L: Logsumexp accumulator
        sm_scale: Softmax scale (1/sqrt(d))
        stride_*: Tensor strides
        Z: Batch size
        H: Number of heads
        N_CTX: Sequence length
        BLOCK_M, BLOCK_N, BLOCK_K: Block sizes
        IS_CAUSAL: Whether to apply causal masking
    """
    # Program ID
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_z = off_hz // H
    off_h = off_hz % H

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_z * stride_qz + off_h * stride_qh + \
             (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    k_ptrs = K + off_z * stride_kz + off_h * stride_kh + \
             (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
    v_ptrs = V + off_z * stride_vz + off_h * stride_vh + \
             (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)

    # Initialize output accumulator
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    # Load Q block
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # Compute attention block by block
    if IS_CAUSAL:
        lo = 0
        hi = (start_m + 1) * BLOCK_M
    else:
        lo = 0
        hi = N_CTX

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        curr_n = start_n + offs_n

        # Load K block
        k = tl.load(k_ptrs + start_n * stride_kn,
                   mask=curr_n[:, None] < N_CTX, other=0.0)

        # Compute QK^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale

        # Apply causal mask
        if IS_CAUSAL:
            mask = offs_m[:, None] >= curr_n[None, :]
            qk = tl.where(mask, qk, float("-inf"))

        # Online softmax: compute max and sum
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)

        # Compute correction factor
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)

        # Update logsumexp
        l_i = l_i * alpha + tl.sum(tl.exp(qk - m_ij[:, None]) * beta[:, None], axis=1)

        # Compute attention weights
        p = tl.exp(qk - m_i_new[:, None])

        # Load V block
        v = tl.load(v_ptrs + start_n * stride_vn,
                   mask=curr_n[:, None] < N_CTX, other=0.0)

        # Update accumulator
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_i_new

    # Normalize output
    acc = acc / l_i[:, None]

    # Store output
    o_ptrs = O + off_z * stride_oz + off_h * stride_oh + \
             (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N_CTX)

    # Store logsumexp for backward
    l_ptrs = L + off_z * H * N_CTX + off_h * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.log(l_i), mask=offs_m < N_CTX)


def flash_attention_triton(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    causal: bool = False,
    sm_scale: Optional[float] = None,
) -> Tensor:
    """Compute Flash Attention using Triton kernel.

    Args:
        q: Query tensor (batch, heads, seq_len, head_dim).
        k: Key tensor (batch, heads, seq_len, head_dim).
        v: Value tensor (batch, heads, seq_len, head_dim).
        causal: Whether to apply causal masking.
        sm_scale: Softmax scale. Defaults to 1/sqrt(head_dim).

    Returns:
        Output tensor (batch, heads, seq_len, head_dim).
    """
    # Ensure contiguous tensors
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    batch, heads, seq_len, head_dim = q.shape

    if sm_scale is None:
        sm_scale = head_dim ** -0.5

    # Allocate output
    o = torch.empty_like(q)
    l = torch.empty((batch, heads, seq_len), device=q.device, dtype=torch.float32)

    # Block sizes (tune for your GPU)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = head_dim

    # Grid
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads)

    # Launch kernel
    _flash_attention_forward_kernel[grid](
        q, k, v, o, l,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        batch, heads, seq_len,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        IS_CAUSAL=causal,
    )

    return o


@AttentionRegistry.register("triton_flash")
@AttentionRegistry.register("triton")
class TritonFlashAttention(AttentionBase):
    """Flash Attention using custom Triton kernel.

    This implementation provides a custom Triton-based Flash Attention
    that can be modified and tuned for specific use cases.

    Benefits:
        - Full control over the kernel implementation
        - Easy to modify for research purposes
        - Portable across different GPU architectures via Triton
    """

    attention_type = AttentionType.FLASH
    backend_name = "triton"
    supports_kv_cache = False
    supports_variable_length = False
    min_compute_capability = (8, 0)  # Ampere+

    def __init__(self, config: AttentionConfig) -> None:
        """Initialize Triton Flash Attention."""
        super().__init__(config)

    def _validate_config(self) -> None:
        """Validate configuration."""
        if self.config.embed_dim % self.config.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.config.embed_dim}) must be divisible by "
                f"num_heads ({self.config.num_heads})"
            )

        # Check head_dim is power of 2 for optimal Triton performance
        head_dim = self.config.head_dim
        if head_dim & (head_dim - 1) != 0:
            import warnings
            warnings.warn(
                f"head_dim={head_dim} is not a power of 2. "
                "Triton performance may be suboptimal."
            )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[tuple[Tensor, Tensor]] = None,
        **kwargs: Any,
    ) -> AttentionOutput:
        """Compute attention using Triton Flash Attention.

        Args:
            query: Query tensor (batch, seq_len, embed_dim) or
                   (batch, seq_len, num_heads, head_dim).
            key: Key tensor.
            value: Value tensor.
            attention_mask: Not supported (use causal=True in config).
            kv_cache: Not supported.
            **kwargs: Additional arguments.

        Returns:
            AttentionOutput with attention result.
        """
        if kv_cache is not None:
            raise NotImplementedError("KV cache not supported in Triton kernel")

        if attention_mask is not None:
            raise NotImplementedError(
                "Custom attention masks not supported. Use causal=True for causal masking."
            )

        # Handle input shape
        needs_reshape = query.dim() == 3
        if needs_reshape:
            batch_size, seq_len, _ = query.shape
            query = self._reshape_to_heads(query, batch_size, seq_len)
            key = self._reshape_to_heads(key, batch_size, key.size(1))
            value = self._reshape_to_heads(value, batch_size, value.size(1))
        else:
            batch_size, _, seq_len, _ = query.shape

        # Call Triton kernel
        output = flash_attention_triton(
            query,
            key,
            value,
            causal=self.config.causal,
            sm_scale=self.config.scale,
        )

        # Reshape back if needed
        if needs_reshape:
            output = self._reshape_from_heads(output, batch_size, seq_len)

        return AttentionOutput(
            output=output,
            attention_weights=None,
            metadata={
                "backend": self.backend_name,
                "kernel": "triton_flash_attention",
            },
        )

    def _reshape_to_heads(
        self, x: Tensor, batch_size: int, seq_len: int
    ) -> Tensor:
        """Reshape (batch, seq, embed) to (batch, heads, seq, head_dim)."""
        x = x.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim)
        return x.transpose(1, 2).contiguous()

    def _reshape_from_heads(
        self, x: Tensor, batch_size: int, seq_len: int
    ) -> Tensor:
        """Reshape (batch, heads, seq, head_dim) to (batch, seq, embed)."""
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.config.embed_dim)

    def estimate_memory(
        self,
        batch_size: int,
        seq_len: int,
        dtype: Optional[torch.dtype] = None,
    ) -> dict[str, int]:
        """Estimate memory for Triton Flash Attention."""
        dtype = dtype or self.config.dtype
        bytes_per_elem = torch.tensor([], dtype=dtype).element_size()

        embed_dim = self.config.embed_dim

        # QKV and output
        qkv_size = 3 * batch_size * seq_len * embed_dim * bytes_per_elem
        output_size = batch_size * seq_len * embed_dim * bytes_per_elem

        # Logsumexp buffer
        l_size = batch_size * self.config.num_heads * seq_len * 4  # float32

        return {
            "qkv_tensors": qkv_size,
            "output": output_size,
            "logsumexp_buffer": l_size,
            "total_estimate": qkv_size + output_size + l_size,
            "memory_complexity": "O(n)",
        }
