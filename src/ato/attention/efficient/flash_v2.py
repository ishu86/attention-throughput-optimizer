"""FlashAttention-2 wrapper for memory-efficient attention."""

from typing import Any, Optional

import torch
from torch import Tensor

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.flash_attn_interface import flash_attn_with_kvcache

    _FLASH_AVAILABLE = True
except ImportError:
    _FLASH_AVAILABLE = False

from ato.attention.base import AttentionBase, AttentionConfig, AttentionOutput, AttentionType
from ato.attention.registry import AttentionRegistry


if _FLASH_AVAILABLE:

    @AttentionRegistry.register("flash_v2")
    @AttentionRegistry.register("flash")
    class FlashAttentionV2(AttentionBase):
        """FlashAttention-2 wrapper for efficient attention computation.

        FlashAttention-2 provides:
            - IO-aware attention that minimizes memory reads/writes
            - O(N) memory complexity instead of O(N^2)
            - 2-4x speedup over standard attention
            - Support for causal masking, dropout, and variable-length sequences

        Requirements:
            - NVIDIA GPU with compute capability >= 8.0 (Ampere+)
            - flash-attn package installed

        Reference:
            "FlashAttention-2: Faster Attention with Better Parallelism and Work
            Partitioning" (Dao, 2023)
        """

        attention_type = AttentionType.FLASH
        backend_name = "flash_v2"
        supports_kv_cache = True
        supports_variable_length = True
        min_compute_capability = (8, 0)  # Ampere+

        def __init__(self, config: AttentionConfig) -> None:
            """Initialize FlashAttention-2 wrapper.

            Args:
                config: Attention configuration.
            """
            super().__init__(config)

        def _validate_config(self) -> None:
            """Validate configuration for FlashAttention-2."""
            if self.config.embed_dim % self.config.num_heads != 0:
                raise ValueError(
                    f"embed_dim ({self.config.embed_dim}) must be divisible by "
                    f"num_heads ({self.config.num_heads})"
                )

            # FlashAttention requires head_dim to be <= 256
            if self.config.head_dim > 256:
                raise ValueError(
                    f"FlashAttention-2 requires head_dim <= 256, got {self.config.head_dim}"
                )

            # Check dtype support
            if self.config.dtype not in (torch.float16, torch.bfloat16):
                raise ValueError(
                    f"FlashAttention-2 requires float16 or bfloat16, got {self.config.dtype}"
                )

        @classmethod
        def is_available(cls) -> bool:
            """Check if FlashAttention-2 is available."""
            if not _FLASH_AVAILABLE:
                return False
            if not torch.cuda.is_available():
                return False
            capability = torch.cuda.get_device_capability()
            return capability >= cls.min_compute_capability

        def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            attention_mask: Optional[Tensor] = None,
            kv_cache: Optional[tuple[Tensor, Tensor]] = None,
            **kwargs: Any,
        ) -> AttentionOutput:
            """Compute attention using FlashAttention-2.

            Args:
                query: Query tensor (batch, seq_len, num_heads, head_dim) or
                       (batch, seq_len, embed_dim).
                key: Key tensor with same format as query.
                value: Value tensor with same format as query.
                attention_mask: Not directly supported; use causal=True in config
                                for causal masking.
                kv_cache: Optional tuple of (cached_keys, cached_values).
                **kwargs: Additional arguments:
                    - cu_seqlens_q: Cumulative sequence lengths for queries (variable-length).
                    - cu_seqlens_k: Cumulative sequence lengths for keys (variable-length).
                    - max_seqlen_q: Max query sequence length (variable-length).
                    - max_seqlen_k: Max key sequence length (variable-length).
                    - softmax_scale: Custom softmax scale.
                    - window_size: Sliding window size (left, right).
                    - return_kv_cache: If True, return updated KV cache.

            Returns:
                AttentionOutput with attention result.
            """
            # Handle input shape
            needs_reshape = query.dim() == 3
            if needs_reshape:
                batch_size, seq_len, _ = query.shape
                query = self._reshape_to_flash(query, batch_size, seq_len)
                key = self._reshape_to_flash(key, batch_size, key.size(1))
                value = self._reshape_to_flash(value, batch_size, value.size(1))
            else:
                batch_size, seq_len = query.shape[:2]

            # Get optional parameters
            softmax_scale = kwargs.get("softmax_scale", self.config.scale)
            window_size = kwargs.get("window_size", self.config.window_size) or (-1, -1)
            dropout_p = self.config.dropout if self.training else 0.0

            # Variable-length support
            cu_seqlens_q = kwargs.get("cu_seqlens_q")
            cu_seqlens_k = kwargs.get("cu_seqlens_k")

            metadata = {"backend": self.backend_name}

            if cu_seqlens_q is not None and cu_seqlens_k is not None:
                # Variable-length attention
                max_seqlen_q = kwargs.get("max_seqlen_q", seq_len)
                max_seqlen_k = kwargs.get("max_seqlen_k", key.size(1))

                output = flash_attn_varlen_func(
                    query.squeeze(0),  # (total_q, num_heads, head_dim)
                    key.squeeze(0),
                    value.squeeze(0),
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    causal=self.config.causal,
                    window_size=window_size,
                )
                output = output.unsqueeze(0)
                metadata["variable_length"] = True

            elif kv_cache is not None:
                # KV cache mode
                cached_k, cached_v = kv_cache
                output = flash_attn_with_kvcache(
                    query,
                    cached_k,
                    cached_v,
                    k=key,
                    v=value,
                    softmax_scale=softmax_scale,
                    causal=self.config.causal,
                )
                if kwargs.get("return_kv_cache", False):
                    # Update cache
                    new_k = torch.cat([cached_k, key], dim=1)
                    new_v = torch.cat([cached_v, value], dim=1)
                    metadata["kv_cache"] = (new_k, new_v)

            else:
                # Standard attention
                output = flash_attn_func(
                    query,
                    key,
                    value,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    causal=self.config.causal,
                    window_size=window_size,
                )

            # Reshape back if needed
            if needs_reshape:
                output = self._reshape_from_flash(output, batch_size, seq_len)

            return AttentionOutput(
                output=output,
                attention_weights=None,  # FlashAttention doesn't return weights
                metadata=metadata,
            )

        def _reshape_to_flash(
            self, x: Tensor, batch_size: int, seq_len: int
        ) -> Tensor:
            """Reshape (batch, seq, embed) to (batch, seq, heads, head_dim)."""
            return x.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim)

        def _reshape_from_flash(
            self, x: Tensor, batch_size: int, seq_len: int
        ) -> Tensor:
            """Reshape (batch, seq, heads, head_dim) to (batch, seq, embed)."""
            return x.reshape(batch_size, seq_len, self.config.embed_dim)

        def estimate_memory(
            self,
            batch_size: int,
            seq_len: int,
            dtype: Optional[torch.dtype] = None,
        ) -> dict[str, int]:
            """Estimate memory usage for FlashAttention-2.

            FlashAttention uses O(N) memory instead of O(N^2) by avoiding
            materialization of the full attention matrix.
            """
            dtype = dtype or self.config.dtype
            bytes_per_elem = torch.tensor([], dtype=dtype).element_size()

            embed_dim = self.config.embed_dim

            # QKV tensors
            qkv_size = 3 * batch_size * seq_len * embed_dim * bytes_per_elem

            # Output tensor
            output_size = batch_size * seq_len * embed_dim * bytes_per_elem

            # FlashAttention uses block-wise computation, not full attention matrix
            # Approximate intermediate memory (much smaller than O(N^2))
            block_size = 128  # Typical block size
            num_blocks = (seq_len + block_size - 1) // block_size
            intermediate = (
                batch_size
                * self.config.num_heads
                * num_blocks
                * block_size
                * block_size
                * bytes_per_elem
            )

            return {
                "qkv_tensors": qkv_size,
                "output": output_size,
                "intermediate_blocks": intermediate,
                "total_estimate": qkv_size + output_size + intermediate,
                "memory_complexity": "O(N)",
                "vs_standard": f"{seq_len / block_size:.1f}x memory reduction",
            }
