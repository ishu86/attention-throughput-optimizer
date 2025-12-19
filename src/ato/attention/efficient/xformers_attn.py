"""xFormers memory-efficient attention wrapper."""

from typing import Any, Optional

import torch
from torch import Tensor

try:
    from xformers.ops import memory_efficient_attention, LowerTriangularMask, AttentionBias

    _XFORMERS_AVAILABLE = True
except ImportError:
    _XFORMERS_AVAILABLE = False

from ato.attention.base import AttentionBase, AttentionConfig, AttentionOutput, AttentionType
from ato.attention.registry import AttentionRegistry


if _XFORMERS_AVAILABLE:

    @AttentionRegistry.register("xformers")
    @AttentionRegistry.register("memory_efficient")
    class XFormersAttention(AttentionBase):
        """xFormers memory-efficient attention wrapper.

        xFormers provides memory-efficient attention that:
            - Avoids materializing the full attention matrix
            - Supports various attention biases and masks
            - Provides good performance across different hardware
            - Falls back gracefully when optimal kernels unavailable

        Requirements:
            - xformers package installed
            - CUDA GPU recommended for best performance

        Reference:
            https://github.com/facebookresearch/xformers
        """

        attention_type = AttentionType.FLASH
        backend_name = "xformers"
        supports_kv_cache = False
        supports_variable_length = True
        min_compute_capability = (7, 0)  # Volta+

        def __init__(self, config: AttentionConfig) -> None:
            """Initialize xFormers attention wrapper.

            Args:
                config: Attention configuration.
            """
            super().__init__(config)

        def _validate_config(self) -> None:
            """Validate configuration for xFormers attention."""
            if self.config.embed_dim % self.config.num_heads != 0:
                raise ValueError(
                    f"embed_dim ({self.config.embed_dim}) must be divisible by "
                    f"num_heads ({self.config.num_heads})"
                )

        @classmethod
        def is_available(cls) -> bool:
            """Check if xFormers is available."""
            if not _XFORMERS_AVAILABLE:
                return False
            return torch.cuda.is_available()

        def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            attention_mask: Optional[Tensor] = None,
            kv_cache: Optional[tuple[Tensor, Tensor]] = None,
            **kwargs: Any,
        ) -> AttentionOutput:
            """Compute attention using xFormers memory-efficient attention.

            Args:
                query: Query tensor (batch, seq_len, num_heads, head_dim) or
                       (batch, seq_len, embed_dim).
                key: Key tensor with same format as query.
                value: Value tensor with same format as query.
                attention_mask: Optional attention bias/mask.
                kv_cache: Not supported, included for API compatibility.
                **kwargs: Additional arguments:
                    - attn_bias: xFormers AttentionBias object.
                    - scale: Custom softmax scale.

            Returns:
                AttentionOutput with attention result.
            """
            if kv_cache is not None:
                raise NotImplementedError(
                    "KV cache not supported in XFormersAttention. Use flash_v2 instead."
                )

            # Handle input shape
            needs_reshape = query.dim() == 3
            if needs_reshape:
                batch_size, seq_len, _ = query.shape
                query = self._reshape_for_xformers(query, batch_size, seq_len)
                key = self._reshape_for_xformers(key, batch_size, key.size(1))
                value = self._reshape_for_xformers(value, batch_size, value.size(1))
            else:
                batch_size, seq_len = query.shape[:2]

            # Determine attention bias
            attn_bias = kwargs.get("attn_bias")

            if attn_bias is None:
                if self.config.causal:
                    attn_bias = LowerTriangularMask()
                elif attention_mask is not None:
                    attn_bias = self._create_bias_from_mask(attention_mask)

            # Get scale
            scale = kwargs.get("scale", self.config.scale)

            # Compute attention
            # xFormers expects (batch, seq, heads, head_dim)
            output = memory_efficient_attention(
                query,
                key,
                value,
                attn_bias=attn_bias,
                scale=scale,
            )

            # Reshape back if needed
            if needs_reshape:
                output = self._reshape_from_xformers(output, batch_size, seq_len)

            return AttentionOutput(
                output=output,
                attention_weights=None,
                metadata={
                    "backend": self.backend_name,
                    "used_causal": self.config.causal,
                },
            )

        def _reshape_for_xformers(
            self, x: Tensor, batch_size: int, seq_len: int
        ) -> Tensor:
            """Reshape (batch, seq, embed) to (batch, seq, heads, head_dim)."""
            return x.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim)

        def _reshape_from_xformers(
            self, x: Tensor, batch_size: int, seq_len: int
        ) -> Tensor:
            """Reshape (batch, seq, heads, head_dim) to (batch, seq, embed)."""
            return x.reshape(batch_size, seq_len, self.config.embed_dim)

        def _create_bias_from_mask(self, mask: Tensor) -> Tensor:
            """Convert attention mask to xFormers bias format.

            Args:
                mask: Boolean or float mask.

            Returns:
                Attention bias tensor.
            """
            if mask.dtype == torch.bool:
                # Convert boolean mask to float bias
                bias = torch.zeros_like(mask, dtype=torch.float32)
                bias.masked_fill_(~mask, float("-inf"))
                return bias.to(mask.device)
            return mask

        def estimate_memory(
            self,
            batch_size: int,
            seq_len: int,
            dtype: Optional[torch.dtype] = None,
        ) -> dict[str, int]:
            """Estimate memory usage for xFormers attention.

            xFormers uses memory-efficient algorithms similar to FlashAttention.
            """
            dtype = dtype or self.config.dtype
            bytes_per_elem = torch.tensor([], dtype=dtype).element_size()

            embed_dim = self.config.embed_dim

            # QKV tensors
            qkv_size = 3 * batch_size * seq_len * embed_dim * bytes_per_elem

            # Output tensor
            output_size = batch_size * seq_len * embed_dim * bytes_per_elem

            # Intermediate memory (block-wise, similar to FlashAttention)
            block_size = 128
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
            }
