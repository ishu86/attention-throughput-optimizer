"""Scaled dot-product attention using PyTorch's native implementation."""

from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ato.attention.base import AttentionBase, AttentionConfig, AttentionOutput, AttentionType
from ato.attention.registry import AttentionRegistry


@AttentionRegistry.register("standard")
@AttentionRegistry.register("sdpa")
class ScaledDotProductAttention(AttentionBase):
    """Scaled dot-product attention using PyTorch's F.scaled_dot_product_attention.

    This implementation uses PyTorch's native SDPA which automatically selects
    the best backend (FlashAttention, Memory-Efficient, or Math) based on
    input characteristics and hardware.

    Supports:
        - Causal masking
        - Custom attention masks
        - Dropout during training
        - Multiple attention backends via PyTorch

    Note:
        This implementation expects pre-projected Q, K, V tensors.
        For full multi-head attention with projections, use MultiHeadAttention.
    """

    attention_type = AttentionType.STANDARD
    backend_name = "pytorch"
    supports_kv_cache = False
    supports_variable_length = True
    min_compute_capability = (7, 0)

    def __init__(self, config: AttentionConfig) -> None:
        """Initialize scaled dot-product attention.

        Args:
            config: Attention configuration.
        """
        super().__init__(config)

    def _validate_config(self) -> None:
        """Validate configuration."""
        if self.config.embed_dim % self.config.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.config.embed_dim}) must be divisible by "
                f"num_heads ({self.config.num_heads})"
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
        """Compute scaled dot-product attention.

        Args:
            query: Query tensor (batch, seq_len, embed_dim) or
                   (batch, num_heads, seq_len, head_dim).
            key: Key tensor with same format as query.
            value: Value tensor with same format as query.
            attention_mask: Optional attention mask. Can be:
                - Boolean mask (batch, seq_len) or (batch, 1, seq_len, seq_len)
                - Float mask with -inf for masked positions
            kv_cache: Not supported, included for API compatibility.
            **kwargs: Additional arguments (return_weights, etc.).

        Returns:
            AttentionOutput with attention result.
        """
        if kv_cache is not None:
            raise NotImplementedError(
                "KV cache not supported in ScaledDotProductAttention. "
                "Use an efficient backend like flash_v2."
            )

        # Handle input shape - convert to (batch, heads, seq, head_dim)
        needs_reshape = query.dim() == 3
        if needs_reshape:
            batch_size, seq_len, embed_dim = query.shape
            query = self._reshape_for_attention(query)
            key = self._reshape_for_attention(key)
            value = self._reshape_for_attention(value)
        else:
            batch_size, _, seq_len, _ = query.shape

        # Process attention mask
        attn_mask = self._process_mask(attention_mask, query)

        # Compute attention using PyTorch's SDPA
        dropout_p = self.config.dropout if self.training else 0.0

        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=self.config.causal and attn_mask is None,
            scale=self.config.scale,
        )

        # Reshape back if needed
        if needs_reshape:
            output = self._reshape_from_attention(output)

        return AttentionOutput(
            output=output,
            attention_weights=None,  # SDPA doesn't return weights by default
            metadata={
                "backend": self.backend_name,
                "used_causal": self.config.causal,
            },
        )

    def _reshape_for_attention(self, x: Tensor) -> Tensor:
        """Reshape (batch, seq, embed) to (batch, heads, seq, head_dim)."""
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim)
        return x.transpose(1, 2)

    def _reshape_from_attention(self, x: Tensor) -> Tensor:
        """Reshape (batch, heads, seq, head_dim) to (batch, seq, embed)."""
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2)
        return x.reshape(batch_size, seq_len, self.config.embed_dim)

    def _process_mask(
        self, mask: Optional[Tensor], query: Tensor
    ) -> Optional[Tensor]:
        """Process attention mask to correct format.

        Args:
            mask: Input mask (various formats supported).
            query: Query tensor for shape reference.

        Returns:
            Processed mask in (batch, 1, seq, seq) float format, or None.
        """
        if mask is None:
            return None

        # Boolean mask to float
        if mask.dtype == torch.bool:
            mask = mask.float().masked_fill(mask == 0, float("-inf"))

        # Expand dimensions if needed
        if mask.dim() == 2:
            # (batch, seq) -> (batch, 1, 1, seq)
            mask = mask.unsqueeze(1).unsqueeze(2)
        elif mask.dim() == 3:
            # (batch, seq, seq) -> (batch, 1, seq, seq)
            mask = mask.unsqueeze(1)

        return mask

    def estimate_memory(
        self,
        batch_size: int,
        seq_len: int,
        dtype: Optional[torch.dtype] = None,
    ) -> dict[str, int]:
        """Estimate memory usage.

        PyTorch's SDPA may use memory-efficient algorithms that avoid
        materializing the full attention matrix.
        """
        base = super().estimate_memory(batch_size, seq_len, dtype)

        # Note: Actual memory may be lower if using efficient backend
        base["note"] = "Actual memory may be lower with efficient SDPA backend"

        return base
