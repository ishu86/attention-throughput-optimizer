"""Multi-head attention with learnable projections."""

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ato.attention.base import AttentionBase, AttentionConfig, AttentionOutput, AttentionType
from ato.attention.registry import AttentionRegistry


@AttentionRegistry.register("multi_head")
@AttentionRegistry.register("mha")
class MultiHeadAttention(AttentionBase):
    """Multi-head attention with learnable Q, K, V, and output projections.

    This is a complete multi-head attention implementation that includes:
        - Learnable linear projections for Q, K, V
        - Multi-head attention computation
        - Output projection

    The implementation uses PyTorch's scaled_dot_product_attention internally
    for efficient computation.
    """

    attention_type = AttentionType.MULTI_HEAD
    backend_name = "pytorch"
    supports_kv_cache = True
    supports_variable_length = True
    min_compute_capability = (7, 0)

    def __init__(self, config: AttentionConfig) -> None:
        """Initialize multi-head attention.

        Args:
            config: Attention configuration.
        """
        super().__init__(config)

        embed_dim = config.embed_dim
        num_heads = config.num_heads
        head_dim = config.head_dim

        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(num_heads * head_dim, embed_dim, bias=False)

        # Initialize weights
        self._init_weights()

    def _validate_config(self) -> None:
        """Validate configuration."""
        if self.config.embed_dim % self.config.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.config.embed_dim}) must be divisible by "
                f"num_heads ({self.config.num_heads})"
            )

    def _init_weights(self) -> None:
        """Initialize projection weights."""
        # Xavier uniform initialization
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[tuple[Tensor, Tensor]] = None,
        **kwargs: Any,
    ) -> AttentionOutput:
        """Compute multi-head attention.

        Args:
            query: Query tensor (batch, seq_len, embed_dim).
            key: Key tensor (batch, seq_len, embed_dim).
            value: Value tensor (batch, seq_len, embed_dim).
            attention_mask: Optional attention mask.
            kv_cache: Optional tuple of (cached_keys, cached_values) with shape
                      (batch, num_heads, cache_len, head_dim).
            **kwargs: Additional arguments.
                - return_kv_cache: If True, return updated KV cache in metadata.

        Returns:
            AttentionOutput with attention result and optionally KV cache.
        """
        batch_size, q_len, _ = query.shape

        # Project queries, keys, values
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        q = self._reshape_for_attention(q, batch_size, q_len)
        k = self._reshape_for_attention(k, batch_size, key.size(1))
        v = self._reshape_for_attention(v, batch_size, value.size(1))

        # Handle KV cache
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        # Process attention mask
        attn_mask = self._process_mask(attention_mask, q, k)

        # Compute attention
        dropout_p = self.config.dropout if self.training else 0.0

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=self.config.causal and attn_mask is None,
            scale=self.config.scale,
        )

        # Reshape back to (batch, seq_len, embed_dim)
        attn_output = self._reshape_from_attention(attn_output, batch_size, q_len)

        # Output projection
        output = self.out_proj(attn_output)

        # Build metadata
        metadata = {"backend": self.backend_name}

        if kwargs.get("return_kv_cache", False):
            metadata["kv_cache"] = (k, v)

        return AttentionOutput(
            output=output,
            attention_weights=None,
            metadata=metadata,
        )

    def _reshape_for_attention(
        self, x: Tensor, batch_size: int, seq_len: int
    ) -> Tensor:
        """Reshape (batch, seq, num_heads * head_dim) to (batch, heads, seq, head_dim)."""
        x = x.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim)
        return x.transpose(1, 2)

    def _reshape_from_attention(
        self, x: Tensor, batch_size: int, seq_len: int
    ) -> Tensor:
        """Reshape (batch, heads, seq, head_dim) to (batch, seq, embed_dim)."""
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.config.num_heads * self.config.head_dim)

    def _process_mask(
        self, mask: Optional[Tensor], query: Tensor, key: Tensor
    ) -> Optional[Tensor]:
        """Process attention mask to correct format."""
        if mask is None:
            return None

        # Boolean mask to float
        if mask.dtype == torch.bool:
            mask = mask.float().masked_fill(~mask, float("-inf")).masked_fill(mask, 0.0)

        # Expand dimensions if needed
        if mask.dim() == 2:
            # (batch, key_len) -> (batch, 1, 1, key_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
        elif mask.dim() == 3:
            # (batch, query_len, key_len) -> (batch, 1, query_len, key_len)
            mask = mask.unsqueeze(1)

        return mask

    def estimate_memory(
        self,
        batch_size: int,
        seq_len: int,
        dtype: Optional[torch.dtype] = None,
    ) -> dict[str, int]:
        """Estimate memory usage including projection weights."""
        base = super().estimate_memory(batch_size, seq_len, dtype)

        # Add projection weight memory
        embed_dim = self.config.embed_dim
        num_heads = self.config.num_heads
        head_dim = self.config.head_dim

        weight_dtype = next(self.parameters()).dtype
        bytes_per_weight = torch.tensor([], dtype=weight_dtype).element_size()

        # Q, K, V projections
        qkv_weights = 3 * embed_dim * (num_heads * head_dim) * bytes_per_weight
        # Output projection
        out_weights = (num_heads * head_dim) * embed_dim * bytes_per_weight

        base["projection_weights"] = qkv_weights + out_weights
        base["total_estimate"] += base["projection_weights"]

        return base
