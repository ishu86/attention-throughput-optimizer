"""Grouped-query attention (GQA) implementation."""

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ato.attention.base import AttentionBase, AttentionConfig, AttentionOutput, AttentionType
from ato.attention.registry import AttentionRegistry


@AttentionRegistry.register("grouped_query")
@AttentionRegistry.register("gqa")
class GroupedQueryAttention(AttentionBase):
    """Grouped-query attention (GQA) with configurable KV heads.

    GQA uses fewer key-value heads than query heads, reducing memory usage
    and computation for KV while maintaining query expressiveness.

    When num_kv_heads == num_heads, this is standard MHA.
    When num_kv_heads == 1, this is multi-query attention (MQA).

    Reference:
        "GQA: Training Generalized Multi-Query Transformer Models from
        Multi-Head Checkpoints" (Ainslie et al., 2023)
    """

    attention_type = AttentionType.GROUPED_QUERY
    backend_name = "pytorch"
    supports_kv_cache = True
    supports_variable_length = True
    min_compute_capability = (7, 0)

    def __init__(self, config: AttentionConfig) -> None:
        """Initialize grouped-query attention.

        Args:
            config: Attention configuration. Must specify num_kv_heads.
        """
        super().__init__(config)

        embed_dim = config.embed_dim
        num_heads = config.num_heads
        num_kv_heads = config.num_kv_heads
        head_dim = config.head_dim

        # Query projection (full num_heads)
        self.q_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)

        # Key/Value projections (reduced num_kv_heads)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(num_heads * head_dim, embed_dim, bias=False)

        # Compute repeat factor for KV heads
        self.num_key_value_groups = num_heads // num_kv_heads

        # Initialize weights
        self._init_weights()

    def _validate_config(self) -> None:
        """Validate configuration."""
        if self.config.embed_dim % self.config.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.config.embed_dim}) must be divisible by "
                f"num_heads ({self.config.num_heads})"
            )

        if self.config.num_heads % self.config.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.config.num_heads}) must be divisible by "
                f"num_kv_heads ({self.config.num_kv_heads})"
            )

    def _init_weights(self) -> None:
        """Initialize projection weights."""
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
        """Compute grouped-query attention.

        Args:
            query: Query tensor (batch, seq_len, embed_dim).
            key: Key tensor (batch, seq_len, embed_dim).
            value: Value tensor (batch, seq_len, embed_dim).
            attention_mask: Optional attention mask.
            kv_cache: Optional tuple of (cached_keys, cached_values) with shape
                      (batch, num_kv_heads, cache_len, head_dim).
            **kwargs: Additional arguments.
                - return_kv_cache: If True, return updated KV cache in metadata.

        Returns:
            AttentionOutput with attention result.
        """
        batch_size, q_len, _ = query.shape
        kv_len = key.size(1)

        # Project queries (full num_heads)
        q = self.q_proj(query)
        q = q.view(batch_size, q_len, self.config.num_heads, self.config.head_dim)
        q = q.transpose(1, 2)  # (batch, num_heads, q_len, head_dim)

        # Project keys and values (reduced num_kv_heads)
        k = self.k_proj(key)
        v = self.v_proj(value)

        k = k.view(batch_size, kv_len, self.config.num_kv_heads, self.config.head_dim)
        v = v.view(batch_size, kv_len, self.config.num_kv_heads, self.config.head_dim)

        k = k.transpose(1, 2)  # (batch, num_kv_heads, kv_len, head_dim)
        v = v.transpose(1, 2)

        # Handle KV cache
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        # Expand KV heads to match query heads
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

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
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(
            batch_size, q_len, self.config.num_heads * self.config.head_dim
        )

        # Output projection
        output = self.out_proj(attn_output)

        # Build metadata
        metadata = {
            "backend": self.backend_name,
            "num_kv_heads": self.config.num_kv_heads,
            "kv_groups": self.num_key_value_groups,
        }

        if kwargs.get("return_kv_cache", False):
            # Return un-expanded KV for caching efficiency
            metadata["kv_cache"] = (
                k[:, :: self.num_key_value_groups],
                v[:, :: self.num_key_value_groups],
            )

        return AttentionOutput(
            output=output,
            attention_weights=None,
            metadata=metadata,
        )

    def _repeat_kv(self, hidden_states: Tensor) -> Tensor:
        """Repeat KV heads to match the number of query heads.

        Args:
            hidden_states: (batch, num_kv_heads, seq_len, head_dim)

        Returns:
            Tensor of shape (batch, num_heads, seq_len, head_dim)
        """
        if self.num_key_value_groups == 1:
            return hidden_states

        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape

        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, self.num_key_value_groups, seq_len, head_dim
        )

        return hidden_states.reshape(
            batch, num_kv_heads * self.num_key_value_groups, seq_len, head_dim
        )

    def _process_mask(
        self, mask: Optional[Tensor], query: Tensor, key: Tensor
    ) -> Optional[Tensor]:
        """Process attention mask to correct format."""
        if mask is None:
            return None

        if mask.dtype == torch.bool:
            mask = mask.float().masked_fill(~mask, float("-inf")).masked_fill(mask, 0.0)

        if mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(2)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)

        return mask

    def estimate_memory(
        self,
        batch_size: int,
        seq_len: int,
        dtype: Optional[torch.dtype] = None,
    ) -> dict[str, int]:
        """Estimate memory usage for GQA.

        GQA uses less memory for KV than standard MHA due to fewer KV heads.
        """
        dtype = dtype or self.config.dtype
        bytes_per_elem = torch.tensor([], dtype=dtype).element_size()

        embed_dim = self.config.embed_dim
        num_heads = self.config.num_heads
        num_kv_heads = self.config.num_kv_heads
        head_dim = self.config.head_dim

        # Query: full num_heads
        q_size = batch_size * seq_len * num_heads * head_dim * bytes_per_elem

        # KV: reduced num_kv_heads
        kv_size = 2 * batch_size * seq_len * num_kv_heads * head_dim * bytes_per_elem

        # Output
        output_size = batch_size * seq_len * embed_dim * bytes_per_elem

        # Attention matrix (after KV expansion)
        attn_matrix_size = batch_size * num_heads * seq_len * seq_len * bytes_per_elem

        # Projection weights
        weight_bytes = torch.tensor([], dtype=torch.float32).element_size()
        q_weights = embed_dim * num_heads * head_dim * weight_bytes
        kv_weights = 2 * embed_dim * num_kv_heads * head_dim * weight_bytes
        out_weights = num_heads * head_dim * embed_dim * weight_bytes

        return {
            "query_tensor": q_size,
            "kv_tensors": kv_size,
            "output": output_size,
            "attention_matrix": attn_matrix_size,
            "projection_weights": q_weights + kv_weights + out_weights,
            "total_estimate": q_size + kv_size + output_size + attn_matrix_size,
            "kv_savings_vs_mha": f"{(1 - num_kv_heads / num_heads) * 100:.1f}%",
        }
