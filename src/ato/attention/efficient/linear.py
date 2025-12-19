"""Linear Attention implementations for O(n) complexity.

Linear attention replaces the softmax attention with kernel-based attention,
achieving O(n) complexity instead of O(n^2) for long sequences.

Standard attention: softmax(QK^T / sqrt(d)) @ V  -> O(n^2)
Linear attention:   φ(Q) @ (φ(K)^T @ V)          -> O(n)

where φ is a feature map function.
"""

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ato.attention.base import AttentionBase, AttentionConfig, AttentionOutput, AttentionType
from ato.attention.registry import AttentionRegistry


def elu_feature_map(x: Tensor) -> Tensor:
    """ELU-based feature map: elu(x) + 1.

    This ensures positive values for valid kernel computation.
    """
    return F.elu(x) + 1


def relu_feature_map(x: Tensor) -> Tensor:
    """ReLU-based feature map."""
    return F.relu(x)


@AttentionRegistry.register("linear")
@AttentionRegistry.register("linear_attention")
class LinearAttention(AttentionBase):
    """Linear Attention with O(n) complexity.

    Uses the associativity of matrix multiplication to compute attention
    in linear time: φ(Q) @ (φ(K)^T @ V) instead of softmax(QK^T) @ V.

    This implementation uses ELU+1 as the feature map following
    "Transformers are RNNs" (Katharopoulos et al., 2020).

    Benefits:
        - O(n) time and memory complexity
        - Efficient for very long sequences (>4096 tokens)
        - Can be computed as an RNN for autoregressive generation

    Limitations:
        - May have lower quality than softmax attention for short sequences
        - No sparse attention patterns
        - Causal version requires cumulative sum computation
    """

    attention_type = AttentionType.LINEAR
    backend_name = "linear"
    supports_kv_cache = True
    supports_variable_length = True
    min_compute_capability = (7, 0)

    def __init__(self, config: AttentionConfig) -> None:
        """Initialize Linear Attention.

        Args:
            config: Attention configuration.
        """
        super().__init__(config)

        embed_dim = config.embed_dim
        num_heads = config.num_heads
        head_dim = config.head_dim

        # Projections
        self.q_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, embed_dim, bias=False)

        # Feature map (can be changed)
        self.feature_map = elu_feature_map

        # Epsilon for numerical stability
        self.eps = 1e-6

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
        """Compute linear attention.

        Args:
            query: Query tensor (batch, seq_len, embed_dim).
            key: Key tensor (batch, seq_len, embed_dim).
            value: Value tensor (batch, seq_len, embed_dim).
            attention_mask: Not supported for linear attention.
            kv_cache: Optional (S, z) state for recurrent computation.
            **kwargs: Additional arguments.

        Returns:
            AttentionOutput with attention result.
        """
        batch_size, q_len, _ = query.shape
        kv_len = key.size(1)

        # Project inputs
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape to (batch, heads, seq, head_dim)
        q = self._reshape(q, batch_size, q_len)
        k = self._reshape(k, batch_size, kv_len)
        v = self._reshape(v, batch_size, kv_len)

        # Apply feature map
        q = self.feature_map(q)
        k = self.feature_map(k)

        if self.config.causal:
            output, new_state = self._causal_linear_attention(q, k, v, kv_cache)
        else:
            output, new_state = self._bidirectional_linear_attention(q, k, v)

        # Reshape back
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, q_len, -1)

        # Output projection
        output = self.out_proj(output)

        metadata = {
            "backend": self.backend_name,
            "complexity": "O(n)",
            "causal": self.config.causal,
        }

        if kwargs.get("return_kv_cache", False) and new_state is not None:
            metadata["kv_cache"] = new_state

        return AttentionOutput(
            output=output,
            attention_weights=None,
            metadata=metadata,
        )

    def _reshape(self, x: Tensor, batch_size: int, seq_len: int) -> Tensor:
        """Reshape to (batch, heads, seq, head_dim)."""
        return x.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)

    def _bidirectional_linear_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> tuple[Tensor, None]:
        """Compute bidirectional linear attention.

        Uses the formula: output = φ(Q) @ (φ(K)^T @ V) / (φ(Q) @ φ(K)^T @ 1)

        Args:
            q: Query (batch, heads, seq_q, head_dim) after feature map.
            k: Key (batch, heads, seq_k, head_dim) after feature map.
            v: Value (batch, heads, seq_k, head_dim).

        Returns:
            Output tensor and None (no state for bidirectional).
        """
        # Compute KV: (batch, heads, head_dim, head_dim)
        kv = torch.einsum("bhnd,bhnv->bhdv", k, v)

        # Compute output: (batch, heads, seq, head_dim)
        output = torch.einsum("bhnd,bhdv->bhnv", q, kv)

        # Normalize
        k_sum = k.sum(dim=2, keepdim=True)  # (batch, heads, 1, head_dim)
        normalizer = torch.einsum("bhnd,bhkd->bhnk", q, k_sum).squeeze(-1)  # (batch, heads, seq)
        normalizer = normalizer.unsqueeze(-1).clamp(min=self.eps)

        output = output / normalizer

        return output, None

    def _causal_linear_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        state: Optional[tuple[Tensor, Tensor]] = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Compute causal linear attention with chunk-based processing.

        Uses chunked computation to achieve O(chunk_size * d²) memory
        instead of O(n * d²), making it practical for long sequences.

        Args:
            q: Query (batch, heads, seq, head_dim) after feature map.
            k: Key (batch, heads, seq, head_dim) after feature map.
            v: Value (batch, heads, seq, head_dim).
            state: Optional (S, z) state from previous step.

        Returns:
            Output tensor and new (S, z) state.
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Initialize or use cached state
        if state is not None:
            S, z = state
            S = S.clone()
            z = z.clone()
        else:
            S = torch.zeros(batch_size, num_heads, head_dim, head_dim,
                           device=q.device, dtype=q.dtype)
            z = torch.zeros(batch_size, num_heads, head_dim,
                           device=q.device, dtype=q.dtype)

        # Single token case (autoregressive generation) - fast path
        if seq_len == 1:
            kv = torch.einsum("bhnd,bhnv->bhdv", k, v)
            S = S + kv
            z = z + k.squeeze(2)

            output = torch.einsum("bhnd,bhdv->bhnv", q, S)
            normalizer = torch.einsum("bhnd,bhd->bhn", q, z)
            normalizer = normalizer.unsqueeze(-1).clamp(min=self.eps)
            output = output / normalizer
            return output, (S, z)

        # Chunk-based processing for memory efficiency
        # Chunk size balances parallelism vs memory: O(chunk * d²) memory
        chunk_size = min(256, seq_len)  # Tune based on GPU memory
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        outputs = []

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, seq_len)
            chunk_len = end - start

            q_chunk = q[:, :, start:end]  # (batch, heads, chunk, d)
            k_chunk = k[:, :, start:end]
            v_chunk = v[:, :, start:end]

            # Process chunk with intra-chunk causal attention
            chunk_output = self._process_chunk_causal(
                q_chunk, k_chunk, v_chunk, S, z, chunk_len
            )
            outputs.append(chunk_output)

            # Update state with chunk contribution
            kv_chunk = torch.einsum("bhnd,bhnv->bhdv", k_chunk, v_chunk)
            S = S + kv_chunk
            z = z + k_chunk.sum(dim=2)

        output = torch.cat(outputs, dim=2)
        return output, (S, z)

    def _process_chunk_causal(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        S_prev: Tensor,
        z_prev: Tensor,
        chunk_len: int,
    ) -> Tensor:
        """Process a single chunk with causal masking.

        Computes attention for positions in chunk, each attending to:
        1. All previous chunks (via S_prev, z_prev state)
        2. Previous positions within this chunk (via masked attention)

        This is O(chunk² + chunk*d²) which is efficient for small chunks.
        """
        batch_size, num_heads, _, head_dim = q.shape

        # Part 1: Contribution from previous chunks (already computed in S_prev, z_prev)
        # output_prev[i] = q[i] @ S_prev / (q[i] @ z_prev)
        prev_output = torch.einsum("bhnd,bhdv->bhnv", q, S_prev)
        prev_norm = torch.einsum("bhnd,bhd->bhn", q, z_prev).unsqueeze(-1)

        # Part 2: Contribution from within this chunk (causal)
        # For small chunks, materialize the chunk attention matrix O(chunk²)
        if chunk_len <= 256:
            # Small chunk: use explicit attention matrix
            # scores[i,j] = q[i] @ k[j] for j <= i
            scores = torch.einsum("bhid,bhjd->bhij", q, k)

            # Causal mask
            causal_mask = torch.triu(
                torch.ones(chunk_len, chunk_len, device=q.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask, 0.0)

            # Weighted sum of values
            chunk_output = torch.einsum("bhij,bhjv->bhiv", scores, v)

            # Normalizer from chunk
            chunk_norm = scores.sum(dim=-1, keepdim=True)  # (batch, heads, chunk, 1)
        else:
            # Larger chunk: use sequential for memory efficiency
            chunk_output, chunk_norm = self._sequential_chunk(q, k, v, chunk_len)

        # Combine contributions
        total_norm = (prev_norm + chunk_norm).clamp(min=self.eps)
        output = (prev_output + chunk_output) / total_norm

        return output

    def _sequential_chunk(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        chunk_len: int,
    ) -> tuple[Tensor, Tensor]:
        """Sequential processing for larger chunks."""
        batch_size, num_heads, _, head_dim = q.shape

        S = torch.zeros(batch_size, num_heads, head_dim, head_dim,
                       device=q.device, dtype=q.dtype)
        z = torch.zeros(batch_size, num_heads, head_dim,
                       device=q.device, dtype=q.dtype)

        outputs = []
        norms = []

        for t in range(chunk_len):
            k_t = k[:, :, t]
            v_t = v[:, :, t]
            q_t = q[:, :, t]

            # Update state first (for causal, k_t contributes to position t)
            kv_t = torch.einsum("bhd,bhv->bhdv", k_t, v_t)
            S = S + kv_t
            z = z + k_t

            # Compute output for position t
            out = torch.einsum("bhd,bhdv->bhv", q_t, S)
            norm = torch.einsum("bhd,bhd->bh", q_t, z).unsqueeze(-1)

            outputs.append(out)
            norms.append(norm)

        return torch.stack(outputs, dim=2), torch.stack(norms, dim=2)

    def estimate_memory(
        self,
        batch_size: int,
        seq_len: int,
        dtype: Optional[torch.dtype] = None,
    ) -> dict[str, int]:
        """Estimate memory usage for linear attention.

        Linear attention uses O(n) memory instead of O(n^2).
        With chunked causal, peak is O(n + chunk² + d²) instead of O(n*d²).
        """
        dtype = dtype or self.config.dtype
        bytes_per_elem = torch.tensor([], dtype=dtype).element_size()

        embed_dim = self.config.embed_dim
        num_heads = self.config.num_heads
        head_dim = self.config.head_dim

        # QKV tensors: O(n)
        qkv_size = 3 * batch_size * seq_len * embed_dim * bytes_per_elem

        # Output: O(n)
        output_size = batch_size * seq_len * embed_dim * bytes_per_elem

        # KV state: O(d^2) per head - constant in sequence length!
        kv_state_size = batch_size * num_heads * head_dim * head_dim * bytes_per_elem

        # For causal: chunk-based computation
        # Peak memory is O(chunk² + d²) per head, NOT O(n * d²)
        chunk_size = min(256, seq_len)
        if self.config.causal:
            # Chunk attention matrix: O(chunk²)
            chunk_attn_size = batch_size * num_heads * chunk_size * chunk_size * bytes_per_elem
            temp_size = kv_state_size + chunk_attn_size
        else:
            temp_size = kv_state_size

        return {
            "qkv_tensors": qkv_size,
            "output": output_size,
            "kv_state": kv_state_size,
            "temp_computation": temp_size,
            "total_estimate": qkv_size + output_size + temp_size,
            "memory_complexity": "O(n + chunk² + d²)",
            "vs_standard": f"No O(n²) attention matrix! Peak chunk={chunk_size}",
        }


@AttentionRegistry.register("performer")
class PerformerAttention(AttentionBase):
    """Performer-style attention with random Fourier features.

    Uses FAVOR+ (Fast Attention Via positive Orthogonal Random features)
    to approximate softmax attention with O(n) complexity.

    Reference:
        "Rethinking Attention with Performers" (Choromanski et al., 2020)
    """

    attention_type = AttentionType.LINEAR
    backend_name = "performer"
    supports_kv_cache = False
    supports_variable_length = True
    min_compute_capability = (7, 0)

    def __init__(
        self,
        config: AttentionConfig,
        num_features: Optional[int] = None,
        redraw_interval: int = 1000,
    ) -> None:
        """Initialize Performer attention.

        Args:
            config: Attention configuration.
            num_features: Number of random features. Defaults to head_dim.
            redraw_interval: Steps between redrawing random features.
        """
        super().__init__(config)

        self.num_features = num_features or config.head_dim
        self.redraw_interval = redraw_interval
        self.steps = 0

        embed_dim = config.embed_dim
        num_heads = config.num_heads
        head_dim = config.head_dim

        # Projections
        self.q_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, embed_dim, bias=False)

        # Register random projection matrix
        self.register_buffer(
            "random_features",
            self._create_random_features(head_dim, self.num_features),
        )

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
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)

    def _create_random_features(self, dim: int, num_features: int) -> Tensor:
        """Create orthogonal random features."""
        # Create random orthogonal matrix
        random_matrix = torch.randn(num_features, dim)
        q, _ = torch.linalg.qr(random_matrix)
        return q * (dim ** 0.5)

    def _redraw_features(self) -> None:
        """Redraw random features (for training stability)."""
        self.random_features = self._create_random_features(
            self.config.head_dim, self.num_features
        ).to(self.random_features.device)

    def _favor_plus_features(self, x: Tensor) -> Tensor:
        """Compute FAVOR+ random features.

        φ(x) = exp(x @ W - ||x||^2 / 2) / sqrt(m)

        Args:
            x: Input tensor (batch, heads, seq, head_dim).

        Returns:
            Feature tensor (batch, heads, seq, num_features).
        """
        # x: (batch, heads, seq, head_dim)
        # random_features: (num_features, head_dim)

        # Project onto random features
        projection = torch.einsum("bhnd,md->bhnm", x, self.random_features)

        # FAVOR+ normalization
        x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True) / 2
        features = torch.exp(projection - x_norm_sq)
        features = features / (self.num_features ** 0.5)

        return features

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[tuple[Tensor, Tensor]] = None,
        **kwargs: Any,
    ) -> AttentionOutput:
        """Compute Performer attention."""
        batch_size, q_len, _ = query.shape
        kv_len = key.size(1)

        # Maybe redraw features during training
        if self.training:
            self.steps += 1
            if self.steps % self.redraw_interval == 0:
                self._redraw_features()

        # Project inputs
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape
        q = q.view(batch_size, q_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)

        # Apply FAVOR+ features
        q_prime = self._favor_plus_features(q)
        k_prime = self._favor_plus_features(k)

        if self.config.causal:
            output = self._causal_attention(q_prime, k_prime, v)
        else:
            # Bidirectional: φ(Q) @ (φ(K)^T @ V) / (φ(Q) @ φ(K)^T @ 1)
            kv = torch.einsum("bhnm,bhnv->bhmv", k_prime, v)
            output = torch.einsum("bhnm,bhmv->bhnv", q_prime, kv)

            # Normalize
            k_sum = k_prime.sum(dim=2)  # (batch, heads, num_features)
            normalizer = torch.einsum("bhnm,bhm->bhn", q_prime, k_sum)
            normalizer = normalizer.unsqueeze(-1).clamp(min=1e-6)
            output = output / normalizer

        # Reshape back
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, q_len, -1)
        output = self.out_proj(output)

        return AttentionOutput(
            output=output,
            attention_weights=None,
            metadata={
                "backend": self.backend_name,
                "complexity": "O(n)",
                "num_features": self.num_features,
            },
        )

    def _causal_attention(
        self,
        q_prime: Tensor,
        k_prime: Tensor,
        v: Tensor,
    ) -> Tensor:
        """Compute causal Performer attention."""
        batch_size, num_heads, seq_len, num_features = q_prime.shape
        head_dim = v.size(-1)

        # Initialize state
        S = torch.zeros(batch_size, num_heads, num_features, head_dim,
                       device=q_prime.device, dtype=q_prime.dtype)
        z = torch.zeros(batch_size, num_heads, num_features,
                       device=q_prime.device, dtype=q_prime.dtype)

        outputs = []
        for t in range(seq_len):
            k_t = k_prime[:, :, t]  # (batch, heads, num_features)
            v_t = v[:, :, t]        # (batch, heads, head_dim)
            q_t = q_prime[:, :, t]  # (batch, heads, num_features)

            # Update state
            S = S + torch.einsum("bhm,bhv->bhmv", k_t, v_t)
            z = z + k_t

            # Compute output
            out = torch.einsum("bhm,bhmv->bhv", q_t, S)
            normalizer = torch.einsum("bhm,bhm->bh", q_t, z).unsqueeze(-1).clamp(min=1e-6)
            out = out / normalizer
            outputs.append(out)

        return torch.stack(outputs, dim=2)

    def estimate_memory(
        self,
        batch_size: int,
        seq_len: int,
        dtype: Optional[torch.dtype] = None,
    ) -> dict[str, int]:
        """Estimate memory for Performer attention."""
        dtype = dtype or self.config.dtype
        bytes_per_elem = torch.tensor([], dtype=dtype).element_size()

        embed_dim = self.config.embed_dim
        num_heads = self.config.num_heads

        qkv_size = 3 * batch_size * seq_len * embed_dim * bytes_per_elem
        output_size = batch_size * seq_len * embed_dim * bytes_per_elem
        features_size = 2 * batch_size * num_heads * seq_len * self.num_features * bytes_per_elem

        return {
            "qkv_tensors": qkv_size,
            "output": output_size,
            "random_features": features_size,
            "total_estimate": qkv_size + output_size + features_size,
            "memory_complexity": "O(n)",
        }
