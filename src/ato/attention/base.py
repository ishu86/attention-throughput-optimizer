"""Base classes and interfaces for attention mechanisms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Union

import torch
from torch import Tensor


class AttentionType(Enum):
    """Types of attention mechanisms."""

    STANDARD = auto()
    MULTI_HEAD = auto()
    GROUPED_QUERY = auto()
    MULTI_LATENT = auto()
    LINEAR = auto()
    SPARSE = auto()
    FLASH = auto()


@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms.

    Attributes:
        embed_dim: Embedding dimension (total dimension across all heads).
        num_heads: Number of attention heads.
        head_dim: Dimension per head (computed from embed_dim/num_heads if not provided).
        num_kv_heads: Number of key/value heads for GQA/MQA (defaults to num_heads).
        dropout: Dropout probability.
        causal: Whether to apply causal masking.
        scale: Custom softmax scale (defaults to 1/sqrt(head_dim)).
        window_size: Sliding window size (left, right) for windowed attention.
        use_rotary: Whether to use rotary position embeddings.
        device: Target device.
        dtype: Data type for computations.
    """

    # Core dimensions
    embed_dim: int
    num_heads: int
    head_dim: Optional[int] = None
    num_kv_heads: Optional[int] = None

    # Attention modifiers
    dropout: float = 0.0
    causal: bool = False
    scale: Optional[float] = None
    window_size: Optional[tuple[int, int]] = None

    # Position encoding
    use_rotary: bool = False

    # Hardware
    device: Union[str, torch.device] = "cuda"
    dtype: torch.dtype = torch.float16

    def __post_init__(self) -> None:
        """Compute derived values."""
        if self.head_dim is None:
            if self.embed_dim % self.num_heads != 0:
                raise ValueError(
                    f"embed_dim ({self.embed_dim}) must be divisible by "
                    f"num_heads ({self.num_heads})"
                )
            self.head_dim = self.embed_dim // self.num_heads

        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads

        if self.scale is None:
            self.scale = self.head_dim**-0.5

        if isinstance(self.device, str):
            self.device = torch.device(self.device)


@dataclass
class AttentionOutput:
    """Standardized output from attention mechanisms.

    Attributes:
        output: The attention output tensor (batch, seq_len, embed_dim).
        attention_weights: Optional attention weights (batch, heads, seq_len, seq_len).
        metadata: Additional metadata from the computation.
    """

    output: Tensor
    attention_weights: Optional[Tensor] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AttentionBase(ABC, torch.nn.Module):
    """Abstract base class for all attention mechanisms.

    All attention implementations should inherit from this class to ensure
    a consistent interface for benchmarking and comparison.

    Class Attributes:
        attention_type: The type of attention mechanism.
        backend_name: Name of the backend (e.g., "pytorch", "flash", "xformers").
        supports_kv_cache: Whether this implementation supports KV caching.
        supports_variable_length: Whether this supports variable-length sequences.
        min_compute_capability: Minimum CUDA compute capability required.
    """

    attention_type: AttentionType = AttentionType.STANDARD
    backend_name: str = "base"
    supports_kv_cache: bool = False
    supports_variable_length: bool = False
    min_compute_capability: tuple[int, int] = (7, 0)  # Volta+

    def __init__(self, config: AttentionConfig) -> None:
        """Initialize attention mechanism.

        Args:
            config: Attention configuration.
        """
        super().__init__()
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate configuration for this attention type.

        Should raise ValueError if configuration is invalid.
        """
        pass

    @abstractmethod
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[tuple[Tensor, Tensor]] = None,
        **kwargs: Any,
    ) -> AttentionOutput:
        """Compute attention.

        Args:
            query: Query tensor (batch, seq_len, embed_dim) or
                   (batch, seq_len, num_heads, head_dim).
            key: Key tensor with same format as query.
            value: Value tensor with same format as query.
            attention_mask: Optional mask tensor. Shape depends on implementation.
            kv_cache: Optional tuple of (cached_keys, cached_values).
            **kwargs: Implementation-specific arguments.

        Returns:
            AttentionOutput containing the attention result and metadata.
        """
        pass

    @classmethod
    def is_available(cls) -> bool:
        """Check if this attention backend is available on current hardware.

        Returns:
            True if the backend can be used, False otherwise.
        """
        if not torch.cuda.is_available():
            return False
        capability = torch.cuda.get_device_capability()
        return capability >= cls.min_compute_capability

    @classmethod
    def get_info(cls) -> dict[str, Any]:
        """Return information about this attention implementation.

        Returns:
            Dictionary with implementation details.
        """
        return {
            "name": cls.__name__,
            "backend": cls.backend_name,
            "type": cls.attention_type.name,
            "supports_kv_cache": cls.supports_kv_cache,
            "supports_variable_length": cls.supports_variable_length,
            "min_compute_capability": cls.min_compute_capability,
            "available": cls.is_available(),
        }

    def estimate_memory(
        self,
        batch_size: int,
        seq_len: int,
        dtype: Optional[torch.dtype] = None,
    ) -> dict[str, int]:
        """Estimate memory usage for given input dimensions.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            dtype: Data type (uses config dtype if not specified).

        Returns:
            Dictionary with memory estimates in bytes.
        """
        dtype = dtype or self.config.dtype
        bytes_per_elem = torch.tensor([], dtype=dtype).element_size()

        embed_dim = self.config.embed_dim
        num_heads = self.config.num_heads
        head_dim = self.config.head_dim

        # QKV tensors
        qkv_size = 3 * batch_size * seq_len * embed_dim * bytes_per_elem

        # Output tensor
        output_size = batch_size * seq_len * embed_dim * bytes_per_elem

        # Attention matrix (for non-flash implementations)
        attn_matrix_size = batch_size * num_heads * seq_len * seq_len * bytes_per_elem

        return {
            "qkv_tensors": qkv_size,
            "output": output_size,
            "attention_matrix": attn_matrix_size,
            "total_estimate": qkv_size + output_size + attn_matrix_size,
        }

    def extra_repr(self) -> str:
        """Return extra representation string for module."""
        return (
            f"embed_dim={self.config.embed_dim}, "
            f"num_heads={self.config.num_heads}, "
            f"head_dim={self.config.head_dim}, "
            f"causal={self.config.causal}"
        )
