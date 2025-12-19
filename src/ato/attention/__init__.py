"""Attention mechanism implementations."""

from ato.attention.base import AttentionBase, AttentionConfig, AttentionOutput, AttentionType
from ato.attention.registry import AttentionRegistry

__all__ = [
    "AttentionBase",
    "AttentionConfig",
    "AttentionOutput",
    "AttentionType",
    "AttentionRegistry",
]


def _register_all_backends() -> None:
    """Register all available attention backends.

    This function imports all attention implementations to trigger their
    registration with the AttentionRegistry.
    """
    # Standard implementations
    from ato.attention.standard import (  # noqa: F401
        scaled_dot_product,
        multi_head,
        grouped_query,
        multi_latent,
    )

    # Efficient implementations (may not be available)
    try:
        from ato.attention.efficient import flash_v2  # noqa: F401
    except ImportError:
        pass

    try:
        from ato.attention.efficient import xformers_attn  # noqa: F401
    except ImportError:
        pass

    try:
        from ato.attention.efficient import linear  # noqa: F401
    except ImportError:
        pass

    # Custom Triton kernels
    try:
        from ato.kernels.triton import flash_attn  # noqa: F401
    except ImportError:
        pass


# Auto-register backends on import
_register_all_backends()
