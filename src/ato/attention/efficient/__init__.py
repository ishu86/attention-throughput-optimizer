"""Efficient attention implementations wrapping optimized libraries."""

# Linear Attention (always available)
from ato.attention.efficient.linear import LinearAttention, PerformerAttention

# FlashAttention-2 (optional)
try:
    from ato.attention.efficient.flash_v2 import FlashAttentionV2

    _HAS_FLASH = True
except ImportError:
    _HAS_FLASH = False
    FlashAttentionV2 = None

# xFormers (optional)
try:
    from ato.attention.efficient.xformers_attn import XFormersAttention

    _HAS_XFORMERS = True
except ImportError:
    _HAS_XFORMERS = False
    XFormersAttention = None


def is_flash_available() -> bool:
    """Check if FlashAttention is available."""
    return _HAS_FLASH


def is_xformers_available() -> bool:
    """Check if xFormers is available."""
    return _HAS_XFORMERS


__all__ = [
    "LinearAttention",
    "PerformerAttention",
    "FlashAttentionV2",
    "XFormersAttention",
    "is_flash_available",
    "is_xformers_available",
]
