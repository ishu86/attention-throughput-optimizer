"""Attention Throughput Optimizer (ATO).

Research toolkit for benchmarking and optimizing attention mechanisms on NVIDIA GPUs.
"""

__version__ = "0.1.0"

from ato.attention.base import AttentionBase, AttentionConfig, AttentionOutput
from ato.attention.registry import AttentionRegistry

__all__ = [
    "AttentionBase",
    "AttentionConfig",
    "AttentionOutput",
    "AttentionRegistry",
    "__version__",
]
