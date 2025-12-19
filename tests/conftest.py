"""Pytest configuration and fixtures for ATO tests."""

import pytest
import torch

from ato.attention.base import AttentionConfig
from ato.attention.registry import AttentionRegistry


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "gpu_required: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if CUDA not available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu_required" in item.keywords:
                item.add_marker(skip_gpu)


@pytest.fixture
def device():
    """Get test device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dtype():
    """Default dtype for tests."""
    return torch.float32


@pytest.fixture
def default_config(device, dtype):
    """Default attention configuration for tests."""
    return AttentionConfig(
        embed_dim=768,
        num_heads=12,
        device=device,
        dtype=dtype,
    )


@pytest.fixture
def small_config(device, dtype):
    """Small configuration for fast tests."""
    return AttentionConfig(
        embed_dim=256,
        num_heads=4,
        device=device,
        dtype=dtype,
    )


@pytest.fixture
def sample_inputs(device, default_config):
    """Generate sample Q, K, V tensors."""
    batch_size, seq_len = 2, 128
    shape = (batch_size, seq_len, default_config.embed_dim)

    return {
        "query": torch.randn(shape, device=device, dtype=default_config.dtype),
        "key": torch.randn(shape, device=device, dtype=default_config.dtype),
        "value": torch.randn(shape, device=device, dtype=default_config.dtype),
    }


@pytest.fixture
def small_inputs(device, small_config):
    """Generate small sample tensors for fast tests."""
    batch_size, seq_len = 2, 32
    shape = (batch_size, seq_len, small_config.embed_dim)

    return {
        "query": torch.randn(shape, device=device, dtype=small_config.dtype),
        "key": torch.randn(shape, device=device, dtype=small_config.dtype),
        "value": torch.randn(shape, device=device, dtype=small_config.dtype),
    }


@pytest.fixture
def reference_attention():
    """Reference PyTorch attention for comparison."""
    return torch.nn.functional.scaled_dot_product_attention


@pytest.fixture(autouse=True)
def cleanup_registry():
    """Clean up registry after each test."""
    # Store original registry
    original = AttentionRegistry._registry.copy()

    yield

    # Restore original registry
    AttentionRegistry._registry = original


@pytest.fixture
def clear_cuda_cache():
    """Clear CUDA cache before and after test."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    yield

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
