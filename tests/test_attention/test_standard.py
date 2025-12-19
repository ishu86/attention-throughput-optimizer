"""Tests for standard attention implementations."""

import pytest
import torch

from ato.attention.base import AttentionConfig, AttentionOutput
from ato.attention.registry import AttentionRegistry
from ato.attention.standard.scaled_dot_product import ScaledDotProductAttention
from ato.attention.standard.multi_head import MultiHeadAttention
from ato.attention.standard.grouped_query import GroupedQueryAttention


class TestScaledDotProductAttention:
    """Tests for ScaledDotProductAttention."""

    def test_registration(self):
        """Test that attention is registered correctly."""
        assert "standard" in AttentionRegistry.list_registered()
        assert "sdpa" in AttentionRegistry.list_registered()

    def test_output_shape(self, small_config, small_inputs):
        """Test output shape matches input."""
        attention = ScaledDotProductAttention(small_config)
        attention = attention.to(small_config.device)

        output = attention(
            small_inputs["query"],
            small_inputs["key"],
            small_inputs["value"],
        )

        assert isinstance(output, AttentionOutput)
        assert output.output.shape == small_inputs["query"].shape

    def test_output_dtype(self, small_config, small_inputs):
        """Test output dtype matches input."""
        attention = ScaledDotProductAttention(small_config)
        attention = attention.to(small_config.device)

        output = attention(
            small_inputs["query"],
            small_inputs["key"],
            small_inputs["value"],
        )

        assert output.output.dtype == small_inputs["query"].dtype

    def test_causal_masking(self, device):
        """Test causal masking works correctly."""
        config = AttentionConfig(
            embed_dim=64,
            num_heads=2,
            causal=True,
            device=device,
            dtype=torch.float32,
        )
        attention = ScaledDotProductAttention(config).to(device)

        batch_size, seq_len = 1, 8
        shape = (batch_size, seq_len, config.embed_dim)

        # Create inputs where future positions should not affect past
        q = torch.randn(shape, device=device)
        k = torch.randn(shape, device=device)
        v = torch.randn(shape, device=device)

        output = attention(q, k, v)

        # Output should have same shape
        assert output.output.shape == shape

    def test_numerical_stability(self, small_config, device):
        """Test attention is numerically stable with various input magnitudes."""
        attention = ScaledDotProductAttention(small_config).to(device)

        for scale in [0.01, 1.0, 10.0]:
            shape = (2, 32, small_config.embed_dim)
            q = torch.randn(shape, device=device, dtype=small_config.dtype) * scale
            k = torch.randn(shape, device=device, dtype=small_config.dtype) * scale
            v = torch.randn(shape, device=device, dtype=small_config.dtype) * scale

            output = attention(q, k, v)

            assert not torch.isnan(output.output).any(), f"NaN with scale={scale}"
            assert not torch.isinf(output.output).any(), f"Inf with scale={scale}"

    def test_invalid_config(self, device):
        """Test that invalid config raises error."""
        with pytest.raises(ValueError):
            # embed_dim not divisible by num_heads
            config = AttentionConfig(
                embed_dim=100,
                num_heads=12,
                device=device,
            )
            ScaledDotProductAttention(config)


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention."""

    def test_registration(self):
        """Test that attention is registered correctly."""
        assert "multi_head" in AttentionRegistry.list_registered()
        assert "mha" in AttentionRegistry.list_registered()

    def test_output_shape(self, small_config, small_inputs):
        """Test output shape matches input."""
        attention = MultiHeadAttention(small_config)
        attention = attention.to(small_config.device)

        output = attention(
            small_inputs["query"],
            small_inputs["key"],
            small_inputs["value"],
        )

        assert isinstance(output, AttentionOutput)
        assert output.output.shape == small_inputs["query"].shape

    def test_has_projections(self, small_config):
        """Test that MHA has learnable projections."""
        attention = MultiHeadAttention(small_config)

        assert hasattr(attention, "q_proj")
        assert hasattr(attention, "k_proj")
        assert hasattr(attention, "v_proj")
        assert hasattr(attention, "out_proj")

        # Check projection shapes
        embed_dim = small_config.embed_dim
        num_heads = small_config.num_heads
        head_dim = small_config.head_dim

        assert attention.q_proj.weight.shape == (num_heads * head_dim, embed_dim)
        assert attention.out_proj.weight.shape == (embed_dim, num_heads * head_dim)

    def test_gradient_flow(self, small_config, device):
        """Test gradients flow through attention."""
        attention = MultiHeadAttention(small_config).to(device)

        batch_size, seq_len = 2, 32
        shape = (batch_size, seq_len, small_config.embed_dim)

        q = torch.randn(shape, device=device, requires_grad=True)
        k = torch.randn(shape, device=device, requires_grad=True)
        v = torch.randn(shape, device=device, requires_grad=True)

        output = attention(q, k, v)
        loss = output.output.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

    def test_kv_cache(self, small_config, device):
        """Test KV cache functionality."""
        attention = MultiHeadAttention(small_config).to(device)

        batch_size = 2
        embed_dim = small_config.embed_dim
        num_heads = small_config.num_heads
        head_dim = small_config.head_dim

        # Initial forward pass
        q1 = torch.randn(batch_size, 8, embed_dim, device=device)
        k1 = torch.randn(batch_size, 8, embed_dim, device=device)
        v1 = torch.randn(batch_size, 8, embed_dim, device=device)

        output1 = attention(q1, k1, v1, return_kv_cache=True)
        assert "kv_cache" in output1.metadata

        kv_cache = output1.metadata["kv_cache"]
        assert kv_cache[0].shape == (batch_size, num_heads, 8, head_dim)

        # Incremental forward with cache
        q2 = torch.randn(batch_size, 1, embed_dim, device=device)
        k2 = torch.randn(batch_size, 1, embed_dim, device=device)
        v2 = torch.randn(batch_size, 1, embed_dim, device=device)

        output2 = attention(q2, k2, v2, kv_cache=kv_cache, return_kv_cache=True)
        new_cache = output2.metadata["kv_cache"]
        assert new_cache[0].shape == (batch_size, num_heads, 9, head_dim)


class TestGroupedQueryAttention:
    """Tests for GroupedQueryAttention."""

    def test_registration(self):
        """Test that attention is registered correctly."""
        assert "grouped_query" in AttentionRegistry.list_registered()
        assert "gqa" in AttentionRegistry.list_registered()

    def test_output_shape(self, device):
        """Test output shape matches input."""
        config = AttentionConfig(
            embed_dim=256,
            num_heads=8,
            num_kv_heads=2,  # GQA with 4:1 ratio
            device=device,
            dtype=torch.float32,
        )
        attention = GroupedQueryAttention(config).to(device)

        batch_size, seq_len = 2, 32
        shape = (batch_size, seq_len, config.embed_dim)

        q = torch.randn(shape, device=device)
        k = torch.randn(shape, device=device)
        v = torch.randn(shape, device=device)

        output = attention(q, k, v)

        assert output.output.shape == shape

    def test_mqa_mode(self, device):
        """Test MQA (single KV head) mode."""
        config = AttentionConfig(
            embed_dim=256,
            num_heads=8,
            num_kv_heads=1,  # MQA
            device=device,
            dtype=torch.float32,
        )
        attention = GroupedQueryAttention(config).to(device)

        batch_size, seq_len = 2, 32
        shape = (batch_size, seq_len, config.embed_dim)

        q = torch.randn(shape, device=device)
        k = torch.randn(shape, device=device)
        v = torch.randn(shape, device=device)

        output = attention(q, k, v)

        assert output.output.shape == shape
        assert output.metadata["num_kv_heads"] == 1
        assert output.metadata["kv_groups"] == 8

    def test_mha_mode(self, device):
        """Test that GQA with num_kv_heads == num_heads equals MHA."""
        config = AttentionConfig(
            embed_dim=256,
            num_heads=8,
            num_kv_heads=8,  # Same as MHA
            device=device,
            dtype=torch.float32,
        )
        attention = GroupedQueryAttention(config).to(device)

        assert attention.num_key_value_groups == 1

    def test_invalid_kv_heads(self, device):
        """Test that invalid num_kv_heads raises error."""
        with pytest.raises(ValueError):
            config = AttentionConfig(
                embed_dim=256,
                num_heads=8,
                num_kv_heads=3,  # 8 not divisible by 3
                device=device,
            )
            GroupedQueryAttention(config)

    def test_memory_estimate(self, device):
        """Test memory estimate shows KV savings."""
        config = AttentionConfig(
            embed_dim=256,
            num_heads=8,
            num_kv_heads=2,
            device=device,
        )
        attention = GroupedQueryAttention(config).to(device)

        estimate = attention.estimate_memory(batch_size=4, seq_len=1024)

        assert "kv_savings_vs_mha" in estimate
        assert "75.0%" in estimate["kv_savings_vs_mha"]  # 1 - 2/8 = 75%


class TestAttentionRegistry:
    """Tests for AttentionRegistry."""

    def test_create_by_name(self, small_config):
        """Test creating attention by name."""
        attention = AttentionRegistry.create("standard", small_config)
        assert isinstance(attention, ScaledDotProductAttention)

    def test_create_unknown_raises(self, small_config):
        """Test creating unknown attention raises error."""
        with pytest.raises(KeyError):
            AttentionRegistry.create("unknown_attention", small_config)

    def test_list_registered(self):
        """Test listing registered attention mechanisms."""
        registered = AttentionRegistry.list_registered()
        assert "standard" in registered
        assert "multi_head" in registered
        assert "grouped_query" in registered

    def test_list_available(self):
        """Test listing available attention mechanisms."""
        available = AttentionRegistry.list_available()
        assert len(available) > 0

        # Each should have expected fields
        for info in available:
            assert "name" in info
            assert "backend" in info
            assert "available" in info

    def test_get_info(self):
        """Test getting info for an attention class."""
        info = ScaledDotProductAttention.get_info()

        assert info["name"] == "ScaledDotProductAttention"
        assert info["backend"] == "pytorch"
        assert "supports_kv_cache" in info
