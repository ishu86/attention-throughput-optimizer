"""Attention mechanism registry for dynamic discovery and instantiation."""

from typing import Any, Optional, Type

from ato.attention.base import AttentionBase, AttentionConfig, AttentionType


class AttentionRegistry:
    """Registry for attention mechanism implementations.

    Enables dynamic registration and discovery of attention backends.
    Use the @register decorator to add new implementations.

    Example:
        @AttentionRegistry.register("my_attention")
        class MyAttention(AttentionBase):
            ...

        # Later, create instance by name
        attention = AttentionRegistry.create("my_attention", config)
    """

    _registry: dict[str, Type[AttentionBase]] = {}

    @classmethod
    def register(cls, name: Optional[str] = None) -> callable:
        """Decorator to register an attention implementation.

        Args:
            name: Registration name. If None, uses lowercase class name.

        Returns:
            Decorator function.

        Example:
            @AttentionRegistry.register("flash_v2")
            class FlashAttentionV2(AttentionBase):
                ...
        """

        def decorator(attention_cls: Type[AttentionBase]) -> Type[AttentionBase]:
            key = name or attention_cls.__name__.lower()
            if key in cls._registry:
                existing = cls._registry[key].__name__
                raise ValueError(
                    f"Attention '{key}' already registered by {existing}. "
                    f"Use a different name or unregister first."
                )
            cls._registry[key] = attention_cls
            return attention_cls

        return decorator

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove an attention implementation from the registry.

        Args:
            name: Registration name to remove.
        """
        if name in cls._registry:
            del cls._registry[name]

    @classmethod
    def get(cls, name: str) -> Type[AttentionBase]:
        """Get attention class by name.

        Args:
            name: Registration name.

        Returns:
            The attention class.

        Raises:
            KeyError: If name not found in registry.
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(f"Attention '{name}' not found. Available: {available}")
        return cls._registry[name]

    @classmethod
    def create(cls, name: str, config: AttentionConfig) -> AttentionBase:
        """Create attention instance by name.

        Args:
            name: Registration name.
            config: Attention configuration.

        Returns:
            Instantiated attention module.

        Raises:
            KeyError: If name not found in registry.
            RuntimeError: If backend is not available on current hardware.
        """
        attention_cls = cls.get(name)

        if not attention_cls.is_available():
            info = attention_cls.get_info()
            raise RuntimeError(
                f"Attention backend '{name}' is not available. "
                f"Requires compute capability >= {info['min_compute_capability']}"
            )

        return attention_cls(config)

    @classmethod
    def list_registered(cls) -> list[str]:
        """List all registered attention names.

        Returns:
            List of registration names.
        """
        return list(cls._registry.keys())

    @classmethod
    def list_available(cls) -> list[dict[str, Any]]:
        """List all registered attention mechanisms with availability info.

        Returns:
            List of dictionaries with implementation info.
        """
        return [attention_cls.get_info() for attention_cls in cls._registry.values()]

    @classmethod
    def list_by_type(cls, attention_type: AttentionType) -> list[str]:
        """List attention implementations of a specific type.

        Args:
            attention_type: The attention type to filter by.

        Returns:
            List of registration names matching the type.
        """
        return [
            name
            for name, attention_cls in cls._registry.items()
            if attention_cls.attention_type == attention_type
        ]

    @classmethod
    def list_by_backend(cls, backend: str) -> list[str]:
        """List attention implementations using a specific backend.

        Args:
            backend: Backend name to filter by (e.g., "flash", "xformers").

        Returns:
            List of registration names using the backend.
        """
        return [
            name
            for name, attention_cls in cls._registry.items()
            if attention_cls.backend_name == backend
        ]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered attention mechanisms.

        Primarily for testing purposes.
        """
        cls._registry.clear()
