"""Configuration management for ATO."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from omegaconf import OmegaConf, DictConfig


@dataclass
class ATOConfig:
    """Global ATO configuration."""

    # Default device settings
    device: str = "cuda"
    dtype: str = "float16"

    # Benchmark defaults
    warmup_iterations: int = 10
    benchmark_iterations: int = 100

    # Profiling settings
    profile_memory: bool = True
    profile_cuda: bool = False

    # Output settings
    output_dir: str = "results"
    save_raw_results: bool = True


def load_config(
    path: Optional[Union[str, Path]] = None,
    overrides: Optional[dict[str, Any]] = None,
) -> DictConfig:
    """Load configuration from YAML file with optional overrides.

    Args:
        path: Path to YAML configuration file. If None, uses defaults.
        overrides: Dictionary of overrides to apply.

    Returns:
        OmegaConf DictConfig with merged configuration.
    """
    # Start with defaults
    default_config = OmegaConf.structured(ATOConfig)

    if path is not None:
        path = Path(path)
        if path.exists():
            file_config = OmegaConf.load(path)
            config = OmegaConf.merge(default_config, file_config)
        else:
            raise FileNotFoundError(f"Configuration file not found: {path}")
    else:
        config = default_config

    # Apply overrides
    if overrides:
        override_config = OmegaConf.create(overrides)
        config = OmegaConf.merge(config, override_config)

    return config


def save_config(config: DictConfig, path: Union[str, Path]) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration to save.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, path)


class ConfigManager:
    """Configuration manager for ATO.

    Provides centralized access to configuration with caching.
    """

    _instance: Optional["ConfigManager"] = None
    _config: Optional[DictConfig] = None

    def __new__(cls) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(
        self,
        path: Optional[Union[str, Path]] = None,
        overrides: Optional[dict[str, Any]] = None,
    ) -> DictConfig:
        """Load and cache configuration."""
        self._config = load_config(path, overrides)
        return self._config

    @property
    def config(self) -> DictConfig:
        """Get current configuration, loading defaults if needed."""
        if self._config is None:
            self._config = load_config()
        return self._config

    def reset(self) -> None:
        """Reset cached configuration."""
        self._config = None


# Global config manager instance
config_manager = ConfigManager()
