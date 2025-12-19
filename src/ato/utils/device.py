"""Device management utilities."""

from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass
class DeviceInfo:
    """Information about a compute device.

    Attributes:
        name: Device name.
        type: Device type (cuda, cpu, mps).
        index: Device index for multi-GPU.
        total_memory_gb: Total memory in GB.
        compute_capability: CUDA compute capability (major, minor).
        driver_version: CUDA driver version.
        cuda_version: CUDA runtime version.
    """

    name: str
    type: str
    index: int
    total_memory_gb: float
    compute_capability: Optional[tuple[int, int]] = None
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None


@dataclass
class GPUMemoryInfo:
    """GPU memory information.

    Attributes:
        total_bytes: Total GPU memory.
        allocated_bytes: Currently allocated memory.
        reserved_bytes: Memory reserved by allocator.
        free_bytes: Free memory available.
    """

    total_bytes: int
    allocated_bytes: int
    reserved_bytes: int
    free_bytes: int

    @property
    def total_gb(self) -> float:
        """Total memory in GB."""
        return self.total_bytes / (1024 ** 3)

    @property
    def allocated_gb(self) -> float:
        """Allocated memory in GB."""
        return self.allocated_bytes / (1024 ** 3)

    @property
    def reserved_gb(self) -> float:
        """Reserved memory in GB."""
        return self.reserved_bytes / (1024 ** 3)

    @property
    def free_gb(self) -> float:
        """Free memory in GB."""
        return self.free_bytes / (1024 ** 3)

    @property
    def utilization(self) -> float:
        """Memory utilization as percentage."""
        return (self.allocated_bytes / self.total_bytes) * 100 if self.total_bytes > 0 else 0


def get_device(
    device: Optional[Union[str, torch.device]] = None,
    fallback_to_cpu: bool = True,
) -> torch.device:
    """Get a torch device.

    Args:
        device: Requested device. If None, uses CUDA if available.
        fallback_to_cpu: If True, falls back to CPU when CUDA unavailable.

    Returns:
        torch.device object.

    Raises:
        RuntimeError: If requested device unavailable and fallback disabled.
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    if isinstance(device, torch.device):
        device_str = str(device)
    else:
        device_str = device

    # Check availability
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            if fallback_to_cpu:
                return torch.device("cpu")
            raise RuntimeError("CUDA requested but not available")

    if device_str == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            if fallback_to_cpu:
                return torch.device("cpu")
            raise RuntimeError("MPS requested but not available")

    return torch.device(device_str)


def get_device_info(device: Optional[torch.device] = None) -> DeviceInfo:
    """Get information about a device.

    Args:
        device: Device to query. Defaults to current CUDA device.

    Returns:
        DeviceInfo with device details.
    """
    device = device or get_device()

    if device.type == "cuda":
        index = device.index or 0
        props = torch.cuda.get_device_properties(index)

        return DeviceInfo(
            name=props.name,
            type="cuda",
            index=index,
            total_memory_gb=props.total_memory / (1024 ** 3),
            compute_capability=(props.major, props.minor),
            driver_version=None,  # Would need pynvml for this
            cuda_version=torch.version.cuda,
        )

    elif device.type == "mps":
        return DeviceInfo(
            name="Apple Silicon GPU",
            type="mps",
            index=0,
            total_memory_gb=0,  # Not easily available
        )

    else:
        return DeviceInfo(
            name="CPU",
            type="cpu",
            index=0,
            total_memory_gb=0,
        )


def check_cuda_capability(
    min_capability: tuple[int, int],
    device: Optional[torch.device] = None,
) -> bool:
    """Check if device meets minimum CUDA compute capability.

    Args:
        min_capability: Minimum required (major, minor) capability.
        device: Device to check. Defaults to current CUDA device.

    Returns:
        True if device meets requirement, False otherwise.
    """
    if not torch.cuda.is_available():
        return False

    device = device or torch.device("cuda")
    if device.type != "cuda":
        return False

    index = device.index or 0
    current = torch.cuda.get_device_capability(index)

    return current >= min_capability


def get_gpu_memory_info(device: Optional[torch.device] = None) -> GPUMemoryInfo:
    """Get current GPU memory information.

    Args:
        device: CUDA device to query.

    Returns:
        GPUMemoryInfo with memory statistics.

    Raises:
        RuntimeError: If device is not CUDA.
    """
    device = device or torch.device("cuda")

    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GPU memory info only available for CUDA devices")

    index = device.index or 0

    total = torch.cuda.get_device_properties(index).total_memory
    allocated = torch.cuda.memory_allocated(index)
    reserved = torch.cuda.memory_reserved(index)
    free = total - reserved

    return GPUMemoryInfo(
        total_bytes=total,
        allocated_bytes=allocated,
        reserved_bytes=reserved,
        free_bytes=free,
    )


def get_optimal_dtype(device: torch.device) -> torch.dtype:
    """Get optimal dtype for a device.

    Args:
        device: Target device.

    Returns:
        Recommended dtype for best performance.
    """
    if device.type == "cuda":
        # Check for BF16 support (Ampere+)
        if check_cuda_capability((8, 0), device):
            return torch.bfloat16
        return torch.float16

    elif device.type == "mps":
        return torch.float16

    else:
        return torch.float32


def synchronize(device: Optional[torch.device] = None) -> None:
    """Synchronize device for accurate timing.

    Args:
        device: Device to synchronize.
    """
    device = device or get_device()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        # MPS synchronization
        torch.mps.synchronize()
