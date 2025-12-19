"""Metrics collection and computation for benchmarks."""

from dataclasses import dataclass, field
from typing import Any, Optional
import time

import numpy as np
import torch


@dataclass
class BenchmarkMetrics:
    """Collected metrics from a benchmark run.

    Attributes:
        times_ms: List of execution times in milliseconds.
        peak_memory_bytes: Peak GPU memory allocated.
        allocated_memory_bytes: Memory allocated at end of run.
    """

    times_ms: list[float] = field(default_factory=list)
    peak_memory_bytes: int = 0
    allocated_memory_bytes: int = 0

    @property
    def mean_ms(self) -> float:
        """Mean execution time in milliseconds."""
        return float(np.mean(self.times_ms)) if self.times_ms else 0.0

    @property
    def std_ms(self) -> float:
        """Standard deviation of execution time in milliseconds."""
        return float(np.std(self.times_ms)) if len(self.times_ms) > 1 else 0.0

    @property
    def min_ms(self) -> float:
        """Minimum execution time in milliseconds."""
        return float(np.min(self.times_ms)) if self.times_ms else 0.0

    @property
    def max_ms(self) -> float:
        """Maximum execution time in milliseconds."""
        return float(np.max(self.times_ms)) if self.times_ms else 0.0

    @property
    def median_ms(self) -> float:
        """Median execution time in milliseconds."""
        return float(np.median(self.times_ms)) if self.times_ms else 0.0

    @property
    def p90_ms(self) -> float:
        """90th percentile execution time in milliseconds."""
        return float(np.percentile(self.times_ms, 90)) if self.times_ms else 0.0

    @property
    def p95_ms(self) -> float:
        """95th percentile execution time in milliseconds."""
        return float(np.percentile(self.times_ms, 95)) if self.times_ms else 0.0

    @property
    def p99_ms(self) -> float:
        """99th percentile execution time in milliseconds."""
        return float(np.percentile(self.times_ms, 99)) if self.times_ms else 0.0

    @property
    def peak_memory_mb(self) -> float:
        """Peak memory in megabytes."""
        return self.peak_memory_bytes / (1024 ** 2)

    @property
    def allocated_memory_mb(self) -> float:
        """Allocated memory in megabytes."""
        return self.allocated_memory_bytes / (1024 ** 2)


class MetricCollector:
    """Collector for benchmark metrics.

    Handles timing measurements and memory profiling during benchmarks.
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        """Initialize metric collector.

        Args:
            device: CUDA device for memory profiling. Defaults to current device.
        """
        self.device = device or torch.device("cuda")
        self._start_time: Optional[float] = None
        self._times: list[float] = []
        self._peak_memory: int = 0

    def reset(self) -> None:
        """Reset collected metrics."""
        self._times = []
        self._peak_memory = 0

    def start_timing(self) -> None:
        """Start timing measurement.

        Uses CUDA synchronization for accurate GPU timing.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        self._start_time = time.perf_counter()

    def stop_timing(self) -> float:
        """Stop timing and record measurement.

        Returns:
            Elapsed time in milliseconds.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

        if self._start_time is None:
            raise RuntimeError("start_timing() must be called before stop_timing()")

        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        self._times.append(elapsed_ms)
        self._start_time = None

        return elapsed_ms

    def record_memory(self) -> int:
        """Record peak memory usage.

        Returns:
            Peak memory allocated in bytes.
        """
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated(self.device)
            self._peak_memory = max(self._peak_memory, peak)
            return peak
        return 0

    def reset_memory_stats(self) -> None:
        """Reset CUDA memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def get_metrics(self) -> BenchmarkMetrics:
        """Get collected metrics.

        Returns:
            BenchmarkMetrics with all collected data.
        """
        allocated = 0
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device)

        return BenchmarkMetrics(
            times_ms=self._times.copy(),
            peak_memory_bytes=self._peak_memory,
            allocated_memory_bytes=allocated,
        )


def compute_throughput(
    batch_size: int,
    seq_length: int,
    time_ms: float,
) -> float:
    """Compute throughput in tokens per second.

    Args:
        batch_size: Batch size.
        seq_length: Sequence length.
        time_ms: Execution time in milliseconds.

    Returns:
        Throughput in tokens per second.
    """
    if time_ms <= 0:
        return 0.0

    total_tokens = batch_size * seq_length
    time_sec = time_ms / 1000
    return total_tokens / time_sec


def compute_tflops(
    batch_size: int,
    seq_length: int,
    num_heads: int,
    head_dim: int,
    time_ms: float,
    causal: bool = False,
) -> float:
    """Compute TFLOPS for attention computation.

    The FLOPs for attention is approximately:
        - 4 * batch * heads * seq^2 * head_dim (for Q@K^T and attn@V)
        - Causal attention uses half the FLOPs

    Args:
        batch_size: Batch size.
        seq_length: Sequence length.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        time_ms: Execution time in milliseconds.
        causal: Whether attention is causal.

    Returns:
        TFLOPS (tera floating point operations per second).
    """
    if time_ms <= 0:
        return 0.0

    # FLOPs calculation
    flops = 4 * batch_size * num_heads * (seq_length ** 2) * head_dim

    if causal:
        flops = flops // 2  # Causal uses half the operations

    time_sec = time_ms / 1000
    tflops = (flops / time_sec) / 1e12

    return tflops


def compute_memory_bandwidth(
    bytes_transferred: int,
    time_ms: float,
) -> float:
    """Compute memory bandwidth utilization.

    Args:
        bytes_transferred: Total bytes transferred.
        time_ms: Execution time in milliseconds.

    Returns:
        Bandwidth in GB/s.
    """
    if time_ms <= 0:
        return 0.0

    time_sec = time_ms / 1000
    bandwidth_gbps = (bytes_transferred / time_sec) / 1e9

    return bandwidth_gbps
