"""GPU memory profiling tools."""

import functools
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Optional

import torch
from rich.console import Console
from rich.table import Table


@dataclass
class MemorySnapshot:
    """Snapshot of GPU memory state.

    Attributes:
        timestamp: Time of snapshot.
        allocated_bytes: Currently allocated memory.
        reserved_bytes: Memory reserved by allocator.
        peak_allocated_bytes: Peak allocated since last reset.
        peak_reserved_bytes: Peak reserved since last reset.
        label: Optional label for this snapshot.
    """

    timestamp: float
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def allocated_mb(self) -> float:
        """Allocated memory in megabytes."""
        return self.allocated_bytes / (1024 ** 2)

    @property
    def reserved_mb(self) -> float:
        """Reserved memory in megabytes."""
        return self.reserved_bytes / (1024 ** 2)

    @property
    def peak_allocated_mb(self) -> float:
        """Peak allocated memory in megabytes."""
        return self.peak_allocated_bytes / (1024 ** 2)

    @property
    def peak_reserved_mb(self) -> float:
        """Peak reserved memory in megabytes."""
        return self.peak_reserved_bytes / (1024 ** 2)


@dataclass
class MemoryProfile:
    """Complete memory profile for an operation.

    Attributes:
        name: Name of the profiled operation.
        snapshots: List of memory snapshots taken during profiling.
    """

    name: str = ""
    snapshots: list[MemorySnapshot] = field(default_factory=list)

    @property
    def peak_allocated_mb(self) -> float:
        """Peak allocated memory across all snapshots."""
        if not self.snapshots:
            return 0.0
        return max(s.peak_allocated_mb for s in self.snapshots)

    @property
    def memory_increase_mb(self) -> float:
        """Memory increase from first to last snapshot."""
        if len(self.snapshots) < 2:
            return 0.0
        return self.snapshots[-1].allocated_mb - self.snapshots[0].allocated_mb

    @property
    def start_memory_mb(self) -> float:
        """Memory at start of profile."""
        if not self.snapshots:
            return 0.0
        return self.snapshots[0].allocated_mb

    @property
    def end_memory_mb(self) -> float:
        """Memory at end of profile."""
        if not self.snapshots:
            return 0.0
        return self.snapshots[-1].allocated_mb

    @property
    def duration_ms(self) -> float:
        """Duration of profiled operation in milliseconds."""
        if len(self.snapshots) < 2:
            return 0.0
        return (self.snapshots[-1].timestamp - self.snapshots[0].timestamp) * 1000


class MemoryProfiler:
    """GPU memory profiler for attention operations.

    Provides context managers and decorators for profiling memory usage.

    Example:
        profiler = MemoryProfiler()

        with profiler.profile("attention_forward"):
            output = attention(q, k, v)

        print(profiler.report())
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        """Initialize memory profiler.

        Args:
            device: CUDA device to profile. Defaults to current device.
        """
        self.device = device or torch.device("cuda")
        self.profiles: dict[str, MemoryProfile] = {}
        self._current_profile: Optional[MemoryProfile] = None
        self.console = Console()

    def snapshot(self, label: str = "") -> MemorySnapshot:
        """Take a memory snapshot.

        Args:
            label: Optional label for the snapshot.

        Returns:
            MemorySnapshot with current memory state.
        """
        if not torch.cuda.is_available():
            return MemorySnapshot(
                timestamp=time.time(),
                allocated_bytes=0,
                reserved_bytes=0,
                peak_allocated_bytes=0,
                peak_reserved_bytes=0,
                label=label,
            )

        torch.cuda.synchronize(self.device)

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            allocated_bytes=torch.cuda.memory_allocated(self.device),
            reserved_bytes=torch.cuda.memory_reserved(self.device),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device),
            label=label,
        )

        if self._current_profile is not None:
            self._current_profile.snapshots.append(snapshot)

        return snapshot

    @contextmanager
    def profile(self, name: str) -> Generator[MemoryProfile, None, None]:
        """Context manager for profiling a code block.

        Args:
            name: Name for this profile.

        Yields:
            MemoryProfile being recorded.

        Example:
            with profiler.profile("forward_pass"):
                output = model(input)
        """
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)

        profile = MemoryProfile(name=name)
        self._current_profile = profile

        self.snapshot(f"{name}_start")

        try:
            yield profile
        finally:
            self.snapshot(f"{name}_end")
            self._current_profile = None
            self.profiles[name] = profile

    def profile_function(self, name: Optional[str] = None) -> Callable:
        """Decorator for profiling a function.

        Args:
            name: Optional profile name. Defaults to function name.

        Returns:
            Decorator function.

        Example:
            @profiler.profile_function("my_function")
            def compute_attention(q, k, v):
                ...
        """

        def decorator(func: Callable) -> Callable:
            profile_name = name or func.__name__

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.profile(profile_name):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def get_profile(self, name: str) -> Optional[MemoryProfile]:
        """Get a specific profile by name.

        Args:
            name: Profile name.

        Returns:
            MemoryProfile if found, None otherwise.
        """
        return self.profiles.get(name)

    def report(self) -> str:
        """Generate a human-readable report.

        Returns:
            Formatted report string.
        """
        lines = ["Memory Profile Report", "=" * 50]

        for name, profile in self.profiles.items():
            lines.append(f"\n{name}:")
            lines.append(f"  Peak allocated: {profile.peak_allocated_mb:.2f} MB")
            lines.append(f"  Memory increase: {profile.memory_increase_mb:.2f} MB")
            lines.append(f"  Duration: {profile.duration_ms:.2f} ms")

            if profile.snapshots:
                lines.append(f"  Start memory: {profile.start_memory_mb:.2f} MB")
                lines.append(f"  End memory: {profile.end_memory_mb:.2f} MB")

        return "\n".join(lines)

    def print_report(self) -> None:
        """Print formatted report to console."""
        table = Table(title="Memory Profile Report")

        table.add_column("Profile", style="cyan")
        table.add_column("Peak (MB)", justify="right", style="yellow")
        table.add_column("Increase (MB)", justify="right")
        table.add_column("Duration (ms)", justify="right")
        table.add_column("Start (MB)", justify="right")
        table.add_column("End (MB)", justify="right")

        for name, profile in self.profiles.items():
            table.add_row(
                name,
                f"{profile.peak_allocated_mb:.2f}",
                f"{profile.memory_increase_mb:.2f}",
                f"{profile.duration_ms:.2f}",
                f"{profile.start_memory_mb:.2f}",
                f"{profile.end_memory_mb:.2f}",
            )

        self.console.print(table)

    def clear(self) -> None:
        """Clear all profiles."""
        self.profiles.clear()

    def compare_profiles(self, *names: str) -> None:
        """Print comparison of multiple profiles.

        Args:
            *names: Profile names to compare.
        """
        profiles = [self.profiles.get(n) for n in names if n in self.profiles]

        if not profiles:
            self.console.print("[red]No matching profiles found[/red]")
            return

        table = Table(title="Profile Comparison")

        table.add_column("Profile", style="cyan")
        table.add_column("Peak (MB)", justify="right")
        table.add_column("vs First", justify="right")

        first_peak = profiles[0].peak_allocated_mb if profiles else 0

        for profile in profiles:
            if profile:
                ratio = profile.peak_allocated_mb / first_peak if first_peak > 0 else 0
                ratio_str = f"{ratio:.2f}x"

                if profile == profiles[0]:
                    ratio_str = "-"
                elif ratio < 1:
                    ratio_str = f"[green]{ratio:.2f}x[/green]"
                elif ratio > 1:
                    ratio_str = f"[red]{ratio:.2f}x[/red]"

                table.add_row(
                    profile.name,
                    f"{profile.peak_allocated_mb:.2f}",
                    ratio_str,
                )

        self.console.print(table)


def main() -> None:
    """CLI entry point for memory profiling."""
    import argparse

    parser = argparse.ArgumentParser(description="Profile attention memory usage")
    parser.add_argument(
        "--attention",
        "-a",
        type=str,
        default="standard",
        help="Attention mechanism to profile",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1,
        help="Batch size",
    )
    parser.add_argument(
        "--seq-length",
        "-s",
        type=int,
        default=1024,
        help="Sequence length",
    )
    parser.add_argument(
        "--embed-dim",
        "-e",
        type=int,
        default=768,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--num-heads",
        "-n",
        type=int,
        default=12,
        help="Number of attention heads",
    )
    args = parser.parse_args()

    from ato.attention import AttentionRegistry, AttentionConfig

    # Create attention
    config = AttentionConfig(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        device="cuda",
        dtype=torch.float16,
    )
    attention = AttentionRegistry.create(args.attention, config).cuda()

    # Generate inputs
    shape = (args.batch_size, args.seq_length, args.embed_dim)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn(shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)

    # Profile
    profiler = MemoryProfiler()

    with profiler.profile(args.attention):
        with torch.no_grad():
            _ = attention(q, k, v)

    profiler.print_report()


if __name__ == "__main__":
    main()
