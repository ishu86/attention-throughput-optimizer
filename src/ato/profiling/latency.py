"""Latency analysis tools for attention mechanisms."""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import torch
from rich.console import Console
from rich.table import Table


@dataclass
class LatencyProfile:
    """Latency profile for an operation.

    Attributes:
        name: Name of the profiled operation.
        times_ms: List of latency measurements in milliseconds.
    """

    name: str
    times_ms: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        """Mean latency in milliseconds."""
        return float(np.mean(self.times_ms)) if self.times_ms else 0.0

    @property
    def std_ms(self) -> float:
        """Standard deviation of latency."""
        return float(np.std(self.times_ms)) if len(self.times_ms) > 1 else 0.0

    @property
    def min_ms(self) -> float:
        """Minimum latency."""
        return float(np.min(self.times_ms)) if self.times_ms else 0.0

    @property
    def max_ms(self) -> float:
        """Maximum latency."""
        return float(np.max(self.times_ms)) if self.times_ms else 0.0

    @property
    def median_ms(self) -> float:
        """Median latency."""
        return float(np.median(self.times_ms)) if self.times_ms else 0.0

    def percentile(self, p: float) -> float:
        """Get latency at given percentile.

        Args:
            p: Percentile (0-100).

        Returns:
            Latency at percentile in milliseconds.
        """
        return float(np.percentile(self.times_ms, p)) if self.times_ms else 0.0

    @property
    def p50_ms(self) -> float:
        """50th percentile (median) latency."""
        return self.percentile(50)

    @property
    def p90_ms(self) -> float:
        """90th percentile latency."""
        return self.percentile(90)

    @property
    def p95_ms(self) -> float:
        """95th percentile latency."""
        return self.percentile(95)

    @property
    def p99_ms(self) -> float:
        """99th percentile latency."""
        return self.percentile(99)

    @property
    def p999_ms(self) -> float:
        """99.9th percentile latency."""
        return self.percentile(99.9)

    @property
    def count(self) -> int:
        """Number of measurements."""
        return len(self.times_ms)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "count": self.count,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "median_ms": self.median_ms,
            "p90_ms": self.p90_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "p999_ms": self.p999_ms,
        }


class LatencyAnalyzer:
    """Latency analyzer for attention operations.

    Measures execution time with CUDA synchronization for accurate GPU timing.

    Example:
        analyzer = LatencyAnalyzer()

        # Warmup
        analyzer.warmup(lambda: attention(q, k, v), iterations=10)

        # Measure
        profile = analyzer.measure(
            "attention_forward",
            lambda: attention(q, k, v),
            iterations=100
        )

        analyzer.print_report()
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        """Initialize latency analyzer.

        Args:
            device: CUDA device for synchronization.
        """
        self.device = device or torch.device("cuda")
        self.profiles: dict[str, LatencyProfile] = {}
        self.console = Console()

    def warmup(
        self,
        func: Callable[[], Any],
        iterations: int = 10,
    ) -> None:
        """Run warmup iterations.

        Args:
            func: Function to execute.
            iterations: Number of warmup iterations.
        """
        for _ in range(iterations):
            func()

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def measure(
        self,
        name: str,
        func: Callable[[], Any],
        iterations: int = 100,
        warmup: int = 10,
    ) -> LatencyProfile:
        """Measure latency of a function.

        Args:
            name: Name for this profile.
            func: Function to measure.
            iterations: Number of measurement iterations.
            warmup: Number of warmup iterations.

        Returns:
            LatencyProfile with measurements.
        """
        # Warmup
        self.warmup(func, warmup)

        # Measure
        profile = LatencyProfile(name=name)

        for _ in range(iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)

            start = time.perf_counter()
            func()

            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)

            elapsed_ms = (time.perf_counter() - start) * 1000
            profile.times_ms.append(elapsed_ms)

        self.profiles[name] = profile
        return profile

    def measure_with_events(
        self,
        name: str,
        func: Callable[[], Any],
        iterations: int = 100,
        warmup: int = 10,
    ) -> LatencyProfile:
        """Measure latency using CUDA events for more accurate timing.

        Args:
            name: Name for this profile.
            func: Function to measure.
            iterations: Number of measurement iterations.
            warmup: Number of warmup iterations.

        Returns:
            LatencyProfile with measurements.
        """
        if not torch.cuda.is_available():
            return self.measure(name, func, iterations, warmup)

        # Warmup
        self.warmup(func, warmup)

        # Create events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Measure
        profile = LatencyProfile(name=name)

        for _ in range(iterations):
            start_event.record()
            func()
            end_event.record()

            torch.cuda.synchronize(self.device)
            elapsed_ms = start_event.elapsed_time(end_event)
            profile.times_ms.append(elapsed_ms)

        self.profiles[name] = profile
        return profile

    def get_profile(self, name: str) -> Optional[LatencyProfile]:
        """Get a specific profile by name.

        Args:
            name: Profile name.

        Returns:
            LatencyProfile if found, None otherwise.
        """
        return self.profiles.get(name)

    def print_report(self) -> None:
        """Print formatted latency report."""
        table = Table(title="Latency Analysis Report")

        table.add_column("Profile", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Mean (ms)", justify="right", style="green")
        table.add_column("Std (ms)", justify="right")
        table.add_column("P50 (ms)", justify="right")
        table.add_column("P95 (ms)", justify="right", style="yellow")
        table.add_column("P99 (ms)", justify="right", style="red")
        table.add_column("Min (ms)", justify="right")
        table.add_column("Max (ms)", justify="right")

        for name, profile in self.profiles.items():
            table.add_row(
                name,
                str(profile.count),
                f"{profile.mean_ms:.3f}",
                f"{profile.std_ms:.3f}",
                f"{profile.p50_ms:.3f}",
                f"{profile.p95_ms:.3f}",
                f"{profile.p99_ms:.3f}",
                f"{profile.min_ms:.3f}",
                f"{profile.max_ms:.3f}",
            )

        self.console.print(table)

    def print_comparison(self, baseline: Optional[str] = None) -> None:
        """Print comparison table.

        Args:
            baseline: Name of baseline profile. Uses first profile if not specified.
        """
        if not self.profiles:
            self.console.print("[red]No profiles to compare[/red]")
            return

        if baseline is None:
            baseline = next(iter(self.profiles.keys()))

        if baseline not in self.profiles:
            self.console.print(f"[red]Baseline '{baseline}' not found[/red]")
            return

        baseline_profile = self.profiles[baseline]
        baseline_mean = baseline_profile.mean_ms

        table = Table(title=f"Latency Comparison (baseline: {baseline})")

        table.add_column("Profile", style="cyan")
        table.add_column("Mean (ms)", justify="right")
        table.add_column("vs Baseline", justify="right")
        table.add_column("Speedup", justify="right")

        for name, profile in self.profiles.items():
            ratio = baseline_mean / profile.mean_ms if profile.mean_ms > 0 else 0

            if name == baseline:
                speedup_str = "-"
            elif ratio > 1:
                speedup_str = f"[green]{ratio:.2f}x faster[/green]"
            elif ratio < 1:
                speedup_str = f"[red]{1/ratio:.2f}x slower[/red]"
            else:
                speedup_str = "1.00x"

            table.add_row(
                name,
                f"{profile.mean_ms:.3f}",
                f"{baseline_mean:.3f}",
                speedup_str,
            )

        self.console.print(table)

    def clear(self) -> None:
        """Clear all profiles."""
        self.profiles.clear()

    def histogram(self, name: str, bins: int = 20) -> None:
        """Print ASCII histogram of latency distribution.

        Args:
            name: Profile name.
            bins: Number of histogram bins.
        """
        profile = self.profiles.get(name)
        if not profile:
            self.console.print(f"[red]Profile '{name}' not found[/red]")
            return

        times = profile.times_ms
        hist, bin_edges = np.histogram(times, bins=bins)
        max_count = max(hist)

        self.console.print(f"\n[bold]Latency Distribution: {name}[/bold]")
        self.console.print(f"  Samples: {len(times)}")
        self.console.print(f"  Range: {min(times):.3f} - {max(times):.3f} ms\n")

        bar_width = 40

        for i, count in enumerate(hist):
            bar_len = int((count / max_count) * bar_width) if max_count > 0 else 0
            bar = "#" * bar_len
            label = f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}"
            self.console.print(f"  {label:>12} | {bar:<{bar_width}} {count}")
