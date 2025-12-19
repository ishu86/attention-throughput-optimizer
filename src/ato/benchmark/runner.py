"""Benchmark runner for attention mechanisms."""

import gc
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from ato.attention.base import AttentionBase, AttentionConfig
from ato.attention.registry import AttentionRegistry
from ato.benchmark.metrics import (
    MetricCollector,
    BenchmarkMetrics,
    compute_throughput,
    compute_tflops,
)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs.

    Attributes:
        attention_names: List of attention mechanism names to benchmark.
        batch_sizes: List of batch sizes to test.
        seq_lengths: List of sequence lengths to test.
        embed_dims: List of embedding dimensions to test.
        num_heads: List of head counts to test.
        warmup_iterations: Number of warmup iterations before timing.
        benchmark_iterations: Number of timed iterations.
        profile_memory: Whether to profile memory usage.
        device: Target device.
        dtype: Data type for tensors.
        causal: Whether to use causal attention.
        verbose: Whether to print progress.
    """

    attention_names: list[str] = field(default_factory=lambda: ["standard"])
    batch_sizes: list[int] = field(default_factory=lambda: [1, 4, 8, 16])
    seq_lengths: list[int] = field(default_factory=lambda: [512, 1024, 2048, 4096])
    embed_dims: list[int] = field(default_factory=lambda: [768])
    num_heads: list[int] = field(default_factory=lambda: [12])
    warmup_iterations: int = 10
    benchmark_iterations: int = 100
    profile_memory: bool = True
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    causal: bool = False
    verbose: bool = True


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run.

    Contains timing, throughput, memory, and configuration information.
    """

    attention_name: str
    batch_size: int
    seq_length: int
    embed_dim: int
    num_heads: int

    # Timing metrics
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    median_time_ms: float
    p95_time_ms: float
    p99_time_ms: float

    # Throughput
    tokens_per_second: float
    tflops: float

    # Memory
    peak_memory_mb: float
    allocated_memory_mb: float

    # Configuration
    causal: bool = False
    dtype: str = "float16"

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "attention_name": self.attention_name,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mean_time_ms": self.mean_time_ms,
            "std_time_ms": self.std_time_ms,
            "min_time_ms": self.min_time_ms,
            "max_time_ms": self.max_time_ms,
            "median_time_ms": self.median_time_ms,
            "p95_time_ms": self.p95_time_ms,
            "p99_time_ms": self.p99_time_ms,
            "tokens_per_second": self.tokens_per_second,
            "tflops": self.tflops,
            "peak_memory_mb": self.peak_memory_mb,
            "allocated_memory_mb": self.allocated_memory_mb,
            "causal": self.causal,
            "dtype": self.dtype,
            **self.metadata,
        }


class BenchmarkRunner:
    """Unified benchmark runner for attention mechanisms.

    Runs benchmarks across multiple configurations and collects metrics.

    Example:
        config = BenchmarkConfig(
            attention_names=["standard", "flash_v2"],
            batch_sizes=[1, 4, 8],
            seq_lengths=[512, 1024, 2048],
        )
        runner = BenchmarkRunner(config)
        results = runner.run()
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize benchmark runner.

        Args:
            config: Benchmark configuration.
        """
        self.config = config
        self.device = torch.device(config.device)
        self.console = Console()
        self.results: list[BenchmarkResult] = []

    def run(self) -> list[BenchmarkResult]:
        """Run benchmarks for all configurations.

        Returns:
            List of benchmark results.
        """
        self.results = []

        # Calculate total number of benchmarks
        total = (
            len(self.config.attention_names)
            * len(self.config.batch_sizes)
            * len(self.config.seq_lengths)
            * len(self.config.embed_dims)
            * len(self.config.num_heads)
        )

        if self.config.verbose:
            self.console.print(f"\n[bold]Running {total} benchmark configurations[/bold]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
            disable=not self.config.verbose,
        ) as progress:
            task = progress.add_task("Benchmarking...", total=total)

            for attn_name in self.config.attention_names:
                for batch_size in self.config.batch_sizes:
                    for seq_len in self.config.seq_lengths:
                        for embed_dim in self.config.embed_dims:
                            for num_heads in self.config.num_heads:
                                progress.update(
                                    task,
                                    description=f"{attn_name} B={batch_size} S={seq_len}",
                                )

                                try:
                                    result = self._run_single(
                                        attn_name,
                                        batch_size,
                                        seq_len,
                                        embed_dim,
                                        num_heads,
                                    )
                                    self.results.append(result)
                                except Exception as e:
                                    if self.config.verbose:
                                        self.console.print(
                                            f"[red]Error benchmarking {attn_name}: {e}[/red]"
                                        )

                                progress.advance(task)

        if self.config.verbose:
            self._print_summary()

        return self.results

    def _run_single(
        self,
        attention_name: str,
        batch_size: int,
        seq_length: int,
        embed_dim: int,
        num_heads: int,
    ) -> BenchmarkResult:
        """Run benchmark for a single configuration.

        Args:
            attention_name: Name of attention mechanism.
            batch_size: Batch size.
            seq_length: Sequence length.
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.

        Returns:
            BenchmarkResult with collected metrics.
        """
        # Create attention config
        attn_config = AttentionConfig(
            embed_dim=embed_dim,
            num_heads=num_heads,
            device=self.device,
            dtype=self.config.dtype,
            causal=self.config.causal,
        )

        # Create attention instance and move to device with correct dtype
        attention = AttentionRegistry.create(attention_name, attn_config)
        attention = attention.to(device=self.device, dtype=self.config.dtype)
        attention.eval()

        # Generate input tensors
        q, k, v = self._generate_inputs(batch_size, seq_length, embed_dim)

        # Create metric collector
        collector = MetricCollector(self.device)

        # Warmup
        self._warmup(attention, q, k, v)

        # Benchmark
        collector.reset()
        collector.reset_memory_stats()

        for _ in range(self.config.benchmark_iterations):
            if self.config.profile_memory:
                collector.reset_memory_stats()

            collector.start_timing()
            with torch.no_grad():
                _ = attention(q, k, v)
            collector.stop_timing()

            if self.config.profile_memory:
                collector.record_memory()

        # Get metrics
        metrics = collector.get_metrics()

        # Compute derived metrics
        head_dim = embed_dim // num_heads
        tokens_per_sec = compute_throughput(batch_size, seq_length, metrics.mean_ms)
        tflops = compute_tflops(
            batch_size, seq_length, num_heads, head_dim, metrics.mean_ms, self.config.causal
        )

        # Cleanup
        del attention, q, k, v
        torch.cuda.empty_cache()
        gc.collect()

        return BenchmarkResult(
            attention_name=attention_name,
            batch_size=batch_size,
            seq_length=seq_length,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mean_time_ms=metrics.mean_ms,
            std_time_ms=metrics.std_ms,
            min_time_ms=metrics.min_ms,
            max_time_ms=metrics.max_ms,
            median_time_ms=metrics.median_ms,
            p95_time_ms=metrics.p95_ms,
            p99_time_ms=metrics.p99_ms,
            tokens_per_second=tokens_per_sec,
            tflops=tflops,
            peak_memory_mb=metrics.peak_memory_mb,
            allocated_memory_mb=metrics.allocated_memory_mb,
            causal=self.config.causal,
            dtype=str(self.config.dtype).split(".")[-1],
        )

    def _generate_inputs(
        self,
        batch_size: int,
        seq_length: int,
        embed_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate random input tensors for benchmarking.

        Args:
            batch_size: Batch size.
            seq_length: Sequence length.
            embed_dim: Embedding dimension.

        Returns:
            Tuple of (query, key, value) tensors.
        """
        shape = (batch_size, seq_length, embed_dim)

        q = torch.randn(shape, device=self.device, dtype=self.config.dtype)
        k = torch.randn(shape, device=self.device, dtype=self.config.dtype)
        v = torch.randn(shape, device=self.device, dtype=self.config.dtype)

        return q, k, v

    def _warmup(
        self,
        attention: AttentionBase,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Run warmup iterations to stabilize GPU.

        Args:
            attention: Attention module.
            q: Query tensor.
            k: Key tensor.
            v: Value tensor.
        """
        for _ in range(self.config.warmup_iterations):
            with torch.no_grad():
                _ = attention(q, k, v)
        torch.cuda.synchronize(self.device)

    def _print_summary(self) -> None:
        """Print summary table of results."""
        if not self.results:
            return

        table = Table(title="Benchmark Results Summary")
        table.add_column("Attention", style="cyan")
        table.add_column("Batch", justify="right")
        table.add_column("SeqLen", justify="right")
        table.add_column("Time (ms)", justify="right")
        table.add_column("Tokens/s", justify="right")
        table.add_column("TFLOPS", justify="right")
        table.add_column("Memory (MB)", justify="right")

        for result in self.results:
            table.add_row(
                result.attention_name,
                str(result.batch_size),
                str(result.seq_length),
                f"{result.mean_time_ms:.2f}",
                f"{result.tokens_per_second:,.0f}",
                f"{result.tflops:.2f}",
                f"{result.peak_memory_mb:.1f}",
            )

        self.console.print(table)


def main() -> None:
    """CLI entry point for benchmarks."""
    import argparse

    parser = argparse.ArgumentParser(description="Run attention benchmarks")
    parser.add_argument(
        "--attention",
        "-a",
        type=str,
        nargs="+",
        default=["standard"],
        help="Attention mechanisms to benchmark",
    )
    parser.add_argument(
        "--batch-sizes",
        "-b",
        type=int,
        nargs="+",
        default=[1, 4, 8],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--seq-lengths",
        "-s",
        type=int,
        nargs="+",
        default=[512, 1024, 2048],
        help="Sequence lengths to test",
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
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--warmup",
        "-w",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Use causal attention",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file for results (CSV)",
    )
    args = parser.parse_args()

    config = BenchmarkConfig(
        attention_names=args.attention,
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        embed_dims=[args.embed_dim],
        num_heads=[args.num_heads],
        benchmark_iterations=args.iterations,
        warmup_iterations=args.warmup,
        causal=args.causal,
    )

    runner = BenchmarkRunner(config)
    results = runner.run()

    if args.output:
        import pandas as pd

        df = pd.DataFrame([r.to_dict() for r in results])
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
