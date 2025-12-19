"""Report generation for benchmark results."""

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.table import Table

from ato.benchmark.runner import BenchmarkResult


class ReportGenerator:
    """Generate reports from benchmark results.

    Supports multiple output formats including tables, CSV, and JSON.
    """

    def __init__(self, results: list[BenchmarkResult]) -> None:
        """Initialize report generator.

        Args:
            results: List of benchmark results.
        """
        self.results = results
        self.console = Console()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame.

        Returns:
            DataFrame with all benchmark results.
        """
        return pd.DataFrame([r.to_dict() for r in self.results])

    def save_csv(self, path: str | Path) -> None:
        """Save results to CSV file.

        Args:
            path: Output file path.
        """
        df = self.to_dataframe()
        df.to_csv(path, index=False)

    def save_json(self, path: str | Path) -> None:
        """Save results to JSON file.

        Args:
            path: Output file path.
        """
        data = [r.to_dict() for r in self.results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def print_summary_table(self) -> None:
        """Print summary table to console."""
        table = Table(title="Benchmark Results")

        table.add_column("Attention", style="cyan")
        table.add_column("Batch", justify="right")
        table.add_column("SeqLen", justify="right")
        table.add_column("Mean (ms)", justify="right", style="green")
        table.add_column("P95 (ms)", justify="right")
        table.add_column("Tokens/s", justify="right", style="yellow")
        table.add_column("TFLOPS", justify="right", style="magenta")
        table.add_column("Peak Mem (MB)", justify="right")

        for r in self.results:
            table.add_row(
                r.attention_name,
                str(r.batch_size),
                str(r.seq_length),
                f"{r.mean_time_ms:.2f}",
                f"{r.p95_time_ms:.2f}",
                f"{r.tokens_per_second:,.0f}",
                f"{r.tflops:.2f}",
                f"{r.peak_memory_mb:.1f}",
            )

        self.console.print(table)

    def print_comparison_table(
        self,
        baseline: str = "standard",
        metric: str = "mean_time_ms",
    ) -> None:
        """Print comparison table relative to a baseline.

        Args:
            baseline: Name of baseline attention mechanism.
            metric: Metric to compare.
        """
        df = self.to_dataframe()

        if baseline not in df["attention_name"].values:
            self.console.print(f"[red]Baseline '{baseline}' not found in results[/red]")
            return

        table = Table(title=f"Comparison vs {baseline} ({metric})")

        table.add_column("Attention", style="cyan")
        table.add_column("Batch", justify="right")
        table.add_column("SeqLen", justify="right")
        table.add_column("Value", justify="right")
        table.add_column("vs Baseline", justify="right")
        table.add_column("Speedup", justify="right", style="green")

        # Group by configuration
        for (batch, seq), group in df.groupby(["batch_size", "seq_length"]):
            baseline_row = group[group["attention_name"] == baseline]
            if baseline_row.empty:
                continue

            baseline_value = baseline_row[metric].values[0]

            for _, row in group.iterrows():
                value = row[metric]
                ratio = baseline_value / value if value > 0 else 0

                speedup_str = f"{ratio:.2f}x"
                if row["attention_name"] == baseline:
                    speedup_str = "-"
                elif ratio > 1:
                    speedup_str = f"[green]{ratio:.2f}x[/green]"
                elif ratio < 1:
                    speedup_str = f"[red]{ratio:.2f}x[/red]"

                table.add_row(
                    row["attention_name"],
                    str(int(batch)),
                    str(int(seq)),
                    f"{value:.2f}",
                    f"{baseline_value:.2f}",
                    speedup_str,
                )

        self.console.print(table)

    def print_memory_comparison(self) -> None:
        """Print memory usage comparison table."""
        table = Table(title="Memory Usage Comparison")

        table.add_column("Attention", style="cyan")
        table.add_column("Batch", justify="right")
        table.add_column("SeqLen", justify="right")
        table.add_column("Peak (MB)", justify="right", style="yellow")
        table.add_column("Allocated (MB)", justify="right")

        for r in self.results:
            table.add_row(
                r.attention_name,
                str(r.batch_size),
                str(r.seq_length),
                f"{r.peak_memory_mb:.1f}",
                f"{r.allocated_memory_mb:.1f}",
            )

        self.console.print(table)

    def generate_report(
        self,
        output_dir: Optional[str | Path] = None,
        include_csv: bool = True,
        include_json: bool = True,
        baseline: str = "standard",
    ) -> None:
        """Generate full report with tables and files.

        Args:
            output_dir: Directory for output files.
            include_csv: Whether to save CSV file.
            include_json: Whether to save JSON file.
            baseline: Baseline for comparison.
        """
        self.console.print("\n[bold]Benchmark Report[/bold]\n")

        # Print summary
        self.print_summary_table()
        self.console.print()

        # Print comparison if multiple attention types
        attention_names = set(r.attention_name for r in self.results)
        if len(attention_names) > 1 and baseline in attention_names:
            self.print_comparison_table(baseline)
            self.console.print()

        # Print memory comparison
        self.print_memory_comparison()

        # Save files if output directory specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            if include_csv:
                csv_path = output_dir / "benchmark_results.csv"
                self.save_csv(csv_path)
                self.console.print(f"\n[green]Saved CSV to {csv_path}[/green]")

            if include_json:
                json_path = output_dir / "benchmark_results.json"
                self.save_json(json_path)
                self.console.print(f"[green]Saved JSON to {json_path}[/green]")
