"""
Progress Tracker
Rich CLI progress tracking for training pipeline
"""

from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.live import Live
from rich.panel import Panel


class ProgressTracker:
    """Tracks and displays progress for training pipeline"""

    def __init__(self, console: Console):
        self.console = console
        self.start_time = None
        self.current_step = None
        self.total_steps = None

        # Create progress bars
        self._overall_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        )

        self._training_progress = Progress(
            TextColumn("[bold green]Training"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            TextColumn("•"),
            TextColumn("Loss: {task.fields[loss]:.4f}"),
            console=console,
        )

        self.overall_task = None
        self.training_task = None

    @contextmanager
    def overall_progress(self):
        """Context manager for overall pipeline progress"""
        self.start_time = datetime.now()

        with self._overall_progress:
            self.overall_task = self._overall_progress.add_task(
                "[cyan]Initializing...",
                total=None,
            )
            yield self

    def start_step(self, description: str):
        """Start a new pipeline step"""
        if self.overall_task is not None:
            self._overall_progress.update(
                self.overall_task,
                description=f"[cyan]{description}...",
            )

    def complete_step(self, message: str):
        """Complete current pipeline step"""
        self.console.print(f"[green]✓[/green] {message}")

    def start_training(self, total_steps: int):
        """Start training progress tracking"""
        self.total_steps = total_steps
        self.training_task = self._training_progress.add_task(
            "Training",
            total=total_steps,
            loss=0.0,
        )

    def update_training(self, step: int, total_steps: int, loss: float):
        """Update training progress"""
        if self.training_task is None:
            self.start_training(total_steps)

        self.current_step = step
        self._training_progress.update(
            self.training_task,
            completed=step,
            loss=loss,
        )

    def show_training_summary(self, stats: dict):
        """Display training summary table"""
        table = Table(title="Training Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for key, value in stats.items():
            table.add_row(key, str(value))

        self.console.print(table)

    def show_model_info(self, info: dict):
        """Display model information"""
        table = Table(title="Model Configuration", show_header=True)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")

        for key, value in info.items():
            # Format large numbers with commas
            if isinstance(value, int) and value > 1000:
                value = f"{value:,}"
            table.add_row(key.replace("_", " ").title(), str(value))

        self.console.print(table)

    def show_dataset_info(self, train_size: int, eval_size: Optional[int]):
        """Display dataset information"""
        self.console.print(
            Panel(
                (
                    f"[bold cyan]Dataset Loaded[/bold cyan]\n\n"
                    f"Training samples: [green]{train_size:,}[/green]\n"
                    f"Evaluation samples: [green]{eval_size:,}[/green]"
                    if eval_size
                    else f"Training samples: [green]{train_size:,}[/green]\n"
                    f"Evaluation samples: [yellow]None[/yellow]"
                ),
                border_style="cyan",
            )
        )

    def get_elapsed_time(self) -> str:
        """Get elapsed time since start"""
        if self.start_time is None:
            return "00:00:00"

        elapsed = datetime.now() - self.start_time
        return str(elapsed).split(".")[0]

    def get_estimated_remaining(self) -> str:
        """Get estimated remaining time for training"""
        if (
            self.start_time is None
            or self.current_step is None
            or self.total_steps is None
            or self.current_step == 0
        ):
            return "Unknown"

        elapsed = datetime.now() - self.start_time
        time_per_step = elapsed / self.current_step
        remaining_steps = self.total_steps - self.current_step
        estimated = time_per_step * remaining_steps

        return str(estimated).split(".")[0]

    def error(self, message: str):
        """Display error message"""
        self.console.print(f"[bold red]✗ Error:[/bold red] {message}")

    def warning(self, message: str):
        """Display warning message"""
        self.console.print(f"[yellow]⚠ Warning:[/yellow] {message}")

    def info(self, message: str):
        """Display info message"""
        self.console.print(f"[blue]ℹ[/blue] {message}")
