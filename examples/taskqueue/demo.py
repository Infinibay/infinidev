#!/usr/bin/env python3
"""Demo script for the TaskQueue with live progress dashboard."""
import signal
import sys
import time
import logging
from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.live import Live

from taskqueue import TaskQueue, Priority

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)


# Sample task functions
def successful_task(value: int) -> str:
    """A task that always succeeds."""
    time.sleep(0.5)
    return f"Success with value {value}"


def failing_task(attempt: int = 1) -> str:
    """A task that fails twice then succeeds."""
    time.sleep(0.3)
    if attempt < 2:
        raise ValueError(f"Simulated failure on attempt {attempt}")
    return "Eventual success after retry"


def slow_task(duration: float) -> str:
    """A task that takes a specified duration."""
    time.sleep(duration)
    return f"Completed slow task after {duration}s"


def immediate_failure() -> str:
    """A task that always fails immediately."""
    raise RuntimeError("This task always fails")


def timeout_task() -> str:
    """A task that takes too long and will timeout."""
    time.sleep(10)
    return "This should not complete"


def quick_task(value: int) -> int:
    """A quick task that returns the squared value."""
    return value * value


class TaskQueueDashboard:
    """Live dashboard for monitoring task queue progress."""

    def __init__(self, queue: TaskQueue):
        self.queue = queue
        self.console = Console()
        self.task_ids: List[str] = []
        self.shutdown = False

    def setup_signal_handlers(self):
        """Setup Ctrl+C handler for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signal."""
        self.console.print("\n[yellow]Shutting down...[/yellow]")
        self.shutdown = True

    def submit_tasks(self):
        """Submit 20 mixed tasks with different behaviors."""
        tasks = [
            # Successful tasks (HIGH priority)
            (successful_task, (1,), {}),
            (successful_task, (2,), {}),
            (successful_task, (3,), {}),
            (successful_task, (4,), {}),
            (successful_task, (5,), {}),

            # Quick tasks (MEDIUM priority)
            (quick_task, (10,), {}),
            (quick_task, (20,), {}),
            (quick_task, (30,), {}),
            (quick_task, (40,), {}),
            (quick_task, (50,), {}),

            # Tasks that will retry then succeed (MEDIUM priority)
            (failing_task, (), {}),
            (failing_task, (), {}),

            # Slow tasks (LOW priority)
            (slow_task, (2,), {}),
            (slow_task, (1,), {}),
            (slow_task, (1.5,), {}),

            # Quick tasks (HIGH priority)
            (quick_task, (60,), {}),
            (quick_task, (70,), {}),

            # Immediate failures (will retry 3 times)
            (immediate_failure, (), {}),
            (immediate_failure, (), {}),
        ]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}")
        ) as progress:
            task = progress.add_task("Submitting tasks...", total=len(tasks))
            for func, args, kwargs in tasks:
                priority = Priority.HIGH if "quick" in func.__name__ or "success" in func.__name__ else Priority.MEDIUM
                if "slow" in func.__name__:
                    priority = Priority.LOW
                task_id = self.queue.submit(func, args=args, kwargs=kwargs, priority=priority)
                self.task_ids.append(task_id)
                progress.advance(task)

        self.console.print(f"[green]Submitted {len(self.task_ids)} tasks[/green]")

    def render_table(self) -> Table:
        """Render the task status table."""
        table = Table(title="Task Status")
        table.add_column("ID", style="cyan")
        table.add_column("Status", style="yellow")
        table.add_column("Priority", style="green")
        table.add_column("Result/Error", style="white")

        # Get progress
        progress = self.queue.get_progress()

        for task_id in self.task_ids:
            status_info = self.queue.status(task_id)
            if status_info:
                status = status_info['status']
                priority = status_info['priority']
                if status == 'completed':
                    result = status_info.get('result', 'N/A')
                    if result and len(str(result)) > 30:
                        result = str(result)[:30] + '...'
                    table.add_row(task_id, f"[green]{status}[/green]", priority, str(result))
                elif status == 'failed':
                    error = status_info.get('error', 'Unknown error')
                    table.add_row(task_id, f"[red]{status}[/red]", priority, f"{error} (retries: {status_info.get('retries', 0)})")
                elif status == 'timeout':
                    table.add_row(task_id, f"[orange]{status}[/orange]", priority, "Timed out")
                elif status == 'running':
                    table.add_row(task_id, f"[blue]{status}[/blue]", priority, "...")
                else:
                    table.add_row(task_id, f"[dim]{status}[/dim]", priority, "waiting...")

        return table

    def render_summary(self, progress: dict) -> Panel:
        """Render the progress summary panel."""
        summary = f"""[bold]Task Queue Progress[/bold]

    Total Tasks: {progress['total']}
    Completed:   [green]{progress['completed']}[/green]
    Failed:      [red]{progress['failed']}[/red]
    Running:     [blue]{progress['running']}[/blue]
    Pending:     [dim]{progress['pending']}[/dim]

    Progress: {progress['progress']:.1f}%
"""
        return Panel(summary, title="Progress", border_style="green")

    def run(self):
        """Run the dashboard."""
        self.setup_signal_handlers()
        self.submit_tasks()

        # Run the live dashboard
        with Live(console=console, refresh_per_second=4) as live:
            while not self.shutdown:
                progress = self.queue.get_progress()

                # Create combined display
                table = self.render_table()
                summary = self.render_summary(progress)

                # Display both summary and table
                display = f"{summary}\n{table}"
                live.update(display)

                # Check if all tasks are done
                if progress['total'] > 0:
                    completed = progress['completed'] + progress['failed']
                    if completed == progress['total']:
                        break

                time.sleep(0.25)

        self._show_results()

    def _show_results(self):
        """Show final results summary."""
        self.console.print("\n[bold]Final Results[/bold]")

        results = self.queue.results()
        completed = sum(1 for r in results.values() if r['status'] == 'completed')
        failed = sum(1 for r in results.values() if r['status'] in ('failed', 'timeout'))

        success_table = Table(title="Completed Tasks")
        success_table.add_column("Task ID", style="cyan")
        success_table.add_column("Result", style="green")

        fail_table = Table(title="Failed Tasks")
        fail_table.add_column("Task ID", style="cyan")
        fail_table.add_column("Error", style="red")

        for task_id, result in results.items():
            if result['status'] == 'completed':
                success_table.add_row(task_id, str(result.get('result', 'N/A')))
            else:
                fail_table.add_row(task_id, str(result.get('error', 'Unknown')))

        self.console.print(success_table)
        self.console.print(fail_table)

        self.console.print(f"\n[bold]Summary:[/bold] {completed} completed, {failed} failed")
        self.console.print(f"Stats: {self.queue.stats}")

        # Graceful shutdown
        self.queue.shutdown()


def main():
    """Main entry point."""
    console.print("[bold magenta]TaskQueue Demo - Live Dashboard[/bold magenta]")
    console.print("Press Ctrl+C to stop early\n")

    # Create queue with 4 workers
    queue = TaskQueue(num_workers=4, retry_delay=0.5, backoff_factor=1.5)

    # Run dashboard
    dashboard = TaskQueueDashboard(queue)
    dashboard.run()


if __name__ == "__main__":
    main()
