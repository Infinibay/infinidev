#!/usr/bin/env python3
"""Terminal-based system monitor dashboard using Python and rich."""

import argparse
import signal
import sys
import time
from typing import Optional

import psutil
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class SystemMonitor:
    """System monitor that collects and formats system metrics."""

    def __init__(self, interval: float = 2.0, use_color: bool = True):
        self.interval = interval
        self.use_color = use_color
        self.running = True
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Set up graceful shutdown on Ctrl+C."""
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        """Handle interrupt signals gracefully."""
        self.running = False

    def get_cpu_usage(self) -> list[dict]:
        """Get per-core CPU usage percentages."""
        per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        return [
            {"core": f"CPU {i}", "usage": usage}
            for i, usage in enumerate(per_core)
        ]

    def get_ram_usage(self) -> dict:
        """Get RAM usage information."""
        mem = psutil.virtual_memory()
        return {
            "used": mem.used,
            "total": mem.total,
            "percent": mem.percent,
            "available": mem.available,
        }

    def get_disk_usage(self) -> list[dict]:
        """Get disk usage per mount point."""
        partitions = psutil.disk_partitions()
        disk_info = []
        for partition in partitions:
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_info.append(
                    {
                        "mountpoint": partition.mountpoint,
                        "total": usage.total,
                        "used": usage.used,
                        "free": usage.free,
                        "percent": usage.percent,
                    }
                )
            except PermissionError:
                continue
        # Sort by mountpoint and limit to top 8
        disk_info.sort(key=lambda x: x["mountpoint"])
        return disk_info[:8]

    def get_network_io(self, prev_stats: Optional[dict] = None) -> tuple[dict, dict]:
        """Get network I/O statistics (bytes per second)."""
        if prev_stats is None:
            # Initial call - get baseline and return empty data
            counters = psutil.net_io_counters()
            baseline = {
                "bytes_sent": counters.bytes_sent,
                "bytes_recv": counters.bytes_recv,
                "time": time.time(),
            }
            return {}, {"baseline": baseline, "prev": baseline}

        prev = prev_stats["prev"]
        curr_counters = psutil.net_io_counters()
        curr_time = time.time()
        elapsed = curr_time - prev["time"]
        if elapsed == 0:
            elapsed = 0.1

        bytes_sent = (curr_counters.bytes_sent - prev["bytes_sent"]) / elapsed
        bytes_recv = (curr_counters.bytes_recv - prev["bytes_recv"]) / elapsed

        network_info = {
            "bytes_sent_per_sec": bytes_sent,
            "bytes_recv_per_sec": bytes_recv,
        }
        new_prev = {
            "bytes_sent": curr_counters.bytes_sent,
            "bytes_recv": curr_counters.bytes_recv,
            "time": curr_time,
        }
        new_prev_stats = {"baseline": prev_stats["baseline"], "prev": new_prev}
        return network_info, new_prev_stats

    def get_top_processes(self, limit: int = 10) -> list[dict]:
        """Get top processes by CPU usage."""
        processes = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent"]):
            try:
                info = proc.info
                if info["name"] and info["cpu_percent"] is not None:
                    processes.append(
                        {
                            "pid": info["pid"],
                            "name": info["name"],
                            "cpu_percent": info["cpu_percent"],
                        }
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        processes.sort(key=lambda x: x["cpu_percent"], reverse=True)
        return processes[:limit]

    def format_bytes(self, bytes_val: int) -> str:
        """Format bytes to human-readable string."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if abs(bytes_val) < 1024.0:
                return f"{bytes_val:.1f}{unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.1f}PB"

    def _create_cpu_table(self, cpu_data: list[dict]) -> Table:
        """Create a table for CPU usage."""
        table = Table(title="CPU Usage (per core)", show_header=False, border_style="blue")
        table.add_column("Core", style="cyan")
        table.add_column("Usage", justify="right", style="green")

        for core_info in cpu_data:
            bar_width = 40
            usage = core_info["usage"]
            filled = int(bar_width * usage / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            table.add_row(core_info["core"], f"{usage:5.1f}%  [{bar}]" )

        return table

    def _create_ram_panel(self, ram_data: dict) -> Panel:
        """Create a panel for RAM usage."""
        used_str = self.format_bytes(ram_data["used"])
        total_str = self.format_bytes(ram_data["total"])
        percent = ram_data["percent"]

        bar_width = 40
        filled = int(bar_width * percent / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        ram_text = Text()
        ram_text.append(f"RAM: {used_str} / {total_str}\n", style="cyan")
        ram_text.append(f"[{bar}] ", style="dim")
        ram_text.append(f"{percent:.1f}%", style="green")
        ram_text.append(f"\nAvailable: {self.format_bytes(ram_data['available'])}", style="dim")

        return Panel(ram_text, title="Memory", border_style="magenta")

    def _create_disk_table(self, disk_data: list[dict]) -> Table:
        """Create a table for disk usage."""
        table = Table(title="Disk Usage", border_style="yellow")
        table.add_column("Mountpoint", style="cyan")
        table.add_column("Used", style="green")
        table.add_column("Total", style="dim")
        table.add_column("Free", style="dim")
        table.add_column("Usage", justify="right", style="green")

        for disk in disk_data:
            used_str = self.format_bytes(disk["used"])
            total_str = self.format_bytes(disk["total"])
            free_str = self.format_bytes(disk["free"])
            percent = disk["percent"]
            bar_width = 20
            filled = int(bar_width * percent / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            table.add_row(
                disk["mountpoint"],
                used_str,
                total_str,
                free_str,
                f"{percent:5.1f}%  [{bar}]",
            )

        return table

    def _create_network_panel(self, net_data: dict) -> Panel:
        """Create a panel for network I/O."""
        if not net_data:
            net_text = Text("Waiting for data...", style="dim")
        else:
            sent_str = self.format_bytes(int(net_data["bytes_sent_per_sec"]))
            recv_str = self.format_bytes(int(net_data["bytes_recv_per_sec"]))
            net_text = Text()
            net_text.append("Network I/O\n", style="cyan")
            net_text.append(f"Sent:   {sent_str}/s\n", style="green")
            net_text.append(f"Recv:   {recv_str}/s", style="blue")

        return Panel(net_text, title="Network", border_style="cyan")

    def _create_processes_table(self, processes: list[dict]) -> Table:
        """Create a table for top processes."""
        table = Table(title="Top 10 Processes by CPU", border_style="red")
        table.add_column("PID", justify="right", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("CPU %", justify="right", style="yellow")

        for proc in processes:
            table.add_row(
                str(proc["pid"]),
                proc["name"],
                f"{proc['cpu_percent']:.1f}%",
            )

        return table

    def render_dashboard(self, prev_net_stats: Optional[dict] = None) -> tuple:
        """Render the complete dashboard."""
        from datetime import datetime
        from rich.console import Group
        from rich.layout import Layout

        # Collect data
        cpu_data = self.get_cpu_usage()
        ram_data = self.get_ram_usage()
        disk_data = self.get_disk_usage()
        net_data, new_net_stats = self.get_network_io(prev_net_stats)
        processes = self.get_top_processes()

        # Create layout using rich Layout with proper size syntax
        layout = Layout()
        header = Layout(name="header", size=3)
        main = Layout(name="main")
        left = Layout(name="left")
        right = Layout(name="right")
        
        main.split_row(left, right)
        layout.split(header, main)

        # Header
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header_text = Text(f"System Monitor - {current_time}", style="bold white")
        layout["header"].update(Panel(header_text, border_style="white"))

        # Left panel - CPU and RAM
        cpu_table = self._create_cpu_table(cpu_data)
        ram_panel = self._create_ram_panel(ram_data)
        left_content = Group(cpu_table, ram_panel)
        layout["left"].update(left_content)

        # Right panel - Disk, Network, Processes
        disk_table = self._create_disk_table(disk_data)
        network_panel = self._create_network_panel(net_data)
        proc_table = self._create_processes_table(processes)
        right_content = Group(disk_table, network_panel, proc_table)
        layout["right"].update(right_content)

        return layout, new_net_stats

    def run(self):
        """Run the monitor loop."""
        from rich.console import Console
        
        console = Console(color_system=None if not self.use_color else "auto")
        live = Live(
            console=console,
            refresh_per_second=1.0 / self.interval,
            transient=False,
        )

        prev_net_stats = None
        with live:
            while self.running:
                layout, prev_net_stats = self.render_dashboard(prev_net_stats)
                live.update(layout, refresh=True)
                time.sleep(self.interval)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Terminal-based system monitor dashboard."
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Refresh interval in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable color output (useful for piping)",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    monitor = SystemMonitor(
        interval=args.interval,
        use_color=not args.no_color,
    )

    try:
        monitor.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)


if __name__ == "__main__":
    main()
