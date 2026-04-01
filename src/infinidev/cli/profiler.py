"""Session profiler for Infinidev.

Wraps yappi to capture multi-threaded wall-clock profiles across all
threads (Textual UI loop, engine workers, FileWatcher, IndexQueue).

Usage::

    with SessionProfiler(enabled=True) as profiler:
        app.run()
    print(profiler.report_path)
"""

from __future__ import annotations

import io
import time
from datetime import datetime
from pathlib import Path

PROFILE_DIR = Path.home() / ".infinidev" / "profiles"

# Packages we want to see in the report (everything else from
# site-packages is filtered out to reduce noise).
_RELEVANT_PACKAGES = ("infinidev", "textual", "litellm")


def _is_relevant(full_name: str) -> bool:
    """Return True if a function path is worth showing in the report."""
    if "site-packages" not in full_name:
        return True  # user code, builtins, stdlib — keep
    return any(pkg in full_name for pkg in _RELEVANT_PACKAGES)


class SessionProfiler:
    """Context manager that profiles the entire application session.

    When *enabled* is False, all methods are no-ops so callers can
    wrap unconditionally without branching.
    """

    def __init__(self, enabled: bool = False, clock_type: str = "wall") -> None:
        self.enabled = enabled
        self.clock_type = clock_type
        self.report_path: Path | None = None
        self._start_time: float = 0.0

    # ── Context manager protocol ────────────────────────────

    def __enter__(self) -> SessionProfiler:
        if self.enabled:
            self.start()
        return self

    def __exit__(self, *exc) -> None:
        if self.enabled:
            self.stop()
            self.report_path = self.save_report()

    # ── Core API ────────────────────────────────────────────

    def start(self) -> None:
        import yappi

        yappi.set_clock_type(self.clock_type)
        yappi.start(builtins=False, profile_threads=True)
        self._start_time = time.monotonic()

    def stop(self) -> None:
        import yappi

        yappi.stop()

    def save_report(self) -> Path:
        """Write a human-readable ``.txt`` and a pstat ``.prof`` file.

        Returns the path to the ``.txt`` report.
        """
        import yappi

        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        txt_path = PROFILE_DIR / f"profile_{stamp}.txt"
        prof_path = PROFILE_DIR / f"profile_{stamp}.prof"

        elapsed = time.monotonic() - self._start_time
        func_stats = yappi.get_func_stats()
        thread_stats = yappi.get_thread_stats()

        # ── Text report ─────────────────────────────────────
        buf = io.StringIO()
        buf.write("Infinidev Session Profile\n")
        buf.write("=" * 50 + "\n")
        buf.write(f"Clock type : {self.clock_type}\n")
        buf.write(f"Duration   : {elapsed:.1f}s\n")
        buf.write(f"Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Thread summary
        buf.write("Thread Summary\n")
        buf.write("-" * 50 + "\n")
        for t in thread_stats:
            buf.write(f"  {t.name:<30s} {t.ttot:>8.2f}s\n")
        buf.write("\n")

        # Filter to relevant functions
        relevant = [s for s in func_stats if _is_relevant(s.full_name)]

        # Top N by cumulative time
        by_ttot = sorted(relevant, key=lambda s: s.ttot, reverse=True)[:30]
        buf.write("Top 30 by Cumulative Time\n")
        buf.write("-" * 50 + "\n")
        buf.write(f"  {'function':<60s} {'ncall':>7s} {'ttot':>9s} {'tsub':>9s}\n")
        for s in by_ttot:
            name = _short_name(s.full_name, 60)
            buf.write(f"  {name:<60s} {s.ncall:>7d} {s.ttot:>8.3f}s {s.tsub:>8.3f}s\n")
        buf.write("\n")

        # Top N by self time
        by_tsub = sorted(relevant, key=lambda s: s.tsub, reverse=True)[:30]
        buf.write("Top 30 by Self Time\n")
        buf.write("-" * 50 + "\n")
        buf.write(f"  {'function':<60s} {'ncall':>7s} {'tsub':>9s} {'ttot':>9s}\n")
        for s in by_tsub:
            name = _short_name(s.full_name, 60)
            buf.write(f"  {name:<60s} {s.ncall:>7d} {s.tsub:>8.3f}s {s.ttot:>8.3f}s\n")

        txt_path.write_text(buf.getvalue())

        # ── pstat file for snakeviz / flamegraph ────────────
        func_stats.save(str(prof_path), type="pstat")

        # Clean up yappi state for a potential next run
        yappi.clear_stats()

        return txt_path


def _short_name(full_name: str, max_len: int) -> str:
    """Shorten a yappi full_name to fit *max_len* characters."""
    if len(full_name) <= max_len:
        return full_name
    # Keep the tail (function name) which is the most useful part
    return "..." + full_name[-(max_len - 3):]
