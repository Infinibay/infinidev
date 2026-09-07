"""Console-script entry point.

Routes ``infinidev web ...`` (also ``serve``) to the browser backend before importing
the Click CLI (which pulls in ``prompt_toolkit`` and reconfigures
logging at import time — neither of which the server wants). Every other
invocation falls through to the existing TUI / classic CLI unchanged.
"""

from __future__ import annotations

import sys


def main() -> None:
    if sys.argv[1:2] and sys.argv[1] in {"serve", "web"}:
        from infinidev.server.cli import run_from_argv

        run_from_argv(sys.argv[2:])
        return
    from infinidev.cli.main import main as _main

    _main()
