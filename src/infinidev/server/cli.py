"""Import-light entry point for the local browser workspace."""

from __future__ import annotations

import argparse
import os
import secrets
import sys
import threading
import webbrowser
from pathlib import Path


def run_server(host: str = "127.0.0.1", port: int = 8765,
               workdir: str | None = None, open_browser: bool = True) -> None:
    """Launch one local workspace, with a capability URL for browser access."""
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("The web harness currently supports loopback hosts only.")
    if workdir:
        os.chdir(Path(workdir).expanduser().resolve())
    try:
        import uvicorn
        from infinidev.server.app import create_app
    except ImportError as exc:
        raise SystemExit(
            "Install web dependencies: uv sync --extra web "
            "(or pip install 'infinidev[web]')"
        ) from exc
    token = secrets.token_urlsafe(32)
    url_host = "[::1]" if host == "::1" else host
    url = f"http://{url_host}:{port}/#token={token}"

    def ready() -> None:
        print(f"\nInfinidev workspace: {Path.cwd()}\nOpen: {url}\n", file=sys.stderr)
        if open_browser:
            timer = threading.Timer(0.3, webbrowser.open, args=(url,))
            timer.daemon = True
            timer.start()

    app = create_app(token=token, on_ready=ready)
    uvicorn.run(app, host=host, port=port, log_level="info")


def run_from_argv(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="infinidev web", description="Open the web harness.")
    parser.add_argument("--host", default="127.0.0.1", help="Loopback bind host.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workdir", help="Project root (defaults to the current directory).")
    parser.add_argument("--no-open", action="store_true", help="Print the URL without opening it.")
    args = parser.parse_args(argv)
    run_server(args.host, args.port, args.workdir, not args.no_open)


def main() -> None:
    run_from_argv(sys.argv[1:])
