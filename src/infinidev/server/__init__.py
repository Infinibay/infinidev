"""Local HTTP/WebSocket frontend for the shared Infinidev task pipeline."""

from __future__ import annotations


def create_app(*args, **kwargs):
    """Import the optional server dependencies only when requested."""
    from infinidev.server.app import create_app as factory

    return factory(*args, **kwargs)


def run_server(*args, **kwargs):
    """Launch the browser workspace."""
    from infinidev.server.cli import run_server as run

    return run(*args, **kwargs)
