"""Read host session metadata without initializing the interactive CLI or database."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from infinidev.engine.session_status import read_status


def run_from_argv(argv: list[str] | None = None) -> None:
    """Print one JSON observation; exit zero also includes unknown observations."""
    parser = argparse.ArgumentParser(prog="infinidev status")
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--session", required=True)
    parser.add_argument("--turn", help="Expected parent host_turn_id")
    parser.add_argument("--run", help="Expected child runtime run_id")
    args = parser.parse_args(argv)
    print(json.dumps(read_status(args.workspace, args.session, expected_turn=args.turn,
                                 expected_run=args.run), ensure_ascii=True))


if __name__ == "__main__":
    run_from_argv()
