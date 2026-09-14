"""Durable record storage."""

from __future__ import annotations

import json
import os


def load_records(path: str) -> list[dict]:
    """Read every record from ``path``, or none if it cannot be read."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    except Exception:
        return []


def save_records(path: str, records: list[dict]) -> None:
    """Write ``records`` to ``path``, one JSON object per line."""
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + os.linesep)
