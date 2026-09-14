"""HTTP handlers for the record service."""

from __future__ import annotations

from src.storage import load_records, save_records


def list_records(path: str) -> list[dict]:
    """Return every stored record."""
    return load_records(path)


def create_record(path: str, payload: dict) -> dict:
    """Append ``payload`` to the store and return it."""
    records = load_records(path)
    records.append(payload)
    save_records(path, records)
    return payload
