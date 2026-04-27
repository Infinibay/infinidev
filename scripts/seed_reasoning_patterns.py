"""Seed (or re-seed) the ``critic_reasoning_patterns`` table.

Reads ``src/infinidev/engine/loop/reasoning_patterns_seed.jsonl``,
computes embeddings for each signature via
:func:`infinidev.tools.base.embeddings.compute_embedding`, and upserts
each row into the ``critic_reasoning_patterns`` table.

Idempotent — the table has ``UNIQUE(pattern_name, signature)`` so a
second run replaces the embedding/threshold/question rather than
duplicating rows. Useful when iterating on the seed file or when the
embedding backend has changed.

Usage::

    uv run python scripts/seed_reasoning_patterns.py
    uv run python scripts/seed_reasoning_patterns.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

# Ensure ``src`` is importable when running directly from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from infinidev.config.settings import get_db_path  # noqa: E402
from infinidev.db.service import init_db  # noqa: E402
from infinidev.tools.base.embeddings import compute_embedding  # noqa: E402

SEED_PATH = (
    _REPO_ROOT / "src" / "infinidev" / "engine" / "loop" /
    "reasoning_patterns_seed.jsonl"
)


def _load_seed(path: Path) -> list[dict]:
    rows: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
    return rows


def _validate(row: dict, lineno: int) -> None:
    required = {"pattern_name", "signature", "threshold", "socratic_question"}
    missing = required - row.keys()
    if missing:
        raise ValueError(f"line {lineno}: missing fields {missing}")
    if not isinstance(row["threshold"], (int, float)):
        raise ValueError(f"line {lineno}: threshold must be numeric")
    if not 0.0 <= row["threshold"] <= 1.0:
        raise ValueError(f"line {lineno}: threshold {row['threshold']} not in [0,1]")


def seed(dry_run: bool = False) -> int:
    """Seed the table. Returns number of rows upserted."""
    if not SEED_PATH.exists():
        raise FileNotFoundError(f"seed file not found: {SEED_PATH}")

    rows = _load_seed(SEED_PATH)
    for i, row in enumerate(rows, 1):
        _validate(row, i)

    init_db()
    db_path = get_db_path()
    upserted = 0

    if dry_run:
        for row in rows:
            print(f"[dry-run] would upsert {row['pattern_name']!r}: "
                  f"{row['signature']!r}")
        return len(rows)

    conn = sqlite3.connect(db_path)
    try:
        for row in rows:
            emb = compute_embedding(row["signature"])
            if emb is None:
                logging.warning(
                    "embedding failed for signature %r; skipping",
                    row["signature"],
                )
                continue
            conn.execute(
                """
                INSERT INTO critic_reasoning_patterns
                    (pattern_name, signature, embedding, threshold,
                     socratic_question, triggered_check)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(pattern_name, signature) DO UPDATE SET
                    embedding = excluded.embedding,
                    threshold = excluded.threshold,
                    socratic_question = excluded.socratic_question,
                    triggered_check = excluded.triggered_check
                """,
                (
                    row["pattern_name"],
                    row["signature"],
                    emb,
                    float(row["threshold"]),
                    row["socratic_question"],
                    row.get("triggered_check"),
                ),
            )
            upserted += 1
        conn.commit()
    finally:
        conn.close()

    return upserted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the seed file and report what would be inserted, "
             "without touching the DB.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    n = seed(dry_run=args.dry_run)
    verb = "would seed" if args.dry_run else "seeded"
    print(f"{verb} {n} reasoning pattern signatures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
