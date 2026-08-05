#!/usr/bin/env python3
"""Select immutable observation subsets for reproducible downstream reports."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from bench.model_behavior import Observation, load_observations


def select_observations(
    rows: Iterable[Observation],
    *,
    repetitions: Iterable[int] = (),
    protocols: Iterable[str] = (),
    probe_ids: Iterable[str] = (),
) -> list[Observation]:
    """Return rows matching every non-empty selector, preserving source order."""
    repetition_set = set(repetitions)
    protocol_set = set(protocols)
    probe_id_set = set(probe_ids)
    return [
        row
        for row in rows
        if (not repetition_set or row.repetition in repetition_set)
        and (not protocol_set or row.elicitation_protocol in protocol_set)
        and (not probe_id_set or row.probe_id in probe_id_set)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repetition", action="append", type=int, default=[])
    parser.add_argument(
        "--protocol", action="append", choices=("choice_only", "self_report"), default=[]
    )
    parser.add_argument("--probe-id", action="append", default=[])
    args = parser.parse_args()
    selected = select_observations(
        load_observations(args.source),
        repetitions=args.repetition,
        protocols=args.protocol,
        probe_ids=args.probe_id,
    )
    if not selected:
        parser.error("selection is empty")
    args.output.write_text(
        "".join(json.dumps(asdict(row), ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
