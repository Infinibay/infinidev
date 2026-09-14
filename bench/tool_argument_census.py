"""Measure what eliding closed tool-call arguments would actually remove.

The claim this checks is a real one and was first made as an estimate: on a task
that writes files, the tool-call arguments are a large share of every request
because a written file's body *is* the argument. `trim_superseded_tool_arguments`
elides long string values inside the calls whose Step already closed. Two things
follow that no amount of reasoning settles:

* **How often it can fire at all.** If real calls rarely carry a long string, the
  lever is theoretical.
* **Whether it ever has to bail out.** The rule refuses to rewrite a call whose
  arguments do not parse — a malformed call is a fact about the run — so a
  corpus with unparseable arguments is a corpus where part of the saving does not
  exist.

Both are counted here from the stored `tool_trace` of every run, which records
each call's name and parsed arguments. The result is the exact elidable share of
the argument bytes, not an assumption about it.

Usage::

    python -m bench.tool_argument_census --runs bench/runs
    python -m bench.tool_argument_census --runs bench/runs --json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

from infinidev.engine.loop.tool_argument_digest import (
    _ELIDE_OVER_CHARS,
    digest_arguments,
)

#: Calls at least this large are the ones whose arguments travel as payload
#: rather than as identity; the aggregate share is reported over these as well
#: as over everything, because the two answer different questions.
_PAYLOAD_CALL_CHARS = 500


def census(run_root: Path) -> dict[str, Any]:
    calls = 0
    unparseable = 0
    touched = 0
    bytes_total = 0
    bytes_elidable = 0
    payload_calls: list[tuple[int, int]] = []
    by_tool: dict[str, dict[str, int]] = {}

    for path in sorted(glob.glob(str(run_root / "**" / "run.json"), recursive=True)):
        try:
            raw = json.loads(Path(path).read_text())
        except Exception:
            continue
        for entry in raw.get("tool_trace") or []:
            if not isinstance(entry, dict):
                continue
            arguments = entry.get("arguments")
            if not isinstance(arguments, dict):
                continue
            name = str(entry.get("tool_name") or "?")
            encoded = json.dumps(
                arguments, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            )
            calls += 1
            bytes_total += len(encoded)
            try:
                json.loads(encoded)
            except ValueError:
                unparseable += 1
            _, saved = digest_arguments(encoded)
            if saved:
                touched += 1
                bytes_elidable += saved
            if len(encoded) >= _PAYLOAD_CALL_CHARS:
                payload_calls.append((len(encoded), saved))
            bucket = by_tool.setdefault(name, {"calls": 0, "bytes": 0, "elidable": 0})
            bucket["calls"] += 1
            bucket["bytes"] += len(encoded)
            bucket["elidable"] += saved

    payload_bytes = sum(size for size, _ in payload_calls)
    payload_elidable = sum(saved for _, saved in payload_calls)
    return {
        "calls": calls,
        "unparseable_calls": unparseable,
        "calls_touched": touched,
        "calls_touched_pct": 100.0 * touched / calls if calls else 0.0,
        "argument_bytes": bytes_total,
        "elidable_bytes": bytes_elidable,
        "elidable_pct": 100.0 * bytes_elidable / bytes_total if bytes_total else 0.0,
        "payload_calls": len(payload_calls),
        "payload_argument_bytes": payload_bytes,
        "payload_elidable_bytes": payload_elidable,
        "payload_elidable_pct": (
            100.0 * payload_elidable / payload_bytes if payload_bytes else 0.0
        ),
        "elide_over_chars": _ELIDE_OVER_CHARS,
        "by_tool": dict(
            sorted(by_tool.items(), key=lambda kv: -kv[1]["elidable"])
        ),
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "=" * 78,
        "What eliding closed tool-call arguments would remove",
        "=" * 78,
        f"real tool calls                    {report['calls']:,}",
        f"  that do not parse                {report['unparseable_calls']}"
        f"   (the rule refuses to rewrite these)",
        f"  carrying a value over {report['elide_over_chars']} chars"
        f"   {report['calls_touched']:,} ({report['calls_touched_pct']:.1f} %)",
        "",
        f"argument bytes                     {report['argument_bytes']:,}",
        f"  elidable                         {report['elidable_bytes']:,}"
        f" ({report['elidable_pct']:.1f} %)",
        "",
        f"calls of {_PAYLOAD_CALL_CHARS}+ chars (the ones that are payload)",
        f"  count                            {report['payload_calls']:,}",
        f"  their argument bytes             {report['payload_argument_bytes']:,}",
        f"  elidable                         {report['payload_elidable_bytes']:,}"
        f" (**{report['payload_elidable_pct']:.1f} %**)",
        "",
        "where the bytes are",
    ]
    for name, bucket in list(report["by_tool"].items())[:8]:
        lines.append(
            f"  {name[:26]:26s} calls={bucket['calls']:5,d}"
            f"  bytes={bucket['bytes']:10,d}  elidable={bucket['elidable']:10,d}"
        )
    lines += [
        "",
        "Read it as: the rule fires on a minority of calls and removes most of",
        "the bytes, because the calls it fires on are the ones that carry a file",
        "body. The payload effect of that is measured per run by",
        "`measure_request_payload`'s `tool_call_argument_chars`, which the",
        "comparison reports as part of the request.",
        "=" * 78,
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("bench/runs"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = census(args.runs)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render(report))


if __name__ == "__main__":
    main()
