"""Score the ``human_review`` rubric items nobody ever scored.

Every campaign in this project compared deterministic verifiers, tokens and
latency. The corpus also carries 289 stored runs whose rubrics include
``human_review`` items — "the fix corrects the constant in the stage that is
wrong rather than compensating in the pipeline", "consequential decisions are
surfaced with a recommendation instead of silently decided" — and not one of
them was ever judged. Those items are the only place code quality, directness
and decision ownership live, so an engine comparison that skips them can
report a large token win and say nothing about whether the work got worse.

This tool makes that judging possible without another model call, and makes it
**blind**: it strips the arm from each run, gives it an opaque id, shuffles,
and writes the key to a separate file that is not opened until the scores are
in. Judging your own change while knowing which arm a diff came from is the
one way to make this worthless.

Usage::

    # 1. Build the packet (writes .review.md, .review.json and .key.json)
    python -m bench.agent_task_blind_review packet \\
        bench/runs/<date>-<name> --output bench/runs/<date>-<name>/blind

    # 2. Read the markdown, write scores.json: {"<opaque-id>": {"<item>": 0..2}}

    # 3. Join and report
    python -m bench.agent_task_blind_review report \\
        bench/runs/<date>-<name>/blind.key.json scores.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

#: How much of each artifact reaches the reviewer. The corpus tasks are small
#: enough that these caps do not truncate a real answer; they exist so a
#: runaway run cannot bury the packet.
_MAX_ANSWER_CHARS = 1_800
_MAX_DIFF_CHARS = 2_600
_MAX_TRACE_STEPS = 40

#: 0 = the criterion is not met, 1 = partially, 2 = met. Three points, because
#: finer granularity would be false precision from a single judge.
SCALE = {0: "not met", 1: "partially", 2: "met"}

_DIFF_HEADER = re.compile(r"^### (?P<path>.+?) \((?P<action>[^)]*)\)$")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def deduplicated_diff(summary: str) -> str:
    """One section per file, not per path spelling (see §6.1.16)."""

    if not summary:
        return ""
    kept: list[str] = []
    seen: set[str] = set()
    for part in summary.split("\n\n"):
        lines = part.splitlines()
        header = lines[0] if lines else ""
        match = _DIFF_HEADER.match(header)
        if match is None:
            kept.append(part)
            continue
        identity = os.path.realpath(match.group("path"))
        if identity in seen:
            continue
        seen.add(identity)
        kept.append(part)
    return "\n\n".join(kept)


def _trace(artifact: dict[str, Any]) -> list[dict[str, str]]:
    """Model-visible tool activity: name, one-line arguments, and outcome.

    The outcome matters: ``failure-recognition`` asks whether the trace shows a
    failed evidence channel and a switch away from it, so a reviewer needs to
    see which calls failed, not only which ran.
    """

    steps: list[dict[str, str]] = []
    for entry in (artifact.get("tool_trace") or [])[:_MAX_TRACE_STEPS]:
        name = str(entry.get("tool_name") or "?")
        args = entry.get("arguments")
        if isinstance(args, dict):
            summary = ", ".join(
                f"{key}={str(value)[:48]}" for key, value in args.items()
            )
        else:
            summary = str(args)[:90]
        steps.append({
            "tool": name,
            "arguments": summary[:120],
            "outcome": _outcome(entry.get("result")),
        })
    return steps


def _outcome(result: Any) -> str:
    """``ok`` / ``failed`` / ``unknown`` for one recorded tool result."""

    if not isinstance(result, str):
        return "unknown"
    lowered = result.lower()
    if '"success": false' in lowered or '"exit_code": 1' in lowered:
        return "failed"
    if '"success": true' in lowered or '"exit_code": 0' in lowered:
        return "ok"
    return "unknown"


def build_packet(
    arm_root: Path,
    output_stem: Path,
    *,
    seed: int = 20_260_914,
    repetitions: set[int] | None = None,
    only_items: set[str] | None = None,
    narrow: bool = False,
) -> dict[str, Any]:
    """Write a blinded review packet plus the key that unblinds it."""

    artifacts: list[dict[str, Any]] = []
    for run_path in sorted(arm_root.rglob("run.json")):
        artifact = _load(run_path)
        items = [
            {"id": item["id"], "description": item["description"]}
            for item in (artifact.get("rubric") or [])
            if item.get("kind") == "human_review"
            and (only_items is None or item["id"] in only_items)
        ]
        if not items:
            continue
        # ``<root>/<campaign>[/<arm>]/artifacts/<run>/run.json``; keep the
        # campaign too, so a packet built over many campaigns still separates
        # them when the key is read.
        arm = run_path.parent.parent.parent.relative_to(arm_root).as_posix()
        if arm == ".":
            arm = arm_root.name
        folder = run_path.parent.name  # <task>.r<rep>.baseline
        task = folder.split(".r")[0]
        repetition = int(folder.split(".r")[1].split(".")[0])
        if repetitions is not None and repetition not in repetitions:
            continue
        artifacts.append({
            "arm": arm,
            "task": task,
            "repetition": repetition,
            "request": artifact["task"].get("request", ""),
            "items": items,
            "verify_exit_code": artifact.get("verify_exit_code"),
            "answer": str(artifact.get("final_answer") or "")[:_MAX_ANSWER_CHARS],
            "diff": deduplicated_diff(
                str(artifact.get("changed_files_summary") or "")
            )[:_MAX_DIFF_CHARS],
            "trace": [] if narrow else _trace(artifact),
            "commands": [
                str((call.get("arguments") or {}).get("command") or "")
                for call in (artifact.get("tool_trace") or [])
                if str(call.get("tool_name")) == "execute_command"
                and isinstance(call.get("arguments"), dict)
            ] if narrow else [],
        })

    if not artifacts:
        raise SystemExit(f"no run with a human_review rubric under {arm_root}")

    rng = random.Random(seed)
    rng.shuffle(artifacts)
    key: dict[str, dict[str, Any]] = {}
    packet: list[dict[str, Any]] = []
    for index, entry in enumerate(artifacts):
        opaque = f"R{index:02d}-{rng.randrange(0x1000, 0xFFFF):04x}"
        key[opaque] = {
            "arm": entry["arm"],
            "task": entry["task"],
            "repetition": entry["repetition"],
        }
        packet.append({"id": opaque, **{k: v for k, v in entry.items() if k != "arm"}})

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    (output_stem.with_suffix(".review.json")).write_text(
        json.dumps(packet, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_stem.with_suffix(".key.json")).write_text(
        json.dumps(key, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_stem.with_suffix(".review.md")).write_text(
        render_packet(packet), encoding="utf-8"
    )
    return {"runs": len(packet), "arms": sorted({v["arm"] for v in key.values()})}


def render_packet(packet: list[dict[str, Any]]) -> str:
    """The reviewer's reading copy: one section per run, no arm anywhere."""

    lines = [
        "# Blinded code-quality review",
        "",
        "Score every item on this scale and write the numbers into `scores.json`",
        "as `{\"<id>\": {\"<item>\": 0|1|2}}`:",
        "",
        "  * **0** — the criterion is not met",
        "  * **1** — partially met",
        "  * **2** — met",
        "",
        "The arm each run came from is in the key file, which is not opened until",
        "every score is written. Nothing else here identifies it.",
        "",
    ]
    for entry in packet:
        lines.append(f"## {entry['id']} · {entry['task']} · r{entry['repetition']}")
        lines.append("")
        lines.append(f"**Request.** {entry['request']}")
        lines.append("")
        lines.append(f"**Verifier exit code.** `{entry['verify_exit_code']}`")
        lines.append("")
        lines.append("**Items to score.**")
        for item in entry["items"]:
            lines.append(f"- `{item['id']}` — {item['description']}")
        lines.append("")
        lines.append("**Final answer.**")
        lines.append("")
        lines.append("```")
        lines.append(entry["answer"] or "(empty)")
        lines.append("```")
        lines.append("")
        if entry.get("commands"):
            lines.append("**Commands run.**")
            lines.append("")
            lines.append("```")
            lines.extend(command or "(empty)" for command in entry["commands"])
            lines.append("```")
            lines.append("")
            lines.append("**Tool sequence.**")
            lines.append("")
            lines.append("```")
            lines.append(f"({len(entry['trace'])} steps, omitted in narrow mode)")
            lines.append("```")
            lines.append("")
            lines.append("**Diff.**")
            lines.append("")
            lines.append("```diff")
            lines.append("(omitted in narrow mode)")
            lines.append("```")
            lines.append("")
            continue
        lines.append("**Tool sequence.**")
        lines.append("")
        lines.append("```")
        for step in entry["trace"]:
            lines.append(f"[{step['outcome']:>7s}] {step['tool']}({step['arguments']})")
        if not entry["trace"]:
            lines.append("(none)")
        lines.append("```")
        lines.append("")
        lines.append("**Diff.**")
        lines.append("")
        lines.append("```diff")
        lines.append(entry["diff"] or "(no diff)")
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def report(key_path: Path, scores_path: Path) -> dict[str, Any]:
    """Join the scores with the key and compare the arms item by item."""

    key = _load(key_path)
    scores = _load(scores_path)
    missing = sorted(set(key) - set(scores))
    if missing:
        raise SystemExit(f"unscored runs: {', '.join(missing)}")

    by_arm: dict[str, list[float]] = {}
    per_item: dict[str, dict[str, list[float]]] = {}
    for opaque, entry in sorted(key.items()):
        values = scores[opaque]
        arm = entry["arm"]
        by_arm.setdefault(arm, [])
        for item, value in values.items():
            score = float(value)
            by_arm[arm].append(score)
            per_item.setdefault(item, {}).setdefault(arm, []).append(score)

    arms = sorted(by_arm)
    if len(arms) != 2:
        raise SystemExit(f"need exactly two arms, found {arms}")

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    items = []
    for item, armscores in sorted(per_item.items()):
        a, b = mean(armscores.get(arms[0], [])), mean(armscores.get(arms[1], []))
        items.append({
            "item": item,
            "n": len(armscores.get(arms[0], [])),
            arms[0]: a,
            arms[1]: b,
            "delta": b - a,
        })

    return {
        "arms": arms,
        "runs": {arm: len(values) for arm, values in by_arm.items()},
        "max_score": 2,
        "mean": {arm: mean(values) for arm, values in by_arm.items()},
        "items": items,
    }


def render_report(result: dict[str, Any]) -> str:
    a, b = result["arms"]
    lines = [
        f"# Blinded rubric review: {a} vs {b}",
        "",
        f"Runs judged: {result['runs'].get(a, 0)} vs {result['runs'].get(b, 0)}. "
        f"Scale 0–{result['max_score']} per item, single blinded judge.",
        "",
        f"Mean score: **{a} {result['mean'][a]:.2f}** vs **{b} {result['mean'][b]:.2f}**",
        "",
        f"| item | n | {a} | {b} | delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result["items"]:
        lines.append(
            f"| `{row['item']}` | {row['n']} | {row[a]:.2f} | {row[b]:.2f} | "
            f"{row['delta']:+.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    packet = sub.add_parser("packet", help="build a blinded review packet")
    packet.add_argument("arm_root", type=Path)
    packet.add_argument("--output", type=Path, required=True)
    packet.add_argument("--seed", type=int, default=20_260_914)
    packet.add_argument(
        "--repetition", type=int, action="append", default=None,
        help="score only this repetition (repeatable); default is every one",
    )
    packet.add_argument(
        "--item", action="append", default=None,
        help="only runs carrying this rubric item (repeatable)",
    )
    packet.add_argument(
        "--narrow", action="store_true",
        help="answer plus commands only; for items that need no diff or trace",
    )

    join = sub.add_parser("report", help="join scores with the key and compare")
    join.add_argument("key", type=Path)
    join.add_argument("scores", type=Path)
    join.add_argument("--markdown", type=Path, default=None)
    join.add_argument("--json", dest="json_out", type=Path, default=None)

    args = parser.parse_args()
    if args.command == "packet":
        info = build_packet(
            args.arm_root,
            args.output,
            seed=args.seed,
            repetitions=set(args.repetition) if args.repetition else None,
            only_items=set(args.item) if args.item else None,
            narrow=args.narrow,
        )
        print(json.dumps(info, indent=2))
        print(f"wrote {args.output.with_suffix('.review.md')}")
        return

    result = report(args.key, args.scores)
    markdown = render_report(result)
    print(markdown)
    if args.markdown:
        args.markdown.write_text(markdown, encoding="utf-8")
    if args.json_out:
        args.json_out.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
