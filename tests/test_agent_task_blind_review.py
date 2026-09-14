"""The blinded rubric review: it must blind, and it must not invent scores.

The corpus carries ``human_review`` rubric items that no campaign ever judged —
the only place code quality and decision ownership are described. Judging them
after the fact is cheap, but only worth anything if the judge cannot see which
arm a run came from and if an incomplete score file is refused rather than
silently averaged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.agent_task_blind_review import build_packet, deduplicated_diff, report


def _run(root: Path, arm: str, folder: str, *, answer: str) -> None:
    path = root / arm / "artifacts" / folder / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "task": {"id": folder.split(".r")[0], "request": "do the thing"},
            "verify_exit_code": 0,
            "final_answer": answer,
            "changed_files_summary": "### /repo/src/a.py (modified)\n```diff\n+x\n```",
            "tool_trace": [
                {"tool_name": "read_file", "arguments": {"file_path": "a.py"},
                 "result": "{}"},
                {"tool_name": "execute_command", "arguments": {"command": "pytest"},
                 "result": '{"exit_code": 1}'},
            ],
            "rubric": [
                {"id": "kept", "kind": "human_review", "description": "the real one"},
                {"id": "gated", "kind": "deterministic", "description": "not judged"},
            ],
        }),
        encoding="utf-8",
    )


def _packet(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "campaign"
    _run(root, "arm-a", "task-one.r0.baseline", answer="answer A")
    _run(root, "arm-b", "task-one.r0.baseline", answer="answer B")
    stem = tmp_path / "blind"
    build_packet(root, stem)
    return stem.with_suffix(".review.md"), stem.with_suffix(".key.json")


def test_the_packet_never_names_the_arm(tmp_path: Path) -> None:
    markdown, key_path = _packet(tmp_path)

    text = markdown.read_text(encoding="utf-8")
    assert "arm-a" not in text and "arm-b" not in text
    # The request, the answer and the deterministic items all stay visible.
    assert "do the thing" in text
    assert "answer A" in text and "answer B" in text
    assert "the real one" in text
    assert "not judged" not in text, "only human_review items are judged"


def test_the_key_covers_every_run_and_hides_in_the_other_file(tmp_path: Path) -> None:
    markdown, key_path = _packet(tmp_path)

    key = json.loads(key_path.read_text(encoding="utf-8"))
    assert len(key) == 2
    assert sorted(v["arm"] for v in key.values()) == ["arm-a", "arm-b"]


def test_the_trace_keeps_whether_a_call_failed(tmp_path: Path) -> None:
    """`failure-recognition` asks whether the trace shows a switch away from a
    failed channel, so the outcome has to survive into the packet."""
    markdown, _ = _packet(tmp_path)

    text = markdown.read_text(encoding="utf-8")
    assert "[ failed]" in text
    assert "[unknown]" in text


def test_an_unscored_run_is_refused_rather_than_averaged(tmp_path: Path) -> None:
    _, key_path = _packet(tmp_path)
    key = json.loads(key_path.read_text(encoding="utf-8"))
    first = sorted(key)[0]
    scores = tmp_path / "scores.json"
    scores.write_text(json.dumps({first: {"kept": 2}}), encoding="utf-8")

    with pytest.raises(SystemExit):
        report(key_path, scores)


def test_the_report_means_both_arms_over_the_same_items(tmp_path: Path) -> None:
    _, key_path = _packet(tmp_path)
    key = json.loads(key_path.read_text(encoding="utf-8"))
    scores = {
        opaque: {"kept": 2 if entry["arm"] == "arm-a" else 0}
        for opaque, entry in key.items()
    }
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps(scores), encoding="utf-8")

    result = report(key_path, scores_path)

    assert result["arms"] == ["arm-a", "arm-b"]
    assert result["mean"]["arm-a"] == 2.0
    assert result["mean"]["arm-b"] == 0.0
    assert result["items"][0]["delta"] == -2.0


def test_the_deduplicated_diff_matches_the_metric_side() -> None:
    summary = (
        "### /var/x/repo/a.py (modified)\n```diff\n+x\n```\n\n"
        "### /private/var/x/repo/a.py (modified)\n```diff\n+x\n```"
    )

    assert deduplicated_diff(summary).count("### ") == 1
