"""The answer-quality proxies, and the honesty of their limits."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from bench.agent_task_answer_quality import answer_quality, compare


def test_a_checkable_answer_passes_every_check() -> None:
    quality = answer_quality(
        "Fixed the rounding in src/pricing.py.\n\n"
        "Verification: `python -m pytest -q` -> 3 passed.\n\n"
        "Nothing is left undone."
    )

    assert quality["names_a_command"] is True
    assert quality["states_a_result"] is True
    assert quality["has_verification_line"] is True
    assert quality["opens_with_narration"] is False
    assert quality["over_word_budget"] is False


def test_an_uncheckable_answer_fails_them() -> None:
    quality = answer_quality("I will now fix the rounding and report back.")

    assert quality["names_a_command"] is False
    assert quality["has_verification_line"] is False
    assert quality["opens_with_narration"] is True


def test_the_word_budget_matches_the_contract() -> None:
    assert answer_quality(" ".join(["word"] * 250))["over_word_budget"] is False
    assert answer_quality(" ".join(["word"] * 251))["over_word_budget"] is True


def _arm(root: Path, name: str, answers: list[tuple[str, int, str]]) -> Path:
    arm = root / name
    artifacts = arm / "artifacts"
    artifacts.mkdir(parents=True)
    rows = []
    for index, (task_id, repetition, answer) in enumerate(answers):
        run = artifacts / f"{task_id}.r{repetition}"
        run.mkdir()
        artifact = run / "run.json"
        artifact.write_text(json.dumps({"final_answer": answer}), encoding="utf-8")
        rows.append({
            "task_id": task_id, "repetition": repetition, "condition": "baseline",
            "run_artifact": str(artifact),
        })
    path = arm / "observations.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8",
    )
    return path


def test_the_report_pairs_the_two_arms() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        a = _arm(root, "a", [("t", 0, "Fixed it. Nothing else.")])
        b = _arm(
            root, "b",
            [("t", 0, "Fixed it.\n\nVerification: `pytest -q` -> 3 passed.")],
        )

        report = compare(a, "default", b, "lean")

    assert "Answer quality: default vs lean" in report
    assert "| has_verification_line | 0/1 | 1/1 |" in report
    assert "proxies" in report, "the report must state what these checks are not"


def test_an_empty_final_answer_scores_as_uncheckable() -> None:
    quality = answer_quality("")

    assert quality["names_a_command"] is False
    assert quality["states_a_result"] is False
    assert quality["words"] == 0
