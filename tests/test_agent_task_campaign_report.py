from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.agent_task_campaign_report import build_campaign_dossier, render_markdown


def _artifact(path: Path, answer: str) -> str:
    path.write_text(
        json.dumps(
            {
                "verify_exit_code": 0,
                "error": "",
                "forbidden_changes": [],
                "missing_expected_changes": [],
                "final_answer": answer,
                "changed_paths": ["result.txt"],
                "verify_stdout": "ok",
                "verify_stderr": "",
                "tool_trace": [],
                "action_records": [],
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _report(tmp_path: Path, model: str) -> dict[str, object]:
    records = []
    for index in range(6):
        records.append(
            {
                "task": {
                    "id": f"task-{index}",
                    "category": f"category-{index}",
                    "request": "Do the task.",
                    "rubric": [],
                },
                "repetition": 0,
                "baseline": {
                    "run_artifact": _artifact(tmp_path / f"{model}-{index}-b.json", "baseline")
                },
                "candidate": {
                    "run_artifact": _artifact(tmp_path / f"{model}-{index}-c.json", "candidate")
                },
                "success_delta": 0,
                "candidate_changed_behavior": True,
                "tool_call_delta": 1,
                "latency_delta_seconds": 2.0,
            }
        )
    return {
        "provider": "provider",
        "model": model,
        "model_identity": f"provider:{model}:snapshot",
        "paired_repetitions": 6,
        "paired_outcomes": {"candidate_improvements": 0, "candidate_regressions": 0},
        "conditions": {
            "baseline": {
                "mean_latency_seconds": 10,
                "mean_tool_calls": 2,
                "prompt_tokens": 100,
                "completion_tokens": 20,
            },
            "candidate": {
                "mean_latency_seconds": 12,
                "mean_tool_calls": 3,
                "prompt_tokens": 120,
                "completion_tokens": 22,
            },
        },
        "task_records": records,
    }


def test_campaign_dossier_retains_all_36_executions_and_raw_answers(tmp_path: Path) -> None:
    plan = {"planned_executions": 36, "routes": [{}, {}, {}]}
    reports = [_report(tmp_path, model) for model in ("sol", "terra", "luna")]

    dossier = build_campaign_dossier(plan, reports)

    assert dossier["observed_executions"] == 36
    assert dossier["models"][0]["category_maps"][0]["candidate"]["final_answer"] == "candidate"
    assert dossier["models"][0]["efficiency"]["tool_call_ratio"] == 1.5
    markdown = render_markdown(dossier)
    assert "observable decisions" in dossier["interpretation_boundary"]
    assert "category-0" in markdown
    assert "candidate" in markdown


def test_campaign_dossier_rejects_incomplete_campaign(tmp_path: Path) -> None:
    plan = {"planned_executions": 36, "routes": [{}, {}, {}]}

    with pytest.raises(ValueError, match="three completed"):
        build_campaign_dossier(plan, [_report(tmp_path, "sol")])
