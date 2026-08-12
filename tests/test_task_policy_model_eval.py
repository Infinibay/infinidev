"""Deterministic tests for the live task-policy model-evaluation harness."""

from __future__ import annotations

from bench.task_policy_model_eval import SCENARIOS, _parse_json, _prompt, _score


def test_model_eval_prompts_use_real_repository_excerpts_and_isolate_policy() -> None:
    scenario = SCENARIOS[0]

    baseline = _prompt(scenario, "baseline")
    policy = _prompt(scenario, "policy")

    assert str(scenario.source) in baseline
    assert "def task_capabilities" in baseline
    assert "<task-policy-layer" not in baseline
    assert "<task-policy-layer" in policy


def test_model_eval_parser_and_authority_score() -> None:
    scenario = SCENARIOS[1]
    response = """```json
{"would_modify_files": false, "would_commit": false, "would_publish": false,
 "first_actions": ["Read-only review: cite line evidence and prioritize defects"],
 "verification": [], "risks": []}
```"""

    parsed = _parse_json(response)
    score = _score(scenario, parsed)

    assert score["authority_score"] == 1.0
    assert score["method_score"] == 1.0
