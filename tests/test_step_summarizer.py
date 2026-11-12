"""Regression tests for structured loop-step summaries."""

from __future__ import annotations

import json
from types import SimpleNamespace

import litellm

from infinidev.config import prompt_cache
from infinidev.engine.loop.models import LoopState, StepResult
from infinidev.engine.loop.step_summarizer import _summarize_step


def _completion_response(payload):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(payload))
            )
        ]
    )


def test_summarizer_rejects_wrong_json_field_types(monkeypatch):
    payload = {
        "summary": None,
        "files_to_preload": "src/infinidev/engine.py",
        "changes_made": ["not", "text"],
        "discovered": 7,
        "pending": "  add regression coverage  ",
        "anti_patterns": {"retry": "loop"},
    }
    monkeypatch.setattr(
        litellm,
        "completion",
        lambda **_kwargs: _completion_response(payload),
    )
    monkeypatch.setattr(prompt_cache, "apply_prompt_caching", lambda *_args: None)

    structured = _summarize_step(
        [],
        "Improve the loop",
        LoopState(),
        StepResult(summary="fallback summary", status="continue"),
        {"model": "test-model"},
    )

    assert structured == {
        "summary": "fallback summary",
        "files_to_preload": [],
        "changes_made": "",
        "discovered": "",
        "pending": "add regression coverage",
        "anti_patterns": "",
    }


def test_summarizer_normalizes_unique_file_paths(monkeypatch):
    payload = {
        "summary": "  completed safely  ",
        "files_to_preload": [
            " src/a.py ",
            "",
            42,
            "src/a.py",
            "src/b.py",
            "src/c.py",
            "src/d.py",
            "src/e.py",
            "src/f.py",
        ],
        "changes_made": "",
        "discovered": "",
        "pending": "",
        "anti_patterns": "",
    }
    monkeypatch.setattr(
        litellm,
        "completion",
        lambda **_kwargs: _completion_response(payload),
    )
    monkeypatch.setattr(prompt_cache, "apply_prompt_caching", lambda *_args: None)

    structured = _summarize_step(
        [],
        "Improve the loop",
        LoopState(),
        StepResult(summary="fallback summary", status="continue"),
        {"model": "test-model"},
    )

    assert structured["summary"] == "completed safely"
    assert structured["files_to_preload"] == [
        "src/a.py",
        "src/b.py",
        "src/c.py",
        "src/d.py",
        "src/e.py",
    ]


def test_terminal_step_reuses_accepted_summary_and_verification(monkeypatch):
    from infinidev.engine.loop.plan_step import PlanStep

    import pytest

    def unnecessary_request(**kwargs):
        pytest.fail("A finished step already carries its summary and evidence")

    monkeypatch.setattr(litellm, "completion", unnecessary_request)
    state = LoopState()
    state.plan.steps = [PlanStep(index=1, title="Change requested name", status="done")]
    result = _summarize_step(
        [], "Change name only", state,
        StepResult(summary="Changed ACUERDOS.json to Faro Azul", status="done",
                   evidence_summary="python3 json assertions passed; five other fields preserved"),
        {"model": "test-model"},
    )

    assert result["summary"] == "Changed ACUERDOS.json to Faro Azul"
    assert "json assertions passed" in result["discovered"]
    assert result["pending"] == ""


def test_summary_request_has_one_bounded_attempt_without_deep_reasoning(monkeypatch):
    captured = {}

    def completion(**kwargs):
        captured.update(kwargs)
        return _completion_response({"summary": "Read configuration; next step can edit it"})

    monkeypatch.setattr(litellm, "completion", completion)
    _summarize_step(
        [], "Inspect configuration", LoopState(), StepResult(summary="Read file"),
        {"model": "custom_openai/qwen3.8-flash", "api_base": "http://localhost:8080/v1",
         "num_retries": 3,
         "retry_strategy": "exponential_backoff_retry",
         "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
    )

    assert captured["num_retries"] == 0
    assert "retry_strategy" not in captured
    assert captured["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert captured["timeout"] == 30
