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
