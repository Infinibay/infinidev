"""Tests for cumulative token accounting in the developer loop."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infinidev.engine.loop.llm_caller import LLMCaller
from infinidev.engine.loop.loop_state import LoopState


def test_track_usage_keeps_last_call_and_cumulative_token_counts() -> None:
    state = LoopState()
    context = SimpleNamespace(state=state)

    LLMCaller._track_usage(
        context,
        SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
            )
        ),
    )
    LLMCaller._track_usage(
        context,
        SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=70,
                completion_tokens=10,
                total_tokens=80,
            )
        ),
    )

    assert state.total_tokens == 200
    assert state.total_prompt_tokens == 170
    assert state.total_completion_tokens == 30
    assert state.last_prompt_tokens == 70
    assert state.last_completion_tokens == 10


def test_loop_state_loads_old_serialized_state_without_cumulative_fields() -> None:
    state = LoopState.model_validate(
        {
            "total_tokens": 120,
            "last_prompt_tokens": 100,
            "last_completion_tokens": 20,
        }
    )

    assert state.total_tokens == 120
    assert state.total_prompt_tokens == 0
    assert state.total_completion_tokens == 0


def test_fc_call_disables_client_retries_for_controlled_run(monkeypatch) -> None:
    captured = {}

    def fail_once(*args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("429 rate limit")

    monkeypatch.setattr("infinidev.engine.loop.llm_caller._call_llm", fail_once)
    context = SimpleNamespace(
        allow_llm_retries=False,
        llm_params={},
        planning_schemas=[],
        tool_schemas=[],
    )

    with pytest.raises(RuntimeError, match="429"):
        LLMCaller()._call_fc(context, [], False, 0)

    assert captured["retry_attempts"] == 1
