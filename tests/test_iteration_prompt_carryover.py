"""Blocks the model needs on every iteration, not just the first.

Each iteration builds a brand-new ``[system, user]`` conversation from
compact summaries — nothing carries over implicitly. So "the model has
already seen this" is never a reason to skip a block: skipping it deletes
the information from the model's context entirely.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infinidev.engine.loop.context_builder import build_iteration_messages
from infinidev.engine.loop.models import LoopState


@pytest.fixture
def engine_and_ctx():
    engine = SimpleNamespace(
        _project_knowledge=[
            {
                "finding_type": "project_context",
                "topic": "test runner",
                "content": "the suite runs with `uv run pytest`",
            },
        ],
        _drain_user_messages=lambda: [],
        session_notes=[],
        _initial_attachments=[],
        _supports_vision_cached=False,
        _cr_hooks=SimpleNamespace(_enabled=False),
        _cr_cached_result=None,
        _cr_last_pivot_key=None,
    )
    ctx = SimpleNamespace(
        state=LoopState(),
        history_window=0,
        start_iteration=0,
        desc="do the thing",
        expected="it is done",
        max_context_tokens=100_000,
        skip_plan=False,
        task=None,
        is_small=False,
        system_prompt="sys",
        project_id=1,
        agent_id="a1",
    )
    return engine, ctx


def _user_turn(messages) -> str:
    content = messages[1]["content"]
    return content if isinstance(content, str) else str(content)


def test_project_knowledge_renders_on_a_later_iteration(engine_and_ctx):
    """The block used to vanish from iteration 2 onwards.

    The findings were fetched once and cached on the engine, then rendered
    only when ``first_turn`` — so the cache had no reader and the project's
    facts left the context after one turn.
    """
    engine, ctx = engine_and_ctx
    messages = build_iteration_messages(engine, ctx, iteration=5)
    assert "uv run pytest" in _user_turn(messages)


def test_project_knowledge_renders_on_the_first_iteration_too(engine_and_ctx, monkeypatch):
    engine, ctx = engine_and_ctx
    import infinidev.db.service as db

    monkeypatch.setattr(
        db, "get_project_knowledge",
        lambda project_id: [{
            "finding_type": "project_context",
            "topic": "test runner",
            "content": "the suite runs with `uv run pytest`",
        }],
        raising=False,
    )
    messages = build_iteration_messages(engine, ctx, iteration=0)
    assert "uv run pytest" in _user_turn(messages)


def test_context_rank_block_is_resent_from_cache_between_pivots(engine_and_ctx):
    """Recomputing between pivots is waste; not re-sending is data loss."""
    from infinidev.engine.loop.context_builder import _rank_at_pivot

    engine, ctx = engine_and_ctx
    engine._cr_hooks._enabled = True
    sentinel = object()
    engine._cr_cached_result = sentinel
    engine._cr_last_pivot_key = (-1, "")

    with pytest.MonkeyPatch.context() as mp:
        from infinidev.config import settings as settings_mod
        mp.setattr(settings_mod.settings, "CONTEXT_RANK_ENABLED", True, raising=False)
        result = _rank_at_pivot(engine, ctx, iteration=3)

    assert result is sentinel, "between pivots the cached ranking must be re-sent"
