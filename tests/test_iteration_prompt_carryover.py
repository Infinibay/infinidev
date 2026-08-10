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
from infinidev.engine.orchestration.task_schema import task_from_free_text


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
        _cr_pending_user_guidance=[],
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


def test_controlled_context_corpus_renders_on_every_iteration(engine_and_ctx):
    engine, ctx = engine_and_ctx
    ctx.context_corpus = "FILE src/rule.py\nROLLBACK_REQUIRES_REVERSE_ORDER = True"

    first = _user_turn(build_iteration_messages(engine, ctx, iteration=0))
    later = _user_turn(build_iteration_messages(engine, ctx, iteration=4))

    assert "<repository-context-corpus>" in first
    assert "ROLLBACK_REQUIRES_REVERSE_ORDER" in first
    assert "ROLLBACK_REQUIRES_REVERSE_ORDER" in later


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


def test_context_rank_query_tracks_active_step_and_new_user_guidance(
    engine_and_ctx, monkeypatch
):
    from infinidev.engine.loop.context_builder import _rank_at_pivot
    from infinidev.engine.loop.plan_step import PlanStep

    engine, ctx = engine_and_ctx
    engine._cr_hooks._enabled = True
    engine._cr_hooks._session_id = "session"
    engine._cr_hooks._task_id = "task"
    engine._cr_hooks._task_embedding = None
    engine._cr_hooks._task_embedding_simplified = None
    ctx.state.plan.steps = [
        PlanStep(
            index=2,
            title="Verify migration rollback",
            explanation="Inspect the transaction boundary",
            expected_output="Rollback test passes",
            status="active",
        )
    ]
    queries: list[str] = []
    sentinel = object()

    def fake_rank(query, *args, **kwargs):
        queries.append(query)
        return sentinel

    from infinidev.config import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "CONTEXT_RANK_ENABLED", True, raising=False)
    monkeypatch.setattr("infinidev.engine.context_rank.ranker.rank", fake_rank)

    result = _rank_at_pivot(
        engine,
        ctx,
        iteration=3,
        user_messages=["Use the PostgreSQL adapter, not SQLite."],
    )

    assert result is sentinel
    assert "Verify migration rollback" in queries[0]
    assert "Rollback test passes" in queries[0]
    assert "PostgreSQL adapter" in queries[0]


def test_new_user_guidance_invalidates_same_step_context_rank_cache(
    engine_and_ctx, monkeypatch
):
    from infinidev.engine.loop.context_builder import _rank_at_pivot

    engine, ctx = engine_and_ctx
    engine._cr_hooks._enabled = True
    engine._cr_hooks._session_id = "session"
    engine._cr_hooks._task_id = "task"
    engine._cr_hooks._task_embedding = None
    engine._cr_hooks._task_embedding_simplified = None
    engine._cr_last_pivot_key = (-1, "")
    engine._cr_cached_result = object()
    calls = 0

    def fake_rank(*args, **kwargs):
        nonlocal calls
        calls += 1
        return object()

    from infinidev.config import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "CONTEXT_RANK_ENABLED", True, raising=False)
    monkeypatch.setattr("infinidev.engine.context_rank.ranker.rank", fake_rank)

    _rank_at_pivot(engine, ctx, iteration=3, user_messages=["Change the target module."])

    assert calls == 1


def test_mid_step_guidance_feeds_next_rank_once(engine_and_ctx, monkeypatch):
    engine, ctx = engine_and_ctx
    engine._cr_pending_user_guidance = ["The user changed the database target."]
    seen: list[list[str] | None] = []

    def fake_rank_at_pivot(engine, ctx, iteration, *, user_messages=None):
        seen.append(user_messages)
        return None

    monkeypatch.setattr(
        "infinidev.engine.loop.context_builder._rank_at_pivot", fake_rank_at_pivot
    )

    build_iteration_messages(engine, ctx, iteration=4)
    build_iteration_messages(engine, ctx, iteration=5)

    assert seen == [["The user changed the database target."], None]
    assert engine._cr_pending_user_guidance == []


def test_ken_turn_context_survives_structured_task_prompt_rebuilds(engine_and_ctx):
    engine, ctx = engine_and_ctx
    ctx.task = task_from_free_text(
        "Continue the grounded infinigpu implementation from its handoff document."
    )
    ctx.desc = """
<task authority="USER_LITERAL">
Continue infinigpu.
</task>

<retrieval-context source="ken" authority="advisory" scope-effect="none">
<context-rank>
Files:
- infinigpu/CONTINUE.md
- infinigpu/guest/icd/infinigpu_cmd_buffer.c
Findings:
- The ICD per-draw UV-byte root cause is already isolated.
</context-rank>
</retrieval-context>
""".strip()

    earlier = _user_turn(build_iteration_messages(engine, ctx, iteration=2))
    later = _user_turn(build_iteration_messages(engine, ctx, iteration=9))

    for prompt in (earlier, later):
        assert "<retrieval-context source=\"ken\"" in prompt
        assert "infinigpu/CONTINUE.md" in prompt
        assert "ICD per-draw UV-byte root cause" in prompt
