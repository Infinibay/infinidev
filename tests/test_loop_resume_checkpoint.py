"""Regression tests for durable, bounded LoopEngine resume checkpoints."""

from __future__ import annotations

import json
from types import SimpleNamespace

from infinidev.engine.loop.loop_plan import LoopPlan, PlanStep
from infinidev.engine.loop.loop_state import LoopState
from infinidev.engine.loop.resume_checkpoint import (
    CHECKPOINT_MAX_UTF8_BYTES,
    build_loop_resume_checkpoint,
    compact_checkpoint_for_chat,
    render_interrupted_step_context,
    resume_state_for_task,
)
from infinidev.engine.orchestration.task_schema import task_from_free_text


def _task():
    return task_from_free_text(
        "Continue the interrupted implementation without replaying the full transcript.",
        title="Resume interrupted implementation",
    )


def _prompt() -> tuple[str, str]:
    return "Implement the active task.", "Return a verified result."


def test_checkpoint_round_trip_is_task_bound() -> None:
    state = LoopState(
        plan=LoopPlan(steps=[
            PlanStep(
                index=1,
                title="Patch resume",
                explanation="Keep only active Step context",
                status="active",
            ),
        ]),
        notes=["The visual transcript is not model context."],
        pending_archive=[
            ("read_file", '{"path":"src/app.py"}', "observed source"),
        ],
    )
    checkpoint = build_loop_resume_checkpoint(state, _prompt(), _task())

    restored = resume_state_for_task(checkpoint, _prompt(), _task())

    assert restored is not None
    assert restored["plan"]["steps"][0]["title"] == "Patch resume"
    assert restored["pending_archive"][0][2] == "observed source"
    assert resume_state_for_task(
        checkpoint,
        ("A different task", "Different result"),
        None,
    ) is None


def test_terminal_checkpoint_is_not_resumed() -> None:
    checkpoint = build_loop_resume_checkpoint(
        LoopState(), _prompt(), _task(), terminal_status="done"
    )

    assert resume_state_for_task(checkpoint, _prompt(), _task()) is None


def test_checkpoint_bounds_large_current_step_outputs() -> None:
    state = LoopState(
        pending_archive=[
            ("execute_command", '{"command":"pytest"}', "x" * 500_000)
            for _ in range(40)
        ],
        last_test_output="failure\n" + "y" * 500_000,
    )

    checkpoint = build_loop_resume_checkpoint(state, _prompt(), _task())
    encoded = json.dumps(checkpoint, ensure_ascii=False).encode("utf-8")

    assert len(encoded) <= CHECKPOINT_MAX_UTF8_BYTES
    assert len(checkpoint["state"]["pending_archive"]) <= 24
    assert "checkpoint truncated" in checkpoint["state"]["pending_archive"][-1][2]


def test_interrupted_step_context_is_plain_bounded_data() -> None:
    state = LoopState(
        pending_archive=[
            ("read_file", '{"path":"src/app.py"}', "observed source"),
        ],
    )

    rendered = render_interrupted_step_context(state)

    assert "<interrupted-step-context>" in rendered
    assert "observed source" in rendered
    assert "untrusted tool exchanges" in rendered


def test_chat_summary_excludes_full_checkpoint_payloads() -> None:
    state = LoopState(
        pending_archive=[
            ("execute_command", "args", "z" * 100_000),
        ],
        last_test_output="q" * 100_000,
    )
    checkpoint = build_loop_resume_checkpoint(state, _prompt(), _task())

    summary = compact_checkpoint_for_chat(checkpoint)
    encoded = json.dumps(summary, ensure_ascii=False)

    assert len(encoded) < 20_000
    assert "active_step_tool_exchanges" in summary
    assert "last_test_output" not in encoded



def test_loop_engine_persists_and_loads_matching_checkpoint(temp_db) -> None:
    from infinidev.db.service import register_session
    from infinidev.engine.loop.engine import LoopEngine
    from infinidev.tools.base.context import set_context

    register_session("resume-loop", "/work")
    set_context(agent_id="developer", session_id="resume-loop")
    task = _task()
    state = LoopState(
        plan=LoopPlan(steps=[
            PlanStep(index=1, title="Resume me", status="active"),
        ]),
        pending_archive=[("read_file", "{}", "observed")],
    )
    ctx = SimpleNamespace(
        state=state,
        desc=_prompt()[0],
        expected=_prompt()[1],
        task=task,
    )
    engine = LoopEngine()

    engine._checkpoint(ctx)
    loaded = engine._load_persisted_resume_state(_prompt(), task)

    assert loaded is not None
    assert loaded["plan"]["steps"][0]["title"] == "Resume me"
    assert loaded["pending_archive"][0][2] == "observed"

    engine._checkpoint(ctx, terminal_status="done")
    assert engine._load_persisted_resume_state(_prompt(), task) is None
