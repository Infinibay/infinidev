"""TUI projection tests for the Goal/Stage/Task hierarchy."""

from __future__ import annotations

from types import SimpleNamespace

from infinidev.ui.hooks_tui import TUIHooks


def test_stage_update_keeps_hierarchy_separate_from_step_list() -> None:
    invalidations: list[bool] = []
    app = SimpleNamespace(
        _staged_planning={},
        _plan_text="",
        _steps_text="> Edit parser\no Run focused test",
        _persist_runtime_state=lambda: None,
        invalidate=lambda: invalidations.append(True),
    )
    hooks = TUIHooks(app)
    snapshot = {
        "goal": {"title": "Finish adaptive planning"},
        "stages": [{
            "number": 2,
            "status": "active",
            "spec": {"title": "Wire persistence"},
            "tasks": [
                {"status": "completed", "spec": {"title": "Store state"}},
                {"status": "active", "spec": {"title": "Resume state"}},
            ],
        }],
    }

    hooks.on_stage_update(snapshot)

    assert app._staged_planning == snapshot
    assert "Goal: Finish adaptive planning" in app._plan_text
    assert "Stage 2: Wire persistence [active]" in app._plan_text
    assert "v Store state" in app._plan_text
    assert "> Resume state" in app._plan_text
    assert app._steps_text == "> Edit parser\no Run focused test"
    assert invalidations
