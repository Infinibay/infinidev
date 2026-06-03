"""Tests for the hidden end-of-task work summary.

Covers two contracts:
  1. ``build_work_summary`` distils loop state + file changes into a
     framed summary, and returns None when there is nothing to record.
  2. The conversation-history split keeps ``work_summary`` turns out of
     the UI repaint / loop compact context but visible to the model.
"""

import os
import tempfile

import pytest

from infinidev.config import settings
from infinidev.engine.loop import work_summary as ws
from infinidev.engine.loop.action_record import ActionRecord
from infinidev.engine.loop.loop_state import LoopState


class _FakeTracker:
    def __init__(self, files):
        self._files = files  # path -> (action, reasons)

    def get_all_paths(self):
        return list(self._files)

    def get_action(self, path):
        return self._files[path][0]

    def get_reasons(self, path):
        return self._files[path][1]


@pytest.fixture(autouse=True)
def _deterministic(monkeypatch):
    # Force the deterministic path so tests never hit a real model. The
    # flags are normally synced onto the module by reload_all() at runtime;
    # raising=False lets us set them even before that sync has happened.
    monkeypatch.setattr(settings, "LOOP_WORK_SUMMARY_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "LOOP_WORK_SUMMARY_USE_LLM", False, raising=False)


def test_summary_contains_required_sections():
    state = LoopState()
    state.history = [
        ActionRecord(step_index=1, summary="patched validate()",
                     changes_made="clamped index to len-1",
                     anti_patterns="first attempt used the wrong bound"),
        ActionRecord(step_index=2, summary="added helper",
                     changes_made="created normalize()",
                     pending_items="no unit test yet"),
    ]
    tracker = _FakeTracker({
        "src/a.py": ("modified", ["fix off-by-one"]),
        "src/b.py": ("created", ["new helper module"]),
    })

    out = ws.build_work_summary(
        state, tracker, final_answer="Done.", status="done",
    )

    assert out is not None
    assert "<work-summary>" in out and "</work-summary>" in out
    # Files + why
    assert "src/a.py" in out and "fix off-by-one" in out
    assert "src/b.py" in out
    # What was done per file (prose, from changes_made)
    assert "clamped index to len-1" in out
    # Challenges / problems
    assert "Challenges" in out
    assert "wrong bound" in out and "no unit test yet" in out


def test_returns_none_when_nothing_happened():
    assert ws.build_work_summary(
        LoopState(), None, final_answer="hi", status="done",
    ) is None


def test_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(settings, "LOOP_WORK_SUMMARY_ENABLED", False, raising=False)
    tracker = _FakeTracker({"x.py": ("modified", ["r"])})
    assert ws.build_work_summary(
        LoopState(), tracker, final_answer="x", status="done",
    ) is None


def test_history_split_hides_from_ui_shows_to_model():
    os.environ["INFINIDEV_DB_PATH"] = tempfile.mktemp(suffix=".db")
    import importlib
    import infinidev.db.service as db
    importlib.reload(db)

    db.init_db()
    sid = "sess-ws"
    db.register_session(sid, "/tmp/ws")
    db.store_conversation_turn(sid, "user", "fix the bug")
    db.store_conversation_turn(sid, "work_summary", "<work-summary>secret</work-summary>")
    db.store_conversation_turn(sid, "assistant", "Done.")

    repaint_roles = [r for r, _ in db.get_all_turns(sid)]
    model_roles = [r for r, _ in db.get_recent_turns_full(sid)]
    compact = db.get_recent_summaries(sid)

    # Hidden from the user-facing surfaces...
    assert "work_summary" not in repaint_roles
    assert not any("secret" in c for c in compact)
    # ...but present for the model.
    assert "work_summary" in model_roles
