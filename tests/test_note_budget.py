"""What happens to a note the 20-slot budget cannot hold.

The cap is task-wide and nothing rotates it, so a long run reaches it and
everything after that used to be dropped in silence — while the model was
told ``{"status": "noted"}`` and, worse, kept being nudged to save notes
by a counter that could no longer be reset.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from infinidev.engine.loop.classified_calls import ClassifiedCalls
from infinidev.engine.loop.models import LoopState
from infinidev.engine.loop.tool_processor import AUTO_NOTE_PREFIX, ToolProcessor


def _ctx(notes: list[str], since: int = 9) -> SimpleNamespace:
    state = LoopState()
    state.notes = list(notes)
    state.tool_calls_since_last_note = since
    return SimpleNamespace(
        state=state, is_small=False,
        project_id=1, agent_id="a1", session_id="s1",
    )


def _note_call(call_id: str, text: str):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name="add_note", arguments=json.dumps({"note": text}),
        ),
    )


def _process(ctx, *calls) -> ClassifiedCalls:
    classified = ClassifiedCalls(notes=list(calls))
    ToolProcessor.process_pseudo_tools(ctx, classified, SimpleNamespace(session_notes=[]))
    return classified


def test_note_under_the_cap_is_stored_and_acknowledged():
    ctx = _ctx([])
    classified = _process(ctx, _note_call("n1", "the bug is in auth.py:412"))
    assert ctx.state.notes == ["the bug is in auth.py:412"]
    assert json.loads(classified.note_results["n1"])["status"] == "noted"


def test_saving_a_note_always_resets_the_nudge_counter():
    ctx = _ctx([f"note {i}" for i in range(20)], since=9)
    _process(ctx, _note_call("n1", "the bug is in auth.py:412"))
    assert ctx.state.tool_calls_since_last_note == 0, (
        "a full budget left the counter high, so the SAVE-NOTES nudge fired "
        "on every remaining iteration demanding a call that could not succeed"
    )


def test_a_real_finding_evicts_an_auto_note():
    """``Read <path>`` is reconstructible; a conclusion is not."""
    notes = [f"{AUTO_NOTE_PREFIX}src/a{i}.py" for i in range(15)]
    notes += [f"finding {i}" for i in range(5)]
    ctx = _ctx(notes)

    classified = _process(ctx, _note_call("n1", "the bug is in auth.py:412"))

    assert "the bug is in auth.py:412" in ctx.state.notes
    assert len(ctx.state.notes) == 20, "the budget still holds"
    assert sum(n.startswith(AUTO_NOTE_PREFIX) for n in ctx.state.notes) == 14
    assert all(f"finding {i}" in ctx.state.notes for i in range(5)), (
        "a model-written note must never be the one evicted"
    )
    assert json.loads(classified.note_results["n1"])["status"] == "noted"


def test_full_budget_of_real_notes_reports_the_drop_honestly():
    ctx = _ctx([f"finding {i}" for i in range(20)])

    classified = _process(ctx, _note_call("n1", "the bug is in auth.py:412"))

    result = json.loads(classified.note_results["n1"])
    assert result["status"] == "dropped", (
        "answering 'noted' for a note that was never saved teaches the model "
        "its context is safe when it is not"
    )
    assert "recall_context" in result["reason"] or "NOT saved" in result["reason"]


def test_dropped_note_reaches_working_memory():
    ctx = _ctx([f"finding {i}" for i in range(20)])
    _process(ctx, _note_call("n1", "the bug is in auth.py:412"))

    from infinidev.engine.working_memory import get_working_memory

    hits = get_working_memory("s1").search("bug auth", limit=5)
    assert any("auth.py:412" in (h.content or "") for h in hits), (
        f"the dropped note should be recallable; got {[h.title for h in hits]}"
    )
