"""The plan block doubles as an index into working memory.

Two renderings of a finished step used to sit in the model's prompt saying
almost the same thing: its line in ``<plan>`` and its collapsed line in
``<previous-actions>``. This makes the first carry what the second cannot —
the labels the archive filed the step's evidence under.

The property that makes it work, and the one worth a test: those labels are
not a description of the evidence, they are the *titles the rows were stored
with*, so pasting one into ``recall_context`` returns the raw tool output
behind the step's conclusion. Anything that changes how ``_format_call``
renders a title, or how ``archive_*`` reports what it stored, breaks the link
silently — the plan would still render plausible-looking labels that retrieve
nothing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infinidev.engine.loop.loop_plan import LoopPlan
from infinidev.engine.loop.plan_step import PlanStep
from infinidev.engine.loop.step_manager import StepManager
from infinidev.engine.working_memory import WorkingMemory


def _step_messages() -> list[dict]:
    return [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "c1",
                "function": {
                    "name": "read_file",
                    "arguments": '{"file_path": "src/auth.py"}',
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "def validate_token(t):\n" + ("    ...\n" * 40),
        },
    ]


@pytest.fixture
def memory(tmp_path) -> WorkingMemory:
    return WorkingMemory("session-index", embed=False, db_path=str(tmp_path / "wm.db"))


# ── the link between a rendered label and a stored row ────────────────────


def test_an_archived_title_is_a_working_recall_query(memory):
    """The whole design rests on this: the label IS the query."""
    titles = memory.archive_step(1, _step_messages(), summary="")
    label = next(t for t in titles if t.startswith("read_file"))

    hits = memory.search(label, limit=3)
    assert hits, f"the rendered label {label!r} retrieves nothing"
    assert any("validate_token" in (h.content or "") for h in hits)


def test_the_step_summary_row_is_not_offered_as_evidence(tmp_path):
    """Recalling it returns the sentence already printed on the plan line."""
    ctx = _ctx_with_plan(tmp_path)
    titles = ["Summary of step 1", "read_file(file_path=src/auth.py)"]
    StepManager._record_outcome(ctx, 1, "Found the missing exp check.", titles)
    assert ctx.state.plan.steps[0].evidence == ["read_file(file_path=src/auth.py)"]


# ── what gets written onto the step ───────────────────────────────────────


def _ctx_with_plan(tmp_path) -> SimpleNamespace:
    plan = LoopPlan()
    plan.steps = [PlanStep(index=1, title="Read auth.py", status="done")]
    state = SimpleNamespace(plan=plan)
    return SimpleNamespace(state=state, workspace_path=str(tmp_path))


def test_the_conclusion_is_the_summarys_first_sentence(tmp_path):
    ctx = _ctx_with_plan(tmp_path)
    StepManager._record_outcome(
        ctx, 1,
        "validate_token() has no exp check. Next: add it and run the tests.",
        [],
    )
    assert ctx.state.plan.steps[0].conclusion == "validate_token() has no exp check"


def test_a_summary_without_sentences_is_truncated_not_dropped(tmp_path):
    ctx = _ctx_with_plan(tmp_path)
    StepManager._record_outcome(ctx, 1, "x" * 400, [])
    assert len(ctx.state.plan.steps[0].conclusion) == 160


def test_recording_against_a_missing_step_is_a_no_op(tmp_path):
    """``explore`` steps and bootstrap records point at indices with no step."""
    ctx = _ctx_with_plan(tmp_path)
    StepManager._record_outcome(ctx, 99, "should not raise", ["read_file(a.py)"])
    assert ctx.state.plan.steps[0].conclusion == ""


def test_an_existing_conclusion_is_not_overwritten(tmp_path):
    ctx = _ctx_with_plan(tmp_path)
    ctx.state.plan.steps[0].conclusion = "already established"
    StepManager._record_outcome(ctx, 1, "a later summary", [])
    assert ctx.state.plan.steps[0].conclusion == "already established"


# ── how it renders ────────────────────────────────────────────────────────


def test_a_closed_step_renders_its_conclusion_and_evidence():
    plan = LoopPlan()
    plan.steps = [
        PlanStep(
            index=1, title="Read auth.py", status="done",
            conclusion="validate_token() has no exp check",
            evidence=["read_file(file_path=src/auth.py)", "code_search(pattern=exp)"],
        ),
        PlanStep(index=2, title="Add the check", status="active"),
    ]
    rendered = plan.render()
    assert "established: validate_token() has no exp check" in rendered
    assert "read_file(file_path=src/auth.py)" in rendered
    assert "recall_context" in rendered, (
        "the labels are useless unless the line says what to do with them"
    )
    assert "2. [active] Add the check" in rendered


def test_evidence_is_capped_and_says_how_much_it_hid():
    plan = LoopPlan()
    plan.steps = [PlanStep(
        index=1, title="Explore", status="done",
        evidence=[f"read_file(file_path=f{i}.py)" for i in range(5)],
    )]
    rendered = plan.render()
    assert "+3 more" in rendered
    assert "f4.py" not in rendered


def test_a_blocked_step_says_so():
    plan = LoopPlan()
    plan.steps = [PlanStep(index=1, title="Run the suite", status="blocked")]
    assert "[blocked]" in plan.render()


def test_no_arrow_glyphs_reach_the_prompt():
    """The project's own prompt rule: words beat arrows, and this is prompt."""
    plan = LoopPlan()
    plan.steps = [PlanStep(
        index=1, title="Read", status="done",
        conclusion="found it", evidence=["read_file(a.py)"],
    )]
    rendered = plan.render()
    assert not any(glyph in rendered for glyph in ("→", "=>", "->"))
