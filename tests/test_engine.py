"""Tests for Infinidev engine components."""
import json
from types import SimpleNamespace

import pytest
from infinidev.engine.loop.models import (
    LoopState,
    LoopPlan,
    PlanStep,
    StepOperation,
    ActionRecord,
    StepResult,
)
from infinidev.engine.summarizer import SmartContextSummarizer
from infinidev.engine.loop.context import build_tools_prompt_section


class TestLoopState:
    """Tests for LoopState model."""

    def test_initialization(self):
        """LoopState initializes with empty plan and history."""
        state = LoopState()
        assert len(state.plan.steps) == 0
        assert len(state.history) == 0
        assert state.current_step_index == 0
        assert state.iteration_count == 0

    def test_active_step_detection(self):
        """Active step is detected correctly."""
        state = LoopState()
        state.plan.steps = [
            PlanStep(index=0, title="Step 1", status="pending"),
            PlanStep(index=1, title="Step 2", status="active"),
            PlanStep(index=2, title="Step 3", status="done"),
        ]
        assert state.plan.active_step is not None
        assert state.plan.active_step.title == "Step 2"
        assert state.plan.active_step.index == 1

    def test_has_pending_property(self):
        """has_pending property works correctly."""
        state = LoopState()
        state.plan.steps = [
            PlanStep(index=0, title="Step 1", status="pending"),
            PlanStep(index=1, title="Step 2", status="done"),
        ]
        assert state.plan.has_pending is True

        state.plan.steps = [
            PlanStep(index=0, title="Step 1", status="done"),
            PlanStep(index=1, title="Step 2", status="skipped"),
        ]
        assert state.plan.has_pending is False

    def test_opened_file_cache_rejects_invalid_entries(self, caplog):
        """Cache bookkeeping is best-effort and cannot abort the engine."""
        state = LoopState()

        state.cache_file(None, "content")  # type: ignore[arg-type]
        state.cache_file("", "content")
        state.cache_file("valid.py", None)  # type: ignore[arg-type]

        assert state.opened_files == {}
        assert "without a valid path" in caplog.text
        assert "without string content" in caplog.text


class TestLoopPlan:
    """Tests for LoopPlan model."""

    def test_from_steps(self):
        """LoopPlan can be created with steps."""
        plan = LoopPlan(steps=[
            PlanStep(index=0, title="Read source code"),
            PlanStep(index=1, title="Analyze requirements"),
        ])
        assert len(plan.steps) == 2
        assert plan.steps[0].title == "Read source code"
        assert plan.steps[1].status == "pending"

    def test_operations(self):
        """LoopPlan operations work correctly."""
        plan = LoopPlan(steps=[
            PlanStep(index=0, title="Step 1"),
            PlanStep(index=1, title="Step 2", status="active"),
        ])

        # Mark active step done
        plan.mark_active_done()
        assert plan.steps[1].status == "done"

        # Activate next pending step (step 0, not appended step)
        plan.activate_next()
        assert plan.steps[0].status == "active"


class TestActionRecord:
    """Tests for ActionRecord model."""

    def test_creation(self):
        """ActionRecord is created with correct structure."""
        record = ActionRecord(
            step_index=0,
            summary="Read main.py file",
            tool_calls_count=1,
        )
        assert record.step_index == 0
        assert record.summary == "Read main.py file"
        assert record.tool_calls_count == 1


class TestStepResult:
    """Tests for StepResult model."""

    def test_creation(self):
        """StepResult is created with correct structure."""
        result = StepResult(
            summary="All tests passed",
            status="continue",
            next_steps=[],
        )
        assert result.summary == "All tests passed"
        assert result.status == "continue"
        assert result.next_steps == []

    def test_with_final_answer(self):
        """StepResult with final_answer."""
        result = StepResult(
            summary="Task complete",
            status="done",
            final_answer="Project setup complete!",
            next_steps=[],
        )
        assert result.status == "done"
        assert result.final_answer == "Project setup complete!"


class TestLoopContext:
    """Tests for loop context prompt builders."""

    def test_build_tools_prompt_section(self):
        """Tools prompt section is built correctly."""
        tool_schemas = [
            {"function": {"name": "read_file", "description": "Read a file", "parameters": {"properties": {}}}},
            {"function": {"name": "write_file", "description": "Write a file", "parameters": {"properties": {}}}},
        ]

        prompt = build_tools_prompt_section(tool_schemas)
        assert "read_file" in prompt
        assert "write_file" in prompt
        assert "tool_calls" in prompt
        # Parameters section is included for schemas with properties (empty dicts skip it)


class TestSummarizer:
    """Tests for SmartContextSummarizer."""

    def test_initialization(self):
        """Summarizer initializes with correct defaults."""
        summarizer = SmartContextSummarizer()
        assert summarizer.max_tokens == 200

    def test_generate_summary_empty(self):
        """Generating summary from empty state returns empty string."""
        summarizer = SmartContextSummarizer()
        state = LoopState()
        result = summarizer.generate_summary(state)
        assert result == ""

    def test_generate_summary_short(self):
        """Short history produces minimal summary."""
        summarizer = SmartContextSummarizer(max_tokens=100)
        state = LoopState()
        state.history = [
            ActionRecord(step_index=0, summary="Read main.py", tool_calls_count=1),
        ]
        state.plan.steps = [
            PlanStep(index=0, title="Next step"),
        ]
        result = summarizer.generate_summary(state)
        assert len(result) <= 100
        assert "Next step" in result  # Pending work is shown


def test_opened_files_block_respects_its_own_budget() -> None:
    """The block advertises a budget, so the budget has to bind.

    The first entry used to bypass it, and ``read_file`` accepts files of
    several megabytes: one read could put the whole file into every later
    round of the task.
    """
    from infinidev.engine.loop.context import _render_opened_files
    from infinidev.engine.loop.models import LoopState
    from infinidev.engine.loop.opened_file import OpenedFile

    state = LoopState()
    state.opened_files_prompt_max_chars = 4_000
    state.opened_files["huge.py"] = OpenedFile(
        path="huge.py", content="x" * 200_000, pinned=True,
    )

    rendered = _render_opened_files(state)

    assert len(rendered) < 5_000, (
        f"the block ignored its own 4 000-character budget and produced "
        f"{len(rendered)} characters"
    )
    assert "truncated at the context budget" in rendered
    assert "200000 characters on disk" in rendered
    assert "read_file" in rendered, "the model has to be told how to get the rest"


def test_a_file_under_the_budget_is_still_sent_whole() -> None:
    from infinidev.engine.loop.context import _render_opened_files
    from infinidev.engine.loop.models import LoopState
    from infinidev.engine.loop.opened_file import OpenedFile

    state = LoopState()
    state.opened_files["small.py"] = OpenedFile(
        path="small.py", content="print('hi')\n", pinned=True,
    )

    rendered = _render_opened_files(state)

    assert "print('hi')" in rendered
    assert "truncated" not in rendered


def test_project_stats_answers_on_an_unindexed_workspace(tmp_path) -> None:
    """Its stated job is the first call of an analysis, when nothing is indexed.

    Returning an error made the orientation tool fail in the one situation it
    exists for; the recorded campaigns spent 16 calls on it. The filesystem
    answers the orientation question without an index.
    """
    from infinidev.tools.base.context import set_context
    from infinidev.tools.code_intel.project_stats_tool import ProjectStatsTool

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "src" / "util.py").write_text("y = 2\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("docs\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("//\n", encoding="utf-8")

    set_context(project_id=991_991, agent_id="probe", workspace_path=str(tmp_path))

    result = ProjectStatsTool()._run()

    assert '"error"' not in result, result
    assert "files on disk:    3" in result, result
    assert ".py 2" in result
    assert "node_modules" not in result
    assert "not indexed" in result


def test_project_stats_survives_an_unreadable_index(tmp_path, monkeypatch) -> None:
    """A missing index table must not fail the orientation call.

    A bare database without the code-intelligence schema is a real state a
    session can be in, and it produced "Stats query failed: no such table:
    ci_files" instead of an answer.
    """
    from infinidev.tools.base import context as tool_context
    from infinidev.tools.code_intel.project_stats_tool import ProjectStatsTool

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    tool_context.set_context(project_id=5, agent_id="p", workspace_path=str(tmp_path))

    def _boom(func, **kwargs):
        raise RuntimeError("no such table: ci_files")

    # The tool imports the helper inside ``_run``, so the patch goes on the
    # module it imports from.
    from infinidev.tools.base import db as tools_db

    monkeypatch.setattr(tools_db, "execute_with_retry", _boom)

    result = ProjectStatsTool()._run()

    assert '"error"' not in result, result
    assert "index unavailable" in result
    assert "files on disk:    1" in result


class _RunnerCtx:
    def __init__(self, paths):
        self.state = SimpleNamespace(opened_files={p: None for p in paths})


def test_an_edit_without_a_path_is_repaired_when_the_file_proves_it(tmp_path) -> None:
    """The dominant malformed call: edit_file with no file_path.

    The model leans on the file it has been working in. Filling the path in is
    safe only when a single file was opened in the Step and the exact
    old_string occurs in it once.
    """
    from infinidev.engine.loop.tool_runner import _repair_missing_edit_target

    target = tmp_path / "cart.py"
    target.write_text("def total():\n    return 1\n", encoding="utf-8")
    ctx = _RunnerCtx([str(target)])

    repaired = _repair_missing_edit_target(
        ctx, "edit_file",
        json.dumps({"old_string": "return 1", "new_string": "return 2"}),
    )

    assert json.loads(repaired)["file_path"] == str(target)


def test_the_repair_refuses_when_the_match_is_not_unique(tmp_path) -> None:
    from infinidev.engine.loop.tool_runner import _repair_missing_edit_target

    target = tmp_path / "twice.py"
    target.write_text("x = 1\ny = 1\n", encoding="utf-8")
    ctx = _RunnerCtx([str(target)])

    arguments = json.dumps({"old_string": "= 1", "new_string": "= 2"})
    assert _repair_missing_edit_target(ctx, "edit_file", arguments) == arguments


def test_the_repair_refuses_with_more_than_one_candidate(tmp_path) -> None:
    from infinidev.engine.loop.tool_runner import _repair_missing_edit_target

    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    for path in (a, b):
        path.write_text("x = 1\n", encoding="utf-8")
    ctx = _RunnerCtx([str(a), str(b)])

    arguments = json.dumps({"old_string": "x = 1", "new_string": "x = 2"})
    assert _repair_missing_edit_target(ctx, "edit_file", arguments) == arguments


def test_the_repair_leaves_a_named_path_and_other_tools_alone(tmp_path) -> None:
    from infinidev.engine.loop.tool_runner import _repair_missing_edit_target

    target = tmp_path / "cart.py"
    target.write_text("x = 1\n", encoding="utf-8")
    ctx = _RunnerCtx([str(target)])

    named = json.dumps({"file_path": "/elsewhere.py", "old_string": "x = 1"})
    assert _repair_missing_edit_target(ctx, "edit_file", named) == named

    other = json.dumps({"old_string": "x = 1"})
    assert _repair_missing_edit_target(ctx, "apply_file_patch", other) == other
