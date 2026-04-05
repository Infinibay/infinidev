"""Tests for Infinidev engine components."""
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
