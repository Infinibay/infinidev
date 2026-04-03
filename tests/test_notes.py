"""Tests for the task notes (scratchpad) system."""

import json

from infinidev.engine.loop.models import LoopState
from infinidev.engine.loop.context import build_iteration_prompt
from infinidev.engine.loop.tools import ADD_NOTE_SCHEMA, build_tool_schemas


class TestLoopStateNotes:
    """Test notes field on LoopState."""

    def test_notes_default_empty(self):
        state = LoopState()
        assert state.notes == []

    def test_notes_append(self):
        state = LoopState()
        state.notes.append("Found config at /etc/app.conf")
        state.notes.append("Using PostgreSQL 15")
        assert len(state.notes) == 2
        assert "PostgreSQL" in state.notes[1]

    def test_notes_serialize_deserialize(self):
        state = LoopState()
        state.notes.append("Important fact")
        data = state.model_dump()
        restored = LoopState.model_validate(data)
        assert restored.notes == ["Important fact"]


class TestNotesInPrompt:
    """Test that notes appear in build_iteration_prompt."""

    def test_no_notes_no_block(self):
        state = LoopState()
        prompt = build_iteration_prompt("Do something", "Expected output", state)
        assert "<notes>" not in prompt

    def test_notes_appear_in_prompt(self):
        state = LoopState()
        state.notes.append("Config is at /etc/app.conf")
        state.notes.append("DB uses port 5433")
        prompt = build_iteration_prompt("Do something", "Expected output", state)
        assert "<notes>" in prompt
        assert "</notes>" in prompt
        assert "1. Config is at /etc/app.conf" in prompt
        assert "2. DB uses port 5433" in prompt

    def test_notes_between_task_and_plan(self):
        """Notes should appear after <task> and before <plan>."""
        state = LoopState()
        state.notes.append("A note")
        prompt = build_iteration_prompt("The task", "Expected", state)
        task_pos = prompt.index("<task>")
        notes_pos = prompt.index("<notes>")
        plan_pos = prompt.index("<plan>")
        assert task_pos < notes_pos < plan_pos


class TestAddNoteSchema:
    """Test add_note schema is present in tool schemas."""

    def test_schema_structure(self):
        func = ADD_NOTE_SCHEMA["function"]
        assert func["name"] == "add_note"
        assert "note" in func["parameters"]["properties"]
        assert "note" in func["parameters"]["required"]

    def test_included_in_build_tool_schemas(self):
        schemas = build_tool_schemas([])  # No agent tools, just engine pseudo-tools
        names = [s["function"]["name"] for s in schemas]
        assert "add_note" in names
        assert "step_complete" in names
