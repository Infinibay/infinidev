"""Tests for the analyst planner (Commit 6).

The planner takes an EscalationPacket and produces a Plan via a
short LLM loop that terminates on emit_plan. These tests mock out
litellm.completion entirely; they verify the contract without making
real calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from infinidev.engine.analysis.plan import Plan
from infinidev.engine.analysis.planner import run_planner
from infinidev.engine.orchestration.escalation_packet import EscalationPacket


# ─────────────────────────────────────────────────────────────────────────
# Mock shapes (minimal versions of LiteLLM's response types)
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class _F:
    name: str
    arguments: str


@dataclass
class _TC:
    id: str
    function: _F
    type: str = "function"


@dataclass
class _M:
    content: str = ""
    tool_calls: list[_TC] | None = None


@dataclass
class _C:
    message: _M


@dataclass
class _R:
    choices: list[_C]


def _tc(name: str, args: dict[str, Any] | None = None, call_id: str = "tc-1") -> _TC:
    return _TC(id=call_id, function=_F(name=name, arguments=json.dumps(args or {})))


def _resp(tool_calls: list[_TC] | None = None, content: str = "") -> _R:
    return _R(choices=[_C(message=_M(content=content, tool_calls=tool_calls))])


class _Scripted:
    def __init__(self, responses: list[_R]):
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("More calls than scripted")
        return self.responses.pop(0)


@pytest.fixture
def patch_litellm(monkeypatch):
    def _install(responses: list[_R]) -> _Scripted:
        scripted = _Scripted(responses)
        import litellm as _lit
        monkeypatch.setattr(_lit, "completion", scripted)
        return scripted

    monkeypatch.setattr(
        "infinidev.engine.analysis.planner.get_litellm_params_for_behavior",
        lambda: {"model": "test/mock", "api_base": "http://localhost"},
    )
    return _install


def _sample_escalation() -> EscalationPacket:
    return EscalationPacket(
        user_request="arreglá el bug del JWT en auth.py",
        understanding="Fix JWT validation bug in src/auth.py",
        opened_files=["src/auth.py"],
        user_visible_preview="Voy a arreglar el JWT.",
        user_signal="dale arreglalo",
    )


class TestBasicEmit:
    def test_single_emit_call_returns_plan(self, patch_litellm):
        patch_litellm([
            _resp([_tc("emit_plan", {
                "overview": "Fix the exp-claim check in validate_token.",
                "steps": [
                    {
                        "title": "Patch exp-claim",
                        "detail": "Update validate_token to reject expired tokens.",
                        "expected_output": "Unit test passes.",
                    },
                    {
                        "title": "Run tests",
                        "detail": "pytest tests/test_auth.py",
                        "expected_output": "All auth tests pass.",
                    },
                ],
            })]),
        ])
        plan = run_planner(_sample_escalation())
        assert isinstance(plan, Plan)
        assert "exp-claim check" in plan.overview
        assert len(plan.steps) == 2
        assert plan.steps[0].title == "Patch exp-claim"
        assert "expired tokens" in plan.steps[0].detail


class TestExplorationBudget:
    def test_exploration_then_emit(self, patch_litellm):
        patch_litellm([
            _resp([_tc("read_file", {"file_path": "src/auth.py"})]),
            _resp([_tc("emit_plan", {
                "overview": "Fix the validate_token exp check.",
                "steps": [{"title": "Patch", "detail": "x", "expected_output": "y"}],
            })]),
        ])
        plan = run_planner(_sample_escalation())
        assert len(plan.steps) == 1
        assert plan.overview.startswith("Fix the validate_token")

    def test_budget_nudge_forces_emit(self, patch_litellm):
        # 4 read calls = budget exhausted; nudge fires; emit on iter 5.
        patch_litellm([
            _resp([_tc("list_directory", {"file_path": "."}, f"tc-{i}")])
            for i in range(4)
        ] + [
            _resp([_tc("emit_plan", {
                "overview": "Budget-limited plan.",
                "steps": [{"title": "Do it", "detail": "d", "expected_output": "e"}],
            })]),
        ])
        plan = run_planner(_sample_escalation())
        assert plan.overview == "Budget-limited plan."


class TestDefensiveFallbacks:
    def test_empty_overview_falls_back(self, patch_litellm):
        patch_litellm([
            _resp([_tc("emit_plan", {
                "overview": "",
                "steps": [{"title": "x", "detail": "y", "expected_output": "z"}],
            })]),
        ])
        plan = run_planner(_sample_escalation())
        # Fallback plan has a neutral overview (no debug-reason prose that
        # would repeat every iteration as <plan-overview>). The user's
        # original request lives in the step's detail instead.
        assert "Carry out the user's request" in plan.overview
        assert len(plan.steps) == 1
        assert "arreglá el bug" in plan.steps[0].detail

    def test_zero_steps_falls_back(self, patch_litellm):
        patch_litellm([
            _resp([_tc("emit_plan", {
                "overview": "Plan text but no steps.",
                "steps": [],
            })]),
        ])
        plan = run_planner(_sample_escalation())
        assert len(plan.steps) >= 1  # fallback plan has at least one step

    def test_text_reply_without_tool_calls_falls_back(self, patch_litellm):
        patch_litellm([_resp(content="Sure, I'll plan that.", tool_calls=None)])
        plan = run_planner(_sample_escalation())
        # Fallback produced a single-step plan.
        assert len(plan.steps) == 1

    def test_llm_error_falls_back(self, patch_litellm, monkeypatch):
        patch_litellm([])
        import litellm as _lit
        def _boom(**kwargs):
            raise RuntimeError("LLM down")
        monkeypatch.setattr(_lit, "completion", _boom)
        plan = run_planner(_sample_escalation())
        assert isinstance(plan, Plan)
        assert len(plan.steps) == 1


class TestToolboxIntegrity:
    def test_planner_schema_has_emit_plan_and_no_write_tools(self, patch_litellm):
        scripted = patch_litellm([
            _resp([_tc("emit_plan", {
                "overview": "ok",
                "steps": [{"title": "x", "detail": "y", "expected_output": "z"}],
            })]),
        ])
        run_planner(_sample_escalation())
        assert len(scripted.calls) == 1
        tools = scripted.calls[0]["tools"]
        names = {t["function"]["name"] for t in tools}
        assert "emit_plan" in names
        # Planner must not have terminators from other tiers.
        assert "respond" not in names
        assert "escalate" not in names
        assert "step_complete" not in names
        # And must not have write tools.
        forbidden = {
            "create_file", "replace_lines", "edit_symbol",
            "execute_command", "git_commit",
        }
        assert not (names & forbidden)


class TestHandoffRendering:
    def test_opened_files_included_in_handoff_prompt(self, patch_litellm):
        scripted = patch_litellm([
            _resp([_tc("emit_plan", {
                "overview": "ok", "steps": [{"title": "x"}],
            })]),
        ])
        run_planner(_sample_escalation())
        # First call's messages contains the handoff text.
        msgs = scripted.calls[0]["messages"]
        handoff = next(m for m in msgs if m["role"] == "user")
        assert "src/auth.py" in handoff["content"]
        assert "do NOT re-open" in handoff["content"]
