"""Tests for planner-authored deterministic step verification.

Covers the per-step objective-verification feature:
  - StepVerification typed model (is_executable, from_loose coercion)
  - ObjectiveVerifier executor (command / test_id / file_contains /
    symbol_exists), including the observable-fragment gate
  - evidence_summary capture in parse_step_complete_args
  - verify threading: emit_plan args -> PlanStepSpec -> PlanStep
  - the <verification-method> prompt block
  - LoopEngine._objective_gate_blocks (pass passes, fail blocks +
    overwrites the tool result, 'blocked' status skips, attempt budget)

The executor runs real subprocesses against temp workspaces — the whole
point is that PASS/FAIL is decided by an exit code, not an LLM claim.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

from infinidev.engine.analysis.step_verification import StepVerification
from infinidev.engine.analysis.objective_verifier import ObjectiveVerifier
from infinidev.engine.analysis.plan import Plan, PlanStepSpec
from infinidev.engine.analysis.planner import _build_plan_from_args
from infinidev.engine.loop.engine import LoopEngine, _seed_state_from_plan
from infinidev.engine.loop.models import LoopState
from infinidev.engine.loop.plan_step import PlanStep
from infinidev.engine.loop.context import build_iteration_prompt
from infinidev.engine.formats.tool_call_parser import parse_step_complete_args


# ── StepVerification model ───────────────────────────────────────────────

class TestStepVerification:
    def test_none_is_not_executable(self):
        assert StepVerification(kind="none", spec="anything").is_executable is False

    def test_command_needs_spec(self):
        assert StepVerification(kind="command", spec="").is_executable is False
        assert StepVerification(kind="command", spec="pytest").is_executable is True

    def test_file_contains_needs_observable(self):
        # file_contains is meaningless without a needle to look for.
        assert StepVerification(kind="file_contains", spec="a.py").is_executable is False
        assert StepVerification(
            kind="file_contains", spec="a.py", observable="def foo"
        ).is_executable is True

    def test_from_loose_flat_authoring_shape(self):
        v = StepVerification.from_loose({
            "title": "step",
            "verify_kind": "command",
            "verify_spec": "pytest -q",
            "verify_observable": "passed",
        })
        assert v is not None
        assert v.kind == "command" and v.spec == "pytest -q" and v.observable == "passed"

    def test_from_loose_nested_shape(self):
        v = StepVerification.from_loose({"kind": "command", "spec": "make test"})
        assert v is not None and v.spec == "make test"

    def test_from_loose_returns_none_for_nonexecutable(self):
        assert StepVerification.from_loose({"verify_kind": "none"}) is None
        assert StepVerification.from_loose({"verify_kind": "command", "verify_spec": ""}) is None
        assert StepVerification.from_loose(None) is None
        assert StepVerification.from_loose("garbage") is None

    def test_from_loose_passthrough_model(self):
        src = StepVerification(kind="command", spec="true")
        assert StepVerification.from_loose(src) is src


# ── ObjectiveVerifier executor ───────────────────────────────────────────

class TestObjectiveVerifierCommand:
    def test_command_pass(self, tmp_path):
        r = ObjectiveVerifier(str(tmp_path)).verify(
            StepVerification(kind="command", spec="true")
        )
        assert r.passed is True

    def test_command_fail(self, tmp_path):
        r = ObjectiveVerifier(str(tmp_path)).verify(
            StepVerification(kind="command", spec="false")
        )
        assert r.passed is False
        assert "exited" in r.summary

    def test_command_observable_required_and_present(self, tmp_path):
        r = ObjectiveVerifier(str(tmp_path)).verify(
            StepVerification(kind="command", spec="echo hello-world", observable="hello-world")
        )
        assert r.passed is True

    def test_command_observable_missing_fails_even_on_exit_zero(self, tmp_path):
        r = ObjectiveVerifier(str(tmp_path)).verify(
            StepVerification(kind="command", spec="echo nope", observable="hello-world")
        )
        assert r.passed is False
        assert "required output not found" in r.summary

    def test_nonexecutable_check_passes_as_skip(self, tmp_path):
        r = ObjectiveVerifier(str(tmp_path)).verify(StepVerification(kind="none"))
        assert r.passed is True
        assert r.commands_run == []


class TestObjectiveVerifierFileContains:
    def test_present(self, tmp_path):
        f = tmp_path / "auth.py"
        f.write_text("def validate_token(t):\n    return payload['exp'] > now\n")
        r = ObjectiveVerifier(str(tmp_path)).verify(
            StepVerification(kind="file_contains", spec="auth.py", observable="payload['exp']")
        )
        assert r.passed is True

    def test_absent(self, tmp_path):
        f = tmp_path / "auth.py"
        f.write_text("def validate_token(t):\n    return True\n")
        r = ObjectiveVerifier(str(tmp_path)).verify(
            StepVerification(kind="file_contains", spec="auth.py", observable="payload['exp']")
        )
        assert r.passed is False

    def test_missing_file(self, tmp_path):
        r = ObjectiveVerifier(str(tmp_path)).verify(
            StepVerification(kind="file_contains", spec="nope.py", observable="x")
        )
        assert r.passed is False
        assert "unreadable" in r.summary


class TestObjectiveVerifierSymbolExists:
    def test_found(self, tmp_path):
        (tmp_path / "m.py").write_text("class WidgetFactory:\n    pass\n")
        r = ObjectiveVerifier(str(tmp_path)).verify(
            StepVerification(kind="symbol_exists", spec="WidgetFactory")
        )
        assert r.passed is True

    def test_not_found(self, tmp_path):
        (tmp_path / "m.py").write_text("class Other:\n    pass\n")
        r = ObjectiveVerifier(str(tmp_path)).verify(
            StepVerification(kind="symbol_exists", spec="WidgetFactory")
        )
        assert r.passed is False


class TestObjectiveVerifierTestId:
    def test_passing_node(self, tmp_path):
        (tmp_path / "test_sample.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n")
        r = ObjectiveVerifier(str(tmp_path)).verify(
            StepVerification(kind="test_id", spec="test_sample.py::test_ok")
        )
        assert r.passed is True

    def test_failing_node(self, tmp_path):
        (tmp_path / "test_sample.py").write_text("def test_bad():\n    assert False\n")
        r = ObjectiveVerifier(str(tmp_path)).verify(
            StepVerification(kind="test_id", spec="test_sample.py::test_bad")
        )
        assert r.passed is False


# ── evidence_summary capture ─────────────────────────────────────────────

class TestEvidenceCapture:
    def test_evidence_summary_captured(self):
        sr = parse_step_complete_args(
            '{"summary":"done","status":"continue",'
            '"evidence_summary":"ran pytest tests/test_auth.py::test_expired — 1 passed"}'
        )
        assert "pytest" in sr.evidence_summary

    def test_evidence_summary_defaults_empty(self):
        sr = parse_step_complete_args('{"summary":"done","status":"continue"}')
        assert sr.evidence_summary == ""


# ── verify threading planner -> spec -> step ─────────────────────────────

class TestVerifyThreading:
    def test_build_plan_from_args_maps_verify(self):
        plan = _build_plan_from_args({
            "overview": "Fix the exp check and prove it with a test.",
            "steps": [{
                "title": "Patch validate_token",
                "detail": "Reject expired tokens.",
                "expected_output": "expired tokens rejected",
                "verify_kind": "test_id",
                "verify_spec": "tests/test_auth.py::test_expired",
            }],
        })
        assert plan is not None
        spec = plan.steps[0]
        assert spec.verify is not None
        assert spec.verify.kind == "test_id"
        assert spec.verify.spec == "tests/test_auth.py::test_expired"

    def test_seed_state_copies_verify_to_step(self):
        plan = Plan(
            overview="x" * 10,
            steps=[PlanStepSpec(
                title="patch",
                verify=StepVerification(kind="command", spec="pytest -q"),
            )],
        )
        state = LoopState()
        _seed_state_from_plan(state, plan)
        step = state.plan.steps[0]
        assert step.verify is not None and step.verify.spec == "pytest -q"
        assert step.user_approved is True

    def test_no_verify_leaves_step_none(self):
        plan = Plan(overview="x" * 10, steps=[PlanStepSpec(title="t")])
        state = LoopState()
        _seed_state_from_plan(state, plan)
        assert state.plan.steps[0].verify is None


# ── <verification-method> prompt block ───────────────────────────────────

class TestVerificationMethodPrompt:
    def _state_with_verify(self, verify):
        state = LoopState()
        state.plan.steps = [PlanStep(index=1, title="patch", verify=verify, status="active")]
        return state

    def test_block_rendered_when_executable(self):
        state = self._state_with_verify(
            StepVerification(kind="test_id", spec="tests/test_auth.py::test_expired")
        )
        prompt = build_iteration_prompt("task", "expected", state)
        assert "<verification-method>" in prompt
        assert "tests/test_auth.py::test_expired" in prompt
        assert "EXTERNAL, automated check" in prompt

    def test_block_absent_without_verify(self):
        state = self._state_with_verify(None)
        prompt = build_iteration_prompt("task", "expected", state)
        assert "<verification-method>" not in prompt

    def test_observable_rendered(self):
        state = self._state_with_verify(
            StepVerification(kind="command", spec="echo x", observable="MARKER-123")
        )
        prompt = build_iteration_prompt("task", "expected", state)
        assert "MARKER-123" in prompt


# ── the gate: LoopEngine._objective_gate_blocks ──────────────────────────

def _make_ctx(verify, workspace, status="continue"):
    state = LoopState()
    state.plan.steps = [PlanStep(index=1, title="patch", verify=verify, status="active")]
    ctx = SimpleNamespace(
        state=state, project_id=1, agent_id="t", workspace_path=str(workspace),
    )
    call = SimpleNamespace(
        id="sc1",
        function=SimpleNamespace(
            arguments='{"summary":"x","status":"%s","evidence_summary":"ran it ok now"}' % status,
        ),
    )
    return ctx, call


class TestObjectiveGate:
    def test_pass_allows_close(self, tmp_path):
        eng = LoopEngine()
        ctx, call = _make_ctx(StepVerification(kind="command", spec="true"), tmp_path)
        messages = [{"role": "tool", "tool_call_id": "sc1", "content": "ok"}]
        assert eng._objective_gate_blocks(ctx, call, messages) is False

    def test_fail_blocks_and_overwrites_result(self, tmp_path):
        eng = LoopEngine()
        ctx, call = _make_ctx(StepVerification(kind="command", spec="false"), tmp_path)
        messages = [{"role": "tool", "tool_call_id": "sc1", "content": "ok"}]
        assert eng._objective_gate_blocks(ctx, call, messages) is True
        # The step_complete tool result was overwritten with the rejection.
        tool_msg = next(m for m in messages if m.get("tool_call_id") == "sc1")
        assert "BLOCKED" in tool_msg["content"]
        assert "attempt 1/3" in tool_msg["content"]

    def test_blocked_status_skips_gate(self, tmp_path):
        eng = LoopEngine()
        # A failing check, but the model declared 'blocked' — don't force it.
        ctx, call = _make_ctx(
            StepVerification(kind="command", spec="false"), tmp_path, status="blocked"
        )
        messages = [{"role": "tool", "tool_call_id": "sc1", "content": "ok"}]
        assert eng._objective_gate_blocks(ctx, call, messages) is False

    def test_no_verify_skips_gate(self, tmp_path):
        eng = LoopEngine()
        ctx, call = _make_ctx(None, tmp_path)
        messages = [{"role": "tool", "tool_call_id": "sc1", "content": "ok"}]
        assert eng._objective_gate_blocks(ctx, call, messages) is False

    def test_attempt_budget_stops_blocking(self, tmp_path):
        eng = LoopEngine()
        eng._verify_attempts = {}
        ctx, call = _make_ctx(StepVerification(kind="command", spec="false"), tmp_path)
        messages = [{"role": "tool", "tool_call_id": "sc1", "content": "ok"}]
        # Cap is 3: first three calls block, the fourth gives up (returns False)
        # and surfaces the unmet objective as a note.
        results = [eng._objective_gate_blocks(ctx, call, messages) for _ in range(4)]
        assert results == [True, True, True, False]
        assert any("NOT verified" in n for n in ctx.state.notes)

    def test_disabled_via_setting(self, tmp_path, monkeypatch):
        from infinidev.config import settings as settings_mod
        monkeypatch.setattr(settings_mod.settings, "LOOP_OBJECTIVE_VERIFY_ENABLED", False)
        eng = LoopEngine()
        ctx, call = _make_ctx(StepVerification(kind="command", spec="false"), tmp_path)
        messages = [{"role": "tool", "tool_call_id": "sc1", "content": "ok"}]
        assert eng._objective_gate_blocks(ctx, call, messages) is False
