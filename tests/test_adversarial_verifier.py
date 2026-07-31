"""Tests for Phase 3: the adversarial cited-evidence LLM verifier.

Covers:
  - StepVerification 'llm_judge' kind: is_executable but NOT is_deterministic
  - the per-step gate skips llm_judge (deferred to task end)
  - AdversarialVerifier verdict handling: grounded PASS, ungrounded-PASS
    demotion, FAIL, UNVERIFIABLE, parse/call failure → unverifiable
  - run_review_rework_loop dispatches deterministic vs llm_judge and surfaces
    unverifiable objectives without reworking them

The LLM call is injected (completion_fn) so no network is touched.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infinidev.engine.analysis.step_verification import StepVerification
from infinidev.engine.analysis.adversarial_verifier import AdversarialVerifier
from infinidev.engine.analysis.review_result import ReviewResult
from infinidev.engine.analysis import review_engine as re_mod
from infinidev.engine.analysis.review_engine import run_review_rework_loop
from infinidev.engine.loop.engine import LoopEngine
from infinidev.engine.loop.models import LoopState
from infinidev.engine.loop.plan_step import PlanStep


# ── kind semantics ───────────────────────────────────────────────────────

class TestLlmJudgeKind:
    def test_llm_judge_executable_not_deterministic(self):
        v = StepVerification(kind="llm_judge", spec="duplication removed")
        assert v.is_executable is True
        assert v.is_deterministic is False

    def test_deterministic_kinds(self):
        assert StepVerification(kind="command", spec="true").is_deterministic is True
        assert StepVerification(kind="none").is_deterministic is False

    def test_from_loose_accepts_llm_judge(self):
        v = StepVerification.from_loose({"verify_kind": "llm_judge", "verify_spec": "reads cleanly"})
        assert v is not None and v.kind == "llm_judge"


class TestPerStepGateSkipsLlmJudge:
    def test_gate_skips_llm_judge(self, tmp_path):
        eng = LoopEngine()
        state = LoopState()
        state.plan.steps = [PlanStep(
            index=1, title="refactor",
            verify=StepVerification(kind="llm_judge", spec="no duplication"),
            status="active",
        )]
        ctx = SimpleNamespace(state=state, project_id=1, agent_id="t", workspace_path=str(tmp_path))
        call = SimpleNamespace(id="sc1", function=SimpleNamespace(
            arguments='{"summary":"x","status":"continue","evidence_summary":"refactored it"}'))
        messages = [{"role": "tool", "tool_call_id": "sc1", "content": "ok"}]
        # llm_judge must NOT block the per-step gate (deferred to task end).
        assert eng._objective_gate_blocks(ctx, call, messages) is False


# ── AdversarialVerifier verdicts ─────────────────────────────────────────

def _verifier(content):
    return AdversarialVerifier(completion_fn=lambda messages: content)


class TestAdversarialVerifier:
    def test_grounded_pass(self):
        files = {"reader.py": "def parse_all():\n    return shared_helper()\n"}
        v = _verifier('{"verdict":"PASS","cited_evidence":"return shared_helper()","reason":"uses one helper"}')
        r = v.verify(StepVerification(kind="llm_judge", spec="dedup"), changed_files=files)
        assert r.passed is True and r.unverifiable is False

    def test_ungrounded_pass_demoted_to_fail(self):
        files = {"reader.py": "def parse_all():\n    return shared_helper()\n"}
        # The judge claims a PASS but quotes text that isn't in the changes.
        v = _verifier('{"verdict":"PASS","cited_evidence":"return totally_made_up_symbol()","reason":"trust me"}')
        r = v.verify(StepVerification(kind="llm_judge", spec="dedup"), changed_files=files)
        assert r.passed is False
        assert "demoted" in r.summary

    def test_pass_with_empty_evidence_demoted(self):
        files = {"a.py": "x = 1\n"}
        v = _verifier('{"verdict":"PASS","cited_evidence":"","reason":"looks good"}')
        r = v.verify(StepVerification(kind="llm_judge", spec="x"), changed_files=files)
        assert r.passed is False

    def test_fail(self):
        v = _verifier('{"verdict":"FAIL","cited_evidence":"","reason":"still duplicated"}')
        r = v.verify(StepVerification(kind="llm_judge", spec="dedup"), changed_files={"a.py": "y=2"})
        assert r.passed is False and r.unverifiable is False
        assert "FAIL" in r.summary

    def test_unverifiable(self):
        v = _verifier('{"verdict":"UNVERIFIABLE","cited_evidence":"","reason":"needs human review"}')
        r = v.verify(StepVerification(kind="llm_judge", spec="ux is nice"), changed_files={"a.py": "z=3"})
        assert r.passed is True and r.unverifiable is True

    def test_unparseable_is_unverifiable(self):
        v = _verifier("this is not json at all")
        r = v.verify(StepVerification(kind="llm_judge", spec="x"), changed_files={"a.py": "q=4"})
        assert r.unverifiable is True

    def test_call_exception_is_unverifiable(self):
        def boom(messages):
            raise RuntimeError("model down")
        v = AdversarialVerifier(completion_fn=boom)
        r = v.verify(StepVerification(kind="llm_judge", spec="x"), changed_files={"a.py": "w=5"})
        assert r.passed is True and r.unverifiable is True

    def test_whitespace_normalised_grounding(self):
        files = {"a.py": "def  f( x ):\n        return   x\n"}  # irregular whitespace
        v = _verifier('{"verdict":"PASS","cited_evidence":"def f( x ): return x","reason":"present"}')
        r = v.verify(StepVerification(kind="llm_judge", spec="f exists"), changed_files=files)
        assert r.passed is True


# ── integration: re-verify loop dispatches + surfaces unverifiable ───────

class _FakeAgent:
    def activate_context(self, **kw): pass
    def deactivate(self): pass


class _FakeReviewer:
    def reset(self): pass
    def _should_multi_pass(self, *a, **k): return False
    @property
    def can_review_again(self): return True
    def review(self, **kw): return ReviewResult(verdict="APPROVED", summary="ok")


class _FakeEngine:
    def __init__(self, workspace, checks, files=None):
        self._workspace = workspace
        self._checks = checks
        self._files = files or {}
        self.execute_calls = []
    def get_objective_checks(self): return self._checks
    def get_plan_steps(self): return []
    def get_file_contents(self): return self._files
    def get_file_tracker(self): return None
    def get_changed_files_summary(self): return ""
    def get_file_change_reasons(self): return {}
    def execute(self, *, agent, task_prompt, verbose=True, **kwargs):
        self.execute_calls.append(task_prompt)
        return "fixed"


def _run(engine, monkeypatch, judge_content=None):
    monkeypatch.setattr(re_mod, "collect_automated_checks", lambda **kw: {"verification_passed": True})
    if judge_content is not None:
        # Patch AdversarialVerifier so the loop's internal construction uses our stub.
        import infinidev.engine.analysis.adversarial_verifier as av_mod
        real = av_mod.AdversarialVerifier
        monkeypatch.setattr(
            av_mod, "AdversarialVerifier",
            lambda *a, **k: real(*a, completion_fn=lambda m: judge_content),
        )
    statuses = []
    def on_status(level, msg): statuses.append((level, msg))
    result, review = run_review_rework_loop(
        engine=engine, agent=_FakeAgent(), session_id="s1",
        task_prompt=("do it", "expected"), initial_result="initial",
        reviewer=_FakeReviewer(), on_status=on_status,
    )
    return result, review, statuses


class TestReworkLoopAdversarialDispatch:
    def test_llm_judge_pass_no_rework(self, tmp_path, monkeypatch):
        files = {"r.py": "return shared_helper()"}
        eng = _FakeEngine(str(tmp_path),
                          checks=[(1, "dedup", StepVerification(kind="llm_judge", spec="dedup"))],
                          files=files)
        _, review, statuses = _run(
            eng, monkeypatch,
            judge_content='{"verdict":"PASS","cited_evidence":"return shared_helper()","reason":"ok"}')
        assert eng.execute_calls == []
        assert any(s[0] == "objectives_pass" for s in statuses)

    def test_llm_judge_fail_triggers_rework(self, tmp_path, monkeypatch):
        from infinidev.config import settings as settings_mod
        monkeypatch.setattr(settings_mod.settings, "REVIEW_OBJECTIVE_REVERIFY_MAX_ROUNDS", 1)
        eng = _FakeEngine(str(tmp_path),
                          checks=[(1, "dedup", StepVerification(kind="llm_judge", spec="dedup"))],
                          files={"r.py": "duplicated"})
        _, _, statuses = _run(
            eng, monkeypatch,
            judge_content='{"verdict":"FAIL","cited_evidence":"","reason":"still dup"}')
        assert len(eng.execute_calls) == 1  # bounded by max_rounds=1
        assert any(s[0] == "objectives_fail" for s in statuses)

    def test_unverifiable_surfaced_not_reworked(self, tmp_path, monkeypatch):
        eng = _FakeEngine(str(tmp_path),
                          checks=[(1, "ux", StepVerification(kind="llm_judge", spec="ux nice"))],
                          files={"r.py": "x=1"})
        _, _, statuses = _run(
            eng, monkeypatch,
            judge_content='{"verdict":"UNVERIFIABLE","cited_evidence":"","reason":"needs human"}')
        assert eng.execute_calls == []  # not reworked
        assert any(s[0] == "objectives_unverified" for s in statuses)
        assert any(s[0] == "objectives_pass" for s in statuses)

    def test_adversarial_disabled_skips(self, tmp_path, monkeypatch):
        from infinidev.config import settings as settings_mod
        monkeypatch.setattr(settings_mod.settings, "REVIEW_ADVERSARIAL_VERIFY_ENABLED", False)
        eng = _FakeEngine(str(tmp_path),
                          checks=[(1, "ux", StepVerification(kind="llm_judge", spec="ux nice"))],
                          files={"r.py": "x=1"})
        _, _, statuses = _run(eng, monkeypatch)  # no judge content needed
        assert eng.execute_calls == []
        assert any(s[0] == "objectives_unverified" for s in statuses)
