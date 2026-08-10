"""Tests for the spec-elaboration loop."""

import json
from unittest.mock import patch

import pytest

from infinidev.config.settings import settings
from infinidev.engine.analysis.grounded_spec import (
    Assumption,
    Clarification,
    GroundedSpec,
    RejectedAlternative,
    ResolvedFact,
)
from infinidev.engine.analysis import spec_elaborator as se
from infinidev.engine.orchestration.escalation_packet import EscalationPacket


# ── Fake litellm response plumbing ────────────────────────────────────────

class _FakeFn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeTC:
    def __init__(self, name, args):
        self.id = "tc-1"
        self.function = _FakeFn(name, json.dumps(args))


class _FakeMsg:
    def __init__(self, tool_calls=None, content=""):
        self.tool_calls = tool_calls
        self.content = content


class _FakeResp:
    def __init__(self, msg):
        self.choices = [type("C", (), {"message": msg})()]


def _tool_resp(name, args):
    return _FakeResp(_FakeMsg(tool_calls=[_FakeTC(name, args)]))


@pytest.fixture
def long_escalation():
    return EscalationPacket(
        user_request="Add rate limiting to the public API so abusive clients are throttled",
        understanding="User wants request throttling on the public API endpoints",
    )


# ── GroundedSpec (pure) ───────────────────────────────────────────────────

class TestGroundedSpec:
    def test_evidence_count_counts_only_facts_with_evidence(self):
        spec = GroundedSpec(
            deliverable="x",
            resolved_facts=[
                ResolvedFact("q1", "a1", evidence="src/a.py:10"),
                ResolvedFact("q2", "a2", evidence=""),
            ],
        )
        assert spec.evidence_count == 1

    def test_render_includes_scope_and_assumptions(self):
        spec = GroundedSpec(
            deliverable="Throttle the API",
            in_scope=["public endpoints"],
            out_of_scope=["admin endpoints"],
            assumptions=[Assumption("per-IP limiting", "no config found")],
            clarifications_needed=[
                Clarification(
                    question="per-user or global?",
                    options=["per-user", "global"],
                    default="per-user",
                    risk="local_reversible",
                )
            ],
            design_direction="token bucket middleware",
            alternatives_rejected=[RejectedAlternative("global dict", "no TTL")],
        )
        rendered = spec.render_for_planner()
        assert "Out of scope" in rendered
        assert "admin endpoints" in rendered
        assert "ASSUMPTIONS" in rendered
        # The planner is told to BUILD the default, not to stall on the question.
        assert "PRODUCT DECISIONS" in rendered
        assert "proceeding with: per-user" in rendered
        assert "alternatives: global" in rendered
        assert "token bucket middleware" in rendered


# ── Gating ────────────────────────────────────────────────────────────────

class TestGating:
    def test_skips_trivial_short_request(self):
        e = EscalationPacket(user_request="fix typo", understanding="typo")
        assert se.should_elaborate(e) is False

    def test_elaborates_substantial_request(self, long_escalation):
        assert se.should_elaborate(long_escalation) is True

    def test_skips_grounding_when_request_already_has_an_execution_contract(self):
        request = (
            "Update src/auth.py so validate_token must reject expired tokens "
            "without changing valid-token behavior, and run pytest tests/test_auth.py."
        )
        escalation = EscalationPacket(user_request=request, understanding=request)

        assert se.should_elaborate(escalation) is False

    def test_elaborates_ambiguous_request_despite_file_and_test_names(self):
        request = (
            "Fix the bug in src/auth.py and run pytest tests/test_auth.py because "
            "users have reported intermittent failures in production recently."
        )
        escalation = EscalationPacket(user_request=request, understanding=request)

        assert se.should_elaborate(escalation) is True


    def test_referenced_continuation_skips_elaboration_for_every_route(self):
        request = "Lee infinigpu/CONTINUE.md y continua el trabajo"
        escalation = EscalationPacket(
            user_request=request,
            understanding=request,
        )

        assert se.should_elaborate(escalation) is False


    def test_disabled_flag_skips(self, long_escalation):
        orig = settings.SPEC_ELABORATION_ENABLED
        settings.SPEC_ELABORATION_ENABLED = False
        try:
            assert se.should_elaborate(long_escalation) is False
        finally:
            settings.SPEC_ELABORATION_ENABLED = orig


# ── Deterministic discard (the core novel piece — no LLM) ─────────────────

class TestDeterministicDiscard:
    def test_kills_candidate_referencing_missing_file(self, tmp_path):
        (tmp_path / "real.py").write_text("x = 1\n")
        candidates = [
            {"summary": "good", "referenced_files": ["real.py"]},
            {"summary": "bad", "referenced_files": ["ghost.py"]},
        ]
        winner, rejected, risks = se._deterministic_discard(
            candidates, str(tmp_path), project_id=None
        )
        assert winner["summary"] == "good"
        assert len(rejected) == 1
        assert "ghost.py" in rejected[0].why_rejected
        assert risks == []  # winner is clean

    def test_winner_residual_problems_become_risks(self, tmp_path):
        # Both reference a missing file; the least-bad still wins but its
        # unresolved reference is surfaced as a risk, not hidden.
        candidates = [{"summary": "only", "referenced_files": ["nope.py"]}]
        winner, rejected, risks = se._deterministic_discard(
            candidates, str(tmp_path), project_id=None
        )
        assert winner["summary"] == "only"
        assert any("nope.py" in r for r in risks)

    def test_new_file_without_extension_not_killed(self, tmp_path):
        # A reference that doesn't look like an existing file path is not checked.
        candidates = [{"summary": "creates module", "referenced_files": ["newpkg"]}]
        winner, rejected, risks = se._deterministic_discard(
            candidates, str(tmp_path), project_id=None
        )
        assert winner["summary"] == "creates module"
        assert risks == []

    def test_empty_candidates(self, tmp_path):
        winner, rejected, risks = se._deterministic_discard([], str(tmp_path), None)
        assert winner is None and rejected == [] and risks == []


# ── Clarification admissibility (the anti-questionnaire gate — no LLM) ────

class TestAdmissibleClarifications:
    def _q(self, question, **kw):
        base = {"question": question, "options": ["a", "b"], "default": "a"}
        base.update(kw)
        return base

    def test_keeps_a_well_formed_decision(self):
        kept, demoted = se._admissible_clarifications([self._q("per-user or global?")], 2)
        assert len(kept) == 1 and demoted == []
        assert kept[0].default == "a"
        assert kept[0].risk == "costly_to_reverse"

    def test_only_explicit_local_decisions_can_default(self):
        kept, _ = se._admissible_clarifications(
            [self._q("format compact or expanded?", risk="local_reversible")],
            2,
        )
        assert kept[0].can_use_default_without_confirmation is True

    def test_missing_risk_fails_safe_to_confirmation(self):
        kept, _ = se._admissible_clarifications([self._q("public API shape?")], 2)
        assert kept[0].can_use_default_without_confirmation is False

    def test_question_without_default_is_demoted_not_asked(self):
        # "What compute budget is available?" — no default to commit to, so it
        # is a survey question, not a decision.
        kept, demoted = se._admissible_clarifications(
            [self._q("what compute budget?", default="")], 2
        )
        assert kept == []
        assert len(demoted) == 1
        assert "what compute budget?" in demoted[0].statement

    def test_question_without_alternatives_is_demoted(self):
        kept, demoted = se._admissible_clarifications(
            [self._q("which datasets are authorised?", options=[], default="the public ones")], 2
        )
        assert kept == []
        assert "the public ones" in demoted[0].statement

    def test_bare_string_is_demoted(self):
        # A pre-gate model emitting plain strings carries nothing actionable.
        kept, demoted = se._admissible_clarifications(["which benchmarks define success?"], 2)
        assert kept == []
        assert len(demoted) == 1

    def test_caps_at_max_and_demotes_the_overflow(self):
        raw = [self._q(f"decision {i}?") for i in range(10)]
        kept, demoted = se._admissible_clarifications(raw, 2)
        assert len(kept) == 2
        # Nothing is lost — the other eight are stated as assumptions.
        assert len(demoted) == 8
        assert all("proceeding with: a" in d.statement for d in demoted)

    def test_zero_max_asks_nothing(self):
        kept, demoted = se._admissible_clarifications([self._q("x?")], 0)
        assert kept == [] and len(demoted) == 1

    def test_dedupes_on_the_question(self):
        raw = [self._q("Per-user or global?"), self._q("per-user or global")]
        kept, demoted = se._admissible_clarifications(raw, 2)
        assert len(kept) == 1 and demoted == []

    def test_default_missing_from_options_is_added(self):
        kept, _ = se._admissible_clarifications(
            [self._q("x?", options=["b", "c"], default="a")], 2
        )
        assert kept[0].options[0] == "a"

    def test_demoted_clarifications_reach_the_spec_as_assumptions(self):
        spec = se._assemble(
            "req", "understanding",
            {"deliverable": "d", "gaps": []},
            {"resolved_facts": [], "assumptions": [],
             "clarifications_needed": [self._q("q1?"), {"question": "q2?"}]},
            None, [], [],
        )
        assert len(spec.clarifications_needed) == 1
        assert any("q2?" in a.statement for a in spec.assumptions)


# ── End-to-end with mocked LLM ────────────────────────────────────────────

class TestElaborateEndToEnd:
    def test_assembles_grounded_spec_and_discards_hallucination(
        self, long_escalation, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(settings, "LLM_MODEL", "ollama_chat/test-model")
        (tmp_path / "api.py").write_text("def handler(): pass\n")

        analyze = _tool_resp("emit_analysis", {
            "deliverable": "Throttle the public API",
            "in_scope": ["public endpoints"],
            "out_of_scope": ["admin"],
            "gaps": [
                {"question": "is there existing middleware?", "kind": "technical"},
                {"question": "per-user or global?", "kind": "product_intent"},
            ],
        })
        ground = _tool_resp("emit_grounding", {
            "resolved_facts": [
                {"question": "is there existing middleware?", "answer": "no", "evidence": "api.py:1"},
            ],
            "assumptions": [],
            "clarifications_needed": [
                {
                    "question": "per-user or global?",
                    "options": ["per-user", "global"],
                    "default": "per-user",
                    "impact": "changes the bucket key",
                    "risk": "local_reversible",
                }
            ],
        })
        candidates = _tool_resp("emit_candidates", {
            "candidates": [
                {"summary": "token bucket in api.py", "referenced_files": ["api.py"]},
                {"summary": "decorator in ghost.py", "referenced_files": ["ghost.py"]},
            ],
        })

        with patch("litellm.completion", side_effect=[analyze, ground, candidates]):
            spec = se.elaborate(
                long_escalation,
                session_id="s1",
                project_id=None,
                workspace_path=str(tmp_path),
            )

        assert spec is not None
        assert spec.deliverable == "Throttle the public API"
        assert spec.out_of_scope == ["admin"]
        assert spec.evidence_count == 1
        assert [c.question for c in spec.clarifications_needed] == ["per-user or global?"]
        assert spec.clarifications_needed[0].default == "per-user"
        # The hallucinated candidate (ghost.py) was deterministically discarded.
        assert spec.design_direction == "token bucket in api.py"
        assert any("ghost.py" in r.why_rejected for r in spec.alternatives_rejected)
        # Rich retrieval key, not the raw request.
        assert "Throttle the public API" in spec.signature_text

    def test_failure_returns_none_not_raise(self, long_escalation, monkeypatch):
        monkeypatch.setattr(settings, "LLM_MODEL", "ollama_chat/test-model")
        with patch("litellm.completion", side_effect=RuntimeError("provider down")):
            spec = se.elaborate(long_escalation, project_id=None, workspace_path="/tmp")
        assert spec is None


# ── Handoff render integration ────────────────────────────────────────────

def test_render_handoff_includes_grounded_spec():
    from infinidev.engine.analysis.planner import _render_handoff
    spec = GroundedSpec(deliverable="Throttle API", in_scope=["public"], design_direction="token bucket")
    packet = EscalationPacket(
        user_request="add rate limiting to the API endpoints please",
        understanding="throttle api",
        grounded_spec=spec,
    )
    rendered = _render_handoff(packet)
    assert "GROUNDED SPEC" in rendered
    assert "token bucket" in rendered
