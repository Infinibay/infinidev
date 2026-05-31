"""Tests for the spec-elaboration loop."""

import json
from unittest.mock import patch

import pytest

from infinidev.config.settings import settings
from infinidev.engine.analysis.grounded_spec import (
    Assumption,
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
            clarifications_needed=["per-user or global?"],
            design_direction="token bucket middleware",
            alternatives_rejected=[RejectedAlternative("global dict", "no TTL")],
        )
        rendered = spec.render_for_planner()
        assert "Out of scope" in rendered
        assert "admin endpoints" in rendered
        assert "ASSUMPTIONS" in rendered
        assert "OPEN PRODUCT QUESTIONS" in rendered
        assert "token bucket middleware" in rendered


# ── Gating ────────────────────────────────────────────────────────────────

class TestGating:
    def test_skips_trivial_short_request(self):
        e = EscalationPacket(user_request="fix typo", understanding="typo")
        assert se.should_elaborate(e) is False

    def test_elaborates_substantial_request(self, long_escalation):
        assert se.should_elaborate(long_escalation) is True

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
            "clarifications_needed": ["per-user or global?"],
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
        assert spec.clarifications_needed == ["per-user or global?"]
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
