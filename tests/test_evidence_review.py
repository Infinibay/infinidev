"""Semantic evidence review for informational outcomes."""

from __future__ import annotations

from types import SimpleNamespace

from infinidev.engine.analysis.evidence_review import (
    EvidenceReviewEngine,
    EvidenceReviewResult,
    run_evidence_review_rework_loop,
)


def test_blocking_issue_must_quote_the_submitted_response() -> None:
    result = EvidenceReviewEngine._ground(
        {
            "verdict": "REJECTED",
            "summary": "Reviewer alleged a claim that is not present.",
            "issues": [{
                "severity": "blocking",
                "category": "unsupported_claim",
                "claim_excerpt": "invented reviewer quote",
                "problem": "Unsupported",
            }],
        },
        "The response contains a different, carefully qualified claim.",
    )

    assert result.verdict == "APPROVED"
    assert result.issues[0]["severity"] == "suggestion"
    assert result.issues[0]["grounded"] is False


def test_exact_unsupported_claim_remains_blocking() -> None:
    response = "The library always retries every network failure."
    result = EvidenceReviewEngine._ground(
        {
            "verdict": "REJECTED",
            "summary": "The absolute claim has no support.",
            "issues": [{
                "severity": "blocking",
                "category": "unsupported_claim",
                "claim_excerpt": response,
                "problem": "No supplied evidence establishes this.",
            }],
        },
        response,
    )

    assert result.verdict == "REJECTED"
    assert result.issues[0]["grounded"] is True


def test_rework_loop_is_bounded_and_preserves_original_scope(monkeypatch) -> None:
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "EVIDENCE_REVIEW_MAX_ROUNDS", 2)
    monkeypatch.setattr(
        "infinidev.engine.analysis.evidence_review._recent_tool_evidence",
        lambda _session_id: "[step 1 · tool_output] web_fetch\nsource text",
    )

    class Reviewer:
        calls = 0

        def review(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return EvidenceReviewResult(
                    "REJECTED",
                    "One claim is overstated.",
                    [{
                        "severity": "blocking",
                        "category": "uncertainty_omitted",
                        "claim_excerpt": "It is certain.",
                        "problem": "Evidence is conditional.",
                        "evidence": "The source says may.",
                        "fix": "Qualify the claim.",
                    }],
                )
            return EvidenceReviewResult("APPROVED", "Claims now match evidence.")

    class Agent:
        def activate_context(self, **_kwargs):
            pass

        def deactivate(self):
            pass

    class Engine:
        prompts = []

        def execute(self, **kwargs):
            self.prompts.append(kwargs["task_prompt"][0])
            return "It may be true."

    engine = Engine()
    final, review = run_evidence_review_rework_loop(
        engine=engine,
        agent=Agent(),
        session_id="s1",
        task_prompt=("Investigate the behavior.", "A sourced answer."),
        initial_result="It is certain.",
        evidence_reviewer=Reviewer(),
    )

    assert final == "It may be true."
    assert review is not None and review.verdict == "APPROVED"
    assert len(engine.prompts) == 1
    assert "preserve the original objective" in engine.prompts[0]
    assert "Do not add new scope" in engine.prompts[0]


def test_pipeline_uses_evidence_review_when_workspace_is_unchanged(monkeypatch) -> None:
    from infinidev.engine.orchestration.pipeline import _run_review_phase

    called = {}

    def fake_loop(**kwargs):
        called.update(kwargs)
        return "grounded answer", EvidenceReviewResult("APPROVED", "Grounded")

    monkeypatch.setattr(
        "infinidev.engine.analysis.evidence_review.run_evidence_review_rework_loop",
        fake_loop,
    )

    class Engine:
        def has_file_changes(self):
            return False

    class Hooks:
        def on_phase(self, _phase):
            pass

        def on_status(self, _level, _message):
            pass

        def notify(self, *_args):
            pass

    result = _run_review_phase(
        engine=Engine(),
        agent=SimpleNamespace(),
        session_id="s1",
        task_prompt=("Research X", "Report"),
        result="draft",
        reviewer=SimpleNamespace(),
        hooks=Hooks(),
    )

    assert result == "grounded answer"
    assert called["initial_result"] == "draft"
