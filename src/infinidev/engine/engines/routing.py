"""Engine selection: resolve the configured mode into a concrete engine.

Implements the Auto coordinator's classification step
(docs/GRAPH_ENGINE_BETA_DESIGN.md §8.4). The output is a structured,
explainable decision — engine, confidence, reasons, risks, reconsider
triggers — that the coordinator persists as an ``engine_selected`` event and
optionally shows to the user. The user can always override via
``TASK_ENGINE_MODE``.

Graph beta is a live engine. An explicit ``graph_beta`` selection is pinned
to it; ``AUTO_ENGINE_ALLOW_GRAPH`` controls only whether ``auto`` may select
Graph for exploratory or branching work.

The classifier is deliberately small: normal work uses one durable Task with
a rolling Step horizon, while Graph remains available for explicitly
branching work. It exists to produce labeled decisions we can measure, not to
invent a planning hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from infinidev.config.settings import settings

ENGINE_REACT = "react"
ENGINE_TASK = "task"
ENGINE_STAGED = "staged"
ENGINE_GRAPH_BETA = "graph_beta"

VALID_MODES = ("auto", ENGINE_TASK, ENGINE_REACT, ENGINE_STAGED, ENGINE_GRAPH_BETA)

# Feature tokens that tilt the auto classifier. Kept bilingual-light: the
# product surface is Spanish, code/requests are often English.
_STAGED_KEYWORDS = (
    "migrate", "migration", "refactor across", "all endpoints", "every endpoint",
    "each endpoint", "step by step", "milestone", "stages", "phases",
    "multi-step", "multi step", "and then", "y luego", "después",
    "primero", "first ", "1.", "2.", "3.",
)
_REACT_KEYWORDS = (
    "what is", "qué es", "explain", "explica", "why does", "por qué",
    "rename", "typo", "print", "log ", "single file", "one file",
)
# Graph fits non-linear, evidence-driven, branching work (§8.3): several
# plausible routes to weigh, hypotheses to test, or a shape that emerges
# while exploring rather than a fixed milestone list.
_GRAPH_KEYWORDS = (
    "investigate", "compare", "alternatives", "trade-off", "tradeoff",
    "hypothesis", "experiment", "explore approaches", "multiple approaches",
    "several ways", "options for", "branching", "non-linear", "weigh",
    "investiga", "compara", "alternativas", "hipótesis", "explora enfoques",
)
_INTERNAL_CONTEXT_MARKERS = (
    "<retrieval-context",
    '<hook-output event="task_start"',
)


def _literal_routing_request(request: str) -> str:
    """Exclude injected retrieval/hook text from engine classification."""
    lowered = request.lower()
    cut_points = [
        index
        for marker in _INTERNAL_CONTEXT_MARKERS
        if (index := lowered.find(marker)) >= 0
    ]
    if cut_points:
        request = request[:min(cut_points)]
    return request.strip()



@dataclass(frozen=True)
class EngineSelection:
    """Structured engine-selection decision.

    ``engine`` is the adapter that will actually run. ``requested_mode`` is
    what configuration asked for (they differ for ``auto``). Explicit modes
    are pinned and therefore do not fall back to another adapter.
    """

    engine: str
    requested_mode: str
    confidence: float = 0.5
    reasons: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    reconsider_if: list[str] = field(default_factory=list)
    estimated_overhead: str = "low"
    fallback_note: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "requested_mode": self.requested_mode,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "risks": list(self.risks),
            "reconsider_if": list(self.reconsider_if),
            "estimated_overhead": self.estimated_overhead,
            "fallback_note": self.fallback_note,
        }


def normalize_mode(raw: str | None) -> str:
    """Clamp an arbitrary mode string onto the valid vocabulary."""
    mode = (raw or "").strip().lower()
    if mode in VALID_MODES:
        return mode
    return ENGINE_TASK


def _extract_features(escalation: Any) -> dict[str, Any]:
    """Cheap, explainable features off the EscalationPacket."""
    request = _literal_routing_request(
        getattr(escalation, "user_request", "") or ""
    )
    text = request.lower()
    features = {
        "request_len": len(request),
        "line_count": request.count("\n") + 1,
        "has_grounded_spec": getattr(escalation, "grounded_spec", None) is not None,
        "has_design_brief": getattr(escalation, "design_brief", None) is not None,
        "council_requested": bool(getattr(escalation, "council_requested", False)),
        "staged_hits": sum(1 for kw in _STAGED_KEYWORDS if kw in text),
        "react_hits": sum(1 for kw in _REACT_KEYWORDS if kw in text),
        "graph_hits": sum(1 for kw in _GRAPH_KEYWORDS if kw in text),
    }
    grounded = getattr(escalation, "grounded_spec", None)
    if grounded is not None:
        features["in_scope_count"] = len(
            getattr(grounded, "in_scope", []) or []
        )
        features["blocking_clarifications"] = len(
            getattr(grounded, "blocking_clarifications", []) or []
        )
    else:
        features["in_scope_count"] = 0
        features["blocking_clarifications"] = 0
    return features


def _classify_auto(escalation: Any) -> EngineSelection:
    """Choose Graph only for branching work; otherwise use the Task loop."""
    feats = _extract_features(escalation)
    reasons: list[str] = []
    risks: list[str] = []

    # Graph candidacy first: non-linear, evidence-driven, branching work
    # (§8.3). Only when the Auto path is allowed to pick it.
    if settings.AUTO_ENGINE_ALLOW_GRAPH:
        graph_reasons: list[str] = []
        if feats["graph_hits"]:
            graph_reasons.append("exploratory_or_branching_request")
        if (
            feats["has_grounded_spec"]
            and feats["has_design_brief"]
            and feats["in_scope_count"] >= 3
        ):
            graph_reasons.append("elaborated_spec_with_design_deliberation")
        if graph_reasons:
            return EngineSelection(
                engine=ENGINE_GRAPH_BETA,
                requested_mode="auto",
                confidence=0.6,
                reasons=graph_reasons,
                risks=["graph_beta_is_experimental"],
                reconsider_if=["more_than_three_components", "new_requirement",
                               "contradicting_evidence"],
                estimated_overhead="high",
            )

    # A developer-owned rolling plan handles both small edits and longer work
    # without promoting orientation into a separate planner phase. Auto only
    # branches away when Graph's explicitly non-linear shape applies above.
    engine = ENGINE_TASK
    confidence = 0.9
    overhead = "low"
    reasons.append("durable_task_with_rolling_steps")
    if feats["staged_hits"] or feats["in_scope_count"] >= 3:
        risks.append("task_may_need_more_step_horizons")

    return EngineSelection(
        engine=engine,
        requested_mode="auto",
        confidence=round(confidence, 2),
        reasons=reasons,
        risks=risks,
        reconsider_if=["more_than_three_components", "new_requirement",
                       "contradicting_evidence"],
        estimated_overhead=overhead,
    )


def select_engine(escalation: Any, mode: str | None = None) -> EngineSelection:
    """Resolve *mode* (or the configured default) into a selection.

    Explicit ``task``/``react``/``staged``/``graph_beta`` modes are honoured verbatim.
    ``auto`` runs the classifier, which may pick Graph only when
    ``AUTO_ENGINE_ALLOW_GRAPH`` permits it.
    """
    resolved_mode = normalize_mode(mode if mode is not None else settings.TASK_ENGINE_MODE)

    if resolved_mode == ENGINE_REACT:
        return EngineSelection(
            engine=ENGINE_REACT,
            requested_mode=resolved_mode,
            confidence=1.0,
            reasons=["user_selected_react"],
            estimated_overhead="low",
        )

    if resolved_mode == ENGINE_TASK:
        return EngineSelection(
            engine=ENGINE_TASK,
            requested_mode=resolved_mode,
            confidence=1.0,
            reasons=["user_selected_task"],
            estimated_overhead="low",
        )

    if resolved_mode == ENGINE_GRAPH_BETA:
        return EngineSelection(
            engine=ENGINE_GRAPH_BETA,
            requested_mode=resolved_mode,
            confidence=1.0,
            reasons=["user_selected_graph_beta"],
            risks=["graph_beta_is_experimental"],
            estimated_overhead="high",
        )

    if resolved_mode == ENGINE_STAGED:
        return EngineSelection(
            engine=ENGINE_STAGED,
            requested_mode=resolved_mode,
            confidence=1.0,
            reasons=["user_selected_staged"],
            estimated_overhead="medium",
        )

    return _classify_auto(escalation)


__all__ = [
    "ENGINE_GRAPH_BETA",
    "ENGINE_REACT",
    "ENGINE_TASK",
    "ENGINE_STAGED",
    "EngineSelection",
    "VALID_MODES",
    "normalize_mode",
    "select_engine",
]
