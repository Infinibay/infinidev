"""Deterministic execution policy selected for one configured model route.

Capabilities answer what a model or provider can do.  This module answers a
different question: how much harness surface should be shown while it does it.
The baseline deliberately preserves existing behaviour; exact, evidence-backed
route rules may make schemas more compact or move code-controlled guardrails.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelExecutionPolicy:
    """Operational knobs that do not change task authority or safety rules."""

    name: str = "baseline"
    compact_tool_schemas: bool = False
    require_step_orientation: bool = True
    step_nudge_fraction: float | None = None
    renew_step_budget_on_progress: bool = False
    semantic_stagnation_control: bool = False

    def step_nudge_threshold(
        self,
        *,
        max_tool_calls: int,
        configured_threshold: int,
    ) -> int:
        """Return the one-shot warning point inside a Step budget."""
        if self.step_nudge_fraction is None:
            return configured_threshold
        if max_tool_calls <= 0:
            return 0
        threshold = int(max_tool_calls * self.step_nudge_fraction)
        return min(max_tool_calls - 1, max(1, threshold))


_BASELINE = ModelExecutionPolicy()

# Live small/medium repository runs showed that M3 executes code correctly but
# follows a mid-budget close instruction too literally and pays heavily for the
# full schema catalogue on every continuation.  Keep the full reasoning prompt;
# adapt only machine-controlled surface and timing.
_MINIMAX_M3 = ModelExecutionPolicy(
    name="minimax-m3-v5",
    compact_tool_schemas=True,
    require_step_orientation=False,
    step_nudge_fraction=0.85,
    # M3 often reaches a concrete edit or a new test result near the ordinary
    # Step boundary. Preserve that in-flight conversation while observable
    # work is still advancing; the engine compares the net workspace diff, so
    # edit-then-revert activity, repeated reads, and identical test outcomes
    # do not renew the window.
    renew_step_budget_on_progress=True,
    # Live large-task traces show M3 paraphrasing the same discovery across
    # Step boundaries. An embedding may detect that meaning, but only hard
    # no-edit/no-new-test evidence is allowed to change the action space.
    semantic_stagnation_control=True,
)


def resolve_model_execution_policy(
    provider: str,
    model: str,
) -> ModelExecutionPolicy:
    """Resolve a conservative policy from non-secret route identity."""
    provider_id = provider.strip().lower()
    bare_model = model.rsplit("/", 1)[-1].strip().lower()
    if provider_id == "minimax" and bare_model == "minimax-m3":
        return _MINIMAX_M3
    return _BASELINE


__all__ = ["ModelExecutionPolicy", "resolve_model_execution_policy"]
