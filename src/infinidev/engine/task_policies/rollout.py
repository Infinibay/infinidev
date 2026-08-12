"""Model-specific rollout gates backed by paired task-policy evidence."""

from __future__ import annotations

from infinidev.engine.prompt_composition import ConditionalPromptFragment


# Rollout is fail-closed: only fragment versions that won a clean paired
# evaluation are injected. Detection and telemetry still run for every policy,
# and an operator can disable evidence gating explicitly for experiments.
_APPROVED_BY_MODEL: dict[str, frozenset[tuple[str, int]]] = {
    "minimax:minimax-m3": frozenset({("refactor.developer", 1)}),
    "openai_subscription:gpt-5.6-terra": frozenset({("bugfix.developer", 3)}),
}


def approved_fragment_ids(provider: str, model: str) -> frozenset[tuple[str, int]]:
    """Return the evidence-approved fragment versions for a known model route."""
    route = f"{provider}:{model}".casefold()
    for key, approved in _APPROVED_BY_MODEL.items():
        if all(part in route for part in key.split(":")):
            return approved
    return frozenset()


def fragment_is_approved(
    fragment: ConditionalPromptFragment,
    *,
    provider: str,
    model: str,
) -> bool:
    """Require an exact model-route and fragment-version approval."""
    approved = approved_fragment_ids(provider, model)
    return (fragment.id, fragment.version) in approved


__all__ = ["approved_fragment_ids", "fragment_is_approved"]
