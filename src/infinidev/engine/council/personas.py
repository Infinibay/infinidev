"""Starter palette of personas for the council.

IMPORTANT: this is NOT a fixed roster. The moderator generates personas
and per-member objectives tailored to each task (via ``seed_council``).
This palette is only a reference the moderator's prompt shows as
few-shot examples, so the personas it invents are diverse and partly in
tension — which is what makes a debate worth more than a single agent.
"""

from __future__ import annotations

# (id, persona) reference examples — the moderator adapts/replaces these.
PERSONA_PALETTE: list[tuple[str, str]] = [
    (
        "advocate-mvp",
        "Pushes for the simplest thing that could work. Attacks "
        "over-engineering and speculative generality; asks 'what is the "
        "smallest change that solves the actual problem?'",
    ),
    (
        "advocate-robust",
        "Prioritises robustness, edge cases, and maintainability. Assumes "
        "the happy path is the easy 20% and hunts the failure modes in the "
        "other 80%.",
    ),
    (
        "skeptic",
        "Refutes every proposal by default. Looks for the hidden bug, the "
        "unstated assumption, the case that breaks it. Demands evidence.",
    ),
    (
        "researcher",
        "Brings facts, not opinions. Uses read-only tools (code_search, "
        "read_file, web) to ground claims in what the codebase and docs "
        "actually say.",
    ),
    (
        "integrator",
        "Cares how the change fits the existing codebase: what it touches, "
        "what conventions it must follow, what it might break elsewhere.",
    ),
]


def render_palette() -> str:
    return "\n".join(f"  * {pid}: {desc}" for pid, desc in PERSONA_PALETTE)


__all__ = ["PERSONA_PALETTE", "render_palette"]
