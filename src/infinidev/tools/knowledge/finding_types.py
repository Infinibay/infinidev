"""Single source of truth for finding types + anchor field metadata.

Previously duplicated across ``record_finding_input.py``,
``record_finding_tool.py``, ``update_finding_input.py``, and
``update_finding_tool.py`` — six literal copies of the same tuple.
Any future additions had to touch all four files in lockstep;
inevitably they drifted. This module centralises the list so the
four tools share one authoritative definition.

The three new types (``lesson``, ``rule``, ``landmine``) back the
anchored-memory subsystem introduced in 2026-04. They are what the
agent writes when it wants a note that will auto-inject on the next
encounter with a matching anchor. The older observational types
(``observation``, ``hypothesis``, ...) continue to work unchanged —
they simply never match an anchor lookup and behave like before.
"""

from __future__ import annotations

# Anchored-memory types — require at least one anchor_* field.
ANCHORED_TYPES: frozenset[str] = frozenset({"lesson", "rule", "landmine"})

# The ordering matters only for the docstring rendered to the LLM —
# the most commonly-needed types are listed first.
FINDING_TYPES: tuple[str, ...] = (
    # Anchored memory — auto-injected when the agent touches the anchor.
    "lesson",
    "rule",
    "landmine",
    # Observational knowledge — loaded via <project-knowledge> block.
    "observation",
    "hypothesis",
    "experiment",
    "proof",
    "conclusion",
    "project_context",
)


# Human-readable description of each type, used in the help tool and
# the record_finding / update_finding input schemas so the LLM sees
# both the list AND the guidance on when to pick each.
FINDING_TYPE_HELP: str = (
    "'lesson' — a fact worth remembering the next time you touch the "
    "anchored file/symbol/tool. 'rule' — a user preference or policy "
    "you must respect. 'landmine' — something that burned you before, "
    "a warning for next time. 'observation' — a general note with no "
    "anchor. 'hypothesis' — a belief that still needs validation. "
    "'experiment' — a result from a specific trial. 'proof' — a "
    "validated fact. 'conclusion' — a top-level decision. "
    "'project_context' — always-loaded structural knowledge about the "
    "project (e.g. 'tests live in tests/, logs at ~/.infinidev/'). "
    "For the first three (lesson/rule/landmine), ALWAYS provide at "
    "least one anchor_* parameter or the memory will never fire."
)
__all__ = ["FINDING_TYPES", "ANCHORED_TYPES", "FINDING_TYPE_HELP"]
