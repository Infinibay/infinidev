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

from typing import Literal, get_args

# The type is the source of truth, not the tuple. Annotating a field with
# ``FindingType`` puts a real JSON ``enum`` in the schema, which is what
# constrained decoding reads — so an invalid type becomes unemittable
# rather than a runtime rejection the model has to recover from. The
# ordering matters only for how the enum renders: most-needed first.
FindingType = Literal[
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
]

FINDING_TYPES: tuple[str, ...] = get_args(FindingType)

# Anchored-memory types — require at least one anchor_* field.
ANCHORED_TYPES: frozenset[str] = frozenset({"lesson", "rule", "landmine"})


# What the enum cannot say on its own. The list of values now lives in the
# schema, so this carries only the two non-obvious facts: which types
# re-appear by themselves, and how. Glossing 'hypothesis' or 'conclusion'
# spent tokens restating the word; the "ALWAYS pass an anchor" warning
# that used to close this string is enforced in ``record_finding_tool``
# with an actionable error, which is where a rule belongs.
FINDING_TYPE_HELP: str = (
    "'lesson' (what to remember here), 'rule' (a user policy to respect) "
    "and 'landmine' (a trap that burned you before) are anchored memories: "
    "they re-appear on their own when a later tool call matches their "
    "anchor_* field. 'project_context' is always loaded. The rest are "
    "plain notes, retrieved on demand."
)
__all__ = ["FindingType", "FINDING_TYPES", "ANCHORED_TYPES", "FINDING_TYPE_HELP"]
