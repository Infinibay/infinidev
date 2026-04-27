"""Structured ``Task`` artefact shared by the principal and the assistant.

The task is the **single shared specification** of what the user asked
for, parsed once and rendered identically in every prompt that needs
it: the developer loop, the assistant critic, the closing review. Free
text is brittle — two LLMs reading the same paragraph reach different
conclusions about scope, success criteria, and constraints. A
structured schema removes that ambiguity at the source.

Design decisions
----------------
* **Simple now, extensible later.** This first cut deliberately omits
  per-step components / BOMs (the larger schema discussed in the
  planning conversation) so the rollout is one commit, not three.
  Fields are forward-compatible: future extensions add fields, never
  rename.
* **`kind` is open with examples.** A closed enum forces every task
  into a pre-decided bucket; an open string with curated examples
  lets the planner stay precise without rejecting weird-but-valid
  cases. ``SUGGESTED_TASK_KINDS`` defines the recommended vocabulary
  but is *not* enforced — non-standard kinds pass validation and only
  emit a soft warning in the log.
* **Backward-compatible construction.** The pipeline historically
  passes raw free-text from ``EscalationPacket.user_request``. The
  helper :func:`task_from_free_text` wraps that into a minimal Task
  so callers that haven't migrated yet keep working.
* **Pydantic, not dataclass.** Matches the rest of the engine
  (``LoopPlan``, ``PlanStep``) and gives us field validation for free.
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


SUGGESTED_TASK_KINDS: dict[str, str] = {
    "feature":       "Adds new user-visible behavior or capability.",
    "bugfix":        "Corrects incorrect or unexpected behavior in existing code.",
    "refactor":      "Restructures code without changing observable behavior.",
    "performance":   "Optimises latency, memory, or throughput.",
    "docs":          "Adds or updates documentation, comments, or examples.",
    "test":          "Adds or updates tests without changing production code.",
    "chore":         "Routine maintenance: dep bumps, lint fixes, formatting.",
    "config":        "Changes configuration, env vars, or settings.",
    "migration":     "Schema/data migration or one-shot transformation.",
    "security":      "Addresses a vulnerability or hardens defences.",
    "investigation": "Pure exploration — read code, produce a report, no code changes.",
}


_NON_FALSIFIABLE_HINTS: tuple[str, ...] = (
    "looks good", "is clean", "is nice", "feels right", "is elegant",
)


class Task(BaseModel):
    """The structured task spec.

    Rendered identically into the prompt of the principal and the
    critic so they share the same understanding. Once built (by the
    chat agent, the planner, or :func:`task_from_free_text`), this
    object is **immutable for the rest of the run** — the pipeline
    treats it as the contract.
    """

    title: str = Field(
        ...,
        min_length=5,
        max_length=120,
        description=(
            "One-line summary, imperative voice. 5-120 chars. "
            "Surfaces in logs, in the prompt header, and in any "
            "human-facing UI."
        ),
    )

    description: str = Field(
        ...,
        min_length=20,
        description=(
            "Full ask in the user's own framing. ≥20 chars. The "
            "verbatim or near-verbatim user request goes here."
        ),
    )

    kind: str = Field(
        ...,
        min_length=2,
        description=(
            "Task category. Prefer one of SUGGESTED_TASK_KINDS but "
            "free-form strings are allowed. Drives downstream "
            "behaviour (e.g. a 'docs' task suppresses the "
            "docstring-only-change guardrail)."
        ),
    )

    acceptance_criteria: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Falsifiable success conditions. At least one. Each one "
            "should be a sentence whose truth can be checked by "
            "running a command, reading a file, or inspecting "
            "behaviour. 'Looks good' is NOT a criterion."
        ),
    )

    out_of_scope: list[str] = Field(
        default_factory=list,
        description=(
            "Things explicitly NOT to do. Prevents scope creep at the "
            "spec level rather than after the fact. Use plain "
            "imperative phrases ('do not touch the auth middleware')."
        ),
    )

    constraints: list[str] = Field(
        default_factory=list,
        description=(
            "Hard rules the work must respect (no breaking changes, "
            "must work offline, must keep CI green, no new "
            "dependencies, etc.). Treated as invariants, not goals."
        ),
    )

    references: list[str] = Field(
        default_factory=list,
        description=(
            "Pointers to external context: ticket IDs, PR URLs, file "
            "paths, doc links. The principal and critic can be told "
            "to consult these but the schema doesn't fetch anything."
        ),
    )

    # --- Validators ---------------------------------------------------------

    @field_validator("title")
    @classmethod
    def _strip_title(cls, v: str) -> str:
        # Defensive: planners sometimes wrap the title in quotes or
        # trailing periods. Normalise so downstream rendering is clean.
        return v.strip().strip('"').strip("'").rstrip(".").strip()

    @field_validator("kind")
    @classmethod
    def _normalise_kind(cls, v: str) -> str:
        normalised = v.strip().lower().replace(" ", "_").replace("-", "_")
        if not normalised:
            raise ValueError("kind cannot be empty after normalisation")
        if normalised not in SUGGESTED_TASK_KINDS:
            logger.debug(
                "Task.kind=%r is not in SUGGESTED_TASK_KINDS; accepted "
                "but unusual.",
                normalised,
            )
        return normalised

    @field_validator("acceptance_criteria")
    @classmethod
    def _validate_criteria(cls, v: list[str]) -> list[str]:
        cleaned: list[str] = []
        for i, raw in enumerate(v):
            if not isinstance(raw, str):
                raise ValueError(
                    f"acceptance_criteria[{i}] must be a string"
                )
            s = raw.strip()
            if len(s) < 5:
                raise ValueError(
                    f"acceptance_criteria[{i}] is too short ({len(s)} "
                    f"chars; need ≥5)"
                )
            lower = s.lower()
            for hint in _NON_FALSIFIABLE_HINTS:
                if hint in lower:
                    logger.warning(
                        "acceptance_criteria[%d]=%r contains non-"
                        "falsifiable phrase %r — accepted but the "
                        "critic will likely flag it.",
                        i, s, hint,
                    )
                    break
            cleaned.append(s)
        return cleaned

    @field_validator("out_of_scope", "constraints", "references")
    @classmethod
    def _strip_string_lists(cls, v: list[str]) -> list[str]:
        return [s.strip() for s in v if isinstance(s, str) and s.strip()]


def task_from_free_text(
    user_request: str,
    *,
    title: str | None = None,
    kind: str = "feature",
) -> Task:
    """Build a minimal :class:`Task` from raw user text.

    Used by the pipeline as a backward-compat fallback when the chat
    agent / planner haven't produced a structured task yet. The
    description is the verbatim user_request; the title is either
    explicit or derived as the first sentence (truncated to 120 chars).

    The critic is told (via prompt) that fields like ``acceptance_criteria``
    were synthesised, not authored — so it can probe them rather than
    treat them as ground truth.
    """
    text = user_request.strip()
    if len(text) < 20:
        # Minimum description constraint — pad with marker so callers
        # see a clear error rather than a confusing pydantic message.
        raise ValueError(
            "user_request is too short (<20 chars) to build a Task; "
            "the chat agent should have asked for clarification first."
        )

    if title is None:
        # Take the first sentence-ish chunk as the title.
        first = re.split(r"(?<=[.!?])\s|\n", text, maxsplit=1)[0].strip()
        if not first:
            first = text
        title = first[:120].rstrip()
        if len(title) < 5:
            # Fallback: pad short titles to satisfy min_length=5.
            title = (title + " task")[:120]

    # Single synthesised acceptance criterion: "the user request is
    # satisfied". Honest about the lack of a real spec — the critic
    # can flag this as low-quality and ask for a refinement.
    return Task(
        title=title,
        description=text,
        kind=kind,
        acceptance_criteria=[
            "The user's request as written in <description> is "
            "satisfied to the user's confirmation."
        ],
    )


def is_synthesised(task: Task) -> bool:
    """True when the task looks like it was built via :func:`task_from_free_text`.

    Heuristic: a single acceptance criterion that mentions
    "user's confirmation" is the synthesised default. Used by prompt
    builders to add a "this task was auto-generated" preamble so the
    critic doesn't trust the criteria as user-authored.
    """
    if len(task.acceptance_criteria) != 1:
        return False
    return "user's confirmation" in task.acceptance_criteria[0].lower()
