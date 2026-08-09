"""Pydantic models for the plan-execute-summarize loop engine."""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Literal

from pydantic import BaseModel, Field

from infinidev.engine.loop.plan_step import PlanStep
from infinidev.engine.loop.step_operation import StepOperation

logger = logging.getLogger(__name__)

_SIMILAR_OPEN_STEP_THRESHOLD = 0.80

_STEP_PHASE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("discover", re.compile(
        r"\b(?:analy[sz]e|explore|find|identify|inspect|investigate|locate|read|search|trace|understand)\b"
    )),
    ("change", re.compile(
        r"\b(?:add|change|create|edit|fix|implement|refactor|rewrite|update)\b"
    )),
    ("verify", re.compile(
        r"\b(?:check|run|test|validate|verification|verify)\b"
    )),
    ("document", re.compile(r"\b(?:document|documentation|explain|write docs?)\b")),
    ("design", re.compile(r"\b(?:design|plan|prototype)\b")),
)
_PATH_RE = re.compile(r"(?:[a-zA-Z0-9_.-]+/)+[a-zA-Z0-9_.-]+")
_IDENTIFIER_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")


def _normalized_step_title(title: str) -> str:
    """Normalize superficial title differences before overlap scoring."""
    return " ".join(re.findall(r"[a-z0-9_./-]+", title.casefold()))


def _step_phase(title: str) -> str:
    """Classify the action without asking another model."""
    normalized = title.casefold()
    for phase, pattern in _STEP_PHASE_PATTERNS:
        if pattern.search(normalized):
            return phase
    return ""


def _step_targets(title: str) -> set[str]:
    """Extract concrete paths and code identifiers named by a plan step."""
    normalized = title.casefold()
    paths = {match.rstrip(".,:;)") for match in _PATH_RE.findall(normalized)}
    identifiers = set(_IDENTIFIER_RE.findall(normalized))
    return paths | identifiers

# Fields the model may refine on a user-approved step.
#
# The freeze protects the user's *scope*, not the planner's prose. A title the
# run has just proved misleading, an explanation naming the wrong file, a
# success criterion written before anyone had read the code — correcting those
# is not the same act as dropping a feature the user asked for, and the single
# ``user_approved`` boolean refused all of them with equal force.
#
# The fields worth defending need no entry here, because ``StepOperation``
# cannot express them: it carries no ``detail``, no ``verify`` and no
# ``status``, so the planner's researched guidance and its adversarial check
# are unreachable by construction. The allowlist is spelled out anyway so that
# adding a field to ``StepOperation`` later cannot silently widen the freeze —
# a new field is refused until someone names it here on purpose.
APPROVED_MUTABLE_FIELDS: tuple[str, ...] = ("title", "explanation", "expected_output")


class LoopPlan(BaseModel):
    """The agent's mutable execution plan."""

    steps: list[PlanStep] = Field(default_factory=list)
    # Stable prose narrative set once by the planner (what, why, which files,
    # validation approach). Rendered every iteration as <plan-overview>;
    # apply_operations never mutates this field — it is immutable after
    # the LoopEngine populates it from an external Plan.
    overview: str = ""
    # ``0`` means unrestricted (legacy plans). A rolling Task keeps at most
    # this many active/pending Steps; completed history remains intact.
    rolling_horizon_limit: int = 0

    @property
    def active_step(self) -> PlanStep | None:
        """Return the first step with status='active', or None."""
        for step in self.steps:
            if step.status == "active":
                return step
        return None

    @property
    def has_pending(self) -> bool:
        """True if any step is pending or active."""
        return any(s.status in ("pending", "active") for s in self.steps)

    def mark_active(self, status: Literal["done", "blocked"]) -> None:
        """Close the active step with an explicit terminal status."""
        for step in self.steps:
            if step.status == "active":
                step.status = status
                break

    def mark_active_done(self) -> None:
        """Mark the current active step as done (without activating next)."""
        self.mark_active("done")

    def undischarged(
        self,
        exclude_index: int | None = None,
        *,
        approved_only: bool = False,
    ) -> list[PlanStep]:
        """Steps the run has not dealt with, one way or another.

        The approved ones among these are the record of what the user asked
        for. They are already in LoopState and already reach the reviewer.
        approved_only selects that scope record for the completion gate; the
        default includes model-authored steps so diagnostics can report every
        stranded item.

        exclude_index skips the step being closed right now. At gate time that
        step is still active, so counting it would refuse every correct close.
        """
        return [
            step for step in self.steps
            if step.status in ("pending", "active")
            and step.index != exclude_index
            and (not approved_only or step.user_approved)
        ]

    def activate_next(self) -> None:
        """Activate the next pending step."""
        for step in self.steps:
            if step.status == "pending":
                step.status = "active"
                break

    def find_similar_open_step(self, title: str) -> PlanStep | None:
        """Return an existing open step that substantially duplicates ``title``.

        Plan titles are short and structured, so character-sequence overlap is
        both cheaper and more predictable than another model call.  The high
        threshold keeps adjacent phases such as "implement" and "test" distinct
        while catching cosmetic restatements of the same work.
        """
        normalized = _normalized_step_title(title)
        if not normalized:
            return None
        phase = _step_phase(title)
        targets = _step_targets(title)
        for step in self.steps:
            if step.status not in ("pending", "active"):
                continue
            existing = _normalized_step_title(step.title)
            if not existing:
                continue
            similarity = SequenceMatcher(None, normalized, existing).ratio()
            same_concrete_work = (
                bool(phase)
                and phase == _step_phase(step.title)
                and bool(targets & _step_targets(step.title))
            )
            if similarity >= _SIMILAR_OPEN_STEP_THRESHOLD or same_concrete_work:
                return step
        return None

    def advance(self) -> None:
        """Mark the active step as done and activate the next pending step."""
        self.mark_active_done()
        self.activate_next()

    def apply_operations(self, ops: list[StepOperation]) -> None:
        """Apply structured add/modify/remove operations to the plan.

        Protections:
        - ``done`` steps are never replaced or removed.
        - ``user_approved`` steps cannot be *removed*, and an add cannot
          displace one that is already running. They can be refined —
          see :data:`APPROVED_MUTABLE_FIELDS` for which fields and why.
        - Bulk removal of all pending steps is blocked (max 50% can be removed
          per call) to prevent the LLM from accidentally wiping the plan.
        - An add never destroys a pending step. Landing on an occupied slot
          renumbers rather than overwrites; see :meth:`_apply_add`.
        """
        approved_indices = {s.index for s in self.steps if s.user_approved}
        filtered_ops: list[StepOperation] = []
        for op in ops:
            if op.op == "remove" and op.index in approved_indices:
                logger.warning(
                    "Rejected remove op on user-approved step %d — dropping a "
                    "step the user asked for is not the model's call",
                    op.index,
                )
                continue
            filtered_ops.append(op)
        ops = filtered_ops

        # Count how many pending/active steps would be removed
        pending_count = sum(1 for s in self.steps if s.status in ("pending", "active"))
        remove_count = sum(1 for op in ops if op.op == "remove")
        if pending_count > 0 and remove_count >= pending_count:
            logger.warning(
                "Blocked bulk removal: %d remove ops for %d pending steps — "
                "keeping existing plan and only applying add/modify ops",
                remove_count, pending_count,
            )
            ops = [op for op in ops if op.op != "remove"]

        for op in ops:
            if op.op == "add":
                self._apply_add(op)

            elif op.op == "modify":
                for step in self.steps:
                    if (
                        step.index == op.index
                        and step.status in ("pending", "active")
                    ):
                        self._apply_modify(step, op)
                        break

            elif op.op == "remove":
                for step in self.steps:
                    if step.index == op.index and step.status in ("pending", "active"):
                        step.status = "skipped"
                        break

        self.steps.sort(key=lambda s: s.index)

    @staticmethod
    def _apply_modify(step: PlanStep, op: StepOperation) -> None:
        """Write the operation's non-empty fields onto ``step``.

        On an approved step only :data:`APPROVED_MUTABLE_FIELDS` are written.
        Today that is every field ``StepOperation`` has, so nothing is dropped;
        the filter exists so that a field added to the operation later has to
        be admitted deliberately rather than by inheritance.
        """
        writable = (
            APPROVED_MUTABLE_FIELDS
            if step.user_approved
            else ("title", "explanation", "expected_output")
        )
        for field in writable:
            value = getattr(op, field, "")
            if value:
                setattr(step, field, value)

    def _apply_add(self, op: StepOperation) -> None:
        """Insert a new step, renumbering rather than overwriting.

        ``add(index=N)`` used to delete whatever pending step held slot N,
        which made "insert a prerequisite I just discovered" indistinguishable
        from "destroy the step that depends on it". An occupied slot now shifts
        the pending tail down by one instead.

        Only *pending* steps are renumbered. A done step's index is a foreign
        key — ``ActionRecord.step_index`` and ``working_memory.step_index``
        both point at it — and the active step's index keys the gate's
        once-per-step guarantees (``_note_fired``, ``_hook_fired``,
        ``_verify_attempts``). Neither may move, so an add that would displace
        one is refused outright rather than silently appended.
        """
        open_steps = sum(
            step.status in ("pending", "active") for step in self.steps
        )
        if self.rolling_horizon_limit and open_steps >= self.rolling_horizon_limit:
            logger.info(
                "Ignored added step %r because the rolling horizon is full (%d/%d)",
                op.title,
                open_steps,
                self.rolling_horizon_limit,
            )
            return

        duplicate = self.find_similar_open_step(op.title)
        if duplicate is not None:
            logger.info(
                "Ignored duplicate open step %r; it overlaps step %d (%r)",
                op.title,
                duplicate.index,
                duplicate.title,
            )
            return

        target = op.index
        occupant = next((s for s in self.steps if s.index == target), None)

        if occupant is not None:
            if occupant.status != "pending":
                logger.warning(
                    "Rejected add at index %d — slot held by a %s step, whose "
                    "index is referenced by archived records or by the "
                    "step gate and cannot move",
                    target, occupant.status,
                )
                return
            immovable = [
                s for s in self.steps
                if s.index >= target and s.status != "pending"
            ]
            if immovable:
                logger.warning(
                    "Rejected add at index %d — renumbering would move %s "
                    "step %d, whose index is already referenced",
                    target, immovable[0].status, immovable[0].index,
                )
                return
            for step in self.steps:
                if step.index >= target and step.status == "pending":
                    step.index += 1

        self.steps.append(PlanStep(
            index=target,
            title=op.title,
            explanation=op.explanation,
            expected_output=op.expected_output,
        ))

    def render(self) -> str:
        """Render the plan as the run's frontier: what is settled, what is not.

        A closed step carries what it established and the labels of the
        evidence behind it. Those labels are the record titles
        ``working_memory`` stored, so each one is a query that pulls the raw
        tool output back — the plan block indexes the archive rather than
        merely naming steps that already happened.
        """
        lines: list[str] = []
        for step in self.steps:
            tag = f"[{step.status}] " if step.status != "pending" else ""
            line = f"{step.index}. {tag}{step.title}"
            if step.conclusion:
                line += f" — established: {step.conclusion}"
            if step.evidence:
                shown = ", ".join(step.evidence[:2])
                extra = len(step.evidence) - 2
                more = f" +{extra} more" if extra > 0 else ""
                line += f" [evidence, recall_context these: {shown}{more}]"
            lines.append(line)
        return "\n".join(lines)
