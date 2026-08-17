"""Budget + continuation policy for the autonomous ("manejate vos") loop.

When the user asks the engine to keep going without intervention, the pipeline
chains multiple plan→execute cycles back-to-back. Without a budget the loop
will happily spend the user's API quota on a thread that lost the plot two
plans ago. The structures here cap the chain by four orthogonal fuses:

* ``max_plans`` — hard cap on how many plans the chain may execute.
* ``token_budget`` — soft cap on cumulative prompt tokens consumed.
* ``wall_seconds`` — wall-clock cap on the whole chain.
* ``idle_passes`` — how many consecutive "nothing to do" reports the chain
  tolerates before giving up. ``idle_passes=2`` means the chain retires after
  seeing two adjacent plans that each surfaced zero new work.

Design decisions
----------------
* **Mutable dataclass, not frozen.** The topes (read once from settings) are
  fixed for a chain, but the counters (``plans_executed``,
  ``tokens_consumed``, ``idle_runs``, ``wall_started_at``) are updated on
  every plan. ``frozen=True`` would force a ``replace`` per update, which is
  the kind of allocation noise that does not belong in a hot loop.
* **Pure query ``should_continue``.** Takes the budget + the last outcome
  string and returns a bool. The caller is responsible for mutating the
  counters; the helper never touches them. This keeps the budget cheap to
  call from anywhere and impossible to mis-use as a side-effect.
* **Defaults are the conservative defaults, not zero.** ``AutonomousBudget()``
  without arguments is a fully usable chain with sensible topes (3 plans,
  50k tokens, 15 minutes, 2 idle passes). The tests can build it without
  first wiring up settings; the pipeline can re-derive it from settings via
  :func:`from_settings` when the user opts in.
* **No I/O.** The module never reads the DB, never calls the LLM, never
  touches the chat agent. Integration glue belongs in the pipeline step.

The outcomes the chain cares about (``done`` / ``blocked`` / ``error`` /
``idle`` / ``continue``) match the values the pipeline emits via
``engine_run.status`` and the chat-agent ``kind`` field. Anything else is
treated as a "continue" signal — useful so new outcome values do not silently
stop the chain.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, Literal

logger = logging.getLogger(__name__)


# Default topes. Used by ``AutonomousBudget()`` so the constructor is usable
# without settings, and by ``from_settings`` when a settings field is missing
# or non-positive (defensive: a user-set 0 in the env should not mean
# "infinite", a positive default is the only contract that keeps the chain
# bounded).
DEFAULT_MAX_PLANS: int = 3
# 200k tokens is enough headroom for an ordinary plan to complete (a typical
# review-rework loop spends 20-60k tokens of prompt alone) while still being
# small enough that three plans cannot accidentally run a user's quota into
# the ground. The previous 50k default tripped the chain on the first plan,
# which is the root cause of the "autonomous mode stops all the time" bug.
DEFAULT_TOKEN_BUDGET: int = 200_000
DEFAULT_WALL_SECONDS: int = 900
DEFAULT_IDLE_PASSES: int = 2


# Outcome vocabulary the chain reacts to. Anything else is treated as a
# "continue" signal so the chain does not silently die when a new outcome
# value is introduced.
#
# ``soft_blocked`` is the non-terminal variant of ``blocked``: the engine has
# something to surface to the user (usually a clarification question) but
# the chain must keep going. The pipeline translates it into a chat-agent
# notification + continuation, instead of stopping the chain.
AutonomousOutcome = Literal[
    "continue", "done", "idle", "blocked", "error", "soft_blocked",
]
TERMINAL_OUTCOMES: frozenset[str] = frozenset({"done", "blocked", "error"})
# Outcomes that mean "the engine asked a question / surfaced something for
# the user, but the chain has more work to do". ``should_continue`` returns
# True for these so the chain does not stop on a clarification.
SOFT_BLOCKED_OUTCOMES: frozenset[str] = frozenset({"soft_blocked"})
_IDLE_OUTCOME: str = "idle"


def _normalise_outcome(outcome: str | None) -> str:
    """Lower-case + strip; missing/empty becomes ``"continue"``.

    The pipeline sometimes emits ``"Completed"`` because the engine surface
    is not strictly normalised; the budget tries to be tolerant of casing
    rather than crashing on a value that's clearly "done".
    """
    if not outcome:
        return "continue"
    return str(outcome).strip().lower() or "continue"


@dataclass
class AutonomousBudget:
    """Per-chain budget + counters for the autonomous loop.

    The topes are fixed at construction (or loaded from settings via
    :func:`from_settings`); the counters are updated by the pipeline as each
    plan completes. ``last_outcome`` carries the most recent result so the
    budget can detect ``idle`` runs without an extra parameter.
    """

    max_plans: int = DEFAULT_MAX_PLANS
    token_budget: int = DEFAULT_TOKEN_BUDGET
    wall_seconds: int = DEFAULT_WALL_SECONDS
    idle_passes: int = DEFAULT_IDLE_PASSES

    # Counters — mutated by the pipeline, never by ``should_continue``.
    plans_executed: int = 0
    tokens_consumed: int = 0
    idle_runs: int = 0
    wall_started_at: float = field(default=0.0)
    last_outcome: str | None = None

    # ── Lifecycle helpers ─────────────────────────────────────────────

    def start(self) -> None:
        """Stamp the wall-clock anchor. Idempotent — calling twice is safe
        only when the counter set is zero; callers that restart mid-chain
        should explicitly ``reset_counters`` first.
        """
        if self.wall_started_at <= 0.0:
            self.wall_started_at = time.monotonic()

    def reset_counters(self) -> None:
        """Zero the counters + re-anchor the wall clock. Useful when the
        chain is recycled (e.g. a fresh user-level toggle without a
        process restart).
        """
        self.plans_executed = 0
        self.tokens_consumed = 0
        self.idle_runs = 0
        self.wall_started_at = time.monotonic()
        self.last_outcome = None

    def record_outcome(self, outcome: str, *, tokens_used: int = 0) -> None:
        """Update counters after a plan completes.

        ``tokens_used`` is added to the cumulative token counter when the
        caller knows the per-plan consumption. It is convenient to pass zero
        when the caller does not track tokens, in which case the chain is
        bounded by the other three fuses alone.
        """
        normalised = _normalise_outcome(outcome)
        self.last_outcome = normalised
        self.plans_executed += 1
        if tokens_used > 0:
            self.tokens_consumed += int(tokens_used)
        if normalised == _IDLE_OUTCOME:
            self.idle_runs += 1

    # ── Derived views ─────────────────────────────────────────────────

    @property
    def wall_elapsed(self) -> float:
        """Seconds since ``start`` was first called. Returns ``0.0`` when
        the chain has not started yet, so callers can render a clean
        "0/N" line instead of a confusing negative-elapsed figure.
        """
        if self.wall_started_at <= 0.0:
            return 0.0
        return max(0.0, time.monotonic() - self.wall_started_at)

    @property
    def wall_remaining(self) -> float:
        """Seconds left on the wall fuse. Can be negative when the cap is
        exceeded; callers should not rely on it being non-negative.
        """
        if self.wall_seconds <= 0:
            return float("inf")
        return float(self.wall_seconds) - self.wall_elapsed

    # ── Bounded construction ───────────────────────────────────────────

    @classmethod
    def from_settings(cls, settings: Any | None = None) -> "AutonomousBudget":
        """Read topes from a settings object (or the default settings).

        Defensive: each field is clamped to a positive integer using the
        module-level defaults when the configured value is missing, None,
        or non-positive. Negative or zero values would otherwise create
        a chain that stops immediately (max_plans=0) or runs forever
        (wall_seconds=0, but we treat that as "use the default").
        """
        if settings is None:
            from infinidev.config.settings import settings as _settings

            settings = _settings

        def _positive(field_name: str, default: int) -> int:
            raw = getattr(settings, field_name, None)
            try:
                value = int(raw) if raw is not None else 0
            except (TypeError, ValueError):
                value = 0
            return value if value > 0 else default

        return cls(
            max_plans=_positive("AUTONOMOUS_MAX_PLANS", DEFAULT_MAX_PLANS),
            token_budget=_positive("AUTONOMOUS_TOKEN_BUDGET", DEFAULT_TOKEN_BUDGET),
            wall_seconds=_positive("AUTONOMOUS_WALL_SECONDS", DEFAULT_WALL_SECONDS),
            idle_passes=_positive("AUTONOMOUS_IDLE_PASSES", DEFAULT_IDLE_PASSES),
        )


def should_continue(budget: AutonomousBudget, last_outcome: str | None) -> bool:
    """Whether the autonomous chain should start another plan.

    Pure query — does not mutate ``budget``. The caller is expected to call
    :meth:`AutonomousBudget.record_outcome` first so the counters reflect
    the result that is being evaluated.

    Returns ``False`` when any of these hold:
      * ``last_outcome`` is one of ``done`` / ``blocked`` / ``error``.
      * ``plans_executed`` has reached ``max_plans``.
      * ``tokens_consumed`` has reached ``token_budget``.
      * ``wall_elapsed`` has reached ``wall_seconds``.
      * ``last_outcome`` is ``idle`` and ``idle_runs`` has reached
        ``idle_passes`` (the chain stops after the configured number of
        consecutive "nothing to do" reports).

    An unknown outcome value is treated as ``continue`` so a new outcome
    added in the pipeline does not silently stop the chain.
    ``soft_blocked`` is treated as ``continue`` — see ``SOFT_BLOCKED_OUTCOMES``
    for the rationale (engine asked a question; chain keeps going).
    """
    outcome = _normalise_outcome(last_outcome)

    if outcome in TERMINAL_OUTCOMES:
        return False

    if budget.plans_executed >= budget.max_plans:
        return False

    if budget.token_budget > 0 and budget.tokens_consumed >= budget.token_budget:
        return False

    if budget.wall_seconds > 0 and budget.wall_elapsed >= budget.wall_seconds:
        return False

    if outcome == _IDLE_OUTCOME and budget.idle_runs >= budget.idle_passes:
        return False

    return True


def budget_status_text(budget: AutonomousBudget) -> str:
    """Short, human-readable summary used in logs and UI banners.

    Always returns a non-empty string so the caller can drop it into a
    ``f"…{budget_status_text(budget)}…"`` without guarding. The wall-clock
    row is rendered as ``mm:ss`` when under an hour, otherwise as ``h:mm:ss``;
    this matches the format the rest of the engine uses for status lines.
    """
    elapsed = int(budget.wall_elapsed)
    if budget.wall_seconds <= 0:
        wall_part = f"wall {elapsed}s/∞"
    else:
        wall_part = f"wall {elapsed}s/{budget.wall_seconds}s"
    return (
        f"plan {budget.plans_executed}/{budget.max_plans} • "
        f"tokens {budget.tokens_consumed}/{budget.token_budget} • "
        f"{wall_part} • "
        f"idle {budget.idle_runs}/{budget.idle_passes}"
    )


def stop_reason(budget: AutonomousBudget) -> str | None:
    """One-line explanation of why ``should_continue`` would stop now.

    Returns ``None`` when the chain is still allowed to continue. Useful for
    the chat-agent's closing message after the chain ends — the user gets a
    concrete reason ("ran out of plans", "wall clock exceeded", "no work
    twice in a row") instead of a vague "stopped".
    """
    outcome = _normalise_outcome(budget.last_outcome)
    if outcome in TERMINAL_OUTCOMES:
        return f"engine reported {outcome}"
    if budget.plans_executed >= budget.max_plans:
        return f"reached max_plans={budget.max_plans}"
    if budget.token_budget > 0 and budget.tokens_consumed >= budget.token_budget:
        return f"reached token_budget={budget.token_budget}"
    if budget.wall_seconds > 0 and budget.wall_elapsed >= budget.wall_seconds:
        return f"reached wall_seconds={budget.wall_seconds}"
    if outcome == _IDLE_OUTCOME and budget.idle_runs >= budget.idle_passes:
        return f"no new work in {budget.idle_runs} consecutive plans"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Intent detection — does the user's text signal "manejate vos" mode?
# ─────────────────────────────────────────────────────────────────────────────

# Phrases the user types when they want the engine to keep going without
# checking in. Permissive list — broader than the regex needs to be, because
# the cost of a missed detection is "the user has to type the same thing
# again next turn", which is annoying but cheap. The cost of a false
# positive is "the chain runs 2-3 plans with no human in the loop", which
# is what the user asked for anyway.
_AUTONOMOUS_PHRASES: tuple[str, ...] = (
    # Spanish (the user's primary language in this project)
    r"\bmanej[áa]te\s+vos\b",
    r"\bmanej[áa]te\s+v[óo]s?\b",
    r"\bsiga\s+investigando\b",
    r"\bsiga\s+vos\b",
    r"\bsiga\s+v[óo]s?\b",
    r"\bvos\s+solo\b",
    r"\bvos\s+sola\b",
    r"\bsegu[ií]\s+vos\b",
    r"\ba\s+todo\s+trapo\b",
    r"\btom[áa]\s+(?:el|lo)\s+control\b",
    r"\bsegu[ií]\s+adelante\b",
    r"\bsin\s+preguntarme\b",
    r"\bno\s+me\s+preguntes\b",
    r"\bsin\s+pedirme\b",
    r"\bmanej[áa]lo\s+vos\b",
    r"\bmanej[áa]lo\s+solo\b",
    # English (model may also surface this)
    r"\byou\s+(?:handle|run|take)\s+it\b",
    r"\bjust\s+(?:handle|do)\s+it\b",
    r"\bdo\s+your\s+thing\b",
    r"\bkeep\s+going\b",
    r"\bwithout\s+(?:asking|checking\s+in)\b",
    r"\bautonomous(?:ly)?\b",
)
_AUTONOMOUS_RE = re.compile(
    "|".join(_AUTONOMOUS_PHRASES), re.IGNORECASE | re.UNICODE,
)


def detect_autonomous_intent(text: str) -> bool:
    """True when ``text`` requests autonomous ("manejate vos") behaviour.

    Used both on the raw user input and on the ``user_signal`` the chat
    agent fills in on escalation. ``user_signal`` often paraphrases the
    user message and frequently contains the literal phrase too — checking
    both lets the detection succeed whether the chat agent echoed the
    user's words verbatim or summarised them.
    """
    if not text:
        return False
    return bool(_AUTONOMOUS_RE.search(text))


def apply_autonomous_to_packet(
    packet: Any,
    user_input: str | None = None,
    *,
    explicit_hint: bool = False,
) -> Any:
    """Return a copy of ``packet`` with ``autonomous=True`` when intent matches.

    ``packet`` is expected to expose an ``autonomous`` boolean attribute
    (``EscalationPacket.autonomous`` added in this step). The helper keeps
    the :class:`dataclasses.replace`-style immutability: it never mutates
    the original packet, so frozen-dataclass semantics are honoured.

    Detection runs when *any* of these hold:
      * the caller passed ``explicit_hint=True`` (used by the pipeline's
        ``autonomous`` kwarg to force-enable the chain);
      * the supplied ``user_input`` matches the autonomous phrase;
      * the packet's stored ``user_request`` matches (useful when no
        ``user_input`` was forwarded and the literal request is the
        only copy);
      * the packet's ``user_signal`` matches (the chat agent frequently
        records the user's literal phrase here even when paraphrasing
        elsewhere).
    """
    already = bool(getattr(packet, "autonomous", False))
    if already:
        return packet
    if (
        explicit_hint
        or detect_autonomous_intent(user_input or "")
        or detect_autonomous_intent(getattr(packet, "user_request", "") or "")
        or detect_autonomous_intent(getattr(packet, "user_signal", "") or "")
    ):
        try:
            return replace(packet, autonomous=True)
        except (TypeError, ValueError):
            # replace() raises when ``autonomous`` is not a dataclass field
            # (older test doubles, mocks). Fall through rather than crashing
            # the chain on a noisy log.
            return packet
    return packet


__all__ = [
    "AutonomousBudget",
    "AutonomousOutcome",
    "DEFAULT_IDLE_PASSES",
    "DEFAULT_MAX_PLANS",
    "DEFAULT_TOKEN_BUDGET",
    "DEFAULT_WALL_SECONDS",
    "SOFT_BLOCKED_OUTCOMES",
    "TERMINAL_OUTCOMES",
    "apply_autonomous_to_packet",
    "budget_status_text",
    "detect_autonomous_intent",
    "from_settings",
    "should_continue",
    "stop_reason",
]


# Late-bound re-export so ``from_settings`` is reachable both as a method on
# the dataclass and as a module-level helper (matches the snake_case usage
# anticipated by the test suite and the eventual pipeline wiring).
from_settings = AutonomousBudget.from_settings
