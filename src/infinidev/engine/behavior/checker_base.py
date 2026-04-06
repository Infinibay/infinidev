"""Base classes for behavior checkers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

# ── Verdict TTL constants ────────────────────────────────────────────────
# Measured in scorer *ticks* — one tick per on_step/on_model_message
# invocation. Checkers assign their natural lifetime via the
# ``ttl_steps`` class attribute so the score stops reflecting stale
# events that no longer describe current behavior.
#
# Rule of thumb:
#   • Ephemeral punishments (chatty thinking, prompt pollution) = 1
#   • Tactical punishments (ignored error, repetition) = 3
#   • Pattern-level punishments (plan drift, lazy work) = 5
#   • Rewards outlive punishments — credit should stick while you
#     weigh subsequent mistakes, not vanish the moment the agent errs.
#   • Infinite (-1) is reserved for truly structural signals.

TTL_EPHEMERAL: int = 1    # visible only on the firing step
TTL_SHORT: int = 3        # next few steps
TTL_MEDIUM: int = 5       # lingers ~5 steps
TTL_LONG: int = 10        # rewards and serious patterns
TTL_INFINITE: int = -1    # never expires


@dataclass
class Verdict:
    """A checker's judgement: positive delta = promote, negative = punish.

    ``trigger_key`` uniquely identifies the *evidence* that made the
    checker fire — a tool-result hash, a command string, a reasoning
    block hash, etc. The scorer dedupes verdicts on
    ``(checker_name, trigger_key)`` so the same underlying event never
    gets scored twice even if a sliding history window exposes it to
    multiple consecutive evaluations. If left empty, the scorer falls
    back to hashing ``reason``.
    """

    delta: int = 0
    reason: str = ""
    trigger_key: str = ""


class BehaviorChecker(ABC):
    """Base class for any checker that scores model behavior.

    Subclasses define ``name``, ``description``, and ``default_enabled``,
    then implement :meth:`check`.
    """

    name: str = "unnamed"
    description: str = ""
    default_enabled: bool = False
    # Optional one-line note shown next to the toggle in the /settings dialog.
    # If empty, ``description`` is used.
    settings_message: str = ""
    # How many scorer ticks a verdict from this checker remains "live"
    # and counted toward the agent's score. ``TTL_INFINITE`` (-1) =
    # never expires. See the TTL_* constants above.
    ttl_steps: int = TTL_INFINITE

    @classmethod
    def settings_label(cls) -> str:
        return cls.settings_message or cls.description or cls.name

    @abstractmethod
    def check(
        self,
        message: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> Verdict:
        """Inspect the latest *message* + recent *history* and return a Verdict.

        ``message`` keys: ``raw_content``, ``reasoning_content``, ``tool_calls``.
        ``history`` is a list of the last N message dicts as fed to the LLM
        (role/content pairs).
        """


class PromptBehaviorChecker(BehaviorChecker):
    """A checker that contributes a single criterion to one batched LLM call.

    Subclasses only declare ``criteria`` (the rule to evaluate) and
    ``delta_range`` (min, max). They never call the LLM themselves —
    :class:`BatchedBehaviorRunner` collects all enabled prompt checkers
    and asks one LLM call to score all of them at once.
    """

    criteria: str = ""
    delta_range: tuple[int, int] = (-3, 3)

    def check(self, message, history):  # pragma: no cover - never called directly
        # Returning a no-op Verdict keeps compatibility if someone runs a
        # PromptBehaviorChecker through the legacy single-call path.
        return Verdict(0, "")


class StochasticChecker(PromptBehaviorChecker):
    """A checker that evaluates deterministically/stochastically.

    Subclasses implement :meth:`evaluate`, which composes primitives
    from :mod:`infinidev.engine.behavior.primitives` to produce a
    :class:`Verdict`. No LLM call is made.

    Inheriting from :class:`PromptBehaviorChecker` keeps the class
    compatible with the legacy LLM judge (``BEHAVIOR_JUDGE_MODE='llm'``)
    and with hybrid escalation: if a subclass also declares ``criteria``,
    it can fall back to the batched LLM runner when stochastic returns
    nothing.
    """

    def evaluate(self, ctx) -> "Verdict | None":  # noqa: F821  (forward ref)
        raise NotImplementedError

    def check(self, message, history):  # pragma: no cover
        return Verdict(0, "")
