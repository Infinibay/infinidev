"""Base classes for behavior checkers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Verdict:
    """A checker's judgement: positive delta = promote, negative = punish."""

    delta: int = 0
    reason: str = ""


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
