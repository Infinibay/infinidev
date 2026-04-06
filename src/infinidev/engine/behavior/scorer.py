"""BehaviorScorer — coordinates checkers and emits banners.

Singleton-ish: a process-global instance is held on the class. Scores
are kept in-memory per (project_id, agent_id) for the session.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from infinidev.engine.behavior.registry import enabled_checkers


@dataclass
class BehaviorEvent:
    """One scored verdict, kept in BehaviorScorer.history for /debug."""
    timestamp: float
    project_id: int
    agent_id: str
    checker: str
    delta: int
    reason: str
    score_after: int

logger = logging.getLogger(__name__)


class BehaviorScorer:
    _instance: "BehaviorScorer | None" = None
    _instance_lock = threading.Lock()

    HISTORY_LIMIT = 200

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._scores: dict[tuple[int, str], int] = {}
        self._history: deque[BehaviorEvent] = deque(maxlen=self.HISTORY_LIMIT)

    @classmethod
    def instance(cls) -> "BehaviorScorer":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def score_for(self, project_id: int, agent_id: str) -> int:
        with self._lock:
            return self._scores.get((project_id, agent_id), 0)

    def reset(self) -> None:
        with self._lock:
            self._scores.clear()
            self._history.clear()

    def history(self, agent_id: str | None = None) -> list[BehaviorEvent]:
        """Return a snapshot of the verdict history (newest last)."""
        with self._lock:
            items = list(self._history)
        if agent_id is None:
            return items
        return [e for e in items if e.agent_id == agent_id]

    def all_scores(self) -> dict[tuple[int, str], int]:
        with self._lock:
            return dict(self._scores)

    def on_model_message(self, ctx) -> None:  # HookContext
        """Run all enabled checkers against the latest message."""
        from infinidev.config.settings import settings

        if not getattr(settings, "BEHAVIOR_CHECKERS_ENABLED", False):
            return

        checkers = enabled_checkers()
        if not checkers:
            return

        meta = ctx.metadata or {}
        message = {
            "role": "assistant",
            "raw_content": meta.get("raw_content", ""),
            "reasoning_content": meta.get("reasoning_content", ""),
            "tool_calls": meta.get("tool_calls"),
        }
        history_window = int(getattr(settings, "BEHAVIOR_HISTORY_WINDOW", 4) or 4)
        full_messages = meta.get("messages") or []
        # Take last N assistant/tool/user messages, excluding system
        history = [m for m in full_messages if m.get("role") != "system"][-history_window:]

        # Split: prompt-only checkers run in ONE batched LLM call;
        # standalone LLMBehaviorChecker subclasses still run individually.
        from infinidev.engine.behavior.checker_base import PromptBehaviorChecker
        from infinidev.engine.behavior.batched_runner import run_batched

        prompt_checkers = [c for c in checkers if isinstance(c, PromptBehaviorChecker)]
        other_checkers = [c for c in checkers if not isinstance(c, PromptBehaviorChecker)]

        task = meta.get("task", "")
        plan_snapshot = meta.get("plan_snapshot") or {}

        if prompt_checkers:
            verdicts = run_batched(
                prompt_checkers, message, history,
                task=task, plan_snapshot=plan_snapshot,
            )
            for checker in prompt_checkers:
                v = verdicts.get(checker.name)
                if v is None or v.delta == 0:
                    continue
                self._apply(ctx, checker.name, v.delta, v.reason)

        for checker in other_checkers:
            try:
                verdict = checker.check(message, history)
            except Exception:
                logger.debug("Checker %s raised", checker.name, exc_info=True)
                continue
            if not verdict or verdict.delta == 0:
                continue
            self._apply(ctx, checker.name, verdict.delta, verdict.reason)

    def _apply(self, ctx, checker_name: str, delta: int, reason: str) -> None:
        key = (ctx.project_id, ctx.agent_id)
        with self._lock:
            new_score = self._scores.get(key, 0) + delta
            self._scores[key] = new_score
            self._history.append(BehaviorEvent(
                timestamp=time.time(),
                project_id=ctx.project_id,
                agent_id=ctx.agent_id,
                checker=checker_name,
                delta=delta,
                reason=reason,
                score_after=new_score,
            ))

        from infinidev.flows.event_listeners import event_bus

        event_bus.emit(
            "loop_behavior_update",
            ctx.project_id,
            ctx.agent_id,
            {
                "agent_id": ctx.agent_id,
                "score": new_score,
                "delta": delta,
                "reason": reason,
                "checker": checker_name,
            },
        )

        # Classic CLI fallback when no UI subscribers
        if not event_bus.has_subscribers:
            from infinidev.engine.engine_logging import log, YELLOW, GREEN, RESET

            color = GREEN if delta > 0 else YELLOW
            sign = "★" if delta > 0 else "⚠"
            verb = "promote" if delta > 0 else "stop doing that"
            log(
                f"{color}{sign} {verb} ({reason}) — behavior points: {new_score}{RESET}"
            )
