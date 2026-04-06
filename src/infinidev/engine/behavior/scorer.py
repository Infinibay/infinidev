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
    """One scored verdict, kept in BehaviorScorer.history for /debug.

    ``tick_at_fire`` / ``ttl_ticks`` define the event's lifetime. The
    scorer expires events whose lifetime has elapsed so the agent's
    live score reflects *current* behavior, not every mistake the
    agent has ever made. ``score_after`` is a snapshot of the score
    at firing time — purely informational, not authoritative.
    """
    timestamp: float
    project_id: int
    agent_id: str
    checker: str
    delta: int
    reason: str
    score_after: int
    tick_at_fire: int = 0
    ttl_ticks: int = -1       # -1 = never expires
    expired: bool = False     # flipped by _expire() when TTL elapses

    def is_live(self, current_tick: int) -> bool:
        if self.expired:
            return False
        if self.ttl_ticks < 0:
            return True
        return (current_tick - self.tick_at_fire) < self.ttl_ticks

logger = logging.getLogger(__name__)


class BehaviorScorer:
    _instance: "BehaviorScorer | None" = None
    _instance_lock = threading.Lock()

    HISTORY_LIMIT = 200

    # Cap the per-agent fingerprint set so long sessions don't grow
    # unbounded. 500 unique verdict fingerprints per agent is plenty —
    # a task producing more than that is either pathological or the
    # checkers themselves are mis-tuned.
    FINGERPRINT_CAP = 500

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._history: deque[BehaviorEvent] = deque(maxlen=self.HISTORY_LIMIT)
        # Dedup set: (project_id, agent_id) -> set of
        # "<checker_name>|<trigger_key or reason_hash>" fingerprints.
        self._fired_fingerprints: dict[tuple[int, str], set[str]] = {}
        # Per-agent monotonic tick counter. Incremented once per
        # on_step / on_model_message invocation; drives TTL expiration.
        self._ticks: dict[tuple[int, str], int] = {}

    @classmethod
    def instance(cls) -> "BehaviorScorer":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def score_for(self, project_id: int, agent_id: str) -> int:
        """Return the agent's *current* score — a view over live events.

        Expired events are excluded. This is a recompute rather than a
        cached sum because expirations can happen between applies, and
        keeping a running counter consistent with TTLs is more error
        prone than just summing ~hundreds of events on demand.
        """
        with self._lock:
            current = self._ticks.get((project_id, agent_id), 0)
            total = 0
            for e in self._history:
                if e.project_id != project_id or e.agent_id != agent_id:
                    continue
                if e.is_live(current):
                    total += e.delta
            return total

    def _tick(self, project_id: int, agent_id: str) -> int:
        """Bump and return the agent's tick counter."""
        key = (project_id, agent_id)
        with self._lock:
            self._ticks[key] = self._ticks.get(key, 0) + 1
            return self._ticks[key]

    def _expire(self, ctx) -> int:
        """Mark newly-expired events for (project_id, agent_id). Returns
        the *number* of events that just expired so callers can decide
        whether to emit a UI update.
        """
        key = (ctx.project_id, ctx.agent_id)
        expired_count = 0
        with self._lock:
            current = self._ticks.get(key, 0)
            for e in self._history:
                if e.project_id != ctx.project_id or e.agent_id != ctx.agent_id:
                    continue
                if e.expired:
                    continue
                if e.ttl_ticks < 0:
                    continue
                if (current - e.tick_at_fire) >= e.ttl_ticks:
                    e.expired = True
                    expired_count += 1
        if expired_count > 0:
            # Notify subscribers so /debug and the banner reflect the
            # new score. The score recompute happens inside score_for.
            new_score = self.score_for(ctx.project_id, ctx.agent_id)
            from infinidev.flows.event_listeners import event_bus

            event_bus.emit(
                "loop_behavior_update",
                ctx.project_id,
                ctx.agent_id,
                {
                    "agent_id": ctx.agent_id,
                    "score": new_score,
                    "delta": 0,
                    "reason": f"{expired_count} event(s) expired",
                    "checker": "_ttl_expiration",
                },
            )
        return expired_count

    def reset(self) -> None:
        with self._lock:
            self._history.clear()
            self._fired_fingerprints.clear()
            self._ticks.clear()
        # Clear per-checker rolling state as well. Imported lazily to
        # avoid a circular dependency at module load time.
        try:
            from infinidev.engine.behavior.checkers.plan_drift import (
                _reset_state as _reset_plan_drift,
            )

            _reset_plan_drift()
        except Exception:
            pass

    def history(self, agent_id: str | None = None) -> list[BehaviorEvent]:
        """Return a snapshot of the verdict history (newest last)."""
        with self._lock:
            items = list(self._history)
        if agent_id is None:
            return items
        return [e for e in items if e.agent_id == agent_id]

    def all_scores(self) -> dict[tuple[int, str], int]:
        """Return current score for every (project, agent) pair that
        has ever fired a verdict, recomputed over live events only."""
        out: dict[tuple[int, str], int] = {}
        with self._lock:
            seen: set[tuple[int, str]] = set()
            for e in self._history:
                seen.add((e.project_id, e.agent_id))
            ticks = dict(self._ticks)
        for pid, aid in seen:
            current = ticks.get((pid, aid), 0)
            total = 0
            with self._lock:
                for e in self._history:
                    if e.project_id != pid or e.agent_id != aid:
                        continue
                    if e.is_live(current):
                        total += e.delta
            out[(pid, aid)] = total
        return out

    @staticmethod
    def _fingerprint(checker_name: str, trigger_key: str, reason: str) -> str:
        """Return a stable dedup key for a verdict.

        Prefers the explicit ``trigger_key`` the checker produced from
        its real evidence. Falls back to hashing the reason string when
        a checker doesn't set one (coarser, but still prevents the most
        obvious double-counting).
        """
        import hashlib

        key = trigger_key or hashlib.md5(
            (reason or "").encode("utf-8", errors="ignore")
        ).hexdigest()[:16]
        return f"{checker_name}|{key}"

    # ------------------------------------------------------------------
    # Entry points — one per HookEvent the scorer is wired into.
    # ------------------------------------------------------------------

    def on_step(self, ctx) -> None:
        """Run all enabled checkers after a completed step (POST_STEP)."""
        from infinidev.config.settings import settings

        if not getattr(settings, "BEHAVIOR_CHECKERS_ENABLED", False):
            return
        # Tick and expire *before* running checkers so new verdicts
        # fire against an already-fresh score.
        self._tick(ctx.project_id, ctx.agent_id)
        self._expire(ctx)
        checkers = enabled_checkers()
        if not checkers:
            return

        from infinidev.engine.behavior.eval_context import StepEvalContext

        eval_ctx = StepEvalContext.from_post_step(
            metadata=ctx.metadata or {},
            project_id=ctx.project_id,
            agent_id=ctx.agent_id,
        )
        mode = str(getattr(settings, "BEHAVIOR_JUDGE_MODE", "stochastic")).lower()
        if mode == "llm":
            self._run_llm(ctx, checkers, eval_ctx)
        elif mode == "hybrid":
            self._run_hybrid(ctx, checkers, eval_ctx)
        else:
            self._run_stochastic(ctx, checkers, eval_ctx)

    def on_model_message(self, ctx) -> None:  # HookContext
        """Legacy per_message hook — dispatches to the current judge mode."""
        from infinidev.config.settings import settings

        if not getattr(settings, "BEHAVIOR_CHECKERS_ENABLED", False):
            return
        # Tick + expire first (see on_step above).
        self._tick(ctx.project_id, ctx.agent_id)
        self._expire(ctx)
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
        history = [m for m in full_messages if m.get("role") != "system"][-history_window:]

        from infinidev.engine.behavior.eval_context import StepEvalContext

        eval_ctx = StepEvalContext.from_single_message(
            message=message,
            history=history,
            task=meta.get("task", ""),
            plan_snapshot=meta.get("plan_snapshot") or {},
            project_id=ctx.project_id,
            agent_id=ctx.agent_id,
        )
        mode = str(getattr(settings, "BEHAVIOR_JUDGE_MODE", "stochastic")).lower()
        if mode == "llm":
            self._run_llm_legacy(ctx, checkers, message, history, meta)
        else:
            # stochastic + hybrid both run stochastic first in message mode.
            self._run_stochastic(ctx, checkers, eval_ctx)
            if mode == "hybrid":
                self._run_llm_legacy(
                    ctx, checkers, message, history, meta,
                    only_low_confidence=True,
                )

    # ------------------------------------------------------------------
    # Judge engines
    # ------------------------------------------------------------------

    def _run_stochastic(self, ctx, checkers, eval_ctx) -> None:
        """Run every StochasticChecker; apply verdicts directly."""
        from infinidev.engine.behavior.checker_base import StochasticChecker

        for checker in checkers:
            if not isinstance(checker, StochasticChecker):
                continue
            try:
                verdict = checker.evaluate(eval_ctx)
            except Exception:
                logger.debug("Stochastic checker %s raised", checker.name, exc_info=True)
                continue
            if verdict is None or verdict.delta == 0:
                continue
            self._apply(
                ctx, checker.name, verdict.delta, verdict.reason,
                trigger_key=getattr(verdict, "trigger_key", "") or "",
                ttl_ticks=getattr(checker, "ttl_steps", -1),
            )

    def _run_llm(self, ctx, checkers, eval_ctx) -> None:
        """Run the batched LLM judge against a POST_STEP context."""
        from infinidev.engine.behavior.checker_base import PromptBehaviorChecker
        from infinidev.engine.behavior.batched_runner import run_batched

        prompt_checkers = [
            c for c in checkers if isinstance(c, PromptBehaviorChecker)
        ]
        if not prompt_checkers:
            return
        # Build a synthetic "latest message" + history from the step.
        message = {
            "role": "assistant",
            "raw_content": eval_ctx.latest_content,
            "reasoning_content": eval_ctx.reasoning_content,
            "tool_calls": [
                {"name": c.name, "arguments": c.raw_args}
                for c in eval_ctx.tool_calls
            ],
        }
        plan_snapshot = {
            "active_step_title": eval_ctx.active_step_title,
            "steps": eval_ctx.plan_steps,
        }
        verdicts = run_batched(
            prompt_checkers, message, eval_ctx.step_messages[-20:],
            task=eval_ctx.task, plan_snapshot=plan_snapshot,
        )
        for checker in prompt_checkers:
            v = verdicts.get(checker.name)
            if v is None or v.delta == 0:
                continue
            self._apply(
                ctx, checker.name, v.delta, v.reason,
                trigger_key=getattr(v, "trigger_key", "") or "",
                ttl_ticks=getattr(checker, "ttl_steps", -1),
            )

    def _run_llm_legacy(
        self, ctx, checkers, message, history, meta,
        only_low_confidence: bool = False,
    ) -> None:
        """Original per_message batched path (used by mode=llm and hybrid)."""
        from infinidev.engine.behavior.checker_base import PromptBehaviorChecker
        from infinidev.engine.behavior.batched_runner import run_batched

        prompt_checkers = [
            c for c in checkers if isinstance(c, PromptBehaviorChecker)
        ]
        if not prompt_checkers:
            return
        task = meta.get("task", "")
        plan_snapshot = meta.get("plan_snapshot") or {}
        verdicts = run_batched(
            prompt_checkers, message, history,
            task=task, plan_snapshot=plan_snapshot,
        )
        for checker in prompt_checkers:
            v = verdicts.get(checker.name)
            if v is None or v.delta == 0:
                continue
            self._apply(
                ctx, checker.name, v.delta, v.reason,
                trigger_key=getattr(v, "trigger_key", "") or "",
                ttl_ticks=getattr(checker, "ttl_steps", -1),
            )

    def _run_hybrid(self, ctx, checkers, eval_ctx) -> None:
        """Stochastic first; escalate nothing-verdicts to the LLM judge.

        In POST_STEP hybrid mode, any checker whose stochastic result was
        ``None`` (i.e. below the confidence threshold) is asked again via
        the batched LLM runner. Checkers that already fired are trusted.
        """
        from infinidev.engine.behavior.checker_base import (
            PromptBehaviorChecker,
            StochasticChecker,
        )
        from infinidev.engine.behavior.batched_runner import run_batched

        fired: set[str] = set()
        for checker in checkers:
            if not isinstance(checker, StochasticChecker):
                continue
            try:
                verdict = checker.evaluate(eval_ctx)
            except Exception:
                logger.debug("Stochastic checker %s raised", checker.name, exc_info=True)
                continue
            if verdict is None or verdict.delta == 0:
                continue
            self._apply(
                ctx, checker.name, verdict.delta, verdict.reason,
                trigger_key=getattr(verdict, "trigger_key", "") or "",
            )
            fired.add(checker.name)

        # Escalate un-fired PromptBehaviorChecker criteria.
        escalate = [
            c for c in checkers
            if isinstance(c, PromptBehaviorChecker) and c.name not in fired
        ]
        if not escalate:
            return
        message = {
            "role": "assistant",
            "raw_content": eval_ctx.latest_content,
            "reasoning_content": eval_ctx.reasoning_content,
            "tool_calls": [
                {"name": c.name, "arguments": c.raw_args}
                for c in eval_ctx.tool_calls
            ],
        }
        plan_snapshot = {
            "active_step_title": eval_ctx.active_step_title,
            "steps": eval_ctx.plan_steps,
        }
        verdicts = run_batched(
            escalate, message, eval_ctx.step_messages[-20:],
            task=eval_ctx.task, plan_snapshot=plan_snapshot,
        )
        for checker in escalate:
            v = verdicts.get(checker.name)
            if v is None or v.delta == 0:
                continue
            self._apply(
                ctx, checker.name, v.delta, v.reason,
                trigger_key=getattr(v, "trigger_key", "") or "",
                ttl_ticks=getattr(checker, "ttl_steps", -1),
            )

    def _apply(
        self, ctx, checker_name: str, delta: int, reason: str,
        trigger_key: str = "", ttl_ticks: int = -1,
    ) -> None:
        key = (ctx.project_id, ctx.agent_id)
        # Dedup: same evidence from the same checker is only scored once.
        fingerprint = self._fingerprint(checker_name, trigger_key, reason)
        with self._lock:
            fired = self._fired_fingerprints.setdefault(key, set())
            if fingerprint in fired:
                logger.debug(
                    "Behavior dedup: skipped %s (%s) — already scored",
                    checker_name, fingerprint,
                )
                return
            if len(fired) >= self.FINGERPRINT_CAP:
                fired.clear()
            fired.add(fingerprint)

            current_tick = self._ticks.get(key, 0)
            event = BehaviorEvent(
                timestamp=time.time(),
                project_id=ctx.project_id,
                agent_id=ctx.agent_id,
                checker=checker_name,
                delta=delta,
                reason=reason,
                score_after=0,          # filled in below after recompute
                tick_at_fire=current_tick,
                ttl_ticks=ttl_ticks,
            )
            self._history.append(event)
        # Recompute AFTER appending so new_score reflects the just-added
        # event (and respects any expirations performed this tick).
        new_score = self.score_for(ctx.project_id, ctx.agent_id)
        event.score_after = new_score

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
