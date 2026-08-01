"""Context window calculator for Infinidev TUI.

Tracks the last prompt token usage against the model's context window limit.
Each LLM call rebuilds the full prompt from scratch, so only the most recent
prompt_tokens value matters for context window usage.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ContextWindowCalculator:
    """Calculates and tracks context window usage.

    Tracks two values:
    - last_prompt_tokens: Tokens used in the most recent LLM call (= real context usage)
    - total_tokens: Cumulative tokens across all LLM calls in the current task
    """

    def __init__(
        self,
        model_name: str = "",
        max_context: int | None = None,
        model_max_context: int | None = None,
    ):
        self.model_name = model_name
        # max_context is None when we don't know the model's context window.
        # Display code should render this as "?" instead of a misleading number.
        self.max_context: int | None = max_context
        self.model_max_context: int | None = model_max_context
        self._last_prompt_tokens: int = 0
        self._task_prompt_tokens: int = 0
        self._warned_over_budget: bool = False

    def resolve_model_context(self) -> None:
        """Resolve the model's *effective* context window, blocking.

        Delegates to :func:`get_model_context_window`, the single source of
        truth shared with the LoopEngine.  For Ollama that is the real ceiling
        (``num_ctx``, capped by the trained length) — NOT the trained length,
        which the server ignores and truncates past.  Sets ``self.max_context``
        to ``None`` when unknown; the TUI renders ``None`` as ``?``.

        Synchronous on purpose. The TUI resolves this during startup, which
        already runs inside prompt_toolkit's event loop, and the async version
        below used to be reached through ``asyncio.run`` — which raises inside
        a running loop. The raise was swallowed, so every model displayed ``?``
        no matter what the catalog said.
        """
        from infinidev.config.llm import get_litellm_params
        from infinidev.config.settings import settings
        from infinidev.engine.loop.model_context import (
            _bare_model,
            get_model_context_window,
            get_model_max_context_window,
        )

        # Building params can fail on a credential problem — the ChatGPT
        # subscription raises when its OAuth login is missing or expired.
        # That is a real error, but it belongs to the first LLM call, where
        # the message reaches the user; here it would only blank the status
        # line, so fall back to the configured name and an unknown window.
        try:
            llm_params = get_litellm_params()
        except Exception as exc:
            logger.info("Context window unresolved (%s); showing '?'", exc)
            self.model_name = _bare_model(settings.LLM_MODEL or "")
            self.max_context = None
            self.model_max_context = None
            return

        model = llm_params.get("model", settings.LLM_MODEL)
        provider_id = getattr(settings, "LLM_PROVIDER", "ollama")

        # Strip provider prefixes to get bare model name for display. Shared
        # with the context lookup so the status line cannot name one model
        # while sizing another (`openai/responses/gpt-5.5` → `gpt-5.5`).
        self.model_name = _bare_model(model)
        self.max_context = get_model_context_window(llm_params, provider_id)
        self.model_max_context = get_model_max_context_window(llm_params, provider_id)

        if self.max_context:
            logger.info(
                f"Model {self.model_name} effective context window: {self.max_context}"
            )
        else:
            logger.info(
                f"Model {self.model_name}: context window unknown, will display as '?'"
            )

    async def update_model_context(self) -> None:
        """Async wrapper: the lookup may block on an HTTP call to Ollama."""
        import asyncio

        await asyncio.to_thread(self.resolve_model_context)

    def update_chat(self, prompt_tokens: int = 0) -> None:
        """Record the chat agent's real prompt size from its last LLM call.

        This used to estimate from the user's message plus the session
        summaries, which measured the one part of the prompt that is
        negligible. The system prompt alone is ~4 400 tokens before a single
        tool schema or tool result — a turn that reported 108 was carrying
        tens of thousands. Only ``usage.prompt_tokens`` off the response knows
        the real number, so that is the only thing this accepts.
        """
        if prompt_tokens > 0:
            self._last_prompt_tokens = prompt_tokens

    def update_task(self, task_prompt_tokens: int = 0) -> None:
        """Update task context with the exact prompt_tokens from the last LLM call."""
        if task_prompt_tokens:
            self._task_prompt_tokens = task_prompt_tokens
            # The prompt overflowed the real window: the backend is silently
            # truncating context this turn.  Warn once so it isn't invisible.
            if (
                self.max_context
                and task_prompt_tokens > self.max_context
                and not self._warned_over_budget
            ):
                self._warned_over_budget = True
                logger.warning(
                    "Prompt (%d tokens) exceeds the model's effective context "
                    "window (%d) — the backend is truncating context. Raise "
                    "INFINIDEV_OLLAMA_NUM_CTX or shorten the task.",
                    task_prompt_tokens,
                    self.max_context,
                )

    def start_task(self) -> None:
        """Mark a new task boundary.

        The developer's prompt is rebuilt from scratch each task, so its
        context usage genuinely drops to zero here. Without this the bar would
        keep showing the previous task's peak until the next call landed.
        """
        self._task_prompt_tokens = 0
        self._warned_over_budget = False

    def get_context_status(self) -> dict[str, Any]:
        """Get current context window status for the UI.

        When ``max_context`` is ``None`` (unknown window), ``remaining_tokens``
        is also ``None`` and ``usage_percentage`` is ``0.0``.  Display code
        should check for ``None`` and render ``?`` instead of computing a
        percentage against a made-up max.
        """
        max_ctx = self.max_context
        prompt = self._last_prompt_tokens
        task = self._task_prompt_tokens

        if max_ctx is not None and max_ctx > 0:
            prompt_remaining: int | None = max(0, max_ctx - prompt)
            prompt_pct = min(1.0, prompt / max_ctx)
            task_remaining: int | None = max(0, max_ctx - task)
            task_pct = min(1.0, task / max_ctx)
        else:
            prompt_remaining = None
            prompt_pct = 0.0
            task_remaining = None
            task_pct = 0.0

        return {
            "model": self.model_name or "unknown",
            "max_context": max_ctx,  # may be None
            "model_max_context": self.model_max_context,
            "chat": {
                "name": "prompt",
                "current_tokens": prompt,
                "max_tokens": max_ctx,
                "remaining_tokens": prompt_remaining,
                "usage_percentage": prompt_pct,
            },
            "tasks": {
                "name": "task",
                "current_tokens": task,
                "max_tokens": max_ctx,
                "remaining_tokens": task_remaining,
                "usage_percentage": task_pct,
            },
        }

    # --- Properties for tests / external access ---

    @property
    def chat_remaining(self) -> int:
        if not self.max_context:
            return 0
        return max(0, self.max_context - self._last_prompt_tokens)

    @property
    def task_remaining(self) -> int:
        if not self.max_context:
            return 0
        return max(0, self.max_context - self._task_prompt_tokens)

    @property
    def chat_usage_percentage(self) -> float:
        if not self.max_context:
            return 0.0
        return min(1.0, self._last_prompt_tokens / self.max_context)

    @property
    def task_usage_percentage(self) -> float:
        if not self.max_context:
            return 0.0
        return min(1.0, self._task_prompt_tokens / self.max_context)

    @property
    def total_remaining(self) -> int:
        return self.chat_remaining

    @property
    def chat_window(self) -> dict[str, Any]:
        return {
            "current_tokens": self._last_prompt_tokens,
            "max_tokens": self.max_context,
            "remaining_tokens": self.chat_remaining,
            "usage_percentage": self.chat_usage_percentage,
        }

    @property
    def task_window(self) -> dict[str, Any]:
        return {
            "current_tokens": self._task_prompt_tokens,
            "max_tokens": self.max_context,
            "remaining_tokens": self.task_remaining,
            "usage_percentage": self.task_usage_percentage,
        }


# Global calculator instance
def _get_initial_model_name() -> str:
    from infinidev.config.settings import settings

    if not settings.LLM_MODEL:
        return "ollama_chat/qwen2.5-coder:7b"
    from infinidev.config.llm import get_litellm_params

    # Runs at import time to seed the module-level calculator, so anything
    # that can raise here takes the whole TUI down before it draws. A
    # misconfigured provider must degrade to a name in the status line, not
    # to an ImportError.
    try:
        return get_litellm_params().get("model", settings.LLM_MODEL)
    except Exception:
        return settings.LLM_MODEL


calculator = ContextWindowCalculator(
    model_name=_get_initial_model_name(), max_context=None
)


async def get_context_status() -> dict[str, Any]:
    await calculator.update_model_context()
    return calculator.get_context_status()
