"""Context window calculator for Infinidev TUI.

Tracks the last prompt token usage against the model's context window limit.
Each LLM call rebuilds the full prompt from scratch, so only the most recent
prompt_tokens value matters for context window usage.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str, model: str) -> int:
    """Estimate tokens for *text* using the model's tokenizer when known.

    litellm.token_counter picks the right tokenizer for known models and
    falls back to a sensible default otherwise — far better than the old
    chars/4 heuristic.  If anything goes wrong (unknown model, import error),
    fall back to ~4 chars per token so the bar still moves.
    """
    if not text:
        return 0
    try:
        import litellm

        return int(litellm.token_counter(model=model or "gpt-3.5-turbo", text=text))
    except Exception:
        return len(text) // 4


class ContextWindowCalculator:
    """Calculates and tracks context window usage.

    Tracks two values:
    - last_prompt_tokens: Tokens used in the most recent LLM call (= real context usage)
    - total_tokens: Cumulative tokens across all LLM calls in the current task
    """

    def __init__(self, model_name: str = "", max_context: int | None = None):
        self.model_name = model_name
        # max_context is None when we don't know the model's context window.
        # Display code should render this as "?" instead of a misleading number.
        self.max_context: int | None = max_context
        self._last_prompt_tokens: int = 0
        self._task_prompt_tokens: int = 0
        self._warned_over_budget: bool = False

    async def update_model_context(self) -> None:
        """Resolve the model's *effective* context window.

        Delegates to :func:`get_model_context_window`, the single source of
        truth shared with the LoopEngine.  For Ollama that is the real ceiling
        (``num_ctx``, capped by the trained length) — NOT the trained length,
        which the server ignores and truncates past.  Sets ``self.max_context``
        to ``None`` when unknown; the TUI renders ``None`` as ``?``.
        """
        import asyncio

        from infinidev.config.llm import get_litellm_params
        from infinidev.config.settings import settings
        from infinidev.engine.loop.model_context import get_model_context_window

        llm_params = get_litellm_params()
        model = llm_params.get("model", settings.LLM_MODEL)
        provider_id = getattr(settings, "LLM_PROVIDER", "ollama")

        # Strip provider prefixes to get bare model name for display.
        self.model_name = model.split("/", 1)[1] if "/" in model else model

        # The resolver may do a blocking HTTP call to Ollama on first lookup
        # (memoized thereafter); keep it off the event loop thread.
        self.max_context = await asyncio.to_thread(
            get_model_context_window,
            llm_params,
            provider_id,
        )

        if self.max_context:
            logger.info(
                f"Model {self.model_name} effective context window: {self.max_context}"
            )
        else:
            logger.info(
                f"Model {self.model_name}: context window unknown, will display as '?'"
            )

    def update_chat(
        self, user_input: str, session_summaries: list[str] | None = None
    ) -> None:
        """Estimate chat context tokens from user input + session history.

        Uses the model's tokenizer via litellm when available, falling back to
        ~4 chars per token.
        """
        text = user_input
        if session_summaries:
            text += "\n".join(session_summaries)
        self._last_prompt_tokens = _estimate_tokens(text, self.model_name)

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

    llm_params = get_litellm_params()
    return llm_params.get("model", settings.LLM_MODEL)


calculator = ContextWindowCalculator(
    model_name=_get_initial_model_name(), max_context=None
)


async def get_context_status() -> dict[str, Any]:
    await calculator.update_model_context()
    return calculator.get_context_status()
