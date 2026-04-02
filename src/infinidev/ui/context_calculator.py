"""Context window calculator for Infinidev TUI.

Tracks the last prompt token usage against the model's context window limit.
Each LLM call rebuilds the full prompt from scratch, so only the most recent
prompt_tokens value matters for context window usage.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class ContextWindowCalculator:
    """Calculates and tracks context window usage.

    Tracks two values:
    - last_prompt_tokens: Tokens used in the most recent LLM call (= real context usage)
    - total_tokens: Cumulative tokens across all LLM calls in the current task
    """

    def __init__(self, model_name: str = "", max_context: int = 4096):
        self.model_name = model_name
        self.max_context: int = max_context
        self._last_prompt_tokens: int = 0
        self._task_prompt_tokens: int = 0

    # Known context windows for cloud models (tokens)
    _KNOWN_CONTEXT: dict[str, int] = {
        # OpenAI
        "gpt-5.4": 1_000_000, "gpt-5.4-mini": 400_000, "gpt-5.4-nano": 400_000,
        "o3": 200_000, "o3-pro": 200_000, "o3-mini": 200_000, "o4-mini": 200_000,
        # Anthropic
        "claude-opus-4-6": 1_000_000, "claude-sonnet-4-6": 1_000_000,
        "claude-haiku-4-5-20251001": 200_000,
        "claude-sonnet-4-5-20250929": 200_000, "claude-opus-4-5-20251101": 200_000,
        "claude-sonnet-4-0": 200_000, "claude-opus-4-0": 200_000,
        # Gemini
        "gemini-3.1-pro-preview": 1_048_576, "gemini-3-flash-preview": 1_048_576,
        "gemini-3.1-flash-lite-preview": 1_048_576,
        "gemini-2.5-pro": 1_048_576, "gemini-2.5-flash": 1_048_576,
        "gemini-2.5-flash-lite": 1_048_576,
        # Z.AI
        "glm-5": 200_000, "glm-5-turbo": 200_000,
        "glm-4.7": 200_000, "glm-4.6": 200_000,
        "glm-4.5": 128_000, "glm-4.5-flash": 128_000, "glm-4.5-air": 128_000,
        # Kimi
        "kimi-k2.5": 256_000,
        "kimi-k2-thinking": 256_000, "kimi-k2-thinking-turbo": 256_000,
        "kimi-k2-0905-preview": 256_000, "kimi-k2-turbo-preview": 256_000,
        # Minimax
        "MiniMax-M2.7": 204_800, "MiniMax-M2.7-highspeed": 204_800,
        "MiniMax-M2.5": 204_800, "MiniMax-M2.1": 204_800,
    }

    async def update_model_context(self) -> None:
        """Fetch model context window. Supports Ollama API and known cloud models."""
        from infinidev.config.llm import get_litellm_params
        from infinidev.config.settings import settings

        llm_params = get_litellm_params()
        model = llm_params.get("model", settings.LLM_MODEL)
        base_url = llm_params.get("base_url", settings.LLM_BASE_URL)

        # Strip provider prefixes to get bare model name
        bare_model = model
        if "/" in bare_model:
            bare_model = bare_model.split("/", 1)[1]

        # Check known cloud models first
        if bare_model in self._KNOWN_CONTEXT:
            self.model_name = bare_model
            self.max_context = self._KNOWN_CONTEXT[bare_model]
            logger.info(f"Model {bare_model} context length (known): {self.max_context}")
            return

        # Try litellm's model cost map for context window
        try:
            from litellm import model_cost
            if model in model_cost:
                ctx = model_cost[model].get("max_input_tokens", 0)
                if ctx:
                    self.model_name = bare_model
                    self.max_context = ctx
                    logger.info(f"Model {bare_model} context length (litellm): {self.max_context}")
                    return
        except Exception:
            pass

        # Fallback: try Ollama API (only works for Ollama provider)
        provider_id = getattr(settings, "LLM_PROVIDER", "ollama")
        if provider_id == "ollama" and base_url:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post(
                        f"{base_url}/api/show",
                        json={"name": bare_model},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        self.model_name = bare_model

                        model_info = data.get("model_info", {})
                        for key, val in model_info.items():
                            if key.endswith(".context_length") and isinstance(val, int):
                                self.max_context = val
                                break

                        logger.info(f"Model {bare_model} context length: {self.max_context}")
                        return

                    # Fallback: try /api/tags
                    resp = await client.get(f"{base_url}/api/tags")
                    if resp.status_code == 200:
                        data = resp.json()
                        for m in data.get("models", []):
                            m_name = m.get("name", "")
                            if bare_model in m_name or m_name in bare_model:
                                self.model_name = m_name
                                ctx = m.get("details", {}).get("context_length", 0)
                                if ctx:
                                    self.max_context = ctx
                                break
            except Exception as e:
                logger.warning(f"Could not fetch model info: {e}")

        # Default for unknown cloud models
        if provider_id != "ollama" and self.max_context <= 4096:
            self.max_context = 128_000
            self.model_name = bare_model
            logger.info(f"Model {bare_model}: using default cloud context {self.max_context}")

    def update_chat(self, user_input: str, session_summaries: list[str] | None = None) -> None:
        """Estimate chat context tokens from user input + session history.

        Uses ~4 chars per token as a rough estimate (standard heuristic).
        """
        text = user_input
        if session_summaries:
            text += "\n".join(session_summaries)
        self._last_prompt_tokens = max(1, len(text) // 4)

    def update_task(self, task_prompt_tokens: int = 0) -> None:
        """Update task context with prompt_tokens from the last LLM call."""
        if task_prompt_tokens:
            self._task_prompt_tokens = task_prompt_tokens

    def get_context_status(self) -> dict[str, Any]:
        """Get current context window status for the UI."""
        max_ctx = self.max_context
        prompt = self._last_prompt_tokens
        task = self._task_prompt_tokens

        prompt_remaining = max(0, max_ctx - prompt)
        prompt_pct = min(1.0, prompt / max_ctx) if max_ctx > 0 else 0.0

        task_remaining = max(0, max_ctx - task)
        task_pct = min(1.0, task / max_ctx) if max_ctx > 0 else 0.0

        return {
            "model": self.model_name or "unknown",
            "max_context": max_ctx,
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
        return max(0, self.max_context - self._last_prompt_tokens)

    @property
    def task_remaining(self) -> int:
        return max(0, self.max_context - self._task_prompt_tokens)

    @property
    def chat_usage_percentage(self) -> float:
        if self.max_context == 0:
            return 0.0
        return min(1.0, self._last_prompt_tokens / self.max_context)

    @property
    def task_usage_percentage(self) -> float:
        if self.max_context == 0:
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
    from infinidev.config.llm import get_litellm_params
    from infinidev.config.settings import settings
    llm_params = get_litellm_params()
    return llm_params.get("model", settings.LLM_MODEL)


calculator = ContextWindowCalculator(model_name=_get_initial_model_name(), max_context=4096)


async def get_context_status() -> dict[str, Any]:
    await calculator.update_model_context()
    return calculator.get_context_status()
