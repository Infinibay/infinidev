"""Context window calculator for Infinidev TUI.

Calculates remaining tokens for both chat and task context windows
based on the selected model's context limits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelInfo(BaseModel):
    """Information about a model from Ollama."""
    name: str = ""
    size: int = 0
    context_length: int = 2048  # Default fallback
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def max_tokens(self) -> int:
        """Get max context length, with fallbacks."""
        # Try known keys for context length
        for key in ["context_length", "context_len", "max_ctx", "ctx_len"]:
            if key in self.details:
                val = self.details[key]
                if isinstance(val, int):
                    return val
        return self.context_length


class ContextWindowCalculator:
    """Calculates and tracks context window usage.

    Manages two separate context windows:
    - Chat context: User messages and responses
    - Task context: Task descriptions and execution summaries

    Both windows share the same max_context limit.
    """

    def __init__(self, model_name: str = "", max_context: int = 4096):
        self.model_name = model_name
        self.max_context: int = max_context
        self._chat_tokens: int = 0
        self._task_tokens: int = 0

    async def update_model_context(self) -> None:
        """Fetch model info from Ollama and update max context.

        Uses /api/show to get the real context_length from model_info,
        which reflects the actual model architecture limit (e.g. 262144
        for qwen3.5) rather than the default num_ctx runtime parameter.
        """
        from infinidev.config.llm import get_litellm_params
        from infinidev.config.settings import settings

        llm_params = get_litellm_params()
        model = llm_params.get("model", settings.LLM_MODEL)
        base_url = llm_params.get("base_url", settings.LLM_BASE_URL)

        # Strip ollama prefixes to get the bare model name for API calls
        bare_model = model
        for prefix in ("ollama_chat/", "ollama/"):
            if bare_model.startswith(prefix):
                bare_model = bare_model[len(prefix):]
                break

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Use /api/show to get full model metadata including real context length
                resp = await client.post(
                    f"{base_url}/api/show",
                    json={"name": bare_model},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    self.model_name = bare_model

                    # Extract context_length from model_info
                    # Keys follow the pattern: <family>.context_length
                    model_info = data.get("model_info", {})
                    for key, val in model_info.items():
                        if key.endswith(".context_length") and isinstance(val, int):
                            self.max_context = val
                            break

                    logger.info(f"Model {bare_model} context length: {self.max_context}")
                    return

                # Fallback: try /api/tags if /api/show fails
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

    def calculate_usage(self, chat_tokens: int, task_tokens: int) -> None:
        """Update usage for both context windows."""
        self._chat_tokens = chat_tokens
        self._task_tokens = task_tokens

    def get_context_status(self) -> dict[str, Any]:
        """Get current context window status."""
        return {
            "model": self.model_name or "unknown",
            "max_context": self.max_context,
            "chat": {
                "name": "chat",
                "current_tokens": self._chat_tokens,
                "max_tokens": self.max_context,
                "remaining_tokens": max(0, self.max_context - self._chat_tokens),
                "usage_percentage": min(1.0, self._chat_tokens / self.max_context) if self.max_context > 0 else 0.0,
            },
            "tasks": {
                "name": "tasks",
                "current_tokens": self._task_tokens,
                "max_tokens": self.max_context,
                "remaining_tokens": max(0, self.max_context - self._task_tokens),
                "usage_percentage": min(1.0, self._task_tokens / self.max_context) if self.max_context > 0 else 0.0,
            },
            "remaining": max(0, self.max_context - (self._chat_tokens + self._task_tokens)),
            "total_used": self._chat_tokens + self._task_tokens,
            "combined_remaining": max(0, self.max_context - (self._chat_tokens + self._task_tokens)),
        }

    @property
    def chat_remaining(self) -> int:
        """Get remaining chat tokens."""
        return max(0, self.max_context - self._chat_tokens)

    @property
    def task_remaining(self) -> int:
        """Get remaining task tokens."""
        return max(0, self.max_context - self._task_tokens)

    @property
    def chat_usage_percentage(self) -> float:
        """Get chat window usage percentage (0.0 to 1.0)."""
        if self.max_context == 0:
            return 0.0
        return min(1.0, self._chat_tokens / self.max_context)

    @property
    def task_usage_percentage(self) -> float:
        """Get task window usage percentage (0.0 to 1.0)."""
        if self.max_context == 0:
            return 0.0
        return min(1.0, self._task_tokens / self.max_context)

    @property
    def total_remaining(self) -> int:
        """Get total remaining tokens across both windows."""
        return self.chat_remaining + self.task_remaining

    @property
    def chat_window(self) -> dict[str, Any]:
        """Get chat window data for tests."""
        return {
            "current_tokens": self._chat_tokens,
            "max_tokens": self.max_context,
            "remaining_tokens": self.chat_remaining,
            "usage_percentage": self.chat_usage_percentage,
        }

    @property
    def task_window(self) -> dict[str, Any]:
        """Get task window data for tests."""
        return {
            "current_tokens": self._task_tokens,
            "max_tokens": self.max_context,
            "remaining_tokens": self.task_remaining,
            "usage_percentage": self.task_usage_percentage,
        }


# Global calculator instance with model name from settings
def _get_initial_model_name() -> str:
    """Get the current model name from settings."""
    from infinidev.config.llm import get_litellm_params
    from infinidev.config.settings import settings

    llm_params = get_litellm_params()
    return llm_params.get("model", settings.LLM_MODEL)


calculator = ContextWindowCalculator(model_name=_get_initial_model_name(), max_context=4096)


async def get_context_status() -> dict[str, Any]:
    """Get current context window status."""
    await calculator.update_model_context()
    return calculator.get_context_status()
