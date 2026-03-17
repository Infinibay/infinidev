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
        """Fetch model info from Ollama and update max context."""
        from infinidev.config.llm import get_litellm_params
        from infinidev.config.settings import settings

        llm_params = get_litellm_params()
        model = llm_params.get("model", settings.LLM_MODEL)

        # Try to get model info from Ollama API
        base_url = llm_params.get("base_url", settings.LLM_BASE_URL)

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{base_url}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    models = data.get("models", [])

                    # Find the current model - handle both prefixed and unprefixed names
                    model_info = None
                    for m in models:
                        m_name = m.get("name", "")
                        # Try multiple matching strategies
                        if model in m_name or m_name in model or model.replace("ollama_chat/", "").replace("ollama_", "") in m_name or m_name.replace("ollama_chat/", "").replace("ollama_", "") in model.replace("ollama_chat/", "").replace("ollama_", ""):
                            model_info = ModelInfo(
                                name=m_name,
                                size=m.get("size", 0),
                                context_length=m.get("details", {}).get("context_length", 2048),
                                details=m.get("details", {}),
                            )
                            break
                    
                    # If no exact match found, try first available model as fallback
                    if not model_info and models:
                        model_info = ModelInfo(
                            name=models[0].get("name", ""),
                            size=models[0].get("size", 0),
                            context_length=models[0].get("details", {}).get("context_length", 2048),
                            details=models[0].get("details", {}),
                        )
                        logger.info(f"Using first model as fallback: {model_info.name}")

                    if model_info:
                        self.model_name = model_info.name
                        self.max_context = model_info.max_tokens
                        logger.info(f"Model {model} context length: {self.max_context}")
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
