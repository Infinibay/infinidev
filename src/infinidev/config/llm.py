"""Centralized LLM configuration for Infinidev CLI."""

from __future__ import annotations
import os
import logging
import re
from typing import Any
from infinidev.config.settings import settings

logger = logging.getLogger(__name__)

def _extract_provider(model: str) -> str:
    """Extract provider prefix from a LiteLLM model string."""
    if "/" in model:
        return model.split("/", 1)[0].lower()
    if model.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return "openai"
    return ""

def _is_native_provider(model: str) -> bool:
    """Return True if LiteLLM handles this provider's endpoint natively."""
    from infinidev.config.providers import get_provider
    provider_id = settings.LLM_PROVIDER
    provider = get_provider(provider_id)
    if provider.is_native:
        return True
    # Fallback: check model prefix for backward compatibility
    return _extract_provider(model) in {"deepseek", "anthropic", "gemini", "openai", "zai"}

def _get_model_size_b(model: str | None = None) -> int:
    """Extract model size in billions from model name.

    Parses patterns like 'qwen2.5-coder:7b', 'llama3.1:8b', 'mistral-7b-instruct'.
    Returns 0 if size cannot be detected.
    """
    model = model or settings.LLM_MODEL or ""
    match = re.search(r"(\d+)\s*[bB]\b", model.lower())
    if match:
        return int(match.group(1))
    return 0


def _is_small_model(model: str | None = None) -> bool:
    """Return True if the model has fewer than 25B parameters.

    Size is detected from the model name string.  Returns False when
    the size cannot be determined (safe default — treat as large).
    """
    size = _get_model_size_b(model)
    return 0 < size < 25


def get_litellm_params() -> dict[str, Any]:
    """Return kwargs suitable for ``litellm.completion(**params, messages=...)``."""
    model = settings.LLM_MODEL
    if not model:
        raise RuntimeError("INFINIDEV_LLM_MODEL is not set.")

    # Auto-correct ollama/ → ollama_chat/ so the /api/chat endpoint is used
    # (ollama/ hits /api/generate which has no function-calling support).
    if model.startswith("ollama/"):
        model = "ollama_chat/" + model[len("ollama/"):]
        logger.info("Auto-corrected model prefix: ollama/ → ollama_chat/ (required for tool calling)")

    params: dict[str, Any] = {"model": model}

    if settings.LLM_API_KEY:
        params["api_key"] = settings.LLM_API_KEY

    if settings.LLM_BASE_URL and not _is_native_provider(model):
        params["api_base"] = settings.LLM_BASE_URL

    if settings.LLM_TIMEOUT:
        params["timeout"] = float(settings.LLM_TIMEOUT)

    # Pass num_ctx for Ollama to control KV cache allocation.
    # Models like gemma4 default to 262k context which hangs on consumer GPUs.
    if settings.LLM_PROVIDER == "ollama" and settings.OLLAMA_NUM_CTX > 0:
        params["num_ctx"] = settings.OLLAMA_NUM_CTX

    return params
