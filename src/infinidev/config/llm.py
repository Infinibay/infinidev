"""Centralized LLM configuration for Infinidev CLI."""

from __future__ import annotations
import os
import logging
import re
from typing import Any
from infinidev.config.settings import settings

logger = logging.getLogger(__name__)


# ── Register models missing from LiteLLM's built-in database ─────────
# LiteLLM rejects requests for unknown models with wrong context-window
# limits.  We register them once at import time so every call_llm() and
# capability probe works correctly.

def _register_custom_models() -> None:
    """Add model entries that LiteLLM doesn't ship yet."""
    try:
        import litellm

        _M27_BASE = {
            "litellm_provider": "minimax",
            "mode": "chat",
            "supports_function_calling": True,
            "supports_tool_choice": True,
            "supports_prompt_caching": True,
            "supports_reasoning": True,
            "supports_system_messages": True,
            "max_input_tokens": 204_800,
            "max_output_tokens": 8192,
            "input_cost_per_token": 3e-07,
            "output_cost_per_token": 1.2e-06,
            "cache_read_input_token_cost": 3e-08,
            "cache_creation_input_token_cost": 3.75e-07,
        }

        custom = {
            "minimax/MiniMax-M2.7": {**_M27_BASE},
            "minimax/MiniMax-M2.7-highspeed": {**_M27_BASE},
        }

        for model_id, info in custom.items():
            if model_id not in litellm.model_cost:
                litellm.model_cost[model_id] = info
                logger.debug("Registered custom model: %s", model_id)
    except Exception as exc:
        logger.debug("Could not register custom models: %s", exc)


_register_custom_models()

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
    return _extract_provider(model) in {"deepseek", "anthropic", "gemini", "openai"}

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


_SMALL_MODEL_NAME_HINTS = (
    # Explicit local / open-weight families that fit on consumer GPUs.
    # Listed lowercase; matched as substrings of the model id.
    "glm-4.7-flash", "glm-4-flash", "glm-flash",
    "gemma2", "gemma3", "gemma4",
    "qwen2.5-coder", "qwen3", "qwen3.5",
    "mistral-small", "mistral-7b", "mixtral-8x7b",
    "nemotron-3-super", "nemotron-cascade",
    "lfm2",
    "gpt-oss:20b",
    # Generic "small" markers
    ":flash", "-flash", "-mini", "-tiny", "-small", "haiku",
)


def _is_small_model(model: str | None = None) -> bool:
    """Return True if the model is in the "small" tier (<~40B effective).

    Detection order:
      1. Explicit size suffix in the name (e.g. "qwen3:9b" → 9 < 40 → True).
      2. Substring match against ``_SMALL_MODEL_NAME_HINTS`` for known
         local/open-weight families that don't carry a size in their tag
         (e.g. ``glm-4.7-flash:latest`` — previously classified as large).
      3. Default False (treat unknown as large; safer for hosted big models).
    """
    name = (model or settings.LLM_MODEL or "").lower()
    size = _get_model_size_b(name)
    if 0 < size < 40:
        return True
    for hint in _SMALL_MODEL_NAME_HINTS:
        if hint in name:
            return True
    return False


def get_litellm_params_for_review_extractor() -> dict[str, Any]:
    """Build litellm params for the review extractor (Pass A).

    Each ``REVIEW_EXTRACTOR_LLM_*`` setting is optional and falls back to
    the matching ``LLM_*`` main setting when empty. Use this to point the
    factual-extraction pass at a small/fast model while the judge keeps
    running on the main one.
    """
    model = (settings.REVIEW_EXTRACTOR_LLM_MODEL or "").strip() or settings.LLM_MODEL
    if not model:
        raise RuntimeError("No review-extractor model and no main LLM_MODEL configured.")

    if model.startswith("ollama/"):
        model = "ollama_chat/" + model[len("ollama/"):]

    provider_id = (settings.REVIEW_EXTRACTOR_LLM_PROVIDER or "").strip() or settings.LLM_PROVIDER
    api_key = (settings.REVIEW_EXTRACTOR_LLM_API_KEY or "").strip() or settings.LLM_API_KEY
    base_url = (settings.REVIEW_EXTRACTOR_LLM_BASE_URL or "").strip() or settings.LLM_BASE_URL

    params: dict[str, Any] = {"model": model}
    if api_key:
        params["api_key"] = api_key

    try:
        from infinidev.config.providers import get_provider
        provider = get_provider(provider_id)
        is_native = bool(getattr(provider, "is_native", False))
    except Exception:
        is_native = _extract_provider(model) in {
            "deepseek", "anthropic", "gemini", "openai",
        }
    if base_url and not is_native:
        params["api_base"] = base_url

    if settings.LLM_TIMEOUT:
        params["timeout"] = float(settings.LLM_TIMEOUT)

    if provider_id == "ollama" and settings.OLLAMA_NUM_CTX > 0:
        params["num_ctx"] = settings.OLLAMA_NUM_CTX

    from importlib.metadata import version as _pkg_version
    try:
        _version = _pkg_version("infinidev")
    except Exception:
        _version = "0.1.0"
    params["extra_headers"] = {
        "User-Agent": f"infinidev/{_version}",
        "X-Client-Name": "infinidev-review-extractor",
        "X-Client-Version": _version,
    }

    return params


def get_litellm_params_for_behavior() -> dict[str, Any]:
    """Build litellm params for the behavior-checker judge.

    Each ``BEHAVIOR_LLM_*`` setting is optional and falls back to the
    matching ``LLM_*`` main setting when empty. This lets users point the
    judge at a small/fast model (e.g. ``ollama/qwen2.5:3b``) without
    affecting the main agent. Returns the same shape as
    :func:`get_litellm_params`.
    """
    model = (settings.BEHAVIOR_LLM_MODEL or "").strip() or settings.LLM_MODEL
    if not model:
        raise RuntimeError("No behavior model and no main LLM_MODEL configured.")

    if model.startswith("ollama/"):
        model = "ollama_chat/" + model[len("ollama/"):]

    provider_id = (settings.BEHAVIOR_LLM_PROVIDER or "").strip() or settings.LLM_PROVIDER
    api_key = (settings.BEHAVIOR_LLM_API_KEY or "").strip() or settings.LLM_API_KEY
    base_url = (settings.BEHAVIOR_LLM_BASE_URL or "").strip() or settings.LLM_BASE_URL

    params: dict[str, Any] = {"model": model}
    if api_key:
        params["api_key"] = api_key

    # Mirror the native-provider rule: only pass api_base for non-native
    # providers, otherwise litellm routes to the wrong endpoint.
    try:
        from infinidev.config.providers import get_provider
        provider = get_provider(provider_id)
        is_native = bool(getattr(provider, "is_native", False))
    except Exception:
        is_native = _extract_provider(model) in {
            "deepseek", "anthropic", "gemini", "openai",
        }
    if base_url and not is_native:
        params["api_base"] = base_url

    if settings.LLM_TIMEOUT:
        params["timeout"] = float(settings.LLM_TIMEOUT)

    # num_ctx only matters for Ollama-style local providers
    if provider_id == "ollama" and settings.OLLAMA_NUM_CTX > 0:
        params["num_ctx"] = settings.OLLAMA_NUM_CTX

    from importlib.metadata import version as _pkg_version
    try:
        _version = _pkg_version("infinidev")
    except Exception:
        _version = "0.1.0"
    params["extra_headers"] = {
        "User-Agent": f"infinidev/{_version}",
        "X-Client-Name": "infinidev-behavior",
        "X-Client-Version": _version,
    }

    return params


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

    # Retry transient provider errors (e.g. OpenRouter mid-stream
    # "Network connection lost"). LiteLLM retries APIError / Timeout /
    # RateLimitError / ServiceUnavailableError automatically.
    if settings.LLM_NUM_RETRIES > 0:
        params["num_retries"] = settings.LLM_NUM_RETRIES
        params["retry_strategy"] = "exponential_backoff_retry"

    # Pass num_ctx for Ollama to control KV cache allocation.
    # Models like gemma4 default to 262k context which hangs on consumer GPUs.
    if settings.LLM_PROVIDER == "ollama" and settings.OLLAMA_NUM_CTX > 0:
        params["num_ctx"] = settings.OLLAMA_NUM_CTX

    # Identify Infinidev to providers via HTTP headers.
    # Providers track client identity for analytics, rate-limit fairness,
    # and partnership eligibility.
    from importlib.metadata import version as _pkg_version
    try:
        _version = _pkg_version("infinidev")
    except Exception:
        _version = "0.1.0"
    params["extra_headers"] = {
        "User-Agent": f"infinidev/{_version}",
        "X-Client-Name": "infinidev",
        "X-Client-Version": _version,
        "anthropic-client-name": "infinidev",
        "anthropic-client-version": _version,
    }

    return params
