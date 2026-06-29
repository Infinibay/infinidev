"""Resolve a model's *effective* context window.

Single source of truth for both the LoopEngine prompt budget and the TUI
context bar.  They used to compute this independently and could disagree —
the TUI even shipped a hand-maintained dict of cloud context sizes that
drifted out of date.

The subtlety that makes a naive implementation wrong: for Ollama the trained
context length reported by ``/api/show`` (e.g. 32k, 256k) is NOT what the
server actually serves.  Ollama allocates exactly ``num_ctx`` of KV cache and
silently truncates anything past it.  So the effective window is
``min(num_ctx, trained)`` — and because Infinidev always sends ``num_ctx``
(default 16384), ``num_ctx`` is the ceiling that actually bites.  Reporting
the trained length makes the usage bar read ~50% while the model is already
truncating at 100%.
"""

from __future__ import annotations

from typing import Any

from infinidev.engine._best_effort import best_effort

# (model, base_url) -> trained context length from Ollama /api/show, or None.
# Memoized: the value is intrinsic to the model and never changes within a
# process, but the fetch is a 5 s-timeout HTTP POST we must not repeat on
# every context rebuild.  ``None`` (unknown) is cached too, so a transient
# Ollama failure doesn't cause repeated timeouts on subsequent calls.
_OLLAMA_CTX_CACHE: dict[tuple[str, str], int | None] = {}

# Cloud context windows used only as a *fallback* when the installed litellm
# doesn't know the model yet (frontier models ship faster than litellm's cost
# map updates).  litellm is consulted first and wins when it has the model, so
# these self-heal on the next litellm upgrade instead of silently going stale.
_CLOUD_CTX_OVERRIDES: dict[str, int] = {
    # OpenAI
    "gpt-5.4": 1_000_000, "gpt-5.4-mini": 400_000, "gpt-5.4-nano": 400_000,
    "o3": 200_000, "o3-pro": 200_000, "o3-mini": 200_000, "o4-mini": 200_000,
    # Anthropic
    "claude-opus-4-8": 1_000_000, "claude-opus-4-6": 1_000_000,
    "claude-sonnet-4-6": 1_000_000, "claude-haiku-4-5-20251001": 200_000,
    "claude-sonnet-4-5-20250929": 200_000, "claude-opus-4-5-20251101": 200_000,
    "claude-sonnet-4-0": 200_000, "claude-opus-4-0": 200_000,
    # Gemini
    "gemini-3.1-pro-preview": 1_048_576, "gemini-3-flash-preview": 1_048_576,
    "gemini-3.1-flash-lite-preview": 1_048_576,
    "gemini-2.5-pro": 1_048_576, "gemini-2.5-flash": 1_048_576,
    "gemini-2.5-flash-lite": 1_048_576,
    # Z.AI
    "glm-5": 200_000, "glm-5-turbo": 200_000, "glm-4.7": 200_000, "glm-4.6": 200_000,
    "glm-4.5": 128_000, "glm-4.5-flash": 128_000, "glm-4.5-air": 128_000,
    # Kimi
    "kimi-k2.5": 256_000, "kimi-k2-thinking": 256_000, "kimi-k2-thinking-turbo": 256_000,
    "kimi-k2-0905-preview": 256_000, "kimi-k2-turbo-preview": 256_000,
    # Minimax
    "MiniMax-M2.7": 204_800, "MiniMax-M2.7-highspeed": 204_800,
    "MiniMax-M2.5": 204_800, "MiniMax-M2.1": 204_800,
}


def _bare_model(model: str) -> str:
    """Strip the provider prefix from a LiteLLM model id."""
    for prefix in ("ollama_chat/", "ollama/"):
        if model.startswith(prefix):
            return model[len(prefix):]
    return model.split("/", 1)[1] if "/" in model else model


def _is_ollama(model: str, provider_id: str | None) -> bool:
    return provider_id == "ollama" or model.startswith(("ollama/", "ollama_chat/"))


def _fetch_ollama_trained_context(model: str, base_url: str) -> int | None:
    """Trained context length from Ollama ``/api/show``, or None if unknown."""
    import httpx

    cache_key = (model, base_url)
    if cache_key in _OLLAMA_CTX_CACHE:
        return _OLLAMA_CTX_CACHE[cache_key]

    bare = _bare_model(model)
    result: int | None = None
    with best_effort("ollama /api/show context length fetch failed"):
        resp = httpx.post(f"{base_url}/api/show", json={"name": bare}, timeout=5.0)
        if resp.status_code == 200:
            model_info = resp.json().get("model_info", {})
            for key, val in model_info.items():
                if key.endswith(".context_length") and isinstance(val, int):
                    result = val
                    break

    _OLLAMA_CTX_CACHE[cache_key] = result
    return result


def _cloud_context_window(model: str) -> int | None:
    """``max_input_tokens`` for a cloud model, or None if unknown.

    litellm's cost map is authoritative (it self-updates on upgrade); the
    hand-maintained override map only fills gaps for models litellm lacks.
    """
    bare = _bare_model(model)
    try:
        import litellm
        for name in (model, bare):
            info = litellm.model_cost.get(name)
            if info:
                ctx = info.get("max_input_tokens") or info.get("max_tokens")
                if ctx:
                    return int(ctx)
    except Exception:
        pass
    for name in (bare, model):
        if name in _CLOUD_CTX_OVERRIDES:
            return _CLOUD_CTX_OVERRIDES[name]
    return None


def get_model_context_window(
    llm_params: dict[str, Any], provider_id: str | None = None,
) -> int | None:
    """The real usable context window the backend enforces, or None if unknown.

    Local (Ollama): ``min(num_ctx, trained)`` — the KV cache Ollama actually
    allocates, capped by the model's trained length.  Cloud: litellm's
    ``max_input_tokens`` (with a small override fallback).
    """
    from infinidev.config.settings import settings

    model = llm_params.get("model") or settings.LLM_MODEL or ""
    if provider_id is None:
        provider_id = getattr(settings, "LLM_PROVIDER", "ollama")

    if _is_ollama(model, provider_id):
        base_url = (
            llm_params.get("base_url")
            or llm_params.get("api_base")
            or settings.LLM_BASE_URL
            or "http://localhost:11434"
        )
        num_ctx = llm_params.get("num_ctx") or (
            settings.OLLAMA_NUM_CTX if settings.OLLAMA_NUM_CTX > 0 else 0
        )
        trained = _fetch_ollama_trained_context(model, base_url)
        if num_ctx and trained:
            return min(num_ctx, trained)
        if num_ctx:
            return num_ctx
        # num_ctx unset: best-effort trained length (note: Ollama's own default
        # window is far smaller, so this can still over-report — set num_ctx).
        return trained

    return _cloud_context_window(model)


def _get_model_max_context(llm_params: dict[str, Any]) -> int:
    """Back-compat wrapper for the LoopEngine.

    Returns 0 when the window is unknown, which disables the context budget in
    the prompt (the engine treats 0 as "no budget").
    """
    return get_model_context_window(llm_params) or 0
