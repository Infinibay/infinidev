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

# Provider-documented cloud context windows.  These entries win over LiteLLM:
# its bundled cost map can know a model while still carrying an older limit
# (for example GPT-5.4 mini at 272k instead of 400k, or MiniMax M2.5 at 1M
# instead of 204.8k).  Models absent here still fall back to LiteLLM so dynamic
# catalogs such as OpenRouter continue to work without a local mirror.
_CLOUD_CTX_OVERRIDES: dict[str, int] = {
    # OpenAI
    "gpt-5.6-sol": 1_050_000,
    "gpt-5.6-terra": 1_050_000,
    "gpt-5.6-luna": 1_050_000,
    "gpt-5.6": 1_050_000,
    "gpt-5.5-pro": 1_050_000,
    "gpt-5.5": 1_050_000,
    "gpt-5.4": 1_050_000,
    "gpt-5.4-mini": 400_000,
    "gpt-5.4-nano": 400_000,
    "o3": 200_000,
    "o3-pro": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    # Anthropic — aliases, matching the catalog in config/providers.py. A
    # dated id here would only ever match a dated id there, and the catalog
    # no longer offers any.
    "claude-opus-5": 1_000_000,
    "claude-sonnet-5": 1_000_000,
    "claude-fable-5": 1_000_000,
    "claude-opus-4-8": 1_000_000,
    "claude-opus-4-7": 1_000_000,
    "claude-opus-4-6": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-haiku-4-5": 200_000,
    "claude-opus-4-5": 200_000,
    "claude-sonnet-4-5": 200_000,
    # Gemini
    "gemini-3.6-flash": 1_048_576,
    "gemini-3.5-flash": 1_048_576,
    "gemini-3.5-flash-lite": 1_048_576,
    "gemini-3.1-pro-preview": 1_048_576,
    "gemini-3.1-flash-lite": 1_048_576,
    "gemini-3-pro-preview": 1_048_576,
    "gemini-3-flash-preview": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.5-flash-lite": 1_048_576,
    # Z.AI
    "glm-5.2": 200_000,
    "glm-5.1": 200_000,
    "glm-5": 200_000,
    "glm-5-turbo": 200_000,
    "glm-4.7": 200_000,
    "glm-4.7-flash": 200_000,
    "glm-4.6": 200_000,
    "glm-4.5": 128_000,
    "glm-4.5-flash": 200_000,
    "glm-4.5-air": 128_000,
    # Qwen — LiteLLM indexes DashScope models under prefixes that never match
    # this provider's `custom_openai/` prefix.  Only limits explicitly listed
    # by Alibaba are included; the unlisted legacy aliases remain unknown.
    "qwen3.8-max-preview": 1_000_000,
    "qwen3.7-max": 1_000_000,
    "qwen3.7-plus": 1_000_000,
    "qwen3.7-flash": 1_000_000,
    "qwen3.6-max-preview": 256_000,
    "qwen3.6-plus": 1_000_000,
    "qwen3.6-flash": 1_000_000,
    "qwen3.5-plus": 1_000_000,
    "qwen3.5-flash": 1_000_000,
    "qwen3.5-397b-a17b": 256_000,
    "qwen3.5-122b-a10b": 256_000,
    "qwen3-coder-plus": 1_000_000,
    # Kimi — K3 is a 1M-context model; the 256k figure here was the K2 line's.
    "kimi-k3": 1_048_576,
    "kimi-k2.7-code": 256_000,
    "kimi-k2.6": 256_000,
    # MiniMax. M3 is a separate 1M model; the M2 API family is 204.8k total
    # input + output despite several LiteLLM entries currently claiming 1M.
    "MiniMax-M3": 1_000_000,
    "MiniMax-M2.7": 204_800,
    "MiniMax-M2.7-highspeed": 204_800,
    "MiniMax-M2.5": 204_800,
    "MiniMax-M2.5-highspeed": 204_800,
    "MiniMax-M2.1": 204_800,
    "MiniMax-M2.1-highspeed": 204_800,
    "MiniMax-M2": 204_800,
    # Mistral aliases. These values track what each current `-latest` alias
    # serves, not the retired snapshots that LiteLLM still associates with a
    # few of the names.
    "mistral-large-latest": 262_144,
    "mistral-medium-latest": 262_144,
    "mistral-small-latest": 262_144,
    "ministral-3b-latest": 262_144,
    "ministral-8b-latest": 262_144,
    "magistral-medium-latest": 131_072,
    "magistral-small-latest": 131_072,
    "codestral-latest": 131_072,
    "devstral-medium-latest": 256_000,
    "devstral-small-latest": 256_000,
    "pixtral-large-latest": 128_000,
    # GMI's hosted DeepSeek-R1 deployment advertises a 128k total window.
    "deepseek-ai/DeepSeek-R1": 128_000,
}


def _bare_model(model: str) -> str:
    """Strip the provider prefix from a LiteLLM model id."""
    for prefix in ("ollama_chat/", "ollama/"):
        if model.startswith(prefix):
            return model[len(prefix) :]
    bare = model.split("/", 1)[1] if "/" in model else model
    # `openai/responses/gpt-5.5` carries a protocol segment as well as a
    # provider one. Both catalogs key on the bare slug, so a single split
    # leaves `responses/gpt-5.5` — a name nothing matches, which reads as
    # "context window unknown" rather than as the bug it is.
    if bare.startswith("responses/"):
        bare = bare[len("responses/") :]
    return bare


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
    """Documented input/context limit for a cloud model, or None if unknown.

    Provider-documented overrides are authoritative for known catalog models;
    LiteLLM fills in dynamic and otherwise-unlisted model ids.
    """
    bare = _bare_model(model)
    for name in (bare, model):
        if name in _CLOUD_CTX_OVERRIDES:
            return _CLOUD_CTX_OVERRIDES[name]
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
    return None


def get_model_context_window(
    llm_params: dict[str, Any],
    provider_id: str | None = None,
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

    # The subscription serves the same model names as the metered API with
    # different limits — litellm's map says gpt-5.5 takes 1 050 000 input
    # tokens, the Codex backend gives it 272 000. Reading the cost map here
    # would hand the loop ~800 000 tokens of headroom that do not exist.
    from infinidev.config.llm import CHATGPT_SUBSCRIPTION_PROVIDER

    if provider_id == CHATGPT_SUBSCRIPTION_PROVIDER:
        from infinidev.config.codex_catalog import context_window

        return context_window(_bare_model(model))

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


def get_model_max_context_window(
    llm_params: dict[str, Any],
    provider_id: str | None = None,
) -> int | None:
    """The model's advertised maximum, independent of a smaller served limit.

    This is display metadata only.  Prompt budgeting must use
    :func:`get_model_context_window`, because a provider surface can enforce a
    lower ceiling than the underlying model (notably ChatGPT subscriptions).
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
        return _fetch_ollama_trained_context(model, base_url)

    return _cloud_context_window(model)


def _get_model_max_context(llm_params: dict[str, Any]) -> int:
    """Back-compat wrapper for the LoopEngine.

    Returns 0 when the window is unknown, which disables the context budget in
    the prompt (the engine treats 0 as "no budget").
    """
    return get_model_context_window(llm_params) or 0
