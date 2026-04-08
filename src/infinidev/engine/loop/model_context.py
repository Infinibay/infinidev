"""Fetch model context window size from Ollama."""

from __future__ import annotations

from typing import Any

# (model, base_url) -> context_length. Memoized because the result
# is intrinsic to the model and never changes within a process — but
# the function used to fire an HTTP POST to ollama /api/show on every
# ``LoopEngine._build_context()`` call, which added ~500 ms to every
# phase transition (analysis → develop, develop → review). Caching
# turns the second-and-onwards calls into ~0 µs dict lookups.
_MAX_CONTEXT_CACHE: dict[tuple[str, str], int] = {}


def _get_model_max_context(llm_params: dict[str, Any]) -> int:
    """Fetch the model's max context window from Ollama /api/show.

    Memoized by ``(model, base_url)`` for the lifetime of the process.
    Returns 0 if unknown (disables context budget in the prompt). The
    zero-result is also cached so a transient ollama failure doesn't
    cause repeated 5-second timeouts on every subsequent call.
    """
    import httpx

    model = llm_params.get("model", "")
    base_url = llm_params.get("base_url", "http://localhost:11434")

    cache_key = (model, base_url)
    cached = _MAX_CONTEXT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    bare_model = model
    for prefix in ("ollama_chat/", "ollama/"):
        if bare_model.startswith(prefix):
            bare_model = bare_model[len(prefix):]
            break

    result = 0
    try:
        resp = httpx.post(
            f"{base_url}/api/show",
            json={"name": bare_model},
            timeout=5.0,
        )
        if resp.status_code == 200:
            model_info = resp.json().get("model_info", {})
            for key, val in model_info.items():
                if key.endswith(".context_length") and isinstance(val, int):
                    result = val
                    break
    except Exception:
        pass

    _MAX_CONTEXT_CACHE[cache_key] = result
    return result

# Max times the LLM can respond with text instead of tool calls before
# forcing a step_complete.  Text responses are kept as context (the model
# may be reasoning), so a higher limit is fine.
