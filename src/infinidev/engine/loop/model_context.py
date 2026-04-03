"""Fetch model context window size from Ollama."""

from __future__ import annotations

from typing import Any


def _get_model_max_context(llm_params: dict[str, Any]) -> int:
    """Fetch the model's max context window from Ollama /api/show.

    Returns 0 if unknown (disables context budget in the prompt).
    """
    import httpx

    model = llm_params.get("model", "")
    base_url = llm_params.get("base_url", "http://localhost:11434")

    bare_model = model
    for prefix in ("ollama_chat/", "ollama/"):
        if bare_model.startswith(prefix):
            bare_model = bare_model[len(prefix):]
            break

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
                    return val
    except Exception:
        pass
    return 0

# Max times the LLM can respond with text instead of tool calls before
# forcing a step_complete.  Text responses are kept as context (the model
# may be reasoning), so a higher limit is fine.
