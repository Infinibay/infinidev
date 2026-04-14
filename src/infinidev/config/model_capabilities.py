"""Model capability detection via runtime probing.

Runs lightweight test calls at startup to determine what the configured LLM
actually supports (function calling, tool_choice=required, JSON mode, etc.)
rather than relying on hardcoded provider lists.

For Ollama models, the ``/api/show`` endpoint is queried first: if the model
template lacks tool-calling markers (``.ToolCalls``, ``if .Tools``), the model
is assumed to lack native function-calling support and the system falls back
to manual (text-based) tool calling without ever sending a slow probe request.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Timeout (seconds) for probe calls — much shorter than the general LLM
# timeout so a failing probe doesn't block the session for minutes.
_PROBE_TIMEOUT = 30


@dataclass
class ModelCapabilities:
    """Runtime-detected capabilities of the configured LLM."""

    supports_function_calling: bool = True
    supports_tool_choice_required: bool = True
    supports_json_mode: bool = True
    has_thinking_sections: bool = False
    needs_schema_sanitization: bool = False
    probed: bool = False
    probe_duration: float = 0.0


# Module-level singleton
_capabilities = ModelCapabilities()

# Tiny tool used for probing — minimal tokens, clear expected behavior
_PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Perform basic arithmetic. Return the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate",
                },
            },
            "required": ["expression"],
        },
    },
}

# Tool with anyOf in schema — tests schema sanitization needs
_PROBE_TOOL_ANYOF = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Perform basic arithmetic. Return the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate",
                },
                "precision": {
                    "anyOf": [
                        {"type": "integer"},
                        {"type": "null"},
                    ],
                    "description": "Decimal places (optional)",
                },
            },
            "required": ["expression"],
        },
    },
}

_PROBE_MESSAGES = [
    {"role": "user", "content": "Calculate 2+2. Use the calculator tool."},
]

# Thinking section markers emitted by various models
_THINKING_MARKERS = ("<thinking>", "<|thinking|>", "<think>")


# Known capability presets per provider — skip probing for these
_PROVIDER_PRESETS: dict[str, ModelCapabilities] = {
    "openai": ModelCapabilities(
        supports_function_calling=True,
        supports_tool_choice_required=True,
        supports_json_mode=True,
        probed=True,
    ),
    "anthropic": ModelCapabilities(
        supports_function_calling=True,
        supports_tool_choice_required=True,
        supports_json_mode=True,
        probed=True,
    ),
    "gemini": ModelCapabilities(
        supports_function_calling=True,
        supports_tool_choice_required=True,
        supports_json_mode=True,
        probed=True,
    ),
    "zai": ModelCapabilities(
        supports_function_calling=True,
        supports_tool_choice_required=False,
        supports_json_mode=True,
        probed=True,
    ),
    "kimi": ModelCapabilities(
        supports_function_calling=True,
        supports_tool_choice_required=False,
        supports_json_mode=True,
        probed=True,
    ),
    "minimax": ModelCapabilities(
        supports_function_calling=True,
        supports_tool_choice_required=False,
        supports_json_mode=True,
        has_thinking_sections=True,  # MiniMax M2.7 sends reasoning_content like DeepSeek
        probed=True,
    ),
    "openrouter": ModelCapabilities(
        supports_function_calling=True,
        supports_tool_choice_required=False,  # varies by model — be conservative
        supports_json_mode=False,             # varies by model — be conservative
        probed=True,
    ),
    "qwen": ModelCapabilities(
        supports_function_calling=True,
        supports_tool_choice_required=True,
        supports_json_mode=True,
        needs_schema_sanitization=True,  # Qwen rejects anyOf and complex schemas
        probed=True,
    ),
}


def get_model_capabilities() -> ModelCapabilities:
    """Return model capabilities. Uses provider presets for known cloud providers.

    For Ollama, queries the ``/api/show`` endpoint to check if the model
    template includes tool-calling markers.  This is fast (~5 ms local) and
    avoids a 30-second probe that would hang for unsupported models.
    """
    global _capabilities
    if not _capabilities.probed:
        try:
            from infinidev.config.settings import settings
            provider_id = getattr(settings, "LLM_PROVIDER", "ollama")
            if provider_id in _PROVIDER_PRESETS:
                _capabilities = _PROVIDER_PRESETS[provider_id]
            elif provider_id == "ollama":
                _capabilities = _detect_ollama_capabilities()
            elif provider_id in ("openai_compatible", "llama_cpp", "vllm"):
                _capabilities = _detect_openai_compatible_capabilities()
        except Exception:
            pass
    return _capabilities


# ── Ollama-specific detection ───────────────────────────────────────────

# Markers in the Go template that indicate tool-calling support.
_TOOL_TEMPLATE_MARKERS = (".ToolCalls", "if .Tools", ".tools")


def _detect_ollama_capabilities() -> ModelCapabilities:
    """Detect capabilities for an Ollama model via /api/show.

    Checks whether the model template contains tool-calling markers.
    Falls back to a short live probe if the template check is inconclusive.
    """
    import httpx
    from infinidev.config.settings import settings

    model = settings.LLM_MODEL or ""
    base_url = settings.LLM_BASE_URL or "http://localhost:11434"

    # Strip litellm prefix to get bare model name
    bare_model = model
    for prefix in ("ollama_chat/", "ollama/"):
        if bare_model.startswith(prefix):
            bare_model = bare_model[len(prefix):]
            break

    caps = ModelCapabilities(probed=True)

    try:
        resp = httpx.post(
            f"{base_url}/api/show",
            json={"name": bare_model},
            timeout=5.0,
        )
        if resp.status_code != 200:
            logger.warning(
                "Ollama /api/show returned %d for %s — assuming full capabilities",
                resp.status_code, bare_model,
            )
            return caps

        data = resp.json()
        template = data.get("template", "")

        has_tool_markers = any(m in template for m in _TOOL_TEMPLATE_MARKERS)

        if has_tool_markers:
            logger.info(
                "Ollama model %s: template has tool-calling markers — FC enabled",
                bare_model,
            )
            caps.supports_function_calling = True
            # tool_choice="required" is generally unreliable on Ollama
            caps.supports_tool_choice_required = False
        else:
            logger.info(
                "Ollama model %s: template lacks tool-calling markers — "
                "falling back to manual tool calling",
                bare_model,
            )
            caps.supports_function_calling = False
            caps.supports_tool_choice_required = False

        # JSON mode is generally supported by Ollama
        caps.supports_json_mode = True

        # Check for thinking markers in the template
        template_lower = template.lower()
        if any(m in template_lower for m in ("<thinking>", "<|thinking|>", "<think>")):
            caps.has_thinking_sections = True

    except Exception as exc:
        logger.warning(
            "Failed to query Ollama /api/show for %s: %s — assuming full capabilities",
            bare_model, str(exc)[:200],
        )

    return caps


def _detect_openai_compatible_capabilities() -> ModelCapabilities:
    """Detect capabilities for an OpenAI-compatible server (llama.cpp, vLLM, TGI, etc.).

    For llama_cpp and vLLM: runs a live probe to test FC support. Modern
    llama-server and vLLM both support native function calling for most models.

    For generic openai_compatible: conservative defaults (manual mode).
    """
    from infinidev.config.settings import settings

    provider_id = getattr(settings, "LLM_PROVIDER", "openai_compatible")

    if provider_id in ("llama_cpp", "vllm"):
        # vLLM has robust FC — probe it
        try:
            from infinidev.config.llm import get_litellm_params
            caps = _run_probes(get_litellm_params())
            caps.probed = True
            return caps
        except Exception as exc:
            logger.warning("vLLM probe failed: %s — falling back to manual TC", str(exc)[:200])

    # Generic openai_compatible: conservative defaults
    return ModelCapabilities(
        supports_function_calling=False,
        supports_tool_choice_required=False,
        supports_json_mode=False,
        probed=True,
    )


def probe_model(llm_params: dict[str, Any]) -> ModelCapabilities:
    """Probe the configured LLM to detect its capabilities.

    Runs 2-3 fast test calls (~3-5s total) with max_tokens=100.
    Updates the module-level singleton and returns it.

    Args:
        llm_params: kwargs dict for litellm.completion (model, api_key, etc.)
    """
    global _capabilities

    caps = ModelCapabilities()
    start = time.monotonic()

    try:
        caps = _run_probes(llm_params)
    except Exception as exc:
        logger.warning(
            "Model probe failed — using defaults: %s", str(exc)[:200]
        )
        caps = ModelCapabilities()

    caps.probe_duration = time.monotonic() - start
    caps.probed = True
    _capabilities = caps

    # Log results
    parts = [
        f"function_calling={'yes' if caps.supports_function_calling else 'NO'}",
        f"tool_choice_required={'yes' if caps.supports_tool_choice_required else 'NO'}",
        f"json_mode={'yes' if caps.supports_json_mode else 'NO'}",
        f"thinking={'yes' if caps.has_thinking_sections else 'no'}",
        f"schema_sanitization={'needed' if caps.needs_schema_sanitization else 'no'}",
    ]
    logger.info(
        "Model capabilities probed in %.1fs: %s",
        caps.probe_duration,
        ", ".join(parts),
    )
    return caps


def _run_probes(llm_params: dict[str, Any]) -> ModelCapabilities:
    """Execute the probe sequence."""
    import litellm

    caps = ModelCapabilities()
    probe_params = {**llm_params, "max_tokens": 100, "timeout": _PROBE_TIMEOUT}

    # ── Probe 1: Function calling + tool_choice=required ─────────────
    try:
        resp = litellm.completion(
            **probe_params,
            messages=_PROBE_MESSAGES,
            tools=[_PROBE_TOOL],
            tool_choice="required",
        )
        choice = resp.choices[0]
        tool_calls = getattr(choice.message, "tool_calls", None)
        content = getattr(choice.message, "content", "") or ""

        if tool_calls:
            # FC works with required — best case
            caps.supports_function_calling = True
            caps.supports_tool_choice_required = True
            # Check for thinking markers in content alongside tool calls
            _check_thinking(caps, content)
        else:
            # Model returned text despite tool_choice=required.
            # Try again with tool_choice=auto to distinguish:
            # "no FC at all" vs "doesn't support required"
            _check_thinking(caps, content)
            caps.supports_tool_choice_required = False
            try:
                resp2 = litellm.completion(
                    **probe_params,
                    messages=_PROBE_MESSAGES,
                    tools=[_PROBE_TOOL],
                    tool_choice="auto",
                )
                tc2 = getattr(resp2.choices[0].message, "tool_calls", None)
                if tc2:
                    caps.supports_function_calling = True
                else:
                    caps.supports_function_calling = False
            except Exception:
                caps.supports_function_calling = False

    except Exception as exc:
        exc_msg = str(exc).lower()
        # Some providers reject tool_choice entirely
        if "tool_choice" in exc_msg:
            caps.supports_tool_choice_required = False
            # Try without tool_choice
            try:
                resp_fallback = litellm.completion(
                    **probe_params,
                    messages=_PROBE_MESSAGES,
                    tools=[_PROBE_TOOL],
                )
                tc_fb = getattr(resp_fallback.choices[0].message, "tool_calls", None)
                caps.supports_function_calling = bool(tc_fb)
            except Exception:
                caps.supports_function_calling = False
        elif "tools" in exc_msg or "function" in exc_msg:
            # Provider rejects tools parameter entirely
            caps.supports_function_calling = False
            caps.supports_tool_choice_required = False
        else:
            raise

    # ── Probe 2: JSON mode ───────────────────────────────────────────
    try:
        litellm.completion(
            **probe_params,
            messages=[{"role": "user", "content": "Return {\"x\": 1}"}],
            response_format={"type": "json_object"},
        )
        caps.supports_json_mode = True
    except Exception as exc:
        exc_msg = str(exc).lower()
        if "response_format" in exc_msg or "json" in exc_msg:
            caps.supports_json_mode = False
        else:
            # Unrelated error — assume JSON mode is fine
            caps.supports_json_mode = True

    # ── Probe 3: Schema sanitization (anyOf) ─────────────────────────
    if caps.supports_function_calling:
        try:
            litellm.completion(
                **probe_params,
                messages=_PROBE_MESSAGES,
                tools=[_PROBE_TOOL_ANYOF],
                tool_choice="auto",
            )
            caps.needs_schema_sanitization = False
        except Exception as exc:
            exc_msg = str(exc).lower()
            if "anyof" in exc_msg or "schema" in exc_msg or "parameter" in exc_msg:
                caps.needs_schema_sanitization = True
            else:
                # Unrelated error — don't assume sanitization needed
                caps.needs_schema_sanitization = False

    return caps


def _check_thinking(caps: ModelCapabilities, content: str) -> None:
    """Check response content for thinking section markers."""
    content_lower = content.lower()
    if any(marker in content_lower for marker in _THINKING_MARKERS):
        caps.has_thinking_sections = True


def _reset_capabilities() -> None:
    """Reset to defaults. For tests only."""
    global _capabilities
    _capabilities = ModelCapabilities()
