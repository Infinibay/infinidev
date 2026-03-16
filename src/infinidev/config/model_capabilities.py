"""Model capability detection via runtime probing.

Runs lightweight test calls at startup to determine what the configured LLM
actually supports (function calling, tool_choice=required, JSON mode, etc.)
rather than relying on hardcoded provider lists.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


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


def get_model_capabilities() -> ModelCapabilities:
    """Return the current model capabilities (probed or defaults)."""
    return _capabilities


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
    probe_params = {**llm_params, "max_tokens": 100}

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
