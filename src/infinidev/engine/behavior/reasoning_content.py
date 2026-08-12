"""Normalize provider-exposed reasoning without inspecting opaque thought data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


ReasoningVisibility = Literal["provider_exposed", "unavailable"]

_TEXT_KEYS = ("text", "thinking", "summary", "reasoning_content")
_OPAQUE_TYPES = frozenset({"redacted_thinking", "encrypted_thinking"})
_HISTORY_FIELDS = ("reasoning_content", "thinking_blocks", "reasoning_details")
_PROVIDER_HISTORY_FIELDS = frozenset({"reasoning_details", "thought_signatures"})


@dataclass(frozen=True)
class ReasoningEnvelope:
    """Visible reasoning text plus its normalized provider source fields."""

    text: str
    sources: tuple[str, ...] = ()
    visibility: ReasoningVisibility = "unavailable"


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return _plain(method(exclude_none=True))
            except TypeError:
                return _plain(method())
    return value


def _visible_text(value: Any) -> list[str]:
    """Extract only displayable text, never signatures or encrypted payloads."""
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value:
            result.extend(_visible_text(item))
        return result

    block_type = str(_get(value, "type", "") or "").casefold()
    if block_type in _OPAQUE_TYPES:
        return []
    for key in _TEXT_KEYS:
        candidate = _get(value, key)
        if candidate is not None:
            return _visible_text(candidate)
    return []


def _append_unique(parts: list[str], text: str) -> None:
    normalized = text.strip()
    if normalized and normalized not in parts:
        parts.append(normalized)


def extract_reasoning(message: Any) -> ReasoningEnvelope:
    """Read reasoning text from LiteLLM and provider-compatible response shapes.

    LiteLLM normally maps Anthropic/Gemini/OpenAI-compatible output to
    ``reasoning_content``. The fallbacks cover native MiniMax
    ``reasoning_details``, Anthropic/Gemini ``thinking_blocks``, and provider
    fields retained by LiteLLM. Opaque signatures and redacted blocks are
    intentionally excluded.
    """
    parts: list[str] = []
    sources: list[str] = []

    candidates: list[tuple[str, Any]] = [
        ("reasoning_content", _get(message, "reasoning_content")),
        ("thinking_blocks", _get(message, "thinking_blocks")),
        ("reasoning_details", _get(message, "reasoning_details")),
        ("reasoning", _get(message, "reasoning")),
    ]
    provider_fields = _get(message, "provider_specific_fields")
    if provider_fields:
        candidates.extend(
            (
                f"provider_specific_fields.{key}",
                _get(provider_fields, key),
            )
            for key in ("reasoning_content", "reasoning_details", "thinking_blocks")
        )

    for source, value in candidates:
        visible = _visible_text(value)
        if not visible:
            continue
        before = len(parts)
        for text in visible:
            _append_unique(parts, text)
        if len(parts) > before:
            sources.append(source)

    text = "\n\n".join(parts)
    return ReasoningEnvelope(
        text=text,
        sources=tuple(sources),
        visibility="provider_exposed" if text else "unavailable",
    )


class ReasoningStreamAccumulator:
    """Convert cumulative provider snapshots into deltas for the UI callback."""

    def __init__(self) -> None:
        self._snapshots: dict[str, str] = {}

    def consume(self, message: Any) -> str:
        envelope = extract_reasoning(message)
        if not envelope.text:
            return ""
        cumulative_sources = tuple(
            source for source in envelope.sources if "reasoning_details" in source
        )
        if not cumulative_sources:
            return envelope.text

        key = cumulative_sources[0]
        previous = self._snapshots.get(key, "")
        current = envelope.text
        self._snapshots[key] = current
        if previous and current.startswith(previous):
            return current[len(previous) :]
        if current == previous:
            return ""
        return current


def reasoning_history_fields(message: Any) -> dict[str, Any]:
    """Return protocol fields that must survive an assistant/tool round trip.

    Visible reasoning is not modified. Provider signatures remain opaque and
    are copied only so Anthropic, Gemini, MiniMax, and GLM can continue a
    multi-turn tool-use chain.
    """
    result: dict[str, Any] = {}
    for key in _HISTORY_FIELDS:
        value = _get(message, key)
        if value is not None:
            result[key] = _plain(value)

    provider_fields = _get(message, "provider_specific_fields")
    if provider_fields:
        safe_fields = {
            key: _plain(_get(provider_fields, key))
            for key in _PROVIDER_HISTORY_FIELDS
            if _get(provider_fields, key) is not None
        }
        if safe_fields:
            result["provider_specific_fields"] = safe_fields
    return result


__all__ = [
    "ReasoningEnvelope",
    "ReasoningStreamAccumulator",
    "extract_reasoning",
    "reasoning_history_fields",
]
