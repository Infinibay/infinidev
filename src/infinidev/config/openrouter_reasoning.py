"""Cache OpenRouter's public per-model controls, including mandatory reasoning."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_expires = 0.0
_models: dict[str, dict[str, Any]] = {}


def model_reasoning(model: str) -> dict[str, Any] | None:
    """Return metadata without inventing controls for dynamic routers or unknown models."""
    global _expires, _models
    if model in {"auto", "free", "openrouter/auto", "openrouter/free"}:
        return None
    with _lock:
        if time.monotonic() >= _expires:
            try:
                response = httpx.get("https://openrouter.ai/api/v1/models", timeout=3)
                response.raise_for_status()
                data = response.json()
                _models = {entry["id"]: entry["reasoning"] for entry in data.get("data", [])
                           if isinstance(entry, dict) and isinstance(entry.get("id"), str)
                           and isinstance(entry.get("reasoning"), dict)}
            except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
                logger.debug("OpenRouter reasoning metadata unavailable: %s", exc)
                _models = {}
            _expires = time.monotonic() + 300
        return _models.get(model)
