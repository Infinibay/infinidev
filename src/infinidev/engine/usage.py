"""Thread-safe usage observed at the shared completion boundary."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from copy import deepcopy
from typing import Any

logger = logging.getLogger(__name__)


def _dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    data = dump() if callable(dump) else None
    return data if isinstance(data, dict) else {}


def _key(params: dict) -> str:
    from infinidev.config.providers import PROVIDERS

    model = str(params.get("model", ""))
    prefix = model.split("/")[0]
    provider = PROVIDERS.get("ollama" if prefix == "ollama_chat" else prefix)
    base = params.get("api_base") or (provider.default_base_url if provider else prefix)
    headers = {k.lower(): v for k, v in (params.get("extra_headers") or {}).items()}
    account = headers.get("chatgpt-account-id") or params.get("api_key", "")
    identity = [str(base).rstrip("/"), model.rsplit("/", 1)[-1], account]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def _empty() -> dict:
    return dict(requests=0, input_tokens=0, output_tokens=0, cached_tokens=0,
                missing_usage=0, cost=0.0, priced_requests=0, headers={}, observed_at=None)


class UsageLedger:
    """Keep process totals scoped to endpoint, model and credential/account."""

    def __init__(self) -> None:
        self._rows: dict[str, dict] = {}
        self._lock = threading.Lock()

    def snapshot(self, params: dict) -> dict:
        with self._lock:
            return deepcopy(self._rows.get(_key(params), _empty()))

    def observe(self, response: Any, params: dict) -> Any:
        """Pass responses through unchanged, recording stream totals only once."""
        key = _key(params)
        if not params.get("stream"):
            self._record(key, getattr(response, "usage", None), _headers(response),
                         _dict(getattr(response, "_hidden_params", {})).get("response_cost"))
            return response

        def stream():
            usage, cost = None, None
            usage_metadata = {}
            headers = _headers(response)
            try:
                for chunk in response:
                    if getattr(chunk, "usage", None) is not None:
                        usage = chunk.usage
                    headers.update(_headers(chunk))
                    hidden = _dict(getattr(chunk, "_hidden_params", {}))
                    if "usage" in hidden:
                        # LiteLLM keeps native stream totals here and can replace
                        # them on exhaustion; retain the mapping until then.
                        usage_metadata = hidden
                    cost = hidden.get("response_cost", cost)
                    yield chunk
            finally:
                if usage_metadata.get("usage") is not None:
                    usage = usage_metadata["usage"]
                self._record(key, usage, headers, cost)
                close = getattr(response, "close", None)
                if callable(close):
                    close()

        return stream()

    def _record(self, key: str, usage: Any, headers: dict, cost: Any) -> None:
        try:
            data = _dict(usage)
            input_tokens = data.get("prompt_tokens", data.get("input_tokens"))
            output_tokens = data.get("completion_tokens", data.get("output_tokens"))
            details = _dict(data.get("prompt_tokens_details", data.get("input_tokens_details")))
            cached = details.get("cached_tokens", data.get("cache_read_input_tokens", 0))
            with self._lock:
                row = self._rows.setdefault(key, _empty())
                row["requests"] += 1
                if input_tokens is None or output_tokens is None:
                    row["missing_usage"] += 1
                else:
                    row["input_tokens"] += max(0, int(input_tokens))
                    row["output_tokens"] += max(0, int(output_tokens))
                    row["cached_tokens"] += max(0, int(cached or 0))
                if isinstance(cost, (int, float)) and cost > 0:
                    row["cost"] += cost
                    row["priced_requests"] += 1
                if headers:
                    row["headers"] = headers
                    row["observed_at"] = time.time()
        except (TypeError, ValueError, OverflowError):
            logger.debug("Provider returned unreadable usage metadata")


def _headers(response: Any) -> dict[str, str]:
    hidden = _dict(getattr(response, "_hidden_params", {}))
    headers = {}
    for source in (hidden.get("additional_headers"), hidden.get("headers"),
                   getattr(response, "response_headers", None)):
        if not hasattr(source, "items"):
            continue
        for raw_key, value in source.items():
            key = str(raw_key).lower().removeprefix("llm_provider-")
            if key.startswith(("x-ratelimit-", "anthropic-ratelimit-")):
                headers[key] = str(value)
    return headers


usage_ledger = UsageLedger()
