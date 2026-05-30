"""ChatGPT Codex subscription provider adapter.

This adapter talks to ChatGPT's Codex Responses backend using OAuth tokens
from ``openai_auth`` and returns a LiteLLM-shaped ChatCompletion response so
Infinidev's existing loop can keep using ``message.content`` and
``message.tool_calls``.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import httpx

from infinidev.config.openai_auth import codex_oauth_headers

_CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
_CODEX_MODELS_URL = "https://chatgpt.com/backend-api/codex/models"
_CLIENT_VERSION = "0.6.0"
_STATIC_MODELS = [
    # GPT-5.5 general presets. Kept as known presets even when the
    # account-specific Codex catalog has not rolled them out yet.
    "gpt-5.5",
    "gpt-5.5-none",
    "gpt-5.5-low",
    "gpt-5.5-medium",
    "gpt-5.5-high",
    "gpt-5.5-xhigh",
    # Modern OpenCode-compatible presets. Variants normalize to the
    # backend model slug while controlling reasoning effort.
    "gpt-5.2",
    "gpt-5.2-none",
    "gpt-5.2-low",
    "gpt-5.2-medium",
    "gpt-5.2-high",
    "gpt-5.2-xhigh",
    "gpt-5.2-codex",
    "gpt-5.2-codex-low",
    "gpt-5.2-codex-medium",
    "gpt-5.2-codex-high",
    "gpt-5.2-codex-xhigh",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex-max-low",
    "gpt-5.1-codex-max-medium",
    "gpt-5.1-codex-max-high",
    "gpt-5.1-codex-max-xhigh",
    "gpt-5.1-codex",
    "gpt-5.1-codex-low",
    "gpt-5.1-codex-medium",
    "gpt-5.1-codex-high",
    "gpt-5.1-codex-mini",
    "gpt-5.1-codex-mini-medium",
    "gpt-5.1-codex-mini-high",
    "gpt-5.1",
    "gpt-5.1-none",
    "gpt-5.1-low",
    "gpt-5.1-medium",
    "gpt-5.1-high",
    # Legacy aliases.
    "codex-mini-latest",
    "gpt-5-codex",
    "gpt-5-codex-mini",
    "gpt-5",
]

def static_models(prefix: str = "openai_codex/") -> list[str]:
    return [f"{prefix}{m}" for m in _STATIC_MODELS]


def fetch_models(
    prefix: str = "openai_codex/",
    base_url: str = "",
    client_version: str = _CLIENT_VERSION,
) -> list[str]:
    """Return known Codex subscription presets.

    The ChatGPT Codex catalog is account-specific and may currently return
    only a subset (for example just ``gpt-5.2``). OpenCode exposes known
    presets and normalizes them at request time, so Infinidev does the same.
    A catalog call is attempted for remote-first ordering, but a catalog
    outage does not hide known presets from the selector.
    """
    try:
        remote = fetch_remote_model_slugs(base_url=base_url, client_version=client_version)
    except Exception:
        remote = []

    ordered: list[str] = []
    seen: set[str] = set()
    for model in [*remote, *_STATIC_MODELS]:
        if model and model not in seen:
            seen.add(model)
            ordered.append(model)
    return [f"{prefix}{m}" for m in ordered]


def fetch_remote_model_slugs(
    *,
    base_url: str = "",
    client_version: str = _CLIENT_VERSION,
) -> list[str]:
    """Fetch the account catalog and raise on auth/network/backend failures."""
    url = _models_url(base_url)
    headers = {"Accept": "application/json", "originator": "codex_cli_rs"}
    headers.update(codex_oauth_headers())
    resp = httpx.get(url, headers=headers, params={"client_version": client_version}, timeout=8)
    resp.raise_for_status()
    return _model_slugs(resp.json().get("models", []))


def _models_url(base_url: str = "") -> str:
    url = (base_url or "https://chatgpt.com/backend-api/codex").rstrip("/")
    if url.endswith("/models"):
        return url
    if url.endswith("/responses"):
        return url[: -len("/responses")] + "/models"
    return f"{url}/models"


def _responses_url(base_url: str = "") -> str:
    url = (base_url or _CODEX_RESPONSES_URL).rstrip("/")
    if url.endswith("/responses"):
        return url
    if url.endswith("/models"):
        return url[: -len("/models")] + "/responses"
    return f"{url}/responses"


def _model_slugs(models: list[Any]) -> list[str]:
    def priority(model: Any) -> int:
        if isinstance(model, dict):
            try:
                return int(model.get("priority", 0))
            except (TypeError, ValueError):
                return 0
        return 0

    slugs: list[str] = []
    seen: set[str] = set()
    for model in sorted(models, key=priority):
        if not isinstance(model, dict):
            continue
        visibility = str(model.get("visibility", "list")).lower()
        if visibility != "list":
            continue
        slug = str(model.get("slug") or model.get("id") or "").strip()
        if slug and slug not in seen:
            seen.add(slug)
            slugs.append(slug)
    return slugs


def completion(
    *,
    params: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] = "auto",
    stream: bool = False,
    on_thinking_chunk: Any | None = None,
    on_stream_status: Any | None = None,
) -> Any:
    """Call ChatGPT Codex backend and return a LiteLLM ModelResponse."""
    request = _build_request(params, messages, tools, tool_choice)
    headers = _headers(request)
    responses_url = _responses_url(str(params.get("api_base") or params.get("base_url") or ""))
    with httpx.Client(timeout=float(params.get("timeout") or 1800)) as client:
        with client.stream("POST", responses_url, headers=headers, json=request) as resp:
            if resp.status_code >= 400:
                text = resp.read().decode(errors="replace")
                if resp.status_code == 404 and ("usage" in text.lower() or "rate" in text.lower()):
                    raise RuntimeError(f"Codex subscription usage limit reached: {text[:300]}")
                raise RuntimeError(
                    f"Codex subscription request failed ({resp.status_code}): {text[:1000] or resp.reason_phrase}"
                )
            events = list(_iter_sse(resp))

    if on_stream_status:
        on_stream_status("done", len(events), None)
    return _response_from_events(events, request["model"])


def _build_request(
    params: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any],
) -> dict[str, Any]:
    requested_model = str(params.get("model") or "gpt-5.2-medium")
    model = _normalize_model(requested_model)
    instructions, input_items = _messages_to_responses_input(messages)
    request: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input_items,
        "tools": _tools_to_responses(tools or []),
        "tool_choice": _tool_choice_to_responses(tool_choice),
        "parallel_tool_calls": True,
        "store": False,
        "stream": True,
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": str(params.get("prompt_cache_key") or "infinidev"),
        "text": {"verbosity": "medium"},
    }
    # ChatGPT's Codex backend rejects sampling parameters like temperature,
    # even though several Infinidev call sites set them for normal LiteLLM providers.
    request["reasoning"] = {"effort": _reasoning_effort(requested_model, model), "summary": "auto"}
    return request


def _headers(request: dict[str, Any]) -> dict[str, str]:
    headers = {
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        "OpenAI-Beta": "responses=experimental",
        "originator": "codex_cli_rs",
    }
    headers.update(codex_oauth_headers())
    cache_key = request.get("prompt_cache_key")
    if cache_key:
        headers["conversation_id"] = str(cache_key)
        headers["session_id"] = str(cache_key)
    return headers


def _strip_provider(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def _normalize_model(model: str) -> str:
    model = _strip_provider(model).strip()
    lower = model.lower()
    compact = lower.replace(" ", "-").replace("_", "-")
    explicit_map = {
        "codex-mini-latest": "gpt-5.1-codex-mini",
        "gpt-5-codex-mini": "gpt-5.1-codex-mini",
        "gpt-5-codex-mini-medium": "gpt-5.1-codex-mini",
        "gpt-5-codex-mini-high": "gpt-5.1-codex-mini",
        "gpt-5-codex": "gpt-5.1-codex",
        "gpt-5": "gpt-5.1",
        "gpt-5-mini": "gpt-5.1",
        "gpt-5-nano": "gpt-5.1",
    }
    if lower in explicit_map:
        return explicit_map[lower]
    if compact in explicit_map:
        return explicit_map[compact]
    for family in (
        "gpt-5.5",
        "gpt-5.2-codex",
        "gpt-5.2",
        "gpt-5.1-codex-max",
        "gpt-5.1-codex-mini",
        "gpt-5.1-codex",
        "gpt-5.1",
    ):
        if compact == family or compact.startswith(f"{family}-"):
            return family
    if "codex-mini" in compact:
        return "gpt-5.1-codex-mini"
    if "codex-max" in compact:
        return "gpt-5.1-codex-max"
    if "codex" in compact:
        return "gpt-5.1-codex"
    if "gpt-5" in compact:
        return "gpt-5.1"
    return model or "gpt-5.2"


def _reasoning_effort(requested_model: str, normalized_model: str | None = None) -> str:
    requested = _strip_provider(requested_model).lower().replace(" ", "-").replace("_", "-")
    suffix = requested.rsplit("-", 1)[-1] if "-" in requested else ""
    if suffix in {"none", "low", "medium", "high", "xhigh"}:
        return suffix
    model = (normalized_model or _normalize_model(requested_model)).lower()
    if "mini" in model:
        return "medium"
    if "codex" in model:
        return "high"
    return "medium"


def response_to_stream_chunks(response: Any) -> Any:
    """Yield one LiteLLM-like streaming chunk for callers expecting a stream."""
    import types

    message = response.choices[0].message if getattr(response, "choices", None) else None
    if message is None:
        return
    delta_tool_calls = []
    for idx, tc in enumerate(getattr(message, "tool_calls", None) or []):
        delta_tool_calls.append(types.SimpleNamespace(
            index=idx,
            id=getattr(tc, "id", None),
            function=types.SimpleNamespace(
                name=getattr(getattr(tc, "function", None), "name", None),
                arguments=getattr(getattr(tc, "function", None), "arguments", None),
            ),
        ))
    delta = types.SimpleNamespace(
        content=getattr(message, "content", None),
        tool_calls=delta_tool_calls or None,
    )
    yield types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)])


def _messages_to_responses_input(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    instructions: list[str] = []
    input_items: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            if content:
                instructions.append(_content_to_text(content))
            continue
        if role == "tool":
            input_items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id") or msg.get("id") or str(uuid.uuid4()),
                "output": _content_to_text(content),
            })
            continue
        if role == "assistant" and msg.get("tool_calls"):
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", None)
                name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", "")
                args = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", "{}")
                input_items.append({
                    "type": "function_call",
                    "call_id": tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", str(uuid.uuid4())),
                    "name": name,
                    "arguments": args or "{}",
                })
            if content:
                input_items.append({"role": "assistant", "content": _content_to_text(content)})
            continue
        if role in ("user", "assistant", "developer"):
            input_items.append({"role": "user" if role == "developer" else role, "content": _content_to_text(content)})
    return "\n\n".join(instructions), input_items


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in ("text", "input_text"):
                    parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def _tools_to_responses(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        fn = tool.get("function") or {}
        out.append({
            "type": "function",
            "name": fn.get("name"),
            "description": fn.get("description") or "",
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    return [t for t in out if t.get("name")]


def _tool_choice_to_responses(tool_choice: str | dict[str, Any]) -> Any:
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") or {}
        name = fn.get("name")
        return {"type": "function", "name": name} if name else "auto"
    return "required" if tool_choice == "required" else "auto"


def _iter_sse(resp: httpx.Response) -> Any:
    event: dict[str, Any] = {}
    data_lines: list[str] = []
    for raw in resp.iter_lines():
        line = raw.strip()
        if not line:
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines = []
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    pass
            event = {}
            continue
        if line.startswith("data:"):
            data = line[5:].strip()
            if data != "[DONE]":
                data_lines.append(data)
        elif line.startswith("event:"):
            event["event"] = line[6:].strip()
    if data_lines:
        try:
            yield json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            pass


def _response_from_events(events: list[dict[str, Any]], model: str) -> Any:
    from litellm import ModelResponse
    from litellm.utils import Choices, Message, ChatCompletionMessageToolCall, Function

    response_obj = None
    output_items: list[dict[str, Any]] = []
    text_by_item: dict[str, list[str]] = {}
    args_by_item: dict[str, list[str]] = {}
    reasoning_parts: list[str] = []

    for ev in events:
        typ = ev.get("type")
        if typ in ("response.done", "response.completed") and isinstance(ev.get("response"), dict):
            response_obj = ev["response"]
            for item in response_obj.get("output") or []:
                if isinstance(item, dict):
                    output_items.append(item)
            continue

        item = ev.get("item")
        if typ == "response.output_item.done" and isinstance(item, dict):
            output_items.append(item)
            continue
        if typ == "response.output_text.delta":
            item_id = str(ev.get("item_id") or ev.get("output_index") or "")
            text_by_item.setdefault(item_id, []).append(str(ev.get("delta") or ""))
            continue
        if typ == "response.output_text.done":
            item_id = str(ev.get("item_id") or ev.get("output_index") or "")
            text = ev.get("text")
            if text is not None:
                text_by_item[item_id] = [str(text)]
            continue
        if typ == "response.function_call_arguments.delta":
            item_id = str(ev.get("item_id") or ev.get("output_index") or "")
            args_by_item.setdefault(item_id, []).append(str(ev.get("delta") or ""))
            continue
        if typ == "response.function_call_arguments.done":
            item_id = str(ev.get("item_id") or ev.get("output_index") or "")
            args = ev.get("arguments")
            if args is not None:
                args_by_item[item_id] = [str(args)]
            continue
        if typ in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            reasoning_parts.append(str(ev.get("delta") or ""))

    if response_obj is None:
        response_obj = events[-1].get("response") if events else None

    content_parts: list[str] = []
    tool_calls: list[Any] = []
    seen_tool_ids: set[str] = set()

    def item_key(item: dict[str, Any], index: int) -> str:
        return str(item.get("id") or item.get("item_id") or item.get("output_index") or index)

    for index, item in enumerate(output_items):
        typ = item.get("type")
        key = item_key(item, index)
        if typ == "message":
            item_text_parts: list[str] = []
            for c in item.get("content") or []:
                if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                    item_text_parts.append(str(c.get("text") or ""))
            if item_text_parts:
                content_parts.append("".join(item_text_parts))
            elif key in text_by_item:
                content_parts.append("".join(text_by_item[key]))
        elif typ == "function_call":
            call_id = str(item.get("call_id") or item.get("id") or f"call_{len(tool_calls)}")
            if call_id in seen_tool_ids:
                continue
            seen_tool_ids.add(call_id)
            arguments = item.get("arguments")
            if not arguments:
                arguments = "".join(args_by_item.get(key, []))
            tool_calls.append(ChatCompletionMessageToolCall(
                id=call_id,
                type="function",
                function=Function(
                    name=str(item.get("name") or ""),
                    arguments=arguments or "{}",
                ),
            ))
        elif typ == "reasoning":
            summary = item.get("summary") or []
            for s in summary:
                if isinstance(s, dict) and s.get("text"):
                    reasoning_parts.append(str(s["text"]))

    if not content_parts and text_by_item:
        for parts in text_by_item.values():
            content_parts.append("".join(parts))

    message = Message(
        role="assistant",
        content="\n".join(p for p in content_parts if p) or None,
        tool_calls=tool_calls or None,
    )
    reasoning = "".join(reasoning_parts).strip()
    if reasoning:
        setattr(message, "reasoning_content", reasoning)
    response = ModelResponse(
        id=(response_obj or {}).get("id") if isinstance(response_obj, dict) else None,
        created=int(time.time()),
        model=model,
        choices=[Choices(index=0, finish_reason="tool_calls" if tool_calls else "stop", message=message)],
    )
    usage = (response_obj or {}).get("usage") if isinstance(response_obj, dict) else None
    if usage:
        response.usage = usage
    return response
