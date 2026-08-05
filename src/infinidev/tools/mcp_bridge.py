"""Every tool an MCP server advertises, exposed under its own name.

Infinidev is an MCP *host*, so the tools a server publishes should reach the
model the way the server named them — ``ken_rank``, ``ken_recall``,
``ken_callgraph`` — not behind hand-written wrappers that rename three of
them and hide the other twenty-seven. Renaming costs twice: the model cannot
follow the server's own documentation, and every tool nobody thought to wrap
is simply unreachable.

So nothing here knows what Ken is. The bridge reads ``tools/list``, turns
each entry into an ``InfinibayBaseTool`` subclass with the server's name,
description and JSON Schema, and that is the whole contract. Point the host
at a different MCP server and its tools show up too.

Three things the raw listing needs before a model can use it well:

**Descriptions get compressed.** Servers built on the Python SDK publish the
function's entire docstring — Ken's thirty tools cost ~6 100 tokens of
schema that way, on top of the ~14 500 the local toolset already spends.
The first paragraph carries the "what", the rest is prose the model does not
need at selection time, so only the first paragraph survives.

**Writers are separated from readers.** ``get_tools_for_role`` builds the
read-only tiers (chat agent, planner, critic, council) from ``is_read_only``,
and that filter is a security boundary, not a hint. The spec has a
``readOnlyHint`` annotation for exactly this and it is honoured when present;
when it is absent — Ken sends no annotations today — the name is matched
against a mutating-verb pattern and anything that matches is treated as a
writer. Unknown means "assume it writes".

**Names are namespaced by the server that owns them.** A local tool always
wins a collision: a remote server must never be able to shadow ``read_file``.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from infinidev.tools.base.base_tool import InfinibayBaseTool

logger = logging.getLogger(__name__)

# Descriptions longer than this are cut at the last sentence that fits.
# Roughly two lines of prose — enough to say what the tool does and when to
# reach for it, which is all the model needs while choosing one.
_MAX_DESCRIPTION = 300

# How much of a tool's output is handed back to the model. Beyond this the
# result is truncated with an explicit marker, because a 200 KB call graph
# silently eating the context window is worse than a short, honest answer.
_MAX_RESULT_CHARS = 12_000

# Verbs that mean a tool changes something. Used only when the server sends
# no ``readOnlyHint``; the default on a match is "this writes", so a new
# mutating tool is kept out of the read-only tiers by default rather than
# leaking into them.
_MUTATING_VERB_RE = re.compile(
    r"(^|_)("
    r"remember|forget|dismiss|record|save|store|persist|"
    r"write|create|make|new|add|insert|append|"
    r"update|modify|edit|patch|set|replace|rename|move|"
    r"delete|remove|drop|clear|reset|purge|prune|"
    r"run|exec|execute|invoke|call|spawn|start|stop|kill|restart|"
    r"install|deploy|publish|push|send|post|commit|apply|import|sync"
    r")(_|$)",
    re.IGNORECASE,
)

_JSON_TO_PYTHON: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def compress_description(text: str) -> str:
    """Reduce a server's docstring to the part that aids tool selection.

    Keeps the first paragraph — the "what" — and drops the parameter walk
    through that follows, which the JSON Schema already carries. If that
    paragraph is still long, it is cut at the last sentence boundary that
    fits so the result never ends mid-clause.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    paragraph = re.split(r"\n\s*\n", cleaned, maxsplit=1)[0]
    paragraph = " ".join(paragraph.split())
    paragraph = re.sub(r"\brelevant\b", "matching", paragraph, flags=re.IGNORECASE)
    paragraph = paragraph.replace("->", "to")
    if len(paragraph) <= _MAX_DESCRIPTION:
        return paragraph
    head = paragraph[:_MAX_DESCRIPTION]
    cut = head.rfind(". ")
    if cut > _MAX_DESCRIPTION // 2:
        return head[: cut + 1]
    return head.rstrip() + "…"


def is_read_only(tool: Any) -> bool:
    """Whether *tool* may be handed to a read-only tier.

    The spec's ``readOnlyHint`` wins when the server sends one. Otherwise
    the name decides, and a mutating verb means "no" — an unrecognised tool
    is only admitted because nothing about it suggests it writes.
    """
    annotations = getattr(tool, "annotations", None) or {}
    hint = annotations.get("readOnlyHint")
    if isinstance(hint, bool):
        return hint
    if annotations.get("destructiveHint") is True:
        return False
    return _MUTATING_VERB_RE.search(tool.name) is None


def _python_type(spec: dict[str, Any]) -> Any:
    """Best-effort JSON Schema → Python type for one property."""
    if not isinstance(spec, dict):
        return Any

    # ``anyOf``/``oneOf`` is how the Python SDK spells "optional". Take the
    # first non-null branch; the schema sanitiser flattens these the same way
    # before they reach the provider.
    for key in ("anyOf", "oneOf"):
        variants = spec.get(key)
        if isinstance(variants, list):
            for variant in variants:
                if isinstance(variant, dict) and variant.get("type") != "null":
                    return _python_type(variant)

    # A closed set of values must stay closed all the way to the provider.
    # Rendering it only into the description leaves the argument a free
    # string, so a strict-mode provider will happily pass through
    # ``scope="everything"`` and the server has to reject it a round trip
    # later. As Literal it becomes a JSON Schema ``enum``, which structured
    # decoding can enforce at generation time — the model cannot emit an
    # invalid value at all.
    choices = spec.get("enum")
    if isinstance(choices, list) and choices and all(
        isinstance(choice, (str, int, bool)) for choice in choices
    ):
        return Literal[tuple(choices)]  # type: ignore[return-value]

    declared = spec.get("type")
    if isinstance(declared, list):  # ["string", "null"]
        declared = next((t for t in declared if t != "null"), None)
    if declared == "array":
        item_type = _python_type(spec.get("items") or {})
        return list[item_type] if item_type is not Any else list
    return _JSON_TO_PYTHON.get(declared, Any) if declared else Any


def _field_description(name: str, spec: dict[str, Any]) -> str:
    """Description for one argument, with the enum spelled out if there is one."""
    parts: list[str] = []
    described = spec.get("description") or spec.get("title")
    if described and str(described).strip().lower() != name.replace("_", " ").lower():
        parts.append(" ".join(str(described).split()))
    choices = spec.get("enum")
    if isinstance(choices, list) and choices:
        parts.append("One of: " + ", ".join(repr(c) for c in choices) + ".")
    return " ".join(parts)


def build_args_model(tool: Any) -> type[BaseModel]:
    """Turn a server's ``inputSchema`` into a pydantic model.

    The model is what validates the model's arguments *and* what
    ``tool_to_openai_schema`` re-serialises for the provider, so a property
    missing here is a property the LLM can never send.
    """
    schema = getattr(tool, "input_schema", None) or {}
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])

    fields: dict[str, Any] = {}
    for name, spec in properties.items():
        if not isinstance(spec, dict):
            spec = {}
        annotation = _python_type(spec)
        description = _field_description(name, spec)
        if name in required:
            default: Any = ...
        elif "default" in spec:
            default = spec["default"]
        else:
            # Optional with no server-provided default: None, and the type
            # widens to accept it so validation does not reject the omission.
            default = None
            annotation = Any if annotation is Any else (annotation | None)
        fields[name] = (annotation, Field(default, description=description or None))

    model_name = "".join(part.title() for part in tool.name.split("_")) + "Args"
    # ``extra="forbid"`` is the point of building this model at all. The
    # engine calls ``tool._run(**args)`` directly (tool_dispatch.py), which
    # bypasses ``BaseTool.run`` and therefore pydantic; and the fallback
    # check there inspects ``_run``'s signature, which for a bridged tool is
    # ``**kwargs`` and so disables itself. Without this, a hallucinated
    # argument is forwarded to the server verbatim — and CLAUDE.md promises
    # the opposite.
    config = ConfigDict(extra="forbid")
    if not fields:
        return create_model(model_name, __config__=config)
    return create_model(model_name, __config__=config, **fields)


def _first_validation_problem(exc: ValidationError) -> str:
    """One readable sentence from a pydantic error, aimed at the model.

    Only the first problem is reported: a model that fixes one argument per
    turn converges, and a wall of nested error dicts is the kind of tool
    output that teaches nothing.
    """
    errors = exc.errors()
    if not errors:
        return "invalid arguments"
    first = errors[0]
    field = ".".join(str(p) for p in first.get("loc", ())) or "argument"
    if first.get("type") == "extra_forbidden":
        return f"no such argument {field!r}"
    if first.get("type") == "missing":
        return f"missing required argument {field!r}"
    return f"argument {field!r}: {first.get('msg', 'is invalid')}"


def render_result(result: Any) -> str:
    """Flatten an ``McpToolResult`` into the text the model reads."""
    text = (getattr(result, "text", "") or "").strip()
    if not text:
        data = getattr(result, "data", None)
        if data is not None:
            try:
                text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                text = str(data)
    if not text:
        return "(the server returned no content)"
    if len(text) > _MAX_RESULT_CHARS:
        dropped = len(text) - _MAX_RESULT_CHARS
        text = (
            text[:_MAX_RESULT_CHARS]
            + f"\n\n… truncated, {dropped:,} more characters. "
            "Narrow the query or lower the limit to see the rest."
        )
    return text


def build_tool_class(tool: Any) -> type[InfinibayBaseTool]:
    """Create the ``InfinibayBaseTool`` subclass that fronts one remote tool."""
    server = tool.server
    remote_name = tool.name
    args_model = build_args_model(tool)
    description = compress_description(tool.description) or (
        f"Tool {remote_name!r} provided by the {server!r} MCP server."
    )

    def _run(self, **kwargs: Any) -> str:
        from infinidev.engine.mcp_client import (
            McpUnavailable,
            get_default_mcp_manager,
        )

        # Validate here, not upstream: the engine reaches ``_run`` directly
        # and the signature-based check it does instead cannot see through
        # ``**kwargs``. A rejected argument is returned as an error the model
        # can learn from rather than forwarded to the server.
        try:
            validated = args_model.model_validate(kwargs)
        except ValidationError as exc:
            accepted = ", ".join(args_model.model_fields) or "(none)"
            return self._error(
                f"{remote_name}: {_first_validation_problem(exc)}. "
                f"Accepted arguments: {accepted}."
            )

        # Optional arguments the model omitted arrive as None. Sending them
        # would override the server's own defaults with null, so they are
        # dropped rather than forwarded.
        arguments = {k: v for k, v in validated.model_dump().items() if v is not None}
        try:
            result = get_default_mcp_manager().call(server, remote_name, arguments)
        except McpUnavailable as exc:
            return self._error(
                f"{server} is not reachable, so {remote_name} cannot run: {exc}"
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("MCP %s.%s raised", server, remote_name, exc_info=True)
            return self._error(f"{remote_name} failed: {exc}")

        rendered = render_result(result)
        if getattr(result, "is_error", False):
            return self._error(f"{remote_name}: {rendered}")
        self._log_tool_usage(f"{server}.{remote_name}")
        return rendered

    namespace: dict[str, Any] = {
        # Pydantic refuses to let a subclass override an inherited field
        # without re-declaring its type, so the annotations come along.
        # ``mcp_server`` is a ClassVar on purpose: it is provenance for the
        # host, not an argument the model may pass.
        "__annotations__": {
            "name": str,
            "description": str,
            "args_schema": type[BaseModel],
            "is_read_only": bool,
            "mcp_server": ClassVar[str],
        },
        "__doc__": tool.description or description,
        "__module__": __name__,
        "name": remote_name,
        "description": description,
        "args_schema": args_model,
        "is_read_only": is_read_only(tool),
        "mcp_server": server,
        "_run": _run,
    }
    class_name = "".join(part.title() for part in remote_name.split("_")) + "McpTool"
    return type(class_name, (InfinibayBaseTool,), namespace)


# ── discovery ────────────────────────────────────────────────────────────
#
# Discovery talks to a subprocess, so it is cached and never on a hot path.
# ``get_tools_for_role`` runs once per turn and must not block on a server
# that is slow to start, which is why the cache is only ever filled from a
# listing that is already in hand: a cold manager returns nothing and warms
# itself in the background for the next turn.

_cache: list[type[InfinibayBaseTool]] | None = None
_cache_lock = threading.Lock()


def _selected(names: list[str]) -> list[str]:
    """Apply the ``MCP_TOOL_FILTER`` glob list to discovered tool names."""
    from fnmatch import fnmatch

    from infinidev.config.settings import settings

    raw = str(getattr(settings, "MCP_TOOL_FILTER", "*") or "*").strip()
    if not raw or raw == "*":
        return names
    patterns = [p.strip() for p in raw.split(",") if p.strip()]
    return [n for n in names if any(fnmatch(n, p) for p in patterns)]


def discover_mcp_tool_classes(
    *, force: bool = False, block: bool = False
) -> list[type[InfinibayBaseTool]]:
    """Tool classes for every MCP tool currently reachable.

    Returns ``[]`` — never raises — when MCP is disabled or no server has
    answered yet. ``block=True`` waits for the servers to hand over their
    listing; the default does not, so the first turn of a session is not
    held up by a subprocess spawn.
    """
    global _cache

    from infinidev.config.settings import settings

    if not getattr(settings, "MCP_ENABLED", True):
        return []
    if _cache is not None and not force:
        return _cache

    with _cache_lock:
        if _cache is not None and not force:
            return _cache
        try:
            from infinidev.engine.mcp_client import get_default_mcp_manager

            manager = get_default_mcp_manager()
            if block:
                manager.warmup()
                tools = manager.list_tools()
            else:
                # Read only what an already-running server can answer from
                # its cache. Spawning from here would put a subprocess
                # launch behind what reads like a list lookup — it is
                # called once per turn, and from every test that asks which
                # tools a role gets. Startup owns the warmup (see
                # ``ui/app.py::_warm_up_mcp``); a cold session simply gets
                # no MCP tools this turn and all of them the next.
                ready = [
                    name
                    for name in manager.all_names()
                    if (client := manager.get(name)) is not None and client.running
                ]
                if not ready:
                    return []
                tools = [t for name in ready for t in manager.list_tools(name)]
        except Exception:
            logger.debug("MCP tool discovery failed", exc_info=True)
            return []

        if not tools:
            return []

        keep = set(_selected([t.name for t in tools]))
        classes: list[type[InfinibayBaseTool]] = []
        seen: set[str] = set()
        for tool in tools:
            if tool.name not in keep or tool.name in seen:
                continue
            seen.add(tool.name)
            try:
                classes.append(build_tool_class(tool))
            except Exception:
                logger.warning(
                    "Could not expose MCP tool %s.%s", tool.server, tool.name,
                    exc_info=True,
                )
        _cache = classes
        _register_writers(classes)
        logger.info(
            "Exposed %d MCP tool(s): %s",
            len(classes),
            ", ".join(sorted(c.model_fields["name"].default for c in classes)),
        )
        return classes


def _register_writers(classes: list[type[InfinibayBaseTool]]) -> None:
    """Tell the executor which discovered tools must not run in parallel.

    ``batch_tool_calls`` decides serial-vs-parallel from a *name* set, since
    all it receives is the call, not the tool. A discovered writer that is
    missing from that set gets batched in with the reads — so ``ken_remember``
    would run concurrently with lookups, against a server that is mutating
    the same index those lookups are reading. The names are only knowable
    after discovery, which is why they are registered rather than listed.
    """
    try:
        from infinidev.engine.tool_executor import WRITE_TOOLS

        WRITE_TOOLS.update(
            cls.model_fields["name"].default
            for cls in classes
            if not cls.model_fields["is_read_only"].default
        )
    except Exception:  # pragma: no cover - defensive
        logger.debug("could not register MCP writers", exc_info=True)


def reset_discovery_cache() -> None:
    """Forget the discovered tools so the next call re-reads ``tools/list``."""
    global _cache
    with _cache_lock:
        _cache = None
