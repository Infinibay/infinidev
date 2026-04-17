"""Chat agent orchestrator — the default entry point of every user turn.

Replaces the legacy conversational fastpath (deleted in Commit 7).
Runs a short, read-only LLM loop that terminates when the model calls
either ``respond`` (conversational reply, turn ends) or ``escalate``
(hand off to the planner).

This is a **self-contained loop**, not a LoopEngine invocation:

  * The LoopEngine is built for plan-execute-summarize with steps,
    notes, guards, behavior tracking, and summarizer — all of which
    are overkill for a 5-iteration conversational turn.
  * The preamble (which we're replacing) used a similar purpose-built
    shape: single litellm call + direct tool_calls parsing. We keep
    that simplicity but add multi-iteration tool dispatch.

Every turn is fresh: we read ``get_recent_turns_full`` from the DB on
entry and accumulate no state across calls. The DB is the single
source of truth for conversation history.
"""

from __future__ import annotations

import json
import logging
import re
import traceback
import types
import uuid
from typing import Any, Optional

from infinidev.config.llm import get_litellm_params_for_behavior
from infinidev.engine.loop.llm_caller import ThinkStreamFilter, strip_think_blocks
from infinidev.engine.loop.schema_sanitizer import tool_to_openai_schema
from infinidev.engine.loop.tools import build_tool_dispatch, execute_tool_call
from infinidev.engine.orchestration.chat_agent_result import ChatAgentResult
from infinidev.engine.orchestration.escalation_packet import EscalationPacket
from infinidev.prompts.chat_agent import build_chat_agent_system_prompt
from infinidev.tools import get_tools_for_role
from infinidev.tools.base.context import (
    bind_tools_to_agent,
    clear_agent_context,
    set_context,
)

logger = logging.getLogger(__name__)


# Generous cap — the chat agent is encouraged to use its read-only
# tools freely, so we don't punish "needed 8 reads to answer a deep
# question" with a rephrase-yourself fallback. The cap is a runaway
# guard, not a quality gate. Two iterations before the cap we inject
# a nudge telling the model to wrap up; see _run_llm_loop.
_DEFAULT_MAX_ITERATIONS = 20
_MAX_RESULT_CHARS = 8000  # trim overly long tool outputs before re-prompting


def run_chat_agent(
    user_input: str,
    *,
    session_id: Optional[str] = None,
    project_id: Optional[int] = None,
    workspace_path: Optional[str] = None,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    hooks: Any | None = None,
) -> ChatAgentResult:
    """Run one turn of the chat agent and return its result.

    The caller (pipeline.run_task) surfaces the result to the user:
    ``kind="respond"`` → show ``reply`` and end the turn;
    ``kind="escalate"`` → continue to the analyst planner with
    ``escalation`` as the handoff packet.

    When ``hooks`` is provided, the LLM call runs in streaming mode and
    the ``respond`` tool's ``message`` field is emitted chunk-by-chunk
    via ``hooks.notify_stream_chunk`` as the JSON arguments form. The
    returned ``ChatAgentResult.streamed`` is True so the caller knows
    the UI already received the text and should not re-notify.

    When the loop can't produce a decision (LLM error, no tool call,
    max-iter exhaustion), a fallback ``respond`` is returned rather
    than raising — the UI must always get something to show the user.
    """
    if not user_input or not user_input.strip():
        return ChatAgentResult(
            kind="respond",
            reply="(empty message)",
        )

    # Ephemeral agent_id isolates this turn's tool-context binding from the
    # developer agent's context. set_context writes into a process-global
    # dict keyed by agent_id, so the chat agent, the planner, and the
    # developer each own independent slots that do not stomp each other.
    # clear_agent_context in `finally` ensures no leak across turns.
    agent_id = f"chat-agent-{uuid.uuid4().hex[:8]}"
    tools = get_tools_for_role("chat_agent")
    bind_tools_to_agent(tools, agent_id)
    set_context(
        agent_id=agent_id,
        project_id=project_id,
        session_id=session_id,
        workspace_path=workspace_path,
    )

    dispatch = build_tool_dispatch(tools)
    tool_schemas = [tool_to_openai_schema(t) for t in tools]

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_chat_agent_system_prompt()},
        {"role": "user", "content": _build_user_message(user_input, session_id)},
    ]

    try:
        return _run_llm_loop(
            messages=messages,
            tool_schemas=tool_schemas,
            dispatch=dispatch,
            user_input=user_input,
            max_iterations=max_iterations,
            hooks=hooks,
        )
    except Exception as exc:
        logger.exception("Chat agent loop failed")
        # If the exception interrupted a stream-in-progress, finalize
        # the partial message so the UI flips it out of streaming mode
        # and re-renders with whatever text was captured. Without this,
        # the TUI would carry a phantom message stuck in streaming=True.
        if hooks is not None:
            try:
                hooks.notify_stream_end("Infinidev", "agent")
            except Exception:
                pass
        return _fallback_respond(_detect_lang(user_input), exc=exc)
    finally:
        clear_agent_context(agent_id)


# ─────────────────────────────────────────────────────────────────────────
# Loop driver
# ─────────────────────────────────────────────────────────────────────────


def _run_llm_loop(
    *,
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    dispatch: dict[str, Any],
    user_input: str,
    max_iterations: int,
    hooks: Any | None = None,
) -> ChatAgentResult:
    import litellm

    base_kwargs = get_litellm_params_for_behavior()
    stream_mode = hooks is not None
    budget_nudged = False

    for iteration in range(max_iterations):
        # Near the end of the budget, nudge the model to wrap up.
        # We don't want to mid-response ambush the LLM, so this fires
        # exactly once on the second-to-last iteration, giving it two
        # chances to produce a terminator.
        if (
            not budget_nudged
            and max_iterations >= 3
            and iteration == max_iterations - 2
        ):
            messages.append({
                "role": "user",
                "content": (
                    "You're approaching your iteration budget. On your "
                    "next call, use `respond` to share what you've found "
                    "so far, or `escalate` if the user clearly asked for "
                    "implementation. Don't start a new investigation — "
                    "summarise and end the turn."
                ),
            })
            budget_nudged = True

        call_kwargs = dict(base_kwargs)
        call_kwargs["messages"] = messages
        call_kwargs["tools"] = tool_schemas
        call_kwargs.setdefault("temperature", 0.1)
        call_kwargs["stream"] = stream_mode
        call_kwargs.setdefault("max_tokens", 2000)

        response = litellm.completion(**call_kwargs)

        if stream_mode:
            content, tool_calls, streamed = _consume_stream(response, hooks)
            # Non-stream responses are normalised globally by the
            # litellm.completion wrapper (see config/llm.py). Streams
            # are assembled locally, so we strip <think> blocks here.
            content = strip_think_blocks(content)
        else:
            message = response.choices[0].message
            content = getattr(message, "content", None) or ""
            tool_calls = getattr(message, "tool_calls", None) or []
            streamed = False

        if not tool_calls:
            # The model chatted in plain text instead of calling a tool.
            # Treat it as a respond — we still terminate cleanly and the
            # user sees the text.
            return ChatAgentResult(
                kind="respond",
                reply=content.strip() or "(no reply)",
                streamed=streamed,
            )

        # Add the assistant turn to the transcript so tool results can
        # reference the tool_use IDs.
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": [_tool_call_to_dict(tc) for tc in tool_calls],
        })

        # Check whether ANY call is a terminator — if yes, we're done.
        for tc in tool_calls:
            name = tc.function.name
            if name == "respond":
                return _build_respond(tc, user_input, streamed=streamed)
            if name == "escalate":
                return _build_escalate(tc, user_input)

        # No terminator — execute read-only tools and continue.
        for tc in tool_calls:
            result = execute_tool_call(
                dispatch, tc.function.name, tc.function.arguments,
            )
            trimmed = result if len(result) <= _MAX_RESULT_CHARS else (
                result[:_MAX_RESULT_CHARS] + "\n...[truncated]"
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": trimmed,
            })

    # Max iterations reached without terminator. Return a graceful
    # respond so the user sees SOMETHING — never a silent return.
    logger.warning("Chat agent hit max_iterations=%d without terminator", max_iterations)
    return _fallback_respond(_detect_lang(user_input), reason="max_iter")


# ─────────────────────────────────────────────────────────────────────────
# Terminator parsing
# ─────────────────────────────────────────────────────────────────────────


def _build_respond(
    tc: Any, user_input: str, *, streamed: bool = False,
) -> ChatAgentResult:
    args = _parse_args(tc)
    message = (args.get("message") or "").strip()
    if not message:
        return _fallback_respond(_detect_lang(user_input), reason="empty_respond")
    return ChatAgentResult(kind="respond", reply=message, streamed=streamed)


def _build_escalate(tc: Any, user_input: str) -> ChatAgentResult:
    args = _parse_args(tc)
    understanding = (args.get("understanding") or "").strip()
    if not understanding:
        # Defensive: escalate with empty understanding is a useless
        # handoff. Fall back to respond so the user isn't stranded.
        return _fallback_respond(_detect_lang(user_input), reason="empty_escalate")
    opened = args.get("opened_files") or []
    if not isinstance(opened, list):
        opened = []
    packet = EscalationPacket(
        user_request=user_input.strip(),
        understanding=understanding,
        opened_files=[str(p) for p in opened],
        user_visible_preview=(args.get("user_visible_preview") or "").strip(),
        user_signal=(args.get("user_signal") or "").strip(),
        suggested_flow="develop",  # v1 restriction
    )
    return ChatAgentResult(kind="escalate", escalation=packet)


def _parse_args(tc: Any) -> dict[str, Any]:
    raw = getattr(tc.function, "arguments", None) or "{}"
    if isinstance(raw, dict):
        return raw
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def _sanitize_tool_arguments(raw: Any) -> str:
    """Normalize tool-call ``arguments`` into a clean JSON string.

    Local models (gemma4, sometimes qwen) emit tool calls whose
    ``arguments`` are either non-strings, strings with trailing noise
    (e.g. the closing ``<tool_call|>`` marker leaked into the field),
    or double-encoded JSON. LiteLLM's Ollama chat transformer runs
    ``json.loads`` on this field when building the *next* request, so
    a malformed value crashes the whole loop one iteration later —
    with a misleading "Extra data" JSONDecodeError at serialization
    time, not at parse time.

    Strategy:
      1. dict/list → ``json.dumps`` directly.
      2. valid JSON string → parse and re-serialize to strip whitespace.
      3. string with extra junk after a valid JSON prefix →
         ``raw_decode`` to keep just the first object.
      4. anything else → ``"{}"`` so downstream code sees a well-formed
         empty args blob instead of crashing.
    """
    if isinstance(raw, (dict, list)):
        try:
            return json.dumps(raw)
        except (TypeError, ValueError):
            return "{}"
    if not isinstance(raw, str):
        return "{}"
    s = raw.strip()
    if not s:
        return "{}"
    try:
        return json.dumps(json.loads(s))
    except json.JSONDecodeError:
        try:
            parsed, _end = json.JSONDecoder().raw_decode(s)
            logger.warning(
                "Truncated malformed tool arguments (extra data after valid "
                "JSON prefix): %r → %r", s[:80], parsed,
            )
            return json.dumps(parsed)
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "Dropped unparseable tool arguments, falling back to '{}': %r",
                s[:80],
            )
            return "{}"


def _tool_call_to_dict(tc: Any) -> dict[str, Any]:
    """Serialize a tool_call object back to the dict shape LiteLLM
    accepts on the next call. Provider-specific message objects don't
    serialize automatically; arguments are sanitized to survive sloppy
    outputs from local models."""
    return {
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.function.name,
            "arguments": _sanitize_tool_arguments(tc.function.arguments),
        },
    }


# Fallback messages keyed by (language, reason). The chat agent's system
# prompt tells the model to match the user's language, but the fallback
# paths bypass the model entirely — so we have to localize ourselves or
# hardcode one language and surprise users in the other. A tiny heuristic
# is good enough here (greetings / action verbs / punctuation cover most
# short chats); the LLM still handles the normal path.
_FALLBACK_MESSAGES: dict[tuple[str, str], str] = {
    # max_iter fires only when even the end-of-budget nudge didn't
    # produce a terminator. Message stays neutral — it's our ceiling,
    # not the user's prompt, that ran out. We don't blame them.
    ("es", "max_iter"): (
        "Investigué bastante y no terminé de cerrar la respuesta. "
        "Si querés seguimos desde acá; contame qué querés que haga."
    ),
    ("en", "max_iter"): (
        "I investigated quite a bit but didn't wrap up a final answer. "
        "Happy to keep going from here — let me know what you want next."
    ),
    ("es", "empty_respond"): "No supe cómo contestarte. ¿Podrías reformular?",
    ("en", "empty_respond"): "I don't know how to answer that. Could you rephrase?",
    ("es", "empty_escalate"): (
        "Detecté que querías que haga algo, pero no me quedó clara la "
        "consigna. ¿Podrías decirme qué implementar?"
    ),
    ("en", "empty_escalate"): (
        "I detected that you wanted me to do something, but the request "
        "isn't clear. Could you tell me what to implement?"
    ),
    ("es", "exception"): "Tuve un problema procesando tu mensaje. ¿Podrías repetirlo?",
    ("en", "exception"): "I ran into a problem processing your message. Could you try again?",
}


def _detect_lang(text: str) -> str:
    """Cheap Spanish-vs-English heuristic for fallback messages only.

    Not used for the normal LLM path — the model handles that. Returns
    ``"es"`` when obvious Spanish markers appear, ``"en"`` otherwise.
    Default English because the codebase and prompts lean English.
    """
    if not text:
        return "en"
    lowered = text.lower()
    # Unicode markers that only appear in Spanish (ñ, ¿, ¡, accented vowels)
    if any(ch in lowered for ch in "ñáéíóúü¿¡"):
        return "es"
    # Common Spanish function words
    spanish_markers = (" que ", " por ", " para ", " con ", " los ", " las ",
                       " una ", " dale", " hacé", " decí", " implementá",
                       " arreglá", " agregá", " borrá", "hola", "gracias", "chau")
    padded = f" {lowered} "
    if any(m in padded for m in spanish_markers):
        return "es"
    return "en"


def _fallback_respond(
    lang: str, *, reason: str = "exception", exc: Exception | None = None,
) -> ChatAgentResult:
    """Build a respond result from a localized fallback message.

    When an exception is supplied the traceback is attached to the
    result so the UI can render it inside a collapsed widget — the
    short message still dominates the chat, but the user can expand
    it to see the real error without digging through log files.
    """
    if exc is not None:
        logger.warning("chat_agent fallback (reason=%s): %s", reason, exc)
    message = _FALLBACK_MESSAGES.get(
        (lang, reason),
        _FALLBACK_MESSAGES[("en", reason if (("en", reason) in _FALLBACK_MESSAGES) else "exception")],
    )
    tb_text: str | None = None
    if exc is not None:
        tb_text = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
    return ChatAgentResult(kind="respond", reply=message, error_traceback=tb_text)


# ─────────────────────────────────────────────────────────────────────────
# Session history
# ─────────────────────────────────────────────────────────────────────────


def _build_user_message(user_input: str, session_id: Optional[str]) -> str:
    """Combine the session-history snapshot and the current user input
    into a SINGLE ``role="user"`` message.

    Two consecutive ``role="user"`` turns trip some providers (Anthropic
    strictly alternates), so we merge: the snapshot is rendered first,
    then the actual request. The snapshot is optional — missing /
    empty session id / DB failure all degrade gracefully.
    """
    trimmed = user_input.strip()
    turns: list[tuple[str, str]] = []
    if session_id:
        try:
            from infinidev.db.service import get_recent_turns_full
            turns = get_recent_turns_full(
                session_id, limit=6, max_chars_per_turn=2000,
            )
        except Exception as exc:
            logger.warning(
                "chat_agent: session history fetch failed (continuing "
                "without snapshot): %s", exc,
            )
            turns = []
    if not turns:
        return trimmed
    lines = [
        "Recent conversation (for context; use tools to reground facts):",
    ]
    for role, content in turns:
        tag = "USER" if role == "user" else "AGENT"
        lines.append(f'<turn role="{tag}">')
        lines.append(content)
        lines.append("</turn>")
    lines.append("")
    lines.append("Current user message:")
    lines.append(trimmed)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# Streaming
# ─────────────────────────────────────────────────────────────────────────


# Matches ``"message"`` field in partial JSON tool_call args. Captures
# the raw content up to (but not including) the unescaped closing quote
# — we may be mid-character, which is fine since we emit diffs. ``\\.``
# handles ``\"`` / ``\n`` / ``\\`` escapes so they don't prematurely
# terminate the match.
_RESPOND_MESSAGE_RE = re.compile(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)')


def _extract_partial_message(args: str) -> str:
    """Return the ``message`` field's content extracted from partial
    JSON tool_call args. Returns "" if the field hasn't started yet."""
    m = _RESPOND_MESSAGE_RE.search(args)
    if not m:
        return ""
    raw = m.group(1)
    # Minimal unescape. Not a full JSON parser — good enough for the
    # common escapes a chat message contains. The final non-streaming
    # pass (`_build_respond` → `_parse_args`) uses real json.loads.
    return (
        raw.replace('\\"', '"')
           .replace('\\n', '\n')
           .replace('\\t', '\t')
           .replace('\\r', '\r')
           .replace('\\\\', '\\')
    )


def _consume_stream(stream: Any, hooks: Any) -> tuple[str, list[Any], bool]:
    """Consume a LiteLLM streaming response, emitting chunks of the
    ``respond`` tool's ``message`` field via ``hooks.notify_stream_chunk``
    as they form.

    Returns ``(content, tool_calls, streamed)`` in the shape the
    non-streaming path produces: ``content`` is the accumulated plain
    text; ``tool_calls`` is a list of objects exposing ``.id``,
    ``.function.name``, and ``.function.arguments`` (synthesised so
    downstream code — ``_build_respond``, ``_build_escalate``,
    ``_tool_call_to_dict``, tool dispatch — works unchanged).
    """
    accumulated: dict[int, dict[str, str]] = {}  # idx → {id, name, args}
    emitted_per_tc: dict[int, str] = {}  # idx → chars already emitted
    content_buffer = ""
    streamed = False
    # Suppress <think>...</think> blocks from reaching the TUI
    # mid-stream. The filter holds back partial open-tag fragments
    # until we know whether a block is starting, so the user never
    # sees a stray '<think>' flash on screen.
    content_filter = ThinkStreamFilter()

    for chunk in stream:
        try:
            delta = chunk.choices[0].delta
        except (AttributeError, IndexError):
            continue

        delta_content = getattr(delta, "content", None)
        if delta_content:
            content_buffer += delta_content
            safe_delta = content_filter.feed(delta_content)
            if safe_delta:
                # Stream plain-text content too. Most chat-agent turns
                # end in a tool call, but some models emit plain text
                # as a respond-equivalent; either way the user sees it
                # live — minus any think-block content.
                try:
                    hooks.notify_stream_chunk("Infinidev", safe_delta, "agent")
                    streamed = True
                except Exception as exc:
                    logger.warning("notify_stream_chunk failed (content): %s", exc)

        delta_tool_calls = getattr(delta, "tool_calls", None) or []
        for tc_delta in delta_tool_calls:
            idx = getattr(tc_delta, "index", 0) or 0
            slot = accumulated.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if getattr(tc_delta, "id", None):
                slot["id"] = tc_delta.id
            fn = getattr(tc_delta, "function", None)
            if fn is None:
                continue
            if getattr(fn, "name", None):
                slot["name"] = (slot["name"] or "") + fn.name
            fn_args = getattr(fn, "arguments", None)
            if fn_args:
                slot["arguments"] += fn_args
                # Emit incremental `message` chars only for the respond tool.
                if slot["name"] == "respond":
                    current = _extract_partial_message(slot["arguments"])
                    emitted = emitted_per_tc.get(idx, "")
                    if current.startswith(emitted) and len(current) > len(emitted):
                        new_chars = current[len(emitted):]
                        try:
                            hooks.notify_stream_chunk("Infinidev", new_chars, "agent")
                            streamed = True
                        except Exception as exc:
                            logger.warning(
                                "notify_stream_chunk failed (tool): %s", exc,
                            )
                        emitted_per_tc[idx] = current

    # Flush any held-back safe tail (e.g., partial open-tag fragments
    # that never resolved into a full <think> block).
    tail = content_filter.flush()
    if tail:
        try:
            hooks.notify_stream_chunk("Infinidev", tail, "agent")
            streamed = True
        except Exception as exc:
            logger.warning("notify_stream_chunk failed (flush): %s", exc)

    tool_calls: list[Any] = []
    for idx, slot in sorted(accumulated.items()):
        if not slot["name"]:
            continue  # skip half-formed entries
        tool_calls.append(types.SimpleNamespace(
            id=slot["id"] or f"stream-tc-{idx}",
            function=types.SimpleNamespace(
                name=slot["name"],
                arguments=slot["arguments"],
            ),
        ))

    return content_buffer, tool_calls, streamed


__all__ = ["run_chat_agent"]
