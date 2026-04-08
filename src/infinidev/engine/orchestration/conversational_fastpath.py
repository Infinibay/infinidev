"""Pre-planning preamble: the agent ALWAYS speaks before working.

Every user turn enters the pipeline through this module. A single
small LLM call ("the preamble") does TWO things in one shot:

  1. Produce a short user-facing reply (1-3 sentences). The reply
     is shown immediately so the user always knows the agent saw
     them — no matter whether the message is a greeting, a task,
     or a context-aware follow-up.

  2. Decide whether more work is needed. If the message is purely
     conversational ("Hola", "thanks for the fix"), the preamble
     short-circuits the pipeline with ``flow="done"``. Otherwise
     the pipeline continues into the analyst → gather → execute.

This is the same architecture a human collaborator uses: hear the
request, say "OK, let me check that", then start working. The user
never waits in silence — and the response itself doubles as the
classification.

Design rules:

1. **Always speaks first**. The preamble runs on EVERY non-empty
   input that hasn't been bypassed by ``skip_analysis``. There is
   no fast/slow path — there's just one path that responds, then
   decides.

2. **Tight prompt budget**. The system prompt is ~150 tokens —
   an order of magnitude smaller than the analyst. With a small
   non-thinking model the call lands in 0.5-2 s.

3. **Context-aware**. Recent session summaries are inlined so
   "thanks for the fix" gets a contextual reply ("you're welcome,
   glad the auth fix worked") instead of a generic one.

4. **Falls open on error**. Any exception in the preamble call
   makes the pipeline fall through to the analyst, so the user
   still gets a reply via the slower path.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from infinidev.engine.analysis.analysis_result import AnalysisResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────


_PREAMBLE_SYSTEM_PROMPT = """You are the first responder of a coding assistant.

Every user message hits you BEFORE the heavy planning machinery runs. \
You have exactly ONE tool available: ``respond``. You MUST call it \
exactly once per turn. Do not write prose, do not think out loud, do \
not ask for other tools — just call ``respond`` immediately.

When you call ``respond`` you give two arguments:

  - ``reply``: a short user-facing message (1-3 sentences max) that
    directly addresses what the user said, in their language
    (Spanish or English).

      * For greetings / thanks / smalltalk: reply conversationally.
      * For real work requests: say in one sentence what you're
        about to start doing. The planning pipeline behind you will
        handle the actual file reads, edits, and tool calls — you
        are just announcing the work.
      * When recent session context is provided, USE it. For
        "gracias" reference what you actually did; for "did that
        work?" reference the change.

  - ``continue_planning``: a boolean.
      * ``true`` if real work is needed (file inspection, edits,
        tests, refactor, code question that needs the codebase).
      * ``false`` for pure conversation (greetings, thanks,
        goodbyes, smalltalk, simple questions about you).

Be DECISIVE. Call ``respond`` on your very first action."""


# The single tool the preamble model is allowed to call. Forcing
# function-calling rather than free-form JSON parsing eliminates
# most format-drift errors small models make on structured output.
_RESPOND_TOOL = {
    "type": "function",
    "function": {
        "name": "respond",
        "description": (
            "Send a short reply to the user AND decide whether the "
            "planning pipeline should continue. This is the only "
            "tool you may call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reply": {
                    "type": "string",
                    "description": (
                        "Your user-facing message, 1-3 sentences max. "
                        "Match the user's language (Spanish or English)."
                    ),
                },
                "continue_planning": {
                    "type": "boolean",
                    "description": (
                        "true if real work is needed (file edits, "
                        "tests, refactor, code questions); false for "
                        "pure conversation (greetings, thanks, etc.)."
                    ),
                },
            },
            "required": ["reply", "continue_planning"],
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────


_MAX_PREAMBLE_LEN = 1500
"""Inputs longer than this skip the preamble entirely and go straight
to the analyst. Below this threshold the LLM call is worth the cost;
above it the message is obviously a real task spec and the preamble
adds latency without changing the outcome."""


def try_conversational_fastpath(
    user_input: str,
    session_summaries: Optional[list[str]] = None,
) -> Optional[tuple[AnalysisResult, str, bool]]:
    """Run the preamble call and return its decision.

    Returns:
        ``None`` to skip the preamble (input empty / too long /
        LLM call failed). The pipeline should continue to the
        normal analyst.

        ``(analysis_result, reply, continue_planning)`` otherwise.
        The caller should:
          - Always notify the user with ``reply``.
          - If ``continue_planning=False``, return immediately with
            ``flow="done"`` using the synthesised analysis_result.
          - If ``continue_planning=True``, fall through to the
            analyst with the reply already shown to the user.
    """
    if not user_input or not user_input.strip():
        return None

    if len(user_input) > _MAX_PREAMBLE_LEN:
        return None

    try:
        return _preamble_via_llm(user_input, session_summaries)
    except Exception as exc:
        logger.debug("Preamble call failed (falling through): %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────────────────────────────────


def _preamble_via_llm(
    user_input: str,
    session_summaries: Optional[list[str]],
) -> Optional[tuple[AnalysisResult, str, bool]]:
    """Run the preamble LLM call and parse its response."""
    import litellm

    from infinidev.config.llm import get_litellm_params_for_behavior

    params = get_litellm_params_for_behavior()

    user_lines: list[str] = []
    if session_summaries:
        recent = [s for s in session_summaries[-3:] if s]
        if recent:
            user_lines.append("RECENT SESSION CONTEXT:")
            for s in recent:
                user_lines.append(f"  - {s}")
            user_lines.append("")
    user_lines.append("USER MESSAGE:")
    user_lines.append(user_input.strip())

    messages = [
        {"role": "system", "content": _PREAMBLE_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_lines)},
    ]

    # Force the model into the single ``respond`` tool. With
    # function-calling enabled and ``tool_choice`` pinned to that
    # one tool, the model can't drift into prose, can't think out
    # loud (most providers serialise reasoning into the tool call
    # arguments directly), and produces a guaranteed-shape result
    # without any free-form JSON parsing.
    call_kwargs = dict(params)
    call_kwargs["messages"] = messages
    call_kwargs["tools"] = [_RESPOND_TOOL]
    call_kwargs["tool_choice"] = {
        "type": "function",
        "function": {"name": "respond"},
    }
    call_kwargs["max_tokens"] = 1500
    call_kwargs["temperature"] = 0.3
    call_kwargs.setdefault("stream", False)
    # Hint thinking models to skip chain-of-thought; ignored when
    # the chat template doesn't honour the kwarg.
    extra_body = dict(call_kwargs.get("extra_body") or {})
    chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
    chat_template_kwargs.setdefault("enable_thinking", False)
    chat_template_kwargs.setdefault("thinking", False)
    extra_body["chat_template_kwargs"] = chat_template_kwargs
    call_kwargs["extra_body"] = extra_body

    response = litellm.completion(**call_kwargs)
    parsed = _extract_tool_args(response)
    if parsed is None:
        # Tool call missing — fall back to scanning the message
        # content / reasoning for an inline JSON object. Some local
        # models ignore tool_choice and emit content instead.
        raw = _extract_text(response)
        if not raw:
            return None
        parsed = _parse_preamble_json(raw)
        if parsed is None:
            return None

    reply = (parsed.get("reply") or "").strip()
    continue_flag = bool(parsed.get("continue_planning", parsed.get("continue", True)))
    if not reply:
        return None

    flow = "develop" if continue_flag else "done"
    result = AnalysisResult(
        action="passthrough",
        original_input=user_input,
        reason=reply,
        flow=flow,
    )
    return result, reply, continue_flag


# ─────────────────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────────────────


def _extract_tool_args(response) -> Optional[dict]:
    """Pull the ``respond`` tool arguments out of a tool-call response.

    Returns the parsed argument dict or None when the model didn't
    actually call a tool. We accept any tool name (not strict
    matching on ``respond``) so that minor disagreements between
    provider serialisations don't break the parser — the parameters
    are what we care about.
    """
    try:
        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            return None
        tc = tool_calls[0]
        fn = getattr(tc, "function", None)
        if fn is None:
            return None
        raw_args = getattr(fn, "arguments", None) or "{}"
        if isinstance(raw_args, dict):
            return raw_args
        try:
            obj = json.loads(raw_args)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    except Exception:
        pass
    return None


def _extract_text(response) -> str:
    """Pull the assistant text out of a litellm completion response.

    Falls back to ``reasoning_content`` when ``content`` is empty —
    thinking models that hit the token limit during their internal
    chain-of-thought leave ``content`` empty but still emit the JSON
    decision somewhere in their reasoning trace. Scanning the
    reasoning lets the parser still find it.
    """
    try:
        msg = response.choices[0].message
        content = (getattr(msg, "content", None) or "").strip()
        if content:
            return content
        reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
        return reasoning
    except Exception:
        return ""


# Loose JSON extractor: finds the FIRST {...} blob in the text and
# parses it. Tolerant to leading/trailing prose, markdown fences,
# stray commentary — small models sometimes ignore the "JSON only"
# instruction in the prompt.
_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_preamble_json(raw: str) -> Optional[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    for match in _JSON_RE.finditer(raw):
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


__all__ = ["try_conversational_fastpath"]
