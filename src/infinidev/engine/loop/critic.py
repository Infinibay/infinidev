"""Assistant LLM — pair-programming critic.

A second LLM that runs on a separate Ollama instance (typically pinned
to a different GPU) and reviews the principal's tool calls in parallel
with their execution. The verdict is purely informative: it never
blocks, never forces retries, never vetoes. The principal sees the
critic's message in the next iteration and decides what to do.

Design notes
------------
* The critic shares the principal's full message history so it can
  reason about <task>, <plan>, <previous-actions>, etc. — these are
  already embedded in the messages by the loop's prompt builder.
* The critic ALSO receives the principal's current-turn reasoning
  (``reasoning_content``) when available. The principal's reasoning
  is normally stripped from message history (see
  ``ContextManager.expire_thinking``), so the engine threads it in
  out-of-band as a transient block in the critic's user prompt. This
  lets the critic intervene on the *thinking*, not just the action —
  e.g. spot a wrong assumption before it crystallises into a bad
  edit. The reasoning is never written back to the principal's
  history; it dies with the critic call.
* The principal's system prompt is REPLACED with the critic system
  prompt. The critic must not see protocol/tool-schema instructions
  meant for the driver — its job is review, not execution.
* The critic's own reasoning/thinking is discarded; only the JSON
  verdict is propagated. Otherwise the critic's chain-of-thought
  would contaminate the principal's next turn.
* A failed/timed-out critic returns None and the engine treats that
  as ``continue`` — the critic must never be a blocker.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from infinidev.config.llm import get_litellm_params_for_assistant
from infinidev.engine.formats.tool_call_parser import (
    _preprocess as _strip_noise,
    safe_json_loads as _safe_json_loads,
)
import re


logger = logging.getLogger(__name__)


CriticAction = Literal["continue", "information", "recommendation", "reject"]
_VALID_ACTIONS: frozenset[str] = frozenset(
    {"continue", "information", "recommendation", "reject"}
)


@dataclass(frozen=True)
class CriticVerdict:
    """The critic's response to a batch of tool calls.

    Attributes:
        action: One of continue / information / recommendation / reject.
            Pure semantic signal for the principal — the engine treats
            them all identically (inject the message, never block).
        message: The text shown to the principal in the next iteration.
            Empty when action == "continue" (no message is injected).
    """

    action: CriticAction
    message: str

    @property
    def is_silent(self) -> bool:
        return self.action == "continue" or not self.message.strip()


_SYSTEM_PROMPT = (
    "You are the assistant LLM paired with a principal developer "
    "LLM. Each turn you see the principal's internal reasoning and "
    "the tool calls they're about to run, and you reply through "
    "emit_verdict — but you also have read-only tools (read_file, "
    "code_search, list_directory, get_symbol_code, find_references, "
    "etc.) that you can call FIRST to verify the principal's claims "
    "before judging them. Use them sparingly: you have a hard cap of "
    "3 read calls per review. If you don't need to verify anything, "
    "skip them and emit_verdict directly — that's the common case.\n\n"
    "Your four objectives, in priority order:\n\n"
    "1. **Help the principal reach the user's goal.** Their task is "
    "your task. Anything you say should move them toward done.\n\n"
    "2. **Safeguard code quality, efficiency, and system security.** "
    "If a tool call would write broken/insecure code, run a "
    "destructive command, edit a file the principal hasn't read, "
    "commit somewhere it shouldn't, or take a clearly wasteful path "
    "— flag it. Reject only when the action is genuinely dangerous "
    "and proceeding would cause real harm.\n\n"
    "3. **Unstick the principal when you see them in trouble.** "
    "Watch for loops, dead ends, and dependency hell. If the "
    "principal has tried the same approach 3+ times with different "
    "variants and none worked (e.g. five `pip install` permutations "
    "for the same package), step in with a recommendation: try a "
    "different angle, drop the blocker, or close the step with what "
    "they have. Their context window doesn't show them the loop "
    "from the outside — yours does.\n\n"
    "4. **Stay silent when you have nothing productive to add.** "
    "Your verdict goes to another LLM, not a human. Cheering "
    "(\"good call\", \"you're on the right track\", \"smart "
    "workaround\") is pure noise — it costs the principal tokens "
    "and dilutes the signal of your real interventions. Agreement "
    "is not a contribution. If the move is sound and you have no "
    "fact, no warning, no alternative, no loop-break to offer: "
    "action='continue' with message=''. That is the correct, "
    "respected, expected output most of the time.\n\n"
    "Channels (call emit_verdict(action, message) with one of):\n"
    "- continue: nothing productive to add. Empty message. This is "
    "the default when the principal is doing fine.\n"
    "- information: you know a concrete fact the principal doesn't "
    "(a file, an API, a convention, a constraint).\n"
    "- recommendation: a better alternative or a course-correction "
    "with the WHAT and the WHY. Use this to break loops.\n"
    "- reject: the action is dangerous/destructive/clearly wrong. "
    "Strong; use sparingly.\n\n"
    "Hard rules:\n"
    "- Your own thinking is discarded; only emit_verdict reaches "
    "the principal.\n"
    "- You have read-only tools available BUT a strict 3-call "
    "cap — once exhausted, you must emit_verdict on your next "
    "turn. Use reads only when verifying a specific claim would "
    "change your verdict.\n"
    "- Never call write tools (replace_lines, execute_command, "
    "etc.) — they aren't exposed to you. Describe what the "
    "principal should do, don't do it yourself.\n"
    "- Don't echo the prefix the engine adds (\"[ASSISTANT - "
    "<action>]:\") inside your message — the engine prepends it; "
    "you'd just be duplicating noise.\n"
    "- Be concise (<150 words). Direct, second person, no "
    "chitchat, no apologies, no asking them to confirm.\n\n"
    "Plain-text responses (no tool call) are delivered as "
    "emit_verdict(action='information', message=<text>) — but "
    "calling the tool explicitly lets you pick the right channel "
    "AND choose 'continue' to stay silent."
)


_EMIT_VERDICT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "emit_verdict",
        "description": (
            "Send your verdict to the principal. The only tool you "
            "have. If you respond with text instead, the text is "
            "treated as emit_verdict(action='information', message=<text>)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["continue", "information", "recommendation", "reject"],
                    "description": (
                        "continue=nothing to add (empty message); "
                        "information=fact/context the principal is missing; "
                        "recommendation=better alternative with rationale; "
                        "reject=dangerous/wrong action that should not proceed."
                    ),
                },
                "message": {
                    "type": "string",
                    "description": (
                        "Text shown to the principal (empty when "
                        "action=continue). Concise, second-person, no "
                        "preamble. <150 words."
                    ),
                },
            },
            "required": ["action", "message"],
        },
    },
}


_USER_TEMPLATE_HEADER = (
    "{reasoning_block}"
    "The principal just decided on these tool calls for the current "
    "turn:\n\n{proposed}\n\n"
    "Principal's tool catalog (reference — these are THEIR tools, "
    "not yours; you only have emit_verdict):\n{catalog}\n\n"
    "Look at their reasoning and their actions in light of the "
    "<task>, <plan>, <previous-actions> and <current-action> you "
    "saw in the conversation. If you see something worth adding — "
    "a piece of repo context they're missing, a cleaner "
    "alternative, a hole in their logic, a risk — call emit_verdict "
    "with it. Or just respond with text and it'll go through as "
    "information."
)


_REASONING_BLOCK_TEMPLATE = (
    "Principal's internal reasoning for this turn (not in their "
    "history — passed to you out-of-band so you can see how they "
    "arrived at the decision):\n\n<<<\n{reasoning}\n>>>\n\n"
)


def _format_proposed_calls(tool_calls: Iterable[Any]) -> str:
    """Render proposed tool calls compactly for the critic prompt.

    Truncates argument values longer than 240 chars so a giant
    ``replace_lines`` payload doesn't blow the critic's context.
    """
    lines: list[str] = []
    for tc in tool_calls:
        try:
            name = tc.function.name
            raw = tc.function.arguments
            args = _safe_json_loads(raw) if isinstance(raw, str) else (raw or {})
        except Exception:
            lines.append(f"- <unparseable tool call: {tc!r}>")
            continue

        rendered_args: list[str] = []
        for k, v in args.items():
            sv = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
            if len(sv) > 240:
                sv = sv[:240] + f"... (+{len(sv) - 240} chars)"
            rendered_args.append(f"{k}={sv}")
        lines.append(f"- {name}({', '.join(rendered_args)})")
    return "\n".join(lines) if lines else "- <none>"


def _format_tool_catalog(tool_descriptions: dict[str, str]) -> str:
    if not tool_descriptions:
        return "(sin catálogo disponible)"
    return "\n".join(
        f"- {name}: {(desc or '').strip().splitlines()[0] if desc else '(sin descripción)'}"
        for name, desc in sorted(tool_descriptions.items())
    )


def _strip_principal_system(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop any ``role=system`` messages from the principal's history.

    The critic gets its own system prompt prepended later. Leaving the
    principal's protocol/tool-schema system in there would tell the
    critic "you must call tools and emit step_complete", which is
    exactly what we don't want.
    """
    return [m for m in messages if m.get("role") != "system"]


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_FIRST_OBJECT_RE = re.compile(r'\{[^{}]*?"action"[^{}]*?(?:"message"[^{}]*?)?\}', re.DOTALL)
# Fallback for models that imitate the engine's downstream prefix
# (``[ASSISTANT - <action>]: <message>``) instead of emitting JSON.
# This happens because the prefix appears verbatim in the principal's
# message history and the model pattern-matches the surrounding text
# rather than the system prompt's schema. Pulls the first such marker
# in the response — anything after the colon is treated as the
# message body, even if it spans multiple lines.
_PREFIX_RE = re.compile(
    r"\[\s*ASSISTANT\s*(?:\([^)]*\))?\s*-\s*"
    r"(continue|information|recommendation|reject)\s*\]\s*:\s*(.*)",
    re.IGNORECASE | re.DOTALL,
)


def _try_parse_dict(text: str) -> dict | None:
    """Best-effort JSON-to-dict for noisy critic output.

    Tries: raw text → markdown-fence-stripped → first ``{...}`` block
    that mentions ``"action"``. Returns the first variant that yields
    a dict; ``None`` if nothing parses.
    """
    if not text:
        return None
    candidates: list[str] = [text]
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1))
    first_obj = _FIRST_OBJECT_RE.search(text)
    if first_obj:
        candidates.append(first_obj.group(0))
    for candidate in candidates:
        try:
            parsed = _safe_json_loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _parse_verdict(content: str) -> CriticVerdict | None:
    """Extract ``{action, message}`` from the critic's response.

    Tolerates models that:
    * wrap JSON in ```json``` code fences,
    * prepend explanatory prose ("Sure, here's my verdict: {...}"),
    * emit ``<thinking>`` / ``<think>`` blocks despite the system
      prompt asking them not to,
    * leak the entire prose response with the JSON object embedded
      somewhere in the middle.

    Returns ``None`` on any failure — the caller treats that as a
    silent ``continue``. The critic must never break the loop, so
    over-tolerance here is intentional.
    """
    if not content:
        return None

    cleaned = _strip_noise(content).strip()
    parsed = _try_parse_dict(cleaned)
    if parsed is not None:
        action = str(parsed.get("action", "")).strip().lower()
        if action in _VALID_ACTIONS:
            message = parsed.get("message", "")
            if not isinstance(message, str):
                try:
                    message = json.dumps(message, ensure_ascii=False)
                except Exception:
                    message = str(message)
            return CriticVerdict(action=action, message=message.strip())  # type: ignore[arg-type]

    # Fallback: model emitted the downstream prefix format
    # ("[ASSISTANT - information]: ...") instead of JSON. Salvage it
    # rather than discarding — the message content is still useful.
    prefix = _PREFIX_RE.search(cleaned)
    if prefix:
        action = prefix.group(1).strip().lower()
        if action in _VALID_ACTIONS:
            message = prefix.group(2).strip()
            return CriticVerdict(action=action, message=message)  # type: ignore[arg-type]

    return None


# Hard caps for the critic's read sub-loop. Kept conservative so a
# misbehaving critic can never balloon a turn's latency or tool budget.
_MAX_READ_CALLS = 3
_MAX_REVIEW_ITERATIONS = _MAX_READ_CALLS + 2  # +1 for budget exhaustion forced verdict, +1 for safety
_MAX_TOOL_RESULT_CHARS = 4000  # truncate long read outputs (e.g. read_file on a big file)


# System prompt for ``consult()`` — different stance than review().
# Review is adversarial-cooperative ("can I poke a hole in this?"),
# consult is collaborative ("the principal asked me a question; help
# them"). The same read sub-loop runs underneath; only the framing
# changes.
_CONSULT_SYSTEM_PROMPT = (
    "You are the assistant LLM. The principal developer just asked "
    "you a question via the consult_assistant tool — they want your "
    "help, not a verdict. Answer directly and usefully.\n\n"
    "You can call read-only tools (read_file, code_search, etc.) "
    "BEFORE answering — same hard cap of 3 calls per consult. Use "
    "them when the question requires verifying something concrete. "
    "Skip them when the answer is already clear from the "
    "conversation context.\n\n"
    "Style:\n"
    "- Be concrete: cite paths, line numbers, function names.\n"
    "- Be concise: <250 words, second person, no preamble.\n"
    "- If you DON'T know, say so explicitly — don't make up "
    "  answers. The principal will then know to investigate.\n"
    "- Never call write tools. You're not editing — you're advising.\n\n"
    "End with your prose answer (no special channel needed; whatever "
    "you say goes back to the principal verbatim as the tool result)."
)


CONSULT_ASSISTANT_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "consult_assistant",
        "description": (
            "Ask the assistant critic for help. Use when you're "
            "stuck, want a second opinion on an approach, or need "
            "the assistant to verify something with a read tool you "
            "don't want to run yourself. The assistant has access to "
            "read-only tools and ~250 words of response budget. The "
            "answer comes back as this tool's result. Do NOT use "
            "this for trivial questions — call it when extra eyes "
            "would actually change your next move."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "Your question, as you'd ask a senior pair "
                        "partner. Be specific — name the file/symbol/"
                        "behaviour you're asking about."
                    ),
                    "minLength": 20,
                },
                "context_hint": {
                    "type": "string",
                    "description": (
                        "Optional. Extra context the assistant might "
                        "need that isn't already in the conversation "
                        "(e.g. which files you've considered, what "
                        "you've already ruled out)."
                    ),
                },
            },
            "required": ["question"],
        },
    },
}


# Process-global registry for the active AssistantCritic, used by
# ``ConsultAssistantTool._run`` to reach the critic without
# dependency-injecting it through every layer of the engine. The
# engine sets this on critic init and clears on shutdown. Multi-loop
# scenarios (which don't currently exist) would need a richer registry.
_ACTIVE_CRITIC: "AssistantCritic | None" = None


def set_active_critic(critic: "AssistantCritic | None") -> None:
    """Register (or clear) the active critic for ConsultAssistantTool.

    Called by ``LoopEngine`` when it builds a critic. Idempotent — a
    second registration replaces the first. Pass ``None`` on shutdown.
    """
    global _ACTIVE_CRITIC
    _ACTIVE_CRITIC = critic


def get_active_critic() -> "AssistantCritic | None":
    """Read the active critic. Returns None when no loop is running
    or when the assistant LLM feature is disabled."""
    return _ACTIVE_CRITIC


class AssistantCritic:
    """Pair-programming critic with a bounded read sub-loop.

    One instance is constructed per developer-loop run (in
    ``LoopEngine.__init__`` when ``ASSISTANT_LLM_ENABLED``) and reused
    across iterations. ``review()`` is safe to call from a worker
    thread — it does not mutate shared state beyond per-call locals.

    The critic owns a small set of read-only tools (read_file,
    code_search, etc.) and may call up to :data:`_MAX_READ_CALLS` of
    them before emitting its verdict. The cap is intentional: cheap
    verdicts ("looks fine, continue") should not pay for tools, and
    a runaway critic must never block the principal indefinitely.
    """

    def __init__(self, tool_descriptions: dict[str, str]):
        # Build params eagerly so a misconfigured assistant LLM raises
        # at engine startup, not in the middle of a step. The principal
        # loop instantiates us only when the feature is enabled.
        self._params = get_litellm_params_for_assistant()
        self._tool_descriptions = tool_descriptions
        # Short, human-readable model name for the [ASSISTANT (...)] prefix.
        full = str(self._params.get("model", ""))
        self._model_short = full.split("/", 1)[-1] if "/" in full else full

        # Read-only tool kit for the sub-loop. Built lazily once and
        # bound to a synthetic critic agent_id so the critic's reads
        # don't pollute the principal's ``opened_files`` tracking.
        self._read_tools: list[Any] = []
        self._read_tool_dispatch: dict[str, Any] = {}
        self._read_tool_schemas: list[dict[str, Any]] = []
        self._init_read_tools()

    def _init_read_tools(self) -> None:
        """Instantiate, bind, and pre-build OpenAI schemas for read tools.

        Failure here is non-fatal: if anything goes wrong (import
        error, schema build error), the critic falls back to its
        legacy "no tools" mode. The principal's loop is never
        affected.
        """
        try:
            from infinidev.tools import get_tools_for_role
            from infinidev.tools.base.context import bind_tools_to_agent
            from infinidev.engine.loop.schema_sanitizer import tool_to_openai_schema

            tools = get_tools_for_role("assistant_critic")
            # Synthetic agent_id keeps the critic's reads isolated from
            # the principal's tracking. Stable per-instance so
            # subsequent reviews share file cache state.
            critic_agent_id = f"assistant-critic-{id(self) & 0xFFFFFF:06x}"
            bind_tools_to_agent(tools, critic_agent_id)
            self._read_tools = tools
            self._read_tool_dispatch = {t.name: t for t in tools}
            self._read_tool_schemas = [tool_to_openai_schema(t) for t in tools]
            logger.debug(
                "critic: %d read tools available (%s)",
                len(tools),
                ", ".join(sorted(self._read_tool_dispatch.keys())),
            )
        except Exception:
            logger.exception("critic: read tool init failed; falling back to verdict-only")
            self._read_tools = []
            self._read_tool_dispatch = {}
            self._read_tool_schemas = []

    @property
    def model_short_name(self) -> str:
        return self._model_short

    def _execute_read_tool(self, name: str, args: dict[str, Any]) -> str:
        """Execute one of the critic's read tools.

        Returns a string result (truncated to
        :data:`_MAX_TOOL_RESULT_CHARS`). Errors become error strings
        that go back to the critic — the loop never breaks because of
        a tool failure.
        """
        tool = self._read_tool_dispatch.get(name)
        if tool is None:
            return f"[critic-error] tool {name!r} is not in the critic's read-only toolkit."
        if name == "emit_verdict":
            return "[critic-error] emit_verdict is not a read tool."
        try:
            result = tool._run(**args) if isinstance(args, dict) else tool._run()
        except TypeError as exc:
            return f"[critic-error] {name}: bad arguments ({exc})"
        except Exception as exc:
            logger.debug("critic: read tool %s raised", name, exc_info=True)
            return f"[critic-error] {name}: {exc}"
        if not isinstance(result, str):
            try:
                result = json.dumps(result, ensure_ascii=False)
            except Exception:
                result = str(result)
        if len(result) > _MAX_TOOL_RESULT_CHARS:
            result = result[:_MAX_TOOL_RESULT_CHARS] + (
                f"\n\n[critic-truncated: +{len(result) - _MAX_TOOL_RESULT_CHARS} chars]"
            )
        return result

    @staticmethod
    def _serialise_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
        """Serialise the LLM response tool_calls back into the OpenAI
        message-format dicts so they can be appended to the conversation
        history for the next sub-loop iteration.
        """
        out: list[dict[str, Any]] = []
        for tc in tool_calls or []:
            try:
                out.append({
                    "id": getattr(tc, "id", "") or "",
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                })
            except Exception:
                continue
        return out

    def review(
        self,
        messages: list[dict[str, Any]],
        tool_calls: Iterable[Any],
        reasoning: str | None = None,
    ) -> CriticVerdict | None:
        """Ask the critic to review a batch of proposed tool calls.

        ``reasoning`` is the principal's current-turn ``reasoning_content``
        when available. It is rendered as a transient block in the user
        prompt so the critic can intervene on the *thinking*, not just
        on the resulting actions. Pass ``None`` (the default) when no
        reasoning was emitted (e.g. non-thinking models).

        Returns ``None`` on any failure (network, timeout, malformed
        JSON, empty response) — the engine treats that as ``continue``
        and proceeds silently. The critic is never allowed to break
        the loop.
        """
        proposed = list(tool_calls)
        if not proposed:
            return None

        reasoning_block = ""
        if reasoning and reasoning.strip():
            reasoning_block = _REASONING_BLOCK_TEMPLATE.format(
                reasoning=reasoning.strip(),
            )

        try:
            crit_messages: list[dict[str, Any]] = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                *_strip_principal_system(messages),
                {
                    "role": "user",
                    "content": _USER_TEMPLATE_HEADER.format(
                        reasoning_block=reasoning_block,
                        proposed=_format_proposed_calls(proposed),
                        catalog=_format_tool_catalog(self._tool_descriptions),
                    ),
                },
            ]
        except Exception:
            logger.exception("critic: prompt assembly failed")
            return None

        # Lazy import — keeps engine startup cheap and avoids a
        # circular import when the engine module loads us.
        try:
            from infinidev.engine.llm_client import call_llm as _call_llm
        except Exception:
            logger.exception("critic: llm_client import failed")
            return None

        reads_done = 0
        for sub_iter in range(_MAX_REVIEW_ITERATIONS):
            # Tool list: read tools available only while under the read
            # budget. On the budget-exhausted iteration we drop them so
            # the critic must emit_verdict — same effect as forcing
            # ``tool_choice``, but provider-agnostic and clearer in
            # logs.
            tools_for_call: list[dict[str, Any]] = [_EMIT_VERDICT_TOOL]
            if self._read_tool_schemas and reads_done < _MAX_READ_CALLS:
                tools_for_call = list(self._read_tool_schemas) + [_EMIT_VERDICT_TOOL]

            try:
                response = _call_llm(
                    self._params,
                    crit_messages,
                    tools=tools_for_call,
                    tool_choice="auto",
                )
            except Exception as exc:
                logger.warning(
                    "critic: LLM call failed at sub-iter %d (%s); silent fallback",
                    sub_iter, exc,
                )
                return None

            # If the model returned a verdict, we're done.
            verdict = self._extract_verdict(response)
            if verdict is not None:
                logger.info(
                    "critic verdict: %s (after %d read calls) | %s",
                    verdict.action, reads_done,
                    verdict.message[:160] + ("..." if len(verdict.message) > 160 else ""),
                )
                return verdict

            # No verdict yet — check if the model called read tools.
            try:
                msg = response.choices[0].message
            except (AttributeError, IndexError, KeyError):
                logger.info("critic: malformed response at sub-iter %d; silent fallback", sub_iter)
                return None

            tool_calls_out = list(getattr(msg, "tool_calls", None) or [])
            non_verdict_calls = [
                tc for tc in tool_calls_out
                if getattr(tc.function, "name", "") != "emit_verdict"
            ]

            # No tool calls AND no verdict means a text-only response
            # we couldn't parse as a verdict. Treat it as silent
            # fallback (the legacy behaviour).
            if not non_verdict_calls:
                logger.info(
                    "critic: empty/unrecognised response at sub-iter %d; silent fallback",
                    sub_iter,
                )
                return None

            # Append the assistant message with its tool_calls so the
            # next LLM call sees the conversation correctly.
            crit_messages.append({
                "role": "assistant",
                "content": getattr(msg, "content", "") or "",
                "tool_calls": self._serialise_tool_calls(tool_calls_out),
            })

            # Execute each read tool call. If we're already at the cap,
            # respond with a "budget exhausted" tool result instead of
            # actually calling the tool.
            for tc in non_verdict_calls:
                tool_name = tc.function.name
                tool_call_id = getattr(tc, "id", "") or ""
                if reads_done >= _MAX_READ_CALLS:
                    crit_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": (
                            "[critic-error] read budget exhausted. "
                            "You must call emit_verdict on your next "
                            "turn — no more read tools available."
                        ),
                    })
                    continue
                try:
                    args_raw = tc.function.arguments or "{}"
                    args = _safe_json_loads(args_raw)
                    if not isinstance(args, dict):
                        args = {}
                except Exception:
                    args = {}
                result = self._execute_read_tool(tool_name, args)
                logger.debug("critic read tool: %s(%s) → %d chars", tool_name, list(args.keys()), len(result))
                crit_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": result,
                })
                reads_done += 1

        # Loop exhausted without a verdict — extremely rare. Fail
        # silently so the principal continues.
        logger.info(
            "critic: sub-loop exhausted after %d iters / %d reads without verdict",
            _MAX_REVIEW_ITERATIONS, reads_done,
        )
        return None

    @staticmethod
    def _extract_verdict(response: Any) -> CriticVerdict | None:
        """Pull a verdict from the LLM response.

        Three paths, in order:

        1. **Native tool call** (``emit_verdict``). The OpenAI/Anthropic
           function-calling grammar enforces the ``{action, message}``
           schema, so this is the strongly-typed happy path.
        2. **Text content** (no tool call). Treated as
           ``information`` — the model said something useful in prose
           even though it didn't pick the channel. The text reaches
           the principal verbatim.
        3. **Legacy in-text JSON / prefix imitation**. ``_parse_verdict``
           still runs as a last layer in case the model returned BOTH
           text and embedded JSON — we'd rather honour the explicit
           channel than collapse to ``information``.

        Returns ``None`` only when all three yield nothing useful
        (empty content + no tool calls).
        """
        try:
            message = response.choices[0].message
        except (AttributeError, IndexError, KeyError):
            return None

        # Path 1: native tool call.
        tool_calls = getattr(message, "tool_calls", None) or []
        for tc in tool_calls:
            try:
                fn = tc.function
                if fn.name != "emit_verdict":
                    continue
                args_raw = fn.arguments or "{}"
                args = _safe_json_loads(args_raw)
            except Exception:
                continue
            if not isinstance(args, dict):
                continue
            action = str(args.get("action", "")).strip().lower()
            if action not in _VALID_ACTIONS:
                continue
            msg = args.get("message", "")
            if not isinstance(msg, str):
                try:
                    msg = json.dumps(msg, ensure_ascii=False)
                except Exception:
                    msg = str(msg)
            return CriticVerdict(action=action, message=msg.strip())  # type: ignore[arg-type]

        content = (message.content or "").strip() if hasattr(message, "content") else ""
        if not content:
            return None

        # Path 3 first (cheap and lets the model pick its channel even
        # when it forgot to use the tool). Path 2 is the catch-all.
        legacy = _parse_verdict(content)
        if legacy is not None:
            return legacy

        # Path 2: text fallback. The model said something but didn't
        # call the tool and didn't emit a parseable JSON / prefix —
        # treat it as ``information`` so the principal still hears it.
        return CriticVerdict(action="information", message=content)

    def consult(
        self,
        question: str,
        *,
        context_hint: str = "",
        principal_messages: list[dict[str, Any]] | None = None,
    ) -> str:
        """Answer a free-form question from the principal.

        Different stance than :meth:`review`: collaborative, not
        adversarial. Same read sub-loop machinery underneath — the
        critic can call up to :data:`_MAX_READ_CALLS` read tools
        before answering. The terminator here is a *text response*,
        not a structured verdict — the LLM's prose comes back to the
        principal verbatim as the consult_assistant tool result.

        Returns a plain string. On any failure (network, malformed
        response, sub-loop exhaustion without a text answer), returns
        a clearly-marked diagnostic string so the principal knows the
        consult didn't go through, rather than mistaking silence for
        an empty answer.
        """
        if not question or not question.strip():
            return "[consult-error] empty question"

        # Build the user prompt: question + optional context hint.
        # The principal's recent message history is forwarded so the
        # consult sees the same task / plan / previous-actions the
        # principal sees — that's what makes this a *pair-programming*
        # consult and not an isolated Q&A.
        user_parts = [f"Question from the principal:\n\n{question.strip()}"]
        if context_hint and context_hint.strip():
            user_parts.append(
                f"\nExtra context the principal supplied:\n\n{context_hint.strip()}"
            )
        user_parts.append(
            "\nAnswer their question directly. If you need to verify "
            "something concrete, call a read tool (read_file, "
            "code_search, etc.) — but only if it would actually "
            "change your answer."
        )

        consult_messages: list[dict[str, Any]] = [
            {"role": "system", "content": _CONSULT_SYSTEM_PROMPT},
            *_strip_principal_system(principal_messages or []),
            {"role": "user", "content": "\n".join(user_parts)},
        ]

        try:
            from infinidev.engine.llm_client import call_llm as _call_llm
        except Exception:
            logger.exception("consult: llm_client import failed")
            return "[consult-error] llm_client unavailable"

        reads_done = 0
        for sub_iter in range(_MAX_REVIEW_ITERATIONS):
            tools_for_call: list[dict[str, Any]] = []
            if self._read_tool_schemas and reads_done < _MAX_READ_CALLS:
                tools_for_call = list(self._read_tool_schemas)

            try:
                response = _call_llm(
                    self._params,
                    consult_messages,
                    tools=tools_for_call if tools_for_call else None,
                    tool_choice="auto" if tools_for_call else None,
                )
            except Exception as exc:
                logger.warning(
                    "consult: LLM call failed at sub-iter %d (%s)",
                    sub_iter, exc,
                )
                return f"[consult-error] LLM call failed: {exc}"

            try:
                msg = response.choices[0].message
            except (AttributeError, IndexError, KeyError):
                return "[consult-error] malformed response"

            tool_calls_out = list(getattr(msg, "tool_calls", None) or [])
            text = (getattr(msg, "content", "") or "").strip()

            # If the model produced text and no tool calls (or only
            # text alongside calls we'd ignore anyway), treat the
            # text as the answer. This is the consult terminator.
            if text and not tool_calls_out:
                logger.info(
                    "consult: answered after %d read calls (%d chars)",
                    reads_done, len(text),
                )
                return text

            # Tool calls without text — execute reads and continue.
            if not tool_calls_out:
                # No tool calls AND no text. Rare. Log and bail.
                logger.info("consult: empty response at sub-iter %d", sub_iter)
                return "[consult-error] empty response from assistant"

            # Append assistant turn before processing reads.
            consult_messages.append({
                "role": "assistant",
                "content": text,
                "tool_calls": self._serialise_tool_calls(tool_calls_out),
            })

            for tc in tool_calls_out:
                tool_name = getattr(tc.function, "name", "")
                tool_call_id = getattr(tc, "id", "") or ""
                if reads_done >= _MAX_READ_CALLS:
                    consult_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": (
                            "[consult-error] read budget exhausted. "
                            "Answer with what you have."
                        ),
                    })
                    continue
                try:
                    args_raw = tc.function.arguments or "{}"
                    args = _safe_json_loads(args_raw)
                    if not isinstance(args, dict):
                        args = {}
                except Exception:
                    args = {}
                result = self._execute_read_tool(tool_name, args)
                consult_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": result,
                })
                reads_done += 1

        logger.info("consult: sub-loop exhausted without text answer")
        return "[consult-error] sub-loop exhausted without an answer"
