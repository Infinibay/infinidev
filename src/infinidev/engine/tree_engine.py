"""Exploration tree engine.

Decomposes complex problems into sub-problems, explores them recursively
with tools, propagates results upward, and synthesizes findings.

Three phases: INIT (decompose) → EXPLORE (loop) → SYNTHESIZE (final).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

from infinidev.engine.base import AgentEngine
from infinidev.engine.tree_context import (
    INIT_TREE_SCHEMA,
    RESOLVE_NODE_SCHEMA,
    SYNTHESIZE_SCHEMA,
    build_explore_prompt,
    build_init_prompt,
    build_synthesize_prompt,
    build_tree_system_prompt,
)
from infinidev.engine.tree_models import (
    Blocker,
    Fact,
    Question,
    TreeNode,
    TreeState,
    propagate,
    select_next_node,
)
from infinidev.engine.loop_tools import (
    build_tool_dispatch,
    build_tool_schemas,
    execute_tool_call,
    tool_to_openai_schema,
)
from infinidev.config.llm import get_litellm_params
from infinidev.config.model_capabilities import get_model_capabilities
from infinidev.db.service import store_exploration_tree

logger = logging.getLogger(__name__)

# ── Event handling ───────────────────────────────────────────────────────────

_event_callback = None


def set_tree_event_callback(callback):
    """Set a callback function to receive tree engine events."""
    global _event_callback
    _event_callback = callback


def _emit_tree_event(
    event_type: str,
    project_id: int,
    agent_id: str,
    data: dict[str, Any],
    user_message: str,
) -> None:
    """Emit event with human-readable message for TUI/classic mode."""
    data["user_message"] = user_message
    if _event_callback:
        _event_callback(event_type, project_id, agent_id, data)
    # Also print in classic mode (no TUI callback)
    else:
        print(user_message, file=sys.stderr, flush=True)


# ── Pretty logging ──────────────────────────────────────────────────────────

_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_MAGENTA = "\033[35m"


def _log(msg: str) -> None:
    if _event_callback is None:
        print(msg, file=sys.stderr, flush=True)


# ── LLM calling (reuse from loop engine) ────────────────────────────────────

# Transient errors for retry
_TRANSIENT_ERRORS = (
    "connection error", "connectionerror", "disconnected",
    "rate limit", "timeout", "503", "502", "429",
    "overloaded", "internal server error",
)
_LLM_RETRIES = 3
_LLM_RETRY_DELAY = 5.0


def _call_llm(
    params: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] = "auto",
) -> Any:
    """Call litellm.completion with retry for transient errors."""
    import litellm
    from infinidev.config.model_capabilities import get_model_capabilities

    caps = get_model_capabilities()
    kwargs: dict[str, Any] = {**params, "messages": messages}

    if tools:
        kwargs["tools"] = tools
        if tool_choice == "required" and not caps.supports_tool_choice_required:
            kwargs["tool_choice"] = "auto"
        else:
            kwargs["tool_choice"] = tool_choice

    last_exc: Exception | None = None
    for attempt in range(1, _LLM_RETRIES + 1):
        try:
            return litellm.completion(**kwargs)
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            is_transient = any(p in msg for p in _TRANSIENT_ERRORS)
            if not is_transient or attempt == _LLM_RETRIES:
                raise
            delay = _LLM_RETRY_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "Transient LLM error (attempt %d/%d), retrying in %.1fs: %s",
                attempt, _LLM_RETRIES, delay, str(exc)[:200],
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


# ── Manual tool call parsing (simplified from loop engine) ───────────────────

class _ManualToolCall:
    """Lightweight stand-in for native tool call objects."""

    __slots__ = ("id", "function")

    class _Function:
        __slots__ = ("name", "arguments")

        def __init__(self, name: str, arguments: str) -> None:
            self.name = name
            self.arguments = arguments

    def __init__(self, id: str, name: str, arguments: str) -> None:
        self.id = id
        self.function = self._Function(name, arguments)


def _parse_text_tool_calls(content: str) -> list[dict[str, Any]] | None:
    """Parse tool calls from model text (reuse loop engine's parser)."""
    from infinidev.engine.loop_engine import _parse_text_tool_calls as _parse
    return _parse(content)


# ── Main Engine ──────────────────────────────────────────────────────────────


class TreeEngine(AgentEngine):
    """Exploration tree engine.

    Decomposes problems into sub-problems, explores them with tools,
    propagates results, and synthesizes findings.
    """

    def execute(
        self,
        agent: Any,
        task_prompt: tuple[str, str],
        *,
        verbose: bool = True,
        guardrail: Any | None = None,
        guardrail_max_retries: int = 5,
        output_pydantic: type | None = None,
        task_tools: list | None = None,
        event_id: int | None = None,
        resume_state: dict | None = None,
    ) -> str:
        """Execute the exploration tree engine."""
        from infinidev.config.settings import settings

        llm_params = get_litellm_params()
        if llm_params is None:
            raise RuntimeError("TreeEngine requires LiteLLM parameters.")

        caps = get_model_capabilities()
        manual_tc = not caps.supports_function_calling

        desc, expected = task_prompt
        project_id = getattr(agent, "project_id", 1)
        agent_id = getattr(agent, "agent_id", "tree_agent")

        # Resolve tools
        tools = task_tools if task_tools is not None else getattr(agent, "tools", [])
        tool_dispatch = build_tool_dispatch(tools) if tools else {}

        # Build tool schemas for explore phase (regular tools + resolve_node)
        regular_schemas = [tool_to_openai_schema(t) for t in tools] if tools else []

        # System prompt
        system_prompt = build_tree_system_prompt(
            getattr(agent, "backstory", ""),
            identity_override=getattr(agent, "_system_prompt_identity", None),
            session_summaries=getattr(agent, "_session_summaries", None),
        )

        # State
        tree = TreeState()
        _last_prompt_tokens = 0  # Tracks prompt_tokens of last LLM call (for TUI context panel)

        _log(f"\n{_BOLD}{_CYAN}🌳 Exploration Tree Engine{_RESET}")
        _log(f"{_DIM}   Problem: {desc[:120]}{_RESET}")
        _log(f"{_DIM}{'─' * 60}{_RESET}")

        # ── PHASE 1: INIT ────────────────────────────────────────────────
        _log(f"\n{_BOLD}Phase 1: Decomposition{_RESET}")

        init_prompt = build_init_prompt(desc)
        init_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": init_prompt},
        ]

        init_result = self._call_with_schema(
            llm_params, init_messages, [INIT_TREE_SCHEMA],
            "init_tree", tree, manual_tc,
        )
        tree.total_llm_calls += 1

        if init_result is None:
            return "Failed to decompose the problem. The LLM did not produce a valid init_tree call."

        # Build tree from init result
        root_problem = init_result.get("root_problem", desc)
        root_logic = init_result.get("logic", "AND")

        root = TreeNode(
            id="1",
            problem_statement=root_problem,
            logic=root_logic,
            depth=0,
        )

        # Add initial facts
        for f_data in init_result.get("facts", []):
            root.facts.append(Fact(
                content=f_data.get("content", ""),
                source="initial",
                evidence=f_data.get("evidence", ""),
                confidence=f_data.get("confidence", "medium"),
            ))

        # Add initial questions
        for q_data in init_result.get("questions", []):
            root.questions.append(Question(
                content=q_data.get("content", ""),
                question_type=q_data.get("question_type", "informational"),
            ))

        # Add sub-problems as children
        for sp_data in init_result.get("sub_problems", []):
            sp_logic = sp_data.get("logic", "AND")
            child = root.add_child(sp_data.get("problem", ""), logic=sp_logic)
            # Add questions to child if provided
            for q_data in sp_data.get("questions", []):
                child.questions.append(Question(
                    content=q_data.get("content", ""),
                    question_type=q_data.get("question_type", "informational"),
                ))

        tree.root = root
        num_children = len(root.children)

        _emit_tree_event("tree_init", project_id, agent_id, {
            "root_problem": root_problem,
            "num_children": num_children,
            "logic": root_logic,
            "total_tokens": tree.total_tokens,
            "total_llm_calls": tree.total_llm_calls,
        }, f"🌳 Tree initialized: \"{root_problem[:80]}\" → {num_children} sub-problems ({root_logic})")

        # ── PHASE 2: EXPLORE LOOP ────────────────────────────────────────
        _log(f"\n{_BOLD}Phase 2: Exploration{_RESET}")

        explore_schemas = regular_schemas + [RESOLVE_NODE_SCHEMA]

        while True:
            # Budget checks
            if tree.total_llm_calls >= settings.TREE_MAX_LLM_CALLS:
                _log(f"{_YELLOW}⚠ LLM call budget exhausted ({tree.total_llm_calls}){_RESET}")
                break
            if tree.total_tool_calls >= settings.TREE_MAX_TOOL_CALLS:
                _log(f"{_YELLOW}⚠ Tool call budget exhausted ({tree.total_tool_calls}){_RESET}")
                break
            if tree.count_nodes() >= settings.TREE_MAX_NODES:
                _log(f"{_YELLOW}⚠ Node budget exhausted ({tree.count_nodes()}){_RESET}")
                break

            # Check if root already resolved
            if tree.root and tree.root.is_resolved():
                _log(f"{_GREEN}Root resolved: {tree.root.state}{_RESET}")
                break

            # Select next node (skip already-explored nodes)
            node = select_next_node(tree)
            if node is None:
                _log(f"{_GREEN}All nodes resolved.{_RESET}")
                break

            node.state = "exploring"
            tree.current_node_id = node.id

            _emit_tree_event("tree_node_exploring", project_id, agent_id, {
                "node_id": node.id,
                "problem": node.problem_statement,
                "depth": node.depth,
                "total_tokens": tree.total_tokens,
                "total_tool_calls": tree.total_tool_calls,
                "prompt_tokens": _last_prompt_tokens,
            }, f"🔍 Exploring [{node.id}]: \"{node.problem_statement[:80]}\"")

            # Build explore prompt
            explore_prompt = build_explore_prompt(desc, tree, node)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": explore_prompt},
            ]

            # Inner tool loop for this node
            node_tool_calls = 0
            inner_iterations = 0
            resolved = False

            while (
                inner_iterations < settings.TREE_INNER_LOOP_MAX
                and node_tool_calls < settings.TREE_MAX_TOOL_CALLS_PER_NODE
                and tree.total_tool_calls < settings.TREE_MAX_TOOL_CALLS
                and tree.total_llm_calls < settings.TREE_MAX_LLM_CALLS
            ):
                inner_iterations += 1

                # Call LLM
                if manual_tc:
                    response = _call_llm(llm_params, messages)
                else:
                    try:
                        response = _call_llm(
                            llm_params, messages, explore_schemas,
                            tool_choice="auto",
                        )
                    except Exception as exc:
                        logger.warning("LLM error in explore phase: %s", exc)
                        break

                tree.total_llm_calls += 1
                usage = getattr(response, "usage", None)
                if usage:
                    tree.total_tokens += getattr(usage, "total_tokens", 0)
                    _last_prompt_tokens = getattr(usage, "prompt_tokens", 0)

                choice = response.choices[0]
                message = choice.message

                # Get tool calls
                if manual_tc:
                    raw_content = (message.content or "").strip()
                    parsed = _parse_text_tool_calls(raw_content)
                    if parsed:
                        tool_calls = [
                            _ManualToolCall(
                                id=f"tree_{tree.total_tool_calls + i}",
                                name=pc["name"],
                                arguments=(
                                    json.dumps(pc["arguments"])
                                    if isinstance(pc["arguments"], dict)
                                    else str(pc["arguments"])
                                ),
                            )
                            for i, pc in enumerate(parsed)
                        ]
                    else:
                        tool_calls = None
                else:
                    tool_calls = getattr(message, "tool_calls", None)

                if not tool_calls:
                    # No tool calls — try to extract useful text
                    content = (message.content or "").strip()
                    if content:
                        messages.append({"role": "assistant", "content": content})
                        messages.append({
                            "role": "user",
                            "content": (
                                "You must call tools to gather evidence, then call "
                                "`resolve_node` when done exploring this node."
                            ),
                        })
                    continue

                # Process tool calls
                resolve_call = None
                regular_calls = []

                for tc in tool_calls:
                    if tc.function.name == "resolve_node":
                        resolve_call = tc
                    else:
                        regular_calls.append(tc)

                # Build assistant message for FC mode
                if not manual_tc:
                    all_tcs = regular_calls + ([resolve_call] if resolve_call else [])
                    assistant_msg: dict[str, Any] = {
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in all_tcs
                        ],
                    }
                    messages.append(assistant_msg)
                else:
                    messages.append({
                        "role": "assistant",
                        "content": getattr(message, "content", "") or "",
                    })

                # Execute regular tools
                tool_results_text: list[str] = []
                for tc in regular_calls:
                    result = execute_tool_call(
                        tool_dispatch, tc.function.name, tc.function.arguments,
                    )
                    node_tool_calls += 1
                    tree.total_tool_calls += 1
                    node.tool_calls_count += 1

                    _emit_tree_event("tree_tool_call", project_id, agent_id, {
                        "node_id": node.id,
                        "tool_name": tc.function.name,
                        "args_preview": tc.function.arguments[:80] if tc.function.arguments else "",
                        "total_tokens": tree.total_tokens,
                        "total_tool_calls": tree.total_tool_calls,
                        "prompt_tokens": _last_prompt_tokens,
                    }, f"  🔧 [{node.id}] {tc.function.name}")

                    if manual_tc:
                        tool_results_text.append(
                            f"[Tool: {tc.function.name}] Result:\n{result}"
                        )
                    else:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })

                # Handle resolve_node
                if resolve_call:
                    if not manual_tc:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": resolve_call.id,
                            "content": '{"status": "acknowledged"}',
                        })

                    resolve_args = self._parse_args(resolve_call.function.arguments)
                    self._apply_resolve(node, resolve_args, tree, settings)

                    _emit_tree_event("tree_node_resolved", project_id, agent_id, {
                        "node_id": node.id,
                        "state": node.state,
                        "confidence": node.confidence,
                        "summary": node.exploration_summary or "",
                        "total_tokens": tree.total_tokens,
                        "total_tool_calls": tree.total_tool_calls,
                        "prompt_tokens": _last_prompt_tokens,
                    }, f"  ✅ [{node.id}] → {node.state} ({node.confidence}): {(node.exploration_summary or '')[:80]}")

                    tree.explored_node_ids.append(node.id)
                    resolved = True

                    # Propagate
                    propagate(tree.root)
                    total = tree.count_nodes()
                    resolved_count = sum(
                        1 for nid in tree.explored_node_ids
                        if tree.get_node(nid) and tree.get_node(nid).is_resolved()
                    )
                    _emit_tree_event("tree_propagation", project_id, agent_id, {
                        "root_state": tree.root.state if tree.root else "unknown",
                        "total_nodes": total,
                        "resolved_nodes": resolved_count,
                        "total_tokens": tree.total_tokens,
                        "total_tool_calls": tree.total_tool_calls,
                    }, f"  📊 Progress: {resolved_count}/{total} nodes resolved, root: {tree.root.state if tree.root else '?'}")

                    break

                # Manual mode: send tool results
                if manual_tc and tool_results_text:
                    messages.append({
                        "role": "user",
                        "content": "\n\n".join(tool_results_text),
                    })

            # If inner loop ended without resolving, force resolution
            if not resolved:
                node.state = "needs_experiment"
                node.confidence = "low"
                node.exploration_summary = "Exploration budget exhausted for this node."
                tree.explored_node_ids.append(node.id)
                propagate(tree.root)
                _log(f"  {_YELLOW}⚠ [{node.id}] forced to needs_experiment (budget){_RESET}")

            tree.iteration_count += 1

        # ── PHASE 3: SYNTHESIZE ──────────────────────────────────────────
        _log(f"\n{_BOLD}Phase 3: Synthesis{_RESET}")

        total_nodes = tree.count_nodes()
        _emit_tree_event("tree_synthesizing", project_id, agent_id, {
            "total_nodes": total_nodes,
        }, f"📝 Synthesizing findings from {total_nodes} nodes...")

        synth_prompt = build_synthesize_prompt(desc, tree)
        synth_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": synth_prompt},
        ]

        synth_result = self._call_with_schema(
            llm_params, synth_messages, [SYNTHESIZE_SCHEMA],
            "synthesize", tree, manual_tc,
        )
        tree.total_llm_calls += 1

        if synth_result:
            synthesis_text = synth_result.get("synthesis", "")
            approach = synth_result.get("recommended_approach", "")
            risks = synth_result.get("risks", [])
            unknowns = synth_result.get("unknowns", [])

            # Build final output
            parts: list[str] = []
            if synthesis_text:
                parts.append(f"## Synthesis\n\n{synthesis_text}")
            if approach:
                parts.append(f"## Recommended Approach\n\n{approach}")
            if risks:
                parts.append("## Risks\n\n" + "\n".join(f"- {r}" for r in risks))
            if unknowns:
                parts.append("## Unknowns\n\n" + "\n".join(f"- {u}" for u in unknowns))

            tree.synthesis = "\n\n".join(parts)
        else:
            tree.synthesis = self._fallback_synthesis(tree)

        # Save to DB
        try:
            store_exploration_tree(
                project_id=project_id,
                problem=desc,
                tree_json=tree.model_dump_json(),
                session_id=getattr(agent, "_session_id", None),
                agent_id=agent_id,
                synthesis=tree.synthesis,
                status="completed" if tree.root and tree.root.is_resolved() else "exhausted",
                total_nodes=tree.count_nodes(),
                total_tool_calls=tree.total_tool_calls,
                total_tokens=tree.total_tokens,
            )
        except Exception as exc:
            logger.warning("Failed to store exploration tree: %s", exc)

        _emit_tree_event("tree_finished", project_id, agent_id, {
            "status": tree.root.state if tree.root else "unknown",
            "total_nodes": tree.count_nodes(),
            "synthesis_preview": (tree.synthesis or "")[:200],
        }, f"🏁 Exploration complete: {tree.count_nodes()} nodes, root: {tree.root.state if tree.root else '?'}")

        _log(f"\n{_DIM}{'─' * 60}{_RESET}")
        _log(
            f"✅ {_BOLD}Exploration complete{_RESET}  "
            f"{_DIM}{tree.iteration_count} iterations · "
            f"{tree.total_tool_calls} tools · "
            f"{tree.total_llm_calls} LLM calls · "
            f"{tree.total_tokens} tokens{_RESET}\n"
        )

        return tree.synthesis or "Exploration completed but no synthesis was produced."

    def explore_subproblem(self, agent: Any, problem: str) -> str:
        """Convenience method for invoking from loop engine."""
        return self.execute(
            agent,
            task_prompt=(problem, "Explore this sub-problem and synthesize findings."),
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _call_with_schema(
        self,
        llm_params: dict[str, Any],
        messages: list[dict[str, Any]],
        schemas: list[dict],
        expected_tool: str,
        tree: TreeState,
        manual_tc: bool,
    ) -> dict[str, Any] | None:
        """Call LLM expecting a specific pseudo-tool call. Returns parsed args or None."""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                if manual_tc:
                    response = _call_llm(llm_params, messages)
                else:
                    response = _call_llm(
                        llm_params, messages, schemas,
                        tool_choice="required",
                    )
            except Exception as exc:
                logger.warning("LLM call failed: %s", exc)
                if attempt < max_attempts - 1:
                    continue
                return None

            usage = getattr(response, "usage", None)
            if usage:
                tree.total_tokens += getattr(usage, "total_tokens", 0)

            choice = response.choices[0]
            message = choice.message

            # Extract tool call
            if manual_tc:
                raw_content = (message.content or "").strip()
                parsed = _parse_text_tool_calls(raw_content)
                if parsed:
                    for pc in parsed:
                        if pc["name"] == expected_tool:
                            args = pc["arguments"]
                            return args if isinstance(args, dict) else {}
            else:
                tool_calls = getattr(message, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        if tc.function.name == expected_tool:
                            return self._parse_args(tc.function.arguments)

            # Retry with nudge
            content = (message.content or "").strip()
            if content:
                messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": f"You must call the `{expected_tool}` tool. Please try again.",
            })

        return None

    def _parse_args(self, arguments: str | dict[str, Any]) -> dict[str, Any]:
        """Parse tool call arguments from string or dict."""
        if isinstance(arguments, dict):
            return arguments
        try:
            return json.loads(arguments) if arguments and arguments.strip() else {}
        except json.JSONDecodeError:
            return {}

    def _apply_resolve(
        self,
        node: TreeNode,
        args: dict[str, Any],
        tree: TreeState,
        settings: Any,
    ) -> None:
        """Apply resolve_node arguments to a tree node."""
        node.state = args.get("state", "solvable")
        node.confidence = args.get("confidence", "medium")
        node.exploration_summary = args.get("summary", "")

        # New facts
        for f_data in args.get("new_facts", []):
            node.facts.append(Fact(
                content=f_data.get("content", ""),
                source="discovered",
                evidence=f_data.get("evidence", ""),
                source_tool=f_data.get("source_tool", ""),
                confidence=f_data.get("confidence", "medium"),
            ))

        # Answered questions
        for aq_data in args.get("answered_questions", []):
            q_text = aq_data.get("question", "")
            for q in node.questions:
                if q.content == q_text and not q.answer:
                    q.answer = aq_data.get("answer", "")
                    q.answered_by_tool = aq_data.get("answered_by_tool")
                    break

        # New constraints
        for c in args.get("new_constraints", []):
            if c not in node.constraints:
                node.constraints.append(c)

        # New blockers
        for b_data in args.get("new_blockers", []):
            node.external_blockers.append(Blocker(
                description=b_data.get("description", ""),
                blocker_type=b_data.get("blocker_type", "unknown"),
                workaround=b_data.get("workaround"),
            ))

        # Problem reformulation
        if args.get("problem_reformulated"):
            node.problem_reformulated = args["problem_reformulated"]

        # Discard reason
        if args.get("discard_reason"):
            node.discard_reason = args["discard_reason"]

        # New sub-problems (further decomposition)
        for sp_data in args.get("new_sub_problems", []):
            if tree.count_nodes() >= settings.TREE_MAX_NODES:
                break
            if node.depth >= settings.TREE_MAX_DEPTH - 1:
                break
            if len(node.children) >= settings.TREE_MAX_CHILDREN:
                break

            child = node.add_child(
                sp_data.get("problem", ""),
                logic=sp_data.get("logic", "AND"),
            )
            for q_data in sp_data.get("questions", []):
                child.questions.append(Question(
                    content=q_data.get("content", ""),
                    question_type=q_data.get("question_type", "informational"),
                ))

    def _fallback_synthesis(self, tree: TreeState) -> str:
        """Produce a synthesis from tree state when LLM fails."""
        if tree.root is None:
            return "Exploration failed: no tree was built."

        parts: list[str] = [
            f"## Exploration Summary\n\nRoot: {tree.root.problem_statement}",
            f"State: {tree.root.state} ({tree.root.confidence} confidence)",
            f"Nodes explored: {len(tree.explored_node_ids)} / {tree.count_nodes()}",
        ]

        # Collect key facts from all nodes
        facts: list[str] = []
        self._collect_all_facts(tree.root, facts)
        if facts:
            parts.append("\n## Key Facts\n\n" + "\n".join(f"- {f}" for f in facts[:20]))

        return "\n".join(parts)

    def _collect_all_facts(self, node: TreeNode, facts: list[str]) -> None:
        for f in node.facts:
            if f.confidence in ("high", "medium"):
                facts.append(f"{f.content} [{f.confidence}]")
        for child in node.children:
            self._collect_all_facts(child, facts)
