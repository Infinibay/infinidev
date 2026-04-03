"""Exploration tree engine.

Decomposes complex problems into sub-problems, explores them recursively
with tools, propagates results upward, and synthesizes findings.

Two modes:
- **explore**: analytical decomposition (INIT -> EXPLORE -> SYNTHESIZE)
- **brainstorm**: creative ideation (ANTI-PATTERN -> DIVERGE -> EXPLORE -> CROSS -> CONVERGE)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

from infinidev.engine.base import AgentEngine
from infinidev.engine.tree.context import (
    INIT_TREE_SCHEMA,
    RESOLVE_NODE_SCHEMA,
    SYNTHESIZE_SCHEMA,
    build_explore_prompt,
    build_init_prompt,
    build_synthesize_prompt,
    build_tree_system_prompt,
)
from infinidev.engine.tree.models import (
    Blocker,
    Fact,
    Question,
    STATE_RANK,
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

# ── Event handling via centralized EventBus ─────────────────────────────────

from infinidev.flows.event_listeners import event_bus


def _emit_tree_event(
    event_type: str,
    project_id: int,
    agent_id: str,
    data: dict[str, Any],
    user_message: str,
) -> None:
    """Emit event with human-readable message for TUI/classic mode."""
    data["user_message"] = user_message
    event_bus.emit(event_type, project_id, agent_id, data)
    # Also print in classic mode (no TUI subscriber)
    if not event_bus.has_subscribers:
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
    if not event_bus.has_subscribers:
        print(msg, file=sys.stderr, flush=True)


# LLM calling — imported from canonical module (includes retry, hooks,
# thinking suppression, JSON mode, and malformed tool call handling)
from infinidev.engine.llm_client import call_llm as _call_llm


# Manual tool call parsing — imported from canonical module
from infinidev.engine.tool_call_parser import (
    ManualToolCall as _ManualToolCall,
    parse_text_tool_calls as _parse_text_tool_calls,
)


# ── Main Engine ──────────────────────────────────────────────────────────────


class TreeEngine(AgentEngine):
    """Tree-based engine with two modes: explore (analytical) and brainstorm (creative)."""

    # Thoroughness presets: (max_depth, max_children)
    _THOROUGHNESS = {
        "quick": (2, 2),
        "medium": (3, 3),
        "thorough": (4, 4),
    }

    def execute(
        self,
        agent: Any,
        task_prompt: tuple[str, str],
        *,
        mode: str = "explore",
        thoroughness: str | None = None,
        verbose: bool = True,
        guardrail: Any | None = None,
        guardrail_max_retries: int = 5,
        output_pydantic: type | None = None,
        task_tools: list | None = None,
        event_id: int | None = None,
        resume_state: dict | None = None,
    ) -> str:
        """Execute the tree engine in the given mode.

        Args:
            mode: "explore" for analytical decomposition, "brainstorm" for creative ideation.
            thoroughness: "quick" (depth 2), "medium" (depth 3), or "thorough" (depth 4).
                Defaults to settings if not specified.
        """
        # Apply thoroughness overrides
        if thoroughness and thoroughness in self._THOROUGHNESS:
            depth, children = self._THOROUGHNESS[thoroughness]
            from infinidev.config.settings import settings
            self._depth_override = depth
            self._children_override = children
        else:
            self._depth_override = None
            self._children_override = None

        if mode == "brainstorm":
            return self._execute_brainstorm(
                agent, task_prompt, task_tools=task_tools,
            )
        return self._execute_explore(
            agent, task_prompt, task_tools=task_tools,
        )

    def explore_subproblem(self, agent: Any, problem: str) -> str:
        """Convenience method for invoking explore mode from loop engine."""
        return self.execute(
            agent,
            task_prompt=(problem, "Explore this sub-problem and synthesize findings."),
            mode="explore",
        )

    def brainstorm_subproblem(self, agent: Any, problem: str) -> str:
        """Convenience method for invoking brainstorm mode from other flows."""
        return self.execute(
            agent,
            task_prompt=(problem, "Generate creative approaches."),
            mode="brainstorm",
        )

    # ── Explore mode ─────────────────────────────────────────────────────

    def _execute_explore(
        self,
        agent: Any,
        task_prompt: tuple[str, str],
        *,
        task_tools: list | None = None,
    ) -> str:
        """Execute analytical exploration: INIT -> EXPLORE -> SYNTHESIZE."""
        from infinidev.config.settings import settings

        ctx = self._setup_context(agent, task_tools)
        desc, _expected = task_prompt
        llm_params, manual_tc = ctx["llm_params"], ctx["manual_tc"]
        project_id, agent_id = ctx["project_id"], ctx["agent_id"]
        tool_dispatch = ctx["tool_dispatch"]
        regular_schemas = ctx["regular_schemas"]
        system_prompt = ctx["system_prompt"]

        tree = TreeState()

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

        root = self._build_tree_from_init(init_result, desc)
        tree.root = root

        _emit_tree_event("tree_init", project_id, agent_id, {
            "root_problem": root.problem_statement,
            "num_children": len(root.children),
            "logic": root.logic,
            "total_tokens": tree.total_tokens,
            "total_llm_calls": tree.total_llm_calls,
        }, f"🌳 Tree initialized: \"{root.problem_statement[:80]}\" → {len(root.children)} sub-problems ({root.logic})")

        # ── PHASE 2: EXPLORE LOOP ────────────────────────────────────────
        _log(f"\n{_BOLD}Phase 2: Exploration{_RESET}")

        explore_schemas = regular_schemas + [RESOLVE_NODE_SCHEMA]
        self._explore_loop(
            tree, desc, system_prompt, explore_schemas,
            llm_params, manual_tc, tool_dispatch,
            project_id, agent_id, settings,
            build_prompt_fn=lambda d, t, n: build_explore_prompt(d, t, n),
        )

        # ── PHASE 3: SYNTHESIZE ──────────────────────────────────────────
        return self._synthesize_and_store(
            tree, desc, system_prompt, llm_params, manual_tc,
            project_id, agent_id, agent, mode="explore",
        )

    # ── Brainstorm mode ──────────────────────────────────────────────────

    def _execute_brainstorm(
        self,
        agent: Any,
        task_prompt: tuple[str, str],
        *,
        task_tools: list | None = None,
    ) -> str:
        """Execute creative brainstorm: ANTI-PATTERN -> DIVERGE -> EXPLORE -> CROSS -> CONVERGE."""
        from infinidev.config.settings import settings
        from infinidev.engine.tree.brainstorm_context import (
            ANTI_PATTERN_SCHEMA,
            CONVERGE_SCHEMA,
            CROSS_SCHEMA,
            DIVERGE_SCHEMA,
            build_anti_pattern_prompt,
            build_brainstorm_explore_prompt,
            build_converge_prompt,
            build_cross_prompt,
            build_diverge_prompt,
            get_random_oblique,
            select_perspectives,
        )

        ctx = self._setup_context(agent, task_tools, flow="brainstorm")
        desc, _expected = task_prompt
        llm_params, manual_tc = ctx["llm_params"], ctx["manual_tc"]
        project_id, agent_id = ctx["project_id"], ctx["agent_id"]
        tool_dispatch = ctx["tool_dispatch"]
        regular_schemas = ctx["regular_schemas"]
        system_prompt = ctx["system_prompt"]

        tree = TreeState()

        _log(f"\n{_BOLD}{_MAGENTA}🧠 Brainstorm Engine{_RESET}")
        _log(f"{_DIM}   Problem: {desc[:120]}{_RESET}")
        _log(f"{_DIM}{'─' * 60}{_RESET}")

        # ── PHASE 0: ANTI-PATTERN ────────────────────────────────────────
        _log(f"\n{_BOLD}Phase 0: Identifying Obvious Solutions{_RESET}")

        anti_prompt = build_anti_pattern_prompt(desc)
        anti_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": anti_prompt},
        ]
        anti_result = self._call_with_schema(
            llm_params, anti_messages, [ANTI_PATTERN_SCHEMA],
            "identify_obvious", tree, manual_tc,
        )
        tree.total_llm_calls += 1

        anti_patterns: list[dict] = []
        assumptions: list[str] = []
        if anti_result:
            anti_patterns = anti_result.get("obvious_solutions", [])
            assumptions = anti_result.get("assumptions", [])

        banned_summary = ", ".join(
            ap.get("approach", "")[:50] for ap in anti_patterns
        )
        _log(f"{_DIM}   Banned: {banned_summary}{_RESET}")

        _emit_tree_event("brainstorm_anti_pattern", project_id, agent_id, {
            "num_banned": len(anti_patterns),
            "assumptions": assumptions,
        }, f"🚫 Banned {len(anti_patterns)} obvious approaches, {len(assumptions)} assumptions identified")

        # ── PHASE 1: DIVERGE ─────────────────────────────────────────────
        _log(f"\n{_BOLD}Phase 1: Diverge (Forced Perspectives){_RESET}")

        perspectives = select_perspectives(5)

        # Create root with OR logic (any good idea wins)
        root = TreeNode(
            id="1",
            problem_statement=desc,
            logic="OR",
            depth=0,
        )
        # Store anti-patterns as facts on root
        for ap in anti_patterns:
            root.facts.append(Fact(
                content=f"BANNED: {ap.get('approach', '')}",
                source="initial",
                confidence="high",
            ))

        tree.root = root

        # Generate one idea per perspective
        for i, perspective in enumerate(perspectives, 1):
            oblique = get_random_oblique()

            diverge_prompt = build_diverge_prompt(
                desc, perspective, anti_patterns, oblique,
            )
            diverge_messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": diverge_prompt},
            ]

            idea_result = self._call_with_schema(
                llm_params, diverge_messages, [DIVERGE_SCHEMA],
                "propose_idea", tree, manual_tc,
            )
            tree.total_llm_calls += 1

            if idea_result:
                idea_title = idea_result.get("idea_title", f"Idea {i}")
                child = root.add_child(idea_title, logic="AND")
                child.facts.append(Fact(
                    content=f"Perspective: {idea_result.get('perspective_used', perspective['name'])}",
                    source="initial",
                    confidence="high",
                ))
                child.facts.append(Fact(
                    content=f"Novelty: {idea_result.get('novelty_claim', '')}",
                    source="initial",
                    confidence="medium",
                ))
                if idea_result.get("description"):
                    child.problem_reformulated = idea_result["description"]
                for step in idea_result.get("verification_steps", []):
                    child.questions.append(Question(
                        content=step,
                        question_type="informational",
                    ))

                _log(f"  {_CYAN}💡 [{child.id}] {idea_title} ({perspective['name']}){_RESET}")
            else:
                _log(f"  {_YELLOW}⚠ Perspective '{perspective['name']}' produced no idea{_RESET}")

        _emit_tree_event("brainstorm_diverge", project_id, agent_id, {
            "num_ideas": len(root.children),
            "perspectives": [p["name"] for p in perspectives],
        }, f"💡 Generated {len(root.children)} ideas from {len(perspectives)} perspectives")

        if not root.children:
            return "Brainstorm failed: no ideas were generated from any perspective."

        # ── PHASE 2: EXPLORE ─────────────────────────────────────────────
        _log(f"\n{_BOLD}Phase 2: Explore Ideas{_RESET}")

        explore_schemas = regular_schemas + [RESOLVE_NODE_SCHEMA]
        self._explore_loop(
            tree, desc, system_prompt, explore_schemas,
            llm_params, manual_tc, tool_dispatch,
            project_id, agent_id, settings,
            build_prompt_fn=lambda d, t, n: build_brainstorm_explore_prompt(
                d, t, n, anti_patterns,
            ),
            stop_on_root_resolved=False,
            brainstorm=True,
        )

        # ── PHASE 3: CROSS ───────────────────────────────────────────────
        _log(f"\n{_BOLD}Phase 3: Cross-Pollinate{_RESET}")

        # Select top 2 explored ideas by state rank
        explored_ideas = [
            c for c in root.children
            if c.id in tree.explored_node_ids and c.is_resolved()
        ]
        explored_ideas.sort(
            key=lambda c: (
                STATE_RANK.get(c.state, -1),
                {"low": 0, "medium": 1, "high": 2}.get(c.confidence, 0),
            ),
            reverse=True,
        )

        if len(explored_ideas) >= 2:
            idea_a = explored_ideas[0]
            idea_b = explored_ideas[1]

            _log(f"  {_MAGENTA}Crossing [{idea_a.id}] × [{idea_b.id}]{_RESET}")

            cross_prompt = build_cross_prompt(desc, idea_a, idea_b, anti_patterns)
            cross_messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": cross_prompt},
            ]

            cross_result = self._call_with_schema(
                llm_params, cross_messages, [CROSS_SCHEMA],
                "cross_ideas", tree, manual_tc,
            )
            tree.total_llm_calls += 1

            if cross_result:
                hybrid_title = cross_result.get("hybrid_title", "Hybrid idea")
                hybrid = root.add_child(hybrid_title, logic="AND")
                hybrid.facts.append(Fact(
                    content=f"From [{idea_a.id}]: {cross_result.get('what_from_a', '')}",
                    source="initial",
                    confidence="medium",
                ))
                hybrid.facts.append(Fact(
                    content=f"From [{idea_b.id}]: {cross_result.get('what_from_b', '')}",
                    source="initial",
                    confidence="medium",
                ))
                hybrid.facts.append(Fact(
                    content=f"Why better: {cross_result.get('why_better', '')}",
                    source="initial",
                    confidence="medium",
                ))
                if cross_result.get("hybrid_description"):
                    hybrid.problem_reformulated = cross_result["hybrid_description"]
                for step in cross_result.get("verification_steps", []):
                    hybrid.questions.append(Question(
                        content=step,
                        question_type="informational",
                    ))

                _log(f"  {_GREEN}🧬 [{hybrid.id}] {hybrid_title}{_RESET}")

                _emit_tree_event("brainstorm_cross", project_id, agent_id, {
                    "idea_a": idea_a.id,
                    "idea_b": idea_b.id,
                    "hybrid_id": hybrid.id,
                }, f"🧬 Crossed [{idea_a.id}] × [{idea_b.id}] → [{hybrid.id}] {hybrid_title}")

                # Explore the hybrid
                _log(f"\n{_BOLD}Phase 3b: Explore Hybrid{_RESET}")
                self._explore_loop(
                    tree, desc, system_prompt, explore_schemas,
                    llm_params, manual_tc, tool_dispatch,
                    project_id, agent_id, settings,
                    build_prompt_fn=lambda d, t, n: build_brainstorm_explore_prompt(
                        d, t, n, anti_patterns,
                    ),
                    brainstorm=True,
                )
        else:
            _log(f"  {_DIM}Not enough explored ideas to cross (need 2, have {len(explored_ideas)}){_RESET}")

        # ── PHASE 4: CONVERGE ────────────────────────────────────────────
        _log(f"\n{_BOLD}Phase 4: Converge{_RESET}")

        converge_prompt = build_converge_prompt(desc, tree, anti_patterns)
        converge_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": converge_prompt},
        ]

        converge_result = self._call_with_schema(
            llm_params, converge_messages, [CONVERGE_SCHEMA],
            "rank_ideas", tree, manual_tc,
        )
        tree.total_llm_calls += 1

        if converge_result:
            tree.synthesis = self._format_brainstorm_synthesis(converge_result)
        else:
            tree.synthesis = self._fallback_synthesis(tree)

        # Save to DB
        self._store_tree(tree, desc, project_id, agent_id, agent, mode="brainstorm")

        _emit_tree_event("brainstorm_finished", project_id, agent_id, {
            "total_nodes": tree.count_nodes(),
            "total_ideas": len(root.children),
            "synthesis_preview": (tree.synthesis or "")[:200],
        }, f"🏁 Brainstorm complete: {len(root.children)} ideas explored, {tree.count_nodes()} total nodes")

        _log(f"\n{_DIM}{'─' * 60}{_RESET}")
        _log(
            f"✅ {_BOLD}Brainstorm complete{_RESET}  "
            f"{_DIM}{tree.iteration_count} iterations · "
            f"{tree.total_tool_calls} tools · "
            f"{tree.total_llm_calls} LLM calls · "
            f"{tree.total_tokens} tokens{_RESET}\n"
        )

        return tree.synthesis or "Brainstorm completed but no synthesis was produced."

    # ── Shared mechanics ─────────────────────────────────────────────────

    def _explore_loop(
        self,
        tree: TreeState,
        desc: str,
        system_prompt: str,
        explore_schemas: list[dict],
        llm_params: dict[str, Any],
        manual_tc: bool,
        tool_dispatch: dict,
        project_id: int,
        agent_id: str,
        settings: Any,
        *,
        build_prompt_fn: Any,
        stop_on_root_resolved: bool = True,
        brainstorm: bool = False,
    ) -> None:
        """Shared explore loop used by both explore and brainstorm modes.

        Iterates over pending nodes, calls LLM with tools, handles resolve_node.
        The build_prompt_fn controls what prompt each node gets.
        Set stop_on_root_resolved=False for brainstorm mode to explore all branches.
        Set brainstorm=True to use reduced per-node budgets for wide exploration.
        """
        _last_prompt_tokens = 0

        # Per-node limits: brainstorm uses tighter budgets for breadth
        inner_loop_max = (
            settings.TREE_BRAINSTORM_INNER_LOOP_MAX if brainstorm
            else settings.TREE_INNER_LOOP_MAX
        )
        tool_calls_per_node = (
            settings.TREE_BRAINSTORM_TOOL_CALLS_PER_NODE if brainstorm
            else settings.TREE_MAX_TOOL_CALLS_PER_NODE
        )
        max_depth = (
            settings.TREE_BRAINSTORM_MAX_DEPTH if brainstorm
            else (getattr(self, '_depth_override', None) or settings.TREE_MAX_DEPTH)
        )

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
            if stop_on_root_resolved and tree.root and tree.root.is_resolved():
                _log(f"{_GREEN}Root resolved: {tree.root.state}{_RESET}")
                break

            # Select next node
            node = select_next_node(tree)
            if node is None:
                _log(f"{_GREEN}All nodes resolved.{_RESET}")
                break

            # Depth limit: skip nodes that are too deep
            if node.depth > max_depth:
                node.state = "hypothesis"
                node.confidence = "low"
                node.exploration_summary = "Skipped: exceeds depth limit for this mode."
                tree.explored_node_ids.append(node.id)
                propagate(tree.root)
                _log(f"  {_YELLOW}⚠ [{node.id}] skipped (depth {node.depth} > max {max_depth}){_RESET}")
                continue

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

            # Build prompt via mode-specific function
            explore_prompt = build_prompt_fn(desc, tree, node)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": explore_prompt},
            ]

            # Inner tool loop for this node
            node_tool_calls = 0
            inner_iterations = 0
            resolved = False

            while (
                inner_iterations < inner_loop_max
                and node_tool_calls < tool_calls_per_node
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

    def _synthesize_and_store(
        self,
        tree: TreeState,
        desc: str,
        system_prompt: str,
        llm_params: dict[str, Any],
        manual_tc: bool,
        project_id: int,
        agent_id: str,
        agent: Any,
        *,
        mode: str = "explore",
    ) -> str:
        """Synthesize findings and store to DB (explore mode)."""
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

        self._store_tree(tree, desc, project_id, agent_id, agent, mode=mode)

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

    # ── Helpers ───────────────────────────────────────────────────────────

    def _setup_context(
        self,
        agent: Any,
        task_tools: list | None,
        flow: str = "explore",
    ) -> dict[str, Any]:
        """Common setup for both modes."""
        llm_params = get_litellm_params()
        if llm_params is None:
            raise RuntimeError("TreeEngine requires LiteLLM parameters.")

        caps = get_model_capabilities()
        manual_tc = not caps.supports_function_calling

        tools = task_tools if task_tools is not None else getattr(agent, "tools", [])
        tool_dispatch = build_tool_dispatch(tools) if tools else {}
        regular_schemas = [tool_to_openai_schema(t) for t in tools] if tools else []

        if flow == "brainstorm":
            from infinidev.prompts.flows.brainstorm import BRAINSTORM_IDENTITY
            identity = getattr(agent, "_system_prompt_identity", None) or BRAINSTORM_IDENTITY
        else:
            identity = getattr(agent, "_system_prompt_identity", None)

        system_prompt = build_tree_system_prompt(
            getattr(agent, "backstory", ""),
            identity_override=identity,
            session_summaries=getattr(agent, "_session_summaries", None),
        )

        return {
            "llm_params": llm_params,
            "manual_tc": manual_tc,
            "project_id": getattr(agent, "project_id", 1),
            "agent_id": getattr(agent, "agent_id", "tree_agent"),
            "tool_dispatch": tool_dispatch,
            "regular_schemas": regular_schemas,
            "system_prompt": system_prompt,
        }

    def _build_tree_from_init(
        self,
        init_result: dict[str, Any],
        desc: str,
    ) -> TreeNode:
        """Build a TreeNode root from init_tree result."""
        root_problem = init_result.get("root_problem", desc)
        root_logic = init_result.get("logic", "AND")

        root = TreeNode(
            id="1",
            problem_statement=root_problem,
            logic=root_logic,
            depth=0,
        )

        for f_data in init_result.get("facts", []):
            root.facts.append(Fact(
                content=f_data.get("content", ""),
                source="initial",
                evidence=f_data.get("evidence", ""),
                confidence=f_data.get("confidence", "medium"),
            ))

        for q_data in init_result.get("questions", []):
            root.questions.append(Question(
                content=q_data.get("content", ""),
                question_type=q_data.get("question_type", "informational"),
            ))

        for sp_data in init_result.get("sub_problems", []):
            sp_logic = sp_data.get("logic", "AND")
            child = root.add_child(sp_data.get("problem", ""), logic=sp_logic)
            for q_data in sp_data.get("questions", []):
                child.questions.append(Question(
                    content=q_data.get("content", ""),
                    question_type=q_data.get("question_type", "informational"),
                ))

        return root

    def _store_tree(
        self,
        tree: TreeState,
        desc: str,
        project_id: int,
        agent_id: str,
        agent: Any,
        *,
        mode: str = "explore",
    ) -> None:
        """Store exploration tree to DB."""
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

        # Hypothesis content
        if "hypothesis_content" in args and args["hypothesis_content"]:
            if node.state not in ("hypothesis",):
                node.state = "hypothesis"
            node.confidence = "low"
            node.hypothesis_content = args["hypothesis_content"]

        # New sub-problems (further decomposition)
        for sp_data in args.get("new_sub_problems", []):
            if tree.count_nodes() >= settings.TREE_MAX_NODES:
                break
            _eff_depth = getattr(self, '_depth_override', None) or settings.TREE_MAX_DEPTH
            if node.depth >= _eff_depth - 1:
                break
            _eff_children = getattr(self, '_children_override', None) or settings.TREE_MAX_CHILDREN
            if len(node.children) >= _eff_children:
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

    def _format_brainstorm_synthesis(self, converge_result: dict[str, Any]) -> str:
        """Format brainstorm convergence result into readable output."""
        parts: list[str] = []

        synthesis = converge_result.get("synthesis", "")
        if synthesis:
            parts.append(f"## Brainstorm Results\n\n{synthesis}")

        ranked = converge_result.get("ranked_ideas", [])
        if ranked:
            idea_lines: list[str] = []
            for idea in sorted(ranked, key=lambda x: x.get("rank", 99)):
                rank = idea.get("rank", "?")
                title = idea.get("idea_title", "Untitled")
                novelty = idea.get("novelty_score", "?")
                feasibility = idea.get("feasibility_score", "?")
                completeness = idea.get("completeness_score", "?")
                justification = idea.get("justification", "")
                next_steps = idea.get("next_steps", [])

                idea_lines.append(
                    f"### #{rank}: {title}\n"
                    f"- Novelty: {novelty}/5 | Feasibility: {feasibility}/5 | "
                    f"Completeness: {completeness}/5\n"
                    f"- {justification}"
                )
                if next_steps:
                    idea_lines.append(
                        "- Next steps:\n" + "\n".join(f"  - {s}" for s in next_steps)
                    )

            parts.append("## Ranked Ideas\n\n" + "\n\n".join(idea_lines))

        surprise = converge_result.get("surprise_finding")
        if surprise:
            parts.append(f"## Surprise Finding\n\n{surprise}")

        return "\n\n".join(parts)

    def _fallback_synthesis(self, tree: TreeState) -> str:
        """Produce a synthesis from tree state when LLM fails."""
        if tree.root is None:
            return "Exploration failed: no tree was built."

        parts: list[str] = [
            f"## Exploration Summary\n\nRoot: {tree.root.problem_statement}",
            f"State: {tree.root.state} ({tree.root.confidence} confidence)",
            f"Nodes explored: {len(tree.explored_node_ids)} / {tree.count_nodes()}",
        ]

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
