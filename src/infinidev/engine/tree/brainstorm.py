"""Creative-brainstorm pipeline for ``TreeEngine``.

Extracted from ``tree/engine.py`` so the engine file can stop juggling
two very different pipelines (analytical explore vs. creative
brainstorm) in one class body. The 5 phases live here:

    ANTI-PATTERN → DIVERGE → EXPLORE → CROSS → CONVERGE

Everything is written as free functions that take the ``TreeEngine``
instance explicitly (``engine`` = what would be ``self``). This keeps
the split mechanically trivial — the engine's own method is a one-line
delegate — and avoids mixin/inheritance gymnastics.
"""

from __future__ import annotations

from typing import Any

from infinidev.engine.tree.context import RESOLVE_NODE_SCHEMA
from infinidev.engine.tree.models import (
    Fact,
    Question,
    STATE_RANK,
    TreeNode,
    TreeState,
)


def execute_brainstorm(
    engine: Any,
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
    # Pull ANSI codes and helpers from the engine module so output
    # stays consistent with the explore pipeline.
    from infinidev.engine.tree.engine import (
        _log,
        _emit_tree_event,
        _BOLD,
        _CYAN,
        _DIM,
        _GREEN,
        _MAGENTA,
        _RESET,
        _YELLOW,
    )

    ctx = engine._setup_context(agent, task_tools, flow="brainstorm")
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
    anti_result = engine._call_with_schema(
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

    root = TreeNode(
        id="1",
        problem_statement=desc,
        logic="OR",
        depth=0,
    )
    for ap in anti_patterns:
        root.facts.append(Fact(
            content=f"BANNED: {ap.get('approach', '')}",
            source="initial",
            confidence="high",
        ))

    tree.root = root

    for i, perspective in enumerate(perspectives, 1):
        oblique = get_random_oblique()

        diverge_prompt = build_diverge_prompt(
            desc, perspective, anti_patterns, oblique,
        )
        diverge_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": diverge_prompt},
        ]

        idea_result = engine._call_with_schema(
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
    engine._explore_loop(
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

        cross_result = engine._call_with_schema(
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

            _log(f"\n{_BOLD}Phase 3b: Explore Hybrid{_RESET}")
            engine._explore_loop(
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

    converge_result = engine._call_with_schema(
        llm_params, converge_messages, [CONVERGE_SCHEMA],
        "rank_ideas", tree, manual_tc,
    )
    tree.total_llm_calls += 1

    if converge_result:
        tree.synthesis = format_brainstorm_synthesis(converge_result)
    else:
        tree.synthesis = engine._fallback_synthesis(tree)

    engine._store_tree(tree, desc, project_id, agent_id, agent, mode="brainstorm")

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


def format_brainstorm_synthesis(converge_result: dict[str, Any]) -> str:
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
