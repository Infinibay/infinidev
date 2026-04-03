"""Prompt construction for the brainstorm engine mode.

Five phases: ANTI-PATTERN → DIVERGE → EXPLORE → CROSS → CONVERGE.
Uses forced perspectives and oblique strategies to inject creativity.
"""

from __future__ import annotations

import random

from infinidev.engine.tree.models import TreeNode, TreeState


# ── Creative lenses ──────────────────────────────────────────────────────────

PERSPECTIVE_LENSES: list[dict] = [
    {
        "id": "inversion",
        "name": "Inversion",
        "prompt": (
            "What if you solve the OPPOSITE problem? "
            "If the goal is X, what solution appears when you aim for not-X "
            "and then flip it?"
        ),
    },
    {
        "id": "analogy",
        "name": "Cross-Domain Analogy",
        "prompt": "How does the {domain} industry solve a similar problem?",
        "domains": [
            "gaming", "biology", "logistics", "music production",
            "urban planning", "restaurant management", "aviation",
            "competitive sports", "film editing", "emergency medicine",
        ],
    },
    {
        "id": "constraint_removal",
        "name": "Constraint Removal",
        "prompt": (
            "Identify the most important constraint or limitation. "
            "Now remove it entirely. What solution becomes possible?"
        ),
    },
    {
        "id": "extreme_user",
        "name": "Extreme User",
        "prompt": "Design this for {persona}. What changes?",
        "personas": [
            "someone with 1 second of attention",
            "a complete beginner who has never coded",
            "a system handling 100M concurrent users",
            "an 85-year-old with low vision",
            "a developer who can only use voice commands",
            "an AI agent with no human in the loop",
        ],
    },
    {
        "id": "stupid_solution",
        "name": "Stupidest Thing That Works",
        "prompt": (
            "What is the dumbest, most embarrassingly simple solution "
            "that could actually work? No elegance allowed."
        ),
    },
    {
        "id": "eliminate",
        "name": "Eliminate the Essential",
        "prompt": (
            "Remove the component that seems most essential to the solution. "
            "What remains? Can you build something useful from just that?"
        ),
    },
    {
        "id": "combine",
        "name": "Unexpected Combination",
        "prompt": "What if this problem was actually a {concept}?",
        "concepts": [
            "card game", "social network", "thermostat", "marketplace",
            "recipe", "conversation", "garden", "assembly line",
            "musical instrument", "voting system",
        ],
    },
    {
        "id": "time_warp",
        "name": "Time Warp",
        "prompt": (
            "How would you solve this in 1 hour with no prep? "
            "Now how would you solve it with 1 year and unlimited budget? "
            "What's the interesting middle ground?"
        ),
    },
    {
        "id": "failure_mode",
        "name": "Embrace Failure",
        "prompt": (
            "Design a solution that is MEANT to fail gracefully. "
            "What if failure is not a bug but a feature?"
        ),
    },
    {
        "id": "reverse_engineer",
        "name": "Reverse Engineer Success",
        "prompt": (
            "Imagine the problem is already solved perfectly. "
            "What does the solution look like? Work backwards from there."
        ),
    },
]

OBLIQUE_STRATEGIES: list[str] = [
    "Use an error as a starting point",
    "What is missing? What is redundant?",
    "Do it backwards",
    "What would you do if you had no fear?",
    "Turn a limitation into a feature",
    "What is the simplest version worth building?",
    "Honor thy mistake as a hidden intention",
    "What would happen if you did nothing?",
    "Remove ambiguities and convert to specifics",
    "Ask your body",
    "Try the opposite extreme",
    "What context would make this idea brilliant?",
    "Make it 10x bigger. Now make it 10x smaller.",
    "What would a child suggest?",
    "Abandon normal instruments",
    "What is the one thing you are avoiding?",
    "Emphasize the flaws",
    "Work at a different speed",
    "What if this was a game?",
    "Disconnect from desire",
    "Go to an extreme, come back to something more sensible",
    "What are the sections? Imagine a caterpillar moving",
    "Only one element of each kind",
    "Do nothing for as long as possible",
    "Destroy the most important thing",
    "What would your enemy do?",
    "Convert a sequence into a loop",
    "Fill every beat with something",
    "Make it the opposite of what it is",
    "Trust the process, distrust the result",
]


# ── Schemas ──────────────────────────────────────────────────────────────────

ANTI_PATTERN_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "identify_obvious",
        "description": (
            "Identify the most obvious/generic solutions to this problem. "
            "These will be BANNED — the brainstorm must go beyond them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "obvious_solutions": {
                    "type": "array",
                    "description": "3-5 obvious/generic approaches to ban",
                    "items": {
                        "type": "object",
                        "properties": {
                            "approach": {
                                "type": "string",
                                "description": "The obvious solution",
                            },
                            "why_obvious": {
                                "type": "string",
                                "description": "Why this is generic/predictable",
                            },
                        },
                        "required": ["approach", "why_obvious"],
                    },
                },
                "assumptions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Common assumptions behind these obvious solutions",
                },
            },
            "required": ["obvious_solutions", "assumptions"],
        },
    },
}


DIVERGE_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "propose_idea",
        "description": (
            "Propose a creative idea based on the assigned perspective. "
            "The idea must differ from banned obvious approaches."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "idea_title": {
                    "type": "string",
                    "description": "Short, descriptive title for the idea",
                },
                "description": {
                    "type": "string",
                    "description": "How the idea works — concrete enough to explore",
                },
                "perspective_used": {
                    "type": "string",
                    "description": "Which creative lens generated this idea",
                },
                "novelty_claim": {
                    "type": "string",
                    "description": "Why this is NOT an obvious solution",
                },
                "feasibility_signals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Early signals suggesting this could work",
                },
                "verification_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "How to validate this idea with tools",
                },
            },
            "required": ["idea_title", "description", "perspective_used", "novelty_claim"],
        },
    },
}


CROSS_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "cross_ideas",
        "description": (
            "Combine two explored ideas into a hybrid that is better than either parent."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "idea_a_id": {
                    "type": "string",
                    "description": "Node ID of first parent idea",
                },
                "idea_b_id": {
                    "type": "string",
                    "description": "Node ID of second parent idea",
                },
                "hybrid_title": {
                    "type": "string",
                    "description": "Title for the hybrid idea",
                },
                "hybrid_description": {
                    "type": "string",
                    "description": "How the hybrid combines both parents",
                },
                "what_from_a": {
                    "type": "string",
                    "description": "What mechanism/insight is taken from idea A",
                },
                "what_from_b": {
                    "type": "string",
                    "description": "What mechanism/insight is taken from idea B",
                },
                "why_better": {
                    "type": "string",
                    "description": "Why the hybrid surpasses both parents",
                },
                "verification_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "How to validate the hybrid",
                },
            },
            "required": [
                "idea_a_id", "idea_b_id", "hybrid_title",
                "hybrid_description", "what_from_a", "what_from_b", "why_better",
            ],
        },
    },
}


CONVERGE_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "rank_ideas",
        "description": "Rank all explored ideas and produce final synthesis.",
        "parameters": {
            "type": "object",
            "properties": {
                "ranked_ideas": {
                    "type": "array",
                    "description": "Ideas ranked from best to worst",
                    "items": {
                        "type": "object",
                        "properties": {
                            "node_id": {"type": "string"},
                            "rank": {"type": "integer"},
                            "idea_title": {"type": "string"},
                            "novelty_score": {
                                "type": "integer",
                                "description": "1-5: how far from obvious approaches",
                            },
                            "feasibility_score": {
                                "type": "integer",
                                "description": "1-5: how viable based on evidence",
                            },
                            "completeness_score": {
                                "type": "integer",
                                "description": "1-5: how fully formed the idea is",
                            },
                            "justification": {"type": "string"},
                            "next_steps": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "node_id", "rank", "idea_title",
                            "novelty_score", "feasibility_score",
                            "completeness_score", "justification",
                        ],
                    },
                },
                "synthesis": {
                    "type": "string",
                    "description": "Overall narrative of the brainstorm results",
                },
                "surprise_finding": {
                    "type": "string",
                    "description": "The most unexpected insight that emerged",
                },
            },
            "required": ["ranked_ideas", "synthesis"],
        },
    },
}


# ── Perspective selection ────────────────────────────────────────────────────


def select_perspectives(
    n: int = 5,
    *,
    must_include: list[str] | None = None,
) -> list[dict]:
    """Select N perspective lenses for diverge phase.

    Always includes 'inversion' and 'stupid_solution' as anchors,
    then fills randomly from the rest. Returns resolved perspectives
    with random domain/persona/concept substitutions.
    """
    anchors = {"inversion", "stupid_solution"}
    if must_include:
        anchors.update(must_include)

    selected = [p for p in PERSPECTIVE_LENSES if p["id"] in anchors]
    remaining = [p for p in PERSPECTIVE_LENSES if p["id"] not in anchors]
    random.shuffle(remaining)

    slots_left = n - len(selected)
    if slots_left > 0:
        selected.extend(remaining[:slots_left])

    # Resolve template variables
    resolved = []
    for p in selected:
        prompt = p["prompt"]
        if "{domain}" in prompt and "domains" in p:
            prompt = prompt.replace("{domain}", random.choice(p["domains"]))
        if "{persona}" in prompt and "personas" in p:
            prompt = prompt.replace("{persona}", random.choice(p["personas"]))
        if "{concept}" in prompt and "concepts" in p:
            prompt = prompt.replace("{concept}", random.choice(p["concepts"]))
        resolved.append({**p, "prompt": prompt})

    return resolved


def get_random_oblique() -> str:
    """Return a random oblique strategy."""
    return random.choice(OBLIQUE_STRATEGIES)


# ── Prompt builders ──────────────────────────────────────────────────────────


def build_anti_pattern_prompt(problem: str) -> str:
    """Build prompt for Phase 0: identify obvious solutions to ban."""
    return (
        f"<task>\n{problem}\n</task>\n\n"
        "<instructions>\n"
        "List the 3-5 most OBVIOUS, GENERIC, PREDICTABLE solutions to this problem.\n"
        "These are the solutions that any developer would suggest first.\n"
        "We want to identify them so we can deliberately go BEYOND them.\n\n"
        "Call `identify_obvious` with:\n"
        "- obvious_solutions: each with the approach and why it's generic\n"
        "- assumptions: the common assumptions behind these obvious solutions\n\n"
        "Be honest — include solutions that are actually good but predictable.\n"
        "The goal is not to avoid good solutions, but to force exploration of\n"
        "ADDITIONAL creative alternatives.\n"
        "</instructions>"
    )


def build_diverge_prompt(
    problem: str,
    perspective: dict,
    anti_patterns: list[dict],
    oblique: str,
) -> str:
    """Build prompt for Phase 1: generate idea from a forced perspective."""
    banned_lines = "\n".join(
        f"- {ap['approach']} ({ap['why_obvious']})"
        for ap in anti_patterns
    )

    return (
        f"<task>\n{problem}\n</task>\n\n"
        f"<banned-approaches>\n"
        f"The following obvious solutions are BANNED — your idea must differ:\n"
        f"{banned_lines}\n"
        f"</banned-approaches>\n\n"
        f"<creative-perspective>\n"
        f"Lens: {perspective['name']}\n"
        f"{perspective['prompt']}\n"
        f"</creative-perspective>\n\n"
        f"<oblique-strategy>\n"
        f"{oblique}\n"
        f"</oblique-strategy>\n\n"
        "<instructions>\n"
        "Using the creative perspective above as your thinking framework, "
        "propose ONE concrete, specific idea for solving this problem.\n\n"
        "The idea must:\n"
        "- Be DIFFERENT from all banned approaches\n"
        "- Be concrete enough to explore with tools\n"
        "- Explain WHY it's novel (not just a repackaging of the obvious)\n\n"
        "Call `propose_idea` with your idea.\n"
        "</instructions>"
    )


def build_brainstorm_explore_prompt(
    problem: str,
    tree: TreeState,
    node: TreeNode,
    anti_patterns: list[dict],
) -> str:
    """Build prompt for Phase 2: explore an idea branch with tools."""
    parts: list[str] = []

    parts.append(f"<task>\n{problem}\n</task>")

    # Banned approaches context
    banned_lines = "\n".join(
        f"- {ap['approach']}" for ap in anti_patterns
    )
    parts.append(
        f"<banned-approaches>\n{banned_lines}\n</banned-approaches>"
    )

    # Tree path
    path = tree.get_path_to_node(node.id)
    path_lines: list[str] = []
    for i, p in enumerate(path):
        indent = "  " * i
        marker = " <-- YOU ARE HERE" if p.id == node.id else ""
        path_lines.append(
            f"{indent}{p.id}: {p.problem_statement} [{p.state}]{marker}"
        )
    parts.append(f"<tree-path>\n{chr(10).join(path_lines)}\n</tree-path>")

    # Sibling ideas for cross-pollination awareness
    siblings = tree.get_siblings(node.id)
    if siblings:
        sib_lines = []
        for s in siblings:
            sib_lines.append(f"- {s.id}: {s.problem_statement} [{s.state}]")
            if s.exploration_summary:
                sib_lines.append(f"  Summary: {s.exploration_summary}")
        parts.append(
            "<sibling-ideas>\nOther ideas being explored:\n"
            + "\n".join(sib_lines)
            + "\n</sibling-ideas>"
        )

    # Current node
    node_parts: list[str] = [f"Idea: {node.problem_statement}"]
    if node.facts:
        node_parts.append("Known facts:")
        for f in node.facts:
            node_parts.append(f"  - {f.content} [{f.confidence}]")
    if node.questions:
        node_parts.append("Open questions:")
        for q in node.questions:
            status = "ANSWERED" if q.answer else "OPEN"
            node_parts.append(f"  - [{status}] {q.content}")
    parts.append(
        "<current-idea>\n" + "\n".join(node_parts) + "\n</current-idea>"
    )

    # Oblique strategy injection
    oblique = get_random_oblique()
    parts.append(f"<oblique-strategy>\n{oblique}\n</oblique-strategy>")

    parts.append(
        "<instructions>\n"
        "You are exploring a CREATIVE idea, not an analytical sub-problem.\n"
        "Use tools to find evidence that makes this idea MORE viable.\n"
        "Also honestly report if it's not feasible.\n\n"
        "Guidelines:\n"
        "- Use tools to validate against the real codebase/environment\n"
        "- State 'hypothesis' is acceptable if direction is promising but "
        "evidence is incomplete\n"
        "- Record feasibility signals as facts\n"
        "- If the idea sparks a BETTER idea, include it as new_sub_problems\n\n"
        "When done, call `resolve_node` with your findings.\n"
        "</instructions>"
    )

    return "\n\n".join(parts)


def build_cross_prompt(
    problem: str,
    idea_a: TreeNode,
    idea_b: TreeNode,
    anti_patterns: list[dict],
) -> str:
    """Build prompt for Phase 3: combine two ideas into a hybrid."""
    banned_lines = "\n".join(
        f"- {ap['approach']}" for ap in anti_patterns
    )

    summary_a = idea_a.exploration_summary or idea_a.problem_statement
    summary_b = idea_b.exploration_summary or idea_b.problem_statement

    facts_a = "\n".join(f"  - {f.content}" for f in idea_a.facts) or "  (none)"
    facts_b = "\n".join(f"  - {f.content}" for f in idea_b.facts) or "  (none)"

    return (
        f"<task>\n{problem}\n</task>\n\n"
        f"<banned-approaches>\n{banned_lines}\n</banned-approaches>\n\n"
        f"<idea-a id=\"{idea_a.id}\">\n"
        f"Title: {idea_a.problem_statement}\n"
        f"State: {idea_a.state} ({idea_a.confidence})\n"
        f"Summary: {summary_a}\n"
        f"Facts:\n{facts_a}\n"
        f"</idea-a>\n\n"
        f"<idea-b id=\"{idea_b.id}\">\n"
        f"Title: {idea_b.problem_statement}\n"
        f"State: {idea_b.state} ({idea_b.confidence})\n"
        f"Summary: {summary_b}\n"
        f"Facts:\n{facts_b}\n"
        f"</idea-b>\n\n"
        "<instructions>\n"
        "Create a HYBRID idea that combines the best aspects of both ideas.\n"
        "The hybrid must be MORE than the sum of its parts — not just A + B,\n"
        "but a synthesis that creates something neither idea achieves alone.\n\n"
        "Call `cross_ideas` with your hybrid.\n"
        "</instructions>"
    )


def build_converge_prompt(
    problem: str,
    tree: TreeState,
    anti_patterns: list[dict],
) -> str:
    """Build prompt for Phase 4: rank ideas and synthesize."""
    parts: list[str] = []

    parts.append(f"<task>\n{problem}\n</task>")

    # Banned approaches for novelty comparison
    banned_lines = "\n".join(
        f"- {ap['approach']}" for ap in anti_patterns
    )
    parts.append(
        f"<banned-approaches>\n"
        f"Obvious solutions (use for novelty scoring — ideas similar to "
        f"these get LOW novelty scores):\n{banned_lines}\n"
        f"</banned-approaches>"
    )

    # All explored ideas
    if tree.root:
        idea_lines: list[str] = []
        _render_idea_tree(tree.root, idea_lines, indent=0)
        parts.append(
            "<explored-ideas>\n"
            + "\n".join(idea_lines)
            + "\n</explored-ideas>"
        )

    parts.append(
        "<instructions>\n"
        "Rank ALL explored ideas and produce a final synthesis.\n\n"
        "Call `rank_ideas` with:\n"
        "- ranked_ideas: each with scores (1-5) for:\n"
        "  - novelty_score: how far from the banned obvious approaches\n"
        "  - feasibility_score: how viable based on tool evidence\n"
        "  - completeness_score: how fully formed the idea is\n"
        "- synthesis: overall narrative of what the brainstorm produced\n"
        "- surprise_finding: the most unexpected insight that emerged\n\n"
        "Scoring guide:\n"
        "- Novelty 1 = basically an obvious approach, 5 = truly unexpected\n"
        "- Feasibility 1 = no evidence, 5 = strong tool-based evidence\n"
        "- Completeness 1 = vague concept, 5 = ready to implement\n"
        "</instructions>"
    )

    return "\n\n".join(parts)


def _render_idea_tree(
    node: TreeNode,
    lines: list[str],
    indent: int,
) -> None:
    """Render the idea tree for convergence prompt."""
    prefix = "  " * indent
    state = f"[{node.state}]" if node.state != "pending" else ""
    conf = f"({node.confidence})" if node.state != "pending" else ""

    lines.append(f"{prefix}{node.id}: {node.problem_statement} {state} {conf}".rstrip())

    if node.exploration_summary:
        lines.append(f"{prefix}  Summary: {node.exploration_summary}")
    if node.hypothesis_content:
        lines.append(f"{prefix}  Hypothesis: {node.hypothesis_content}")
    if node.facts:
        for fact in node.facts:
            lines.append(f"{prefix}  Fact [{fact.confidence}]: {fact.content}")

    for child in node.children:
        _render_idea_tree(child, lines, indent + 1)
