"""Prompt construction for the exploration tree engine.

Three phases: INIT (decompose), EXPLORE (investigate node), SYNTHESIZE (final).
All prompts use XML blocks following the loop engine convention.
"""

from __future__ import annotations

from infinidev.engine.tree_models import TreeNode, TreeState


# ── Protocol (analogous to LOOP_PROTOCOL) ────────────────────────────────────

TREE_PROTOCOL = """\
## Exploration Tree Protocol

You operate in an exploration tree engine. Follow these rules:

### Node States
- **pending** — Not yet explored
- **exploring** — Currently being investigated
- **solvable** — Solution found with evidence
- **unsolvable** — Proven impossible with current constraints
- **mitigable** — Not fully solvable but can be managed/reduced
- **needs_decision** — Requires human input, not more information
- **needs_experiment** — Answer only obtainable by running code/tests
- **discarded** — Dead branch (in OR logic, another path was chosen)
- **hypothesis** — Speculative approach (used in brainstorm mode)

### Logic Modes
- **AND** — All children must be resolved; parent gets WORST child state
- **OR** — Any child can resolve it; parent gets BEST child state

### Exploration Rules
- Maximum 4 children per node, maximum 4 levels of depth
- Every fact MUST have evidence from tool output — no speculation
- Pivot questions restructure the tree; informational ones add data
- When a node seems unsolvable, decompose the assumptions behind "unsolvable"
- Prefer OR logic when exploring alternatives to a blocked path
- Constraints and blockers always propagate upward from children

### CRITICAL
- VERIFY with tools before asserting facts
- Do NOT speculate about code behavior — read the code
- Do NOT assume APIs work a certain way — check documentation
- When done exploring a node, you MUST call `resolve_node`
"""


# ── Pseudo-tool schemas ──────────────────────────────────────────────────────

INIT_TREE_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "init_tree",
        "description": (
            "Initialize the exploration tree by decomposing the problem into "
            "2-4 sub-problems with initial facts and questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "root_problem": {
                    "type": "string",
                    "description": "The root problem, reformulated for clarity",
                },
                "logic": {
                    "type": "string",
                    "enum": ["AND", "OR"],
                    "description": "How sub-problems combine: AND = all needed, OR = any path works",
                },
                "facts": {
                    "type": "array",
                    "description": "Known facts about the problem",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "evidence": {"type": "string"},
                            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        },
                        "required": ["content"],
                    },
                },
                "questions": {
                    "type": "array",
                    "description": "Key questions to investigate",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "question_type": {"type": "string", "enum": ["pivot", "informational"]},
                        },
                        "required": ["content"],
                    },
                },
                "sub_problems": {
                    "type": "array",
                    "description": "2-4 sub-problems to explore",
                    "items": {
                        "type": "object",
                        "properties": {
                            "problem": {"type": "string"},
                            "logic": {"type": "string", "enum": ["AND", "OR"]},
                        },
                        "required": ["problem"],
                    },
                },
            },
            "required": ["root_problem", "sub_problems"],
        },
    },
}


RESOLVE_NODE_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "resolve_node",
        "description": (
            "Resolve the current node after exploration. Report state, "
            "facts discovered, questions answered, and optional decomposition."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "enum": [
                        "solvable", "unsolvable", "mitigable",
                        "needs_decision", "needs_experiment",
                        "hypothesis",
                    ],
                    "description": "New state for this node",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                },
                "summary": {
                    "type": "string",
                    "description": "1-2 sentence summary of findings",
                },
                "hypothesis_content": {
                    "type": "string",
                    "description": "Speculative best-guess approach (for brainstorm mode)",
                },
                "new_facts": {
                    "type": "array",
                    "description": "Facts discovered during exploration",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "evidence": {"type": "string"},
                            "source_tool": {"type": "string"},
                            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        },
                        "required": ["content"],
                    },
                },
                "answered_questions": {
                    "type": "array",
                    "description": "Questions that were answered",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "answer": {"type": "string"},
                            "answered_by_tool": {"type": "string"},
                        },
                        "required": ["question", "answer"],
                    },
                },
                "new_sub_problems": {
                    "type": "array",
                    "description": "Further decomposition if needed",
                    "items": {
                        "type": "object",
                        "properties": {
                            "problem": {"type": "string"},
                            "logic": {"type": "string", "enum": ["AND", "OR"]},
                            "questions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "content": {"type": "string"},
                                        "question_type": {"type": "string", "enum": ["pivot", "informational"]},
                                    },
                                    "required": ["content"],
                                },
                            },
                        },
                        "required": ["problem"],
                    },
                },
                "new_constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Constraints discovered during exploration",
                },
                "new_blockers": {
                    "type": "array",
                    "description": "External blockers found",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "blocker_type": {
                                "type": "string",
                                "enum": ["api", "library", "permission", "infra", "unknown"],
                            },
                            "workaround": {"type": "string"},
                        },
                        "required": ["description"],
                    },
                },
                "problem_reformulated": {
                    "type": "string",
                    "description": "Reformulated problem statement based on findings",
                },
                "discard_reason": {
                    "type": "string",
                    "description": "Why this branch was discarded (if state=unsolvable in OR parent)",
                },
            },
            "required": ["state", "confidence", "summary"],
        },
    },
}


SYNTHESIZE_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "synthesize",
        "description": "Produce the final synthesis of the exploration.",
        "parameters": {
            "type": "object",
            "properties": {
                "synthesis": {
                    "type": "string",
                    "description": "Complete synthesis of all findings",
                },
                "recommended_approach": {
                    "type": "string",
                    "description": "Recommended approach based on evidence",
                },
                "risks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Identified risks",
                },
                "unknowns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Remaining unknowns",
                },
            },
            "required": ["synthesis", "recommended_approach"],
        },
    },
}


# ── Prompt builders ──────────────────────────────────────────────────────────


def build_init_prompt(problem: str) -> str:
    """Build the user prompt for the INIT phase (tree decomposition)."""
    return (
        f"<task>\n{problem}\n</task>\n\n"
        "<instructions>\n"
        "Analyze this problem and decompose it into an exploration tree.\n"
        "Call `init_tree` with:\n"
        "- root_problem: the problem reformulated for clarity\n"
        "- logic: AND (all sub-problems needed) or OR (any path works)\n"
        "- facts: what is already known (with evidence if available)\n"
        "- questions: key questions to investigate (mark pivots)\n"
        "- sub_problems: 2-4 concrete sub-problems to explore\n\n"
        "Keep sub-problems specific and actionable. Each should be explorable "
        "with file reads, code searches, command execution, or web lookups.\n"
        "</instructions>"
    )


def build_explore_prompt(
    problem: str,
    tree: TreeState,
    node: TreeNode,
) -> str:
    """Build the user prompt for the EXPLORE phase (investigate a node)."""
    parts: list[str] = []

    # Original task
    parts.append(f"<task>\n{problem}\n</task>")

    # Tree path visualization
    path = tree.get_path_to_node(node.id)
    path_lines: list[str] = []
    for i, p in enumerate(path):
        indent = "  " * i
        marker = " <-- YOU ARE HERE" if p.id == node.id else ""
        path_lines.append(f"{indent}{p.id}: {p.problem_statement} [{p.state}]{marker}")
    parts.append(f"<tree-path>\n{chr(10).join(path_lines)}\n</tree-path>")

    # Inherited facts from resolved siblings
    siblings = tree.get_siblings(node.id)
    inherited_facts: list[str] = []
    for sib in siblings:
        if sib.is_resolved():
            for fact in sib.collect_inheritable_facts():
                inherited_facts.append(
                    f"- [{sib.id}] {fact.content}"
                    + (f" (evidence: {fact.evidence[:100]})" if fact.evidence else "")
                )
    if inherited_facts:
        parts.append(
            "<inherited-facts>\n"
            "Facts from resolved sibling nodes:\n"
            + "\n".join(inherited_facts)
            + "\n</inherited-facts>"
        )

    # Sibling state
    if siblings:
        sib_lines = [f"- {s.id}: {s.problem_statement} [{s.state}]" for s in siblings]
        parts.append(
            "<sibling-state>\n"
            + "\n".join(sib_lines)
            + "\n</sibling-state>"
        )

    # Current node details
    node_parts: list[str] = [f"Problem: {node.problem_statement}"]
    if node.problem_reformulated:
        node_parts.append(f"Reformulated: {node.problem_reformulated}")
    if node.facts:
        node_parts.append("Known facts:")
        for f in node.facts:
            node_parts.append(f"  - {f.content} [{f.confidence}]")
    if node.questions:
        node_parts.append("Open questions:")
        for q in node.questions:
            status = "ANSWERED" if q.answer else "OPEN"
            prefix = "PIVOT" if q.question_type == "pivot" else "info"
            node_parts.append(f"  - [{prefix}][{status}] {q.content}")
            if q.answer:
                node_parts.append(f"    Answer: {q.answer}")
    if node.constraints:
        node_parts.append("Constraints: " + "; ".join(node.constraints))

    parts.append(
        "<current-node>\n"
        + "\n".join(node_parts)
        + "\n</current-node>"
    )

    # Instructions
    parts.append(
        "<instructions>\n"
        "Explore this sub-problem using available tools to gather evidence.\n"
        "When done, call `resolve_node` with your findings.\n\n"
        "Guidelines:\n"
        "- Use tools to verify assumptions — don't guess\n"
        "- Record new facts with evidence from tool output\n"
        "- Answer open questions where possible\n"
        "- If the problem needs further decomposition, include new_sub_problems\n"
        "- If blocked by external factors, report them as new_blockers\n"
        "</instructions>"
    )

    return "\n\n".join(parts)


def build_synthesize_prompt(problem: str, tree: TreeState) -> str:
    """Build the user prompt for the SYNTHESIZE phase (final synthesis)."""
    parts: list[str] = []

    parts.append(f"<task>\n{problem}\n</task>")

    # Render the complete tree
    tree_lines: list[str] = []
    if tree.root:
        _render_tree(tree.root, tree_lines, indent=0)
    parts.append(
        "<exploration-tree>\n"
        + "\n".join(tree_lines)
        + "\n</exploration-tree>"
    )

    parts.append(
        "<instructions>\n"
        "Synthesize the findings from the exploration tree.\n"
        "Call `synthesize` with:\n"
        "- synthesis: comprehensive summary of what was found\n"
        "- recommended_approach: the best path forward based on evidence\n"
        "- risks: identified risks and caveats\n"
        "- unknowns: remaining questions that couldn't be answered\n\n"
        "Base your synthesis on evidence, not speculation. Reference specific "
        "nodes and facts from the tree.\n"
        "</instructions>"
    )

    return "\n\n".join(parts)


def _render_tree(node: TreeNode, lines: list[str], indent: int) -> None:
    """Recursively render a tree node into text lines."""
    prefix = "  " * indent
    state_marker = f"[{node.state}]"
    conf_marker = f"({node.confidence})"
    logic_marker = f"[{node.logic}]" if node.children else ""

    lines.append(
        f"{prefix}{node.id}: {node.problem_statement} "
        f"{state_marker} {conf_marker} {logic_marker}".rstrip()
    )

    if node.exploration_summary:
        lines.append(f"{prefix}  Summary: {node.exploration_summary}")

    if node.facts:
        for fact in node.facts:
            src = f" (via {fact.source_tool})" if fact.source_tool else ""
            lines.append(f"{prefix}  Fact [{fact.confidence}]: {fact.content}{src}")

    if node.constraints:
        lines.append(f"{prefix}  Constraints: {'; '.join(node.constraints)}")

    if node.external_blockers:
        for b in node.external_blockers:
            lines.append(f"{prefix}  Blocker [{b.blocker_type}]: {b.description}")

    if node.discard_reason:
        lines.append(f"{prefix}  Discarded: {node.discard_reason}")

    for child in node.children:
        _render_tree(child, lines, indent + 1)


def build_tree_system_prompt(
    backstory: str,
    *,
    identity_override: str | None = None,
    session_summaries: list[str] | None = None,
) -> str:
    """Build the system prompt for tree engine phases.

    Uses the explore flow identity + TREE_PROTOCOL instead of LOOP_PROTOCOL.
    """
    from infinidev.prompts.flows.explore import EXPLORE_IDENTITY

    parts: list[str] = [identity_override if identity_override else EXPLORE_IDENTITY]

    if session_summaries:
        numbered = "\n".join(
            f"{i+1}. {s}" for i, s in enumerate(session_summaries)
        )
        parts.append(f"<session-context>\n{numbered}\n</session-context>")

    parts.append(TREE_PROTOCOL)

    return "\n\n".join(parts)
