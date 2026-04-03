"""Integration tests for the exploration tree engine with mocked LLM."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from infinidev.engine.tree.engine import TreeEngine
from infinidev.engine.tree.models import (
    TreeNode,
    TreeState,
    propagate,
    select_next_node,
)


def _make_agent():
    """Create a minimal mock agent."""
    agent = MagicMock()
    agent.agent_id = "test_agent"
    agent.project_id = 1
    agent.backstory = "Test backstory"
    agent.tools = []
    agent._system_prompt_identity = None
    agent._session_summaries = None
    agent._session_id = None
    agent._tech_hints = None
    return agent


def _mock_response(tool_calls=None, content="", usage_tokens=100):
    """Create a mock LLM response."""
    response = MagicMock()
    response.usage = MagicMock()
    response.usage.total_tokens = usage_tokens
    response.usage.prompt_tokens = usage_tokens // 2
    response.usage.completion_tokens = usage_tokens // 2

    choice = MagicMock()
    choice.message.content = content

    if tool_calls:
        tc_objects = []
        for i, tc in enumerate(tool_calls):
            tc_obj = MagicMock()
            tc_obj.id = f"call_{i}"
            tc_obj.function.name = tc["name"]
            tc_obj.function.arguments = (
                json.dumps(tc["arguments"]) if isinstance(tc["arguments"], dict)
                else tc["arguments"]
            )
            tc_objects.append(tc_obj)
        choice.message.tool_calls = tc_objects
    else:
        choice.message.tool_calls = None

    response.choices = [choice]
    return response


def _patch_engine():
    """Return a combined patch context for the tree engine dependencies."""
    caps = MagicMock()
    caps.supports_function_calling = True
    caps.supports_tool_choice_required = True
    caps.supports_json_mode = False

    patches = {
        "llm_params": patch(
            "infinidev.engine.tree.engine.get_litellm_params",
            return_value={"model": "test"},
            create=True,
        ),
        "caps": patch(
            "infinidev.engine.tree.engine.get_model_capabilities",
            return_value=caps,
            create=True,
        ),
        "store": patch(
            "infinidev.engine.tree.engine.store_exploration_tree",
            create=True,
        ),
        "call_llm": patch("infinidev.engine.tree.engine._call_llm"),
    }
    return patches


# ── Explore mode integration tests ──────────────────────────────────────────


class TestTreeEngineInit:
    """Test the INIT phase of tree engine."""

    def test_basic_init_and_synthesize(self):
        """Test that a simple problem gets decomposed and synthesized."""
        patches = _patch_engine()

        with patches["llm_params"], patches["caps"], patches["store"] as mock_store, \
             patches["call_llm"] as mock_llm:

            mock_llm.side_effect = [
                # Init
                _mock_response(tool_calls=[{
                    "name": "init_tree",
                    "arguments": {
                        "root_problem": "How to implement hot-reload",
                        "logic": "AND",
                        "facts": [{"content": "Project uses Python", "confidence": "high"}],
                        "questions": [{"content": "Does it use importlib?", "question_type": "pivot"}],
                        "sub_problems": [
                            {"problem": "Detect file changes", "logic": "AND"},
                            {"problem": "Reload modules safely", "logic": "OR"},
                        ],
                    },
                }]),
                # Explore child 1.1
                _mock_response(tool_calls=[{
                    "name": "resolve_node",
                    "arguments": {
                        "state": "solvable",
                        "confidence": "high",
                        "summary": "Use watchdog for file change detection",
                        "new_facts": [{"content": "watchdog library available", "source_tool": "web_search"}],
                    },
                }]),
                # Explore child 1.2
                _mock_response(tool_calls=[{
                    "name": "resolve_node",
                    "arguments": {
                        "state": "mitigable",
                        "confidence": "medium",
                        "summary": "importlib.reload works but has caveats with state",
                        "new_constraints": ["Module-level state is lost on reload"],
                    },
                }]),
                # Synthesize
                _mock_response(tool_calls=[{
                    "name": "synthesize",
                    "arguments": {
                        "synthesis": "Hot-reload is feasible using watchdog + importlib.reload",
                        "recommended_approach": "Use watchdog for detection, importlib.reload with state preservation",
                        "risks": ["Module state loss", "Circular imports"],
                        "unknowns": ["Performance impact on large modules"],
                    },
                }]),
            ]

            engine = TreeEngine()
            result = engine.execute(
                _make_agent(),
                task_prompt=("How to implement hot-reload?", "Analysis"),
            )

            assert "Hot-reload is feasible" in result
            assert "Recommended Approach" in result
            assert mock_store.called


class TestTreeEngineDecomposition:
    """Test further decomposition during exploration."""

    def test_node_creates_sub_problems(self):
        """Test that resolve_node can create new sub-problems."""
        patches = _patch_engine()

        with patches["llm_params"], patches["caps"], patches["store"], \
             patches["call_llm"] as mock_llm:

            mock_llm.side_effect = [
                # Init -- single child
                _mock_response(tool_calls=[{
                    "name": "init_tree",
                    "arguments": {
                        "root_problem": "Complex problem",
                        "sub_problems": [{"problem": "Investigate X"}],
                    },
                }]),
                # Explore 1.1 -> needs decomposition
                _mock_response(tool_calls=[{
                    "name": "resolve_node",
                    "arguments": {
                        "state": "solvable",
                        "confidence": "medium",
                        "summary": "X has two parts",
                        "new_sub_problems": [
                            {"problem": "Part A of X"},
                            {"problem": "Part B of X"},
                        ],
                    },
                }]),
                # Explore 1.1.1
                _mock_response(tool_calls=[{
                    "name": "resolve_node",
                    "arguments": {
                        "state": "solvable",
                        "confidence": "high",
                        "summary": "Part A solved",
                    },
                }]),
                # Explore 1.1.2
                _mock_response(tool_calls=[{
                    "name": "resolve_node",
                    "arguments": {
                        "state": "solvable",
                        "confidence": "high",
                        "summary": "Part B solved",
                    },
                }]),
                # Synthesize
                _mock_response(tool_calls=[{
                    "name": "synthesize",
                    "arguments": {
                        "synthesis": "Both parts solved",
                        "recommended_approach": "Do A then B",
                    },
                }]),
            ]

            engine = TreeEngine()
            result = engine.execute(
                _make_agent(),
                task_prompt=("Complex problem", "Analysis"),
            )

            assert "Both parts solved" in result


class TestTreeEngineOR:
    """Test OR logic exploration."""

    def test_or_logic_first_solvable_wins(self):
        """Test that OR logic resolves root when first child is solvable."""
        patches = _patch_engine()

        with patches["llm_params"], patches["caps"], patches["store"], \
             patches["call_llm"] as mock_llm:

            mock_llm.side_effect = [
                # Init -- OR with 2 alternatives
                _mock_response(tool_calls=[{
                    "name": "init_tree",
                    "arguments": {
                        "root_problem": "Find approach",
                        "logic": "OR",
                        "sub_problems": [
                            {"problem": "Approach A"},
                            {"problem": "Approach B"},
                        ],
                    },
                }]),
                # Explore 1.1 -> solvable
                _mock_response(tool_calls=[{
                    "name": "resolve_node",
                    "arguments": {
                        "state": "solvable",
                        "confidence": "high",
                        "summary": "Approach A works",
                    },
                }]),
                # Synthesize (root resolved via OR after first child)
                _mock_response(tool_calls=[{
                    "name": "synthesize",
                    "arguments": {
                        "synthesis": "Approach A is viable",
                        "recommended_approach": "Use approach A",
                    },
                }]),
            ]

            engine = TreeEngine()
            result = engine.execute(
                _make_agent(),
                task_prompt=("Find approach", "Analysis"),
            )

            assert "Approach A" in result
            assert mock_llm.call_count == 3


class TestTreeEngineToolCalls:
    """Test that regular tool calls work during exploration."""

    def test_tool_calls_before_resolve(self):
        """Test that regular tools execute before resolve_node."""
        patches = _patch_engine()

        with patches["llm_params"], patches["caps"], patches["store"], \
             patches["call_llm"] as mock_llm, \
             patch("infinidev.engine.tree.engine.execute_tool_call", return_value="file content here"):

            mock_llm.side_effect = [
                # Init
                _mock_response(tool_calls=[{
                    "name": "init_tree",
                    "arguments": {
                        "root_problem": "Check codebase",
                        "sub_problems": [{"problem": "Read main.py"}],
                    },
                }]),
                # Explore 1.1 -- first call reads a file
                _mock_response(tool_calls=[{
                    "name": "read_file",
                    "arguments": {"file_path": "src/main.py"},
                }]),
                # Then resolve
                _mock_response(tool_calls=[{
                    "name": "resolve_node",
                    "arguments": {
                        "state": "solvable",
                        "confidence": "high",
                        "summary": "Found the entry point",
                        "new_facts": [{
                            "content": "Entry point is main()",
                            "evidence": "file content here",
                            "source_tool": "read_file",
                        }],
                    },
                }]),
                # Synthesize
                _mock_response(tool_calls=[{
                    "name": "synthesize",
                    "arguments": {
                        "synthesis": "Entry point found",
                        "recommended_approach": "Start from main()",
                    },
                }]),
            ]

            engine = TreeEngine()
            result = engine.execute(
                _make_agent(),
                task_prompt=("Check codebase", "Analysis"),
            )

            assert "Entry point found" in result


class TestExploreSubproblem:
    """Test the convenience method for loop engine integration."""

    def test_explore_subproblem(self):
        patches = _patch_engine()

        with patches["llm_params"], patches["caps"], patches["store"], \
             patches["call_llm"] as mock_llm:

            mock_llm.side_effect = [
                _mock_response(tool_calls=[{
                    "name": "init_tree",
                    "arguments": {
                        "root_problem": "Sub-problem from loop",
                        "sub_problems": [{"problem": "Part 1"}],
                    },
                }]),
                _mock_response(tool_calls=[{
                    "name": "resolve_node",
                    "arguments": {
                        "state": "solvable",
                        "confidence": "high",
                        "summary": "Solved",
                    },
                }]),
                _mock_response(tool_calls=[{
                    "name": "synthesize",
                    "arguments": {
                        "synthesis": "Sub-problem resolved",
                        "recommended_approach": "Do X",
                    },
                }]),
            ]

            engine = TreeEngine()
            result = engine.explore_subproblem(_make_agent(), "Sub-problem from loop")
            assert "Sub-problem resolved" in result


# ── Brainstorm mode integration tests ────────────────────────────────────────


class TestBrainstormMode:
    """Test the brainstorm execution path with mocked LLM."""

    def test_brainstorm_full_flow(self):
        """Test anti-pattern -> diverge -> explore -> cross -> converge."""
        patches = _patch_engine()

        with patches["llm_params"], patches["caps"], patches["store"], \
             patches["call_llm"] as mock_llm:

            mock_llm.side_effect = [
                # Phase 0: Anti-pattern
                _mock_response(tool_calls=[{
                    "name": "identify_obvious",
                    "arguments": {
                        "obvious_solutions": [
                            {"approach": "Use a cache", "why_obvious": "First thing everyone suggests"},
                            {"approach": "Add more servers", "why_obvious": "Throw money at it"},
                        ],
                        "assumptions": ["More resources = better performance"],
                    },
                }]),
                # Phase 1: Diverge (5 perspectives, but we mock responses for each)
                # Perspective 1
                _mock_response(tool_calls=[{
                    "name": "propose_idea",
                    "arguments": {
                        "idea_title": "Lazy evaluation pipeline",
                        "description": "Only compute what's needed at render time",
                        "perspective_used": "Inversion",
                        "novelty_claim": "Instead of pre-computing everything, compute nothing until asked",
                    },
                }]),
                # Perspective 2
                _mock_response(tool_calls=[{
                    "name": "propose_idea",
                    "arguments": {
                        "idea_title": "Event-sourced state",
                        "description": "Store events, replay to build state",
                        "perspective_used": "Cross-Domain Analogy",
                        "novelty_claim": "Borrowed from banking/ledger systems",
                    },
                }]),
                # Perspective 3
                _mock_response(tool_calls=[{
                    "name": "propose_idea",
                    "arguments": {
                        "idea_title": "Single shared buffer",
                        "description": "One mutable buffer everyone writes to",
                        "perspective_used": "Stupidest Thing That Works",
                        "novelty_claim": "Eliminates all allocation overhead",
                    },
                }]),
                # Perspective 4
                _mock_response(tool_calls=[{
                    "name": "propose_idea",
                    "arguments": {
                        "idea_title": "Remove the database",
                        "description": "Keep everything in-memory with snapshots",
                        "perspective_used": "Eliminate the Essential",
                        "novelty_claim": "Question whether persistence is needed at all",
                    },
                }]),
                # Perspective 5
                _mock_response(tool_calls=[{
                    "name": "propose_idea",
                    "arguments": {
                        "idea_title": "Constraint as feature",
                        "description": "Use the 100ms budget as a design constraint",
                        "perspective_used": "Constraint Removal",
                        "novelty_claim": "Turn latency limit into an architectural guide",
                    },
                }]),
                # Phase 2: Explore each idea (5 resolve_node calls)
                _mock_response(tool_calls=[{
                    "name": "resolve_node",
                    "arguments": {
                        "state": "hypothesis",
                        "confidence": "medium",
                        "summary": "Lazy eval is promising but needs benchmark",
                        "hypothesis_content": "Lazy pipeline could cut 60% of compute",
                    },
                }]),
                _mock_response(tool_calls=[{
                    "name": "resolve_node",
                    "arguments": {
                        "state": "solvable",
                        "confidence": "high",
                        "summary": "Event sourcing works well for this use case",
                    },
                }]),
                _mock_response(tool_calls=[{
                    "name": "resolve_node",
                    "arguments": {
                        "state": "mitigable",
                        "confidence": "low",
                        "summary": "Shared buffer has race condition risks",
                    },
                }]),
                _mock_response(tool_calls=[{
                    "name": "resolve_node",
                    "arguments": {
                        "state": "unsolvable",
                        "confidence": "high",
                        "summary": "Can't remove DB due to compliance requirements",
                    },
                }]),
                _mock_response(tool_calls=[{
                    "name": "resolve_node",
                    "arguments": {
                        "state": "solvable",
                        "confidence": "medium",
                        "summary": "100ms budget works as architectural guide",
                    },
                }]),
                # Phase 3: Cross (top 2 ideas)
                _mock_response(tool_calls=[{
                    "name": "cross_ideas",
                    "arguments": {
                        "idea_a_id": "1.2",
                        "idea_b_id": "1.5",
                        "hybrid_title": "Event-sourced with time budget",
                        "hybrid_description": "Event sourcing with 100ms replay window",
                        "what_from_a": "Event sourcing architecture",
                        "what_from_b": "Time budget as design constraint",
                        "why_better": "Combines proven pattern with performance guarantee",
                    },
                }]),
                # Phase 3b: Explore hybrid
                _mock_response(tool_calls=[{
                    "name": "resolve_node",
                    "arguments": {
                        "state": "solvable",
                        "confidence": "high",
                        "summary": "Hybrid approach validated with benchmarks",
                    },
                }]),
                # Phase 4: Converge
                _mock_response(tool_calls=[{
                    "name": "rank_ideas",
                    "arguments": {
                        "ranked_ideas": [
                            {
                                "node_id": "1.6",
                                "rank": 1,
                                "idea_title": "Event-sourced with time budget",
                                "novelty_score": 4,
                                "feasibility_score": 5,
                                "completeness_score": 4,
                                "justification": "Best of both worlds",
                            },
                            {
                                "node_id": "1.2",
                                "rank": 2,
                                "idea_title": "Event-sourced state",
                                "novelty_score": 3,
                                "feasibility_score": 5,
                                "completeness_score": 4,
                                "justification": "Proven pattern",
                            },
                        ],
                        "synthesis": "The brainstorm produced a novel hybrid approach",
                        "surprise_finding": "Time budget as architectural guide was unexpected",
                    },
                }]),
            ]

            engine = TreeEngine()
            result = engine.execute(
                _make_agent(),
                task_prompt=("Optimize API response time", "Creative approaches"),
                mode="brainstorm",
            )

            assert "Brainstorm Results" in result
            assert "Ranked Ideas" in result
            assert "Event-sourced with time budget" in result
            assert "Surprise Finding" in result

    def test_brainstorm_subproblem_convenience(self):
        """Test brainstorm_subproblem convenience method calls execute with mode=brainstorm."""
        engine = TreeEngine()
        with patch.object(engine, "execute", return_value="ideas") as mock_exec:
            result = engine.brainstorm_subproblem(_make_agent(), "test problem")
            assert result == "ideas"
            mock_exec.assert_called_once()
            call_kwargs = mock_exec.call_args
            assert call_kwargs.kwargs["mode"] == "brainstorm"


# ── Unit tests for model logic ───────────────────────────────────────────────


class TestORLogicPropagation:
    """Test OR logic propagation on TreeNode."""

    def test_or_logic_propagates_best(self):
        """OR logic: best state propagates up."""
        parent = TreeNode(
            id="1",
            problem_statement="Test OR logic",
            logic="OR",
            depth=0,
        )
        child1 = TreeNode(
            id="1.1",
            problem_statement="Subproblem 1",
            state="solvable",
            confidence="high",
            depth=1,
        )
        child2 = TreeNode(
            id="1.2",
            problem_statement="Subproblem 2",
            state="unsolvable",
            confidence="low",
            depth=1,
        )
        parent.children = [child1, child2]
        propagate(parent)
        assert parent.state == "solvable"
        assert parent.confidence == "high"

    def test_hypothesis_ranks_below_mitigable(self):
        """In OR logic, mitigable should rank above hypothesis."""
        parent = TreeNode(
            id="1",
            problem_statement="Test ranking",
            logic="OR",
            depth=0,
        )
        hypo_child = TreeNode(
            id="1.1",
            problem_statement="Hypothesis approach",
            state="hypothesis",
            confidence="low",
            depth=1,
        )
        mitigable_child = TreeNode(
            id="1.2",
            problem_statement="Mitigable approach",
            state="mitigable",
            confidence="medium",
            depth=1,
        )
        parent.children = [hypo_child, mitigable_child]
        propagate(parent)
        assert parent.state == "mitigable"
        assert parent.confidence == "medium"


class TestTreeNodeHypothesis:
    """Tests for hypothesis state on TreeNode."""

    def test_hypothesis_is_resolved(self):
        """Hypothesis state counts as resolved."""
        node = TreeNode(id="1.1", problem_statement="Test", state="hypothesis", depth=1)
        assert node.is_resolved() is True

    def test_hypothesis_content_stored(self):
        """hypothesis_content field can be set on TreeNode."""
        node = TreeNode(
            id="1.1",
            problem_statement="Test",
            state="hypothesis",
            hypothesis_content="Maybe the auth module uses JWT",
            depth=1,
        )
        assert node.hypothesis_content == "Maybe the auth module uses JWT"


# ── Brainstorm context unit tests ────────────────────────────────────────────


class TestBrainstormContext:
    """Tests for brainstorm_context.py prompt/schema utilities."""

    def test_select_perspectives_returns_n(self):
        from infinidev.engine.tree.brainstorm_context import select_perspectives
        result = select_perspectives(5)
        assert len(result) == 5

    def test_select_perspectives_includes_anchors(self):
        from infinidev.engine.tree.brainstorm_context import select_perspectives
        result = select_perspectives(3)
        ids = [p["id"] for p in result]
        assert "inversion" in ids
        assert "stupid_solution" in ids

    def test_select_perspectives_resolves_templates(self):
        from infinidev.engine.tree.brainstorm_context import select_perspectives
        result = select_perspectives(10)
        for p in result:
            assert "{domain}" not in p["prompt"]
            assert "{persona}" not in p["prompt"]
            assert "{concept}" not in p["prompt"]

    def test_get_random_oblique_returns_string(self):
        from infinidev.engine.tree.brainstorm_context import get_random_oblique
        result = get_random_oblique()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_build_anti_pattern_prompt(self):
        from infinidev.engine.tree.brainstorm_context import build_anti_pattern_prompt
        prompt = build_anti_pattern_prompt("Optimize API")
        assert "Optimize API" in prompt
        assert "identify_obvious" in prompt
        assert "BANNED" in prompt or "OBVIOUS" in prompt

    def test_build_diverge_prompt_includes_banned(self):
        from infinidev.engine.tree.brainstorm_context import build_diverge_prompt
        perspective = {"id": "test", "name": "Test Lens", "prompt": "Try something"}
        anti_patterns = [{"approach": "Use cache", "why_obvious": "generic"}]
        prompt = build_diverge_prompt("problem", perspective, anti_patterns, "oblique hint")
        assert "Use cache" in prompt
        assert "Test Lens" in prompt
        assert "oblique hint" in prompt

    def test_build_converge_prompt(self):
        from infinidev.engine.tree.brainstorm_context import build_converge_prompt
        tree = TreeState()
        tree.root = TreeNode(id="1", problem_statement="test", depth=0)
        prompt = build_converge_prompt("problem", tree, [])
        assert "rank_ideas" in prompt
        assert "novelty_score" in prompt
