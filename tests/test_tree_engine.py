"""Integration tests for the exploration tree engine with mocked LLM."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from infinidev.engine.tree_engine import TreeEngine
from infinidev.engine.tree_models import TreeState


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
            "infinidev.engine.tree_engine.get_litellm_params",
            return_value={"model": "test"},
            create=True,
        ),
        "caps": patch(
            "infinidev.engine.tree_engine.get_model_capabilities",
            return_value=caps,
            create=True,
        ),
        "store": patch(
            "infinidev.engine.tree_engine.store_exploration_tree",
            create=True,
        ),
        "call_llm": patch("infinidev.engine.tree_engine._call_llm"),
    }
    return patches


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
                # Init — single child
                _mock_response(tool_calls=[{
                    "name": "init_tree",
                    "arguments": {
                        "root_problem": "Complex problem",
                        "sub_problems": [{"problem": "Investigate X"}],
                    },
                }]),
                # Explore 1.1 → needs decomposition
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
                # Init — OR with 2 alternatives
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
                # Explore 1.1 → solvable
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
             patch("infinidev.engine.tree_engine.execute_tool_call", return_value="file content here"):

            mock_llm.side_effect = [
                # Init
                _mock_response(tool_calls=[{
                    "name": "init_tree",
                    "arguments": {
                        "root_problem": "Check codebase",
                        "sub_problems": [{"problem": "Read main.py"}],
                    },
                }]),
                # Explore 1.1 — first call reads a file
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
