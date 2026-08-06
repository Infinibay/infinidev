"""Semantic contracts shared by Infinidev's specialized role prompts."""

from __future__ import annotations

from infinidev.prompts.analyst.planner_prompt import ANALYST_PLANNER_SYSTEM_PROMPT
from infinidev.prompts.analyst.stage_planner_prompt import STAGE_PLANNER_SYSTEM_PROMPT
from infinidev.prompts.analyst.task_planner_prompt import TASK_PLANNER_SYSTEM_PROMPT
from infinidev.prompts.chat_agent.system import CHAT_AGENT_SYSTEM_PROMPT_TEMPLATE
from infinidev.prompts.flows.brainstorm import BRAINSTORM_IDENTITY
from infinidev.prompts.flows.document import DOCUMENT_IDENTITY
from infinidev.prompts.flows.explore import EXPLORE_IDENTITY
from infinidev.prompts.flows.sysadmin import SYSADMIN_IDENTITY
from infinidev.prompts.reviewer.judge_system import JUDGE_SYSTEM_PROMPT
from infinidev.prompts.variants import get_variant


def test_chat_router_models_authority_and_ambiguous_targets() -> None:
    prompt = CHAT_AGENT_SYSTEM_PROMPT_TEMPLATE

    assert "Future or conditional approval is not current permission" in prompt
    assert "Do not choose a candidate" in prompt
    assert "broadened to all candidates" in prompt


def test_planner_does_not_promote_defaults_to_requirements() -> None:
    prompt = ANALYST_PLANNER_SYSTEM_PROMPT

    assert "Keep literal requirements separate from defaults" in prompt
    assert "cannot become a Goal or Task acceptance condition" in prompt
    assert "replace the singular target with every candidate" in prompt


def test_stage_and_task_planners_share_authority_vocabulary() -> None:
    for prompt in (STAGE_PLANNER_SYSTEM_PROMPT, TASK_PLANNER_SYSTEM_PROMPT):
        assert "A **Goal** is the user-owned outcome" in prompt
        assert "``USER_LITERAL``" in prompt
        assert "``DERIVED``" in prompt
        assert "``OBSERVED_EVIDENCE``" in prompt
        assert "Derived material" in prompt
        assert "cannot expand the Goal" in prompt


def test_stage_planner_separates_goal_closure_from_next_horizon() -> None:
    prompt = STAGE_PLANNER_SYSTEM_PROMPT

    assert "Judge the Goal's finish separately from the distance to it" in prompt
    assert "A Goal with a decidable finish can still require many Stages" in prompt
    assert "A completed plan, an empty queue or a confident assessment" in prompt
    assert "one Stage can cover it" in prompt
    assert "leave that later work out of the current Stage" in prompt
    assert "do not pre-plan Tasks for a later Stage" in prompt
    assert "Do not create ceremonial Tasks to fill a count" in prompt


def test_task_planner_treats_steps_as_adaptable_tactics() -> None:
    prompt = TASK_PLANNER_SYSTEM_PROMPT

    assert "Steps are model-inferred tactics" in prompt
    assert "add, revise, reorder or remove them" in prompt
    assert "Step count follows those evidence boundaries" in prompt
    assert "A planner-authored command is untrusted model output" in prompt
    assert "Its paths, test names and behavior are not evidence" in prompt


def test_document_prompt_uses_examples_for_reader_tasks() -> None:
    assert "where they help the intended" in DOCUMENT_IDENTITY
    assert "Always include examples" not in DOCUMENT_IDENTITY
    assert "label the gap instead of inventing content" in DOCUMENT_IDENTITY


def test_sysadmin_does_not_request_duplicate_authorization() -> None:
    assert "Do not ask twice" in SYSADMIN_IDENTITY
    assert "dangerous or newly discovered external effects" in SYSADMIN_IDENTITY
    assert "NEVER do these without EXPLICIT user approval" in SYSADMIN_IDENTITY


def test_explore_separates_observed_facts_from_hypotheses() -> None:
    assert "consequential factual claims" in EXPLORE_IDENTITY
    assert "Label hypotheses and" in EXPLORE_IDENTITY


def test_brainstorm_does_not_optimize_for_novelty_by_default() -> None:
    assert "Establish the baseline" in BRAINSTORM_IDENTITY
    assert "Novelty is an exploration axis, not an acceptance criterion" in BRAINSTORM_IDENTITY


def test_reviewer_severity_table_matches_its_authority_rules() -> None:
    blocking_row = next(
        line for line in JUDGE_SYSTEM_PROMPT.splitlines() if line.startswith("| **Blocking**")
    )

    assert "Unmet original requirement" in blocking_row
    assert "missing plan steps" not in blocking_row
    assert "report discrepancies" not in blocking_row


def test_research_variants_use_evidence_gaps_instead_of_confidence_scores() -> None:
    generalized = get_variant("flow.research.identity", "generalized")
    coding = get_variant("flow.research.identity", "coding")

    assert generalized is not None
    assert coding is not None
    assert "Do not use confidence as a substitute for missing evidence" in generalized
    assert "state_evidence_gaps_and_uncertainty" in coding
    assert "cross_reference(min_sources=2)" not in coding
