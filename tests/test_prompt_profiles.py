"""Regression tests for project-local JSON prompt profiles."""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from infinidev.prompts.analyst.stage_planner_prompt import (
    build_stage_planner_system_prompt,
)
from infinidev.prompts.analyst.task_planner_prompt import (
    build_task_planner_system_prompt,
)
from infinidev.prompts.profiles import (
    EffectivePromptConfiguration,
    PromptProfileError,
    apply_prompt_profile,
    resolve_prompt_profile,
)


def _configuration(tmp_path, document: dict) -> EffectivePromptConfiguration:
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return EffectivePromptConfiguration.compile(path)


_FIXED_PROFILE_IDS = {
    "loop.identity",
    "loop.protocol",
    "loop.behavior_guidelines",
    "loop.technology_guidance",
    "loop.project_instructions",
    "loop.critic_guidance",
    "loop.session_context",
    "iteration.smart_summary",
    "iteration.project_knowledge",
    "iteration.context_corpus",
    "iteration.context_rank",
    "iteration.workspace",
    "iteration.background_completions",
    "iteration.background_tasks",
    "iteration.reactive_guidance",
    "iteration.opened_files",
    "iteration.session_notes",
    "iteration.working_notes",
    "iteration.note_nudge",
    "iteration.previous_actions",
    "iteration.anti_patterns",
    "iteration.behavior_summary",
    "iteration.next_actions",
    "iteration.context_budget",
    "task_planner.identity",
    "task_planner.methodology",
    "task_planner.planning_vocabulary",
    "task_planner.handoff_guidance",
    "task_planner.decomposition_guidance",
    "task_planner.verification_guidance",
    "task_planner.examples",
    "stage_planner.identity",
    "stage_planner.methodology",
    "stage_planner.planning_vocabulary",
    "stage_planner.authority_guidance",
    "stage_planner.horizon_guidance",
    "stage_planner.decision_guidance",
    "stage_planner.decomposition_guidance",
    "stage_planner.examples",
    "reviewer.identity",
    "reviewer.input_guidance",
    "reviewer.authority_guidance",
    "reviewer.evaluation_guidance",
    "reviewer.severity_guidance",
    "extractor.identity",
    "judge.identity",
    "judge.input_guidance",
    "judge.authority_guidance",
    "judge.evaluation_guidance",
    "judge.severity_guidance",
    "evidence.identity",
    "evidence.evaluation_guidance",
    "adversarial.identity",
    "adversarial.evaluation_guidance",
    "chat.identity",
    "chat.language_guidance",
    "chat.council_guidance",
    "chat.followup_guidance",
    "chat.project_instructions",
    "chat.model_guidance",
    "council.seed_identity",
    "council.member_identity",
    "council.judge_identity",
    "council.synthesis_identity",
    "council.language_guidance",
    "council.persona_palette",
    "gather.identity_guidance",
    "gather.classifier_guidance",
    "gather.synthesis_guidance",
    "gather.question_guidance",
    "summary.step_guidance",
}


def test_documented_catalog_enumerates_every_supported_profile_id() -> None:
    documentation = (
        Path(__file__).parents[1] / "docs" / "prompt-profiles.md"
    ).read_text(encoding="utf-8")
    prefixes = (
        "loop|iteration|task_planner|stage_planner|reviewer|extractor|judge|"
        "evidence|adversarial|chat|council|gather|summary|phase"
    )
    documented = set(re.findall(rf"`(({prefixes})\.[a-z_.]+)`", documentation))
    documented_ids = {profile_id for profile_id, _prefix in documented}
    phase_ids = {
        f"phase.{task_type}.{phase}{suffix}"
        for task_type in ("bug", "feature", "refactor", "other", "sysadmin")
        for phase in ("investigate", "plan", "execute")
        for suffix in ("", "_identity")
    }
    expected = _FIXED_PROFILE_IDS | phase_ids

    assert len(expected) == 101
    assert documented_ids == expected


def test_missing_profile_keeps_fragment_enabled(tmp_path) -> None:
    assert apply_prompt_profile("base prompt", "develop", "loop.identity") == "base prompt"
    assert resolve_prompt_profile("develop", "loop.identity", tmp_path / "missing.json").enabled


def test_phase_profile_can_disable_a_prompt(tmp_path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps({"develop": {"loop.identity": False}}), encoding="utf-8")

    assert resolve_prompt_profile("develop", "loop.identity", path).enabled is False


def test_exact_model_profile_precedes_general_phase(tmp_path, monkeypatch) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps(
            {
                "develop": {"loop.identity": False},
                "models": {
                    "test-provider/test-model": {
                        "develop": {"loop.identity": {"detail": 2}}
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("infinidev.prompts.profiles.settings.LLM_PROVIDER", "test-provider")
    monkeypatch.setattr("infinidev.prompts.profiles.settings.LLM_MODEL", "test-model")

    profile = resolve_prompt_profile("develop", "loop.identity", path)

    assert profile.enabled
    assert profile.parameters == {"detail": 2}


def test_provider_profile_precedes_general_phase(tmp_path, monkeypatch) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps(
            {
                "develop": {"loop.protocol": False},
                "models": {"test-provider": {"develop": {"loop.protocol": True}}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("infinidev.prompts.profiles.settings.LLM_PROVIDER", "test-provider")
    monkeypatch.setattr("infinidev.prompts.profiles.settings.LLM_MODEL", "other-model")

    configuration = EffectivePromptConfiguration.compile(path)

    assert configuration.resolve("develop", "loop.protocol").enabled is True


def test_loop_identity_profile_disables_the_builtin_fragment(tmp_path, monkeypatch) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps({"develop": {"loop.identity": False}}), encoding="utf-8")
    monkeypatch.setattr("infinidev.prompts.profiles.get_prompt_profile_path", lambda: path)

    from infinidev.engine.loop.context import build_system_prompt

    prompt = build_system_prompt("", workspace_path=str(tmp_path))

    assert "You are Infinidev" not in prompt


def test_compiled_configuration_is_immutable_until_the_next_run(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps({"develop": {"loop.identity": False}}), encoding="utf-8")
    from infinidev.prompts import profiles

    real_loader = profiles.load_prompt_profiles
    reads = 0

    def tracked_loader(profile_path=None):
        nonlocal reads
        reads += 1
        return real_loader(profile_path)

    monkeypatch.setattr(profiles, "load_prompt_profiles", tracked_loader)

    first_run = EffectivePromptConfiguration.compile(path)
    path.write_text(json.dumps({"develop": {"loop.identity": True}}), encoding="utf-8")

    assert first_run.resolve("develop", "loop.identity").enabled is False
    assert first_run.resolve("develop", "loop.identity").enabled is False
    with pytest.raises(TypeError):
        first_run.profiles["develop"]["loop.identity"] = object()
    assert reads == 1

    second_run = EffectivePromptConfiguration.compile(path)

    assert second_run.resolve("develop", "loop.identity").enabled is True
    assert reads == 2


def test_optional_system_blocks_share_the_compiled_configuration(
    tmp_path, monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("PROJECT-MARKER", encoding="utf-8")
    configuration = _configuration(
        tmp_path,
        {
            "develop": {
                "loop.behavior_guidelines": False,
                "loop.technology_guidance": False,
                "loop.project_instructions": False,
                "loop.session_context": False,
            }
        },
    )
    monkeypatch.setattr(
        "infinidev.prompts.tech.get_tech_prompt",
        lambda _hint: "TECHNOLOGY-MARKER",
    )

    from infinidev.engine.loop.context import build_system_prompt

    prompt = build_system_prompt(
        "",
        tech_hints=["python"],
        session_summaries=["SESSION-MARKER"],
        workspace_path=str(workspace),
        prompt_configuration=configuration,
    )

    assert "expert software engineer and researcher assisting" in prompt
    assert "Product bars and working guidance" not in prompt
    assert "TECHNOLOGY-MARKER" not in prompt
    assert "PROJECT-MARKER" not in prompt
    assert "SESSION-MARKER" not in prompt
    assert "plan-execute-summarize" in prompt


def test_optional_iteration_context_can_be_disabled_without_contracts(tmp_path) -> None:
    configuration = _configuration(
        tmp_path,
        {
            "develop": {
                "iteration.project_knowledge": False,
                "iteration.session_notes": False,
                "iteration.workspace": False,
            }
        },
    )
    from infinidev.engine.loop.context import build_iteration_prompt
    from infinidev.engine.loop.loop_state import LoopState
    from infinidev.engine.loop.plan_step import PlanStep

    state = LoopState()
    state.plan.steps = [
        PlanStep(
            index=1,
            title="Required current action",
            status="active",
            expected_output="REQUIRED-EXPECTED",
        )
    ]
    prompt = build_iteration_prompt(
        "REQUIRED-TASK",
        "fallback",
        state,
        project_knowledge=[{
            "id": 1,
            "finding_type": "fact",
            "status": "active",
            "confidence": 1.0,
            "topic": "OPTIONAL-KNOWLEDGE",
            "content": "OPTIONAL-CONTENT",
        }],
        session_notes=["OPTIONAL-NOTE"],
        prompt_configuration=configuration,
    )

    assert "OPTIONAL-KNOWLEDGE" not in prompt
    assert "OPTIONAL-NOTE" not in prompt
    assert "<workspace>" not in prompt
    assert "<task>\nREQUIRED-TASK\n</task>" in prompt
    assert "<plan>" in prompt
    assert "<current-action>" in prompt
    assert "Required current action" in prompt
    assert "<expected-output>" in prompt
    assert "REQUIRED-EXPECTED" in prompt


def test_task_planner_profiles_remove_guidance_but_keep_terminal_contract(tmp_path) -> None:
    configuration = _configuration(
        tmp_path,
        {
            "plan": {
                "task_planner.identity": False,
                "task_planner.methodology": False,
                "task_planner.planning_vocabulary": False,
                "task_planner.decomposition_guidance": False,
                "task_planner.verification_guidance": False,
                "task_planner.examples": False,
            }
        },
    )

    prompt = build_task_planner_system_prompt(
        "hard",
        configuration=configuration,
    )

    assert "You are the task planner" not in prompt
    assert "## Planning vocabulary" not in prompt
    assert "## Turn evidence into Steps" not in prompt
    assert "## Verification" not in prompt
    assert "## Output-shape example" not in prompt
    assert "## Machine facts" in prompt
    assert "Call ``emit_task_plan`` exactly once" in prompt


def test_stage_planner_profiles_remove_guidance_but_keep_terminal_contract(tmp_path) -> None:
    configuration = _configuration(
        tmp_path,
        {
            "plan": {
                "stage_planner.identity": False,
                "stage_planner.methodology": False,
                "stage_planner.planning_vocabulary": False,
                "stage_planner.authority_guidance": False,
                "stage_planner.horizon_guidance": False,
                "stage_planner.decision_guidance": False,
                "stage_planner.decomposition_guidance": False,
                "stage_planner.examples": False,
            }
        },
    )

    prompt = build_stage_planner_system_prompt(configuration=configuration)

    assert "You are the stage planner" not in prompt
    assert "## Planning vocabulary" not in prompt
    assert "## Decide from evidence" not in prompt
    assert "## Shape the Stage and its Tasks" not in prompt
    assert "## Example of the planning boundary" not in prompt
    assert "## Machine facts" in prompt
    assert "Call exactly one of" in prompt
    assert "``emit_stage``, ``complete_goal`` or ``block_goal``" in prompt


def test_standalone_planners_compile_profiles_once_per_invocation(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "prompts.json"
    path.write_text("{}", encoding="utf-8")
    from infinidev.prompts import profiles

    real_loader = profiles.load_prompt_profiles
    reads = 0

    def tracked_loader(profile_path=None):
        nonlocal reads
        reads += 1
        return real_loader(profile_path)

    monkeypatch.setattr(profiles, "get_prompt_profile_path", lambda: path)
    monkeypatch.setattr(profiles, "load_prompt_profiles", tracked_loader)

    from infinidev.engine.analysis import planner, stage_planner
    from infinidev.engine.analysis.plan import Plan
    from infinidev.engine.analysis.staged_planning import (
        BlockGoalDecision,
        GoalSpec,
        StagedPlanningState,
    )
    from infinidev.engine.orchestration.escalation_packet import EscalationPacket

    monkeypatch.setattr(
        planner,
        "_run_llm_loop",
        lambda **_kwargs: Plan(overview="profile test", steps=[]),
    )
    monkeypatch.setattr(
        stage_planner,
        "_run_llm_loop",
        lambda **_kwargs: BlockGoalDecision(
            reason="profile test",
            missing="nothing",
            evidence=[],
        ),
    )

    planner.run_planner(EscalationPacket(user_request="test", understanding="test"))
    assert reads == 1

    stage_planner.run_stage_planner(
        StagedPlanningState(goal=GoalSpec(title="test", user_request="test"))
    )
    assert reads == 2


def test_evaluation_profiles_keep_json_contracts_and_compile_once(
    tmp_path, monkeypatch,
) -> None:
    configuration = _configuration(
        tmp_path,
        {
            "review": {
                "reviewer.identity": False,
                "reviewer.evaluation_guidance": False,
                "evidence.identity": False,
                "evidence.evaluation_guidance": False,
                "adversarial.identity": False,
                "adversarial.evaluation_guidance": False,
            }
        },
    )

    from infinidev.engine.analysis.adversarial_verifier import AdversarialVerifier
    from infinidev.engine.analysis.review_engine import ReviewEngine
    from infinidev.prompts.reviewer.evidence_system import (
        build_evidence_review_system_prompt,
    )
    from infinidev.prompts.reviewer.system import REVIEWER_SYSTEM_PROMPT

    reviewer_prompt = ReviewEngine._compose_system_prompt(
        REVIEWER_SYSTEM_PROMPT,
        None,
        section_names={
            "Identity": "reviewer.identity",
            "Review Criteria": "reviewer.evaluation_guidance",
        },
        configuration=configuration,
    )
    evidence_prompt = build_evidence_review_system_prompt(configuration)
    verifier = AdversarialVerifier(
        completion_fn=lambda _messages: "{}",
        prompt_configuration=configuration,
    )
    verifier_prompt = verifier._build_messages(
        __import__(
            "infinidev.engine.analysis.step_verification",
            fromlist=["StepVerification"],
        ).StepVerification(kind="llm_judge", spec="verify"),
        {},
        "",
    )[0]["content"]

    assert "You are an independent, meticulous code reviewer" not in reviewer_prompt
    assert "## Review Criteria" not in reviewer_prompt
    assert "## Response Format" in reviewer_prompt
    assert '"verdict": "APPROVED"' in reviewer_prompt
    assert "You are an evidence reviewer" not in evidence_prompt
    assert "Return JSON only:" in evidence_prompt
    assert '"claim_excerpt"' in evidence_prompt
    assert "SKEPTICAL, INDEPENDENT" not in verifier_prompt
    assert "Respond with ONLY a JSON object" in verifier_prompt
    assert '"cited_evidence"' in verifier_prompt


def test_evaluation_engines_compile_profiles_once_per_invocation(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "prompts.json"
    path.write_text("{}", encoding="utf-8")
    from infinidev.prompts import profiles

    real_loader = profiles.load_prompt_profiles
    reads = 0

    def tracked_loader(profile_path=None):
        nonlocal reads
        reads += 1
        return real_loader(profile_path)

    monkeypatch.setattr(profiles, "get_prompt_profile_path", lambda: path)
    monkeypatch.setattr(profiles, "load_prompt_profiles", tracked_loader)

    from infinidev.engine.analysis.adversarial_verifier import AdversarialVerifier
    from infinidev.engine.analysis.evidence_review import EvidenceReviewEngine
    from infinidev.engine.analysis.review_engine import ReviewEngine

    review_engine = ReviewEngine()
    assert reads == 1
    review_engine._compose_system_prompt(
        "## Identity\n\noptional\n\n## Response Format\n\nrequired",
        None,
        section_names={"Identity": "reviewer.identity"},
        configuration=review_engine._prompt_configuration,
    )
    review_engine._compose_system_prompt(
        "## Identity\n\noptional\n\n## Response Format\n\nrequired",
        None,
        section_names={"Identity": "reviewer.identity"},
        configuration=review_engine._prompt_configuration,
    )
    assert reads == 1

    EvidenceReviewEngine()
    assert reads == 2
    AdversarialVerifier(completion_fn=lambda _messages: "{}")
    assert reads == 3


def test_remaining_prompt_families_disable_guidance_but_keep_contracts(tmp_path) -> None:
    configuration = _configuration(
        tmp_path,
        {
            "chat": {
                "chat.identity": False,
                "chat.language_guidance": False,
                "chat.council_guidance": False,
                "chat.followup_guidance": False,
                "chat.project_instructions": False,
                "chat.model_guidance": False,
            },
            "council": {
                "council.seed_identity": False,
                "council.member_identity": False,
                "council.judge_identity": False,
                "council.synthesis_identity": False,
                "council.language_guidance": False,
                "council.persona_palette": False,
            },
            "gather": {
                "gather.identity_guidance": False,
                "gather.classifier_guidance": False,
                "gather.synthesis_guidance": False,
                "gather.question_guidance": False,
            },
            "summarize": {"summary.step_guidance": False},
        },
    )

    from infinidev.engine.council.brief import MemberAssignment
    from infinidev.engine.council.prompts import (
        build_member_system_prompt,
        build_moderator_judge_prompt,
        build_moderator_seed_prompt,
        build_moderator_synth_prompt,
    )
    from infinidev.engine.loop.step_summarizer import (
        _SUMMARIZER_GUIDANCE,
        _SUMMARIZER_OUTPUT_CONTRACT,
    )
    from infinidev.gather.classifier import (
        _CLASSIFIER_CONTRACT,
        _CLASSIFIER_GUIDANCE,
    )
    from infinidev.gather.mini_agent import (
        GatherSession,
        _INVESTIGATOR_CONTRACT,
        _INVESTIGATOR_GUIDANCE,
    )
    from infinidev.gather.runner import (
        _DYNAMIC_QUESTIONS_CONTRACT,
        _DYNAMIC_QUESTIONS_GUIDANCE,
        _SYNTHESIZER_CONTRACT,
        _SYNTHESIZER_GUIDANCE,
    )
    from infinidev.prompts.chat_agent.system import (
        CHAT_AGENT_SYSTEM_PROMPT_TEMPLATE,
        compose_chat_agent_system_prompt,
    )

    chat = compose_chat_agent_system_prompt(
        CHAT_AGENT_SYSTEM_PROMPT_TEMPLATE.format(
            chat_agent_toolbox="read_file",
            developer_toolset="edit_file",
        ),
        configuration=configuration,
    )
    assert "conversational coding assistant" not in chat
    assert "## CRITICAL: Reply in the user's language" not in chat
    assert "## Convening a council" not in chat
    assert "## Self-referential follow-ups" not in chat
    assert "Every user turn starts here" in chat
    assert "exactly ONE tool call — respond OR escalate" in chat

    seed = build_moderator_seed_prompt(configuration)
    member = build_member_system_prompt(
        MemberAssignment("skeptic", "challenge assumptions", "find risks"),
        "What should we build?",
        configuration,
    )
    judge = build_moderator_judge_prompt(configuration)
    synth = build_moderator_synth_prompt(configuration)
    assert "You are the MODERATOR" not in seed
    assert "Persona palette" not in seed
    assert "seed_council`` EXACTLY once" in seed
    assert "You are a member" not in member
    assert "channel_post`` — your contribution" in member
    assert "You are the MODERATOR judging" not in judge
    assert "council_verdict`` EXACTLY once" in judge
    assert "You are the MODERATOR closing" not in synth
    assert "synthesize_brief`` EXACTLY once" in synth

    gather = GatherSession(configuration)
    assert configuration.resolve("gather", "gather.identity_guidance").enabled is False
    assert "You ONLY gather information" in _INVESTIGATOR_CONTRACT
    assert "step_complete with status=\"done\"" in _INVESTIGATOR_CONTRACT
    assert "codebase investigator" in _INVESTIGATOR_GUIDANCE
    assert gather._prompt_configuration is configuration
    assert "ticket classifier" in _CLASSIFIER_GUIDANCE
    assert '"ticket_type"' in _CLASSIFIER_CONTRACT
    assert "self-contained task description" in _SYNTHESIZER_GUIDANCE
    assert "Output ONLY the description text" in _SYNTHESIZER_CONTRACT
    assert "additional questions" in _DYNAMIC_QUESTIONS_GUIDANCE
    assert "output a JSON array" in _DYNAMIC_QUESTIONS_CONTRACT

    summary_guidance = configuration.resolve("summarize", "summary.step_guidance")
    assert summary_guidance.enabled is False
    assert "step summarizer" in _SUMMARIZER_GUIDANCE
    assert "Output EXACTLY this JSON format" in _SUMMARIZER_OUTPUT_CONTRACT
    assert '"files_to_preload"' in _SUMMARIZER_OUTPUT_CONTRACT


def test_remaining_standalone_invocations_compile_profiles_once(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "prompts.json"
    path.write_text("{}", encoding="utf-8")
    from infinidev.prompts import profiles

    real_loader = profiles.load_prompt_profiles
    reads = 0

    def tracked_loader(profile_path=None):
        nonlocal reads
        reads += 1
        return real_loader(profile_path)

    monkeypatch.setattr(profiles, "get_prompt_profile_path", lambda: path)
    monkeypatch.setattr(profiles, "load_prompt_profiles", tracked_loader)

    from infinidev.engine.council import moderator
    from infinidev.engine.council.brief import CouncilRoster
    from infinidev.engine.council.runner import run_council
    from infinidev.gather.mini_agent import GatherSession
    from infinidev.prompts.chat_agent import build_chat_agent_system_prompt

    build_chat_agent_system_prompt()
    assert reads == 1
    GatherSession()
    assert reads == 2

    monkeypatch.setattr(
        moderator,
        "seed_council",
        lambda *_args, **_kwargs: CouncilRoster(
            question="test",
            members=[],
            opening_threads=[],
        ),
    )
    run_council("test")
    assert reads == 3


def test_invalid_known_setting_is_rejected(tmp_path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps({"develop": {"loop.identity": ["invalid"]}}), encoding="utf-8")

    with pytest.raises(PromptProfileError, match="boolean or object"):
        resolve_prompt_profile("develop", "loop.identity", path)
