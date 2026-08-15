"""Regression coverage for conditional task-policy routing and rendering."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infinidev.engine.orchestration.task_renderer import render_task_xml
from infinidev.engine.orchestration.task_schema import task_from_free_text
from infinidev.engine.task_policies.models import ClassifierResult, TaskProfile
from infinidev.engine.task_policies.linear_classifier import (
    CLASSIFIER_VERSION as LINEAR_CLASSIFIER_VERSION,
)
from infinidev.engine.task_policies.registry import POLICY_BY_ID
from infinidev.engine.task_policies.rendering import (
    compose_task_aware_system_prompt,
    render_all_conditional_task_policy_layer,
    render_task_policy_layer,
    select_task_policy_fragments,
)
from infinidev.engine.task_policies.router import resolve_task_profile
from infinidev.engine.task_policies.semantic import (
    SEMANTIC_CLASSIFIER_VERSION,
    SemanticPolicyCandidate,
    SemanticRetrieval,
)
from infinidev.engine.task_policies.semantic_prototypes import PROTOTYPES


def _selected(profile: TaskProfile) -> set[str]:
    return {selection.id for selection in profile.selected_policies}


def test_semantic_prototypes_are_separate_and_cover_operation_policies() -> None:
    operation_policy_ids = {
        policy.id for policy in POLICY_BY_ID.values() if policy.operations
    }

    assert set(PROTOTYPES) == operation_policy_ids
    assert all(len(item.positive) >= 20 for item in PROTOTYPES.values())
    assert all(len(item.negative) >= 20 for item in PROTOTYPES.values())
    assert all(len(set(item.positive)) == len(item.positive) for item in PROTOTYPES.values())
    assert all(len(set(item.negative)) == len(item.negative) for item in PROTOTYPES.values())
    assert not hasattr(POLICY_BY_ID["bugfix.root_cause"], "positive_examples")


def test_composed_research_fix_preserves_literal_authority() -> None:
    profile = resolve_task_profile(
        "Investiga por qué falla y después corrígelo sin cambiar la API pública."
    )

    assert profile.operations == ("bugfix", "research")
    assert {"diagnose", "modify"} <= set(profile.authority)
    assert "preserve_public_api" in profile.constraints
    assert {
        "bugfix.root_cause",
        "compatibility.preserve_public_api",
        "research.evidence_first",
    } == _selected(profile)
    assert profile.sequence == ("investigate", "implement", "verify")


def test_natural_bugfix_verbs_grant_literal_modify_authority() -> None:
    requests = (
        "The cache returns stale entries after invalidation. Find the cause and correct it.",
        "After reconnecting the handler emits duplicates; restore the existing contract.",
        "Una entrada válida ahora falla; restablece el resultado anterior.",
        "Le parseur rejette les fichiers valides; rétablis le comportement attendu.",
    )

    for request in requests:
        profile = resolve_task_profile(request, enable_embeddings=True)

        assert "modify" in profile.authority, request


def test_negated_refactor_does_not_authorize_changes() -> None:
    profile = resolve_task_profile("No refactorices; solo explícame el problema.")

    assert "refactor" not in profile.operations
    assert profile.authority == ("answer",)
    assert profile.constraints == ("read_only",)
    assert "refactor.preserve_behavior" not in _selected(profile)


def test_quoted_policy_language_is_not_treated_as_intent() -> None:
    profile = resolve_task_profile('El error dice "refactor required", ¿qué significa?')

    assert profile.operations == ()
    assert profile.authority == ("answer",)
    assert not profile.selected_policies


def test_read_only_review_selects_review_policy() -> None:
    profile = resolve_task_profile("Revisa el PR, pero no cambies archivos.")

    assert profile.operations == ("review",)
    assert profile.authority == ("answer", "diagnose")
    assert _selected(profile) == {"review.read_only"}


def test_review_report_write_does_not_become_implementation_work() -> None:
    profile = resolve_task_profile(
        "Review auth.py and write REVIEW.md. Report correctness or security blockers "
        "first with precise evidence, then maintainability concerns and optional "
        "observations. Do not modify the implementation."
    )

    assert profile.operations == ("review", "security")
    assert "modify" in profile.authority
    assert _selected(profile) == {"review.read_only"}
    assert not profile.rejected_candidates
    assert profile.result == ("report",)
    assert profile.sequence == ("review",)


def test_optimization_with_api_constraint_is_a_modifying_task() -> None:
    profile = resolve_task_profile("Optimiza esta función sin modificar la API pública.")

    assert profile.operations == ("performance",)
    assert "modify" in profile.authority
    assert "preserve_public_api" in profile.constraints
    assert _selected(profile) == {
        "compatibility.preserve_public_api",
        "performance.measure_first",
    }


def test_read_only_performance_analysis_keeps_measurement_policy() -> None:
    profile = resolve_task_profile(
        "Profile the supplied workload and report p95 and allocations; do not change code."
    )

    assert profile.operations == ("performance",)
    assert profile.authority == ("answer", "diagnose")
    assert profile.constraints == ("read_only",)
    assert profile.result == ("report",)
    assert profile.sequence == ("investigate",)
    assert _selected(profile) == {"performance.measure_first"}


def test_commit_and_push_require_literal_external_authority() -> None:
    profile = resolve_task_profile("Implementa la feature, haz commit y push a main.")

    assert {"modify", "commit", "publish"} <= set(profile.authority)
    assert "external_write" in profile.risks
    assert profile.sequence[-2:] == ("commit", "publish")


def test_llm_fallback_cannot_grant_modify_authority() -> None:
    def classify(_: str) -> ClassifierResult:
        return ClassifierResult(
            operations=["feature"], result=["code"], sequence=["implement", "publish"]
        )

    profile = resolve_task_profile(
        "Could you deal with this component?",
        enable_llm_fallback=True,
        classifier=classify,
    )

    assert profile.llm_fallback_used
    assert profile.operations == ("feature",)
    assert profile.authority == ("answer",)
    assert "feature.contract_first" not in _selected(profile)
    assert profile.sequence == ()
    assert profile.rejected_candidates[0].reason.startswith("literal request")


def test_preferred_main_model_classifier_runs_before_and_skips_local_encoder(
    monkeypatch,
) -> None:
    from infinidev.engine.task_policies import router

    calls = []

    def classify(text: str) -> ClassifierResult:
        calls.append(text)
        return ClassifierResult(operations=["feature"], confidence=0.93)

    monkeypatch.setattr(
        router,
        "classify_task_methods",
        lambda *args, **kwargs: pytest.fail("local encoder should not run"),
    )

    profile = resolve_task_profile(
        "Please improve this component.",
        enable_embeddings=True,
        classifier=classify,
        encoder_checkpoint="/unused/checkpoint",
        llm_classifier_mode="preferred",
    )

    assert calls == ["Please improve this component."]
    assert profile.llm_classifier_used
    assert not profile.llm_fallback_used
    assert _selected(profile) == {"feature.contract_first"}
    assert profile.selected_policies[0].source == "llm"
    assert profile.selected_policies[0].score == 0.93
    rendered = render_task_policy_layer(
        profile,
        role="developer",
        phase="execute",
        force=True,
    )
    assert 'id="feature.developer"' in rendered


def test_preferred_main_model_classifier_still_cannot_grant_authority() -> None:
    profile = resolve_task_profile(
        "What would be a useful capability here?",
        classifier=lambda _: ClassifierResult(
            operations=["feature"], confidence=0.9,
        ),
        llm_classifier_mode="preferred",
    )

    assert profile.operations == ("feature",)
    assert profile.authority == ("answer",)
    assert not profile.selected_policies
    assert profile.rejected_candidates[0].id == "feature.contract_first"


def test_preferred_main_model_replaces_literal_method_but_keeps_authority() -> None:
    profile = resolve_task_profile(
        "Implement the requested feature.",
        classifier=lambda _: ClassifierResult(
            operations=["bugfix"], confidence=0.91,
        ),
        llm_classifier_mode="preferred",
    )

    assert profile.operations == ("bugfix",)
    assert "modify" in profile.authority
    assert _selected(profile) == {"bugfix.root_cause"}
    assert profile.selected_policies[0].source == "llm"


def test_llm_classifier_mode_validation() -> None:
    with pytest.raises(ValueError, match="llm_classifier_mode"):
        resolve_task_profile("test", llm_classifier_mode="sometimes")


def test_default_classifier_uses_selected_main_model_params(monkeypatch) -> None:
    from infinidev.config import llm
    from infinidev.engine import llm_client
    from infinidev.engine.task_policies import router

    captured = {}
    monkeypatch.setattr(
        llm,
        "get_litellm_params",
        lambda: {"model": "selected/main-model", "api_base": "https://example.test"},
    )

    def call(params, messages, **kwargs):
        captured.update({"params": params, "messages": messages, "kwargs": kwargs})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content='{"operations":["research","performance"],"confidence":0.94}',
        ))])

    monkeypatch.setattr(llm_client, "call_llm", call)

    result = router._default_llm_classifier("Investigate the slow path.", max_tokens=73)

    assert result is not None
    assert result.operations == ["research", "performance"]
    assert result.confidence == 0.94
    assert captured["params"] == {
        "model": "selected/main-model",
        "api_base": "https://example.test",
        "max_tokens": 73,
    }
    assert captured["messages"][-1]["content"] == "Investigate the slow path."
    assert captured["kwargs"] == {
        "thinking_enabled": False,
        "use_json_mode": False,
    }


def test_embedding_double_agreement_can_suggest_method_but_not_write_authority(
    monkeypatch,
) -> None:
    from infinidev.engine.task_policies import router
    from infinidev.engine.task_policies.linear_classifier import TaskMethodPrediction

    policy = POLICY_BY_ID["research.evidence_first"]
    monkeypatch.setattr(
        router,
        "retrieve_policy_candidates",
        lambda *args, **kwargs: SemanticRetrieval(
            candidates=(SemanticPolicyCandidate(
                policy=policy,
                score=0.81,
                runner_up_margin=0.19,
                negative_margin=0.22,
                evidence="positive-example:research.evidence_first:2",
                space_id="ken/static-qwen3-r512-v2:1024:test",
            ),),
            space_id="ken/static-qwen3-r512-v2:1024:test",
            classifier_version=SEMANTIC_CLASSIFIER_VERSION,
            abstained=False,
        ),
    )
    monkeypatch.setattr(
        router,
        "classify_task_method",
        lambda _: TaskMethodPrediction(
            policy_id="research.evidence_first",
            score=0.84,
            threshold=0.2,
            runner_up_margin=0.2,
            space_id="ken/static-qwen3-r512-v2:1024:test",
        ),
    )

    profile = resolve_task_profile(
        "Please look into this component.",
        enable_embeddings=True,
        embedding_threshold=0.7,
    )

    assert profile.operations == ("research",)
    assert profile.authority == ("answer",)
    assert _selected(profile) == {"research.evidence_first"}
    assert profile.semantic_classifier_version == LINEAR_CLASSIFIER_VERSION
    assert profile.semantic_space_id == "ken/static-qwen3-r512-v2:1024:test"


def test_fine_tuned_encoder_selects_multiple_methods_without_granting_authority(
    monkeypatch,
) -> None:
    from infinidev.engine.task_policies import router
    from infinidev.engine.task_policies.encoder_classifier import (
        EncoderTaskPrediction,
        PolicyScore,
    )

    monkeypatch.setattr(
        router,
        "classify_task_methods",
        lambda *args, **kwargs: EncoderTaskPrediction(
            scores=(
                PolicyScore("performance.measure_first", 0.92, 0.7, True),
                PolicyScore("research.evidence_first", 0.88, 0.8, True),
            ),
            task_score=0.97,
            task_threshold=0.5,
            classifier_version="fine-tuned-test-v1",
            space_id="infinidev/task-policy-encoder:test",
        ),
    )

    profile = resolve_task_profile(
        "Please improve this component.",
        enable_embeddings=True,
        encoder_checkpoint="/unused/checkpoint",
    )

    assert _selected(profile) == {
        "performance.measure_first",
        "research.evidence_first",
    }
    assert profile.operations == ("research", "performance")
    assert profile.authority == ("answer", "modify")
    assert profile.semantic_classifier_version == "fine-tuned-test-v1"


def test_fine_tuned_encoder_cannot_grant_modify_authority(monkeypatch) -> None:
    from infinidev.engine.task_policies import router
    from infinidev.engine.task_policies.encoder_classifier import (
        EncoderTaskPrediction,
        PolicyScore,
    )

    monkeypatch.setattr(
        router,
        "classify_task_methods",
        lambda *args, **kwargs: EncoderTaskPrediction(
            scores=(PolicyScore("refactor.preserve_behavior", 0.95, 0.8, True),),
            task_score=0.99,
            task_threshold=0.5,
            classifier_version="fine-tuned-test-v1",
            space_id="infinidev/task-policy-encoder:test",
        ),
    )

    profile = resolve_task_profile(
        "Tell me what approach would make this component easier to maintain.",
        enable_embeddings=True,
        encoder_checkpoint="/unused/checkpoint",
    )

    assert "modify" not in profile.authority
    assert "refactor.preserve_behavior" not in _selected(profile)
    assert profile.rejected_candidates[0].reason.startswith("literal request")


def test_bundled_static_router_handles_action_paraphrases() -> None:
    cases = {
        "Make warning badges easier to scan while keeping all other output stable.":
            "refactor.preserve_behavior",
        "Tell me which queue option best fits and support the recommendation with sources.":
            "research.evidence_first",
        "This endpoint is too slow. Measure where the time goes and improve it.":
            "performance.measure_first",
    }

    for request, expected in cases.items():
        profile = resolve_task_profile(
            request,
            enable_embeddings=True,
            embedding_threshold=0.18,
            embedding_margin=0.04,
        )

        assert expected in _selected(profile), request
        assert profile.semantic_space_id is not None
        assert profile.semantic_space_id.startswith("ken/static-qwen3-r512-v2:1024:")


def test_literal_method_resolves_a_matching_mini_head_margin_tie() -> None:
    profile = resolve_task_profile(
        "Fix pages_needed so a partial final page is included, exact multiples stay "
        "unchanged, zero items need zero pages, and invalid page sizes keep raising "
        "ValueError. Change only the implementation and run the relevant tests.",
        enable_embeddings=True,
    )

    bugfix = next(
        item for item in profile.selected_policies if item.id == "bugfix.root_cause"
    )
    assert bugfix.evidence == ("mini-head+literal",)
    assert not profile.semantic_abstained


def test_bundled_static_router_resolves_contract_restoration_by_agreement() -> None:
    profile = resolve_task_profile(
        "The cache returns stale values. Make it behave as specified again.",
        enable_embeddings=True,
        embedding_threshold=0.18,
        embedding_margin=0.04,
    )

    assert profile.operations == ("bugfix",)
    assert _selected(profile) == {"bugfix.root_cause"}
    assert profile.selected_policies[0].evidence == ("mini-head+contrastive",)
    assert not profile.semantic_abstained


def test_hierarchical_head_and_retriever_agree_on_timing_bugfixes() -> None:
    cases = (
        "The lease renewer waits after its last allowed attempt; make it stop at the limit.",
        "El debounce pierde la última edición cuando vence el timer; "
        "recupera la garantía existente.",
    )

    for text in cases:
        profile = resolve_task_profile(
            text,
            enable_embeddings=True,
            embedding_threshold=0.18,
            embedding_margin=0.04,
        )

        assert profile.operations == ("bugfix",), text
        assert _selected(profile) == {"bugfix.root_cause"}, text
        assert profile.selected_policies[0].evidence == ("mini-head+contrastive",)


def test_bugfix_literals_do_not_match_correct_results_or_fixed_delays() -> None:
    performance = resolve_task_profile(
        "The results remain correct but latency is too high; profile the endpoint."
    )
    new_strategy = resolve_task_profile(
        "Introduce jittered backoff; only fixed delays are available today."
    )

    assert performance.operations == ("performance",)
    assert "bugfix.root_cause" not in _selected(performance)
    assert "bugfix" not in new_strategy.operations
    assert "bugfix.root_cause" not in _selected(new_strategy)


def test_bundled_mini_head_abstention_vetoes_prototype_fallback() -> None:
    profile = resolve_task_profile(
        "Responde únicamente con LISTO para confirmar que recibiste este mensaje. "
        "No inspecciones ni modifiques archivos.",
        enable_embeddings=True,
        embedding_threshold=0.18,
        embedding_margin=0.04,
    )

    assert profile.operations == ()
    assert not profile.selected_policies
    assert profile.semantic_abstained
    assert profile.semantic_abstention_reason


def test_semantic_router_abstains_for_quoted_action_explanation() -> None:
    profile = resolve_task_profile(
        'The log says "please implement fix". Explain that message.',
        enable_embeddings=True,
    )

    assert profile.operations == ()
    assert not profile.selected_policies
    assert profile.semantic_abstained
    assert profile.semantic_abstention_reason == "quoted action is explanatory context"


def test_all_conditional_layer_contains_every_developer_method(monkeypatch) -> None:
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "TASK_POLICIES_SHADOW_MODE", False)
    layer = render_all_conditional_task_policy_layer(
        role="developer",
        phase="execute",
        max_utf8_bytes=12_000,
    )

    expected = (
        "compatibility.developer",
        "review.developer",
        "bugfix.developer",
        "refactor.developer",
        "feature.developer",
        "performance.developer",
        "research.developer",
    )
    assert all(f'id="{fragment_id}"' in layer for fragment_id in expected)
    assert layer.count("<if reason=") == len(expected)
    assert "not a claim that its reason is true" in layer
    assert "never grants permission" in layer
    assert "feature.planner" not in layer
    assert 'provenance="all-conditional-v1:developer:execute"' in layer


def test_default_composer_uses_all_conditions_without_profile_gating(monkeypatch) -> None:
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "TASK_POLICIES_RENDER_ALL_CONDITIONAL", True)
    monkeypatch.setattr(settings, "TASK_POLICIES_SHADOW_MODE", False)
    profile = resolve_task_profile("Implementa una nueva opción para los usuarios.")

    prompt = compose_task_aware_system_prompt(
        "stable identity and protocol",
        profile,
        role="developer",
        phase="execute",
        max_utf8_bytes=12_000,
    )

    assert 'id="feature.developer"' in prompt
    assert 'id="bugfix.developer"' in prompt
    assert 'id="review.developer"' in prompt


def test_selected_fragment_mode_remains_available(monkeypatch) -> None:
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "TASK_POLICIES_RENDER_ALL_CONDITIONAL", False)
    profile = resolve_task_profile("Implementa una nueva opción para los usuarios.")
    prompt = compose_task_aware_system_prompt(
        "stable", profile, role="developer", phase="execute", force=True,
    )

    assert 'id="feature.developer"' in prompt
    assert 'id="bugfix.developer"' not in prompt


def test_policy_rendering_is_role_phase_filtered_and_bounded(monkeypatch) -> None:
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "TASK_POLICIES_SHADOW_MODE", False)
    profile = resolve_task_profile("Refactoriza este módulo sin cambiar comportamiento.")

    layer = render_task_policy_layer(
        profile, role="developer", phase="execute", max_utf8_bytes=3600,
        force=True,
    )

    assert '<task-policy-layer provenance="task-profile-v1:developer:execute">' in layer
    assert "refactor.preserve_behavior@1" in layer
    assert len(layer.encode("utf-8")) < 4000


def test_same_task_gets_distinct_planner_developer_and_reviewer_methods() -> None:
    profile = resolve_task_profile("Implementa una nueva opción para los usuarios.")

    planner = render_task_policy_layer(
        profile, role="planner", phase="plan", force=True,
    )
    developer = render_task_policy_layer(
        profile, role="developer", phase="execute", force=True,
    )
    reviewer = render_task_policy_layer(
        profile, role="reviewer", phase="review", force=True,
    )

    assert "feature.planner" in planner
    assert "Define the new user-visible contract" in planner
    assert "feature.developer" in developer
    assert "smallest complete slice" in developer
    assert "feature.reviewer" in reviewer
    assert "Map the new user workflow" in reviewer
    assert "bugfix." not in planner + developer + reviewer


def test_researcher_role_has_an_evidence_fragment_without_coding_guidance() -> None:
    profile = resolve_task_profile(
        "Investiga las alternativas y respalda la recomendación con fuentes."
    )

    layer = render_task_policy_layer(
        profile,
        role="researcher",
        phase="investigate",
        force=True,
    )

    assert "research.researcher" in layer
    assert "Gather primary, current evidence" in layer
    assert "smallest complete slice" not in layer


def test_unselected_methods_are_omitted_and_auditable() -> None:
    profile = resolve_task_profile("Optimiza esta función y mide la mejora.")

    selection = select_task_policy_fragments(
        profile,
        role="developer",
        phase="execute",
    )

    assert [fragment.id for fragment in selection.fragments] == [
        "performance.developer"
    ]
    assert ("bugfix.developer", "policy-not-selected") in selection.omitted


def test_task_method_is_after_stable_cache_prefix() -> None:
    from infinidev.engine.prompt_composition import CACHE_BREAKPOINT_MARKER

    profile = resolve_task_profile("Refactoriza este módulo sin cambiar comportamiento.")
    prompt = compose_task_aware_system_prompt(
        "stable identity and protocol",
        profile,
        role="developer",
        phase="execute",
        force=True,
        cache_boundary=True,
    )

    assert prompt.startswith(
        f"stable identity and protocol\n\n{CACHE_BREAKPOINT_MARKER}"
    )
    assert prompt.index(CACHE_BREAKPOINT_MARKER) < prompt.index("refactor.developer")


def test_stable_role_cores_do_not_embed_a_bugfix_method() -> None:
    from infinidev.prompts.analyst.task_planner_prompt import (
        TASK_PLANNER_SYSTEM_PROMPT,
    )
    from infinidev.prompts.flows.develop import get_develop_identity

    assert "Bug-Fix Workflow Example" not in get_develop_identity(set())
    assert "validate_token rejects" not in TASK_PLANNER_SYSTEM_PROMPT


def test_reviewer_composes_policy_into_its_system_prompt(monkeypatch) -> None:
    from infinidev.config.settings import settings
    from infinidev.engine.analysis.review_engine import ReviewEngine

    monkeypatch.setattr(settings, "TASK_POLICIES_EVIDENCE_GATED", False)
    profile = resolve_task_profile("Corrige este fallo reproducible.")
    prompt = ReviewEngine._compose_system_prompt("stable reviewer", profile)

    assert prompt.startswith("stable reviewer")
    assert "bugfix.reviewer" in prompt
    assert "bugfix.developer" not in prompt


def test_bugfix_developer_fragment_is_outcome_focused_and_versioned() -> None:
    profile = resolve_task_profile("Corrige este fallo reproducible.")
    prompt = render_task_policy_layer(
        profile, role="developer", phase="execute", force=True,
    )

    assert 'id="bugfix.developer" version="3"' in prompt
    assert "narrowest demonstrated contract violation" in prompt
    assert "directly affected contract" in prompt


def test_minimax_rollout_keeps_only_e2e_approved_developer_fragment(
    monkeypatch,
) -> None:
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "LLM_PROVIDER", "minimax")
    monkeypatch.setattr(settings, "LLM_MODEL", "minimax/MiniMax-M3")
    monkeypatch.setattr(settings, "TASK_POLICIES_EVIDENCE_GATED", True)

    refactor = resolve_task_profile("Refactoriza sin cambiar comportamiento.")
    bugfix = resolve_task_profile("Corrige este fallo reproducible.")

    assert "refactor.developer" in render_task_policy_layer(
        refactor, role="developer", phase="execute",
    )
    assert render_task_policy_layer(
        bugfix, role="developer", phase="execute",
    ) == ""
    assert "bugfix.developer" in render_task_policy_layer(
        bugfix, role="developer", phase="execute", force=True,
    )


def test_terra_rollout_enables_only_its_e2e_approved_bugfix_fragment(
    monkeypatch,
) -> None:
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai_subscription")
    monkeypatch.setattr(settings, "LLM_MODEL", "gpt-5.6-terra")
    monkeypatch.setattr(settings, "TASK_POLICIES_EVIDENCE_GATED", True)
    bugfix = resolve_task_profile("Corrige este fallo reproducible.")
    refactor = resolve_task_profile("Refactoriza sin cambiar comportamiento.")

    bugfix_layer = render_task_policy_layer(
        bugfix, role="developer", phase="execute",
    )
    assert 'id="bugfix.developer" version="3"' in bugfix_layer
    assert render_task_policy_layer(
        refactor, role="developer", phase="execute",
    ) == ""


def test_task_xml_keeps_internal_profile_out_of_the_user_prompt(monkeypatch) -> None:
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "TASK_POLICIES_SHADOW_MODE", False)
    profile = resolve_task_profile("Revisa el cambio, pero no edites archivos.")
    task = task_from_free_text(
        "Revisa el cambio, pero no edites archivos.", task_profile=profile,
    )

    rendered = render_task_xml(task)

    assert "<task-profile" not in rendered
    assert "review.read_only" not in rendered
    assert task.task_profile is profile


def test_profile_event_payload_is_replayable() -> None:
    profile = resolve_task_profile("Investiga la causa de este fallo.")

    payload = profile.event_payload()

    assert payload["task_profile_version"] == 1
    assert payload["router_version"] == 2
    assert payload["selected_policies"][0]["id"] == "research.evidence_first"
    assert len(payload["selected_policies"][0]["policy_hash"]) == 64
