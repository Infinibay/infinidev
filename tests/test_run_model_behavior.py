from __future__ import annotations

from types import SimpleNamespace

import pytest

from bench.model_behavior import Observation, Probe, UtilityProfile
from bench.run_model_behavior import (
    Condition,
    ModelReply,
    RunConfig,
    parse_model_reply,
    pending_runs,
    present_probe,
    probe_messages,
    litellm_completion,
    run_one,
    run_sequentially,
    select_probe_checkpoint,
    select_probes,
    validate_option_order_protocol,
    validate_preference_context,
    validate_run_checkpoint,
)


def _probe() -> Probe:
    return Probe(
        "p1", "tool_choice", "What next?", {"A": "Read", "B": "Guess"}, "A"
    )


def _config() -> RunConfig:
    return RunConfig(
        "test-model",
        "provider/test@v1",
        (Condition("base", "Be careful"),),
        elicitation_protocol="self_report",
    )


def test_run_config_enforces_two_second_request_floor() -> None:
    raw = {
        "model": "test-model",
        "model_identity": "provider/test@v1",
        "conditions": {"raw": None},
        "min_request_interval_seconds": 1.99,
    }

    with pytest.raises(ValueError, match="must be at least 2.0"):
        RunConfig.from_dict(raw)


def test_parse_model_reply_accepts_json_inside_surrounding_text() -> None:
    parsed = parse_model_reply(
        'Result:\n```json\n{"answer":"a","confidence":0.8,'
        '"decision_criterion":"inspect evidence","missing_context":""}\n```',
        {"A": "Read", "B": "Guess"},
    )
    assert parsed["answer"] == "A"
    assert parsed["confidence"] == 0.8


def test_run_one_records_raw_feedback_identity_and_hash() -> None:
    def completion(config: RunConfig, condition: Condition, probe: Probe) -> ModelReply:
        return ModelReply(
            '{"answer":"A","confidence":0.9,'
            '"decision_criterion":"read first","missing_context":"file tree"}',
            100,
            20,
        )

    observation = run_one(_probe(), _config().conditions[0], _config(), 2, completion)
    assert observation.answer == "A"
    assert observation.repetition == 2
    assert observation.model_identity == "provider/test@v1"
    assert observation.condition_sha256
    assert observation.missing_context == "file tree"
    assert observation.input_tokens == 100


def test_run_one_records_invalid_output_as_error() -> None:
    def completion(config: RunConfig, condition: Condition, probe: Probe) -> ModelReply:
        return ModelReply("not json")

    observation = run_one(_probe(), _config().conditions[0], _config(), 0, completion)
    assert observation.error and "did not contain a JSON object" in observation.error
    assert observation.confidence is None


def test_pending_runs_resumes_by_condition_hash_and_repetition() -> None:
    condition = _config().conditions[0]
    completed = [
        Observation(
            "p1", "base", "A", 0.9, repetition=0, condition_sha256=condition.sha256
        )
    ]
    work = pending_runs([_probe()], [condition], 2, completed)
    assert [(probe.id, repetition) for probe, _, repetition in work] == [("p1", 1)]


def test_changed_prompt_is_not_mistaken_for_completed_condition() -> None:
    old = Condition("base", "Old prompt")
    new = Condition("base", "New prompt")
    completed = [
        Observation("p1", "base", "A", 0.9, condition_sha256=old.sha256)
    ]
    assert len(pending_runs([_probe()], [new], 1, completed)) == 1


def test_probe_selection_excludes_drafts_unless_explicitly_requested() -> None:
    approved = Probe(
        "approved", "tools", "?", {"A": "x", "B": "y"}, "A",
        review_status="approved",
    )
    draft = Probe("draft", "tools", "?", {"A": "x", "B": "y"}, "A")
    assert [probe.id for probe in select_probes([approved, draft])] == ["approved"]
    assert {probe.id for probe in select_probes([approved, draft], include_drafts=True)} == {
        "approved",
        "draft",
    }


def test_probe_selection_filters_evaluation_mode_before_execution() -> None:
    normative = Probe(
        "normative", "tools", "?", {"A": "x", "B": "y"}, "A",
        review_status="approved",
    )
    preference = Probe(
        "preference", "tools", "", {"A": "x", "B": "y"}, None,
        scenario="A reversible choice.",
        user_request="Choose based on my preference.",
        evaluation_mode="preference",
        choice_effects={"A": {"speed": 1.0}, "B": {"quality": 1.0}},
        review_status="approved",
    )

    assert select_probes(
        [normative, preference], evaluation_mode="normative"
    ) == [normative]
    assert select_probes(
        [normative, preference], evaluation_mode="preference"
    ) == [preference]
    assert select_probes(
        [normative, preference], probe_ids=["preference"]
    ) == [preference]

    with pytest.raises(ValueError, match="non-scored raw-prior baseline"):
        validate_preference_context([preference], _config())
    validate_preference_context([preference], _config(), allow_unprofiled=True)


def test_run_one_records_utility_profile_identity() -> None:
    profile = UtilityProfile.from_dict(
        {"name": "high-control", "weights": {"user_control": 1.0}}
    )
    config = RunConfig(
        "test-model",
        "provider/test@v1",
        (Condition("base", "Be careful"),),
        utility_profile=profile,
    )

    def completion(config: RunConfig, condition: Condition, probe: Probe) -> ModelReply:
        return ModelReply('{"answer":"A","confidence":0.9}')

    observation = run_one(_probe(), config.conditions[0], config, 0, completion)
    assert observation.utility_profile == "high-control"
    assert observation.utility_profile_sha256 == profile.sha256


def test_pending_runs_does_not_mix_different_utility_profiles() -> None:
    condition = _config().conditions[0]
    completed = [
        Observation(
            "p1",
            "base",
            "A",
            0.9,
            condition_sha256=condition.sha256,
            utility_profile_sha256="profile-one",
        )
    ]
    assert len(
        pending_runs(
            [_probe()], [condition], 1, completed, utility_profile_sha256="profile-two"
        )
    ) == 1


def test_balanced_rotation_presents_each_action_at_each_letter_once() -> None:
    probe = Probe(
        "p4",
        "tools",
        "Choose",
        {"A": "one", "B": "two", "C": "three", "D": "four"},
        "A",
    )
    mappings = [
        present_probe(probe, repetition, "balanced_rotation", seed=17)[1]
        for repetition in range(4)
    ]
    for displayed_letter in probe.choices:
        assert {mapping[displayed_letter] for mapping in mappings} == set(probe.choices)


def test_run_one_maps_provider_letter_back_to_canonical_action() -> None:
    probe = Probe(
        "p4",
        "tools",
        "Choose",
        {"A": "one", "B": "two", "C": "three", "D": "four"},
        "A",
    )
    config = RunConfig(
        "test",
        "test@v1",
        (Condition("raw", None),),
        elicitation_protocol="choice_only",
        option_order_protocol="balanced_rotation",
        seed=17,
    )

    def completion(config: RunConfig, condition: Condition, shown: Probe) -> ModelReply:
        return ModelReply('{"answer":"A"}')

    observation = run_one(
        probe,
        config.conditions[0],
        config,
        1,
        completion,
        dataset_sha256="dataset-v2",
        manifest_sha256="manifest-v1",
    )
    assert observation.provider_answer == "A"
    assert observation.answer == observation.choice_mapping["A"]
    assert observation.option_order_protocol == "balanced_rotation"
    assert observation.presentation_id.startswith("balanced_rotation:")
    assert observation.dataset_sha256 == "dataset-v2"
    assert observation.manifest_sha256 == "manifest-v1"


def test_pending_runs_treats_changed_dataset_or_manifest_as_new_evidence() -> None:
    config = RunConfig("test", "test@v1", (Condition("raw", None),))
    completed = [
        Observation(
            "p1",
            "raw",
            "A",
            None,
            condition_sha256=config.conditions[0].sha256,
            dataset_sha256="old-dataset",
            manifest_sha256="old-manifest",
        )
    ]
    assert pending_runs(
        [_probe()],
        config.conditions,
        1,
        completed,
        elicitation_protocol="self_report",
        dataset_sha256="new-dataset",
        manifest_sha256="new-manifest",
    )
    assert not pending_runs(
        [_probe()],
        config.conditions,
        1,
        completed,
        elicitation_protocol="self_report",
        dataset_sha256="old-dataset",
        manifest_sha256="old-manifest",
    )


def test_balanced_rotation_requires_complete_choice_cycles() -> None:
    config = RunConfig(
        "test",
        "test@v1",
        (Condition("raw", None),),
        option_order_protocol="balanced_rotation",
    )
    with pytest.raises(ValueError, match="repetitions divisible"):
        validate_option_order_protocol([_probe()], config, repetitions=3)
    validate_option_order_protocol([_probe()], config, repetitions=2)


def test_balanced_checkpoint_limits_whole_probes_not_individual_calls() -> None:
    probes = [_probe(), Probe("p2", "tools", "Other", {"A": "x", "B": "y"}, "A")]
    assert [probe.id for probe in select_probe_checkpoint(probes, 1)] == ["p1"]
    config = RunConfig(
        "test",
        "test@v1",
        (Condition("raw", None),),
        option_order_protocol="balanced_rotation",
    )
    validate_run_checkpoint(config, max_runs=0, max_probes=1)
    with pytest.raises(ValueError, match="cannot use max_runs"):
        validate_run_checkpoint(config, max_runs=4, max_probes=0)
    with pytest.raises(ValueError, match="either max_runs or max_probes"):
        validate_run_checkpoint(config, max_runs=1, max_probes=1)


def test_raw_condition_has_no_system_message_and_calls_are_isolated() -> None:
    config = RunConfig("test", "test@v1", (Condition("raw", None),))
    first = _probe()
    second = Probe("p2", "tools", "Different question?", {"A": "x", "B": "y"}, "A")

    first_messages = probe_messages(config, config.conditions[0], first)
    second_messages = probe_messages(config, config.conditions[0], second)

    assert [message["role"] for message in first_messages] == ["user"]
    assert "What next?" in first_messages[0]["content"]
    assert "What next?" not in second_messages[0]["content"]
    assert first_messages is not second_messages
    assert "confidence" not in first_messages[0]["content"]
    assert "decision_criterion" not in first_messages[0]["content"]


def test_choice_only_does_not_invent_unreported_feedback() -> None:
    config = RunConfig("test", "test@v1", (Condition("raw", None),))

    def completion(config: RunConfig, condition: Condition, probe: Probe) -> ModelReply:
        return ModelReply('{"answer":"A"}')

    observation = run_one(_probe(), config.conditions[0], config, 0, completion)
    assert observation.answer == "A"
    assert observation.confidence is None
    assert observation.decision_criterion == ""
    assert observation.missing_context == ""
    assert observation.elicitation_protocol == "choice_only"


def test_pending_runs_keeps_elicitation_protocols_separate() -> None:
    condition = Condition("raw", None)
    completed = [
        Observation(
            "p1", "raw", "A", None,
            condition_sha256=condition.sha256,
            elicitation_protocol="choice_only",
        )
    ]
    assert len(
        pending_runs(
            [_probe()], [condition], 1, completed,
            elicitation_protocol="self_report",
        )
    ) == 1


def test_subscription_runner_applies_shared_oauth_transport(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def apply_transport(params: dict[str, object], provider: str) -> None:
        assert provider == "openai_subscription"
        params["model"] = "openai/responses/gpt-5.6-sol"
        params["api_key"] = "secret-not-recorded"
        params["api_base"] = "https://chatgpt.com/backend-api/codex"
        params["extra_body"] = {"store": False}
        params.pop("temperature", None)

    def completion(**kwargs: object) -> object:
        calls.append(kwargs)
        message = SimpleNamespace(content='{"answer":"A"}')
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage={"prompt_tokens": 10, "completion_tokens": 3},
        )

    monkeypatch.setattr("infinidev.config.llm.apply_provider_transport", apply_transport)
    monkeypatch.setattr("litellm.completion", completion)
    config = RunConfig(
        "gpt-5.6-sol",
        "openai_subscription/gpt-5.6-sol@catalog-test",
        (Condition("raw", None),),
        provider="openai_subscription",
        reasoning_effort="medium",
    )

    reply = litellm_completion(config, config.conditions[0], _probe())

    assert reply.text == '{"answer":"A"}'
    assert calls[0]["model"] == "openai/responses/gpt-5.6-sol"
    assert calls[0]["reasoning_effort"] == "medium"
    assert calls[0]["extra_body"] == {"store": False}
    assert "temperature" not in calls[0]


def test_subscription_runner_uses_completion_boundary_installed_by_transport(
    monkeypatch,
) -> None:
    """The subscription import may replace litellm.completion with its stream repair."""
    import litellm

    def stale_completion(**kwargs: object) -> object:
        raise AssertionError("captured completion before transport setup")

    def repaired_completion(**kwargs: object) -> object:
        message = SimpleNamespace(content='{"answer":"A"}')
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage={"prompt_tokens": 8, "completion_tokens": 2},
        )

    def apply_transport(params: dict[str, object], provider: str) -> None:
        assert provider == "openai_subscription"
        monkeypatch.setattr(litellm, "completion", repaired_completion)

    monkeypatch.setattr(litellm, "completion", stale_completion)
    monkeypatch.setattr("infinidev.config.llm.apply_provider_transport", apply_transport)
    config = RunConfig(
        "gpt-5.6-sol",
        "openai_subscription/gpt-5.6-sol@catalog-test",
        (Condition("raw", None),),
        provider="openai_subscription",
    )

    reply = litellm_completion(config, config.conditions[0], _probe())

    assert reply.text == '{"answer":"A"}'
    assert reply.input_tokens == 8


def test_run_sequentially_never_overlaps_and_stops_on_rate_limit() -> None:
    config = RunConfig(
        "test", "test@v1", (Condition("raw", None),),
        min_request_interval_seconds=2.0,
    )
    work = [(_probe(), config.conditions[0], index) for index in range(3)]
    active = 0
    maximum_active = 0
    calls = 0
    recorded: list[Observation] = []
    clock = [0.0]

    def completion(config: RunConfig, condition: Condition, probe: Probe) -> ModelReply:
        nonlocal active, maximum_active, calls
        active += 1
        maximum_active = max(maximum_active, active)
        calls += 1
        active -= 1
        if calls == 2:
            raise RuntimeError("status code: 429 rate limit")
        return ModelReply('{"answer":"A","confidence":0.9}')

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    completed = run_sequentially(
        work,
        config,
        completion,
        recorded.append,
        sleep=sleep,
        monotonic=lambda: clock[0],
    )

    assert maximum_active == 1
    assert calls == completed == len(recorded) == 2
    assert clock[0] == 2.0
    assert recorded[-1].error and "429" in recorded[-1].error


def test_run_sequentially_rejects_a_direct_config_below_request_floor() -> None:
    config = RunConfig(
        "test",
        "test@v1",
        (Condition("raw", None),),
        min_request_interval_seconds=1.99,
    )

    with pytest.raises(ValueError, match="must be at least 2.0"):
        run_sequentially([], config, lambda *_: ModelReply('{"answer":"A"}'), lambda _: None)
