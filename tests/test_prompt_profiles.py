"""Regression tests for project-local JSON prompt profiles."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import errno
from importlib.resources import files
import json
import os
from pathlib import Path
import re
from threading import Event

import pytest

from infinidev.prompts.analyst.stage_planner_prompt import (
    build_stage_planner_system_prompt,
)
from infinidev.prompts.analyst.task_planner_prompt import (
    build_task_planner_system_prompt,
)
from infinidev.engine.loop.context import build_system_prompt
from infinidev.prompts import catalog as prompt_catalog
from infinidev.prompts.catalog import STARTER_PROMPT_PROFILES, USER_OVERRIDES_FILE
from infinidev.prompts.optional_capabilities import (
    CAPABILITY_RESOURCE_PACKAGE,
    OPTIONAL_CAPABILITIES,
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
    "team.orchestrator_guidance",
    "team.worker_guidance",
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


def _expected_profile_ids() -> set[str]:
    phase_ids = {
        f"phase.{task_type}.{phase}{suffix}"
        for task_type in ("bug", "feature", "refactor", "other", "sysadmin")
        for phase in ("investigate", "plan", "execute")
        for suffix in ("", "_identity")
    }
    return _FIXED_PROFILE_IDS | phase_ids


def test_documented_catalog_enumerates_every_supported_profile_id() -> None:
    documentation = (
        Path(__file__).parents[1] / "docs" / "prompt-profiles.md"
    ).read_text(encoding="utf-8")
    prefixes = (
        "loop|iteration|task_planner|stage_planner|reviewer|extractor|judge|"
        "evidence|adversarial|chat|council|gather|summary|phase|team"
    )
    documented = set(re.findall(rf"`(({prefixes})\.[a-z_.]+)`", documentation))
    documented_ids = {profile_id for profile_id, _prefix in documented}
    expected = _expected_profile_ids()

    assert len(expected) == 103
    assert documented_ids == expected


def test_documented_optional_capabilities_match_runtime_catalog_once() -> None:
    documentation = (
        Path(__file__).parents[1] / "docs" / "prompt-profiles.md"
    ).read_text(encoding="utf-8")
    table = documentation.split("## Capacidades opcionales curadas", 1)[1].split(
        "Para activar sólo depuración", 1
    )[0]
    documented_ids = re.findall(r"^\| `(capability\.[a-z_]+)` \|", table, re.MULTILINE)
    expected_ids = [name for name, _body in OPTIONAL_CAPABILITIES]
    catalog_count = re.search(
        r"ofrece un catálogo de (\d+) guías",
        documentation,
    )

    assert catalog_count is not None
    assert int(catalog_count.group(1)) == len(expected_ids)
    assert len(documented_ids) == len(set(documented_ids))
    assert documented_ids == expected_ids


def test_optional_capabilities_load_from_individual_ordered_json_resources() -> None:
    resources = sorted(
        (
            resource
            for resource in files(CAPABILITY_RESOURCE_PACKAGE).iterdir()
            if resource.name.endswith(".json")
        ),
        key=lambda resource: resource.name,
    )
    documents = [json.loads(resource.read_text(encoding="utf-8")) for resource in resources]

    assert len(resources) == len(OPTIONAL_CAPABILITIES) == 42
    assert [
        int(resource.name.split("-", 1)[0]) for resource in resources
    ] == list(range(1, 43))
    assert [document["id"] for document in documents] == [
        name for name, _body in OPTIONAL_CAPABILITIES
    ]
    assert all(set(document) == {"id", "reason", "guidance"} for document in documents)
    assert all(
        isinstance(document["guidance"], list) and all(
            isinstance(line, str) for line in document["guidance"]
        )
        for document in documents
    )


def test_starter_catalog_enumerates_every_supported_profile_id_once() -> None:
    occurrences: list[tuple[str, str, dict[str, bool]]] = []
    for _filename, content in STARTER_PROMPT_PROFILES:
        document = json.loads(content)
        for phase, entries in document.items():
            occurrences.extend((phase, name, entry) for name, entry in entries.items())

    names = [name for _phase, name, _entry in occurrences]
    optional_names = {name for name, _body in OPTIONAL_CAPABILITIES}
    assert len(names) == 103 + len(optional_names)
    assert len(names) == len(set(names))
    assert set(names) == _expected_profile_ids() | optional_names
    for _phase, name, entry in occurrences:
        assert entry == {"enabled_by_default": name not in optional_names}


def test_missing_profile_keeps_fragment_enabled(tmp_path) -> None:
    assert apply_prompt_profile("base prompt", "develop", "loop.identity") == "base prompt"
    profile = resolve_prompt_profile("develop", "loop.identity", tmp_path / "missing.json")
    assert profile.enabled
    assert profile.enabled_by_default


def test_structured_profile_declares_default_and_runtime_override(tmp_path) -> None:
    configuration = _configuration(
        tmp_path,
        {
            "develop": {
                "loop.identity": {
                    "enabled_by_default": False,
                    "enabled": True,
                    "parameters": {"detail": "compact"},
                },
                "loop.protocol": {"enabled_by_default": False},
            }
        },
    )

    identity = configuration.resolve("develop", "loop.identity")
    assert identity.enabled_by_default is False
    assert identity.enabled is True
    assert dict(identity.parameters or {}) == {"detail": "compact"}
    protocol = configuration.resolve("develop", "loop.protocol")
    assert protocol.enabled_by_default is False
    assert protocol.enabled is False


def test_structured_profile_accepts_concise_enabled_override(tmp_path) -> None:
    configuration = _configuration(
        tmp_path,
        {
            "develop": {
                "capability.testing": {"enabled": True},
                "loop.identity": {"parameters": {"detail": "compact"}},
            }
        },
    )

    capability = configuration.resolve(
        "develop", "capability.testing", default_enabled=False
    )
    assert capability.enabled is True
    assert capability.enabled_by_default is True
    identity = configuration.resolve("develop", "loop.identity")
    assert dict(identity.parameters or {}) == {"detail": "compact"}


def test_default_catalog_merges_files_then_project_override(
    tmp_path, monkeypatch,
) -> None:
    catalog = tmp_path / "user" / "prompts"
    catalog.mkdir(parents=True)
    project_profile = tmp_path / "project" / "prompts.json"
    project_profile.parent.mkdir()
    (catalog / "10-base.json").write_text(
        json.dumps(
            {
                "develop": {
                    "loop.identity": False,
                    "loop.protocol": {"detail": 1},
                },
                "models": {
                    "test-provider": {
                        "develop": {"loop.protocol": {"detail": 2}}
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (catalog / "20-local.json").write_text(
        json.dumps({"develop": {"loop.identity": True}}),
        encoding="utf-8",
    )
    project_profile.write_text(
        json.dumps(
            {
                "models": {
                    "test-provider": {
                        "develop": {"loop.protocol": False}
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "infinidev.prompts.profiles.get_prompt_catalog_path", lambda: catalog
    )
    monkeypatch.setattr(
        "infinidev.prompts.profiles.get_prompt_profile_path", lambda: project_profile
    )
    monkeypatch.setattr(
        "infinidev.prompts.profiles.settings.LLM_PROVIDER", "test-provider"
    )
    monkeypatch.setattr(
        "infinidev.prompts.profiles.settings.LLM_MODEL", "test-model"
    )

    configuration = EffectivePromptConfiguration.compile()

    assert configuration.resolve("develop", "loop.identity").enabled is True
    assert configuration.resolve("develop", "loop.protocol").enabled is False


def test_default_catalog_materializes_starters_lazily(tmp_path, monkeypatch) -> None:
    catalog = tmp_path / "missing" / "prompts"
    monkeypatch.setattr(
        "infinidev.prompts.profiles.get_prompt_catalog_path", lambda: catalog
    )
    monkeypatch.setattr(
        "infinidev.prompts.profiles.get_prompt_profile_path",
        lambda: tmp_path / "missing-project.json",
    )

    configuration = EffectivePromptConfiguration.compile()

    assert catalog.is_dir()
    assert sorted(path.name for path in catalog.iterdir()) == [
        filename for filename, _content in STARTER_PROMPT_PROFILES
    ]
    optional_names = {name for name, _body in OPTIONAL_CAPABILITIES}
    for profile_path in catalog.glob("*.json"):
        document = json.loads(profile_path.read_text(encoding="utf-8"))
        for entries in document.values():
            for name, entry in entries.items():
                assert entry == {"enabled_by_default": name not in optional_names}

    for phase, name in (
        ("develop", "loop.identity"),
        ("plan", "task_planner.methodology"),
        ("review", "reviewer.evaluation_guidance"),
    ):
        profile = configuration.resolve(phase, name)
        assert profile.enabled is True
        assert profile.enabled_by_default is True


@pytest.mark.parametrize("error_code", [errno.EACCES, errno.EPERM, errno.EROFS])
def test_read_only_catalog_uses_packaged_defaults_and_existing_overrides(
    tmp_path, monkeypatch, error_code,
) -> None:
    from infinidev.prompts import profiles

    catalog = tmp_path / "prompts"
    catalog.mkdir()
    (catalog / "99-custom.json").write_text(
        json.dumps({"develop": {"loop.identity": False}}), encoding="utf-8",
    )
    project = tmp_path / "project.json"
    project.write_text(
        json.dumps({"develop": {"loop.protocol": False}}), encoding="utf-8",
    )
    monkeypatch.setattr(profiles, "get_prompt_catalog_path", lambda: catalog)
    monkeypatch.setattr(profiles, "get_prompt_profile_path", lambda: project)

    def cannot_publish(path):
        raise OSError(error_code, "read-only catalog")

    monkeypatch.setattr(profiles, "materialize_starter_prompt_profiles", cannot_publish)

    configuration = EffectivePromptConfiguration.compile()

    assert not configuration.resolve("develop", "loop.identity").enabled
    assert not configuration.resolve("develop", "loop.protocol").enabled
    for name, _body in OPTIONAL_CAPABILITIES:
        assert not configuration.resolve("develop", name).enabled
    assert len(list(catalog.iterdir())) == 1


def test_catalog_publication_io_errors_are_not_hidden(tmp_path, monkeypatch) -> None:
    from infinidev.prompts import profiles

    monkeypatch.setattr(profiles, "get_prompt_catalog_path", lambda: tmp_path)

    def broken_storage(path):
        raise OSError(errno.EIO, "storage failed")

    monkeypatch.setattr(profiles, "materialize_starter_prompt_profiles", broken_storage)
    with pytest.raises(OSError, match="storage failed"):
        EffectivePromptConfiguration.compile()


def test_default_catalog_publishes_complete_starters_during_concurrent_startup(
    tmp_path, monkeypatch,
) -> None:
    catalog = tmp_path / "missing" / "prompts"
    expected_contents = dict(STARTER_PROMPT_PROFILES)

    def unsupported_link(*_args, **_kwargs) -> None:
        raise OSError("hard links are unavailable")

    monkeypatch.setattr(prompt_catalog.os, "link", unsupported_link)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = tuple(executor.map(
            prompt_catalog.materialize_starter_prompt_profiles,
            [catalog] * 4,
        ))

    assert sum(len(created) for created in results) == len(STARTER_PROMPT_PROFILES)
    assert {
        path.name: path.read_text(encoding="utf-8") for path in catalog.iterdir()
    } == expected_contents


def test_concurrent_materializer_waits_for_complete_starter_write(
    tmp_path, monkeypatch,
) -> None:
    catalog = tmp_path / "missing" / "prompts"
    first_filename, first_content = STARTER_PROMPT_PROFILES[0]
    writer_paused = Event()
    release_writer = Event()
    second_started = Event()
    original_write = prompt_catalog._write_starter_profile
    paused_once = False

    def paused_write(target: Path, content: str) -> None:
        nonlocal paused_once
        if paused_once:
            original_write(target, content)
            return
        paused_once = True
        with target.open("x", encoding="utf-8") as profile_file:
            profile_file.write(content[:1])
            profile_file.flush()
            writer_paused.set()
            assert release_writer.wait(timeout=2.0)
            profile_file.write(content[1:])
            profile_file.flush()
            os.fsync(profile_file.fileno())

    def materialize_after_signal() -> tuple[Path, ...]:
        second_started.set()
        return prompt_catalog.materialize_starter_prompt_profiles(catalog)

    monkeypatch.setattr(prompt_catalog, "_write_starter_profile", paused_write)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(prompt_catalog.materialize_starter_prompt_profiles, catalog)
        assert writer_paused.wait(timeout=2.0)
        assert (catalog / first_filename).read_text(encoding="utf-8") == first_content[:1]

        second = executor.submit(materialize_after_signal)
        assert second_started.wait(timeout=2.0)
        assert not second.done()

        release_writer.set()
        first_created = first.result(timeout=2.0)
        second_created = second.result(timeout=2.0)

    assert len(first_created) == len(STARTER_PROMPT_PROFILES)
    assert second_created == ()
    assert (catalog / first_filename).read_text(encoding="utf-8") == first_content


def test_starter_materialization_syncs_catalog_directory(tmp_path, monkeypatch) -> None:
    catalog = tmp_path / "prompts"
    synced: list[Path] = []
    monkeypatch.setattr(prompt_catalog, "fsync_directory", synced.append)

    created = prompt_catalog.materialize_starter_prompt_profiles(catalog)

    assert len(created) == len(STARTER_PROMPT_PROFILES)
    assert synced == [catalog]

    assert prompt_catalog.materialize_starter_prompt_profiles(catalog) == ()
    assert synced == [catalog]


def test_failed_starter_directory_sync_rolls_back_new_files_and_allows_retry(
    tmp_path, monkeypatch,
) -> None:
    catalog = tmp_path / "prompts"
    catalog.mkdir()
    existing_filename, existing_content = STARTER_PROMPT_PROFILES[0]
    existing = catalog / existing_filename
    existing.write_text(existing_content, encoding="utf-8")
    custom = catalog / "70-team-conventions.json"
    custom.write_text('{"develop":{"loop.identity":true}}\n', encoding="utf-8")

    sync_attempts: list[Path] = []

    def fail_directory_sync(path: Path) -> None:
        sync_attempts.append(path)
        if len(sync_attempts) == 1:
            raise OSError("directory sync failed")

    monkeypatch.setattr(prompt_catalog, "fsync_directory", fail_directory_sync)

    with pytest.raises(OSError, match="directory sync failed"):
        prompt_catalog.materialize_starter_prompt_profiles(catalog)

    assert sync_attempts == [catalog, catalog]

    assert existing.read_text(encoding="utf-8") == existing_content
    assert custom.read_text(encoding="utf-8") == '{"develop":{"loop.identity":true}}\n'
    assert sorted(path.name for path in catalog.glob("*.json")) == [
        existing_filename,
        custom.name,
    ]

    synced: list[Path] = []
    monkeypatch.setattr(prompt_catalog, "fsync_directory", synced.append)
    created = prompt_catalog.materialize_starter_prompt_profiles(catalog)

    assert created == tuple(catalog / filename for filename, _ in STARTER_PROMPT_PROFILES[1:])
    assert synced == [catalog]


def test_failed_starter_write_removes_partial_destination_and_allows_retry(
    tmp_path, monkeypatch,
) -> None:
    catalog = tmp_path / "prompts"
    catalog.mkdir()
    existing_filename, _existing_content = STARTER_PROMPT_PROFILES[0]
    existing = catalog / existing_filename
    custom_content = '{"develop":{"loop.identity":false}}\n'
    existing.write_text(custom_content, encoding="utf-8")
    failed_filename, _failed_content = STARTER_PROMPT_PROFILES[1]
    failed_target = catalog / failed_filename
    original_fsync = prompt_catalog.os.fsync

    def failed_fsync(_file_descriptor: int) -> None:
        raise OSError("simulated storage failure")

    monkeypatch.setattr(prompt_catalog.os, "fsync", failed_fsync)

    with pytest.raises(OSError, match="simulated storage failure"):
        prompt_catalog.materialize_starter_prompt_profiles(catalog)

    assert not failed_target.exists()
    assert existing.read_text(encoding="utf-8") == custom_content

    monkeypatch.setattr(prompt_catalog.os, "fsync", original_fsync)
    created = prompt_catalog.materialize_starter_prompt_profiles(catalog)

    assert created == tuple(catalog / filename for filename, _ in STARTER_PROMPT_PROFILES[1:])
    assert existing.read_text(encoding="utf-8") == custom_content
    for target, (_filename, content) in zip(created, STARTER_PROMPT_PROFILES[1:]):
        assert target.read_text(encoding="utf-8") == content


def test_default_catalog_completes_starters_alongside_custom_profile(
    tmp_path, monkeypatch,
) -> None:
    catalog = tmp_path / "prompts"
    catalog.mkdir()
    custom = catalog / "70-team-conventions.json"
    custom_content = json.dumps({"develop": {"loop.identity": True}})
    custom.write_text(custom_content, encoding="utf-8")
    monkeypatch.setattr(
        "infinidev.prompts.profiles.get_prompt_catalog_path", lambda: catalog
    )
    monkeypatch.setattr(
        "infinidev.prompts.profiles.get_prompt_profile_path",
        lambda: tmp_path / "missing-project.json",
    )

    EffectivePromptConfiguration.compile()

    assert custom.read_text(encoding="utf-8") == custom_content
    assert all((catalog / filename).is_file() for filename, _ in STARTER_PROMPT_PROFILES)


def test_default_catalog_recovers_interrupted_starter_materialization(
    tmp_path, monkeypatch,
) -> None:
    catalog = tmp_path / "prompts"
    catalog.mkdir()
    first_filename, first_content = STARTER_PROMPT_PROFILES[0]
    (catalog / first_filename).write_text(first_content, encoding="utf-8")
    monkeypatch.setattr(
        "infinidev.prompts.profiles.get_prompt_catalog_path", lambda: catalog
    )
    monkeypatch.setattr(
        "infinidev.prompts.profiles.get_prompt_profile_path",
        lambda: tmp_path / "missing-project.json",
    )

    configuration = EffectivePromptConfiguration.compile()

    assert sorted(path.name for path in catalog.iterdir()) == [
        filename for filename, _content in STARTER_PROMPT_PROFILES
    ]
    assert configuration.resolve("develop", "loop.identity").enabled is True
    assert configuration.resolve("review", "reviewer.identity").enabled is True


def test_default_catalog_upgrade_adds_new_starter_without_touching_customizations(
    tmp_path,
) -> None:
    catalog = tmp_path / "prompts"
    catalog.mkdir()
    legacy_starters = STARTER_PROMPT_PROFILES[:-1]
    for filename, content in legacy_starters:
        (catalog / filename).write_text(content, encoding="utf-8")

    customized = catalog / legacy_starters[0][0]
    custom_content = json.dumps({"develop": {"loop.identity": False}})
    customized.write_text(custom_content, encoding="utf-8")
    user_profile = catalog / "70-team-conventions.json"
    user_content = '{"develop":{"loop.protocol":{"enabled":false}}}\n'
    user_profile.write_text(user_content, encoding="utf-8")
    new_filename, new_content = STARTER_PROMPT_PROFILES[-1]

    created = prompt_catalog.materialize_starter_prompt_profiles(catalog)

    assert created == (catalog / new_filename,)
    assert (catalog / new_filename).read_text(encoding="utf-8") == new_content
    assert customized.read_text(encoding="utf-8") == custom_content
    assert user_profile.read_text(encoding="utf-8") == user_content


def test_default_catalog_recovers_legacy_override_only_directory(
    tmp_path, monkeypatch,
) -> None:
    catalog = tmp_path / "prompts"
    catalog.mkdir()
    override = catalog / USER_OVERRIDES_FILE
    content = json.dumps({"develop": {"capability.testing": {"enabled": True}}})
    override.write_text(content, encoding="utf-8")
    monkeypatch.setattr(
        "infinidev.prompts.profiles.get_prompt_catalog_path", lambda: catalog
    )
    monkeypatch.setattr(
        "infinidev.prompts.profiles.get_prompt_profile_path",
        lambda: tmp_path / "missing-project.json",
    )

    configuration = EffectivePromptConfiguration.compile()

    assert sorted(path.name for path in catalog.iterdir()) == sorted(
        [filename for filename, _content in STARTER_PROMPT_PROFILES] + [USER_OVERRIDES_FILE]
    )
    assert override.read_text(encoding="utf-8") == content
    assert configuration.resolve("develop", "capability.testing").enabled is True
    assert configuration.resolve("develop", "loop.identity").enabled is True


def test_default_catalog_never_overwrites_existing_starter_path(
    tmp_path, monkeypatch,
) -> None:
    catalog = tmp_path / "prompts"
    catalog.mkdir()
    existing = catalog / "10-development.json"
    content = json.dumps({"develop": {"loop.identity": False}})
    existing.write_text(content, encoding="utf-8")
    monkeypatch.setattr(
        "infinidev.prompts.profiles.get_prompt_catalog_path", lambda: catalog
    )
    monkeypatch.setattr(
        "infinidev.prompts.profiles.get_prompt_profile_path",
        lambda: tmp_path / "missing-project.json",
    )

    configuration = EffectivePromptConfiguration.compile()

    assert existing.read_text(encoding="utf-8") == content
    assert all((catalog / filename).exists() for filename, _ in STARTER_PROMPT_PROFILES)
    assert configuration.resolve("develop", "loop.identity").enabled is False


def test_catalog_validation_error_names_source_file(tmp_path, monkeypatch) -> None:
    catalog = tmp_path / "prompts"
    catalog.mkdir()
    invalid = catalog / "broken.json"
    invalid.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        "infinidev.prompts.profiles.get_prompt_catalog_path", lambda: catalog
    )
    monkeypatch.setattr(
        "infinidev.prompts.profiles.get_prompt_profile_path",
        lambda: tmp_path / "missing-project.json",
    )

    with pytest.raises(PromptProfileError, match=re.escape(str(invalid))):
        EffectivePromptConfiguration.compile()


def test_optional_capabilities_are_neutral_until_enabled(tmp_path) -> None:
    configuration = _configuration(tmp_path, {})

    prompt = build_system_prompt("", prompt_configuration=configuration)

    assert all(body not in prompt for _name, body in OPTIONAL_CAPABILITIES)


def test_optional_capability_opt_in_reaches_developer_system_prompt(tmp_path) -> None:
    capability_name, capability_body = OPTIONAL_CAPABILITIES[0]
    configuration = _configuration(
        tmp_path,
        {"develop": {capability_name: {"enabled_by_default": False, "enabled": True}}},
    )

    prompt = build_system_prompt("", prompt_configuration=configuration)

    assert capability_body in prompt
    assert all(body not in prompt for _name, body in OPTIONAL_CAPABILITIES[1:])


def test_optional_capability_opt_in_reaches_small_model_prompt(tmp_path) -> None:
    capability_name, capability_body = OPTIONAL_CAPABILITIES[0]
    configuration = _configuration(
        tmp_path,
        {"develop": {capability_name: {"enabled": True}}},
    )

    prompt = build_system_prompt(
        "",
        prompt_configuration=configuration,
        small_model=True,
    )

    assert capability_body in prompt
    assert all(body not in prompt for _name, body in OPTIONAL_CAPABILITIES[1:])


def test_correctness_boundary_capabilities_compose_independently(tmp_path) -> None:
    capability_names = {
        "capability.requirements_clarity",
        "capability.architecture_impact",
        "capability.api_contracts",
        "capability.concurrency",
        "capability.resource_lifecycle",
        "capability.error_recovery",
        "capability.test_quality",
        "capability.algorithmic_correctness",
    }
    capabilities = dict(OPTIONAL_CAPABILITIES)

    assert capability_names <= capabilities.keys()
    for capability_name in capability_names:
        configuration = _configuration(
            tmp_path,
            {"develop": {capability_name: {"enabled": True}}},
        )

        prompt = build_system_prompt("", prompt_configuration=configuration)

        assert capabilities[capability_name] in prompt
        assert all(
            body not in prompt
            for name, body in OPTIONAL_CAPABILITIES
            if name != capability_name
        )


def test_production_boundary_capabilities_compose_independently(tmp_path) -> None:
    capability_names = {
        "capability.observability",
        "capability.release_readiness",
        "capability.dependency_change",
        "capability.configuration_change",
        "capability.data_integrity",
        "capability.privacy",
        "capability.incident_response",
    }
    capabilities = dict(OPTIONAL_CAPABILITIES)

    assert capability_names <= capabilities.keys()
    for capability_name in capability_names:
        configuration = _configuration(
            tmp_path,
            {"develop": {capability_name: {"enabled": True}}},
        )

        prompt = build_system_prompt("", prompt_configuration=configuration)

        assert capabilities[capability_name] in prompt
        assert all(
            body not in prompt
            for name, body in OPTIONAL_CAPABILITIES
            if name != capability_name
        )


def test_interaction_capabilities_compose_independently(tmp_path) -> None:
    capability_names = {
        "capability.cross_platform",
        "capability.cli_ux",
        "capability.ui_state_completeness",
        "capability.browser_verification",
    }
    capabilities = dict(OPTIONAL_CAPABILITIES)

    assert capability_names <= capabilities.keys()
    for capability_name in capability_names:
        configuration = _configuration(
            tmp_path,
            {"develop": {capability_name: {"enabled": True}}},
        )

        prompt = build_system_prompt("", prompt_configuration=configuration)

        assert capabilities[capability_name] in prompt
        assert all(
            body not in prompt
            for name, body in OPTIONAL_CAPABILITIES
            if name != capability_name
        )


def test_continuity_and_localization_capabilities_compose_independently(tmp_path) -> None:
    capability_names = {
        "capability.localization",
        "capability.handoff",
        "capability.context_management",
    }
    capabilities = dict(OPTIONAL_CAPABILITIES)

    assert capability_names <= capabilities.keys()
    for capability_name in capability_names:
        configuration = _configuration(
            tmp_path,
            {"develop": {capability_name: {"enabled": True}}},
        )

        prompt = build_system_prompt("", prompt_configuration=configuration)

        assert capabilities[capability_name] in prompt
        assert all(
            body not in prompt
            for name, body in OPTIONAL_CAPABILITIES
            if name != capability_name
        )


def test_task_mode_capabilities_compose_independently(tmp_path) -> None:
    capability_names = {
        "capability.research_evidence",
        "capability.refactoring_discipline",
        "capability.dead_code_cleanup",
        "capability.git_hygiene",
        "capability.cost_awareness",
        "capability.mentoring",
        "capability.root_cause_analysis",
        "capability.ai_system_evaluation",
    }
    capabilities = dict(OPTIONAL_CAPABILITIES)

    assert capability_names <= capabilities.keys()
    for capability_name in capability_names:
        configuration = _configuration(
            tmp_path,
            {"develop": {capability_name: {"enabled": True}}},
        )

        prompt = build_system_prompt("", prompt_configuration=configuration)

        assert capabilities[capability_name] in prompt
        assert all(
            body not in prompt
            for name, body in OPTIONAL_CAPABILITIES
            if name != capability_name
        )


def test_complaint_mitigations_compose_independently_when_enabled(tmp_path) -> None:
    mitigation_names = {
        "capability.authority_boundaries",
        "capability.verification_integrity",
        "capability.change_preservation",
        "capability.progress_discipline",
    }
    capabilities = dict(OPTIONAL_CAPABILITIES)

    assert mitigation_names <= capabilities.keys()
    for capability_name in mitigation_names:
        configuration = _configuration(
            tmp_path,
            {
                "develop": {
                    capability_name: {
                        "enabled_by_default": False,
                        "enabled": True,
                    }
                }
            },
        )

        prompt = build_system_prompt("", prompt_configuration=configuration)

        assert capabilities[capability_name] in prompt
        assert all(
            body not in prompt
            for name, body in OPTIONAL_CAPABILITIES
            if name != capability_name
        )


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


def test_system_prompt_compiles_default_configuration_once(tmp_path, monkeypatch) -> None:
    from infinidev.prompts import profiles

    path = tmp_path / "prompts.json"
    path.write_text("{}", encoding="utf-8")
    real_loader = profiles.load_prompt_profiles
    reads = 0

    def tracked_loader(profile_path=None):
        nonlocal reads
        reads += 1
        return real_loader(profile_path)

    monkeypatch.setattr(profiles, "load_prompt_profiles", tracked_loader)
    monkeypatch.setattr(profiles, "get_prompt_profile_path", lambda: path)

    build_system_prompt("")

    assert reads == 1


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
    # This test is about which blocks a compiled configuration includes, so it
    # pins the style it asserts wording from instead of tracking the default.
    from infinidev.config.settings import settings as _settings

    monkeypatch.setattr(_settings, "PROMPT_STYLE", "generalized")

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


@pytest.mark.parametrize(
    "setting, message",
    [
        ({"enabled_by_default": "yes"}, "boolean enabled fields"),
        ({"enabled_by_default": True, "unknown": 1}, "unknown fields"),
        ({"enabled_by_default": True, "parameters": {"nested": {}}}, "invalid parameters"),
    ],
)
def test_invalid_structured_setting_is_rejected(tmp_path, setting, message) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps({"develop": {"loop.identity": setting}}), encoding="utf-8")

    with pytest.raises(PromptProfileError, match=message):
        resolve_prompt_profile("develop", "loop.identity", path)
