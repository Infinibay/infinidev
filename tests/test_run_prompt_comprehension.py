from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from bench.prompt_comprehension import ComprehensionCondition
from bench.run_prompt_comprehension import (
    AttemptResult,
    ComprehensionRunConfig,
    ProviderReply,
    append_attempt,
    preflight_baseline,
    run_one,
    run_sequentially,
)
from tests.test_prompt_comprehension import _case


def _reply() -> ProviderReply:
    payload = {
        "understanding": "Prepare a migration but do not deploy it.",
        "objective": "Prepare the migration.",
        "deliverables": [],
        "constraints": ["Do not deploy."],
        "user_owned_decisions": [],
        "authorized_actions": ["Inspect files."],
        "unauthorized_actions": ["Deploy."],
        "verification": ["Run migration tests."],
        "ambiguities": ["Target schema is unknown."],
        "stop_conditions": ["Ask before deployment."],
        "conflicts": [],
        "priority_resolution": "",
        "interpretation_risks": ["Deploying despite the prohibition."],
        "confidence": 0.9,
    }
    return ProviderReply(json.dumps(payload), 100, 80)


def _write_cases(path: Path, *, approved: bool = True, count: int = 2) -> str:
    cases = [
        replace(
            _case(),
            id=f"case-{index}",
            review_status="approved" if approved else "draft",
        )
        for index in range(count)
    ]
    path.write_text(
        "".join(json.dumps(asdict(case), sort_keys=True) + "\n" for case in cases),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(dataset_sha256: str) -> dict[str, object]:
    return {
        "provider": "provider",
        "model": "model",
        "revision": "revision-2025-01-01",
        "model_identity": "provider:model:revision-2025-01-01",
        "manifest_id": "baseline-manifest-1",
        "approved_dataset_sha256": dataset_sha256,
        "conditions": {"raw": None},
    }


def _write_manifest(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_config_defaults_to_one_second_and_accepts_exactly_raw() -> None:
    config = ComprehensionRunConfig.from_dict(_manifest("a" * 64))

    assert config.min_request_interval_seconds == 1.0
    assert config.conditions == (ComprehensionCondition("raw"),)
    assert config.revision == "revision-2025-01-01"
    assert config.dataset_stage == "approved"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(conditions={"raw": {}, "behavior": None}), "exactly raw"),
        (lambda value: value.update(conditions={"raw": {}}), "exactly raw"),
        (lambda value: value.update(provider=""), "explicit provider"),
        (lambda value: value.update(model=""), "explicit provider"),
        (lambda value: value.update(revision=""), "explicit provider"),
        (lambda value: value.update(model_identity=""), "explicit provider"),
        (lambda value: value.update(manifest_id=""), "manifest_id"),
        (lambda value: value.update(approved_dataset_sha256="not-a-hash"), "SHA-256"),
    ],
    ids=[
        "extra-condition",
        "raw-with-prompt",
        "provider-missing",
        "model-missing",
        "revision-missing",
        "identity-missing",
        "manifest-id-missing",
        "approved-hash-invalid",
    ],
)
def test_config_rejects_every_non_baseline_identity(
    mutation: object,
    match: str,
) -> None:
    """The baseline manifest fails closed unless all pinned raw-only fields are present."""
    value = _manifest("a" * 64)
    assert callable(mutation)
    mutation(value)

    with pytest.raises(ValueError, match=match):
        ComprehensionRunConfig.from_dict(value)


def test_preflight_rejects_unapproved_or_substituted_dataset_before_claim(
    tmp_path: Path,
) -> None:
    """Review status and the approved byte hash are both mandatory before provider use."""
    cases_path = tmp_path / "cases.jsonl"
    dataset_sha = _write_cases(cases_path, approved=False)
    manifest_path = tmp_path / "manifest.json"
    ledger_path = tmp_path / "ledger.jsonl"
    _write_manifest(manifest_path, _manifest(dataset_sha))

    with pytest.raises(ValueError, match="approved dataset must contain only approved cases"):
        preflight_baseline(cases_path, manifest_path, ledger_path, resume=False)

    dataset_sha = _write_cases(cases_path, approved=True)
    substituted = _manifest("b" * 64)
    _write_manifest(manifest_path, substituted)
    with pytest.raises(ValueError, match="do not match the manifest SHA-256"):
        preflight_baseline(cases_path, manifest_path, ledger_path, resume=False)
    assert dataset_sha != substituted["approved_dataset_sha256"]


def test_preflight_allows_explicit_hash_pinned_exploratory_draft(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    dataset_sha = _write_cases(cases_path, approved=False)
    manifest_path = tmp_path / "manifest.json"
    ledger_path = tmp_path / "ledger.jsonl"
    manifest = _manifest(dataset_sha)
    manifest.pop("approved_dataset_sha256")
    manifest.update(
        dataset_sha256=dataset_sha,
        dataset_stage="exploratory_draft",
        manifest_id="exploratory-manifest-1",
    )
    _write_manifest(manifest_path, manifest)

    plan = preflight_baseline(
        cases_path,
        manifest_path,
        ledger_path,
        resume=False,
        split="all",
    )

    assert plan.config.dataset_stage == "exploratory_draft"
    assert len(plan.cases) == 2


def test_preflight_claims_manifest_exclusively_and_resume_requires_same_bytes(
    tmp_path: Path,
) -> None:
    """A manifest is new exactly once and its byte identity cannot change on resume."""
    cases_path = tmp_path / "cases.jsonl"
    dataset_sha = _write_cases(cases_path)
    manifest_path = tmp_path / "manifest.json"
    ledger_path = tmp_path / "ledger.jsonl"
    manifest = _manifest(dataset_sha)
    _write_manifest(manifest_path, manifest)

    first = preflight_baseline(cases_path, manifest_path, ledger_path, resume=False)
    assert first.terminal_tuple_ids == frozenset()

    with pytest.raises(ValueError, match="manifest is not new"):
        preflight_baseline(cases_path, manifest_path, ledger_path, resume=False)

    resumed = preflight_baseline(cases_path, manifest_path, ledger_path, resume=True)
    assert resumed.manifest_sha256 == first.manifest_sha256

    manifest["manifest_id"] = "changed-after-claim"
    _write_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="manifest claim changed"):
        preflight_baseline(cases_path, manifest_path, ledger_path, resume=True)


def test_preflight_all_split_selects_every_approved_case(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases = [
        replace(_case(), id="calibration", split="calibration", review_status="approved"),
        replace(_case(), id="validation", split="validation", review_status="approved"),
    ]
    cases_path.write_text(
        "".join(json.dumps(asdict(case), sort_keys=True) + "\n" for case in cases),
        encoding="utf-8",
    )
    dataset_sha = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    ledger_path = tmp_path / "ledger.jsonl"
    _write_manifest(manifest_path, _manifest(dataset_sha))

    plan = preflight_baseline(
        cases_path,
        manifest_path,
        ledger_path,
        resume=False,
        split="all",
    )

    assert [case.id for case in plan.cases] == ["calibration", "validation"]


def test_runner_persists_rate_limit_then_resumes_without_repeating_tuple(
    tmp_path: Path,
) -> None:
    """Every attempted tuple is durable and terminal before stop or later resume."""
    cases_path = tmp_path / "cases.jsonl"
    dataset_sha = _write_cases(cases_path)
    manifest_path = tmp_path / "manifest.json"
    ledger_path = tmp_path / "ledger.jsonl"
    _write_manifest(manifest_path, _manifest(dataset_sha))
    plan = preflight_baseline(cases_path, manifest_path, ledger_path, resume=False)

    def rate_limited(*_: object) -> ProviderReply:
        raise RuntimeError("429 rate limit exceeded")

    count = run_sequentially(
        plan.config,
        plan.cases,
        rate_limited,
        lambda result: append_attempt(
            ledger_path,
            result,
            plan.config,
            manifest_sha256=plan.manifest_sha256,
            dataset_sha256=plan.dataset_sha256,
        ),
        dataset_sha256=plan.dataset_sha256,
        manifest_sha256=plan.manifest_sha256,
        terminal_tuple_ids=plan.terminal_tuple_ids,
    )

    assert count == 1
    first_rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert first_rows[0]["status"] == "failure"
    assert first_rows[0]["failure"]["type"] == "provider_error"
    assert first_rows[0]["terminal"] is True

    resumed = preflight_baseline(cases_path, manifest_path, ledger_path, resume=True)
    seen_case_ids: list[str] = []

    def success(
        _config: ComprehensionRunConfig,
        _condition: ComprehensionCondition,
        case: object,
    ) -> ProviderReply:
        seen_case_ids.append(str(getattr(case, "id")))
        return _reply()

    resumed_count = run_sequentially(
        resumed.config,
        resumed.cases,
        success,
        lambda result: append_attempt(
            ledger_path,
            result,
            resumed.config,
            manifest_sha256=resumed.manifest_sha256,
            dataset_sha256=resumed.dataset_sha256,
        ),
        dataset_sha256=resumed.dataset_sha256,
        manifest_sha256=resumed.manifest_sha256,
        terminal_tuple_ids=resumed.terminal_tuple_ids,
    )

    final_rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert resumed_count == 1
    assert seen_case_ids == ["case-1"]
    assert len(final_rows) == 2
    assert len({row["tuple_id"] for row in final_rows}) == 2
    assert [row["status"] for row in final_rows] == ["failure", "success"]


def test_run_one_separates_provider_and_parse_failures() -> None:
    config = ComprehensionRunConfig.from_dict(_manifest("a" * 64))
    condition = config.conditions[0]

    provider_result = run_one(
        config,
        condition,
        _case(),
        lambda *_: (_ for _ in ()).throw(ConnectionError("offline")),
        dataset_sha256="a" * 64,
    )
    parse_result = run_one(
        config,
        condition,
        _case(),
        lambda *_: ProviderReply("not JSON", 1, 2),
        dataset_sha256="a" * 64,
    )

    assert isinstance(provider_result, AttemptResult)
    assert provider_result.failure_type == "provider_error"
    assert parse_result.failure_type == "parse_error"
    assert parse_result.observation.response_text == "not JSON"


def test_runner_is_sequential_and_paces_every_raw_call() -> None:
    config = ComprehensionRunConfig.from_dict(_manifest("a" * 64))
    cases = [_case(), replace(_case(), id="second-case")]
    clock = iter([0.0, 0.0, 0.2, 1.0])
    sleeps: list[float] = []
    observed: list[AttemptResult] = []

    count = run_sequentially(
        config,
        cases,
        lambda *_: _reply(),
        observed.append,
        dataset_sha256="a" * 64,
        manifest_sha256="b" * 64,
        sleep=sleeps.append,
        monotonic=lambda: next(clock),
    )

    assert count == 2
    assert len(observed) == 2
    assert sleeps == [0.8]
    assert [item.observation.condition for item in observed] == ["raw", "raw"]


def test_runner_can_stop_after_a_bounded_provider_pilot() -> None:
    config = ComprehensionRunConfig.from_dict(_manifest("a" * 64))
    cases = [_case(), replace(_case(), id="second-case")]
    observed: list[AttemptResult] = []

    count = run_sequentially(
        config,
        cases,
        lambda *_: _reply(),
        observed.append,
        dataset_sha256="a" * 64,
        manifest_sha256="b" * 64,
        max_cases=1,
    )

    assert count == 1
    assert [item.observation.case_id for item in observed] == [_case().id]


def test_runner_stops_after_first_non_rate_limit_provider_error() -> None:
    config = ComprehensionRunConfig.from_dict(_manifest("a" * 64))
    cases = [_case(), replace(_case(), id="must-not-run")]
    observed: list[AttemptResult] = []

    count = run_sequentially(
        config,
        cases,
        lambda *_: (_ for _ in ()).throw(ConnectionError("provider offline")),
        observed.append,
        dataset_sha256="a" * 64,
        manifest_sha256="b" * 64,
    )

    assert count == 1
    assert len(observed) == 1
    assert observed[0].failure_type == "provider_error"
