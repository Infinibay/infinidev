from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

from bench.prompt_comprehension import ComprehensionCase, ComprehensionObservation
from bench.prompt_comprehension_report import (
    build_report,
    canonical_json_bytes,
    load_observations,
    render_markdown,
)
from tests.test_prompt_comprehension import _case


def test_report_preserves_raw_answers_and_treats_numbers_as_collection_only(
    tmp_path: Path,
) -> None:
    cases = tmp_path / "cases.jsonl"
    case = _case()
    cases.write_text(json.dumps({**case.__dict__, "tags": list(case.tags)}) + "\n")
    dataset_sha = hashlib.sha256(cases.read_bytes()).hexdigest()
    observation = ComprehensionObservation(
        case_id=case.id,
        category=case.category,
        condition="raw",
        condition_sha256="condition",
        model_identity="provider:model:snapshot",
        response_text="raw provider bytes",
        parsed={"understanding": "Do not deploy.", "objective": "Plan."},
        latency_seconds=1.0,
        input_tokens=100,
        output_tokens=20,
        dataset_sha256=dataset_sha,
    )

    report = build_report(cases, [observation])

    record = report["categories"]["requirements"][0]
    assert record["observation"]["response_text"] == "raw provider bytes"
    assert "no globally optimal" in report["interpretation_boundary"]
    family = report["family_analysis"][case.id]
    assert family["variants"][0]["case"]["variant_id"] == "anchor"
    markdown = render_markdown(report)
    assert "Do not deploy." in markdown
    assert "Controlled families" in markdown
    assert "Reviewed interpretation key" in markdown


def _write_cases(path: Path, cases: list[ComprehensionCase]) -> str:
    path.write_text(
        "".join(json.dumps(asdict(case), sort_keys=True) + "\n" for case in cases),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _observation(
    case: ComprehensionCase,
    dataset_sha256: str,
    *,
    response_text: str = "provider response",
    parsed: dict[str, object] | None = None,
    error: str = "",
) -> ComprehensionObservation:
    return ComprehensionObservation(
        case_id=case.id,
        category=case.category,
        condition="raw",
        condition_sha256="raw-condition-sha",
        model_identity="provider:model:revision-1",
        response_text=response_text,
        parsed=parsed or {},
        latency_seconds=0.5,
        input_tokens=10,
        output_tokens=5,
        error=error,
        dataset_sha256=dataset_sha256,
    )


def test_report_separates_typed_failures_and_preserves_registry_groups(tmp_path: Path) -> None:
    """Provider and parse failures remain distinct within family and scenario evidence."""
    anchor = replace(
        _case(),
        id="case-anchor",
        problem_id="scenario-reviewed",
        family_id="family-reviewed",
        variant_id="anchor",
        intended_relation="anchor",
    )
    equivalent = replace(
        anchor,
        id="case-equivalent",
        variant_id="equivalent-1",
        intended_relation="equivalent",
    )
    contrast = replace(
        anchor,
        id="case-contrast",
        variant_id="contrast-1",
        intended_relation="contrast",
    )
    cases_path = tmp_path / "cases.jsonl"
    dataset_sha = _write_cases(cases_path, [contrast, anchor, equivalent])
    observations = [
        _observation(
            anchor,
            dataset_sha,
            parsed={"understanding": "Preserve the constraint.", "objective": "Plan safely."},
        ),
        _observation(
            equivalent,
            dataset_sha,
            error="ValueError: response did not contain a JSON object",
        ),
        _observation(
            contrast,
            dataset_sha,
            error="RateLimitError: provider quota exhausted",
        ),
    ]

    report = build_report(cases_path, observations)

    summary = report["condition_summaries"]["raw"]
    assert summary == {
        "calls": 3,
        "successes": 1,
        "provider_errors": 1,
        "parse_errors": 1,
        "evidence_status": "evidencia insuficiente",
        "mean_latency_seconds": 0.5,
        "input_tokens": 30,
        "output_tokens": 15,
    }
    family = report["family_analysis"]["family-reviewed"]
    scenario = report["scenario_analysis"]["scenario-reviewed"]
    assert family["evidence_status"] == "evidencia insuficiente"
    assert scenario["evidence_status"] == "evidencia insuficiente"
    assert family["registry"]["variant_ids"] == ["anchor", "contrast-1", "equivalent-1"]
    assert family["registry"]["intended_relations"] == [
        "anchor",
        "contrast",
        "equivalent",
    ]
    outcomes = {
        record["case"]["id"]: record["outcome"]["error_type"]
        for record in family["records"]
    }
    assert outcomes == {
        "case-anchor": None,
        "case-contrast": "provider_error",
        "case-equivalent": "parse_error",
    }
    assert "score" not in report
    assert "global_score" not in report
    assert all("score" not in summary for summary in report["condition_summaries"].values())
    markdown = render_markdown(report)
    assert "evidencia insuficiente" in markdown
    assert "parse_error" in markdown
    assert "provider_error" in markdown


def test_two_independent_derivations_are_byte_identical(tmp_path: Path) -> None:
    """The same persisted raw inputs derive identical canonical JSON and Markdown bytes."""
    case = replace(
        _case(),
        id="stable-case",
        problem_id="stable-scenario",
        family_id="stable-family",
        variant_id="anchor",
    )
    cases_path = tmp_path / "cases.jsonl"
    dataset_sha = _write_cases(cases_path, [case])
    observation = _observation(
        case,
        dataset_sha,
        parsed={"understanding": "Stable reconstruction.", "objective": "Stable objective."},
    )
    observations_path = tmp_path / "observations.jsonl"
    observations_path.write_text(
        json.dumps(asdict(observation), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    derived: list[tuple[bytes, bytes]] = []
    for directory_name in ("derivation-a", "derivation-b"):
        output_dir = tmp_path / directory_name
        output_dir.mkdir()
        report = build_report(cases_path, load_observations(observations_path))
        json_path = output_dir / "report.json"
        markdown_path = output_dir / "report.md"
        json_path.write_bytes(canonical_json_bytes(report))
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
        derived.append((json_path.read_bytes(), markdown_path.read_bytes()))

    assert derived[0][0] == derived[1][0]
    assert derived[0][1] == derived[1][1]
    assert derived[0][0].endswith(b"\n")
    assert json.loads(derived[0][0])["schema_version"] == 2


def test_report_loads_durable_runner_ledger_and_family_research(tmp_path: Path) -> None:
    case = replace(_case(), id="ledger-case", family_id="ledger-family")
    cases_path = tmp_path / "cases.jsonl"
    dataset_sha = _write_cases(cases_path, [case])
    observation = _observation(
        case,
        dataset_sha,
        parsed={"understanding": "Ledger response.", "objective": "Inspect."},
    )
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "tuple_id": "a" * 64,
                "status": "success",
                "terminal": True,
                "failure": None,
                "observation": asdict(observation),
            }
        )
        + "\n"
    )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "families": [
                    {
                        "family_id": "ledger-family",
                        "research_question": "What did the model understand?",
                        "product_utility": "Improve objective delivery.",
                        "information_needed_about_model": "Concrete reconstruction.",
                        "possible_interventions": ["objective template"],
                    }
                ]
            }
        )
    )

    loaded = load_observations(ledger)
    report = build_report(cases_path, loaded, registry_path=registry)

    research = report["family_analysis"]["ledger-family"]["research"]
    assert research["research_question"] == "What did the model understand?"
    assert "Product utility" in render_markdown(report)
