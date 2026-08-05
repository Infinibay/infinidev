from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

from bench.prompt_comprehension_campaign_audit import audit_campaign, tuple_id


@dataclass(frozen=True)
class CampaignFixture:
    root: Path
    closure: Path
    dataset: Path
    manifest: Path
    ledger: Path
    dataset_sha256: str
    manifest_sha256: str


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join(_canonical(row) for row in rows))


def _campaign(tmp_path: Path) -> CampaignFixture:
    dataset = tmp_path / "approved.jsonl"
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    closure = tmp_path / "closure.json"
    dataset_rows = [
        {"id": "case-a", "review_status": "approved"},
        {"id": "case-b", "review_status": "approved"},
    ]
    _write_jsonl(dataset, dataset_rows)
    dataset_sha = _sha256(dataset.read_bytes())
    models = [
        {
            "provider": "provider-a",
            "model": "model-a",
            "revision": "revision-1",
            "model_identity": "provider-a:model-a:revision-1",
        },
        {
            "provider": "provider-b",
            "model": "model-b",
            "revision": "revision-2",
            "model_identity": "provider-b:model-b:revision-2",
        },
    ]
    manifest.write_bytes(
        _canonical(
            {
                "schema_version": 1,
                "manifest_id": "campaign-1",
                "condition": "raw",
                "dataset_sha256": dataset_sha,
                "models": models,
            }
        )
    )
    manifest_sha = _sha256(manifest.read_bytes())
    rows = []
    for case in dataset_rows:
        for model in models:
            model_key = (
                str(model["provider"]),
                str(model["model"]),
                str(model["revision"]),
                str(model["model_identity"]),
            )
            rows.append(
                {
                    "tuple_id": tuple_id(str(case["id"]), model_key, manifest_sha, dataset_sha),
                    "case_id": case["id"],
                    "condition": "raw",
                    "dataset_sha256": dataset_sha,
                    "manifest_sha256": manifest_sha,
                    **model,
                    "status": "success",
                    "terminal": True,
                    "failure": None,
                    "observation": {
                        "case_id": case["id"],
                        "category": "test",
                        "condition": "raw",
                        "condition_sha256": "raw-condition",
                        "model_identity": model["model_identity"],
                        "response_text": "raw response",
                        "parsed": {"understanding": "understood"},
                        "latency_seconds": 0.1,
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "error": "",
                        "dataset_sha256": dataset_sha,
                    },
                }
            )
    _write_jsonl(ledger, rows)
    closure.write_bytes(
        _canonical(
            {
                "schema_version": 1,
                "condition": "raw",
                "dataset": {
                    "path": dataset.name,
                    "sha256": dataset_sha,
                    "review_status": "approved",
                    "case_count": len(dataset_rows),
                },
                "manifest": {"path": manifest.name, "sha256": manifest_sha},
                "ledger": {"path": ledger.name, "sha256": _sha256(ledger.read_bytes())},
                "counts": {
                    "planned_tuples": len(rows),
                    "terminal_tuples": len(rows),
                    "successes": len(rows),
                    "failures": 0,
                    "pending": 0,
                },
            }
        )
    )
    return CampaignFixture(
        tmp_path,
        closure,
        dataset,
        manifest,
        ledger,
        dataset_sha,
        manifest_sha,
    )


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _reseal(fixture: CampaignFixture, artifact: str) -> None:
    closure = _load_json(fixture.closure)
    descriptor = closure[artifact]
    assert isinstance(descriptor, dict)
    path = getattr(fixture, artifact)
    descriptor["sha256"] = _sha256(path.read_bytes())
    fixture.closure.write_bytes(_canonical(closure))


def _audit(fixture: CampaignFixture) -> None:
    audit_campaign(
        fixture.closure,
        artifact_root=fixture.root,
        expected_dataset_sha256=fixture.dataset_sha256,
        expected_manifest_sha256=fixture.manifest_sha256,
    )


def test_audit_recalculates_bytes_and_accepts_exact_terminal_product(tmp_path: Path) -> None:
    """A complete case-by-model product with trusted hashes is accepted."""
    fixture = _campaign(tmp_path)

    result = audit_campaign(
        fixture.closure,
        artifact_root=fixture.root,
        expected_dataset_sha256=fixture.dataset_sha256,
        expected_manifest_sha256=fixture.manifest_sha256,
    )

    assert result.planned_tuples == 4
    assert result.successes == 4
    assert result.failures == 0
    assert result.dataset_sha256 == _sha256(fixture.dataset.read_bytes())
    assert result.manifest_sha256 == _sha256(fixture.manifest.read_bytes())
    assert result.ledger_sha256 == _sha256(fixture.ledger.read_bytes())


def _alter_bytes(fixture: CampaignFixture) -> None:
    fixture.ledger.write_bytes(fixture.ledger.read_bytes().replace(b'"case-a"', b'"case-x"', 1))


def _truncate(fixture: CampaignFixture) -> None:
    rows = _load_jsonl(fixture.ledger)
    _write_jsonl(fixture.ledger, rows[:-1])
    _reseal(fixture, "ledger")


def _duplicate(fixture: CampaignFixture) -> None:
    rows = _load_jsonl(fixture.ledger)
    _write_jsonl(fixture.ledger, [*rows, rows[0]])
    _reseal(fixture, "ledger")


def _extra(fixture: CampaignFixture) -> None:
    rows = _load_jsonl(fixture.ledger)
    extra = {**rows[0], "case_id": "case-extra", "tuple_id": "f" * 64}
    _write_jsonl(fixture.ledger, [*rows, extra])
    _reseal(fixture, "ledger")


def _mix_revision(fixture: CampaignFixture) -> None:
    rows = _load_jsonl(fixture.ledger)
    rows[0]["revision"] = "revision-other"
    _write_jsonl(fixture.ledger, rows)
    _reseal(fixture, "ledger")


def _mix_manifest(fixture: CampaignFixture) -> None:
    rows = _load_jsonl(fixture.ledger)
    rows[0]["manifest_sha256"] = "a" * 64
    _write_jsonl(fixture.ledger, rows)
    _reseal(fixture, "ledger")


def _substitute_dataset(fixture: CampaignFixture) -> None:
    rows = _load_jsonl(fixture.dataset)
    rows[0]["id"] = "substituted-case"
    _write_jsonl(fixture.dataset, rows)
    _reseal(fixture, "dataset")


def _substitute_manifest(fixture: CampaignFixture) -> None:
    manifest = _load_json(fixture.manifest)
    manifest["manifest_id"] = "campaign-substitute"
    fixture.manifest.write_bytes(_canonical(manifest))
    _reseal(fixture, "manifest")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (_alter_bytes, "ledger SHA-256 mismatch"),
        (_truncate, "ledger is truncated"),
        (_duplicate, "duplicate planned tuple"),
        (_extra, "extra or mixed-revision tuple"),
        (_mix_revision, "extra or mixed-revision tuple"),
        (_mix_manifest, "mixes a manifest identity"),
        (_substitute_dataset, "trusted approved SHA-256"),
        (_substitute_manifest, "trusted campaign SHA-256"),
    ],
    ids=[
        "alteration",
        "truncation",
        "duplicate",
        "extra",
        "mixed-revision",
        "mixed-manifest",
        "dataset-substitution",
        "manifest-substitution",
    ],
)
def test_audit_rejects_independent_artifact_corruption(
    tmp_path: Path,
    mutation: Callable[[CampaignFixture], None],
    match: str,
) -> None:
    """Each provenance or tuple-set mutation fails closed independently."""
    fixture = _campaign(tmp_path)
    mutation(fixture)

    with pytest.raises(ValueError, match=match):
        _audit(fixture)
