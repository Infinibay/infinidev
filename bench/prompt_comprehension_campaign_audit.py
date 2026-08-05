#!/usr/bin/env python3
"""Audit prompt-comprehension campaign artifacts from their persisted bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


TERMINAL_FAILURE_TYPES = {"provider_error", "parse_error", "preflight_blocked"}


@dataclass(frozen=True)
class CampaignAudit:
    """Verified identities and terminal tuple counts for one campaign."""

    closure_sha256: str
    dataset_sha256: str
    manifest_sha256: str
    ledger_sha256: str
    planned_tuples: int
    successes: int
    failures: int


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _object(data: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from err
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _jsonl(data: bytes, label: str) -> list[dict[str, object]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as err:
        raise ValueError(f"{label} is not valid UTF-8") from err
    if not text or not text.endswith("\n"):
        raise ValueError(f"{label} is empty or truncated")
    rows: list[dict[str, object]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ValueError(f"{label} contains a blank row at line {number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as err:
            raise ValueError(f"{label} has invalid JSON at line {number}") from err
        if not isinstance(row, dict):
            raise ValueError(f"{label} row {number} must be an object")
        rows.append(row)
    return rows


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be explicit non-empty text")
    return value


def _artifact_path(root: Path, value: object, label: str) -> Path:
    relative = Path(_text(value, f"{label}.path"))
    if relative.is_absolute():
        raise ValueError(f"{label}.path must be relative to the artifact root")
    root = root.resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label}.path escapes the artifact root")
    if not resolved.is_file():
        raise ValueError(f"{label} does not exist: {relative}")
    return resolved


def _linked_bytes(
    root: Path,
    descriptor: Mapping[str, object],
    label: str,
) -> tuple[bytes, str]:
    path = _artifact_path(root, descriptor.get("path"), label)
    data = path.read_bytes()
    actual = _sha256(data)
    declared = _text(descriptor.get("sha256"), f"{label}.sha256")
    if actual != declared:
        raise ValueError(f"{label} SHA-256 mismatch")
    return data, actual


def _model_key(model: Mapping[str, object]) -> tuple[str, str, str, str]:
    return (
        _text(model.get("provider"), "model.provider"),
        _text(model.get("model"), "model.model"),
        _text(model.get("revision"), "model.revision"),
        _text(model.get("model_identity"), "model.model_identity"),
    )


def tuple_id(
    case_id: str,
    model_key: tuple[str, str, str, str],
    manifest_sha256: str,
    dataset_sha256: str,
) -> str:
    """Return the stable identity of one planned raw campaign tuple."""
    provider, model, revision, model_identity = model_key
    coordinates = {
        "case_id": case_id,
        "condition": "raw",
        "dataset_sha256": dataset_sha256,
        "manifest_sha256": manifest_sha256,
        "model": model,
        "model_identity": model_identity,
        "provider": provider,
        "revision": revision,
    }
    encoded = json.dumps(coordinates, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256(encoded)


def audit_campaign(
    closure_path: Path,
    *,
    artifact_root: Path,
    expected_dataset_sha256: str,
    expected_manifest_sha256: str,
) -> CampaignAudit:
    """Fail closed unless bytes form one complete, immutable raw-only campaign."""
    closure_bytes = closure_path.read_bytes()
    closure = _object(closure_bytes, "closure")
    if closure.get("schema_version") != 1:
        raise ValueError("unsupported closure schema_version")
    if closure.get("condition") != "raw":
        raise ValueError("campaign condition must be exactly raw")

    dataset_descriptor = _mapping(closure.get("dataset"), "closure.dataset")
    manifest_descriptor = _mapping(closure.get("manifest"), "closure.manifest")
    ledger_descriptor = _mapping(closure.get("ledger"), "closure.ledger")
    dataset_bytes, dataset_sha = _linked_bytes(artifact_root, dataset_descriptor, "dataset")
    manifest_bytes, manifest_sha = _linked_bytes(artifact_root, manifest_descriptor, "manifest")
    ledger_bytes, ledger_sha = _linked_bytes(artifact_root, ledger_descriptor, "ledger")
    if dataset_sha != expected_dataset_sha256:
        raise ValueError("dataset does not match the trusted approved SHA-256")
    if manifest_sha != expected_manifest_sha256:
        raise ValueError("manifest does not match the trusted campaign SHA-256")
    if dataset_descriptor.get("review_status") != "approved":
        raise ValueError("closure dataset review_status must be approved")

    dataset_rows = _jsonl(dataset_bytes, "dataset")
    case_ids: list[str] = []
    for row in dataset_rows:
        case_ids.append(_text(row.get("id"), "dataset.id"))
        if row.get("review_status") != "approved":
            raise ValueError("dataset mixes non-approved review statuses")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("dataset contains duplicate case IDs")
    if int(dataset_descriptor.get("case_count", -1)) != len(case_ids):
        raise ValueError("dataset case_count does not match its bytes")

    manifest = _object(manifest_bytes, "manifest")
    _text(manifest.get("manifest_id"), "manifest.manifest_id")
    if manifest.get("condition") != "raw":
        raise ValueError("manifest condition must be exactly raw")
    if manifest.get("dataset_sha256") != dataset_sha:
        raise ValueError("manifest substitutes or mixes the dataset identity")
    raw_models = manifest.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("manifest.models must be a non-empty list")
    model_keys = [_model_key(_mapping(model, "manifest model")) for model in raw_models]
    if len(model_keys) != len(set(model_keys)):
        raise ValueError("manifest contains duplicate model revisions")

    expected = {
        (case_id, model_key): tuple_id(case_id, model_key, manifest_sha, dataset_sha)
        for case_id in case_ids
        for model_key in model_keys
    }
    ledger_rows = _jsonl(ledger_bytes, "ledger")
    observed: dict[tuple[str, tuple[str, str, str, str]], str] = {}
    successes = 0
    failures = 0
    for row in ledger_rows:
        if row.get("condition") != "raw":
            raise ValueError("ledger mixes a non-raw condition")
        if row.get("dataset_sha256") != dataset_sha:
            raise ValueError("ledger mixes a dataset identity")
        if row.get("manifest_sha256") != manifest_sha:
            raise ValueError("ledger mixes a manifest identity")
        if row.get("terminal") is not True:
            raise ValueError("ledger contains a non-terminal tuple")
        case_id = _text(row.get("case_id"), "ledger.case_id")
        model_key = _model_key(row)
        coordinate = (case_id, model_key)
        if coordinate in observed:
            raise ValueError("ledger contains a duplicate planned tuple")
        stored_tuple_id = _text(row.get("tuple_id"), "ledger.tuple_id")
        expected_tuple_id = expected.get(coordinate)
        if expected_tuple_id is None:
            raise ValueError("ledger contains an extra or mixed-revision tuple")
        if stored_tuple_id != expected_tuple_id:
            raise ValueError("ledger tuple_id does not match its byte-level provenance")
        observed[coordinate] = stored_tuple_id
        status = row.get("status")
        if status == "success":
            if row.get("failure") not in (None, {}):
                raise ValueError("successful tuple contains failure metadata")
            observation = _mapping(row.get("observation"), "ledger.observation")
            if (
                observation.get("case_id") != case_id
                or observation.get("condition") != "raw"
                or observation.get("model_identity") != model_key[3]
                or observation.get("dataset_sha256") != dataset_sha
                or not isinstance(observation.get("response_text"), str)
                or not isinstance(observation.get("parsed"), dict)
            ):
                raise ValueError("successful tuple observation does not match its coordinates")
            successes += 1
        elif status == "failure":
            failure = _mapping(row.get("failure"), "ledger.failure")
            if failure.get("type") not in TERMINAL_FAILURE_TYPES:
                raise ValueError("ledger failure is not terminally typed")
            if failure.get("type") != "preflight_blocked":
                observation = _mapping(row.get("observation"), "ledger.observation")
                if (
                    observation.get("case_id") != case_id
                    or observation.get("condition") != "raw"
                    or observation.get("model_identity") != model_key[3]
                    or observation.get("dataset_sha256") != dataset_sha
                ):
                    raise ValueError("failed tuple observation does not match its coordinates")
            failures += 1
        else:
            raise ValueError("ledger status must be success or failure")

    missing = set(expected) - set(observed)
    if missing:
        raise ValueError(f"ledger is truncated: {len(missing)} planned tuples are missing")
    if len(observed) != len(expected):
        raise ValueError("ledger tuple cardinality does not match the plan")

    counts = _mapping(closure.get("counts"), "closure.counts")
    required_counts = {
        "planned_tuples": len(expected),
        "terminal_tuples": len(observed),
        "successes": successes,
        "failures": failures,
        "pending": 0,
    }
    for name, expected_count in required_counts.items():
        if counts.get(name) != expected_count:
            raise ValueError(f"closure count mismatch: {name}")

    return CampaignAudit(
        closure_sha256=_sha256(closure_bytes),
        dataset_sha256=dataset_sha,
        manifest_sha256=manifest_sha,
        ledger_sha256=ledger_sha,
        planned_tuples=len(expected),
        successes=successes,
        failures=failures,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("closure", type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    result = audit_campaign(
        args.closure,
        artifact_root=args.artifact_root,
        expected_dataset_sha256=args.expected_dataset_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
    )
    print(json.dumps(result.__dict__, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
