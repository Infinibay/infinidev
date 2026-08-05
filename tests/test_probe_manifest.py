from __future__ import annotations

import pytest

from bench.model_behavior import Probe
from bench.probe_manifest import build_explicit_manifest, build_manifest, manifest_probe_ids


def _probe(
    probe_id: str,
    category: str,
    mode: str = "preference",
    group: str | None = None,
) -> Probe:
    return Probe(
        probe_id,
        category,
        "Choose",
        {"A": "fast", "B": "careful"},
        None if mode == "preference" else "B",
        group=group,
        evaluation_mode=mode,
        choice_effects=(
            {"A": {"speed": 1.0}, "B": {"quality": 1.0}}
            if mode == "preference"
            else {}
        ),
    )


def test_manifest_is_stratified_deterministic_and_excludes_prior_coverage() -> None:
    probes = [
        _probe("a1", "alpha"),
        _probe("a2", "alpha"),
        _probe("b1", "beta"),
        _probe("b2", "beta"),
    ]
    first = build_manifest(
        probes,
        dataset_sha256="dataset",
        seed=7,
        per_category=1,
        evaluation_mode="preference",
        excluded_probe_ids=["a1"],
    )
    second = build_manifest(
        reversed(probes),
        dataset_sha256="dataset",
        seed=7,
        per_category=1,
        evaluation_mode="preference",
        excluded_probe_ids=["a1"],
    )
    assert first == second
    assert first["probe_count"] == 2
    assert {row["category"] for row in first["probes"]} == {"alpha", "beta"}
    assert "a1" not in {row["probe_id"] for row in first["probes"]}


def test_manifest_validation_binds_dataset_and_metadata() -> None:
    probe = _probe("p1", "tools")
    manifest = build_manifest(
        [probe],
        dataset_sha256="dataset",
        seed=0,
        per_category=1,
        evaluation_mode="preference",
    )
    assert manifest_probe_ids(manifest, {probe.id: probe}, dataset_sha256="dataset") == ["p1"]
    with pytest.raises(ValueError, match="dataset_sha256"):
        manifest_probe_ids(manifest, {probe.id: probe}, dataset_sha256="changed")


def test_manifest_can_exclude_whole_observed_families() -> None:
    probes = [
        _probe("a1", "alpha", group="seen"),
        _probe("a2", "alpha", group="seen"),
        _probe("a3", "alpha", group="unseen"),
        _probe("b1", "beta", group="fresh"),
    ]
    manifest = build_manifest(
        probes,
        dataset_sha256="dataset",
        seed=0,
        per_category=1,
        evaluation_mode="preference",
        excluded_probe_ids=["a1"],
        excluded_families=["seen"],
    )
    selected = {row["probe_id"] for row in manifest["probes"]}
    assert selected == {"a3", "b1"}
    assert manifest["selection"]["excluded_family_count"] == 1


def test_manifest_records_explicit_allowed_category_shortfalls() -> None:
    probes = [
        _probe("a1", "alpha", group="seen"),
        _probe("b1", "beta", group="fresh"),
    ]
    manifest = build_manifest(
        probes,
        dataset_sha256="dataset",
        seed=0,
        per_category=1,
        evaluation_mode="preference",
        excluded_families=["seen"],
        allow_category_shortfalls=True,
    )
    assert [row["probe_id"] for row in manifest["probes"]] == ["b1"]
    assert manifest["selection"]["category_shortfalls"] == {"alpha": 1}


def test_explicit_manifest_preserves_predeclared_order_and_purpose() -> None:
    first = _probe("first", "alpha")
    second = _probe("second", "beta")
    manifest = build_explicit_manifest(
        {first.id: first, second.id: second},
        dataset_sha256="dataset",
        probe_ids=["second", "first"],
        purpose="replicate divergent families under user profiles",
    )
    assert [row["probe_id"] for row in manifest["probes"]] == ["second", "first"]
    assert manifest["selection"] == {
        "method": "explicit",
        "purpose": "replicate divergent families under user profiles",
    }
