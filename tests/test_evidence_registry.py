from __future__ import annotations

from bench.evidence_registry import REQUIRED_DECISIONS, audit_registry


def _registry() -> dict[str, object]:
    decisions = []
    for decision_id in sorted(REQUIRED_DECISIONS):
        decisions.append(
            {
                "id": decision_id,
                "status": "supported",
                "claim": "Bounded claim",
                "supported_scope": "Measured scope",
                "does_not_establish": "Universal validity",
                "sources": ["paper"],
                "controls": ["held-out check"],
            }
        )
    return {
        "schema_version": 1,
        "sources": [
            {
                "id": "paper",
                "kind": "paper",
                "title": "Paper",
                "url": "https://example.test/paper",
                "primary_finding_used": "A bounded result",
                "limits": "A different setting",
            }
        ],
        "decisions": decisions,
    }


def test_registry_audit_accepts_bounded_cited_claims(tmp_path) -> None:
    report = audit_registry(_registry(), root=tmp_path)
    assert report["passes"] is True
    assert report["decision_count"] == 11


def test_registry_audit_rejects_supported_claim_without_source(tmp_path) -> None:
    registry = _registry()
    registry["decisions"][0]["sources"] = []
    report = audit_registry(registry, root=tmp_path)
    assert report["passes"] is False
    assert any("without evidence" in error for error in report["errors"])


def test_registry_audit_requires_internal_artifact(tmp_path) -> None:
    registry = _registry()
    registry["sources"].append(
        {
            "id": "internal",
            "kind": "internal_experiment",
            "title": "Run",
            "path": "missing.json",
            "primary_finding_used": "One observation",
            "limits": "Small sample",
        }
    )
    report = audit_registry(registry, root=tmp_path)
    assert any("internal evidence is missing" in error for error in report["errors"])
