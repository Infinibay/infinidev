"""Tests for manual-review family consistency auditing."""

from __future__ import annotations

from bench.external_review_family_audit import audit_reviewed_families


def _candidate(identifier: str, text: str) -> dict[str, object]:
    return {
        "candidate_id": identifier,
        "source": {"repo": identifier},
        "issue_text": text,
    }


def _review(identifier: str, policies: list[str], reason: str | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": identifier,
        "include": True,
        "policies": policies,
        "notes": "manually reviewed",
    }
    if reason:
        row["uncategorized_reason"] = reason
    return row


def test_audit_reports_conflicts_and_unreviewed_family_members() -> None:
    shared = " ".join(f"statement{index}" for index in range(70))
    candidates = [
        _candidate("a", f"Fix this parser {shared} alpha"),
        _candidate("b", f"Extend this parser {shared} beta"),
        _candidate("c", f"Review this parser {shared} gamma"),
        _candidate("other", "Explain transaction isolation levels"),
    ]
    decisions = {
        "a": _review("a", ["bugfix"]),
        "b": _review("b", ["feature"]),
    }

    report = audit_reviewed_families(candidates, decisions)

    assert report["reviewed_rows"] == 2
    assert report["reviewed_families"] == 1
    assert report["reviewed_family_rows"] == 3
    assert report["unreviewed_reviewed_family_rows"] == 1
    assert report["conflicting_reviewed_families"] == 1
    assert report["unreviewed"][0]["candidate_id"] == "c"
    assert {tuple(item["policies"]) for item in report["conflicts"][0]["decisions"]} == {
        ("bugfix",),
        ("feature",),
    }


def test_audit_accepts_consistent_zero_label_family() -> None:
    candidates = [
        _candidate("a", "write a script about alpha versus beta"),
        _candidate("b", "write a script about gamma versus delta"),
    ]
    decisions = {
        "a": _review("a", [], "ambiguous_method"),
        "b": _review("b", [], "ambiguous_method"),
    }

    report = audit_reviewed_families(candidates, decisions)

    assert report["conflicting_reviewed_families"] == 0
    assert report["unreviewed_reviewed_family_rows"] == 0
