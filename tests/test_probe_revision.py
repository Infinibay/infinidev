from __future__ import annotations

from bench.probe_revision import materialize_revision


def _records() -> list[dict[str, object]]:
    return [
        {
            "id": "p1",
            "group": "g1",
            "evaluation_mode": "preference",
            "review_status": "draft",
            "answer": None,
            "reviewer": "",
            "scenario": "old",
            "analysis": {"variant_axis": "old axis"},
        }
    ]


def test_materialize_revision_tracks_evidence_and_nested_updates() -> None:
    revised, lineage = materialize_revision(
        _records(),
        {
            "revision_id": "r1",
            "base_dataset_sha256": "base",
            "changes": [
                {
                    "probe_id": "p1",
                    "updates": {
                        "scenario": "new",
                        "analysis.variant_axis": "new axis",
                    },
                    "rationale": "Independent diagnostics agreed.",
                    "evidence": ["report.md#p1"],
                }
            ],
        },
        base_sha256="base",
    )
    assert revised[0]["scenario"] == "new"
    assert revised[0]["analysis"]["variant_axis"] == "new axis"
    assert lineage["changed_probe_count"] == 1
    assert lineage["changes"][0]["evidence"] == ["report.md#p1"]


def test_materialize_revision_rejects_hash_mismatch_and_protected_fields() -> None:
    spec = {
        "revision_id": "r1",
        "base_dataset_sha256": "base",
        "changes": [
            {
                "probe_id": "p1",
                "updates": {"review_status": "approved"},
                "rationale": "Bad update.",
                "evidence": ["none"],
            }
        ],
    }
    for base_hash in ("wrong", "base"):
        try:
            materialize_revision(_records(), spec, base_sha256=base_hash)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe revision was accepted")
