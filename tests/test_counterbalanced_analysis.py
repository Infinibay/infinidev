from __future__ import annotations

from bench.counterbalanced_analysis import classify_probe


def _record(key: str, *, stable: bool) -> dict[str, object]:
    return {"modal_keys": [key], "stable": stable}


def test_classification_distinguishes_stability_from_modal_agreement() -> None:
    assert classify_probe({"a": _record("A", stable=True), "b": _record("A", stable=True)}) == (
        "stable_shared"
    )
    assert classify_probe({"a": _record("A", stable=True), "b": _record("B", stable=True)}) == (
        "stable_divergence"
    )
    assert classify_probe({"a": _record("A", stable=False), "b": _record("A", stable=True)}) == (
        "shared_modal"
    )
    assert classify_probe({"a": _record("A", stable=False), "b": _record("B", stable=True)}) == (
        "divergent_modal"
    )


def test_classification_retains_modal_ties() -> None:
    assert classify_probe(
        {"a": {"modal_keys": ["A", "B"], "stable": False}, "b": _record("A", stable=True)}
    ) == "modal_tie"
