from __future__ import annotations

import json
from pathlib import Path

from bench.generate_prompt_comprehension_battery import materialize
from bench.prompt_comprehension_battery_audit import audit_battery


def test_generator_materializes_complete_balanced_battery() -> None:
    cases, registry = materialize()

    assert len(cases) == 672
    assert len(registry["families"]) == 224
    assert len({case["id"] for case in cases}) == 672
    assert len(registry["questions"]) == 98

    expected_cases = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in cases
    )
    expected_registry = json.dumps(registry, ensure_ascii=False, indent=2) + "\n"
    assert Path("bench/prompt_comprehension_battery.draft.jsonl").read_text() == expected_cases
    assert Path("bench/prompt_comprehension_family_registry.json").read_text() == expected_registry


def test_checked_in_battery_passes_fail_closed_audit() -> None:
    report = audit_battery(
        Path("bench/prompt_comprehension_battery.draft.jsonl"),
        Path("bench/prompt_comprehension_family_registry.json"),
    )

    assert report["all_passed"] is True
    assert report["counts"]["cases"] == 672
    assert report["counts"]["families"] == 224
