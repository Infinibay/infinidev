"""Tests for the sequential held-out context-delivery collector."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.context_delivery_eval import ContextTask
from bench.context_delivery_run import (
    RunConfig,
    _delivered_evidence,
    build_full_corpus,
    single_flight_lock,
)


def _write_config(path: Path, **overrides: object) -> None:
    value = {
        "provider": "openai_subscription",
        "model": "gpt-5.6-sol",
        "model_identity": "openai_subscription:gpt-5.6-sol:test-snapshot",
        **overrides,
    }
    path.write_text(json.dumps(value), encoding="utf-8")


def _task() -> ContextTask:
    return ContextTask.from_dict(
        {
            "id": "t1",
            "family": "f1",
            "split": "validation",
            "repository_fixture": "fixture",
            "request": "repair it",
            "verify_command": "pytest -q",
            "required_evidence": ["src/a.py:A.fix", "tests/test_a.py:test_fix"],
            "review_status": "approved",
        }
    )


def test_run_config_enforces_two_second_minimum(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write_config(path, min_task_interval_seconds=1.99)

    with pytest.raises(ValueError, match="at least 2.0"):
        RunConfig.from_path(path)


def test_full_corpus_has_stable_paths_and_ignores_runtime_state(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "b.py").write_text("B = 2\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("A = 1\n", encoding="utf-8")
    (tmp_path / ".infinidev").mkdir()
    (tmp_path / ".infinidev" / "state.db").write_bytes(b"not utf8: \xff")

    corpus, paths = build_full_corpus(tmp_path)

    assert paths == ("a.py", "src/b.py")
    assert corpus.index("FILE: a.py") < corpus.index("FILE: src/b.py")
    assert "state.db" not in corpus


def test_full_corpus_places_relevant_files_at_declared_position(tmp_path: Path) -> None:
    for name in ("a.py", "b.py", "c.py", "relevant.py"):
        (tmp_path / name).write_text(f"NAME = {name!r}\n", encoding="utf-8")

    _, front = build_full_corpus(
        tmp_path, relevant_paths=("relevant.py",), relevant_position="front"
    )
    _, middle = build_full_corpus(
        tmp_path, relevant_paths=("relevant.py",), relevant_position="middle"
    )
    _, end = build_full_corpus(
        tmp_path, relevant_paths=("relevant.py",), relevant_position="end"
    )

    assert front[0] == "relevant.py"
    assert middle == ("a.py", "relevant.py", "b.py", "c.py")
    assert end[-1] == "relevant.py"


def test_evidence_delivery_records_only_exact_supplied_paths() -> None:
    task = _task()
    ranked = SimpleNamespace(
        files=[SimpleNamespace(target="src/a.py")],
        symbols=[],
        findings=[],
    )
    engine = SimpleNamespace(_cr_cached_result=ranked)

    assert _delivered_evidence(task, "baseline", engine, ()) == ()
    assert _delivered_evidence(task, "ranked", engine, ()) == ("src/a.py:A.fix",)
    assert _delivered_evidence(
        task, "full", engine, ("src/a.py", "tests/test_a.py")
    ) == task.required_evidence


def test_ranked_evidence_uses_every_delivered_pivot_not_only_last_one() -> None:
    task = _task()
    engine = SimpleNamespace(
        _cr_delivered_targets={"/tmp/repo/src/a.py", "/tmp/repo/tests/test_a.py"},
        _cr_cached_result=SimpleNamespace(files=[], symbols=[], findings=[]),
    )

    assert _delivered_evidence(task, "ranked", engine, ()) == task.required_evidence


def test_single_flight_lock_rejects_concurrent_owner(tmp_path: Path) -> None:
    lock = tmp_path / "campaign.lock"

    with single_flight_lock(lock):
        with pytest.raises(RuntimeError, match="another subscription-backed evaluation"):
            with single_flight_lock(lock):
                pass
