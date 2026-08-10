from __future__ import annotations

from bench.build_multilingual_embedding_mixture import select_balanced


def _row(identity: str, repository: str, **overrides):
    row = {
        "id": identity,
        "source": "commitpackft_python",
        "source_dataset": "commitpack",
        "repository": repository,
        "kind": "instruction_to_code_change",
        "language": "en",
        "programming_language": "python",
        "text": f"Fix behavior {identity}",
        "parallel_text": f"@@ change {identity}",
        "split": "train",
    }
    row.update(overrides)
    return row


def test_balances_groups_and_is_order_independent():
    rows = [_row(f"py-{index}", f"repo-{index}") for index in range(10)]
    rows += [
        _row(f"java-{index}", f"java-repo-{index}", programming_language="java")
        for index in range(10)
    ]

    first, _ = select_balanced(
        rows, limits={"instruction_to_code_change": 3}, default_limit=1, seed=4
    )
    second, _ = select_balanced(
        reversed(rows),
        limits={"instruction_to_code_change": 3},
        default_limit=1,
        seed=4,
    )

    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert sum(row["programming_language"] == "python" for row in first) == 3
    assert sum(row["programming_language"] == "java" for row in first) == 3


def test_excludes_upstream_holdouts_and_duplicate_pairs():
    duplicate = _row("duplicate", "other", text="Fix behavior one", parallel_text="@@ change one")
    rows = [_row("one", "repo"), duplicate, _row("heldout", "heldout", split="test")]

    selected, rejected = select_balanced(
        rows, limits={"instruction_to_code_change": 10}, default_limit=1, seed=4
    )

    assert [row["id"] for row in selected] == ["one"]
    assert rejected["duplicate_pair"] == 1
    assert rejected["ineligible_or_upstream_holdout"] == 1


def test_excludes_deprioritized_programming_languages():
    selected, rejected = select_balanced(
        [_row("zig", "repo", programming_language="zig")],
        limits={"instruction_to_code_change": 10},
        default_limit=1,
        seed=4,
    )

    assert selected == []
    assert rejected["ineligible_or_upstream_holdout"] == 1


def test_repository_family_never_crosses_internal_splits():
    rows = [_row(f"same-{index}", "same/repository") for index in range(5)]
    selected, _ = select_balanced(
        rows, limits={"instruction_to_code_change": 10}, default_limit=1, seed=17
    )

    assert len({row["split"] for row in selected}) == 1
    assert len({row["split_family"] for row in selected}) == 1
