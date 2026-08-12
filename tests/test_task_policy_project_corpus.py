"""Diversity and leakage gates for the project-inspired draft corpus."""

from __future__ import annotations

from bench.task_policy_project_corpus import (
    PROJECTS,
    audit_project_corpus,
    build_project_corpus,
)


def test_project_corpus_has_broad_language_and_domain_coverage() -> None:
    examples = build_project_corpus()
    report = audit_project_corpus(examples)

    assert report["examples"] >= 650
    assert report["projects"] >= 12
    assert len(report["programming_languages"]) >= 12
    assert set(report["natural_languages"]) == {"en", "es", "fr", "pt"}
    assert len(report["operations"]) >= 12
    assert report["uncategorized"] >= 90
    assert all(project.inspiration_url.startswith("https://github.com/") for project in PROJECTS)


def test_project_corpus_is_split_by_project_and_explicit_about_negatives() -> None:
    report = audit_project_corpus(build_project_corpus())

    assert report["duplicate_ids"] == 0
    assert report["duplicate_texts"] == 0
    assert report["family_split_leakage"] == []
    assert report["phrase_split_leakage"]
    assert report["malformed_uncategorized"] == []
    assert report["all_rows_are_draft"] is True
    assert report["release_ready"] is False
    assert {"conversation", "out_of_domain", "quoted_action"} <= set(
        report["uncategorized_reasons"]
    )
