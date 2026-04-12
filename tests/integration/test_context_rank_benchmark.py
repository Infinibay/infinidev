"""End-to-end benchmark for ContextRank v3 on a real-world fixture.

Uses a trimmed snapshot of an infinidev-indexed TypeScript project
(backend-refactor, ~3500 symbols across 110 files) to exercise the
ranker's fuzzy symbol search channel against queries that mirror what
a developer would actually ask a coding assistant.

**Why this exists.**  Unit tests in ``test_context_rank.py`` exercise
each ranker component in isolation with small synthetic fixtures
(2-3 symbols, controlled embeddings).  That catches regressions in
logic but not in *calibration* — the thresholds, scale factors, and
guards in ``ranker.py`` were tuned against this specific corpus, and
a refactor that passes all unit tests can still break the live hit
rate.  This benchmark is the integration-level regression guard.

**Test isolation.**  The fixture DB at
``tests/fixtures/cr_bench_corpus.db`` is **never written to** by the
tests.  The ``cr_benchmark_db_path`` fixture copies it to a pytest
tempdir at session scope, runs ``init_db()`` on the copy (so the v3
schema migrations apply even if the committed fixture is older than
the current code), and points ``INFINIDEV_DB_PATH`` at the copy.
The committed fixture byte stream is immutable from the tests'
perspective — git status stays clean after ``pytest``.

**Why session scope.**  Copying 8 MB and initializing the schema takes
~200 ms; running ``init_db()`` on a fresh copy per test would be
wasteful.  Session scope means all benchmark tests share one
tempdir copy.  Since no test writes to the DB by design, this is
safe — but if future tests *do* need isolated writes, switch the
fixture to function scope.

**Paths.**  The fixture was built with all absolute workspace
prefixes stripped (``/home/andres/swe/runs/backend-refactor/`` →
``""``), so ``file_path`` values are all relative.  The ranker's
path normalization via ``_normalize_path`` is a no-op on already-
relative paths, so behavior is identical to a live run.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest


FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "cr_bench_corpus.db"


@pytest.fixture(scope="session")
def cr_benchmark_db_path(tmp_path_factory):
    """Copy the committed fixture to a tempdir and point env vars at it.

    Returns the path to the tempdir copy.  The original fixture file
    is guaranteed to remain byte-identical after the test session
    because no code in the test ever touches it — all writes (if
    any) go to the tempdir copy which pytest cleans up automatically.
    """
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"benchmark fixture missing: {FIXTURE_PATH}. "
            "Run the backfill script to regenerate it."
        )

    tmp_dir = tmp_path_factory.mktemp("cr_bench")
    db_copy = tmp_dir / "cr_bench_corpus.db"
    shutil.copy(FIXTURE_PATH, db_copy)

    # Point the settings layer at the copy.  INFINIDEV_DB_PATH is
    # read via pydantic-settings at import time, but settings.DB_PATH
    # can be overridden in-process by writing to the module attribute
    # before any DB call.  We do both: set the env var (for any
    # sub-process spawn) and patch the live settings object.
    previous_env = os.environ.get("INFINIDEV_DB_PATH")
    os.environ["INFINIDEV_DB_PATH"] = str(db_copy)

    from infinidev.config.settings import settings as _settings
    previous_db_path = _settings.DB_PATH
    _settings.DB_PATH = str(db_copy)

    # Apply schema migrations to the copy so the fixture is usable
    # even if the committed file was generated before later v3
    # columns were added.  init_db() is idempotent — only adds the
    # missing columns and indexes.
    from infinidev.db.service import init_db
    init_db()

    yield db_copy

    # Restore env + settings so downstream tests aren't polluted
    if previous_env is None:
        os.environ.pop("INFINIDEV_DB_PATH", None)
    else:
        os.environ["INFINIDEV_DB_PATH"] = previous_env
    _settings.DB_PATH = previous_db_path


def _fmt(result):
    """Format a rank result as a compact string list for failure messages."""
    out = []
    for section_name, items in (
        ("files", result.files),
        ("symbols", result.symbols),
        ("findings", result.findings),
    ):
        for item in items:
            out.append(f"{section_name}:{item.target}@{item.score:.2f}")
    return out or ["(empty)"]


def _rank_with_embedding(query: str, *, project_id: int = 1):
    """Run ``rank()`` with a pre-computed query embedding.

    Mirrors how the loop engine invokes it in production: the task
    embedding is computed once in a background thread and passed to
    every pivot's rank call via ``cached_embedding``.
    """
    from infinidev.engine.context_rank.ranker import rank
    from infinidev.tools.base.embeddings import compute_embedding

    emb = compute_embedding(query)
    return rank(
        query,
        session_id="bench-test",
        task_id="bench-test",
        iteration=0,
        cached_embedding=emb,
        project_id=project_id,
    )


def _contains_keyword(result, keyword: str) -> bool:
    """True if any ranked item's target or reason contains the keyword."""
    kl = keyword.lower()
    for section in (result.files, result.symbols, result.findings):
        for item in section:
            if kl in item.target.lower() or kl in item.reason.lower():
                return True
    return False


# ── Per-query benchmark tests ───────────────────────────────────────────
#
# Each test runs one representative query and asserts at least one
# expected keyword appears in the top-K.  The tests are separate so
# pytest output makes it obvious which query regressed if something
# breaks (vs one monolithic test reporting "4/6 passed").


class TestContextRankBenchmark:
    """Integration benchmark against the backend-refactor fixture."""

    def test_q1_exact_method_mention(self, cr_benchmark_db_path):
        """Exact method name in the query: should dominate via high cosine."""
        result = _rank_with_embedding(
            "Fix the bug in handleCriticalError method"
        )
        assert _contains_keyword(result, "handleCriticalError"), (
            f"expected handleCriticalError in top-K; got {_fmt(result)}"
        )

    def test_q2_typo_tolerance_on_method_name(self, cr_benchmark_db_path):
        """Typo in the method name — should still recover via fuzzy embedding."""
        result = _rank_with_embedding(
            "Show me the handleCrticalError method"  # typo: missing 'i'
        )
        assert not result.empty, f"expected at least one match; got {_fmt(result)}"
        # The target may not be match #1 (a closer spelling like
        # ``handleError`` can outrank it), but it must appear in the top-K.
        assert _contains_keyword(result, "handleCriticalError"), (
            f"expected handleCriticalError in top-K; got {_fmt(result)}"
        )

    def test_q3_semantic_description_firewall_cleanup(self, cr_benchmark_db_path):
        """Descriptive natural-language query with two domain keywords."""
        result = _rank_with_embedding("How does the firewall cleanup work?")
        # Any match on 'firewall' or 'cleanup' is acceptable — both are
        # common enough in this corpus that multiple symbols will rank.
        assert _contains_keyword(result, "firewall") or _contains_keyword(
            result, "cleanup"
        ), f"expected firewall/cleanup match; got {_fmt(result)}"

    def test_q4_concept_no_exact_name(self, cr_benchmark_db_path):
        """Concept-only query with multiple content words."""
        result = _rank_with_embedding(
            "Where do we validate ISO files before upload?"
        )
        # At least one of the validateISO / ISOService targets must surface
        assert any(
            _contains_keyword(result, k) for k in ("validateISO", "ISOService", "isoUpload")
        ), f"expected ISO validation match; got {_fmt(result)}"

    def test_q5_multi_hop_semantic(self, cr_benchmark_db_path):
        """Query that lost by 5 thousandths against the old 0.45 threshold.

        After the Zipf-based query simplification, the embedding of
        'manage virtual machine templates' (stop words dropped) lifts
        the cosines just enough for createMockMachineTemplate and
        machineTemplates to clear the threshold.  This is the
        regression-anchor test for that fix.
        """
        result = _rank_with_embedding(
            "How do we manage virtual machine templates?"
        )
        assert not result.empty, f"expected at least one match; got {_fmt(result)}"
        assert any(
            _contains_keyword(result, k)
            for k in ("machineTemplate", "machine_template", "createMockMachine")
        ), f"expected machine-template match; got {_fmt(result)}"

    def test_q6_negative_conversational_noise(self, cr_benchmark_db_path):
        """Unrelated conversational query must not produce a match.

        This is the regression guard against over-simplification:
        'what's the weather today' used to match a symbol literally
        named ``today`` in the codebase when the threshold was too
        low.  After the Zipf guard (< 2 surviving tokens falls back
        to raw), the raw query embedding doesn't match anything
        above the confidence gate.
        """
        result = _rank_with_embedding("what's the weather today")
        assert result.empty, (
            f"expected empty result for unrelated query; got {_fmt(result)}"
        )


# ── Corpus-level sanity test ────────────────────────────────────────────


class TestBenchmarkCorpusSanity:
    """Guards against regressions in the fixture itself."""

    def test_fixture_is_loaded(self, cr_benchmark_db_path):
        """The fixture must contain a non-trivial corpus."""
        import sqlite3
        conn = sqlite3.connect(str(cr_benchmark_db_path))
        n_symbols = conn.execute("SELECT COUNT(*) FROM ci_symbols").fetchone()[0]
        n_files = conn.execute("SELECT COUNT(*) FROM ci_files").fetchone()[0]
        n_embeddings = conn.execute(
            "SELECT COUNT(*) FROM ci_symbols WHERE embedding IS NOT NULL"
        ).fetchone()[0]
        assert n_symbols >= 500, f"fixture too small: {n_symbols} symbols"
        assert n_files >= 50, f"fixture too small: {n_files} files"
        # Every symbol in the fixture must have an embedding (we stripped
        # symbols with NULL embeddings when building it).
        assert n_embeddings == n_symbols, (
            f"{n_symbols - n_embeddings} symbols missing embeddings"
        )

    def test_fixture_paths_are_relative(self, cr_benchmark_db_path):
        """Paths must not contain any absolute leaks from the build machine."""
        import sqlite3
        conn = sqlite3.connect(str(cr_benchmark_db_path))
        leaks = conn.execute(
            "SELECT COUNT(*) FROM ci_symbols WHERE file_path LIKE '/%'"
        ).fetchone()[0]
        assert leaks == 0, f"{leaks} symbols still have absolute paths"

    def test_error_handler_classes_indexed(self, cr_benchmark_db_path):
        """Regression guard for the parser bug we diagnosed.

        The original fixture had the 3 classes in ErrorHandler.ts
        indexed with empty names (stale data from a pre-fix parser).
        We force-reindexed them when building the fixture; this test
        verifies the committed file still has them.
        """
        import sqlite3
        conn = sqlite3.connect(str(cr_benchmark_db_path))
        rows = conn.execute(
            "SELECT name FROM ci_symbols "
            "WHERE file_path LIKE '%ErrorHandler.ts' AND kind = 'class'"
        ).fetchall()
        names = {r[0] for r in rows}
        assert "ErrorHandler" in names, f"ErrorHandler missing; got {names}"
        assert "AppError" in names, f"AppError missing; got {names}"
        assert "ErrorLogger" in names, f"ErrorLogger missing; got {names}"
