"""Tests for the post-development code review engine."""

import json
import pytest
from unittest.mock import patch, MagicMock

from infinidev.engine.analysis.review_engine import ReviewEngine, ReviewResult


class TestReviewResult:
    """Test ReviewResult data class."""

    def test_approved_properties(self):
        result = ReviewResult(verdict="APPROVED", summary="All good")
        assert result.is_approved is True
        assert result.is_rejected is False

    def test_rejected_properties(self):
        result = ReviewResult(verdict="REJECTED", summary="Has issues")
        assert result.is_approved is False
        assert result.is_rejected is True

    def test_skipped_properties(self):
        result = ReviewResult(verdict="SKIPPED", summary="No changes")
        assert result.is_approved is False
        assert result.is_rejected is False

    def test_format_feedback_for_developer_approved(self):
        result = ReviewResult(verdict="APPROVED")
        assert result.format_feedback_for_developer() == ""

    def test_format_feedback_for_developer_rejected(self):
        result = ReviewResult(
            verdict="REJECTED",
            summary="Security issues found",
            issues=[
                {
                    "severity": "blocking",
                    "file": "src/auth.py",
                    "description": "SQL injection in login query",
                    "why": "Allows arbitrary SQL execution",
                    "fix": "Use parameterized queries",
                },
                {
                    "severity": "important",
                    "file": "src/utils.py",
                    "description": "Missing error handling",
                },
            ],
            notes=["Consider adding input validation"],
        )
        feedback = result.format_feedback_for_developer()
        assert "## Code Review Feedback — REJECTED" in feedback
        assert "Security issues found" in feedback
        assert "SQL injection" in feedback
        assert "src/auth.py" in feedback
        assert "parameterized queries" in feedback
        assert "Missing error handling" in feedback
        assert "input validation" in feedback

    def test_format_feedback_blocking_only(self):
        result = ReviewResult(
            verdict="REJECTED",
            summary="Bug found",
            issues=[
                {
                    "severity": "blocking",
                    "description": "Off-by-one error",
                    "fix": "Use < instead of <=",
                },
            ],
        )
        feedback = result.format_feedback_for_developer()
        assert "Off-by-one error" in feedback
        assert "< instead of <=" in feedback

    def test_format_for_user_approved(self):
        result = ReviewResult(
            verdict="APPROVED",
            summary="Code looks good",
            notes=["Minor style suggestion"],
        )
        text = result.format_for_user()
        assert "APPROVED" in text
        assert "Code looks good" in text
        assert "Minor style suggestion" in text

    def test_format_for_user_rejected(self):
        result = ReviewResult(
            verdict="REJECTED",
            summary="3 blocking issues",
            issues=[
                {"severity": "blocking", "description": "issue 1"},
                {"severity": "blocking", "description": "issue 2"},
                {"severity": "blocking", "description": "issue 3"},
            ],
        )
        text = result.format_for_user()
        assert "REJECTED" in text
        assert "3 blocking issue(s)" in text

    def test_format_for_user_skipped(self):
        result = ReviewResult(verdict="SKIPPED")
        assert result.format_for_user() == ""


class TestReviewEngine:
    """Test ReviewEngine."""

    def test_reset(self):
        engine = ReviewEngine()
        engine._review_count = 3
        engine.reset()
        assert engine._review_count == 0

    def test_can_review_again(self):
        engine = ReviewEngine()
        assert engine.can_review_again is True
        engine._review_count = 3
        assert engine.can_review_again is False

    def test_review_no_file_changes(self):
        engine = ReviewEngine()
        result = engine.review(
            task_description="Fix the bug",
            developer_result="Fixed it",
            file_changes_summary="",
        )
        assert result.verdict == "SKIPPED"

    def test_review_whitespace_only_changes(self):
        engine = ReviewEngine()
        result = engine.review(
            task_description="Fix the bug",
            developer_result="Fixed it",
            file_changes_summary="   \n  \n  ",
        )
        assert result.verdict == "SKIPPED"

    @patch("infinidev.config.llm.get_litellm_params")
    def test_review_no_llm(self, mock_params):
        mock_params.return_value = None
        engine = ReviewEngine()
        result = engine.review(
            task_description="Fix the bug",
            developer_result="Fixed it",
            file_changes_summary="--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new",
        )
        assert result.verdict == "SKIPPED"

    @patch("litellm.completion")
    @patch("infinidev.config.llm.get_litellm_params")
    def test_review_approved(self, mock_params, mock_completion):
        mock_params.return_value = {"model": "test"}
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content=json.dumps({
                    "verdict": "APPROVED",
                    "summary": "Code is clean and correct",
                    "notes": ["Good test coverage"],
                })
            ))]
        )
        engine = ReviewEngine()
        result = engine.review(
            task_description="Add login",
            developer_result="Added login endpoint",
            file_changes_summary="--- a/auth.py\n+++ b/auth.py\n+def login():\n+    pass",
        )
        assert result.is_approved
        assert "clean and correct" in result.summary

    @patch("litellm.completion")
    @patch("infinidev.config.llm.get_litellm_params")
    def test_review_rejected(self, mock_params, mock_completion):
        mock_params.return_value = {"model": "test"}
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content=json.dumps({
                    "verdict": "REJECTED",
                    "summary": "Missing input validation",
                    "issues": [
                        {
                            "severity": "blocking",
                            "file": "auth.py",
                            "description": "No password validation",
                            "why": "Allows empty passwords",
                            "fix": "Add length check",
                        }
                    ],
                })
            ))]
        )
        engine = ReviewEngine()
        result = engine.review(
            task_description="Add login",
            developer_result="Added login",
            file_changes_summary="--- a/auth.py\n+++ b/auth.py\n+def login(pw):\n+    pass",
        )
        assert result.is_rejected
        assert len(result.issues) == 1
        assert result.issues[0]["file"] == "auth.py"

    @patch("litellm.completion")
    @patch("infinidev.config.llm.get_litellm_params")
    def test_review_json_in_markdown(self, mock_params, mock_completion):
        mock_params.return_value = {"model": "test"}
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='```json\n{"verdict": "APPROVED", "summary": "ok"}\n```'
            ))]
        )
        engine = ReviewEngine()
        result = engine.review(
            task_description="x",
            developer_result="y",
            file_changes_summary="diff here",
        )
        assert result.is_approved

    @patch("litellm.completion")
    @patch("infinidev.config.llm.get_litellm_params")
    def test_review_llm_error_skips(self, mock_params, mock_completion):
        mock_params.return_value = {"model": "test"}
        mock_completion.side_effect = Exception("LLM down")
        engine = ReviewEngine()
        result = engine.review(
            task_description="x",
            developer_result="y",
            file_changes_summary="diff here",
        )
        assert result.verdict == "SKIPPED"  # Don't auto-approve on errors

    @patch("litellm.completion")
    @patch("infinidev.config.llm.get_litellm_params")
    def test_review_invalid_json_skips(self, mock_params, mock_completion):
        mock_params.return_value = {"model": "test"}
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="I think the code looks fine overall."
            ))]
        )
        engine = ReviewEngine()
        result = engine.review(
            task_description="x",
            developer_result="y",
            file_changes_summary="diff here",
        )
        assert result.verdict == "SKIPPED"  # Don't auto-approve unparseable responses

    @patch("litellm.completion")
    @patch("infinidev.config.llm.get_litellm_params")
    def test_review_with_previous_feedback(self, mock_params, mock_completion):
        mock_params.return_value = {"model": "test"}
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='{"verdict": "APPROVED", "summary": "Issues fixed"}'
            ))]
        )
        engine = ReviewEngine()
        engine._review_count = 1  # Simulate second round
        result = engine.review(
            task_description="Add login",
            developer_result="Fixed the issues",
            file_changes_summary="diff here",
            previous_feedback="## Code Review Feedback\nFix the validation",
        )
        assert result.is_approved
        # Verify the prompt included previous feedback
        call_args = mock_completion.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        user_msg = messages[1]["content"]
        assert "Previous Review Feedback" in user_msg
        assert "Fix the validation" in user_msg

    @patch("litellm.completion")
    @patch("infinidev.config.llm.get_litellm_params")
    def test_events_emitted_to_bus(self, mock_params, mock_completion):
        mock_params.return_value = {"model": "test"}
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='{"verdict": "APPROVED", "summary": "ok"}'
            ))]
        )
        events = []
        def callback(event_type, *args):
            events.append(event_type)

        from infinidev.flows.event_listeners import event_bus
        event_bus.subscribe(callback)
        try:
            engine = ReviewEngine()
            engine.review(
                task_description="x",
                developer_result="y",
                file_changes_summary="diff here",
            )
            assert "review_start" in events
            assert "review_complete" in events
        finally:
            event_bus.unsubscribe(callback)

    def test_build_review_prompt_structure(self):
        engine = ReviewEngine()
        prompt = engine._build_review_prompt(
            task_description="Add auth",
            developer_result="Added JWT auth",
            file_changes_summary="--- a/auth.py\n+++ b/auth.py",
            previous_feedback="",
        )
        assert "## Original Task" in prompt
        assert "Add auth" in prompt
        assert "## Developer's Report" in prompt
        assert "Added JWT auth" in prompt
        assert "## Diffs" in prompt

    def test_build_review_prompt_with_feedback(self):
        engine = ReviewEngine()
        engine._review_count = 2
        prompt = engine._build_review_prompt(
            task_description="Add auth",
            developer_result="Fixed issues",
            file_changes_summary="diff",
            previous_feedback="Fix the SQL injection",
        )
        assert "## Previous Review Feedback" in prompt
        assert "Fix the SQL injection" in prompt
        assert "Round 2" in prompt

    def test_build_review_prompt_with_conversation_context(self):
        engine = ReviewEngine()
        prompt = engine._build_review_prompt(
            task_description="Add auth",
            developer_result="Done",
            file_changes_summary="diff",
            previous_feedback="",
            recent_messages=[
                "[user] Set up the project structure",
                "[assistant] Created base files",
                "[user] Now add JWT authentication",
            ],
        )
        assert "## Conversation Context" in prompt
        assert "Set up the project structure" in prompt
        assert ">>> CURRENT REQUEST <<<" in prompt
        assert "Now add JWT authentication" in prompt

    def test_build_review_prompt_with_file_reasons_and_contents(self):
        engine = ReviewEngine()
        prompt = engine._build_review_prompt(
            task_description="Add auth",
            developer_result="Done",
            file_changes_summary="diff",
            previous_feedback="",
            file_reasons={
                "/home/user/project/auth.py": ["Added JWT token generation"],
                "/home/user/project/config.py": ["Added JWT secret config"],
            },
            file_contents={
                "/home/user/project/auth.py": "import jwt\n\ndef generate_token():\n    pass\n",
                "/home/user/project/config.py": "JWT_SECRET = 'changeme'\n",
            },
        )
        assert "## Files Changed" in prompt
        assert "auth.py" in prompt
        assert "config.py" in prompt
        assert "Added JWT token generation" in prompt
        assert "Added JWT secret config" in prompt
        assert "import jwt" in prompt
        assert "JWT_SECRET" in prompt

    @patch("litellm.completion")
    @patch("infinidev.config.llm.get_litellm_params")
    def test_review_passes_enriched_context(self, mock_params, mock_completion):
        """Verify that file_reasons, file_contents, and recent_messages reach the prompt."""
        mock_params.return_value = {"model": "test"}
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='{"verdict": "APPROVED", "summary": "ok"}'
            ))]
        )
        engine = ReviewEngine()
        engine.review(
            task_description="Add auth",
            developer_result="Done",
            file_changes_summary="diff here",
            file_reasons={"/tmp/auth.py": ["JWT implementation"]},
            file_contents={"/tmp/auth.py": "import jwt\n"},
            recent_messages=["[user] Add auth"],
        )
        call_args = mock_completion.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        user_msg = messages[1]["content"]
        assert "JWT implementation" in user_msg
        assert "import jwt" in user_msg
        assert "Conversation Context" in user_msg
        assert "[user] Add auth" in user_msg

    @patch("litellm.completion")
    @patch("infinidev.config.llm.get_litellm_params")
    def test_review_no_diffs_but_file_contents_proceeds(self, mock_params, mock_completion):
        """Review should proceed when file_contents is provided even without diffs."""
        mock_params.return_value = {"model": "test"}
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='{"verdict": "REJECTED", "summary": "Found issues", "issues": [{"severity": "blocking", "file": "app.py", "description": "SQL injection", "why": "unsafe", "fix": "use params"}]}'
            ))]
        )
        engine = ReviewEngine()
        result = engine.review(
            task_description="Review the app",
            developer_result="Reviewing",
            file_changes_summary="",
            file_contents={"/tmp/app.py": "import sqlite3\nconn.execute(f'SELECT * FROM users WHERE id={user_id}')"},
        )
        assert result.verdict != "SKIPPED"
        assert result.is_rejected

    def test_review_no_diffs_no_contents_skips(self):
        """Review should skip when neither diffs nor file_contents are provided."""
        engine = ReviewEngine()
        result = engine.review(
            task_description="Review the app",
            developer_result="Reviewing",
            file_changes_summary="",
        )
        assert result.verdict == "SKIPPED"


class TestMultiPassReview:
    """Test multi-pass review: extraction + judgment."""

    def _set_mode(self, monkeypatch, mode: str, threshold: int = 400):
        from infinidev.config.settings import settings as s
        monkeypatch.setattr(s, "REVIEW_MULTI_PASS_MODE", mode)
        monkeypatch.setattr(s, "REVIEW_MULTI_PASS_COMPLEXITY_THRESHOLD", threshold)

    def _extraction_response(self, extraction: dict) -> MagicMock:
        return MagicMock(choices=[MagicMock(message=MagicMock(
            content=json.dumps(extraction)
        ))])

    def _verdict_response(self, verdict: dict) -> MagicMock:
        return MagicMock(choices=[MagicMock(message=MagicMock(
            content=json.dumps(verdict)
        ))])

    @patch("litellm.completion")
    @patch("infinidev.config.llm.get_litellm_params_for_review_extractor")
    @patch("infinidev.config.llm.get_litellm_params")
    def test_multi_pass_happy_path(
        self, mock_params, mock_ext_params, mock_completion, monkeypatch,
    ):
        self._set_mode(monkeypatch, "always")
        mock_params.return_value = {"model": "judge"}
        mock_ext_params.return_value = {"model": "extractor"}

        mock_completion.side_effect = [
            self._extraction_response({
                "changes": [{"file": "a.py", "kind": "modified",
                             "symbols_added": ["f"], "symbols_removed": [],
                             "line_range": "1-5", "summary": "add f", "notable_lines": []}],
                "plan_coverage": [],
                "public_api_impact": {"new_exports": ["f"], "removed_exports": [], "signature_changes": []},
                "report_discrepancies": [],
            }),
            self._verdict_response({"verdict": "APPROVED", "summary": "ok"}),
        ]

        engine = ReviewEngine()
        result = engine.review(
            task_description="Add f",
            developer_result="Added f",
            file_changes_summary="--- a/a.py\n+++ b/a.py\n+def f(): pass",
            file_contents={"a.py": "def f(): pass"},
        )
        assert result.is_approved
        assert mock_completion.call_count == 2

        # The second call is the judge — its user message must contain the
        # Extraction block and must NOT contain a "## Diffs" section.
        second_call = mock_completion.call_args_list[1]
        user_msg = next(
            m["content"] for m in second_call.kwargs["messages"]
            if m["role"] == "user"
        )
        assert "## Extraction" in user_msg
        assert "## Diffs" not in user_msg

    @patch("litellm.completion")
    @patch("infinidev.config.llm.get_litellm_params_for_review_extractor")
    @patch("infinidev.config.llm.get_litellm_params")
    def test_multi_pass_extraction_malformed_falls_back(
        self, mock_params, mock_ext_params, mock_completion, monkeypatch,
    ):
        self._set_mode(monkeypatch, "always")
        mock_params.return_value = {"model": "judge"}
        mock_ext_params.return_value = {"model": "extractor"}

        garbage = MagicMock(choices=[MagicMock(message=MagicMock(
            content="not json at all"
        ))])
        single_pass_resp = self._verdict_response(
            {"verdict": "APPROVED", "summary": "fallback ok"},
        )
        mock_completion.side_effect = [garbage, garbage, single_pass_resp]

        engine = ReviewEngine()
        result = engine.review(
            task_description="task",
            developer_result="done",
            file_changes_summary="--- a/a.py\n+++ b/a.py\n+x",
            file_contents={"a.py": "x"},
        )
        assert result.is_approved
        assert "fallback ok" in result.summary
        # 2 extractor attempts + 1 single-pass fallback
        assert mock_completion.call_count == 3

    @patch("litellm.completion")
    @patch("infinidev.config.llm.get_litellm_params_for_review_extractor")
    @patch("infinidev.config.llm.get_litellm_params")
    def test_multi_pass_extraction_raises_falls_back(
        self, mock_params, mock_ext_params, mock_completion, monkeypatch,
    ):
        self._set_mode(monkeypatch, "always")
        mock_params.return_value = {"model": "judge"}
        mock_ext_params.return_value = {"model": "extractor"}

        fallback_resp = self._verdict_response(
            {"verdict": "APPROVED", "summary": "recovered"},
        )
        mock_completion.side_effect = [RuntimeError("boom"), fallback_resp]

        engine = ReviewEngine()
        result = engine.review(
            task_description="task",
            developer_result="done",
            file_changes_summary="--- a/a.py\n+++ b/a.py\n+x",
        )
        assert result.is_approved

    @pytest.mark.parametrize(
        "mode,lines,expected_calls",
        [
            ("off", 5000, 1),       # off ignores threshold
            ("auto", 10, 1),         # below threshold
            ("auto", 800, 2),        # above threshold
            ("always", 10, 2),       # always regardless
        ],
    )
    @patch("litellm.completion")
    @patch("infinidev.config.llm.get_litellm_params_for_review_extractor")
    @patch("infinidev.config.llm.get_litellm_params")
    def test_mode_and_threshold_gating(
        self, mock_params, mock_ext_params, mock_completion,
        mode, lines, expected_calls, monkeypatch,
    ):
        self._set_mode(monkeypatch, mode, threshold=400)
        mock_params.return_value = {"model": "judge"}
        mock_ext_params.return_value = {"model": "extractor"}

        if expected_calls == 2:
            mock_completion.side_effect = [
                self._extraction_response({
                    "changes": [], "plan_coverage": [],
                    "public_api_impact": {"new_exports": [], "removed_exports": [], "signature_changes": []},
                    "report_discrepancies": [],
                }),
                self._verdict_response({"verdict": "APPROVED", "summary": "ok"}),
            ]
        else:
            mock_completion.side_effect = [
                self._verdict_response({"verdict": "APPROVED", "summary": "ok"}),
            ]

        diff = "--- a/a.py\n+++ b/a.py\n" + "\n".join(f"+line{i}" for i in range(lines))
        engine = ReviewEngine()
        engine.review(
            task_description="task",
            developer_result="done",
            file_changes_summary=diff,
            file_contents={"a.py": "x"},
        )
        assert mock_completion.call_count == expected_calls

    def test_compute_complexity_score(self):
        """Score = changed_lines + 50 * changed_files."""
        diff = "\n".join(f"+line{i}" for i in range(10))
        score = ReviewEngine._compute_complexity(
            diff, {"a.py": "", "b.py": "", "c.py": ""},
        )
        # 10 content lines + 3 files * 50 = 160
        assert score == 160

    def test_extractor_prompt_builder(self):
        eng = ReviewEngine()
        prompt = eng._build_extractor_prompt(
            task_description="Add login",
            developer_result="done",
            file_changes_summary="--- a/auth.py\n+++ b/auth.py\n+def login():",
            file_contents={"auth.py": "def login(): pass"},
            plan_steps=[{"step": 1, "title": "write login"}],
        )
        assert "## Original Task" in prompt
        assert "## Plan" in prompt
        assert "## Diffs" in prompt
        # Extractor prompt body should not mention severity/judgment language
        assert "severity" not in prompt.lower()
        assert "blocking" not in prompt.lower()
        assert "reject" not in prompt.lower()

    def test_judge_prompt_omits_diffs(self):
        eng = ReviewEngine()
        extraction = {
            "changes": [{"file": "a.py", "summary": "added f"}],
            "plan_coverage": [],
            "public_api_impact": {},
            "report_discrepancies": [],
        }
        prompt = eng._build_judge_prompt(
            extraction=extraction,
            task_description="task",
            developer_result="done",
            plan_steps=[],
            automated_checks={},
            previous_feedback="",
            recent_messages=[],
        )
        assert "## Extraction" in prompt
        assert "added f" in prompt
        assert "## Diffs" not in prompt

    def test_parallel_extraction_and_checks_integration(
        self, tmp_path, monkeypatch,
    ):
        """run_review_rework_loop runs Pass A in parallel with
        collect_automated_checks when multi-pass fires."""
        from infinidev.engine.analysis.review_engine import (
            run_review_rework_loop,
        )
        from infinidev.engine.file_change_tracker import FileChangeTracker

        self._set_mode(monkeypatch, "always")

        # Minimal stub engine exposing the API the loop needs.
        class _StubEngine:
            def __init__(self):
                self._tracker = FileChangeTracker()
                self._workspace = str(tmp_path)

            def get_file_contents(self):
                return {"a.py": "def f(): pass"}

            def get_file_change_reasons(self):
                return {"a.py": ["added f"]}

            def get_changed_files_summary(self):
                return "--- a/a.py\n+++ b/a.py\n+def f(): pass"

            def get_file_tracker(self):
                return self._tracker

            def get_plan_steps(self):
                return [{"step": 1, "title": "add f"}]

            def has_file_changes(self):
                return True

            def execute(self, **_kw):
                return "Done"

        class _StubAgent:
            def activate_context(self, **_kw):
                pass

            def deactivate(self):
                pass

        extraction = {
            "changes": [], "plan_coverage": [],
            "public_api_impact": {"new_exports": [], "removed_exports": [], "signature_changes": []},
            "report_discrepancies": [],
        }
        verdict = {"verdict": "APPROVED", "summary": "ok"}

        with patch("litellm.completion") as mock_completion, \
                patch("infinidev.config.llm.get_litellm_params_for_review_extractor",
                      return_value={"model": "ext"}), \
                patch("infinidev.config.llm.get_litellm_params",
                      return_value={"model": "judge"}), \
                patch("infinidev.engine.analysis.review_engine.collect_automated_checks") as mock_collect, \
                patch("infinidev.engine.analysis.verification_engine.VerificationEngine") as MockVE:
            mock_completion.side_effect = [
                self._extraction_response(extraction),
                self._verdict_response(verdict),
            ]
            mock_collect.return_value = {
                "verification_passed": True,
                "orphaned_references": [],
                "missing_docstrings": [],
            }

            reviewer = ReviewEngine()
            _result, review = run_review_rework_loop(
                engine=_StubEngine(),
                agent=_StubAgent(),
                session_id="sid",
                task_prompt=("add f", "Complete the task."),
                initial_result="Added f",
                reviewer=reviewer,
                recent_messages=[],
                on_status=None,
            )

            assert review is not None
            assert review.is_approved
            assert mock_completion.call_count == 2
            assert mock_collect.call_count == 1


class TestFileChangeTrackerReasons:
    """Test reason tracking in FileChangeTracker."""

    def test_record_and_get_reasons(self):
        from infinidev.engine.file_change_tracker import FileChangeTracker
        tracker = FileChangeTracker()
        tracker.record_reason("/tmp/foo.py", "Fix off-by-one error")
        tracker.record_reason("/tmp/foo.py", "Also fix edge case")
        tracker.record_reason("/tmp/bar.py", "New utility module")

        assert tracker.get_reasons("/tmp/foo.py") == [
            "Fix off-by-one error",
            "Also fix edge case",
        ]
        assert tracker.get_reasons("/tmp/bar.py") == ["New utility module"]
        assert tracker.get_reasons("/tmp/unknown.py") == []

    def test_empty_reason_ignored(self):
        from infinidev.engine.file_change_tracker import FileChangeTracker
        tracker = FileChangeTracker()
        tracker.record_reason("/tmp/foo.py", "")
        tracker.record_reason("/tmp/foo.py", "   ")
        assert tracker.get_reasons("/tmp/foo.py") == []

    def test_reset_clears_reasons(self):
        from infinidev.engine.file_change_tracker import FileChangeTracker
        tracker = FileChangeTracker()
        tracker.record_reason("/tmp/foo.py", "some reason")
        tracker.reset()
        assert tracker.get_reasons("/tmp/foo.py") == []

    def test_record_deleted_symbols(self):
        from infinidev.engine.file_change_tracker import FileChangeTracker
        tracker = FileChangeTracker()
        tracker.record_deleted_symbols("/tmp/foo.py", ["old_func", "HelperClass"])
        tracker.record_deleted_symbols("/tmp/foo.py", ["another_func"])
        tracker.record_deleted_symbols("/tmp/bar.py", ["MyClass.my_method"])

        deleted = tracker.get_deleted_symbols()
        assert "/tmp/foo.py" in deleted
        assert "old_func" in deleted["/tmp/foo.py"]
        assert "HelperClass" in deleted["/tmp/foo.py"]
        assert "another_func" in deleted["/tmp/foo.py"]
        assert "/tmp/bar.py" in deleted
        assert "MyClass.my_method" in deleted["/tmp/bar.py"]

    def test_record_deleted_symbols_empty_list_ignored(self):
        from infinidev.engine.file_change_tracker import FileChangeTracker
        tracker = FileChangeTracker()
        tracker.record_deleted_symbols("/tmp/foo.py", [])
        assert tracker.get_deleted_symbols() == {}

    def test_reset_clears_deleted_symbols(self):
        from infinidev.engine.file_change_tracker import FileChangeTracker
        tracker = FileChangeTracker()
        tracker.record_deleted_symbols("/tmp/foo.py", ["removed_func"])
        tracker.reset()
        assert tracker.get_deleted_symbols() == {}

    def test_maybe_emit_file_change_forwards_removed_symbols(self, tmp_path):
        """Integration: the hook called by the tool executor must forward
        `removed_symbols` from the tool result into the tracker. This is the
        wiring that makes the orphaned-references check actually run."""
        from infinidev.engine.file_change_tracker import FileChangeTracker
        from infinidev.engine.tool_executor import maybe_emit_file_change

        target = tmp_path / "mod.py"
        target.write_text("# reduced content\n")

        tracker = FileChangeTracker()
        result = json.dumps({
            "file_path": str(target),
            "action": "modified",
            "size_bytes": 20,
            "warning": "deleted symbol",
            "removed_symbols": ["deleted_func", "GoneClass"],
        })

        maybe_emit_file_change(
            tool_name="write_file",
            arguments={"file_path": str(target), "content": "# reduced\n"},
            result=result,
            pre_content="def deleted_func():\n    return 1\n",
            tracker=tracker,
            project_id=1,
            agent_id="test-agent",
        )

        deleted = tracker.get_deleted_symbols()
        abs_target = str(target.resolve())
        assert abs_target in deleted
        assert deleted[abs_target] == {"deleted_func", "GoneClass"}

    def test_prompt_builder_includes_plan_section(self):
        eng = ReviewEngine()
        plan = [
            {"step": 1, "title": "Add schema", "explanation": "Create Users table", "files": ["schema.sql"]},
            {"step": 2, "title": "Wire up migration"},
        ]
        prompt = eng._build_review_prompt(
            task_description="Add user auth",
            developer_result="Done",
            file_changes_summary="",
            previous_feedback="",
            plan_steps=plan,
        )
        assert "## Plan (what the developer committed to)" in prompt
        assert "1. Add schema [schema.sql]" in prompt
        assert "Create Users table" in prompt
        assert "2. Wire up migration" in prompt

    def test_prompt_builder_includes_automated_checks(self):
        eng = ReviewEngine()
        checks = {
            "verification_passed": True,
            "orphaned_references": [
                {"file": "a.py", "line": 10, "message": "Symbol 'foo' removed but used at a.py:10."},
            ],
            "missing_docstrings": [
                {"file": "b.py", "line": 3, "message": "Class 'Bar' is missing a docstring."},
            ],
        }
        prompt = eng._build_review_prompt(
            task_description="t",
            developer_result="r",
            file_changes_summary="",
            previous_feedback="",
            automated_checks=checks,
        )
        assert "## Automated Checks" in prompt
        assert "orphaned_references: 1" in prompt
        assert "Symbol 'foo' removed" in prompt
        assert "missing_docstrings: 1" in prompt
        assert "Class 'Bar'" in prompt
        assert "tests/import-check: PASSED" in prompt

    def test_prompt_builder_omits_sections_when_empty(self):
        eng = ReviewEngine()
        prompt = eng._build_review_prompt(
            task_description="t",
            developer_result="r",
            file_changes_summary="",
            previous_feedback="",
        )
        assert "## Plan" not in prompt
        assert "## Automated Checks" not in prompt

    def test_collect_automated_checks_empty_when_no_changes(self):
        from infinidev.engine.analysis.review_engine import collect_automated_checks
        checks = collect_automated_checks(
            changed_files=[], file_tracker=None, verification_passed=True,
        )
        assert checks["verification_passed"] is True
        assert checks["orphaned_references"] == []
        assert checks["missing_docstrings"] == []

    def test_maybe_emit_file_change_no_removed_symbols(self, tmp_path):
        """When the tool result has no removed_symbols, tracker stays empty
        for deletions (regular diff tracking still happens)."""
        from infinidev.engine.file_change_tracker import FileChangeTracker
        from infinidev.engine.tool_executor import maybe_emit_file_change

        target = tmp_path / "clean.py"
        target.write_text("def added():\n    pass\n")

        tracker = FileChangeTracker()
        result = json.dumps({
            "file_path": str(target),
            "action": "modified",
            "size_bytes": 20,
        })

        maybe_emit_file_change(
            tool_name="write_file",
            arguments={"file_path": str(target), "content": "def added():\n    pass\n"},
            result=result,
            pre_content="",
            tracker=tracker,
            project_id=1,
            agent_id="test-agent",
        )

        assert tracker.get_deleted_symbols() == {}


class TestPhaseSettings:
    """Test ANALYSIS_ENABLED and REVIEW_ENABLED settings."""

    def test_analysis_enabled_default_true(self):
        from infinidev.config.settings import Settings
        s = Settings()
        assert s.ANALYSIS_ENABLED is True

    def test_review_enabled_default_true(self):
        from infinidev.config.settings import Settings
        s = Settings()
        assert s.REVIEW_ENABLED is True
