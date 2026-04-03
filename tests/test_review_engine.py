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
