"""Tests for the post-development code review engine."""

import json
import pytest
from unittest.mock import patch, MagicMock

from infinidev.engine.review_engine import ReviewEngine, ReviewResult


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
    def test_review_llm_error_approves(self, mock_params, mock_completion):
        mock_params.return_value = {"model": "test"}
        mock_completion.side_effect = Exception("LLM down")
        engine = ReviewEngine()
        result = engine.review(
            task_description="x",
            developer_result="y",
            file_changes_summary="diff here",
        )
        assert result.is_approved  # Fails gracefully

    @patch("litellm.completion")
    @patch("infinidev.config.llm.get_litellm_params")
    def test_review_invalid_json_approves(self, mock_params, mock_completion):
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
        assert result.is_approved

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
    def test_event_callback_called(self, mock_params, mock_completion):
        mock_params.return_value = {"model": "test"}
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='{"verdict": "APPROVED", "summary": "ok"}'
            ))]
        )
        events = []
        def callback(event_type, *args):
            events.append(event_type)

        engine = ReviewEngine()
        engine.review(
            task_description="x",
            developer_result="y",
            file_changes_summary="diff here",
            event_callback=callback,
        )
        assert "review_start" in events
        assert "review_complete" in events

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
        assert "## Code Changes to Review" in prompt

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
