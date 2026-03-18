"""Tests for the pre-development analysis engine."""

import json
import pytest
from unittest.mock import patch, MagicMock

from infinidev.engine.analysis_engine import AnalysisEngine, AnalysisResult


class TestAnalysisResult:
    """Test AnalysisResult data class."""

    def test_passthrough_build_developer_prompt(self):
        result = AnalysisResult(
            action="passthrough",
            original_input="hello world",
        )
        desc, expected = result.build_developer_prompt()
        assert desc == "hello world"
        assert "Complete the task" in expected

    def test_proceed_build_developer_prompt(self):
        result = AnalysisResult(
            action="proceed",
            original_input="add auth to the API",
            specification={
                "summary": "Add JWT authentication to the REST API",
                "requirements": [
                    "JWT-based auth on all endpoints",
                    "Login endpoint at /auth/login",
                ],
                "hidden_requirements": [
                    "Session management with token refresh",
                ],
                "assumptions": [
                    "Using bcrypt for password hashing",
                ],
                "out_of_scope": [
                    "OAuth2 / social login",
                ],
                "technical_notes": "Use PyJWT library",
            },
        )
        desc, expected = result.build_developer_prompt()
        assert "## User Request" in desc
        assert "add auth to the API" in desc
        assert "## Analysis Summary" in desc
        assert "JWT authentication" in desc
        assert "## Requirements" in desc
        assert "JWT-based auth" in desc
        assert "## Identified Hidden Requirements" in desc
        assert "Session management" in desc
        assert "## Assumptions" in desc
        assert "bcrypt" in desc
        assert "## Out of Scope" in desc
        assert "OAuth2" in desc
        assert "## Technical Notes" in desc
        assert "PyJWT" in desc

    def test_proceed_empty_specification(self):
        result = AnalysisResult(
            action="proceed",
            original_input="do something",
            specification={},
        )
        desc, expected = result.build_developer_prompt()
        # Falls back to original input since spec is empty
        assert desc == "do something"

    def test_ask_build_developer_prompt_returns_original(self):
        """ask action returns original input (shouldn't be called normally)."""
        result = AnalysisResult(
            action="ask",
            original_input="build an app",
        )
        desc, expected = result.build_developer_prompt()
        assert desc == "build an app"

    def test_format_questions_for_user(self):
        result = AnalysisResult(
            action="ask",
            original_input="build an app",
            questions=[
                {
                    "question": "What platform?",
                    "why": "Determines the tech stack",
                    "options": ["Web app", "Mobile app", "CLI tool"],
                },
            ],
            context="I understand you want to build an app.",
        )
        text = result.format_questions_for_user()
        assert "What platform?" in text
        assert "Determines the tech stack" in text
        assert "Web app" in text
        assert "I understand you want to build an app" in text

    def test_format_questions_empty(self):
        result = AnalysisResult(action="ask", original_input="x")
        assert result.format_questions_for_user() == ""


class TestAnalysisEngine:
    """Test AnalysisEngine."""

    def test_reset(self):
        engine = AnalysisEngine()
        engine._qa_history = [{"q": "a", "a": "b"}]
        engine._analysis_rounds = 3
        engine.reset()
        assert engine._qa_history == []
        assert engine._analysis_rounds == 0

    def test_add_answer(self):
        engine = AnalysisEngine()
        engine.add_answer("What platform?", "Web app")
        assert len(engine._qa_history) == 1
        assert engine._qa_history[0]["answer"] == "Web app"

    def test_can_ask_more(self):
        engine = AnalysisEngine()
        assert engine.can_ask_more is True
        engine._analysis_rounds = 3
        assert engine.can_ask_more is False

    @patch("infinidev.config.llm.get_litellm_params")
    def test_analyze_no_llm(self, mock_params):
        """Without LLM params, should passthrough."""
        mock_params.return_value = None
        engine = AnalysisEngine()
        result = engine.analyze("hello")
        assert result.action == "passthrough"

    @patch("litellm.completion")
    @patch("infinidev.config.llm.get_litellm_params")
    def test_analyze_passthrough(self, mock_params, mock_completion):
        mock_params.return_value = {"model": "test"}
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content=json.dumps({
                    "action": "passthrough",
                    "reason": "Simple greeting",
                })
            ))]
        )
        engine = AnalysisEngine()
        result = engine.analyze("hello")
        assert result.action == "passthrough"

    @patch("litellm.completion")
    @patch("infinidev.config.llm.get_litellm_params")
    def test_analyze_proceed(self, mock_params, mock_completion):
        mock_params.return_value = {"model": "test"}
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content=json.dumps({
                    "action": "proceed",
                    "specification": {
                        "summary": "Add a login page",
                        "requirements": ["Login form", "Session management"],
                    },
                })
            ))]
        )
        engine = AnalysisEngine()
        result = engine.analyze("add login to the app")
        assert result.action == "proceed"
        assert result.specification["summary"] == "Add a login page"

    @patch("litellm.completion")
    @patch("infinidev.config.llm.get_litellm_params")
    def test_analyze_ask(self, mock_params, mock_completion):
        mock_params.return_value = {"model": "test"}
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content=json.dumps({
                    "action": "ask",
                    "questions": [{"question": "Web or mobile?", "why": "Stack choice", "options": ["Web", "Mobile"]}],
                    "context": "You want an app",
                })
            ))]
        )
        engine = AnalysisEngine()
        result = engine.analyze("build an app")
        assert result.action == "ask"
        assert len(result.questions) == 1
        assert "Web or mobile?" in result.questions[0]["question"]

    @patch("litellm.completion")
    @patch("infinidev.config.llm.get_litellm_params")
    def test_analyze_json_in_markdown(self, mock_params, mock_completion):
        """Should handle JSON wrapped in markdown code blocks."""
        mock_params.return_value = {"model": "test"}
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='```json\n{"action": "passthrough", "reason": "simple"}\n```'
            ))]
        )
        engine = AnalysisEngine()
        result = engine.analyze("hi")
        assert result.action == "passthrough"

    @patch("litellm.completion")
    @patch("infinidev.config.llm.get_litellm_params")
    def test_analyze_json_with_preamble(self, mock_params, mock_completion):
        """Should handle JSON with text before/after."""
        mock_params.return_value = {"model": "test"}
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='Here is the analysis:\n{"action": "passthrough", "reason": "simple"}\nDone.'
            ))]
        )
        engine = AnalysisEngine()
        result = engine.analyze("hi")
        assert result.action == "passthrough"

    @patch("litellm.completion")
    @patch("infinidev.config.llm.get_litellm_params")
    def test_analyze_llm_error_passthroughs(self, mock_params, mock_completion):
        """LLM errors should result in passthrough."""
        mock_params.return_value = {"model": "test"}
        mock_completion.side_effect = Exception("LLM is down")
        engine = AnalysisEngine()
        result = engine.analyze("do something complex")
        assert result.action == "passthrough"

    @patch("litellm.completion")
    @patch("infinidev.config.llm.get_litellm_params")
    def test_analyze_invalid_json_passthroughs(self, mock_params, mock_completion):
        """Unparseable response should result in passthrough."""
        mock_params.return_value = {"model": "test"}
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="I'm not sure what you want me to do."
            ))]
        )
        engine = AnalysisEngine()
        result = engine.analyze("do something")
        assert result.action == "passthrough"

    def test_build_analysis_prompt_with_history(self):
        engine = AnalysisEngine()
        engine.add_answer("What platform?", "Web")
        engine._analysis_rounds = 1
        prompt = engine._build_analysis_prompt(
            "build an app",
            session_summaries=["User asked about auth previously"],
        )
        assert "## Conversation History" in prompt
        assert "auth previously" in prompt
        assert "## Previous Clarifications" in prompt
        assert "Web" in prompt
        assert "## User Request" in prompt
        assert "build an app" in prompt

    def test_build_analysis_prompt_final_round(self):
        engine = AnalysisEngine()
        engine._analysis_rounds = 3  # max rounds
        prompt = engine._build_analysis_prompt("build an app", None)
        assert "FINAL ROUND" in prompt
        assert "MUST produce a specification" in prompt

    @patch("litellm.completion")
    @patch("infinidev.config.llm.get_litellm_params")
    def test_event_callback_called(self, mock_params, mock_completion):
        """Event callback should be called on start and complete."""
        mock_params.return_value = {"model": "test"}
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='{"action": "passthrough", "reason": "simple"}'
            ))]
        )
        events = []
        def callback(event_type, *args):
            events.append(event_type)

        engine = AnalysisEngine()
        engine.analyze("hello", event_callback=callback)
        assert "analysis_start" in events
        assert "analysis_complete" in events

    def test_proceed_with_partial_spec(self):
        """Specification with only some fields should work."""
        result = AnalysisResult(
            action="proceed",
            original_input="fix the bug",
            specification={
                "summary": "Fix the null pointer in auth.py",
                "requirements": ["Handle None return from get_user()"],
            },
        )
        desc, _ = result.build_developer_prompt()
        assert "Fix the null pointer" in desc
        assert "Handle None" in desc
        assert "Hidden Requirements" not in desc  # empty list, not included


class TestAnalysisEngineResearch:
    """Test research functionality in AnalysisEngine."""

    @patch("litellm.completion")
    @patch("infinidev.config.llm.get_litellm_params")
    def test_analyze_research_action(self, mock_params, mock_completion):
        """Research action triggers web search, then second LLM call produces proceed."""
        mock_params.return_value = {"model": "test"}
        # First call returns research, second returns proceed
        mock_completion.side_effect = [
            MagicMock(choices=[MagicMock(message=MagicMock(
                content=json.dumps({
                    "action": "research",
                    "queries": ["Stripe webhook API v2"],
                    "reason": "Need current Stripe API details",
                })
            ))]),
            MagicMock(choices=[MagicMock(message=MagicMock(
                content=json.dumps({
                    "action": "proceed",
                    "specification": {
                        "summary": "Integrate Stripe webhooks",
                        "requirements": ["Verify webhook signatures"],
                    },
                })
            ))]),
        ]
        with patch("infinidev.tools.web.backends.search_ddg", return_value=[
            {"title": "Stripe Docs", "url": "https://stripe.com/docs", "snippet": "Webhook guide"},
        ]), patch("infinidev.tools.web.backends.fetch_with_trafilatura", return_value="Stripe webhook content"):
            engine = AnalysisEngine()
            result = engine.analyze("integrate stripe webhooks")

        assert result.action == "proceed"
        assert "Stripe webhooks" in result.specification["summary"]
        assert mock_completion.call_count == 2

    @patch("litellm.completion")
    @patch("infinidev.config.llm.get_litellm_params")
    def test_analyze_research_no_results(self, mock_params, mock_completion):
        """Empty search results still produce a spec on second call."""
        mock_params.return_value = {"model": "test"}
        mock_completion.side_effect = [
            MagicMock(choices=[MagicMock(message=MagicMock(
                content=json.dumps({
                    "action": "research",
                    "queries": ["obscure API"],
                    "reason": "Need info",
                })
            ))]),
            MagicMock(choices=[MagicMock(message=MagicMock(
                content=json.dumps({
                    "action": "proceed",
                    "specification": {"summary": "Best effort spec"},
                })
            ))]),
        ]
        with patch("infinidev.tools.web.backends.search_ddg", return_value=[]), \
             patch("infinidev.tools.web.backends.fetch_with_trafilatura", return_value=None):
            engine = AnalysisEngine()
            result = engine.analyze("use obscure API")

        assert result.action == "proceed"

    @patch("litellm.completion")
    @patch("infinidev.config.llm.get_litellm_params")
    def test_analyze_research_prevents_loop(self, mock_params, mock_completion):
        """Two consecutive research actions → passthrough to prevent loop."""
        mock_params.return_value = {"model": "test"}
        mock_completion.side_effect = [
            MagicMock(choices=[MagicMock(message=MagicMock(
                content=json.dumps({
                    "action": "research",
                    "queries": ["query1"],
                    "reason": "Need info",
                })
            ))]),
            MagicMock(choices=[MagicMock(message=MagicMock(
                content=json.dumps({
                    "action": "research",
                    "queries": ["query2"],
                    "reason": "Still need info",
                })
            ))]),
        ]
        with patch("infinidev.tools.web.backends.search_ddg", return_value=[]), \
             patch("infinidev.tools.web.backends.fetch_with_trafilatura", return_value=None):
            engine = AnalysisEngine()
            result = engine.analyze("complex request")

        assert result.action == "passthrough"
        assert "loop" in result.reason.lower()

    @patch("litellm.completion")
    @patch("infinidev.config.llm.get_litellm_params")
    def test_analyze_research_max_queries(self, mock_params, mock_completion):
        """Only first 3 queries are searched even if more are provided."""
        mock_params.return_value = {"model": "test"}
        mock_completion.side_effect = [
            MagicMock(choices=[MagicMock(message=MagicMock(
                content=json.dumps({
                    "action": "research",
                    "queries": ["q1", "q2", "q3", "q4", "q5"],
                    "reason": "Need lots of info",
                })
            ))]),
            MagicMock(choices=[MagicMock(message=MagicMock(
                content=json.dumps({
                    "action": "proceed",
                    "specification": {"summary": "Done"},
                })
            ))]),
        ]
        search_calls = []

        def mock_search(query, num_results=3):
            search_calls.append(query)
            return []

        with patch("infinidev.tools.web.backends.search_ddg", side_effect=mock_search), \
             patch("infinidev.tools.web.backends.fetch_with_trafilatura", return_value=None):
            engine = AnalysisEngine()
            result = engine.analyze("big request")

        assert len(search_calls) == 3
        assert result.action == "proceed"

    @patch("litellm.completion")
    @patch("infinidev.config.llm.get_litellm_params")
    def test_analyze_research_fetch_failure(self, mock_params, mock_completion):
        """Fetch failure degrades gracefully without error."""
        mock_params.return_value = {"model": "test"}
        mock_completion.side_effect = [
            MagicMock(choices=[MagicMock(message=MagicMock(
                content=json.dumps({
                    "action": "research",
                    "queries": ["test query"],
                    "reason": "Need info",
                })
            ))]),
            MagicMock(choices=[MagicMock(message=MagicMock(
                content=json.dumps({
                    "action": "proceed",
                    "specification": {"summary": "Spec without fetch"},
                })
            ))]),
        ]
        with patch("infinidev.tools.web.backends.search_ddg", return_value=[
            {"title": "Result", "url": "https://example.com", "snippet": "A result"},
        ]), patch("infinidev.tools.web.backends.fetch_with_trafilatura", side_effect=Exception("timeout")):
            engine = AnalysisEngine()
            result = engine.analyze("test request")

        assert result.action == "proceed"

    def test_perform_research_formatting(self):
        """Test _perform_research formats results correctly."""
        with patch("infinidev.tools.web.backends.search_ddg", return_value=[
            {"title": "Doc Page", "url": "https://example.com/doc", "snippet": "The documentation"},
            {"title": "Blog Post", "url": "https://blog.com/post", "snippet": "A blog post"},
        ]), patch("infinidev.tools.web.backends.fetch_with_trafilatura", return_value="Full page content here"):
            engine = AnalysisEngine()
            result = engine._perform_research(["test query"])

        assert "## Research Results" in result
        assert "### Query: test query" in result
        assert "Doc Page" in result
        assert "https://example.com/doc" in result
        assert "The documentation" in result
        assert "Full page content here" in result
        assert "Blog Post" in result
