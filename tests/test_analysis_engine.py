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


def _mock_loop_execute(return_json):
    """Helper: patch _run_analyst_loop to return a specific JSON string."""
    if isinstance(return_json, dict):
        return_json = json.dumps(return_json)
    return patch.object(
        AnalysisEngine,
        "_run_analyst_loop",
        return_value=AnalysisResult(
            **_parse_json_to_result_kwargs(return_json)
        ),
    )


def _parse_json_to_result_kwargs(raw):
    """Parse JSON into kwargs for AnalysisResult (mirrors _parse_response logic)."""
    data = json.loads(raw)
    action = data.get("action", "passthrough")
    kwargs = {"action": action, "original_input": "test"}
    if action == "passthrough":
        kwargs["reason"] = data.get("reason", "")
    elif action == "ask":
        kwargs["questions"] = data.get("questions", [])
        kwargs["context"] = data.get("context", "")
    elif action == "research":
        kwargs["research_queries"] = data.get("queries", [])
        kwargs["research_reason"] = data.get("reason", "")
    elif action == "proceed":
        kwargs["specification"] = data.get("specification", {})
    return kwargs


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

    @patch("infinidev.config.llm.get_litellm_params")
    def test_analyze_passthrough(self, mock_params):
        mock_params.return_value = {"model": "test"}
        passthrough_result = AnalysisResult(
            action="passthrough",
            original_input="hello",
            reason="Simple greeting",
        )
        with patch.object(AnalysisEngine, "_run_analyst_loop", return_value=passthrough_result):
            engine = AnalysisEngine()
            result = engine.analyze("hello")
        assert result.action == "passthrough"

    @patch("infinidev.config.llm.get_litellm_params")
    def test_analyze_proceed(self, mock_params):
        mock_params.return_value = {"model": "test"}
        proceed_result = AnalysisResult(
            action="proceed",
            original_input="add login to the app",
            specification={
                "summary": "Add a login page",
                "requirements": ["Login form", "Session management"],
            },
        )
        with patch.object(AnalysisEngine, "_run_analyst_loop", return_value=proceed_result):
            engine = AnalysisEngine()
            result = engine.analyze("add login to the app")
        assert result.action == "proceed"
        assert result.specification["summary"] == "Add a login page"

    @patch("infinidev.config.llm.get_litellm_params")
    def test_analyze_ask(self, mock_params):
        mock_params.return_value = {"model": "test"}
        ask_result = AnalysisResult(
            action="ask",
            original_input="build an app",
            questions=[{"question": "Web or mobile?", "why": "Stack choice", "options": ["Web", "Mobile"]}],
            context="You want an app",
        )
        with patch.object(AnalysisEngine, "_run_analyst_loop", return_value=ask_result):
            engine = AnalysisEngine()
            result = engine.analyze("build an app")
        assert result.action == "ask"
        assert len(result.questions) == 1
        assert "Web or mobile?" in result.questions[0]["question"]

    @patch("infinidev.config.llm.get_litellm_params")
    def test_analyze_loop_error_passthroughs(self, mock_params):
        """Agent loop errors should result in passthrough."""
        mock_params.return_value = {"model": "test"}
        with patch.object(AnalysisEngine, "_run_analyst_loop", side_effect=Exception("LLM is down")):
            engine = AnalysisEngine()
            result = engine.analyze("do something complex")
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

    def test_build_analysis_prompt_has_instructions(self):
        engine = AnalysisEngine()
        engine._analysis_rounds = 1
        prompt = engine._build_analysis_prompt("add auth", None)
        assert "## Instructions" in prompt
        assert "exploring the codebase" in prompt
        assert "Do NOT write or modify" in prompt

    @patch("infinidev.config.llm.get_litellm_params")
    def test_events_emitted_to_bus(self, mock_params):
        """Analysis events should be emitted to the EventBus."""
        mock_params.return_value = {"model": "test"}
        passthrough_result = AnalysisResult(
            action="passthrough", original_input="hello", reason="simple",
        )
        events = []
        def callback(event_type, *args):
            events.append(event_type)

        from infinidev.flows.event_listeners import event_bus
        event_bus.subscribe(callback)
        try:
            with patch.object(AnalysisEngine, "_run_analyst_loop", return_value=passthrough_result):
                engine = AnalysisEngine()
                engine.analyze("hello")
            assert "analysis_start" in events
            assert "analysis_complete" in events
        finally:
            event_bus.unsubscribe(callback)

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

    def test_parse_response_passthrough(self):
        engine = AnalysisEngine()
        result = engine._parse_response(
            '{"action": "passthrough", "reason": "simple"}', "hello"
        )
        assert result.action == "passthrough"

    def test_parse_response_json_in_markdown(self):
        engine = AnalysisEngine()
        result = engine._parse_response(
            '```json\n{"action": "passthrough", "reason": "simple"}\n```', "hello"
        )
        assert result.action == "passthrough"

    def test_parse_response_json_with_preamble(self):
        engine = AnalysisEngine()
        result = engine._parse_response(
            'Here is the analysis:\n{"action": "passthrough", "reason": "simple"}\nDone.',
            "hello",
        )
        assert result.action == "passthrough"

    def test_parse_response_invalid_json(self):
        engine = AnalysisEngine()
        result = engine._parse_response(
            "I'm not sure what you want me to do.", "do something"
        )
        assert result.action == "passthrough"

    def test_parse_response_research(self):
        engine = AnalysisEngine()
        result = engine._parse_response(
            json.dumps({"action": "research", "queries": ["q1"], "reason": "need info"}),
            "test",
        )
        assert result.action == "research"
        assert result.research_queries == ["q1"]

    def test_parse_response_proceed(self):
        engine = AnalysisEngine()
        result = engine._parse_response(
            json.dumps({"action": "proceed", "specification": {"summary": "Test"}}),
            "test",
        )
        assert result.action == "proceed"
        assert result.specification["summary"] == "Test"


class TestAnalysisEngineResearch:
    """Test research functionality in AnalysisEngine."""

    @patch("infinidev.config.llm.get_litellm_params")
    def test_analyze_research_action(self, mock_params):
        """Research action triggers web search, then second loop produces proceed."""
        mock_params.return_value = {"model": "test"}

        research_result = AnalysisResult(
            action="research",
            original_input="integrate stripe webhooks",
            research_queries=["Stripe webhook API v2"],
            research_reason="Need current Stripe API details",
        )
        proceed_result = AnalysisResult(
            action="proceed",
            original_input="integrate stripe webhooks",
            specification={
                "summary": "Integrate Stripe webhooks",
                "requirements": ["Verify webhook signatures"],
            },
        )

        call_count = [0]
        def mock_run_loop(user_input, session_summaries):
            call_count[0] += 1
            if call_count[0] == 1:
                return research_result
            return proceed_result

        with patch.object(AnalysisEngine, "_run_analyst_loop", side_effect=mock_run_loop), \
             patch("infinidev.tools.web.backends.search_ddg", return_value=[
                 {"title": "Stripe Docs", "url": "https://stripe.com/docs", "snippet": "Webhook guide"},
             ]), \
             patch("infinidev.tools.web.backends.fetch_with_trafilatura", return_value="Stripe webhook content"):
            engine = AnalysisEngine()
            result = engine.analyze("integrate stripe webhooks")

        assert result.action == "proceed"
        assert "Stripe webhooks" in result.specification["summary"]
        assert call_count[0] == 2

    @patch("infinidev.config.llm.get_litellm_params")
    def test_analyze_research_no_results(self, mock_params):
        """Empty search results still produce a spec on second loop."""
        mock_params.return_value = {"model": "test"}

        research_result = AnalysisResult(
            action="research",
            original_input="use obscure API",
            research_queries=["obscure API"],
            research_reason="Need info",
        )
        proceed_result = AnalysisResult(
            action="proceed",
            original_input="use obscure API",
            specification={"summary": "Best effort spec"},
        )

        call_count = [0]
        def mock_run_loop(user_input, session_summaries):
            call_count[0] += 1
            if call_count[0] == 1:
                return research_result
            return proceed_result

        with patch.object(AnalysisEngine, "_run_analyst_loop", side_effect=mock_run_loop), \
             patch("infinidev.tools.web.backends.search_ddg", return_value=[]), \
             patch("infinidev.tools.web.backends.fetch_with_trafilatura", return_value=None):
            engine = AnalysisEngine()
            result = engine.analyze("use obscure API")

        assert result.action == "proceed"

    @patch("infinidev.config.llm.get_litellm_params")
    def test_analyze_research_prevents_loop(self, mock_params):
        """Two consecutive research actions → passthrough to prevent loop."""
        mock_params.return_value = {"model": "test"}

        research_result = AnalysisResult(
            action="research",
            original_input="complex request",
            research_queries=["query1"],
            research_reason="Need info",
        )

        with patch.object(AnalysisEngine, "_run_analyst_loop", return_value=research_result), \
             patch("infinidev.tools.web.backends.search_ddg", return_value=[]), \
             patch("infinidev.tools.web.backends.fetch_with_trafilatura", return_value=None):
            engine = AnalysisEngine()
            result = engine.analyze("complex request")

        assert result.action == "passthrough"
        assert "loop" in result.reason.lower()

    @patch("infinidev.config.llm.get_litellm_params")
    def test_analyze_research_max_queries(self, mock_params):
        """Only first 3 queries are searched even if more are provided."""
        mock_params.return_value = {"model": "test"}

        research_result = AnalysisResult(
            action="research",
            original_input="big request",
            research_queries=["q1", "q2", "q3", "q4", "q5"],
            research_reason="Need lots of info",
        )
        proceed_result = AnalysisResult(
            action="proceed",
            original_input="big request",
            specification={"summary": "Done"},
        )

        call_count = [0]
        def mock_run_loop(user_input, session_summaries):
            call_count[0] += 1
            if call_count[0] == 1:
                return research_result
            return proceed_result

        search_calls = []
        def mock_search(query, num_results=3):
            search_calls.append(query)
            return []

        with patch.object(AnalysisEngine, "_run_analyst_loop", side_effect=mock_run_loop), \
             patch("infinidev.tools.web.backends.search_ddg", side_effect=mock_search), \
             patch("infinidev.tools.web.backends.fetch_with_trafilatura", return_value=None):
            engine = AnalysisEngine()
            result = engine.analyze("big request")

        assert len(search_calls) == 3
        assert result.action == "proceed"

    @patch("infinidev.config.llm.get_litellm_params")
    def test_analyze_research_fetch_failure(self, mock_params):
        """Fetch failure degrades gracefully without error."""
        mock_params.return_value = {"model": "test"}

        research_result = AnalysisResult(
            action="research",
            original_input="test request",
            research_queries=["test query"],
            research_reason="Need info",
        )
        proceed_result = AnalysisResult(
            action="proceed",
            original_input="test request",
            specification={"summary": "Spec without fetch"},
        )

        call_count = [0]
        def mock_run_loop(user_input, session_summaries):
            call_count[0] += 1
            if call_count[0] == 1:
                return research_result
            return proceed_result

        with patch.object(AnalysisEngine, "_run_analyst_loop", side_effect=mock_run_loop), \
             patch("infinidev.tools.web.backends.search_ddg", return_value=[
                 {"title": "Result", "url": "https://example.com", "snippet": "A result"},
             ]), \
             patch("infinidev.tools.web.backends.fetch_with_trafilatura", side_effect=Exception("timeout")):
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


class TestAnalysisEngineRunLoop:
    """Test _run_analyst_loop creates agent with correct config."""

    @patch("infinidev.engine.loop_engine.LoopEngine", new_callable=MagicMock)
    @patch("infinidev.agents.base.InfinidevAgent", new_callable=MagicMock)
    def test_analyst_agent_has_identity_override(self, mock_agent_cls, mock_engine_cls):
        """Analyst agent should have _system_prompt_identity set."""
        from infinidev.prompts.analyst.system import ANALYST_SYSTEM_PROMPT

        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent
        mock_engine = MagicMock()
        mock_engine.execute.return_value = '{"action": "passthrough", "reason": "test"}'
        mock_engine_cls.return_value = mock_engine

        engine = AnalysisEngine()
        engine._analysis_rounds = 1
        result = engine._run_analyst_loop("test", None)

        # Verify agent was created with analyst role
        mock_agent_cls.assert_called_once()
        call_kwargs = mock_agent_cls.call_args[1]
        assert call_kwargs["role"] == "analyst"
        assert call_kwargs["agent_id"] == "analyst"

        # Verify identity override was set
        assert mock_agent._system_prompt_identity == ANALYST_SYSTEM_PROMPT

        # Verify loop engine was called
        mock_engine.execute.assert_called_once()

    @patch("infinidev.engine.loop_engine.LoopEngine", new_callable=MagicMock)
    @patch("infinidev.agents.base.InfinidevAgent", new_callable=MagicMock)
    def test_analyst_loop_parses_output(self, mock_agent_cls, mock_engine_cls):
        """Loop engine output should be parsed into AnalysisResult."""
        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent
        mock_engine = MagicMock()
        mock_engine.execute.return_value = json.dumps({
            "action": "proceed",
            "specification": {"summary": "Built from codebase analysis"},
        })
        mock_engine_cls.return_value = mock_engine

        engine = AnalysisEngine()
        engine._analysis_rounds = 1
        result = engine._run_analyst_loop("add feature", None)

        assert result.action == "proceed"
        assert result.specification["summary"] == "Built from codebase analysis"

    @patch("infinidev.engine.loop_engine.LoopEngine", new_callable=MagicMock)
    @patch("infinidev.agents.base.InfinidevAgent", new_callable=MagicMock)
    def test_analyst_context_activated_and_deactivated(self, mock_agent_cls, mock_engine_cls):
        """Agent context should be activated before and deactivated after loop."""
        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent
        mock_engine = MagicMock()
        mock_engine.execute.return_value = '{"action": "passthrough", "reason": "test"}'
        mock_engine_cls.return_value = mock_engine

        engine = AnalysisEngine()
        engine._analysis_rounds = 1
        engine._run_analyst_loop("test", None)

        mock_agent.activate_context.assert_called_once()
        mock_agent.deactivate.assert_called_once()

    @patch("infinidev.engine.loop_engine.LoopEngine", new_callable=MagicMock)
    @patch("infinidev.agents.base.InfinidevAgent", new_callable=MagicMock)
    def test_analyst_deactivates_on_error(self, mock_agent_cls, mock_engine_cls):
        """Agent should be deactivated even if loop engine raises."""
        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent
        mock_engine = MagicMock()
        mock_engine.execute.side_effect = RuntimeError("boom")
        mock_engine_cls.return_value = mock_engine

        engine = AnalysisEngine()
        engine._analysis_rounds = 1
        with pytest.raises(RuntimeError):
            engine._run_analyst_loop("test", None)

        mock_agent.deactivate.assert_called_once()
