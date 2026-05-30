"""Tests for configuration modules: settings, LLM config, model capabilities."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from infinidev.config.settings import Settings, settings, reload_all, SETTINGS_FILE


# ── Settings ─────────────────────────────────────────────────────────────────


class TestSettings:
    """Tests for Settings class."""

    def test_default_values(self):
        """Key defaults match expected values."""
        s = Settings()
        assert s.LOOP_MAX_ITERATIONS == 50
        assert s.LOOP_MAX_TOTAL_TOOL_CALLS == 1000
        assert s.MAX_FILE_SIZE_BYTES == 5 * 1024 * 1024
        assert s.SANDBOX_ENABLED is False
        assert s.DEDUP_SIMILARITY_THRESHOLD == 0.82

    def test_load_from_json_file(self, tmp_path):
        """Settings loaded from JSON file."""
        sf = tmp_path / "settings.json"
        sf.write_text(json.dumps({"LLM_MODEL": "test/model"}))
        with patch("infinidev.config.settings.SETTINGS_FILE", sf):
            s = Settings.load_user_settings()
        assert s.LLM_MODEL == "test/model"

    def test_load_missing_file(self, tmp_path):
        """Missing file uses defaults."""
        sf = tmp_path / "nonexistent.json"
        with patch("infinidev.config.settings.SETTINGS_FILE", sf):
            s = Settings.load_user_settings()
        assert s.LLM_MODEL == "ollama_chat/qwen2.5-coder:7b"

    def test_load_corrupted_file(self, tmp_path):
        """Malformed JSON prints warning, uses defaults."""
        sf = tmp_path / "settings.json"
        sf.write_text("{not valid json!!!")
        with patch("infinidev.config.settings.SETTINGS_FILE", sf):
            s = Settings.load_user_settings()
        # Should not crash; defaults used
        assert s.LOOP_MAX_ITERATIONS == 50

    def test_env_var_override(self):
        """Env var INFINIDEV_LLM_MODEL takes precedence."""
        with patch.dict(os.environ, {"INFINIDEV_LLM_MODEL": "env/override"}):
            s = Settings()
        assert s.LLM_MODEL == "env/override"

    def test_save_user_settings_creates_file(self, tmp_path):
        """Save creates file with JSON content."""
        sf = tmp_path / "settings.json"
        with patch("infinidev.config.settings.SETTINGS_FILE", sf):
            s = Settings()
            s.save_user_settings({"LLM_MODEL": "saved/model"})
        data = json.loads(sf.read_text())
        assert data["LLM_MODEL"] == "saved/model"

    def test_save_user_settings_merges(self, tmp_path):
        """Save partial update preserves existing keys."""
        sf = tmp_path / "settings.json"
        sf.write_text(json.dumps({"LLM_MODEL": "old", "LOOP_MAX_ITERATIONS": 10}))
        with patch("infinidev.config.settings.SETTINGS_FILE", sf):
            s = Settings()
            s.save_user_settings({"LLM_MODEL": "new"})
        data = json.loads(sf.read_text())
        assert data["LLM_MODEL"] == "new"
        assert data["LOOP_MAX_ITERATIONS"] == 10

    def test_reload_all(self, tmp_path):
        """reload_all() updates the global settings object."""
        sf = tmp_path / "settings.json"
        sf.write_text(json.dumps({"LOOP_MAX_ITERATIONS": 99}))
        original = settings.LOOP_MAX_ITERATIONS
        with patch("infinidev.config.settings.SETTINGS_FILE", sf):
            reload_all()
        assert settings.LOOP_MAX_ITERATIONS == 99
        settings.LOOP_MAX_ITERATIONS = original


# ── LLM Config ───────────────────────────────────────────────────────────────


class TestLLMConfig:
    """Tests for LLM configuration."""

    def test_get_litellm_params_basic(self):
        """Returns model, api_key, api_base for ollama."""
        from infinidev.config.llm import get_litellm_params
        params = get_litellm_params()
        assert "model" in params
        assert params["model"]  # Not empty

    def test_auto_correct_ollama_prefix(self):
        """ollama/ prefix is corrected to ollama_chat/."""
        orig = settings.LLM_MODEL
        settings.LLM_MODEL = "ollama/test-model"
        try:
            from infinidev.config.llm import get_litellm_params
            params = get_litellm_params()
            assert params["model"].startswith("ollama_chat/")
        finally:
            settings.LLM_MODEL = orig

    def test_empty_model_raises(self):
        """Empty LLM_MODEL raises RuntimeError."""
        orig = settings.LLM_MODEL
        settings.LLM_MODEL = ""
        try:
            from infinidev.config.llm import get_litellm_params
            with pytest.raises(RuntimeError):
                get_litellm_params()
        finally:
            settings.LLM_MODEL = orig


# ── Model Capabilities ──────────────────────────────────────────────────────


class TestModelCapabilities:
    """Tests for model capability detection."""

    def test_default_capabilities(self):
        """Unprobed defaults have probed=False."""
        from infinidev.config.model_capabilities import ModelCapabilities
        caps = ModelCapabilities()
        assert caps.probed is False
        assert caps.supports_function_calling is True  # optimistic default

    @patch("litellm.completion")
    def test_probe_fc_success(self, mock_completion):
        """Model that returns tool_calls has FC support."""
        from infinidev.config.model_capabilities import probe_model
        mock_tc = MagicMock()
        mock_tc.function.name = "test_tool"
        mock_tc.function.arguments = "{}"
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                tool_calls=[mock_tc], content=None
            ))]
        )
        caps = probe_model({"model": "test"})
        assert caps.supports_function_calling is True
        assert caps.probed is True

    @patch("litellm.completion")
    def test_probe_fc_failure(self, mock_completion):
        """Model that never returns tool_calls has no FC."""
        from infinidev.config.model_capabilities import probe_model
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                tool_calls=None, content="I cannot use tools"
            ))]
        )
        caps = probe_model({"model": "test"})
        assert caps.supports_function_calling is False
        assert caps.probed is True

    @patch("litellm.completion")
    def test_probe_exception_uses_defaults(self, mock_completion):
        """Per-probe exceptions set conservative negatives, not raw defaults."""
        from infinidev.config.model_capabilities import probe_model
        mock_completion.side_effect = Exception("Connection refused")
        caps = probe_model({"model": "test"})
        assert caps.probed is True
        # When every probe call raises, the safer assumption is "no FC"
        # — emitting tools to a model that can't handle them produces
        # worse failures than running in manual mode.
        assert caps.supports_function_calling is False
        assert caps.supports_tool_choice_required is False

    def test_reset_capabilities(self):
        """_reset_capabilities restores defaults, then auto-detection re-probes."""
        import infinidev.config.model_capabilities as mc
        mc._reset_capabilities()
        # After reset, the module-level singleton is unprobed
        assert mc._capabilities.probed is False
        # But get_model_capabilities() triggers auto-detection (for ollama),
        # so the returned object is always probed
        caps = mc.get_model_capabilities()
        assert caps.probed is True



# ── Provider Model Discovery ─────────────────────────────────────────────────


class TestProviderModelDiscovery:
    def test_openai_codex_fetches_known_presets_with_remote_first(self):
        from infinidev.config.providers import fetch_models

        response = MagicMock()
        response.json.return_value = {
            "models": [
                {"slug": "hidden-model", "visibility": "hide", "priority": 0},
                {"slug": "gpt-later", "visibility": "list", "priority": 20},
                {"slug": "gpt-first", "visibility": "list", "priority": 1},
            ]
        }
        response.raise_for_status.return_value = None

        with patch("infinidev.config.codex_subscription.codex_oauth_headers", return_value={"Authorization": "Bearer token"}), \
             patch("infinidev.config.codex_subscription.httpx.get", return_value=response) as mock_get:
            models = fetch_models("openai_codex", raise_on_error=True)

        assert models[:2] == ["openai_codex/gpt-first", "openai_codex/gpt-later"]
        assert "openai_codex/gpt-5.5" in models
        assert "openai_codex/gpt-5.2-medium" in models
        mock_get.assert_called_once()
        url = mock_get.call_args.args[0]
        kwargs = mock_get.call_args.kwargs
        assert url == "https://chatgpt.com/backend-api/codex/models"
        assert kwargs["params"]["client_version"] == "0.6.0"
        assert kwargs["headers"]["Authorization"] == "Bearer token"

    def test_openai_codex_model_fetch_falls_back_to_known_presets_on_error(self):
        from infinidev.config.providers import fetch_models

        with patch("infinidev.config.codex_subscription.codex_oauth_headers", side_effect=RuntimeError("not logged in")):
            models = fetch_models("openai_codex")

        assert "openai_codex/gpt-5.5" in models
        assert "openai_codex/gpt-5.2-codex-xhigh" in models



# ── Codex OAuth ───────────────────────────────────────────────────────────────


class TestCodexOAuth:
    def test_complete_codex_oauth_reports_missing_pending_flow(self, tmp_path):
        from infinidev.config.openai_auth import complete_codex_oauth_flow

        with patch.dict(os.environ, {"INFINIDEV_CODEX_HOME": str(tmp_path / "codex")}), \
             pytest.raises(RuntimeError, match="No pending OAuth login found"):
            complete_codex_oauth_flow("http://localhost:1455/auth/callback?code=abc&state=state")


    def test_codex_oauth_headers_include_required_account_id_header(self):
        from pathlib import Path
        from infinidev.config.openai_auth import CodexOAuthToken, codex_oauth_headers

        token = CodexOAuthToken(
            access_token="access-token",
            refresh_token="refresh-token",
            account_id="account-123",
            plan_type=None,
            is_fedramp_account=False,
            source=Path("auth.json"),
        )
        with patch("infinidev.config.openai_auth.load_codex_oauth_token", return_value=token):
            headers = codex_oauth_headers()

        assert headers["Authorization"] == "Bearer access-token"
        assert headers["chatgpt-account-id"] == "account-123"
        assert "ChatGPT-Account-ID" not in headers

    def test_codex_oauth_headers_require_account_id(self):
        from pathlib import Path
        from infinidev.config.openai_auth import CodexOAuthToken, codex_oauth_headers

        token = CodexOAuthToken(
            access_token="access-token",
            refresh_token="refresh-token",
            account_id=None,
            plan_type=None,
            is_fedramp_account=False,
            source=Path("auth.json"),
        )
        with patch("infinidev.config.openai_auth.load_codex_oauth_token", return_value=token), \
             pytest.raises(RuntimeError, match="chatgpt_account_id"):
            codex_oauth_headers()



class TestCodexSubscriptionAdapter:
    def test_normalizes_modern_presets_and_reasoning_effort(self):
        from infinidev.config.codex_subscription import _build_request

        request = _build_request(
            {"model": "openai_codex/gpt-5.5-high"},
            [{"role": "user", "content": "hi"}],
            [],
            "auto",
        )
        assert request["model"] == "gpt-5.5"
        assert request["reasoning"]["effort"] == "high"

        request = _build_request(
            {"model": "openai_codex/gpt-5.2-codex-xhigh"},
            [{"role": "user", "content": "hi"}],
            [],
            "auto",
        )
        assert request["model"] == "gpt-5.2-codex"
        assert request["reasoning"]["effort"] == "xhigh"

        request = _build_request(
            {"model": "openai_codex/codex-mini-latest"},
            [{"role": "user", "content": "hi"}],
            [],
            "auto",
        )
        assert request["model"] == "gpt-5.1-codex-mini"
        assert request["reasoning"]["effort"] == "medium"

    def test_global_litellm_wrapper_routes_openai_codex(self):
        import litellm
        import infinidev.config.llm  # noqa: F401 - installs wrapper

        fake_response = MagicMock()
        with patch("infinidev.config.codex_subscription.completion", return_value=fake_response) as mock_codex:
            response = litellm.completion(
                model="openai_codex/gpt-5.5-high",
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "respond"}}],
                tool_choice="auto",
            )

        assert response is fake_response
        kwargs = mock_codex.call_args.kwargs
        assert kwargs["params"]["model"] == "openai_codex/gpt-5.5-high"
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
        assert kwargs["tools"]

    def test_global_litellm_wrapper_routes_positional_openai_codex_model(self):
        import litellm
        import infinidev.config.llm  # noqa: F401 - installs wrapper

        fake_response = MagicMock()
        with patch("infinidev.config.codex_subscription.completion", return_value=fake_response) as mock_codex:
            response = litellm.completion(
                "openai_codex/gpt-5.5-xhigh",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert response is fake_response
        assert mock_codex.call_args.kwargs["params"]["model"] == "openai_codex/gpt-5.5-xhigh"

    def test_fetch_remote_model_slugs_raises_for_oauth_validation(self):
        from infinidev.config.codex_subscription import fetch_remote_model_slugs

        response = MagicMock()
        response.raise_for_status.side_effect = RuntimeError("unauthorized")
        with patch("infinidev.config.codex_subscription.codex_oauth_headers", return_value={}), \
             patch("infinidev.config.codex_subscription.httpx.get", return_value=response), \
             pytest.raises(RuntimeError, match="unauthorized"):
            fetch_remote_model_slugs()

    def test_codex_request_omits_unsupported_temperature(self):
        from infinidev.config.codex_subscription import _build_request

        request = _build_request(
            {"model": "openai_codex/gpt 5.5 high", "temperature": 0.1},
            [{"role": "user", "content": "hi"}],
            [],
            "auto",
        )

        assert request["model"] == "gpt-5.5"
        assert request["reasoning"]["effort"] == "high"
        assert "temperature" not in request

    def test_codex_completion_reports_backend_error_body(self):
        from infinidev.config.codex_subscription import completion

        class FakeResponse:
            status_code = 400
            reason_phrase = "Bad Request"

            def read(self):
                return b'{"detail":"Unsupported parameter: temperature"}'

        class FakeStream:
            def __enter__(self):
                return FakeResponse()

            def __exit__(self, *args):
                return False

        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        client.stream.return_value = FakeStream()
        with patch("infinidev.config.codex_subscription.codex_oauth_headers", return_value={}), \
             patch("infinidev.config.codex_subscription.httpx.Client", return_value=client), \
             pytest.raises(RuntimeError, match="Unsupported parameter: temperature"):
            completion(
                params={"model": "openai_codex/gpt-5.5-medium"},
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_codex_completion_uses_configured_base_url(self):
        from infinidev.config.codex_subscription import completion

        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                return None

            def iter_lines(self):
                yield 'data: {"type":"response.completed","response":{"output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}]}}'
                yield ''

        class FakeStream:
            def __enter__(self):
                return FakeResponse()

            def __exit__(self, *args):
                return False

        client = MagicMock()
        client.__enter__.return_value = client
        client.__exit__.return_value = False
        client.stream.return_value = FakeStream()
        with patch("infinidev.config.codex_subscription.codex_oauth_headers", return_value={}), \
             patch("infinidev.config.codex_subscription.httpx.Client", return_value=client):
            response = completion(
                params={"model": "openai_codex/gpt-5.5-medium", "api_base": "https://example.test/codex"},
                messages=[{"role": "user", "content": "hi"}],
            )

        assert client.stream.call_args.args[:2] == ("POST", "https://example.test/codex/responses")
        assert response.choices[0].message.content == "ok"

    def test_response_parser_reads_codex_output_item_done_tool_call(self):
        from infinidev.config.codex_subscription import _response_from_events

        response = _response_from_events([
            {"type": "response.output_item.added", "item": {
                "type": "function_call",
                "id": "fc_123",
                "call_id": "call_123",
                "name": "respond",
                "arguments": "",
                "status": "in_progress",
            }},
            {"type": "response.function_call_arguments.delta", "item_id": "fc_123", "delta": '{"message":"'},
            {"type": "response.function_call_arguments.delta", "item_id": "fc_123", "delta": 'Hola'},
            {"type": "response.function_call_arguments.delta", "item_id": "fc_123", "delta": '"}'},
            {"type": "response.function_call_arguments.done", "item_id": "fc_123", "arguments": '{"message":"Hola"}'},
            {"type": "response.output_item.done", "item": {
                "type": "function_call",
                "id": "fc_123",
                "call_id": "call_123",
                "name": "respond",
                "arguments": '{"message":"Hola"}',
                "status": "completed",
            }},
            {"type": "response.completed", "response": {"id": "resp_123", "output": []}},
        ], "gpt-5.5")

        message = response.choices[0].message
        assert message.content is None
        assert message.tool_calls[0].id == "call_123"
        assert message.tool_calls[0].function.name == "respond"
        assert message.tool_calls[0].function.arguments == '{"message":"Hola"}'

    def test_response_parser_reads_codex_output_text_deltas_without_final_output(self):
        from infinidev.config.codex_subscription import _response_from_events

        response = _response_from_events([
            {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "Ho"},
            {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "la"},
            {"type": "response.completed", "response": {"id": "resp_123", "output": []}},
        ], "gpt-5.5")

        assert response.choices[0].message.content == "Hola"

    def test_response_parser_does_not_duplicate_final_message_and_deltas(self):
        from infinidev.config.codex_subscription import _response_from_events

        response = _response_from_events([
            {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "Hola"},
            {"type": "response.output_item.done", "item": {
                "type": "message",
                "id": "msg_1",
                "content": [{"type": "output_text", "text": "Hola"}],
            }},
            {"type": "response.completed", "response": {"id": "resp_123", "output": []}},
        ], "gpt-5.5")

        assert response.choices[0].message.content == "Hola"

    def test_global_litellm_wrapper_routes_streaming_openai_codex(self):
        import litellm
        import infinidev.config.llm  # noqa: F401 - installs wrapper
        from litellm.utils import Choices, Message, ModelResponse

        fake_response = ModelResponse(
            choices=[Choices(index=0, message=Message(role="assistant", content="hi"))],
            model="gpt-5.5",
        )
        with patch("infinidev.config.codex_subscription.completion", return_value=fake_response):
            stream = litellm.completion(
                model="openai_codex/gpt-5.5-high",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
            chunks = list(stream)

        assert chunks[0].choices[0].delta.content == "hi"
