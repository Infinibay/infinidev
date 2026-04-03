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
        """Top-level exception falls back to defaults."""
        from infinidev.config.model_capabilities import probe_model
        mock_completion.side_effect = Exception("Connection refused")
        caps = probe_model({"model": "test"})
        assert caps.probed is True
        # Defaults to True (optimistic)
        assert caps.supports_function_calling is True

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
