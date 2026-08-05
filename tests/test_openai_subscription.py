"""Tests for the ChatGPT (Codex) subscription provider.

Everything here runs against a throwaway ``CODEX_HOME``: the module reads
that env var to find ``auth.json``, so no test can reach — let alone
rewrite — the developer's real credentials.
"""

from __future__ import annotations

import base64
import json
import os
import stat
import time

import pytest

from infinidev.config import codex_catalog, openai_oauth
from infinidev.config.openai_oauth import CodexAuthError


# ── Helpers ──────────────────────────────────────────────────────────


def _jwt(claims: dict) -> str:
    """A syntactically valid unsigned JWT carrying *claims*."""

    def seg(obj: dict) -> str:
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{seg({'alg': 'none'})}.{seg(claims)}.signature"


def _access_token(*, expires_in: float = 3600, account: str = "acct-123") -> str:
    return _jwt(
        {
            "exp": time.time() + expires_in,
            "https://api.openai.com/auth": {
                "chatgpt_account_id": account,
                "chatgpt_plan_type": "pro",
            },
        }
    )


def _write_auth(home, **overrides) -> "os.PathLike[str]":
    data = {
        "auth_mode": "chatgpt",
        "OPENAI_API_KEY": None,
        "tokens": {
            "id_token": _jwt({"email": "dev@example.com"}),
            "access_token": _access_token(),
            "refresh_token": "refresh-original",
            "account_id": "acct-123",
        },
        "last_refresh": "2026-01-01T00:00:00.000000000Z",
    }
    data.update(overrides)
    path = home / "auth.json"
    path.write_text(json.dumps(data))
    os.chmod(path, 0o600)
    return path


@pytest.fixture
def codex_home(tmp_path, monkeypatch):
    """An isolated CODEX_HOME with the catalog cache reset."""
    home = tmp_path / "codex"
    home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(home))
    # The catalog memoises on mtime; a fresh tmp dir would otherwise inherit
    # whatever a previous test (or the real machine) left parsed.
    monkeypatch.setattr(codex_catalog, "_cache", None)
    return home


# ── Reading credentials ──────────────────────────────────────────────


def test_missing_auth_file_names_the_fix(codex_home):
    with pytest.raises(CodexAuthError) as exc:
        openai_oauth.load_credentials()
    # A credential error that doesn't say what to run is a dead end.
    assert "codex login" in str(exc.value)
    assert str(codex_home) in str(exc.value)


def test_api_key_mode_is_rejected(codex_home):
    _write_auth(codex_home, auth_mode="apikey")
    with pytest.raises(CodexAuthError) as exc:
        openai_oauth.load_credentials()
    assert "API key" in str(exc.value)


def test_claims_come_from_the_token_not_the_file(codex_home):
    # The file's account_id can go stale after an account switch; the claim
    # inside the token is the one the backend will honour.
    _write_auth(
        codex_home,
        tokens={
            "access_token": _access_token(account="acct-from-token"),
            "refresh_token": "r",
            "id_token": "",
            "account_id": "acct-stale-in-file",
        },
    )
    creds = openai_oauth.load_credentials()
    assert creds.account_id == "acct-from-token"
    assert creds.plan_type == "pro"
    assert not creds.needs_refresh()


def test_credentials_without_refresh_token_are_unusable(codex_home):
    _write_auth(codex_home, tokens={"access_token": _access_token(), "refresh_token": ""})
    with pytest.raises(CodexAuthError):
        openai_oauth.load_credentials()


def test_is_configured_never_raises(codex_home):
    assert openai_oauth.is_configured() is False
    _write_auth(codex_home)
    assert openai_oauth.is_configured() is True


# ── Refreshing ───────────────────────────────────────────────────────


class _Response:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def _stub_post(monkeypatch, response, calls=None):
    import httpx

    def fake_post(url, **kwargs):
        if calls is not None:
            calls.append((url, kwargs))
        return response

    monkeypatch.setattr(httpx, "post", fake_post)


def test_valid_token_is_used_without_a_network_call(codex_home, monkeypatch):
    _write_auth(codex_home)
    import httpx

    def explode(*a, **k):  # pragma: no cover - must not run
        raise AssertionError("refreshed a token that was still valid")

    monkeypatch.setattr(httpx, "post", explode)
    assert openai_oauth.resolve().access_token


def test_expired_token_triggers_a_refresh(codex_home, monkeypatch):
    _write_auth(
        codex_home,
        tokens={
            "access_token": _access_token(expires_in=-100),
            "refresh_token": "refresh-original",
            "id_token": "",
            "account_id": "acct-123",
        },
    )
    calls: list = []
    _stub_post(
        monkeypatch,
        _Response(payload={"access_token": _access_token(), "refresh_token": "refresh-NEW"}),
        calls,
    )

    creds = openai_oauth.resolve()

    assert creds.expires_in() > 0
    (url, kwargs) = calls[0]
    assert url == "https://auth.openai.com/oauth/token"
    assert kwargs["json"]["grant_type"] == "refresh_token"
    assert kwargs["json"]["client_id"] == openai_oauth.CLIENT_ID
    assert kwargs["json"]["refresh_token"] == "refresh-original"


def test_a_rotated_refresh_token_is_written_back(codex_home, monkeypatch):
    """The whole reason this module writes to another tool's file.

    OpenAI retires the old refresh token when it issues a new one. Keeping
    the replacement only in memory would leave a dead token in auth.json and
    silently log the user out of the Codex CLI.
    """
    path = _write_auth(
        codex_home,
        tokens={
            "access_token": _access_token(expires_in=-1),
            "refresh_token": "refresh-original",
            "id_token": "",
            "account_id": "acct-123",
        },
    )
    _stub_post(
        monkeypatch,
        _Response(payload={"access_token": _access_token(), "refresh_token": "refresh-NEW"}),
    )

    openai_oauth.resolve()

    on_disk = json.loads(path.read_text())
    assert on_disk["tokens"]["refresh_token"] == "refresh-NEW"
    assert on_disk["auth_mode"] == "chatgpt"
    assert on_disk["last_refresh"] != "2026-01-01T00:00:00.000000000Z"


def test_refresh_preserves_unrelated_fields_and_permissions(codex_home, monkeypatch):
    path = _write_auth(
        codex_home,
        some_future_field={"kept": True},
        tokens={
            "access_token": _access_token(expires_in=-1),
            "refresh_token": "r",
            "id_token": "",
            "account_id": "acct-123",
        },
    )
    _stub_post(monkeypatch, _Response(payload={"access_token": _access_token()}))

    openai_oauth.resolve()

    on_disk = json.loads(path.read_text())
    assert on_disk["some_future_field"] == {"kept": True}
    # A credential must never land world-readable, not even briefly.
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_refresh_keeps_the_old_refresh_token_when_none_is_returned(codex_home, monkeypatch):
    path = _write_auth(
        codex_home,
        tokens={
            "access_token": _access_token(expires_in=-1),
            "refresh_token": "refresh-original",
            "id_token": "",
            "account_id": "acct-123",
        },
    )
    _stub_post(monkeypatch, _Response(payload={"access_token": _access_token()}))

    openai_oauth.resolve()

    assert json.loads(path.read_text())["tokens"]["refresh_token"] == "refresh-original"


def test_rejected_refresh_tells_the_user_to_log_in(codex_home, monkeypatch):
    _write_auth(
        codex_home,
        tokens={
            "access_token": _access_token(expires_in=-1),
            "refresh_token": "revoked",
            "id_token": "",
            "account_id": "a",
        },
    )
    _stub_post(monkeypatch, _Response(status_code=400, text='{"error":"invalid_grant"}'))

    with pytest.raises(CodexAuthError) as exc:
        openai_oauth.resolve()
    assert "codex login" in str(exc.value)
    assert "invalid_grant" in str(exc.value)


def test_token_url_override_is_honoured(codex_home, monkeypatch):
    monkeypatch.setenv("CODEX_REFRESH_TOKEN_URL_OVERRIDE", "https://gateway.internal/token")
    _write_auth(
        codex_home,
        tokens={
            "access_token": _access_token(expires_in=-1),
            "refresh_token": "r",
            "id_token": "",
            "account_id": "a",
        },
    )
    calls: list = []
    _stub_post(monkeypatch, _Response(payload={"access_token": _access_token()}), calls)

    openai_oauth.resolve()

    assert calls[0][0] == "https://gateway.internal/token"


def test_status_reports_without_leaking_tokens(codex_home):
    _write_auth(codex_home)
    info = openai_oauth.status()
    assert info["configured"] is True
    assert info["plan"] == "pro"
    assert info["expired"] is False
    assert "token" not in json.dumps(info).lower().replace("access_token", "")


# ── Request headers ──────────────────────────────────────────────────


def test_request_headers_carry_account_and_originator(codex_home):
    headers = openai_oauth.request_headers(account_id="acct-9")
    assert headers["ChatGPT-Account-Id"] == "acct-9"
    assert headers["originator"] == "codex_cli_rs"
    assert headers["session_id"] == openai_oauth.session_id()  # stable per process


def test_originator_is_overridable(codex_home, monkeypatch):
    monkeypatch.setenv("INFINIDEV_CODEX_ORIGINATOR", "custom_client")
    assert openai_oauth.request_headers()["originator"] == "custom_client"


# ── Model catalog ────────────────────────────────────────────────────


def _write_catalog(home, models):
    (home / "models_cache.json").write_text(json.dumps({"models": models}))


def test_catalog_lists_visible_models_by_priority(codex_home):
    """`priority` ranks ascending — it is a position, not a score.

    The real catalog ships gpt-5.5 at 9 and gpt-5.2 at 29, so reading it the
    intuitive way (highest wins) puts the weakest model first in the picker.
    """
    _write_catalog(
        codex_home,
        [
            {"slug": "third", "priority": 29, "visibility": "list"},
            {"slug": "hidden", "priority": 1, "visibility": "hide"},
            {"slug": "first", "priority": 9, "visibility": "list"},
            {"slug": "second", "priority": 16, "visibility": "list"},
        ],
    )
    models = codex_catalog.list_models()
    assert models[:3] == ["first", "second", "third"]
    assert "hidden" not in models


def test_unranked_models_sort_last(codex_home):
    _write_catalog(
        codex_home,
        [
            {"slug": "unranked", "visibility": "list"},
            {"slug": "ranked", "priority": 50, "visibility": "list"},
        ],
    )
    assert codex_catalog.list_models()[:2] == ["ranked", "unranked"]


def test_context_window_applies_the_effective_percentage(codex_home):
    # The remaining 5% is the model's output budget; counting it as input is
    # how a prompt overflows at the last moment.
    _write_catalog(
        codex_home,
        [{"slug": "m", "context_window": 272_000, "effective_context_window_percent": 95}],
    )
    assert codex_catalog.context_window("m") == 258_400


def test_context_window_without_a_percentage_uses_the_raw_window(codex_home):
    _write_catalog(codex_home, [{"slug": "m", "context_window": 128_000}])
    assert codex_catalog.context_window("m") == 128_000


def test_catalog_falls_back_when_the_cache_is_absent(codex_home):
    assert "gpt-5.5" in codex_catalog.list_models()
    # Same 95 % output reserve as a real entry — the fallback should not
    # report a roomier window than the catalog would.
    assert codex_catalog.context_window("gpt-5.5") == 258_400


def test_a_loaded_catalog_is_the_whole_list(codex_home):
    """The fallback used to be appended to a catalog that had loaded fine.

    That only pays if the hardcoded tuple is fresher than a cache the CLI
    refreshes itself, and it never is. What it produced in practice was the
    reverse: the catalog carried the three real 5.6 slugs and the fallback
    added ``gpt-5.6``, which the backend rejects outright. The settings
    dialog renders this list as a *closed* dropdown, so every such entry is
    a selectable trap rather than a harmless suggestion.
    """
    _write_catalog(
        codex_home,
        [{"slug": "gpt-5.5", "context_window": 272_000, "visibility": "list", "priority": 9}],
    )
    assert codex_catalog.list_models() == ["gpt-5.5"]


def test_a_model_newer_than_the_cache_still_gets_a_context_window(codex_home):
    """Tolerating an unknown slug and *offering* one are different things.

    The cache lags real releases, so a user who types a model newer than it
    must still get a usable context budget — an unknown window would disable
    the loop's accounting for a model that works perfectly well. That is the
    half of the old behaviour worth keeping.
    """
    _write_catalog(
        codex_home,
        [{"slug": "gpt-5.5", "context_window": 272_000, "visibility": "list", "priority": 9}],
    )
    assert codex_catalog.context_window("gpt-5.7-unreleased") == 258_400


def test_the_offline_fallback_names_only_real_slugs():
    """A floor made of slugs the backend rejects is not a floor.

    The 5.6 generation ships as named variants; the bare family name earns
    "not supported when using codex with ChatGPT account".
    """
    assert "gpt-5.6" not in codex_catalog._FALLBACK_MODELS
    assert "gpt-5.6-sol" in codex_catalog._FALLBACK_MODELS


def test_catalog_entries_are_not_duplicated_by_the_fallback(codex_home):
    _write_catalog(codex_home, [{"slug": "gpt-5.5", "visibility": "list", "priority": 9}])
    models = codex_catalog.list_models()
    assert models.count("gpt-5.5") == 1


def test_hidden_catalog_entries_are_not_resurrected_by_the_fallback(codex_home):
    _write_catalog(codex_home, [{"slug": "gpt-5.5", "visibility": "hide"}])
    assert "gpt-5.5" not in codex_catalog.list_models()


def test_reasoning_levels_are_read_from_the_catalog(codex_home):
    _write_catalog(
        codex_home,
        [{"slug": "m", "supported_reasoning_levels": [{"effort": "low"}, {"effort": "xhigh"}]}],
    )
    assert codex_catalog.reasoning_levels("m") == ["low", "xhigh"]


@pytest.mark.parametrize(
    ("capability_fields", "expected"),
    [
        ({"supports_image_generation": True}, True),
        ({"supports_image_generation": False}, False),
        ({}, None),
        ({"supports_image_generation": "yes"}, None),
    ],
)
def test_image_generation_announcement_is_strictly_tristate(
    codex_home, capability_fields, expected
):
    _write_catalog(
        codex_home,
        [{"slug": "gpt-5.6-sol", **capability_fields}],
    )

    assert codex_catalog.supports_image_generation("gpt-5.6-sol") is expected


def test_announced_gpt_56_sol_without_a_generation_route_does_not_register_tool(
    codex_home, monkeypatch
):
    import infinidev.config.model_capabilities as capabilities
    import infinidev.tools as tools

    _write_catalog(
        codex_home,
        [{"slug": "gpt-5.6-sol", "supports_image_generation": True}],
    )
    monkeypatch.setattr(capabilities, "_generation_route_from_settings", lambda: None)
    resolver = capabilities.CapabilityResolver(
        metadata_detector=lambda route: capabilities.CapabilityAssessment(
            status=capabilities.CapabilityStatus.SUPPORTED
        ),
        local_detector=lambda route: capabilities.CapabilityAssessment(),
    )
    snapshot = resolver.resolve(
        capabilities.ModelRoute(
            "openai_subscription", "openai/responses/gpt-5.6-sol"
        )
    )

    assert snapshot.image_generation.status is capabilities.CapabilityStatus.UNSUPPORTED
    assert snapshot.generation_profile is None
    assert snapshot.generation_route is None
    assert snapshot.image_input.status is capabilities.CapabilityStatus.SUPPORTED

    monkeypatch.setattr(capabilities, "get_capability_snapshot", lambda: snapshot)
    monkeypatch.setattr(tools, "discover_mcp_tool_classes", lambda: [])
    names = {tool.name for tool in tools.get_tools_for_role("developer")}
    assert "generate_image" not in names


@pytest.mark.parametrize(
    "capability_fields",
    [
        {"supports_image_generation": False},
        {},
        {"supports_image_generation": "yes"},
    ],
)
def test_gpt_56_sol_generation_is_not_registered_without_affirmative_announcement(
    codex_home, monkeypatch, capability_fields
):
    import infinidev.config.model_capabilities as capabilities
    import infinidev.tools as tools

    _write_catalog(
        codex_home,
        [{"slug": "gpt-5.6-sol", **capability_fields}],
    )
    monkeypatch.setattr(capabilities, "_generation_route_from_settings", lambda: None)
    snapshot = capabilities.CapabilityResolver(
        metadata_detector=lambda route: capabilities.CapabilityAssessment(),
        local_detector=lambda route: capabilities.CapabilityAssessment(),
    ).resolve(
        capabilities.ModelRoute(
            "openai_subscription", "openai/responses/gpt-5.6-sol"
        )
    )

    assert snapshot.image_generation.supported is False
    assert snapshot.generation_profile is None

    monkeypatch.setattr(capabilities, "get_capability_snapshot", lambda: snapshot)
    monkeypatch.setattr(tools, "discover_mcp_tool_classes", lambda: [])
    names = {tool.name for tool in tools.get_tools_for_role("developer")}
    assert "generate_image" not in names


def test_gpt_56_sol_announcement_does_not_enable_other_routes(codex_home, monkeypatch):
    import infinidev.config.model_capabilities as capabilities

    _write_catalog(
        codex_home,
        [{"slug": "gpt-5.6-sol", "supports_image_generation": True}],
    )
    monkeypatch.setattr(capabilities, "_generation_route_from_settings", lambda: None)
    resolver = capabilities.CapabilityResolver(
        metadata_detector=lambda route: capabilities.CapabilityAssessment(),
        local_detector=lambda route: capabilities.CapabilityAssessment(),
    )

    routes = (
        (
            capabilities.ModelRoute("openai", "openai/responses/gpt-5.6-sol"),
            capabilities.CapabilityStatus.UNKNOWN,
        ),
        (
            capabilities.ModelRoute(
                "openai_subscription", "openai/responses/gpt-5.6-terra"
            ),
            capabilities.CapabilityStatus.UNSUPPORTED,
        ),
    )
    for route, expected_status in routes:
        snapshot = resolver.resolve(route)
        assert snapshot.image_generation.status is expected_status
        assert snapshot.generation_profile is None
        assert snapshot.generation_route is None


# ── Wiring into LiteLLM params ───────────────────────────────────────


@pytest.fixture
def subscription_settings(codex_home, monkeypatch):
    from infinidev.config.settings import settings

    _write_auth(codex_home)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai_subscription")
    monkeypatch.setattr(settings, "LLM_MODEL", "gpt-5.5")
    # Deliberately wrong leftovers from a previous provider — the subscription
    # must override both rather than inherit them.
    monkeypatch.setattr(settings, "LLM_BASE_URL", "http://localhost:11434")
    monkeypatch.setattr(settings, "LLM_API_KEY", "ollama")
    return settings


def test_params_target_the_codex_backend(subscription_settings):
    from infinidev.config.llm import get_litellm_params

    params = get_litellm_params()

    assert params["model"] == "openai/responses/gpt-5.5"
    assert params["api_base"] == "https://chatgpt.com/backend-api/codex"
    # The OAuth token stands in for the api key — LiteLLM turns it into the
    # Authorization: Bearer header.
    assert params["api_key"].count(".") == 2
    assert params["api_key"] != "ollama"
    assert params["extra_headers"]["ChatGPT-Account-Id"] == "acct-123"
    assert params["extra_headers"]["originator"] == "codex_cli_rs"
    assert params["extra_body"]["store"] is False


def test_temperature_is_dropped_for_reasoning_models(subscription_settings):
    """GPT-5.x rejects an explicit temperature; the loop pins 0.2 by default."""
    from infinidev.config.llm import get_litellm_params

    subscription_settings.LLM_TEMPERATURE = 0.2
    assert "temperature" not in get_litellm_params()


def test_assistant_params_also_use_the_subscription(subscription_settings, monkeypatch):
    from infinidev.config.llm import get_litellm_params_for_assistant

    monkeypatch.setattr(subscription_settings, "ASSISTANT_LLM_PROVIDER", "")
    monkeypatch.setattr(subscription_settings, "ASSISTANT_LLM_MODEL", "")
    params = get_litellm_params_for_assistant()
    assert params["api_base"] == "https://chatgpt.com/backend-api/codex"
    assert params["model"] == "openai/responses/gpt-5.5"


def test_registered_provider_flag_beats_the_prefix_heuristic():
    """The bug this guards: `openai/...` models that must not reach OpenAI.

    ``_is_native_provider`` used to fall through to a prefix check whenever a
    provider's own flag was False. For the subscription that check sees
    ``openai/`` and returns True, api_base is dropped, and the request goes to
    api.openai.com — where an OAuth token is not a valid key.
    """
    from infinidev.config.llm import _is_native_provider_id

    assert _is_native_provider_id("openai_subscription", "openai/responses/gpt-5.5") is False
    assert _is_native_provider_id("openai", "openai/gpt-5.4") is True
    assert _is_native_provider_id("ollama", "ollama_chat/qwen3") is False
    # Unregistered ids keep the legacy heuristic.
    assert _is_native_provider_id("some_new_id", "anthropic/claude") is True


# ── Context window and reasoning ─────────────────────────────────────


def test_context_window_uses_the_subscription_catalog(codex_home):
    """litellm says gpt-5.5 takes 1 050 000 input tokens — on the API. The
    subscription serves 272 000, and believing the API number would hand the
    loop ~800 000 tokens of headroom that do not exist."""
    from infinidev.engine.loop.model_context import get_model_context_window

    _write_catalog(
        codex_home,
        [{"slug": "gpt-5.5", "context_window": 272_000, "effective_context_window_percent": 95}],
    )
    window = get_model_context_window(
        {"model": "openai/responses/gpt-5.5"}, "openai_subscription"
    )
    assert window == 258_400


def test_gpt_56_subscription_does_not_use_the_larger_api_window(codex_home):
    """The API model has 1.05M context, but Codex subscriptions are capped by
    their own catalog and must not inherit the metered API limit."""
    from infinidev.engine.loop.model_context import (
        get_model_context_window,
        get_model_max_context_window,
    )

    _write_catalog(
        codex_home,
        [
            {
                "slug": "gpt-5.6-sol",
                "context_window": 272_000,
                "effective_context_window_percent": 95,
            }
        ],
    )
    window = get_model_context_window(
        {"model": "openai/responses/gpt-5.6-sol"},
        "openai_subscription",
    )
    assert window == 258_400
    assert get_model_max_context_window(
        {"model": "openai/responses/gpt-5.6-sol"},
        "openai_subscription",
    ) == 1_050_000


def test_bare_model_strips_the_protocol_segment():
    from infinidev.engine.loop.model_context import _bare_model

    assert _bare_model("openai/responses/gpt-5.5") == "gpt-5.5"
    assert _bare_model("ollama_chat/qwen3:8b") == "qwen3:8b"
    assert _bare_model("anthropic/claude-opus-4-6") == "claude-opus-4-6"


def test_ultra_reaches_xhigh_when_the_model_supports_it(codex_home, monkeypatch):
    from infinidev.config import thinking_budget
    from infinidev.config.settings import settings

    _write_catalog(
        codex_home,
        [
            {
                "slug": "gpt-5.5",
                "supported_reasoning_levels": [
                    {"effort": "low"},
                    {"effort": "medium"},
                    {"effort": "high"},
                    {"effort": "xhigh"},
                ],
            }
        ],
    )
    monkeypatch.setattr(settings, "THINKING_ENABLED", True)
    monkeypatch.setattr(settings, "THINKING_BUDGET", "ultra")

    kwargs: dict = {}
    thinking_budget.apply_thinking_budget(
        kwargs, "openai_subscription", "openai/responses/gpt-5.5"
    )
    assert kwargs["reasoning_effort"] == "xhigh"


def test_unsupported_effort_falls_back_to_a_listed_one(codex_home, monkeypatch):
    from infinidev.config import thinking_budget
    from infinidev.config.settings import settings

    _write_catalog(
        codex_home,
        [{"slug": "small", "supported_reasoning_levels": [{"effort": "low"}]}],
    )
    monkeypatch.setattr(settings, "THINKING_ENABLED", True)
    monkeypatch.setattr(settings, "THINKING_BUDGET", "ultra")

    kwargs: dict = {}
    thinking_budget.apply_thinking_budget(
        kwargs, "openai_subscription", "openai/responses/small"
    )
    assert kwargs["reasoning_effort"] == "low"


def test_disabled_thinking_sends_the_lowest_effort(codex_home, monkeypatch):
    from infinidev.config import thinking_budget
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "THINKING_ENABLED", False)
    kwargs: dict = {}
    thinking_budget.apply_thinking_budget(
        kwargs, "openai_subscription", "openai/responses/gpt-5.5"
    )
    # Reasoning cannot be switched off on these models; "low" is as near as
    # the API gets, and no /no_think tag should be injected.
    assert kwargs["reasoning_effort"] == "low"
    assert "max_tokens" not in kwargs


# ── Provider registry ────────────────────────────────────────────────


def test_provider_is_registered_and_selectable():
    from infinidev.config.providers import get_provider, list_provider_ids

    assert "openai_subscription" in list_provider_ids()
    provider = get_provider("openai_subscription")
    assert provider.api_key_required is False
    assert provider.is_native is False
    assert provider.prefix == "openai/responses/"


def test_settings_dialog_offers_every_registered_provider():
    """The picker is derived from the registry, not a fourth copy of the list."""
    from infinidev.config.providers import list_provider_ids
    from infinidev.ui.dialogs.settings_editor_state import SETTINGS_SECTIONS

    field = next(f for f in SETTINGS_SECTIONS["LLM"] if f[0] == "LLM_PROVIDER")
    offered = field[2].removeprefix("select:").split(",")
    assert offered == list_provider_ids()


def test_model_discovery_needs_no_network(codex_home):
    from infinidev.config.providers import fetch_models

    _write_catalog(codex_home, [{"slug": "gpt-5.5", "visibility": "list", "priority": 9}])
    models = fetch_models("openai_subscription")
    assert models[0] == "openai/responses/gpt-5.5"
    assert all(m.startswith("openai/responses/") for m in models)


# ── Degrading without a login ────────────────────────────────────────


def test_status_line_survives_a_missing_login(codex_home, monkeypatch):
    """A credential error belongs to the first LLM call, not to startup.

    ``_get_initial_model_name`` runs at import time to seed the module-level
    calculator, so an exception there takes the TUI down before it draws.
    """
    from infinidev.config.settings import settings
    from infinidev.ui.context_calculator import (
        ContextWindowCalculator,
        _get_initial_model_name,
    )

    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai_subscription")
    monkeypatch.setattr(settings, "LLM_MODEL", "gpt-5.5")
    # No auth.json was written: resolving the token raises.

    assert _get_initial_model_name() == "gpt-5.5"

    calc = ContextWindowCalculator()
    calc.resolve_model_context()  # must not raise
    assert calc.max_context is None
    assert calc.model_name == "gpt-5.5"
    assert calc.get_context_status()["max_context"] is None


def test_capabilities_preset_exists():
    from infinidev.config.model_capabilities import _PROVIDER_PRESETS

    caps = _PROVIDER_PRESETS["openai_subscription"]
    assert caps.supports_function_calling is True
    assert caps.probed is True


def test_subscription_tool_request_omits_unsupported_tool_choice(monkeypatch):
    from infinidev.config.settings import settings
    from infinidev.engine.llm_client import call_llm

    seen = {}
    response = object()
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai_subscription")
    monkeypatch.setattr("litellm.completion", lambda **kwargs: seen.update(kwargs) or response)

    assert call_llm(
        {"model": "openai/responses/gpt-5.6-sol"},
        [{"role": "user", "content": "x"}],
        [{"type": "function", "function": {"name": "done", "parameters": {}}}],
        tool_choice="required",
        retry_attempts=1,
    ) is response

    assert "tools" in seen
    assert "tool_choice" not in seen


# ── The backend answers streams only ─────────────────────────────────
#
# Codex rejects every non-streaming request with HTTP 400 and
# {"detail":"Stream must be set to true"}, which cannot be reproduced
# without the network. What IS testable is the repair: which requests get
# rewritten, which are left alone, and that the rebuilt response carries
# the tool call the planner terminates on.


class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Chunk:
    def __init__(self, delta):
        self.choices = [type("C", (), {"delta": delta})()]


def test_a_codex_request_without_stream_is_rewritten():
    from infinidev.config.llm import _needs_forced_streaming

    assert _needs_forced_streaming(
        {"api_base": "https://chatgpt.com/backend-api/codex"}
    )
    # A trailing slash is the same backend.
    assert _needs_forced_streaming(
        {"api_base": "https://chatgpt.com/backend-api/codex/"}
    )


def test_a_real_stream_is_left_alone():
    """Rewriting a request that already streams would consume the generator."""
    from infinidev.config.llm import _needs_forced_streaming

    assert not _needs_forced_streaming(
        {"api_base": "https://chatgpt.com/backend-api/codex", "stream": True}
    )


def test_other_backends_are_untouched():
    from infinidev.config.llm import _needs_forced_streaming

    assert not _needs_forced_streaming({"api_base": "http://localhost:11434"})
    assert not _needs_forced_streaming({"api_base": "https://api.openai.com/v1"})
    assert not _needs_forced_streaming({})


def test_the_rebuilt_response_carries_the_tool_call(monkeypatch):
    """What the planner reads back must survive the round trip.

    ``planner.py`` terminates on ``tc.function.name == "emit_plan"`` and
    reports usage, so a rebuild that drops either one fails silently: the
    planner would fall back to a plan with no steps.
    """
    import litellm

    from infinidev.config.llm import _completion_via_forced_stream

    seen: dict = {}

    def _fake_original(*args, **kwargs):
        seen.update(kwargs)
        return iter([_Chunk(_Delta(content="ignored"))])

    class _Rebuilt:
        choices = [
            type("C", (), {
                "message": type("M", (), {
                    "content": "",
                    "tool_calls": [
                        type("TC", (), {
                            "id": "call_1",
                            "function": type("F", (), {
                                "name": "emit_plan",
                                "arguments": '{"overview":"x","steps":[]}',
                            })(),
                        })()
                    ],
                })(),
            })()
        ]
        usage = type("U", (), {"prompt_tokens": 77, "completion_tokens": 148})()

    monkeypatch.setattr(litellm, "stream_chunk_builder", lambda *a, **k: _Rebuilt)

    messages = [{"role": "user", "content": "plan it"}]
    result = _completion_via_forced_stream(
        _fake_original, (), {"messages": messages, "stream": False},
    )

    assert seen["stream"] is True, "the backend was asked for a stream"
    call = result.choices[0].message.tool_calls[0]
    assert call.function.name == "emit_plan"
    assert json.loads(call.function.arguments)["overview"] == "x"
    assert result.usage.prompt_tokens == 77


def test_the_callers_kwargs_are_not_mutated(monkeypatch):
    """The caller asked for a whole response and must still believe it did.

    ``planner.py`` uses ``setdefault("stream", False)`` on a dict it reuses
    across iterations; flipping that value under it would turn every later
    call into an unconsumed generator.
    """
    import litellm

    from infinidev.config.llm import _completion_via_forced_stream

    monkeypatch.setattr(litellm, "stream_chunk_builder", lambda *a, **k: "rebuilt")

    kwargs = {"messages": [], "stream": False}
    _completion_via_forced_stream(lambda *a, **k: iter([]), (), kwargs)

    assert kwargs["stream"] is False


def test_an_empty_stream_names_the_fix(monkeypatch):
    """A None rebuild must not reach the caller as an AttributeError."""
    import litellm

    from infinidev.config.llm import _completion_via_forced_stream

    monkeypatch.setattr(litellm, "stream_chunk_builder", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="rebuilt into no response"):
        _completion_via_forced_stream(lambda *a, **k: iter([]), (), {"messages": []})


# ── Parameters the backend refuses ───────────────────────────────────


def test_max_tokens_is_dropped_for_codex():
    """LiteLLM renders max_tokens as the Responses API's max_output_tokens.

    The Codex backend answers that with "Unsupported parameter:
    max_output_tokens", and nine call sites in the engine set it — the
    planner, both spec-elaborator passes, the council, the summarisers and
    the chat agent.
    """
    from infinidev.config.llm import _sanitized_for_codex

    cleaned = _sanitized_for_codex({"model": "m", "max_tokens": 3000, "messages": []})
    assert "max_tokens" not in cleaned
    assert cleaned["messages"] == []


def test_temperature_is_dropped_even_when_a_caller_puts_it_back():
    """_apply_chatgpt_subscription pops it; planner.py setdefaults it back.

    The builder runs once when params are assembled, the caller runs after,
    so only a check at the request boundary catches this one.
    """
    from infinidev.config.llm import _sanitized_for_codex

    assert "temperature" not in _sanitized_for_codex({"temperature": 0.1})


def test_sanitizing_does_not_edit_the_callers_dict():
    """Callers reuse one kwargs dict across loop iterations."""
    from infinidev.config.llm import _sanitized_for_codex

    original = {"max_tokens": 3000, "temperature": 0.2}
    _sanitized_for_codex(original)
    assert original == {"max_tokens": 3000, "temperature": 0.2}


def test_a_clean_request_is_passed_through_unchanged():
    from infinidev.config.llm import _sanitized_for_codex

    kwargs = {"model": "m", "messages": []}
    assert _sanitized_for_codex(kwargs) is kwargs


# ── Reasoning effort follows the model, not a hardcoded ladder ───────


def test_a_level_the_model_publishes_is_passed_through(codex_home):
    """This is what lets /effort max mean max instead of a rounded tier."""
    from infinidev.config.thinking_budget import _subscription_effort

    _write_catalog(
        codex_home,
        [{"slug": "m", "supported_reasoning_levels": [
            {"effort": "low"}, {"effort": "high"}, {"effort": "max"},
        ]}],
    )
    assert _subscription_effort("openai/responses/m", "max", 0) == "max"


def test_ultra_reaches_the_deepest_level_the_model_has(codex_home):
    """It used to hardcode xhigh — the one level LiteLLM refuses on 5.6.

    LiteLLM's own version check recognises "gpt-5.4+" but not the named 5.6
    variants, so ultra died before reaching a backend that accepts max and
    ultra happily.
    """
    from infinidev.config.thinking_budget import _subscription_effort

    _write_catalog(
        codex_home,
        [{"slug": "sol", "supported_reasoning_levels": [
            {"effort": "low"}, {"effort": "medium"}, {"effort": "high"},
            {"effort": "xhigh"}, {"effort": "max"}, {"effort": "ultra"},
        ]}],
    )
    assert _subscription_effort("openai/responses/sol", "ultra", 0) == "ultra"


def test_ultra_settles_for_the_top_of_a_shallower_model(codex_home):
    from infinidev.config.thinking_budget import _subscription_effort

    _write_catalog(
        codex_home,
        [{"slug": "small", "supported_reasoning_levels": [
            {"effort": "low"}, {"effort": "medium"}, {"effort": "high"},
        ]}],
    )
    assert _subscription_effort("openai/responses/small", "ultra", 0) == "high"


def test_an_unavailable_level_falls_back_to_medium(codex_home):
    from infinidev.config.thinking_budget import _subscription_effort

    _write_catalog(
        codex_home,
        [{"slug": "small", "supported_reasoning_levels": [
            {"effort": "low"}, {"effort": "medium"},
        ]}],
    )
    assert _subscription_effort("openai/responses/small", "high", 0) == "medium"


def test_efforts_are_offered_shallowest_first(codex_home):
    """The command lists these, so catalog order must not leak into the UI."""
    from infinidev.config.thinking_budget import subscription_efforts

    _write_catalog(
        codex_home,
        [{"slug": "sol", "supported_reasoning_levels": [
            {"effort": "max"}, {"effort": "low"}, {"effort": "ultra"},
            {"effort": "high"},
        ]}],
    )
    assert subscription_efforts("openai/responses/sol") == [
        "low", "high", "max", "ultra",
    ]


def test_efforts_are_empty_without_a_catalog(codex_home):
    """The command then offers the generic presets instead of guessing."""
    from infinidev.config.thinking_budget import subscription_efforts

    assert subscription_efforts("openai/responses/unknown") == []
