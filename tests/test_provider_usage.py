"""Provider usage: quota contracts, observed requests, and missing-data behavior."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import httpx
import pytest


def test_stream_usage_counts_final_totals_once_and_preserves_chunks():
    from infinidev.engine.usage import UsageLedger

    ledger = UsageLedger()
    params = {"model": "openai/gpt-6-astra", "api_key": "test-key", "stream": True}
    chunks = [SimpleNamespace(usage=None), SimpleNamespace(usage={
        "prompt_tokens": 100, "completion_tokens": 20,
        "prompt_tokens_details": {"cached_tokens": 40},
    }, _hidden_params={"additional_headers": {"x-ratelimit-remaining-requests": "8"}})]
    assert list(ledger.observe(iter(chunks), params)) == chunks
    snapshot = ledger.snapshot(params)
    assert snapshot["requests"] == 1
    assert snapshot["input_tokens"] == 100
    assert snapshot["output_tokens"] == 20
    assert snapshot["cached_tokens"] == 40
    assert snapshot["headers"]["x-ratelimit-remaining-requests"] == "8"
    assert ledger.snapshot({**params, "api_key": "other-key"})["requests"] == 0


def test_missing_usage_is_not_reported_as_measured_zero():
    from infinidev.engine.usage import UsageLedger

    ledger = UsageLedger()
    params = {"model": "ollama_chat/qwen3:8b"}
    response = SimpleNamespace(usage=None)
    assert ledger.observe(response, params) is response
    assert ledger.snapshot(params)["missing_usage"] == 1


def test_native_stream_totals_are_read_after_sdk_finishes_updating_metadata():
    from infinidev.engine.usage import UsageLedger

    ledger = UsageLedger()
    params = {"model": "anthropic/claude-sonnet-4-6", "stream": True}
    hidden = {"usage": {"prompt_tokens": 12, "completion_tokens": 0}}

    def chunks():
        yield SimpleNamespace(usage=None, _hidden_params=hidden)
        hidden["usage"] = {"prompt_tokens": 12, "completion_tokens": 2}

    list(ledger.observe(chunks(), params))
    snapshot = ledger.snapshot(params)
    assert snapshot["requests"] == 1
    assert snapshot["input_tokens"] == 12
    assert snapshot["output_tokens"] == 2


@pytest.mark.parametrize("stream", [False, True])
def test_anthropic_usage_survives_sdk_and_matches_selected_connection(monkeypatch, stream):
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    from infinidev.config.llm import get_litellm_params
    from infinidev.config.settings import settings
    from infinidev.config.usage import UsageSelection
    from infinidev.engine import usage
    from infinidev.engine.llm_client import call_llm

    monkeypatch.setattr(settings, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(settings, "LLM_MODEL", "anthropic/claude-sonnet-4-6")
    monkeypatch.setattr(settings, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(settings, "LLM_BASE_URL", "https://api.anthropic.com")
    ledger = usage.UsageLedger()
    monkeypatch.setattr(usage, "usage_ledger", ledger)
    requests = []

    def post(self, url, **kwargs):
        requests.append(url)
        message = {"id": "msg_test", "type": "message", "role": "assistant",
                   "model": "claude-sonnet-4-6", "content": [{"type": "text", "text": "Checked."}],
                   "stop_reason": "end_turn", "stop_sequence": None,
                   "usage": {"input_tokens": 8, "output_tokens": 2,
                             "cache_read_input_tokens": 4, "cache_creation_input_tokens": 0}}
        headers = {"anthropic-ratelimit-requests-remaining": "19"}
        if kwargs.get("stream"):
            events = [
                {"type": "message_start", "message": {
                    **message, "content": [], "stop_reason": None,
                    "usage": {**message["usage"], "output_tokens": 0},
                }},
                {"type": "content_block_start", "index": 0,
                 "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0,
                 "delta": {"type": "text_delta", "text": "Checked."}},
                {"type": "content_block_stop", "index": 0},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                 "usage": {"output_tokens": 2}},
                {"type": "message_stop"},
            ]
            data = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
            return httpx.Response(200, request=httpx.Request("POST", url),
                                  headers={**headers, "content-type": "text/event-stream"},
                                  text=data)
        return httpx.Response(200, request=httpx.Request("POST", url),
                              json=message, headers=headers)

    monkeypatch.setattr(HTTPHandler, "post", post)
    response = call_llm(get_litellm_params(), [{"role": "user", "content": "Inspect the source"}],
                        retry_attempts=1, on_thinking_chunk=(lambda text: None) if stream else None)
    assert response.choices[0].message.content == "Checked."
    assert requests == ["https://api.anthropic.com/v1/messages"]
    snapshot = ledger.snapshot(UsageSelection.current().request_params())
    assert snapshot["requests"] == 1
    assert snapshot["input_tokens"] == 12
    assert snapshot["output_tokens"] == 2
    assert snapshot["cached_tokens"] == 4
    assert snapshot["headers"]["anthropic-ratelimit-requests-remaining"] == "19"


def test_codex_quota_uses_reported_windows_not_assumed_durations():
    from infinidev.config.usage import format_codex_limits

    result = format_codex_limits({"rateLimitsByLimitId": {"codex": {
        "primary": {"usedPercent": 25, "windowDurationMins": 300, "resetsAt": 1800000000},
        "secondary": {"usedPercent": 10, "windowDurationMins": 10080, "resetsAt": 1800100000},
    }}})
    assert "75% remaining" in result
    assert "90% remaining" in result
    assert "5h" in result and "7d" in result
    assert "reset" in result
    assert "unavailable" in format_codex_limits({}).lower()


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_api_usage_without_admin_key_does_not_query_billing(provider, monkeypatch):
    from infinidev.config.usage import UsageSelection, render_usage

    monkeypatch.setattr(httpx, "get", lambda *a, **k: pytest.fail("No admin key configured"))
    report = render_usage(UsageSelection(provider, "test-model", "test-key", ""))
    assert "admin" in report.lower()
    assert "subscription" in report.lower()


@pytest.mark.parametrize("provider,expected", [("openai", "$1.2500"), ("anthropic", "$1.2500")])
def test_admin_usage_uses_native_auth_and_cost_units(provider, expected, monkeypatch):
    from infinidev.config.usage import UsageSelection, render_usage

    requests = []

    def get(url, **kwargs):
        requests.append((url, kwargs))
        if "cost" in url:
            row = {"amount": {"value": 1.25, "currency": "usd"}} if provider == "openai" else {
                "amount": "125", "currency": "USD",
            }
        else:
            row = {"input_tokens": 100, "output_tokens": 20} if provider == "openai" else {
                "uncached_input_tokens": 50, "cache_read_input_tokens": 25,
                "cache_creation": {"ephemeral_5m_input_tokens": 25}, "output_tokens": 20,
            }
        return httpx.Response(200, request=httpx.Request("GET", url),
                              json={"data": [{"results": [row]}], "has_more": False})

    monkeypatch.setattr(httpx, "get", get)
    report = render_usage(UsageSelection(provider, "test-model", "inference-key", "", "admin-key"))
    assert "100 input" in report and "20 output" in report
    assert expected in report
    assert len(requests) == 2
    for url, kwargs in requests:
        headers = kwargs["headers"]
        assert "inference-key" not in str(headers)
        if provider == "openai":
            assert url.startswith("https://api.openai.com/v1/organization/")
            assert headers["Authorization"] == "Bearer admin-key"
        else:
            assert url.startswith("https://api.anthropic.com/v1/organizations/")
            assert headers["x-api-key"] == "admin-key"


def test_rate_limit_auth_failure_is_actionable_and_never_echoes_secrets(monkeypatch):
    from infinidev.config.usage import UsageSelection, render_usage

    monkeypatch.setattr(httpx, "get", lambda url, **kwargs: httpx.Response(
        403, request=httpx.Request("GET", url), text="secret-provider-response",
    ))
    report = render_usage(UsageSelection("openai", "gpt-6-astra", "key", "", "admin-secret"))
    assert "403" in report and "unavailable" in report.lower()
    assert "secret" not in report


@pytest.mark.parametrize("timeout", [False, True])
def test_codex_client_handshake_deadline_and_process_cleanup(tmp_path, monkeypatch, timeout):
    from infinidev.config import codex_usage

    calls = tmp_path / "calls.jsonl"
    executable = tmp_path / "codex"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, sys, time\n"
        f"log = open({str(calls)!r}, 'a', buffering=1)\n"
        "for line in sys.stdin:\n"
        "    log.write(line)\n"
        "    message = json.loads(line)\n"
        "    if message['method'] == 'initialize':\n"
        "        print(json.dumps({'method': 'notification'}), flush=True)\n"
        "        print(json.dumps({'id': 1, 'result': {}}), flush=True)\n"
        "    elif message['method'] == 'account/rateLimits/read':\n"
        + ("        time.sleep(30)\n" if timeout else
           "        print(json.dumps({'id': 2, 'result': {'rateLimits': {}}}), flush=True)\n")
    )
    executable.chmod(0o755)
    monkeypatch.setattr(codex_usage.shutil, "which", lambda name: str(executable))
    processes = []
    popen = codex_usage.subprocess.Popen

    def capture(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(codex_usage.subprocess, "Popen", capture)
    if timeout:
        with pytest.raises(TimeoutError):
            codex_usage.read_codex_limits(timeout=0.5)
    else:
        assert codex_usage.read_codex_limits() == {"rateLimits": {}}
    assert processes[0].poll() is not None
    assert [json.loads(line)["method"] for line in calls.read_text().splitlines()] == [
        "initialize", "initialized", "account/rateLimits/read",
    ]


def test_usage_command_wiring_keeps_tui_query_off_the_ui_thread(monkeypatch, capsys):
    from infinidev.cli.commands import handle_command as classic_command
    from infinidev.ui.handlers.commands import handle_command

    monkeypatch.setattr("infinidev.config.usage.render_usage", lambda *args: "usage report")
    assert classic_command("/usage") is True
    assert "usage report" in capsys.readouterr().out
    queued = []
    monkeypatch.setattr("infinidev.ui.workers.run_in_background", lambda app, fn: queued.append(fn))
    messages = []
    app = SimpleNamespace(_usage_pending=False, flash_status=lambda message: None,
                          add_message=lambda *args: messages.append(args))
    handle_command(app, "/usage")
    assert not messages and app._usage_pending
    handle_command(app, "/usage")
    assert len(queued) == 1
    queued[0]()
    assert messages == [("System", "usage report", "system")]
    assert not app._usage_pending
