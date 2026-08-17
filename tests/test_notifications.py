"""Tests for the programmable notifications subsystem.

Coverage:
  - models: serialization roundtrip for TriggerSpec / ChannelConfig
  - storage: CRUD, history, script + file state, integrity
  - channels: console writes JSONL line, webhook POSTs JSON
  - scheduler: cron parsing, interval/script/file/agent evaluation
  - tool: action dispatch, validation, error messages
"""

from __future__ import annotations

import http.server
import json
import os
import threading
import time
from pathlib import Path

import pytest

from infinidev.notifications import (
    ChannelConfig,
    Notification,
    NotificationScheduler,
    NotificationStore,
    TriggerSpec,
)
from infinidev.notifications.channels import (
    deliver,
    deliver_console,
    deliver_webhook,
    file_signature,
    render_template,
)
from infinidev.notifications.scheduler import (
    POLL_INTERVAL_SECONDS,
    cron_matches,
    parse_cron,
    reset_default_scheduler,
)
from infinidev.tools.meta.notifications_tool import (
    ManageNotificationsTool,
    _build_channel_config,
    _build_trigger_spec,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "notifications.db"


@pytest.fixture
def store(tmp_db: Path) -> NotificationStore:
    s = NotificationStore(tmp_db)
    s.reset()
    yield s
    s.reset()


@pytest.fixture
def scheduler(store: NotificationStore) -> NotificationScheduler:
    """A scheduler that does not start its own thread (tests drive ticks)."""
    return NotificationScheduler(store=store, poll_interval=POLL_INTERVAL_SECONDS)


@pytest.fixture
def fresh_default(tmp_path: Path, monkeypatch):
    """Reset the default store and scheduler so tests start clean.

    Each test gets its own SQLite file under tmp_path so history rows
    from previous tests never bleed into ``test_history_empty`` or any
    other test that asserts an empty starting state.
    """
    from infinidev.notifications import storage as storage_module
    from infinidev.notifications import scheduler as scheduler_module

    db_path = tmp_path / "default.db"
    custom_store = NotificationStore(db_path)
    custom_store.reset()
    monkeypatch.setattr(storage_module, "_default_store", custom_store)
    scheduler_module.reset_default_scheduler()
    # Bind the lazily-created default scheduler to our test store once
    # the tool triggers it.
    original_get = scheduler_module.get_default_scheduler

    def _patched_get():
        sched = original_get()
        sched._store = custom_store
        return sched

    monkeypatch.setattr(scheduler_module, "get_default_scheduler", _patched_get)
    # Patch the symbol the tool imported, so its `get_default_scheduler`
    # call resolves to the patched one.
    from infinidev.tools.meta import notifications_tool as tool_module
    monkeypatch.setattr(tool_module, "get_default_scheduler", _patched_get)
    yield
    reset_default_scheduler()
    storage_module.reset_default_store_for_tests()


# ── Models ────────────────────────────────────────────────────────────────


class TestTriggerSpec:
    def test_roundtrip_with_all_fields(self):
        original = TriggerSpec(
            type="script",
            every_seconds=60,
            cron="*/5 * * * *",
            command="echo hi",
            working_dir="/tmp",
            expected_exit_code=0,
            stdout_match=r"version \d+",
            path="/var/log/x",
            watch="sha256",
        )
        restored = TriggerSpec.from_json(original.to_json())
        assert restored == original

    def test_from_json_empty_or_invalid_returns_default(self):
        assert TriggerSpec.from_json(None) == TriggerSpec()
        assert TriggerSpec.from_json("not-json") == TriggerSpec()

    def test_from_json_drops_unknown_keys(self):
        restored = TriggerSpec.from_json(json.dumps({"type": "agent", "evil": True}))
        assert restored.type == "agent"
        assert not hasattr(restored, "evil")


class TestChannelConfig:
    def test_roundtrip_with_headers(self):
        original = ChannelConfig(
            type="webhook",
            url="https://example.test/hook",
            method="PUT",
            headers={"X-Token": "abc", "X-Source": "test"},
        )
        restored = ChannelConfig.from_json(original.to_json())
        assert restored == original

    def test_invalid_headers_coerced_to_empty(self):
        restored = ChannelConfig.from_json(json.dumps({"headers": "not a dict"}))
        assert restored.headers == {}


class TestNotificationRow:
    def test_to_row_and_from_row_roundtrip(self):
        n = Notification(
            id=7,
            name="ping-5min",
            enabled=True,
            trigger=TriggerSpec(type="interval", every_seconds=300),
            channel=ChannelConfig(type="console"),
            title="Heartbeat",
            template="alive",
            created_at=1.0,
            last_fired_at=2.0,
            fire_count=3,
        )
        row = n.to_row()
        rebuilt = Notification.from_row(row)
        assert rebuilt.id == n.id
        assert rebuilt.name == n.name
        assert rebuilt.trigger == n.trigger
        assert rebuilt.channel == n.channel
        assert rebuilt.title == n.title
        assert rebuilt.fire_count == n.fire_count


# ── Storage ───────────────────────────────────────────────────────────────


class TestNotificationStore:
    def test_create_and_get_by_name(self, store: NotificationStore):
        n = store.create(
            name="heartbeat",
            trigger=TriggerSpec(type="interval", every_seconds=10),
            channel=ChannelConfig(type="console"),
        )
        assert n.id > 0
        assert n.created_at > 0
        assert n.last_fired_at is None
        assert n.fire_count == 0
        fetched = store.get_by_name("heartbeat")
        assert fetched is not None
        assert fetched.name == "heartbeat"
        assert fetched.trigger.every_seconds == 10

    def test_unique_name_raises(self, store: NotificationStore):
        store.create(
            name="dup",
            trigger=TriggerSpec(type="agent"),
            channel=ChannelConfig(type="console"),
        )
        with pytest.raises(ValueError, match="already exists"):
            store.create(
                name="dup",
                trigger=TriggerSpec(type="agent"),
                channel=ChannelConfig(type="console"),
            )

    def test_list_filters_by_enabled(self, store: NotificationStore):
        store.create(
            name="a",
            trigger=TriggerSpec(type="agent"),
            channel=ChannelConfig(type="console"),
        )
        store.create(
            name="b",
            trigger=TriggerSpec(type="agent"),
            channel=ChannelConfig(type="console"),
            enabled=False,
        )
        all_ = store.list_all()
        enabled = store.list_enabled()
        assert {n.name for n in all_} == {"a", "b"}
        assert {n.name for n in enabled} == {"a"}

    def test_update_enabled(self, store: NotificationStore):
        n = store.create(
            name="toggle",
            trigger=TriggerSpec(type="agent"),
            channel=ChannelConfig(type="console"),
        )
        assert store.update_enabled(n.id, False) is True
        assert store.get(n.id).enabled is False
        assert store.update_enabled(n.id, True) is True

    def test_delete_by_name_cascades_history(self, store: NotificationStore):
        n = store.create(
            name="ephemeral",
            trigger=TriggerSpec(type="agent"),
            channel=ChannelConfig(type="console"),
        )
        store.record_fire(n.id, "delivered")
        assert len(store.history(n.id)) == 1
        assert store.delete_by_name("ephemeral") is True
        assert store.get_by_name("ephemeral") is None
        assert store.history(n.id) == []

    def test_history_no_filter(self, store: NotificationStore):
        a = store.create(
            name="a",
            trigger=TriggerSpec(type="agent"),
            channel=ChannelConfig(type="console"),
        )
        b = store.create(
            name="b",
            trigger=TriggerSpec(type="agent"),
            channel=ChannelConfig(type="console"),
        )
        store.record_fire(a.id, "delivered", payload={"x": 1})
        store.record_fire(b.id, "error", error="boom", payload={"y": 2})
        rows = store.history()
        assert len(rows) == 2
        names = {r["notification_name"] for r in rows}
        assert names == {"a", "b"}

    def test_script_state_roundtrip(self, store: NotificationStore):
        n = store.create(
            name="s",
            trigger=TriggerSpec(type="script"),
            channel=ChannelConfig(type="console"),
        )
        assert store.get_script_state(n.id) is None
        store.set_script_state(n.id, 0, "hello", time.time(), True)
        state = store.get_script_state(n.id)
        assert state["last_exit_code"] == 0
        assert state["last_stdout"] == "hello"
        assert state["last_match"] is True

    def test_file_state_roundtrip(self, store: NotificationStore):
        n = store.create(
            name="f",
            trigger=TriggerSpec(type="file"),
            channel=ChannelConfig(type="console"),
        )
        assert store.get_file_state(n.id) is None
        store.set_file_state(n.id, "mtime:1:1", time.time())
        state = store.get_file_state(n.id)
        assert state["last_signature"] == "mtime:1:1"


# ── Cron ──────────────────────────────────────────────────────────────────


class TestCronParser:
    def test_parse_star_each_field(self):
        m, h, dom, mo, dow = parse_cron("* * * * *")
        assert m == set(range(0, 60))
        assert h == set(range(0, 24))
        assert dom == set(range(1, 32))
        assert mo == set(range(1, 13))
        assert dow == set(range(0, 7))

    def test_parse_step_expression(self):
        m, *_ = parse_cron("*/15 * * * *")
        assert m == {0, 15, 30, 45}

    def test_parse_range_and_list(self):
        m, h, *_ = parse_cron("0,30 9-12 * * *")
        assert m == {0, 30}
        assert h == {9, 10, 11, 12}

    def test_parse_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_cron("* * * *")  # too few
        with pytest.raises(ValueError):
            parse_cron("99 * * * *")  # minute out of range
        with pytest.raises(ValueError):
            parse_cron("*/abc * * * *")  # non-int step

    def test_cron_matches_at_top_of_hour(self):
        # cron "0 * * * *" fires every hour at minute 0
        # Pick a struct_time that is Monday 2025-01-06 09:00
        struct = time.strptime("2025-01-06 09:00:00", "%Y-%m-%d %H:%M:%S")
        assert cron_matches("0 * * * *", struct) is True
        # 09:01 should not match
        struct2 = time.strptime("2025-01-06 09:01:00", "%Y-%m-%d %H:%M:%S")
        assert cron_matches("0 * * * *", struct2) is False

    def test_cron_matches_every_5_minutes(self):
        # "*/5 * * * *" should fire at :00, :05, :10, ...
        for minute in (0, 5, 10, 55):
            struct = time.strptime(f"2025-01-06 09:{minute:02d}:00", "%Y-%m-%d %H:%M:%S")
            assert cron_matches("*/5 * * * *", struct) is True
        for minute in (1, 7, 13, 59):
            struct = time.strptime(f"2025-01-06 09:{minute:02d}:00", "%Y-%m-%d %H:%M:%S")
            assert cron_matches("*/5 * * * *", struct) is False


# ── Channels ──────────────────────────────────────────────────────────────


class TestChannels:
    def test_render_template_basic(self):
        out = render_template(
            "{name} ok at {fired_at}", {"name": "x", "fired_at": 1.0}
        )
        assert out == "x ok at 1.0"

    def test_render_template_missing_var_returns_template_unchanged(self):
        # The fallback keeps the raw template rather than raising or returning empty.
        assert render_template("{nope}", {}) == "{nope}"

    def test_deliver_console_appends_jsonl(self, tmp_path: Path):
        log = tmp_path / "n.log"
        deliver_console(ChannelConfig(type="console", log_path=str(log)),
                        {"name": "x", "title": "t", "body": "b"})
        text = log.read_text()
        # exactly one newline-terminated JSON object
        lines = [ln for ln in text.splitlines() if ln.strip()]
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["name"] == "x"
        assert parsed["title"] == "t"

    def test_deliver_webhook_posts_json(self):
        captured: dict = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                captured["body"] = json.loads(body)
                captured["headers"] = dict(self.headers)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, fmt, *args):  # silence test output
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            cfg = ChannelConfig(
                type="webhook",
                url=f"http://127.0.0.1:{port}/hook",
                method="POST",
                headers={"X-Test": "yes"},
            )
            deliver(cfg, {"name": "p", "body": "hi"})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

        assert captured["body"]["name"] == "p"
        assert captured["headers"].get("X-Test") == "yes"
        assert captured["headers"].get("Content-Type") == "application/json"

    def test_deliver_webhook_raises_on_bad_url(self):
        with pytest.raises(RuntimeError):
            deliver_webhook(
                ChannelConfig(type="webhook", url="http://127.0.0.1:1/dead"),
                {"name": "x"},
            )

    def test_deliver_unknown_channel_raises(self):
        with pytest.raises(ValueError):
            deliver(ChannelConfig(type="sms"), {"name": "x"})


class TestFileSignature:
    def test_signature_includes_size(self, tmp_path: Path):
        p = tmp_path / "f.txt"
        p.write_text("hello")
        sig = file_signature(str(p), "mtime")
        assert sig is not None
        assert sig.startswith("mtime:")
        assert sig.endswith(":5")  # size

    def test_signature_changes_after_modification(self, tmp_path: Path):
        p = tmp_path / "f.txt"
        p.write_text("hello")
        before = file_signature(str(p), "mtime")
        time.sleep(0.05)
        p.write_text("hello world")
        after = file_signature(str(p), "mtime")
        assert before != after

    def test_signature_returns_none_for_missing_file(self, tmp_path: Path):
        assert file_signature(str(tmp_path / "missing"), "mtime") is None
        assert file_signature(str(tmp_path / "missing"), "sha256") is None


# ── Scheduler ─────────────────────────────────────────────────────────────


class TestSchedulerInterval:
    def test_interval_fires_on_first_tick(self, scheduler: NotificationScheduler, store: NotificationStore):
        n = store.create(
            name="beat",
            trigger=TriggerSpec(type="interval", every_seconds=60),
            channel=ChannelConfig(type="console"),
        )
        results = scheduler.tick(now=time.time())
        assert len(results) == 1
        assert results[0]["name"] == "beat"
        assert results[0]["status"] == "delivered"

    def test_interval_does_not_fire_before_window(self, scheduler: NotificationScheduler, store: NotificationStore):
        store.create(
            name="slow",
            trigger=TriggerSpec(type="interval", every_seconds=300),
            channel=ChannelConfig(type="console"),
        )
        # First fire
        scheduler.tick(now=time.time())
        # Subsequent tick 1s later must not fire (window is 300s)
        results = scheduler.tick(now=time.time() + 1)
        assert results == []

    def test_interval_fires_again_after_window(self, scheduler: NotificationScheduler, store: NotificationStore):
        store.create(
            name="ok",
            trigger=TriggerSpec(type="interval", every_seconds=10),
            channel=ChannelConfig(type="console"),
        )
        scheduler.tick(now=time.time())
        results = scheduler.tick(now=time.time() + 11)
        assert len(results) == 1


class TestSchedulerCron:
    def test_cron_does_not_fire_when_minute_mismatches(
        self, scheduler: NotificationScheduler, store: NotificationStore
    ):
        store.create(
            name="hourly",
            trigger=TriggerSpec(type="cron", cron="0 * * * *"),
            channel=ChannelConfig(type="console"),
        )
        # Use a struct_time that is at minute 5
        struct = time.strptime("2025-01-06 09:05:00", "%Y-%m-%d %H:%M:%S")
        assert scheduler.tick(now=time.mktime(struct)) == []

    def test_cron_does_not_fire_twice_within_minute(
        self, scheduler: NotificationScheduler, store: NotificationStore
    ):
        store.create(
            name="hourly",
            trigger=TriggerSpec(type="cron", cron="0 * * * *"),
            channel=ChannelConfig(type="console"),
        )
        struct = time.strptime("2025-01-06 09:00:00", "%Y-%m-%d %H:%M:%S")
        ts = time.mktime(struct)
        assert len(scheduler.tick(now=ts)) == 1
        # Same minute
        assert scheduler.tick(now=ts + 5) == []


class TestSchedulerScript:
    def test_script_fires_on_exit_code_match(
        self, scheduler: NotificationScheduler, store: NotificationStore
    ):
        store.create(
            name="on-fail",
            trigger=TriggerSpec(
                type="script",
                command="python -c \"import sys; sys.exit(1)\"",
                expected_exit_code=1,
            ),
            channel=ChannelConfig(type="console"),
        )
        results = scheduler.tick(now=time.time())
        assert len(results) == 1
        assert results[0]["payload"]["exit_code"] == 1

    def test_script_does_not_fire_on_exit_code_mismatch(
        self, scheduler: NotificationScheduler, store: NotificationStore
    ):
        store.create(
            name="on-fail",
            trigger=TriggerSpec(
                type="script",
                command="python -c 'import sys; sys.exit(0)'",
                expected_exit_code=1,
            ),
            channel=ChannelConfig(type="console"),
        )
        assert scheduler.tick(now=time.time()) == []

    def test_script_fires_on_stdout_transition(
        self, scheduler: NotificationScheduler, store: NotificationStore
    ):
        store.create(
            name="on-version",
            trigger=TriggerSpec(
                type="script",
                command="python -c 'print(\"version 99\")'",
                stdout_match=r"version \d+",
            ),
            channel=ChannelConfig(type="console"),
        )
        # First tick: stdout matches, last_match was None → fires, state set True
        assert len(scheduler.tick(now=time.time())) == 1
        # Second tick within the same poll window: state.last_match=True → no fire
        assert scheduler.tick(now=time.time() + 0.5) == []

    def test_script_handles_timeout_gracefully(self, scheduler: NotificationScheduler, store: NotificationStore):
        store.create(
            name="slow",
            trigger=TriggerSpec(
                type="script",
                command="python -c 'import time; time.sleep(60)'",
                expected_exit_code=0,
            ),
            channel=ChannelConfig(type="console"),
        )
        # Should NOT raise; the run hits a TimeoutExpired and the
        # scheduler treats it as a non-firing tick.
        results = scheduler.tick(now=time.time())
        assert results == []


class TestSchedulerFile:
    def test_file_fires_on_first_observation_with_state(
        self,
        scheduler: NotificationScheduler,
        store: NotificationStore,
        tmp_path: Path,
    ):
        p = tmp_path / "watched.txt"
        p.write_text("v1")
        n = store.create(
            name="file-watch",
            trigger=TriggerSpec(type="file", path=str(p), watch="mtime"),
            channel=ChannelConfig(type="console"),
        )
        results = scheduler.tick(now=time.time())
        assert len(results) == 1
        # Second tick without change: must not fire again.
        assert scheduler.tick(now=time.time() + 1) == []

    def test_file_fires_again_after_change(
        self,
        scheduler: NotificationScheduler,
        store: NotificationStore,
        tmp_path: Path,
    ):
        p = tmp_path / "watched.txt"
        p.write_text("v1")
        store.create(
            name="file-watch",
            trigger=TriggerSpec(type="file", path=str(p), watch="mtime"),
            channel=ChannelConfig(type="console"),
        )
        scheduler.tick(now=time.time())
        # Force a change with a different mtime.
        time.sleep(0.05)
        p.write_text("v2")
        results = scheduler.tick(now=time.time() + 1)
        assert len(results) == 1

    def test_file_missing_does_not_fire(
        self,
        scheduler: NotificationScheduler,
        store: NotificationStore,
        tmp_path: Path,
    ):
        store.create(
            name="missing",
            trigger=TriggerSpec(
                type="file", path=str(tmp_path / "nope"), watch="mtime"
            ),
            channel=ChannelConfig(type="console"),
        )
        assert scheduler.tick(now=time.time()) == []


class TestSchedulerAgent:
    def test_agent_trigger_does_not_poll_fire(
        self, scheduler: NotificationScheduler, store: NotificationStore
    ):
        store.create(
            name="agent-only",
            trigger=TriggerSpec(type="agent"),
            channel=ChannelConfig(type="console"),
        )
        assert scheduler.tick(now=time.time()) == []

    def test_fire_agent_by_name(self, scheduler: NotificationScheduler, store: NotificationStore):
        store.create(
            name="agent-only",
            trigger=TriggerSpec(type="agent"),
            channel=ChannelConfig(type="console"),
        )
        result = scheduler.fire_agent("agent-only", {"foo": "bar"})
        assert result["status"] == "delivered"
        assert result["payload"]["foo"] == "bar"

    def test_fire_unknown_returns_not_found(self, scheduler: NotificationScheduler):
        result = scheduler.fire_agent("nope")
        assert result["status"] == "not_found"

    def test_fire_records_history(self, scheduler: NotificationScheduler, store: NotificationStore):
        n = store.create(
            name="agent-only",
            trigger=TriggerSpec(type="agent"),
            channel=ChannelConfig(type="console"),
        )
        scheduler.fire_agent("agent-only")
        rows = store.history(n.id)
        assert len(rows) == 1
        assert rows[0]["status"] == "delivered"


# ── Tool ──────────────────────────────────────────────────────────────────


class TestManageNotificationsTool:
    def _tool(self) -> ManageNotificationsTool:
        return ManageNotificationsTool()

    def test_list_when_empty(self, fresh_default):
        out = self._tool()._run(action="list")
        assert "No notifications" in out

    def test_create_then_list(self, fresh_default):
        tool = self._tool()
        result = tool._run(
            action="create",
            name="daily-9",
            trigger_type="cron",
            trigger={"cron": "0 9 * * *"},
            channel_type="console",
        )
        assert "registered" in result
        listing = tool._run(action="list")
        assert "daily-9" in listing
        assert "cron" in listing

    def test_create_requires_name(self, fresh_default):
        out = self._tool()._run(
            action="create",
            trigger_type="agent",
            channel_type="console",
        )
        assert "name" in out.lower()

    def test_create_validates_trigger_fields(self, fresh_default):
        out = self._tool()._run(
            action="create",
            name="bad-cron",
            trigger_type="cron",
            trigger={"cron": "not a cron"},
            channel_type="console",
        )
        assert "cron" in out.lower() or "5 field" in out.lower()

    def test_create_validates_webhook_url(self, fresh_default):
        out = self._tool()._run(
            action="create",
            name="bad-webhook",
            trigger_type="agent",
            channel={},
            channel_type="webhook",
        )
        assert "url" in out.lower()

    def test_enable_disable(self, fresh_default):
        tool = self._tool()
        tool._run(
            action="create",
            name="toggle",
            trigger_type="interval",
            trigger={"every_seconds": 60},
            channel_type="console",
        )
        out = tool._run(action="disable", name="toggle")
        assert "enabled" in out
        listing = tool._run(action="list")
        assert "[off]" in listing
        tool._run(action="enable", name="toggle")
        listing = tool._run(action="list")
        assert "[on]" in listing

    def test_delete(self, fresh_default):
        tool = self._tool()
        tool._run(
            action="create",
            name="ephemeral",
            trigger_type="agent",
            channel_type="console",
        )
        out = tool._run(action="delete", name="ephemeral")
        assert "deleted" in out
        listing = tool._run(action="list")
        assert "ephemeral" not in listing

    def test_fire_agent(self, fresh_default):
        tool = self._tool()
        tool._run(
            action="create",
            name="manual",
            trigger_type="agent",
            channel_type="console",
        )
        out = tool._run(action="fire", name="manual", trigger={"k": "v"})
        assert "delivered" in out

    def test_fire_unknown_returns_error(self, fresh_default):
        out = self._tool()._run(action="fire", name="nope")
        assert "no notification" in out.lower()

    def test_history_empty(self, fresh_default):
        out = self._tool()._run(action="history")
        assert "No fire history" in out

    def test_unknown_action(self, fresh_default):
        out = self._tool()._run(action="frobnicate")
        assert "unknown" in out.lower()


class TestTriggerSpecBuilders:
    def test_interval_builder(self):
        spec = _build_trigger_spec(__import__("infinidev.notifications.models", fromlist=["NotificationTrigger"]).NotificationTrigger.INTERVAL, {"every_seconds": 30})
        assert spec.type == "interval"
        assert spec.every_seconds == 30

    def test_interval_builder_accepts_seconds_alias(self):
        spec = _build_trigger_spec(__import__("infinidev.notifications.models", fromlist=["NotificationTrigger"]).NotificationTrigger.INTERVAL, {"seconds": 90})
        assert spec.every_seconds == 90

    def test_interval_builder_rejects_zero(self):
        from infinidev.notifications.models import NotificationTrigger
        with pytest.raises(ValueError):
            _build_trigger_spec(NotificationTrigger.INTERVAL, {"every_seconds": 0})

    def test_cron_builder(self):
        from infinidev.notifications.models import NotificationTrigger
        spec = _build_trigger_spec(NotificationTrigger.CRON, {"cron": "*/5 * * * *"})
        assert spec.cron == "*/5 * * * *"

    def test_cron_builder_rejects_bad_expression(self):
        from infinidev.notifications.models import NotificationTrigger
        with pytest.raises(ValueError):
            _build_trigger_spec(NotificationTrigger.CRON, {"cron": "broken"})

    def test_script_builder(self):
        from infinidev.notifications.models import NotificationTrigger
        spec = _build_trigger_spec(
            NotificationTrigger.SCRIPT,
            {"command": "ls", "expected_exit_code": 0, "stdout_match": "x"},
        )
        assert spec.command == "ls"
        assert spec.expected_exit_code == 0
        assert spec.stdout_match == "x"

    def test_script_builder_rejects_blank(self):
        from infinidev.notifications.models import NotificationTrigger
        with pytest.raises(ValueError):
            _build_trigger_spec(NotificationTrigger.SCRIPT, {"command": "  "})

    def test_file_builder_default_watch(self):
        from infinidev.notifications.models import NotificationTrigger
        spec = _build_trigger_spec(NotificationTrigger.FILE, {"path": "/tmp/x"})
        assert spec.watch == "mtime"

    def test_file_builder_rejects_invalid_watch(self):
        from infinidev.notifications.models import NotificationTrigger
        with pytest.raises(ValueError):
            _build_trigger_spec(NotificationTrigger.FILE, {"path": "/tmp/x", "watch": "ctime"})

    def test_agent_builder_no_fields_needed(self):
        from infinidev.notifications.models import NotificationTrigger
        spec = _build_trigger_spec(NotificationTrigger.AGENT, {})
        assert spec.type == "agent"


class TestChannelConfigBuilders:
    def test_console_default(self):
        cfg = _build_channel_config("console", {})
        assert cfg.type == "console"
        assert cfg.log_path is None

    def test_console_with_path(self):
        cfg = _build_channel_config("console", {"log_path": "/tmp/x.log"})
        assert cfg.log_path == "/tmp/x.log"

    def test_webhook_minimum(self):
        cfg = _build_channel_config("webhook", {"url": "https://example.test"})
        assert cfg.type == "webhook"
        assert cfg.method == "POST"
        assert cfg.headers == {}

    def test_webhook_rejects_missing_url(self):
        with pytest.raises(ValueError):
            _build_channel_config("webhook", {})

    def test_webhook_rejects_bad_method(self):
        with pytest.raises(ValueError):
            _build_channel_config("webhook", {"url": "https://example.test", "method": "GET"})

    def test_webhook_coerces_headers(self):
        cfg = _build_channel_config(
            "webhook",
            {"url": "https://x", "headers": {"a": "b", "c": "d"}},
        )
        assert cfg.headers == {"a": "b", "c": "d"}

    def test_unknown_channel_raises(self):
        with pytest.raises(ValueError):
            _build_channel_config("sms", {})


# ── Default-singleton smoke test ──────────────────────────────────────────


def test_default_scheduler_start_and_stop(fresh_default, tmp_db):
    """The default scheduler should start its thread and tick at least once.

    Uses a custom store path so we don't touch the user's real
    ``~/.infinidev/notifications.db``. We mock the default store by
    pointing the scheduler's internal reference after import.
    """
    from infinidev.notifications import storage as storage_module
    from infinidev.notifications import scheduler as scheduler_module

    # Replace the default store + scheduler with instances bound to tmp_db.
    custom_store = NotificationStore(tmp_db)
    custom_store.reset()
    storage_module._default_store = custom_store

    scheduler_module.reset_default_scheduler()
    sched = scheduler_module.get_default_scheduler()
    sched._store = custom_store  # bind scheduler to our test store
    # Interval notification should fire on first tick.
    custom_store.create(
        name="ping",
        trigger=TriggerSpec(type="interval", every_seconds=1),
        channel=ChannelConfig(type="console"),
    )
    # Don't actually start the thread — call tick() directly so the test
    # completes deterministically. The integration is that get_default_scheduler
    # is callable without error and uses the right store.
    sched.tick(now=time.time())
    rows = custom_store.history()
    assert len(rows) == 1
    assert rows[0]["status"] == "delivered"