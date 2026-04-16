"""Tests for chat-agent primitives: EscalationPacket, ChatAgentResult,
RespondTool, EscalateTool. Commit 3 of the pipeline redesign.

These are the leaf-node data contracts the chat-agent orchestrator
(Commit 4) and the pipeline rewrite (Commit 7) will consume. The
tests pin down: (1) the dataclass invariants, (2) the tool schemas,
(3) the read-only classification that keeps both new tools inside
the chat-agent whitelist.
"""

import json
import pytest

from infinidev.engine.orchestration.escalation_packet import EscalationPacket
from infinidev.engine.orchestration.chat_agent_result import ChatAgentResult
from infinidev.tools.chat_agent import RespondTool, EscalateTool
from infinidev.tools import get_tools_for_role


class TestEscalationPacket:
    def test_minimal_packet_defaults(self):
        pkt = EscalationPacket(
            user_request="fix the bug",
            understanding="user wants me to fix a bug",
        )
        assert pkt.suggested_flow == "develop"
        assert pkt.opened_files == []
        assert pkt.user_visible_preview == ""
        assert pkt.user_signal == ""

    def test_full_packet(self):
        pkt = EscalationPacket(
            user_request="¿podés arreglar el JWT?",
            understanding="JWT validation is broken in auth.py",
            opened_files=["src/auth.py", "src/jwt_utils.py"],
            user_visible_preview="Voy a arreglar la validación del JWT.",
            user_signal="sí dale",
        )
        assert pkt.opened_files == ["src/auth.py", "src/jwt_utils.py"]
        assert pkt.user_signal == "sí dale"

    def test_packet_is_frozen(self):
        pkt = EscalationPacket(user_request="x", understanding="y")
        with pytest.raises(Exception):
            pkt.user_request = "mutated"  # type: ignore[misc]


class TestChatAgentResult:
    def test_respond_variant(self):
        r = ChatAgentResult(kind="respond", reply="hola!")
        assert r.kind == "respond"
        assert r.reply == "hola!"
        assert r.escalation is None

    def test_respond_requires_non_empty_reply(self):
        with pytest.raises(ValueError, match="non-empty reply"):
            ChatAgentResult(kind="respond", reply="")

    def test_respond_cannot_carry_escalation(self):
        pkt = EscalationPacket(user_request="x", understanding="y")
        with pytest.raises(ValueError, match="must not carry an escalation"):
            ChatAgentResult(kind="respond", reply="hi", escalation=pkt)

    def test_escalate_variant(self):
        pkt = EscalationPacket(user_request="x", understanding="y")
        r = ChatAgentResult(kind="escalate", escalation=pkt)
        assert r.kind == "escalate"
        assert r.escalation is pkt
        assert r.reply == ""

    def test_escalate_requires_packet(self):
        with pytest.raises(ValueError, match="requires an escalation packet"):
            ChatAgentResult(kind="escalate")

    def test_rejects_unknown_kind(self):
        with pytest.raises(ValueError, match="Invalid kind"):
            ChatAgentResult(kind="done")  # type: ignore[arg-type]


class TestRespondTool:
    def test_schema_name_and_args(self):
        tool = RespondTool()
        assert tool.name == "respond"
        assert tool.is_read_only is True
        fields = tool.args_schema.model_fields
        assert "message" in fields
        assert "language" in fields

    def test_run_returns_structured_payload(self):
        tool = RespondTool()
        raw = tool._run(message="hola", language="es")
        obj = json.loads(raw)
        assert obj == {"kind": "respond", "message": "hola", "language": "es"}


class TestEscalateTool:
    def test_schema_name_and_args(self):
        tool = EscalateTool()
        assert tool.name == "escalate"
        assert tool.is_read_only is True
        fields = tool.args_schema.model_fields
        for key in (
            "understanding",
            "user_visible_preview",
            "opened_files",
            "user_signal",
            "suggested_flow",
        ):
            assert key in fields

    def test_run_returns_structured_payload(self):
        tool = EscalateTool()
        raw = tool._run(
            understanding="fix auth bug",
            user_visible_preview="Voy a arreglarlo",
            opened_files=["src/auth.py"],
            user_signal="dale",
            suggested_flow="develop",
        )
        obj = json.loads(raw)
        assert obj["kind"] == "escalate"
        assert obj["understanding"] == "fix auth bug"
        assert obj["opened_files"] == ["src/auth.py"]
        assert obj["user_signal"] == "dale"
        assert obj["suggested_flow"] == "develop"

    def test_run_defaults(self):
        tool = EscalateTool()
        raw = tool._run(understanding="something")
        obj = json.loads(raw)
        assert obj["opened_files"] == []
        assert obj["suggested_flow"] == "develop"


class TestChatAgentRoleIncludesTerminators:
    def test_respond_and_escalate_in_chat_agent_toolbox(self):
        tools = get_tools_for_role("chat_agent")
        names = {t.name for t in tools}
        assert "respond" in names
        assert "escalate" in names

    def test_respond_and_escalate_not_in_developer_toolbox(self):
        tools = get_tools_for_role("developer")
        names = {t.name for t in tools}
        assert "respond" not in names, (
            "respond must be chat-agent-exclusive — the developer "
            "terminates via step_complete"
        )
        assert "escalate" not in names, (
            "escalate must be chat-agent-exclusive — the developer "
            "cannot escalate to itself"
        )
