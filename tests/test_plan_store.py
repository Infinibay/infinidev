"""Tests for the durable mini-plan store and tools."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from infinidev.engine import plan_store
from infinidev.tools.meta.plan_tools import (
    PlanAddTool,
    PlanListTool,
    PlanRemoveTool,
    PlanUpdateTool,
)


@pytest.fixture
def plan_base(monkeypatch, tmp_path):
    monkeypatch.setenv("INFINIDEV_BASE_DIR", str(tmp_path))
    yield tmp_path


def test_add_item_is_idempotent(plan_base):
    first = plan_store.add_plan_item(
        "rewrite",
        "Wire Ken MCP",
        idempotency_key="ken-mcp",
    )
    second = plan_store.add_plan_item(
        "rewrite",
        "Wire Ken MCP again",
        idempotency_key="ken-mcp",
    )
    assert first.id == second.id


def test_update_and_remove_items(plan_base):
    item = plan_store.add_plan_item("rewrite", "Design runtime")
    updated = plan_store.update_plan_item(
        "rewrite", item.id, status="completed", notes="ok"
    )
    assert updated.status == "completed"
    assert plan_store.remove_plan_item("rewrite", item.id) is True
    assert plan_store.remove_plan_item("rewrite", item.id) is False


def test_next_pending_resolves_dependencies(plan_base):
    a = plan_store.add_plan_item("rewrite", "First")
    b = plan_store.add_plan_item("rewrite", "Second", depends_on=[a.id])
    plan = plan_store.load_plan("rewrite")
    runnable = plan_store.next_pending_items(plan)
    assert [item.id for item in runnable] == [a.id]
    plan_store.update_plan_item("rewrite", a.id, status="completed")
    plan = plan_store.load_plan("rewrite")
    runnable = plan_store.next_pending_items(plan)
    assert [item.id for item in runnable] == [b.id]


def test_plan_tools_round_trip(plan_base):
    add = PlanAddTool()
    item = json.loads(
        add._run(name="rewrite", title="Hook task runtime", idempotency_key="hook")
    )
    assert item["status"] == "ok"

    listing = json.loads(PlanListTool()._run(name="rewrite"))
    assert len(listing["items"]) == 1
    assert "hook" in [it["idempotency_key"] for it in listing["items"]]

    updater = PlanUpdateTool()
    update = json.loads(
        updater._run(name="rewrite", item_id=item["item"]["id"], status="completed")
    )
    assert update["status"] == "ok"

    remover = PlanRemoveTool()
    removal = json.loads(remover._run(name="rewrite", item_id=item["item"]["id"]))
    assert removal["status"] == "ok"

    listing = json.loads(PlanListTool()._run(name="rewrite"))
    assert listing["items"] == []


def test_the_mini_plan_is_not_in_the_developer_toolbox():
    """The store works, and stays unbound on purpose.

    Nothing in the engine, the prompt, or the review reads a mini-plan back,
    so binding these would give the model a second way to "manage a plan"
    whose writes never reach the run. `add_step` is the one that steers it.
    Rebinding is a line in `tools/__init__.py:META_TOOLS` — this test is the
    reminder that a consumer has to exist first.
    """
    from infinidev.tools import get_tools_for_role

    names = {t.name for t in get_tools_for_role("developer")}
    assert not names & {"plan_add", "plan_list", "plan_update", "plan_remove"}
    assert "add_step" in names
