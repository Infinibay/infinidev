"""Durable mini-plan storage for the rewrite session.

A ``MiniPlan`` is a small list of tasks the harness keeps visible so the
user (or a future prompt) can resume work without losing track. Items
have explicit lifecycle states and an ``idempotency_key`` so re-running
the tool with the same title is a no-op rather than a duplicate.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PlanItem:
    """A single plan step with optional dependencies and result notes."""

    id: str
    title: str
    status: str = "pending"
    depends_on: list[str] = field(default_factory=list)
    notes: str = ""
    idempotency_key: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MiniPlan:
    """A small plan file persisted under ``.infinidev/plans``."""

    name: str
    items: list[PlanItem] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "items": [item.to_dict() for item in self.items],
        }


def _plans_dir(base: Path | None = None) -> Path:
    root = base or Path(os.environ.get("INFINIDEV_BASE_DIR", ".infinidev"))
    directory = root / "plans"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, data: dict) -> None:
    """Write *data* to *path* atomically so a crash never leaves a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=str(path.parent), delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(json.dumps(data, indent=2, ensure_ascii=False))
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def load_plan(name: str, base: Path | None = None) -> MiniPlan:
    """Load a plan by *name* or return a fresh empty plan if it doesn't exist."""
    path = _plans_dir(base) / f"{name}.json"
    if not path.exists():
        return MiniPlan(name=name, created_at=_now(), updated_at=_now())
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Plan %s is not valid JSON; starting fresh", path)
        return MiniPlan(name=name, created_at=_now(), updated_at=_now())
    items = [
        PlanItem(
            id=str(entry.get("id", str(uuid.uuid4()))),
            title=str(entry.get("title", "")),
            status=str(entry.get("status", "pending")),
            depends_on=list(entry.get("depends_on", []) or []),
            notes=str(entry.get("notes", "")),
            idempotency_key=str(entry.get("idempotency_key", "")),
            created_at=str(entry.get("created_at", "")),
            updated_at=str(entry.get("updated_at", "")),
        )
        for entry in data.get("items", [])
    ]
    return MiniPlan(
        name=name,
        items=items,
        created_at=str(data.get("created_at", "")),
        updated_at=str(data.get("updated_at", "")),
    )


def save_plan(plan: MiniPlan, base: Path | None = None) -> Path:
    """Persist *plan* and return the resolved path."""
    plan.updated_at = _now()
    path = _plans_dir(base) / f"{plan.name}.json"
    _atomic_write(path, plan.to_dict())
    return path


def add_plan_item(
    name: str,
    title: str,
    *,
    depends_on: list[str] | None = None,
    idempotency_key: str = "",
    base: Path | None = None,
) -> PlanItem:
    """Add a new item to *plan* unless the idempotency key already exists."""
    plan = load_plan(name, base)
    if idempotency_key:
        for item in plan.items:
            if item.idempotency_key == idempotency_key:
                return item
    item = PlanItem(
        id=str(uuid.uuid4()),
        title=title.strip(),
        depends_on=list(depends_on or []),
        idempotency_key=idempotency_key,
        created_at=_now(),
        updated_at=_now(),
    )
    plan.items.append(item)
    save_plan(plan, base)
    return item


def update_plan_item(
    name: str,
    item_id: str,
    *,
    status: str | None = None,
    notes: str | None = None,
    base: Path | None = None,
) -> PlanItem | None:
    """Update an existing item; returns ``None`` if not found."""
    plan = load_plan(name, base)
    for item in plan.items:
        if item.id == item_id:
            if status is not None:
                item.status = status
            if notes is not None:
                item.notes = notes
            item.updated_at = _now()
            save_plan(plan, base)
            return item
    return None


def remove_plan_item(name: str, item_id: str, base: Path | None = None) -> bool:
    """Remove an item by id; returns ``True`` if removed."""
    plan = load_plan(name, base)
    before = len(plan.items)
    plan.items = [item for item in plan.items if item.id != item_id]
    if len(plan.items) == before:
        return False
    save_plan(plan, base)
    return True


def next_pending_items(plan: MiniPlan, *, limit: int = 10) -> list[PlanItem]:
    """Return up to *limit* runnable items (pending + dependencies satisfied)."""
    completed = {item.id for item in plan.items if item.status == "completed"}
    pending = [
        item
        for item in plan.items
        if item.status == "pending" and all(dep in completed for dep in item.depends_on)
    ]
    return pending[:limit]


def iter_items(plan: MiniPlan) -> Iterable[PlanItem]:
    return iter(plan.items)
