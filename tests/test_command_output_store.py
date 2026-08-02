"""Security and durability contracts for private command-output storage."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import time
from pathlib import Path

import pytest

from infinidev.engine.command_output_store import (
    CommandOutputHandle,
    CommandOutputIntegrityError,
    CommandOutputQuotaError,
    CommandOutputStore,
    CommandOutputStoreConfig,
)


def _config(
    *,
    artifact: int = 100_000,
    session: int = 200_000,
    project: int = 400_000,
    retention: int = 3600,
    grace: int = 60,
) -> CommandOutputStoreConfig:
    return CommandOutputStoreConfig(
        max_artifact_bytes=artifact,
        max_session_bytes=session,
        max_project_bytes=project,
        timeout_seconds=5,
        retention_seconds=retention,
        sweep_grace_seconds=grace,
    )


@pytest.fixture
def store(tmp_path, temp_db):
    return CommandOutputStore(
        root=tmp_path / "private" / "command_output",
        db_path=temp_db,
        config=_config(),
    )


def _row(db_path: str, artifact_id: int) -> sqlite3.Row:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
        ).fetchone()
        assert row is not None
        return row
    finally:
        conn.close()


def _storage_id(db_path: str, artifact_id: int) -> str:
    """Extract the random identity from the path-free, digest-bound reference."""
    return str(_row(db_path, artifact_id)["file_path"]).split(".", 1)[0]


def test_store_round_trip_catalog_is_opaque_and_private(store, temp_db):
    secret = "API_TOKEN=super-secret-value\n" + "π" * 200
    handles = store.store_streams(
        project_id=1,
        session_id="session-a",
        streams={"stdout": secret},
    )
    handle = handles["stdout"]

    assert store.read_text(handle, project_id=1, session_id="session-a") == secret
    row = _row(temp_db, handle.artifact_id)
    assert row["type"] == "command_output"
    assert row["name"] is None
    assert row["description"] is None
    assert row["content"] is None
    assert row["file_path"] and "/" not in row["file_path"]
    assert "secret" not in row["file_path"]

    conn = sqlite3.connect(temp_db)
    try:
        fts = conn.execute(
            "SELECT name, description, content FROM artifacts_fts WHERE rowid = ?",
            (handle.artifact_id,),
        ).fetchone()
        assert fts == (None, None, None)
        serialized = "\n".join(
            str(value or "")
            for value in conn.execute(
                "SELECT name, file_path, description, content FROM artifacts"
            ).fetchone()
        )
        assert "super-secret-value" not in serialized
    finally:
        conn.close()

    assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
    for path in store.root.iterdir():
        assert stat.S_IMODE(path.lstat().st_mode) == 0o600


def test_equal_text_gets_random_distinct_occurrences(store):
    first = store.store_streams(
        project_id=1, session_id="s", streams={"stdout": "same"}
    )["stdout"]
    second = store.store_streams(
        project_id=1, session_id="s", streams={"stdout": "same"}
    )["stdout"]
    assert first.artifact_id != second.artifact_id


def test_cross_session_and_project_handles_fail_closed(store):
    handle = store.store_streams(
        project_id=1, session_id="owner", streams={"stdout": "private"}
    )["stdout"]
    with pytest.raises(CommandOutputIntegrityError, match="scope"):
        store.read_text(handle, project_id=1, session_id="other")
    with pytest.raises(CommandOutputIntegrityError, match="scope"):
        store.read_text(handle, project_id=2, session_id="owner")


def test_catalog_traversal_and_wrong_type_fail_closed(store, temp_db):
    handle = store.store_streams(
        project_id=1, session_id="s", streams={"stdout": "private"}
    )["stdout"]
    conn = sqlite3.connect(temp_db)
    conn.execute(
        "UPDATE artifacts SET file_path = '../../etc/passwd' WHERE id = ?",
        (handle.artifact_id,),
    )
    conn.commit()
    conn.close()
    with pytest.raises(CommandOutputIntegrityError, match="storage reference"):
        store.read_text(handle, project_id=1, session_id="s")

    wrong = CommandOutputHandle(
        artifact_id=handle.artifact_id,
        artifact_type="report",
        stream="stdout",
        char_count=7,
        byte_count=7,
    )
    with pytest.raises(CommandOutputIntegrityError, match="invalid"):
        store.read_text(wrong, project_id=1, session_id="s")


def test_blob_symlink_corruption_and_substitution_fail_closed(store, temp_db, tmp_path):
    handle = store.store_streams(
        project_id=1, session_id="s", streams={"stdout": "original"}
    )["stdout"]
    storage_id = _storage_id(temp_db, handle.artifact_id)
    blob = store.root / f"{storage_id}.blob"
    outside = tmp_path / "outside"
    outside.write_text("do-not-read")
    blob.unlink()
    blob.symlink_to(outside)
    with pytest.raises(CommandOutputIntegrityError, match="unavailable"):
        store.read_text(handle, project_id=1, session_id="s")

    blob.unlink()
    blob.write_text("same-len")
    os.chmod(blob, 0o600)
    with pytest.raises(CommandOutputIntegrityError, match="hash"):
        store.read_text(handle, project_id=1, session_id="s")


def test_sidecar_corruption_and_unsafe_catalog_metadata_fail_closed(store, temp_db):
    handle = store.store_streams(
        project_id=1, session_id="s", streams={"stdout": "original"}
    )["stdout"]
    storage_id = _storage_id(temp_db, handle.artifact_id)
    sidecar = store.root / f"{storage_id}.json"
    sidecar.write_text("not-json")
    os.chmod(sidecar, 0o600)
    with pytest.raises(CommandOutputIntegrityError, match="sidecar"):
        store.read_text(handle, project_id=1, session_id="s")

    # A seeded secret in indexed catalog metadata is rejected rather than read.
    conn = sqlite3.connect(temp_db)
    conn.execute(
        "UPDATE artifacts SET description = 'SECRET_SEEDED' WHERE id = ?",
        (handle.artifact_id,),
    )
    conn.commit()
    conn.close()
    with pytest.raises(CommandOutputIntegrityError, match="metadata"):
        store.read_text(handle, project_id=1, session_id="s")


def test_valid_looking_sidecar_replacement_fails_digest_binding(store, temp_db):
    handle = store.store_streams(
        project_id=1, session_id="s", streams={"stdout": "original"}
    )["stdout"]
    storage_id = _storage_id(temp_db, handle.artifact_id)
    sidecar = store.root / f"{storage_id}.json"
    metadata = json.loads(sidecar.read_text())
    metadata["session_id"] = "s"
    metadata["created_at"] += 1
    sidecar.write_text(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    os.chmod(sidecar, 0o600)

    with pytest.raises(CommandOutputIntegrityError, match="substitution"):
        store.read_text(handle, project_id=1, session_id="s")


def test_per_artifact_session_and_project_quotas(tmp_path, temp_db):
    too_small = CommandOutputStore(
        root=tmp_path / "artifact",
        db_path=temp_db,
        config=_config(artifact=4, session=20, project=40),
    )
    with pytest.raises(CommandOutputQuotaError, match="per-artifact"):
        too_small.store_streams(
            project_id=1, session_id="s", streams={"stdout": "12345"}
        )

    session_limited = CommandOutputStore(
        root=tmp_path / "session",
        db_path=temp_db,
        config=_config(artifact=10, session=10, project=40),
    )
    session_limited.store_streams(
        project_id=1, session_id="session-q", streams={"stdout": "123456"}
    )
    with pytest.raises(CommandOutputQuotaError, match="session"):
        session_limited.store_streams(
            project_id=1, session_id="session-q", streams={"stderr": "12345"}
        )

    # Keep one physical root per catalog. Production has exactly one fixed
    # root; switching roots against one DB intentionally fails integrity.
    project_limited = CommandOutputStore(
        root=tmp_path / "session",
        db_path=temp_db,
        config=_config(artifact=10, session=10, project=15),
    )
    project_limited.store_streams(
        project_id=1, session_id="p1", streams={"stdout": "123456"}
    )
    with pytest.raises(CommandOutputQuotaError, match="project"):
        project_limited.store_streams(
            project_id=1, session_id="p2", streams={"stdout": "12345"}
        )


def test_partial_write_failure_leaves_no_catalog_or_managed_files(
    store, temp_db, monkeypatch
):
    real_atomic_write = store._atomic_write
    calls = 0

    def fail_second(path: Path, payload: bytes, deadline: float) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated crash before sidecar rename")
        real_atomic_write(path, payload, deadline)

    monkeypatch.setattr(store, "_atomic_write", fail_second)
    with pytest.raises(OSError, match="simulated crash"):
        store.store_streams(
            project_id=1, session_id="s", streams={"stdout": "output"}
        )

    conn = sqlite3.connect(temp_db)
    try:
        assert conn.execute(
            "SELECT count(*) FROM artifacts WHERE type = 'command_output'"
        ).fetchone()[0] == 0
    finally:
        conn.close()
    assert not list(store.root.glob("*.blob"))
    assert not list(store.root.glob("*.json"))


def test_db_insert_failure_removes_private_files_and_announces_no_artifact(
    store, temp_db, monkeypatch
):
    import infinidev.engine.command_output_store as store_module

    real_execute = store_module.execute_with_retry

    def fail_insert(operation, *args, **kwargs):
        if getattr(operation, "__name__", "") == "_insert":
            raise sqlite3.OperationalError("simulated catalog failure")
        return real_execute(operation, *args, **kwargs)

    monkeypatch.setattr(store_module, "execute_with_retry", fail_insert)
    with pytest.raises(sqlite3.OperationalError, match="catalog failure"):
        store.store_streams(
            project_id=1, session_id="s", streams={"stdout": "private"}
        )

    conn = sqlite3.connect(temp_db)
    try:
        assert conn.execute(
            "SELECT count(*) FROM artifacts WHERE type = 'command_output'"
        ).fetchone()[0] == 0
    finally:
        conn.close()
    assert not list(store.root.glob("*.blob"))
    assert not list(store.root.glob("*.json"))


def test_sweep_removes_expired_catalog_and_old_orphans_but_not_foreign_files(
    tmp_path, temp_db
):
    store = CommandOutputStore(
        root=tmp_path / "private" / "command_output",
        db_path=temp_db,
        config=_config(retention=1, grace=1),
    )
    handle = store.store_streams(
        project_id=1, session_id="s", streams={"stdout": "old"}
    )["stdout"]
    storage_id = _storage_id(temp_db, handle.artifact_id)
    sidecar = store.root / f"{storage_id}.json"
    metadata = json.loads(sidecar.read_text())
    metadata["created_at"] = int(time.time()) - 10
    sidecar_bytes = json.dumps(
        metadata, sort_keys=True, separators=(",", ":")
    ).encode()
    sidecar.write_bytes(sidecar_bytes)
    os.chmod(sidecar, 0o600)
    # Retention is based on a catalog-bound sidecar timestamp. Simulate a
    # legitimately old artifact by updating both parts of that opaque binding.
    import hashlib

    conn = sqlite3.connect(temp_db)
    conn.execute(
        "UPDATE artifacts SET file_path = ? WHERE id = ?",
        (
            f"{storage_id}.{hashlib.sha256(sidecar_bytes).hexdigest()}",
            handle.artifact_id,
        ),
    )
    conn.commit()
    conn.close()

    orphan = store.root / ("a" * 48 + ".blob")
    orphan.write_text("orphan")
    os.chmod(orphan, 0o600)
    temp = store.root / (".tmp-" + "b" * 48)
    temp.write_text("partial")
    os.chmod(temp, 0o600)
    foreign = store.root / "keep-me.txt"
    foreign.write_text("foreign")
    os.chmod(foreign, 0o600)
    old = time.time() - 10
    os.utime(orphan, (old, old))
    os.utime(temp, (old, old))

    store.sweep()
    conn = sqlite3.connect(temp_db)
    try:
        assert conn.execute(
            "SELECT count(*) FROM artifacts WHERE id = ?", (handle.artifact_id,)
        ).fetchone()[0] == 0
    finally:
        conn.close()
    assert not (store.root / f"{storage_id}.blob").exists()
    assert not sidecar.exists()
    assert not orphan.exists()
    assert not temp.exists()
    assert foreign.exists()


def test_symlink_store_root_is_rejected(tmp_path, temp_db):
    target = tmp_path / "target"
    target.mkdir()
    root = tmp_path / "linked-root"
    root.symlink_to(target, target_is_directory=True)
    store = CommandOutputStore(root=root, db_path=temp_db, config=_config())
    with pytest.raises(CommandOutputIntegrityError, match="directory"):
        store.store_streams(
            project_id=1, session_id="s", streams={"stdout": "private"}
        )


def test_sweep_removes_old_dangling_catalog_row_after_grace(store, temp_db):
    conn = sqlite3.connect(temp_db)
    cursor = conn.execute(
        "INSERT INTO artifacts "
        "(project_id, session_id, type, file_path, content, created_at) "
        "VALUES (1, 's', 'command_output', ?, NULL, datetime('now', '-2 hours'))",
        (f"{'c' * 48}.{'d' * 64}",),
    )
    artifact_id = int(cursor.lastrowid)
    conn.commit()
    conn.close()

    store.sweep()

    conn = sqlite3.connect(temp_db)
    try:
        assert conn.execute(
            "SELECT count(*) FROM artifacts WHERE id = ?", (artifact_id,)
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_sweep_keeps_recent_dangling_catalog_row_during_grace(store, temp_db):
    conn = sqlite3.connect(temp_db)
    cursor = conn.execute(
        "INSERT INTO artifacts "
        "(project_id, session_id, type, file_path, content) "
        "VALUES (1, 's', 'command_output', ?, NULL)",
        (f"{'e' * 48}.{'f' * 64}",),
    )
    artifact_id = int(cursor.lastrowid)
    conn.commit()
    conn.close()

    store.sweep()

    conn = sqlite3.connect(temp_db)
    try:
        assert conn.execute(
            "SELECT count(*) FROM artifacts WHERE id = ?", (artifact_id,)
        ).fetchone()[0] == 1
    finally:
        conn.close()
