"""Content-free observations of a bounded, explicitly selected trace window."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from infinidev.code_intel._db import execute_with_retry

RUN = 'a' * 32
OTHER_RUN = 'b' * 32
HASH = hashlib.sha256(b'empty search result').hexdigest()
OTHER_HASH = hashlib.sha256(b'different result').hexdigest()


@pytest.fixture
def trace_db(workspace_dir):
    directory = Path(workspace_dir) / '.infinidev'
    directory.mkdir(exist_ok=True)
    path = directory / 'infinidev.db'
    conn = sqlite3.connect(path)
    conn.executescript('''CREATE TABLE working_memory (
        session_id TEXT, kind TEXT, content TEXT, created_at TEXT);
        CREATE INDEX recent ON working_memory(session_id, created_at);''')
    yield Path(workspace_dir), conn, path
    conn.close()


def record(conn, event='tool_result', *, session='session', run=RUN,
           tool='code_search', digest=HASH, **extra):
    body = dict(trace='turn_provenance', run_id=run, event=event, iteration=1,
                source='dispatcher', tool=tool, output_sha256=digest, output_bytes=19)
    body.update(extra)
    conn.execute('INSERT INTO working_memory VALUES (?,?,?,?)',
                 (session, 'runtime_trace', json.dumps(body), '2026-11-16T12:00:00+00:00'))
    conn.commit()


def observe(root, run=RUN):
    from infinidev.engine.result_observation import read_tool_results
    return read_tool_results(root, 'session', run)


def test_repeated_empty_results_count_without_exporting_content(trace_db):
    root, conn, _ = trace_db
    record(conn, digest=OTHER_HASH)
    for _ in range(306):
        record(conn, arguments={'secret': 'PRIVATE_ARGUMENT'}, content='PRIVATE_BODY',
               reasoning_content='PRIVATE_REASONING')
    result = observe(root)
    assert result['state'] == 'observed'
    assert result['results']['count'] == 307
    assert result['results']['latest_occurrences'] == 306
    assert result['results']['trailing_identical'] == 306
    assert result['results']['last_different']['output_sha256'] == OTHER_HASH
    assert result['results']['by_tool'] == [dict(tool='code_search', count=307,
                                               distinct_output_hashes=2)]
    assert 'PRIVATE_' not in json.dumps(result)
    assert result['coverage']['scope'] == 'retained_window'
    assert result['coverage']['history_complete'] is False


def test_interleaved_tools_break_streak_but_model_messages_do_not(trace_db):
    root, conn, _ = trace_db
    record(conn)
    record(conn, tool='read_file')
    record(conn)
    record(conn, 'model_response', source='model')
    record(conn)
    result = observe(root)['results']
    assert result['trailing_identical'] == 2
    assert result['latest_occurrences'] == 3
    assert result['last_different']['tool'] == 'read_file'


def test_session_run_and_kind_are_exact_not_global(trace_db):
    root, conn, _ = trace_db
    record(conn)
    record(conn, session='other')
    record(conn, run=OTHER_RUN)
    conn.execute('INSERT INTO working_memory VALUES (?,?,?,?)',
                 ('session', 'reasoning', 'PRIVATE_NOT_JSON', '2026-11-16'))
    conn.commit()
    assert observe(root)['results']['count'] == 1
    assert observe(root, 'c' * 32)['reason'] == 'trace_not_found'


def test_closed_child_is_metadata_not_session_idle(trace_db):
    root, conn, _ = trace_db
    record(conn)
    record(conn, 'turn_closed', status='cancelled', closure='interrupted', dispatches=1)
    result = observe(root)
    assert result['closure']['status'] == 'cancelled'
    assert result['closure']['dispatches'] == 1
    assert 'idle' not in json.dumps(result)


@pytest.mark.parametrize('change', [dict(output_sha256='bad'), dict(output_bytes=-1),
                                   dict(source='model'), dict(tool='x' * 500)])
def test_invalid_selected_result_is_unavailable_not_zero(trace_db, change):
    root, conn, _ = trace_db
    record(conn, **change)
    result = observe(root)
    assert result['state'] == 'unavailable'
    assert result['reason'] == 'invalid_trace'
    assert result['results'] is None


def test_row_and_result_windows_report_clipping(trace_db, monkeypatch):
    from infinidev.engine import result_observation as module
    monkeypatch.setattr(module, '_SCAN_ROWS', 5)
    monkeypatch.setattr(module, '_RESULT_ROWS', 3)
    root, conn, _ = trace_db
    for _ in range(9):
        record(conn)
    result = observe(root)
    assert result['results']['count'] == 3
    assert result['coverage']['scan_clipped'] is True
    assert result['coverage']['results_clipped'] is True
    assert result['results']['trailing_identical'] == 3
    assert result['results']['last_different'] is None


@pytest.mark.parametrize('body', ['{bad json', '{"content":"' + 'x' * 18000 + '"}'],
                         ids=['invalid-json', 'oversized'])
def test_bad_or_oversized_trace_fails_closed(trace_db, body):
    root, conn, _ = trace_db
    conn.execute('INSERT INTO working_memory VALUES (?,?,?,?)',
                 ('session', 'runtime_trace', body, '2026-11-16'))
    conn.commit()
    assert observe(root)['state'] == 'unavailable'


def test_missing_database_does_not_create_state(tmp_path):
    result = observe(tmp_path)
    assert result['reason'] == 'missing_database'
    assert not (tmp_path / '.infinidev').exists()


def test_old_schema_is_unavailable_and_unchanged(trace_db):
    root, conn, path = trace_db
    conn.execute('DROP TABLE working_memory'); conn.commit()
    before = path.read_bytes()
    assert observe(root)['state'] == 'unavailable'
    assert path.read_bytes() == before


def test_symlink_database_is_rejected(trace_db, tmp_path):
    root, conn, path = trace_db
    outside = root / 'outside.db'
    conn.close(); path.rename(outside); path.symlink_to(outside)
    assert observe(root)['reason'] == 'unsafe_database'


def test_sql_budget_exhaustion_is_unavailable(trace_db, monkeypatch):
    from infinidev.code_intel import _db
    root, conn, _ = trace_db
    for _ in range(50):
        record(conn)
    monkeypatch.setattr(_db, '_READ_ONLY_STEPS', 1)
    assert observe(root)['state'] == 'unavailable'


def test_readonly_shared_helper_sees_wal_and_rejects_writes(trace_db):
    _, conn, path = trace_db
    conn.execute('PRAGMA journal_mode=WAL')
    record(conn)
    found = execute_with_retry(lambda c: c.execute('SELECT count(*) FROM working_memory')
                              .fetchone()[0], db_path=str(path), read_only=True)
    assert found == 1
    with pytest.raises(sqlite3.OperationalError):
        execute_with_retry(lambda c: c.execute('DELETE FROM working_memory'),
                           db_path=str(path), read_only=True)
    assert conn.execute('SELECT count(*) FROM working_memory').fetchone()[0] == 1


@pytest.mark.parametrize('change', [dict(output_bytes=True), dict(iteration=False),
                                   dict(tool={'private': 'PRIVATE_BODY'})])
def test_wrong_json_primitive_types_are_unavailable(trace_db, change):
    root, conn, _ = trace_db
    record(conn, **change)
    assert observe(root)['state'] == 'unavailable'


def test_expired_read_budget_does_not_return_measurements(trace_db, monkeypatch):
    from infinidev.code_intel import _db
    _, _, path = trace_db
    monkeypatch.setattr(_db, '_READ_ONLY_SECONDS', -1)
    with pytest.raises(sqlite3.OperationalError):
        execute_with_retry(lambda conn: conn.execute('SELECT 1').fetchone(),
                           db_path=str(path), read_only=True)


def test_trace_with_no_results_has_only_a_window_zero(trace_db):
    root, conn, _ = trace_db
    record(conn, 'turn_setup')
    result = observe(root)
    assert result['results']['count'] == 0
    assert result['results']['latest'] is None
    assert result['coverage']['history_complete'] is False


def test_persisted_order_does_not_depend_on_event_timestamp(trace_db, monkeypatch):
    from infinidev.engine import result_observation as module
    root, conn, _ = trace_db
    monkeypatch.setattr(module, '_SCAN_ROWS', 1)
    record(conn, digest=OTHER_HASH)
    conn.execute("UPDATE working_memory SET created_at='2099-01-01'"); conn.commit()
    record(conn)
    assert observe(root)['results']['latest']['output_sha256'] == HASH


def test_busy_database_is_unavailable_without_retry_or_writes(trace_db):
    root, conn, path = trace_db
    record(conn)
    conn.execute('BEGIN EXCLUSIVE')
    try:
        assert observe(root)['state'] == 'unavailable'
    finally:
        conn.rollback()
    assert conn.execute('SELECT count(*) FROM working_memory').fetchone()[0] == 1


@pytest.mark.parametrize('event', [None, {}, ''])
def test_selected_trace_requires_an_event_label(trace_db, event):
    root, conn, _ = trace_db
    record(conn, event)
    assert observe(root)['state'] == 'unavailable'
