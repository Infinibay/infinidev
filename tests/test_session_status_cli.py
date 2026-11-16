"""The status entry point remains usable without agent bootstrap or a database."""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys


def test_status_route_imports_no_agent_or_database(tmp_path):
    source = '''
import importlib.abc
import sys
class RejectHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"infinidev.cli.main", "infinidev.config.settings"} or fullname.startswith((
            "infinidev.db", "infinidev.server", "litellm", "textual")):
            raise AssertionError("unexpected bootstrap: " + fullname)
sys.meta_path.insert(0, RejectHeavy())
from infinidev.cli.entry import main
sys.argv = ["infinidev", "status", "--workspace", sys.argv[1], "--session", "nonexistent"]
main()
'''
    result = subprocess.run([sys.executable, "-c", source, str(tmp_path)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["state"] == "unknown"
    assert not (tmp_path / ".infinidev").exists()


def test_status_module_requires_explicit_workspace_and_session():
    result = subprocess.run([sys.executable, "-m", "infinidev.cli.status"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert "--workspace" in result.stderr and "--session" in result.stderr


def test_deeply_nested_record_is_unknown_in_standalone_reader(tmp_path):
    bucket = tmp_path / ".infinidev/session-status/v1" / hashlib.sha256(b"session").hexdigest()
    bucket.mkdir(parents=True)
    (bucket / ("a" * 32 + ".json")).write_bytes(b"[" * 2000 + b"]" * 2000)
    result = subprocess.run([sys.executable, "-m", "infinidev.cli.status", "--workspace",
                             str(tmp_path), "--session", "session"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["state"] == "unknown"


def test_tool_results_opt_in_requires_explicit_run(tmp_path):
    result = subprocess.run([sys.executable, '-m', 'infinidev.cli.status', '--workspace',
                             str(tmp_path), '--session', 'session', '--tool-results'],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert '--tool-results requires --run' in result.stderr


def test_tool_results_opt_in_does_not_bootstrap_settings_or_model(tmp_path):
    import sqlite3

    directory = tmp_path / '.infinidev'
    directory.mkdir()
    conn = sqlite3.connect(directory / 'infinidev.db')
    conn.execute('CREATE TABLE working_memory '
                 '(session_id TEXT, kind TEXT, content TEXT, created_at TEXT)')
    conn.close()
    source = '''
import importlib.abc
import sys
class RejectHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'infinidev.cli.main','infinidev.config.settings'} or fullname.startswith((
            'infinidev.db','infinidev.server','litellm','textual','infinidev.tools')):
            raise AssertionError('unexpected bootstrap: '+fullname)
sys.meta_path.insert(0, RejectHeavy())
from infinidev.cli.entry import main
sys.argv = ['infinidev','status','--workspace',sys.argv[1],'--session','session',
            '--run','a'*32,'--tool-results']
main()
'''
    result = subprocess.run([sys.executable, '-c', source, str(tmp_path)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data['tool_results']['state'] == 'unavailable'
    assert data['state'] == 'unknown'
    assert sorted(path.name for path in directory.iterdir()) == ['infinidev.db']
