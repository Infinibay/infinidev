"""Tests for the heuristic code analyzer."""

import os
import json

import pytest

from infinidev.code_intel.analyzer import (
    analyze_code, check_broken_imports, check_undefined_symbols,
    check_unused_imports, check_unused_definitions,
    store_diagnostics, get_diagnostics, Diagnostic,
)
from infinidev.code_intel.indexer import index_file
from infinidev.code_intel.stdlib_modules import get_stdlib_modules
from infinidev.code_intel.import_resolver import resolve_import, ResolvedImport
from infinidev.code_intel.models import Import


# ── Stdlib modules ──────────────────────────────────────────────────────────


class TestStdlibModules:
    def test_common_modules_present(self):
        mods = get_stdlib_modules()
        for name in ["os", "sys", "json", "re", "pathlib", "typing", "collections"]:
            assert name in mods, f"{name} should be in stdlib"

    def test_not_third_party(self):
        mods = get_stdlib_modules()
        for name in ["requests", "flask", "django", "numpy", "pandas"]:
            assert name not in mods, f"{name} should NOT be in stdlib"


# ── Import resolver ─────────────────────────────────────────────────────────


class TestImportResolver:
    def test_stdlib_resolved(self):
        imp = Import(source="os", name="path", file_path="test.py", line=1, language="python")
        result = resolve_import(1, imp, "/workspace")
        assert result.status == "stdlib"

    def test_stdlib_dotted(self):
        imp = Import(source="os.path", name="join", file_path="test.py", line=1, language="python")
        result = resolve_import(1, imp, "/workspace")
        assert result.status == "stdlib"

    def test_local_module(self, workspace_dir):
        """Local module found as file in workspace."""
        (workspace_dir / "mymodule.py").write_text("def foo(): pass\n")
        imp = Import(source="mymodule", name="foo", file_path=str(workspace_dir / "main.py"), line=1, language="python")
        result = resolve_import(1, imp, str(workspace_dir))
        assert result.status == "local"

    def test_local_package(self, workspace_dir):
        """Local package found as directory with __init__.py."""
        pkg = workspace_dir / "mypkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        imp = Import(source="mypkg", name="something", file_path=str(workspace_dir / "main.py"), line=1, language="python")
        result = resolve_import(1, imp, str(workspace_dir))
        assert result.status == "local"

    def test_relative_import(self, workspace_dir):
        """Relative import resolves within directory."""
        (workspace_dir / "sibling.py").write_text("x = 1\n")
        imp = Import(
            source=".sibling", name="x",
            file_path=str(workspace_dir / "main.py"), line=1,
            language="python",
        )
        result = resolve_import(1, imp, str(workspace_dir))
        assert result.status == "local"

    def test_unresolved_import(self, workspace_dir):
        """Nonexistent module is unresolved."""
        imp = Import(
            source="nonexistent_module_xyz", name="foo",
            file_path=str(workspace_dir / "main.py"), line=1,
            language="python",
        )
        result = resolve_import(1, imp, str(workspace_dir))
        assert result.status in ("unresolved", "third_party")  # depends on installed packages


# ── Analyzer checks ─────────────────────────────────────────────────────────


@pytest.fixture
def indexed_file(workspace_dir, temp_db):
    """Create and index a Python file with known issues."""
    code = '''\
import os
import json
import nonexistent_pkg

from collections import OrderedDict

def used_function():
    result = os.path.join("a", "b")
    return result

def unused_function():
    return 42

class MyClass:
    def method(self):
        undefined_var.do_something()
        return used_function()
'''
    path = workspace_dir / "buggy.py"
    path.write_text(code)
    index_file(1, str(path))
    return str(path)


class TestBrokenImports:
    def test_detects_nonexistent_module(self, indexed_file, workspace_dir):
        diags = check_broken_imports(1, str(workspace_dir), indexed_file)
        broken_sources = [d.message for d in diags]
        # nonexistent_pkg should be flagged
        assert any("nonexistent_pkg" in m for m in broken_sources)

    def test_stdlib_not_flagged(self, indexed_file, workspace_dir):
        diags = check_broken_imports(1, str(workspace_dir), indexed_file)
        messages = " ".join(d.message for d in diags)
        assert "os" not in messages.split("'") or "Cannot resolve import 'os'" not in messages
        assert "'json'" not in messages or "Cannot resolve import 'json'" not in messages


class TestUnusedImports:
    def test_detects_unused(self, indexed_file):
        diags = check_unused_imports(1, indexed_file)
        unused_names = [d.message for d in diags]
        # json is imported but never referenced
        assert any("json" in m for m in unused_names)
        # OrderedDict is imported but never referenced
        assert any("OrderedDict" in m for m in unused_names)

    def test_used_import_not_flagged(self, indexed_file):
        diags = check_unused_imports(1, indexed_file)
        unused_names = " ".join(d.message for d in diags)
        # os is used (os.path.join)
        assert "Import 'os'" not in unused_names


class TestUnusedDefinitions:
    def test_detects_unused(self, indexed_file):
        diags = check_unused_definitions(1, indexed_file)
        names = [d.message for d in diags]
        # unused_function is defined but never called
        assert any("unused_function" in m for m in names)


class TestAnalyzeCode:
    def test_full_analysis(self, indexed_file, workspace_dir):
        report = analyze_code(1, str(workspace_dir), indexed_file)
        assert len(report.diagnostics) > 0
        assert report.scope == "file"
        # Should have stats for each check
        assert "broken_imports" in report.stats

    def test_specific_checks(self, indexed_file, workspace_dir):
        report = analyze_code(1, str(workspace_dir), indexed_file, checks=["unused_imports"])
        assert "unused_imports" in report.stats
        assert "broken_imports" not in report.stats


# ── Diagnostics persistence ─────────────────────────────────────────────────


class TestDiagnosticsPersistence:
    def test_store_and_retrieve(self, temp_db):
        diags = [
            Diagnostic("test.py", 10, "error", "broken_imports", "bad import"),
            Diagnostic("test.py", 20, "warning", "unused_imports", "unused os"),
        ]
        store_diagnostics(1, "test.py", diags)
        retrieved = get_diagnostics(1, "test.py")
        assert len(retrieved) == 2
        assert retrieved[0].severity == "error"
        assert retrieved[1].severity == "warning"

    def test_replace_old_diagnostics(self, temp_db):
        store_diagnostics(1, "test.py", [Diagnostic("test.py", 1, "error", "x", "old")])
        store_diagnostics(1, "test.py", [Diagnostic("test.py", 2, "warning", "y", "new")])
        retrieved = get_diagnostics(1, "test.py")
        assert len(retrieved) == 1
        assert retrieved[0].message == "new"
