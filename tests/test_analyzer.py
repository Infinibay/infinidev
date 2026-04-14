"""Tests for the heuristic code analyzer."""

import os
import json

import pytest

from infinidev.code_intel.analyzer import (
    analyze_code, check_broken_imports, check_undefined_symbols,
    check_unused_imports, check_unused_definitions, check_missing_docstrings,
    check_orphaned_references,
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


# ── Missing docstrings ────────────────────────────────────────────────────────


class TestMissingDocstrings:
    def test_detects_missing_docstring_on_function(self, workspace_dir, temp_db):
        """A public function without a docstring is flagged."""
        code = '''\
def undocumented_func():
    return 42
'''
        path = workspace_dir / "nodoc.py"
        path.write_text(code)
        index_file(1, str(path))

        diags = check_missing_docstrings(1, str(path))
        names = [d.message for d in diags]
        assert any("undocumented_func" in m for m in names)

    def test_public_class_without_docstring_flagged(self, workspace_dir, temp_db):
        """A public class without a docstring is flagged."""
        code = '''\
class MyClass:
    def method(self):
        pass
'''
        path = workspace_dir / "class_nodoc.py"
        path.write_text(code)
        index_file(1, str(path))

        diags = check_missing_docstrings(1, str(path))
        names = [d.message for d in diags]
        assert any("MyClass" in m for m in names)

    def test_function_with_docstring_not_flagged(self, workspace_dir, temp_db):
        """A public function that has a docstring is NOT flagged."""
        code = '''\
def documented_func():
    """Does something useful."""
    return 42
'''
        path = workspace_dir / "hasdoc.py"
        path.write_text(code)
        index_file(1, str(path))

        diags = check_missing_docstrings(1, str(path))
        names = [d.message for d in diags]
        assert not any("documented_func" in m for m in names)

    def test_private_symbols_not_flagged(self, workspace_dir, temp_db):
        """Private functions/classes (starting with _) are skipped."""
        code = '''\
def _private():
    """Private helper."""
    return 1

class _Internal:
    pass
'''
        path = workspace_dir / "private.py"
        path.write_text(code)
        index_file(1, str(path))

        diags = check_missing_docstrings(1, str(path))
        names = [d.message for d in diags]
        assert not any("_private" in m for m in names)
        assert not any("_Internal" in m for m in names)

    def test_dunder_methods_not_flagged(self, workspace_dir, temp_db):
        """Dunder methods (__str__, __repr__, etc.) are skipped."""
        code = '''\
class MyClass:
    def __str__(self):
        return "x"

    def __init__(self):
        self.x = 1
'''
        path = workspace_dir / "dunders.py"
        path.write_text(code)
        index_file(1, str(path))

        diags = check_missing_docstrings(1, str(path))
        names = [d.message for d in diags]
        assert not any("__str__" in m for m in names)
        assert not any("__init__" in m for m in names)

    def test_project_scope_finds_all_files(self, workspace_dir, temp_db):
        """Without a file_path, check_missing_docstrings scans the whole project."""
        code = '''\
def only_in_this_file():
    return 1
'''
        path = workspace_dir / "project_nodoc.py"
        path.write_text(code)
        index_file(1, str(path))

        diags = check_missing_docstrings(1)
        names = [d.message for d in diags]
        assert any("only_in_this_file" in m for m in names)


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

# ── Orphaned references ───────────────────────────────────────────────────────


class TestOrphanedReferences:
    def test_empty_mapping_returns_no_diags(self):
        assert check_orphaned_references(1, {}) == []
        assert check_orphaned_references(1, None) == []

    def test_not_in_auto_dispatch(self):
        """orphaned_references must stay OUT of ALL_CHECKS because it needs
        a deleted_by_file mapping, not a plain file path."""
        from infinidev.code_intel.analyzer import ALL_CHECKS
        assert "orphaned_references" not in ALL_CHECKS

    def test_skips_when_same_name_defined_elsewhere(
        self, workspace_dir, temp_db,
    ):
        """If another symbol with the same simple name exists in another
        file, we assume the reference resolves there — no diagnostic."""
        # Module A defines and uses `shared_name`
        a = workspace_dir / "mod_a.py"
        a.write_text("def shared_name():\n    return 1\n")
        # Module B also defines `shared_name` and calls it
        b = workspace_dir / "mod_b.py"
        b.write_text("def shared_name():\n    return 2\n\nshared_name()\n")
        index_file(1, str(a))
        index_file(1, str(b))

        # Pretend we deleted `shared_name` from mod_a.py only
        diags = check_orphaned_references(1, {str(a): ["shared_name"]})
        # Another definition exists in mod_b.py, so skip
        assert diags == []

    def test_flags_when_fully_gone(self, workspace_dir, temp_db):
        """When the deleted symbol has no sibling definition, flag refs."""
        src = workspace_dir / "defs.py"
        src.write_text("def unique_func():\n    return 1\n")
        caller = workspace_dir / "caller.py"
        caller.write_text("from defs import unique_func\nunique_func()\n")
        index_file(1, str(src))
        index_file(1, str(caller))

        # Re-index the "source" file as if unique_func was deleted
        src.write_text("# unique_func removed\n")
        index_file(1, str(src))

        diags = check_orphaned_references(1, {str(src): ["unique_func"]})
        assert len(diags) >= 1
        assert all(d.severity == "error" for d in diags)
        assert all("caller.py" in d.file_path for d in diags)

    def test_does_not_flag_refs_inside_source_file(
        self, workspace_dir, temp_db,
    ):
        """References in the same file where the symbol was deleted are not
        flagged — the edit that removed the symbol presumably also cleaned
        up local refs, or they'll surface as syntax/import errors instead."""
        src = workspace_dir / "self_ref.py"
        src.write_text(
            "def local_fn():\n    return 1\n\nlocal_fn()\n"
        )
        index_file(1, str(src))

        diags = check_orphaned_references(1, {str(src): ["local_fn"]})
        assert diags == []
