"""Regression tests for VerificationEngine._file_to_module path handling.

Found via end-to-end run: an ABSOLUTE changed-file path was turned into a
garbage dotted module ("/tmp/x/calc.py" -> ".tmp.x.calc"), so the import
check failed spuriously and triggered an unnecessary developer re-run.
"""

from __future__ import annotations

from unittest.mock import patch

from infinidev.engine.analysis.verification_engine import VerificationEngine


class TestFileToModule:
    def setup_method(self):
        self.eng = VerificationEngine(workspace="/work/proj")

    def test_absolute_path_relativised(self):
        # The exact bug: an absolute path inside the workspace.
        assert self.eng._file_to_module("/work/proj/calc.py") == "calc"

    def test_absolute_nested_path(self):
        assert self.eng._file_to_module("/work/proj/pkg/mod.py") == "pkg.mod"

    def test_no_leading_dot(self):
        mod = self.eng._file_to_module("/work/proj/calc.py")
        assert mod is not None and not mod.startswith(".")

    def test_relative_path_unchanged(self):
        assert self.eng._file_to_module("pkg/mod.py") == "pkg.mod"

    def test_src_prefix_stripped(self):
        assert self.eng._file_to_module("src/pkg/mod.py") == "pkg.mod"
        assert self.eng._file_to_module("/work/proj/src/pkg/mod.py") == "pkg.mod"

    def test_outside_workspace_skipped(self):
        # A path outside the workspace can't be imported from here.
        assert self.eng._file_to_module("/somewhere/else/mod.py") is None

    def test_test_and_dunder_files_skipped(self):
        assert self.eng._file_to_module("/work/proj/test_calc.py") is None
        assert self.eng._file_to_module("/work/proj/conftest.py") is None
        assert self.eng._file_to_module("/work/proj/pkg/__init__.py") is None

    def test_non_python_skipped(self):
        assert self.eng._file_to_module("/work/proj/readme.md") is None

    def test_invalid_module_syntax_is_skipped(self):
        path = "/work/proj/x;__import__('os').system('false').py"
        assert self.eng._file_to_module(path) is None


def test_permission_denial_prevents_verification_command(tmp_path):
    marker = tmp_path / "should-not-exist"
    engine = VerificationEngine(workspace=str(tmp_path))

    with patch(
        "infinidev.tools.shell.execute_command_tool.check_command_permission",
        return_value="Command denied by user",
    ):
        result = engine._run(["touch", str(marker)])

    assert result["exit_code"] == -1
    assert "denied" in result["output"].lower()
    assert not marker.exists()
