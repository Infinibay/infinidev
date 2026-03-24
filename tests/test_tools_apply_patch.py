"""Tests for ApplyPatchTool."""

import json
import os

import pytest

from infinidev.tools.file.apply_patch import ApplyPatchTool, _extract_files_from_patch, _parse_unified_diff


class TestApplyPatch:
    """Tests for ApplyPatchTool."""

    def test_single_file_patch(self, bound_tool, workspace_dir):
        """Apply a simple single-file patch."""
        tool = bound_tool(ApplyPatchTool)
        fpath = workspace_dir / "sample.txt"

        patch = """\
--- a/sample.txt
+++ b/sample.txt
@@ -1,5 +1,5 @@
-line one
+LINE ONE
 line two
 line three
 line four
 line five
"""
        result = tool._run(patch=patch, strip=1)
        # Check it succeeded (either via patch binary or Python fallback)
        assert "error" not in result.lower() or "files_modified" in result

        content = fpath.read_text()
        assert "LINE ONE" in content
        assert "line two" in content

    def test_empty_patch(self, bound_tool):
        """Empty patch returns error."""
        tool = bound_tool(ApplyPatchTool)
        result = tool._run(patch="")
        data = json.loads(result)
        assert "error" in data

    def test_malformed_patch(self, bound_tool, workspace_dir):
        """Malformed patch returns error."""
        tool = bound_tool(ApplyPatchTool)
        result = tool._run(patch="this is not a patch")
        data = json.loads(result)
        assert "error" in data


class TestPatchHelpers:
    """Tests for patch parsing helpers."""

    def test_extract_files(self):
        """Extract file paths from a unified diff."""
        patch = """\
--- a/src/auth.py
+++ b/src/auth.py
@@ -1,3 +1,3 @@
--- a/src/utils.py
+++ b/src/utils.py
@@ -10,3 +10,3 @@
"""
        files = _extract_files_from_patch(patch, strip=1)
        assert files == ["src/auth.py", "src/utils.py"]

    def test_extract_files_strip_0(self):
        """Strip=0 keeps full path."""
        patch = "+++ b/src/auth.py\n"
        files = _extract_files_from_patch(patch, strip=0)
        assert files == ["b/src/auth.py"]

    def test_parse_unified_diff(self):
        """Parse a simple unified diff."""
        patch = """\
--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 line1
-old_line
+new_line
 line3
"""
        result = _parse_unified_diff(patch, strip=1)
        assert "file.py" in result
        hunks = result["file.py"]
        assert len(hunks) == 1
        assert hunks[0]["old_start"] == 1
        assert hunks[0]["old_count"] == 3
        assert "new_line" in hunks[0]["add_lines"]
        assert "old_line" not in hunks[0]["add_lines"]

    def test_parse_multi_file_diff(self):
        """Parse a diff with multiple files."""
        patch = """\
--- a/a.py
+++ b/a.py
@@ -1,1 +1,1 @@
-old_a
+new_a
--- a/b.py
+++ b/b.py
@@ -1,1 +1,1 @@
-old_b
+new_b
"""
        result = _parse_unified_diff(patch, strip=1)
        assert "a.py" in result
        assert "b.py" in result
