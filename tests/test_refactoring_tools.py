"""Tests for rename_symbol and move_symbol refactoring tools."""

import json
import os

import pytest

from infinidev.code_intel.indexer import index_file
from infinidev.tools.code_intel.rename_symbol import RenameSymbolTool
from infinidev.tools.code_intel.move_symbol import MoveSymbolTool


@pytest.fixture
def python_project(workspace_dir, temp_db):
    """Create a mini Python project with multiple files for refactoring tests."""
    # Main module
    main_py = workspace_dir / "main.py"
    main_py.write_text(
        "from utils import helper_func\n"
        "\n"
        "def run():\n"
        "    result = helper_func(42)\n"
        "    return result\n"
    )

    # Utils module
    utils_py = workspace_dir / "utils.py"
    utils_py.write_text(
        "def helper_func(x):\n"
        "    return x * 2\n"
        "\n"
        "def other_func():\n"
        "    return helper_func(10)\n"
    )

    # Class module
    models_py = workspace_dir / "models.py"
    models_py.write_text(
        "class User:\n"
        "    def __init__(self, name):\n"
        "        self.name = name\n"
        "\n"
        "    def greet(self):\n"
        "        return f'Hello, {self.name}'\n"
        "\n"
        "    def validate(self):\n"
        "        return bool(self.name)\n"
    )

    # Index all files
    index_file(1, str(main_py))
    index_file(1, str(utils_py))
    index_file(1, str(models_py))

    return workspace_dir


# ── rename_symbol ────────────────────────────────────────────────────────────


class TestRenameSymbol:
    def test_rename_function(self, bound_tool, python_project):
        """Renames a function and updates references."""
        tool = bound_tool(RenameSymbolTool)
        result = tool._run(
            symbol="helper_func",
            new_name="compute",
            file_path=str(python_project / "utils.py"),
        )
        data = json.loads(result)
        assert data["old_name"] == "helper_func"
        assert data["new_name"] == "compute"
        assert data["total_replacements"] > 0

        # Check definition was renamed
        with open(python_project / "utils.py") as f:
            content = f.read()
        assert "def compute(x):" in content
        assert "def helper_func" not in content

        # Check reference in same file was updated
        assert "return compute(10)" in content

    def test_rename_updates_other_files(self, bound_tool, python_project):
        """Rename updates references in other files."""
        tool = bound_tool(RenameSymbolTool)
        tool._run(
            symbol="helper_func",
            new_name="compute",
            file_path=str(python_project / "utils.py"),
        )

        with open(python_project / "main.py") as f:
            content = f.read()
        assert "compute" in content

    def test_rename_method(self, bound_tool, python_project):
        """Renames a class method."""
        tool = bound_tool(RenameSymbolTool)
        result = tool._run(
            symbol="User.greet",
            new_name="say_hello",
            file_path=str(python_project / "models.py"),
        )
        data = json.loads(result)
        assert data["total_replacements"] >= 1

        with open(python_project / "models.py") as f:
            content = f.read()
        assert "def say_hello(self):" in content
        assert "def greet" not in content

    def test_rename_invalid_identifier(self, bound_tool, python_project):
        """Rejects invalid Python identifiers."""
        tool = bound_tool(RenameSymbolTool)
        result = tool._run(symbol="helper_func", new_name="123bad")
        data = json.loads(result)
        assert "error" in data

    def test_rename_same_name(self, bound_tool, python_project):
        """Rejects renaming to the same name."""
        tool = bound_tool(RenameSymbolTool)
        result = tool._run(
            symbol="helper_func",
            new_name="helper_func",
            file_path=str(python_project / "utils.py"),
        )
        data = json.loads(result)
        assert "error" in data
        assert "already named" in data["error"]


# ── move_symbol ──────────────────────────────────────────────────────────────


class TestMoveSymbol:
    def test_move_function_to_new_file(self, bound_tool, python_project):
        """Moves a function to another file."""
        tool = bound_tool(MoveSymbolTool)
        target = str(python_project / "helpers.py")

        # Create target file first
        with open(target, "w") as f:
            f.write("# Helper functions\n")
        index_file(1, target)

        result = tool._run(
            symbol="helper_func",
            target_file=target,
        )
        data = json.loads(result)
        assert data["lines_moved"] >= 2

        # Check function is in target
        with open(target) as f:
            content = f.read()
        assert "def helper_func" in content

        # Check function is removed from source
        with open(python_project / "utils.py") as f:
            content = f.read()
        assert "def helper_func" not in content
        assert "def other_func" in content  # other function stays

    def test_move_method_to_class(self, bound_tool, python_project):
        """Moves a method into a different class."""
        # Create a target class
        service_py = python_project / "service.py"
        service_py.write_text(
            "class UserService:\n"
            "    def __init__(self):\n"
            "        pass\n"
        )
        index_file(1, str(service_py))

        tool = bound_tool(MoveSymbolTool)
        result = tool._run(
            symbol="User.validate",
            target_file=str(service_py),
            target_class="UserService",
        )
        data = json.loads(result)
        assert data["target_class"] == "UserService"

        # Check method is in target class
        with open(service_py) as f:
            content = f.read()
        assert "validate" in content

        # Check method is removed from source
        with open(python_project / "models.py") as f:
            content = f.read()
        assert "def validate" not in content
        assert "def greet" in content  # other method stays

    def test_move_to_nonexistent_target_fails_gracefully(self, bound_tool, python_project):
        """Moving to a file that doesn't exist creates it."""
        tool = bound_tool(MoveSymbolTool)
        target = str(python_project / "new_module.py")
        result = tool._run(symbol="other_func", target_file=target)
        data = json.loads(result)
        # Should succeed — creates the file
        assert os.path.isfile(target)
        with open(target) as f:
            assert "other_func" in f.read()

    def test_move_same_file_without_class_errors(self, bound_tool, python_project):
        """Moving to same file without target_class is an error."""
        tool = bound_tool(MoveSymbolTool)
        result = tool._run(
            symbol="helper_func",
            target_file=str(python_project / "utils.py"),
        )
        data = json.loads(result)
        assert "error" in data
